from _future_ import annotations

import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError


# ============================================================
#  환경 설정
# ============================================================

class Settings(BaseSettings):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "ap-northeast-2"
    s3_bucket_name: str

    gemini_api_key: Optional[str] = None
    gemini_enabled: bool = False

    google_application_credentials: Optional[str] = None

    # True면 OCR/Gemini 비활성 + 더미 데이터
    stub_mode: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


# ============================================================
#  FastAPI & CORS
# ============================================================

app = FastAPI(title="PetHealth+ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ============================================================
#  S3 유틸
# ============================================================

def get_s3_client():
    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        raise RuntimeError("AWS credentials missing")
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
        config=Config(s3={"addressing_style": "path"}),
    )


def upload_bytes_to_s3(
    data: bytes,
    key: str,
    content_type: str,
    metadata: Optional[dict] = None,
) -> str:
    try:
        s3 = get_s3_client()
        extra = {"ContentType": content_type}
        if metadata:
            extra["Metadata"] = metadata

        s3.put_object(
            Bucket=settings.s3_bucket_name,
            Key=key,
            Body=data,
            **extra,
        )

        # presigned URL
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket_name, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
        return url
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"S3 upload error: {e}")
        # ============================================================
#  OCR / Gemini 초기화 (서버 죽지 않도록 보호)
# ============================================================

# Vision
try:
    from google.cloud import vision  # type: ignore
except Exception as e:
    print("⚠️ google.cloud.vision import 실패:", e)
    vision = None

# Gemini
try:
    import google.generativeai as genai  # type: ignore
except Exception as e:
    print("⚠️ google.generativeai import 실패:", e)
    genai = None


# Vision Client
vision_client: Optional["vision.ImageAnnotatorClient"] = None
if vision is not None:
    try:
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        vision_client = vision.ImageAnnotatorClient()
        print("✅ Vision OCR 초기화 성공")
    except Exception as e:
        print("⚠️ Vision OCR 초기화 실패:", e)
        vision_client = None


# Gemini Client
if settings.gemini_enabled and settings.gemini_api_key and genai is not None:
    try:
        genai.configure(api_key=settings.gemini_api_key)
        print("✅ Gemini 초기화 성공")
    except Exception as e:
        print("⚠️ Gemini 초기화 실패:", e)
        settings.gemini_enabled = False
else:
    settings.gemini_enabled = False


# ============================================================
#  OCR 함수
# ============================================================

async def run_vision_ocr(image_bytes: bytes) -> str:
    """ Vision OCR → 원문 텍스트 """
    if settings.stub_mode:
        return "테스트 OCR 결과 입니다."

    if vision_client is None:
        raise HTTPException(status_code=500, detail="Vision OCR 사용 불가 (환경 설정 오류)")

    try:
        img = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=img)
        if response.error.message:
            raise RuntimeError(response.error.message)

        texts = [t.description for t in response.text_annotations]
        return texts[0] if texts else ""

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 오류: {e}")


# ============================================================
#  Gemini (LLM) 파싱
# ============================================================

async def parse_with_gemini(ocr_text: str) -> dict:
    """
    OCR 결과를 기반으로 병원명/날짜/항목/금액 파싱
    """
    if settings.stub_mode:
        return {
            "clinicName": "테스트동물병원",
            "timestamp": "2025-11-20 12:30",
            "items": [
                {"name": "DHPPi", "price": 30000},
                {"name": "Corona", "price": 25000},
            ],
            "totalAmount": 55000,
        }

    if not settings.gemini_enabled or genai is None:
        raise HTTPException(status_code=500, detail="Gemini 사용 불가 (환경 설정 오류)")

    prompt = f"""
다음 OCR 텍스트에서 병원명, 날짜/시간, 진료 항목과 금액을 JSON으로 정확하게 추출하라.

OCR 텍스트:
{ocr_text}

JSON 구조:
{{
  "clinicName": "...",
  "timestamp": "YYYY-MM-DD HH:MM",
  "items": [
      {{"name": "...", "price": 0000}}
  ],
  "totalAmount": 0000
}}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        parsed = json.loads(response.text)
        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini 파싱 오류: {e}")
        # ============================================================
#  데이터 모델
# ============================================================

class ReceiptParsed(BaseModel):
    clinicName: Optional[str]
    timestamp: Optional[str]
    items: List[dict] = []
    totalAmount: Optional[int]


class ReceiptAnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: Optional[str] = None


class PdfRecord(BaseModel):
    id: str
    petId: str
    title: str
    memo: Optional[str] = None
    s3Url: str
    createdAt: str


# ============================================================
#  헬스 체크
# ============================================================

@app.get("/")
async def root():
    return {"msg": "PetHealth+ server alive"}


@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ocr": vision_client is not None,
        "gemini": settings.gemini_enabled,
        "stub": settings.stub_mode,
    }


# ============================================================
#  1) 영수증 분석 (OCR → Gemini → S3 저장)
# ============================================================

@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str = Form(...),
    image: UploadFile = File(...)
):
    print("📥 /api/receipt/analyze 요청 들어옴")

    img_bytes = await image.read()

    # OCR 수행
    ocr_text = await run_vision_ocr(img_bytes)

    # Gemini 파싱
    parsed = await parse_with_gemini(ocr_text)

    # S3 저장
    object_key = f"receipt/{petId}/{uuid.uuid4()}.jpg"
    s3_url = upload_bytes_to_s3(
        img_bytes,
        object_key,
        content_type="image/jpeg",
    )

    response = ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=ReceiptParsed(**parsed),
        notes=ocr_text[:500]
    )
    return response


# ============================================================
#  2) 검사결과 PDF 업로드
# ============================================================

@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form(...),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    print("📥 /api/lab/upload-pdf 요청")

    pdf_bytes = await file.read()

    object_key = f"lab/{petId}/{uuid.uuid4()}.pdf"
    s3_url = upload_bytes_to_s3(
        pdf_bytes,
        object_key,
        content_type="application/pdf",
    )

    return PdfRecord(
        id=str(uuid.uuid4()),
        petId=petId,
        title=title,
        memo=memo,
        s3Url=s3_url,
        createdAt=datetime.now(timezone.utc).isoformat()
    )


# ============================================================
#  3) 증명서 PDF 업로드
# ============================================================

@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form(...),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    print("📥 /api/cert/upload-pdf 요청")

    pdf_bytes = await file.read()

    object_key = f"cert/{petId}/{uuid.uuid4()}.pdf"
    s3_url = upload_bytes_to_s3(
        pdf_bytes,
        object_key,
        content_type="application/pdf",
    )

    return PdfRecord(
        id=str(uuid.uuid4()),
        petId=petId,
        title=title,
        memo=memo,
        s3Url=s3_url,
        createdAt=datetime.now(timezone.utc).isoformat()
    )
