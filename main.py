# ================================
# PetHealth+ Backend (AI + S3)
# ================================

import io
import os
import json
import uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.client import Config
from PIL import Image

# Gemini AI
import google.generativeai as genai


# =========================
# 환경변수 / 설정
# =========================

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB 제한


# =========================
# S3 클라이언트
# =========================

s3 = None
if AWS_S3_BUCKET:
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4"),
        endpoint_url=AWS_S3_ENDPOINT_URL or None,
    )


def build_public_url(key: str) -> str:
    if AWS_S3_PUBLIC_BASE_URL:
        return f"{AWS_S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    if AWS_S3_ENDPOINT_URL:
        return f"{AWS_S3_ENDPOINT_URL.rstrip('/')}/{key}"
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload_to_s3(file_bytes: bytes, ext: str) -> str:
    if not s3 or not AWS_S3_BUCKET:
        raise RuntimeError("S3가 올바르게 설정되지 않았습니다.")

    key = f"receipts/{uuid.uuid4().hex}.{ext}"

    s3.put_object(
        Bucket=AWS_S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=f"image/{ext}",
        ACL="private",
    )

    return build_public_url(key)


# =========================
# Pydantic DTO
# =========================

class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsed(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None
    diseaseName: Optional[str] = None
    symptomsSummary: Optional[str] = None
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None


class ReceiptAnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: Optional[str] = None


# =========================
# FastAPI 앱
# =========================

app = FastAPI(
    title="PetHealth+ Backend",
    description="반려동물 영수증 분석 / 저장 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 연결 성공",
        "s3": bool(AWS_S3_BUCKET),
        "gemini": bool(GEMINI_API_KEY),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# =========================
# 영수증 분석 API
# =========================

@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    # 1) 파일 읽기
    raw = await file.read()

    if not raw:
        raise HTTPException(400, detail="빈 파일입니다.")

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(400, detail="이미지 용량 초과 (15MB 이하만 가능).")

    # 2) 이미지 → JPEG 변환 시도
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        image_bytes = buf.read()
        ext = "jpg"
    except:
        # 이미지 아닐 경우 원본 업로드
        image_bytes = raw
        ext = file.filename.split(".")[-1].lower()

    # 3) S3 업로드
    try:
        s3_url = upload_to_s3(image_bytes, ext)
    except Exception as e:
        print("S3 업로드 실패:", e)
        s3_url = f"https://dummy-s3.pethealthplus/{petId}/{file.filename}"

    # 4) Gemini AI 분석
    if not GEMINI_API_KEY:
        raise HTTPException(500, detail="Gemini API Key 미설정")

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
        너는 동물병원 영수증 분석 전문가야.
        아래 영수증 사진에서 다음 JSON 형태로만 출력해줘.

        {
            "clinicName": "",
            "visitDate": "",
            "items": [
                { "name": "", "price": 0 }
            ],
            "totalAmount": 0
        }

        JSON 외의 텍스트는 절대 넣지 마.
        """

        response = model.generate_content(
            [prompt, image_bytes],
            safety_settings=None,
        )

        ai_text = response.text.strip()

        data = json.loads(ai_text)

    except Exception as e:
        print("Gemini 오류:", e)
        raise HTTPException(500, detail="AI 분석 실패. JSON 파싱 오류 또는 모델 문제.")

    # 5) 최종 응답 구성
    parsed = ReceiptParsed(
        clinicName=data.get("clinicName"),
        visitDate=data.get("visitDate"),
        items=data.get("items", []),
        totalAmount=data.get("totalAmount"),
    )

    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes="AI 분석 완료",
    )
