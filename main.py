import os
import io
import json
import logging
from uuid import uuid4
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3

# OCR & LLM
from google.cloud import vision
import google.generativeai as genai

# -------------------------------------------------------------------
# 기본 설정 / 로거
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "true").lower() == "true"

# -------------------------------------------------------------------
# AWS S3 설정
# -------------------------------------------------------------------

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not STUB_MODE:
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME):
        logger.warning("S3 환경변수 일부가 비어 있습니다. STUB_MODE = false 이지만 S3 업로드에 실패할 수 있습니다.")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# -------------------------------------------------------------------
# Google Vision / Gemini 설정
# -------------------------------------------------------------------

vision_client = None
gemini_model = None

if not STUB_MODE:
    try:
        # GOOGLE_APPLICATION_CREDENTIALS 환경변수는 Render 쪽에서 설정해둔 경로 사용
        vision_client = vision.ImageAnnotatorClient()
        logger.info("Google Vision 클라이언트 초기화 완료")
    except Exception as e:
        logger.error(f"Vision 클라이언트 초기화 실패: {e}")

    if GEMINI_ENABLED:
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini 모델 초기화 완료")
        except Exception as e:
            logger.error(f"Gemini 초기화 실패: {e}")
    else:
        logger.info("GEMINI_ENABLED = false, Gemini 사용 안 함")

# -------------------------------------------------------------------
# Pydantic 모델
# -------------------------------------------------------------------

class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsed(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # "YYYY-MM-DD"
    totalAmount: Optional[int] = None
    items: List[ReceiptItem] = []


class ReceiptAnalyzeResponse(BaseModel):
    parsed: ReceiptParsed
    rawOcrText: Optional[str] = None


class PdfRecord(BaseModel):
    id: str
    petId: str
    title: str = ""
    memo: str = ""
    s3Url: str
    kind: str  # "lab" or "cert"


# -------------------------------------------------------------------
# FastAPI 앱
# -------------------------------------------------------------------

app = FastAPI(title="PetHealth+ Server", version="1.0.0")

# CORS: iOS + 테스트용 웹에서 다 접근할 수 있게 전체 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# 헬스 체크
# -------------------------------------------------------------------

@app.get("/")
async def root():
    """Render 브라우저에서 확인용 루트 엔드포인트."""
    return {"status": "ok", "stubMode": STUB_MODE}


@app.get("/health")
async def health():
    return {"status": "ok"}


# -------------------------------------------------------------------
# 공통: S3 업로드 + presigned URL 생성
# -------------------------------------------------------------------

def upload_pdf_and_presign(kind: str, pet_id: str, file: UploadFile) -> str:
    """
    S3에 PDF 업로드 후, 기간 제한된 다운로드 URL(presigned URL)을 리턴.
    """
    if STUB_MODE:
        # 개발용 더미 URL
        return f"https://example.com/{kind}/{pet_id}/{uuid4()}.pdf"

    if not S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME 이 설정되어 있지 않습니다.")

    # 파일 키: kind/petId/uuid.pdf
    key = f"{kind}/{pet_id}/{uuid4()}.pdf"

    # UploadFile.file 은 파일 객체이므로 그대로 upload_fileobj 사용 가능
    file.file.seek(0)
    s3_client.upload_fileobj(
        file.file,
        S3_BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": "application/pdf"},
    )

    # presigned URL 생성 (예: 7일 유효)
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": key},
        ExpiresIn=60 * 60 * 24 * 7,
    )
    return url


# -------------------------------------------------------------------
# OCR + Gemini: 영수증 분석 로직
# -------------------------------------------------------------------

def call_vision_ocr(image_bytes: bytes) -> str:
    """
    Google Vision 으로 전체 텍스트 추출.
    """
    if STUB_MODE or vision_client is None:
        # stub 모드에서는 간단한 더미 텍스트
        return "펫동물병원 2025-11-18 진료비 총액 88000원 진찰료 30000원 혈액검사 40000원 기타 18000원"

    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision 오류: {response.error.message}")

    if response.full_text_annotation and response.full_text_annotation.text:
        return response.full_text_annotation.text
    elif response.text_annotations:
        return response.text_annotations[0].description
    else:
        return ""


def call_gemini_for_receipt(ocr_text: str) -> ReceiptParsed:
    """
    OCR 텍스트를 Gemini에게 넘겨 구조화된 JSON으로 파싱.
    """
    if STUB_MODE or not GEMINI_ENABLED or gemini_model is None:
        # 간단한 하드코딩 더미
        return ReceiptParsed(
            clinicName="펫동물병원",
            visitDate="2025-11-18",
            totalAmount=88000,
            items=[
                ReceiptItem(name="진찰료", price=30000),
                ReceiptItem(name="혈액검사", price=40000),
                ReceiptItem(name="기타", price=18000),
            ],
        )

    prompt = """
너는 한국 동물병원 영수증을 분석하는 도우미야.
다음 OCR 텍스트를 보고 아래 JSON 형식으로만 답해줘.

반드시 이 스키마를 지켜:
{
  "clinicName": "병원 이름 또는 null",
  "visitDate": "YYYY-MM-DD 형식 문자열 또는 null",
  "totalAmount": 정수 또는 null,
  "items": [
    { "name": "항목명", "price": 정수 또는 null },
    ...
  ]
}

불필요한 설명, 코드블록, 마크다운은 절대 넣지 말고 오직 JSON만 출력해.
"""
    content = prompt + "\n\n==== OCR TEXT ====\n" + ocr_text

    response = gemini_model.generate_content(content)
    text = response.text.strip()

    # ⁠ json ...  ⁠ 같은 코드블록 제거
    if text.startswith("```"):
        text = text.strip("`")
        # "json\n{ ... }" 형태일 수 있음
        if text.startswith("json"):
            text = text[4:]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Gemini JSON 파싱 실패: {e} / 원본: {text}")
        # 최소 구조만 채운 기본값 리턴
        return ReceiptParsed(
            clinicName=None,
            visitDate=None,
            totalAmount=None,
            items=[],
        )

    clinic = data.get("clinicName")
    visit = data.get("visitDate")
    total = data.get("totalAmount")

    items_raw = data.get("items") or []
    items: List[ReceiptItem] = []
    for it in items_raw:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or ""
        if not name:
            continue
        price_val = it.get("price")
        try:
            price_int = int(price_val) if price_val is not None else None
        except (TypeError, ValueError):
            price_int = None
        items.append(ReceiptItem(name=name, price=price_int))

    return ReceiptParsed(
        clinicName=clinic,
        visitDate=visit,
        totalAmount=total if isinstance(total, int) else None,
        items=items,
    )


# -------------------------------------------------------------------
# 엔드포인트: 영수증 OCR 분석
# -------------------------------------------------------------------

@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str = Form(...),  # 현재는 petId 는 그냥 받아만 두고 사용은 안 함
    file: UploadFile = File(...),
):
    """
    iOS에서 호출하는 영수증 OCR 엔드포인트.
    - multipart/form-data
      - petId: 문자열
      - file: 이미지 (jpg/png 등)
    """
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        # 1) OCR
        ocr_text = call_vision_ocr(contents)

        # 2) Gemini로 구조화
        parsed = call_gemini_for_receipt(ocr_text)

        return ReceiptAnalyzeResponse(parsed=parsed, rawOcrText=ocr_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"/api/receipt/analyze 오류: {e}")
        raise HTTPException(status_code=500, detail="영수증 분석 중 오류가 발생했습니다.")


# -------------------------------------------------------------------
# 엔드포인트: PDF 업로드 (검사결과 / 증명서)
# -------------------------------------------------------------------

@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form(""),
    memo: str = Form(""),
    file: UploadFile = File(...),
):
    """
    검사결과 PDF 업로드.
    iOS: APIClient.uploadLabPDF(...) 와 매칭.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        url = upload_pdf_and_presign("lab", petId, file)
        record = PdfRecord(
            id=str(uuid4()),
            petId=petId,
            title=title or "",
            memo=memo or "",
            s3Url=url,
            kind="lab",
        )
        return record
    except Exception as e:
        logger.exception(f"/api/lab/upload-pdf 오류: {e}")
        raise HTTPException(status_code=500, detail="검사결과 업로드 중 오류가 발생했습니다.")


@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form(""),
    memo: str = Form(""),
    file: UploadFile = File(...),
):
    """
    증명서 PDF 업로드.
    iOS: APIClient.uploadCertPDF(...) 와 매칭.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        url = upload_pdf_and_presign("cert", petId, file)
        record = PdfRecord(
            id=str(uuid4()),
            petId=petId,
            title=title or "",
            memo=memo or "",
            s3Url=url,
            kind="cert",
        )
        return record
    except Exception as e:
        logger.exception(f"/api/cert/upload-pdf 오류: {e}")
        raise HTTPException(status_code=500, detail="증명서 업로드 중 오류가 발생했습니다.")


# -------------------------------------------------------------------
# 로컬 개발용 실행 엔트리
# -------------------------------------------------------------------

if _name_ == "_main_":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
