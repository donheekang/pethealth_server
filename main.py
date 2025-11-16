# main.py
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# S3
import boto3
from botocore.client import Config

# Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Google Vision OCR
try:
    from google.cloud import vision
except ImportError:
    vision = None


# ============================================================
# 환경 변수
# ============================================================
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

GCV_ENABLED = os.getenv("GCV_ENABLED", "false").lower() == "true"
GCV_CREDENTIALS_JSON = os.getenv("GCV_CREDENTIALS_JSON")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# FastAPI 앱
# ============================================================
app = FastAPI(title="PetHealth+ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 루트 (/) — 앱 상태 확인용
# ============================================================
@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 연결 성공 ✅",
        "ocr": GCV_ENABLED,
        "stubMode": STUB_MODE,
    }


# ============================================================
# 헬스체크 (/health)
# ============================================================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc),
        "stubMode": STUB_MODE
    }


# ============================================================
# S3 Client
# ============================================================
def s3_client():
    if STUB_MODE:
        return None

    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def upload_to_s3(file_bytes: bytes, filename: str):
    if STUB_MODE:
        return f"https://stub.s3/{filename}"

    client = s3_client()
    client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=filename,
        Body=file_bytes,
        ContentType="application/octet-stream"
    )

    return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"


# ============================================================
# Google Vision OCR
# ============================================================
def run_vision_ocr(image_bytes: bytes) -> str:
    if STUB_MODE or not GCV_ENABLED:
        return "토리동물병원\n진료 15,000원\n피부약 22,000원"

    if not GCV_CREDENTIALS_JSON:
        raise HTTPException(500, "Google OCR 자격증명이 설정되지 않음")

    client = vision.ImageAnnotatorClient()
    img = vision.Image(content=image_bytes)
    response = client.text_detection(image=img)
    if not response.text_annotations:
        return ""
    return response.text_annotations[0].description


# ============================================================
# Gemini 분석
# ============================================================
def analyze_with_gemini(text: str) -> dict:
    if STUB_MODE:
        return {
            "clinicName": "토리동물병원",
            "items": [
                {"name": "진료", "price": 15000},
                {"name": "피부약", "price": 22000}
            ],
            "totalAmount": 37000
        }

    if not GEMINI_API_KEY or genai is None:
        raise HTTPException(500, "Gemini API Key 미설정")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
다음 텍스트는 동물병원 영수증입니다.
병원명, 진료항목 이름, 금액, 총금액을 JSON으로 변환하세요.

텍스트:
{text}

JSON 형식:
{{
  "clinicName": "",
  "items": [
    {{"name": "", "price": 0}}
  ],
  "totalAmount": 0
}}
"""

    res = model.generate_content(prompt)
    cleaned = res.text.replace("⁠  json", "").replace("  ⁠", "").strip()

    import json
    return json.loads(cleaned)


# ============================================================
# API 1: 영수증 이미지 OCR + 분석
# ============================================================
@app.post("/api/receipt/analyze")
async def upload_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        raw = await file.read()
        ext = file.filename.split(".")[-1].lower()
        filename = f"receipts/{petId}/{uuid.uuid4()}.{ext}"

        # 업로드
        s3_url = upload_to_s3(raw, filename)

        # OCR
        ocr_text = run_vision_ocr(raw)

        # Gemini 분석
        result = analyze_with_gemini(ocr_text)

        return {
            "petId": petId,
            "s3Url": s3_url,
            "ocrText": ocr_text,
            "parsed": result
        }

    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================
# API 2: PDF (검사결과·증명서) 업로드 + Gemini 분석
# ============================================================
@app.post("/api/pdf/analyze")
async def upload_pdf(
    petId: str = Form(...),
    file: UploadFile = File(...)
):
    raw = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext not in ["pdf"]:
        raise HTTPException(400, "PDF만 업로드 가능")

    filename = f"pdf/{petId}/{uuid.uuid4()}.pdf"
    s3_url = upload_to_s3(raw, filename)

    # Gemini PDF 분석
    if STUB_MODE:
        analysis = {"summary": "PDF 분석 결과 (stub)"}
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        analysis = model.generate_content(
            "다음 PDF 검사를 해석해 요약해주세요.",
            blob={"mime_type": "application/pdf", "data": raw}
        ).text

    return {
        "petId": petId,
        "s3Url": s3_url,
        "analysis": analysis
    }
