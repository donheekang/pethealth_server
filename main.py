import os
import io
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import boto3
from botocore.exceptions import NoCredentialsError

from pydantic_settings import BaseSettings
from pydantic import Field

# ============================================
# SETTINGS (Render 환경변수 자동 매핑)
# ============================================

class Settings(BaseSettings):
    STUB_MODE: bool = False

    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str | None = None
    S3_BUCKET_NAME: str | None = None

    GEMINI_API_KEY: str | None = None
    GEMINI_ENABLED: bool = True

    GOOGLE_APPLICATION_CREDENTIALS: str | None = None

settings = Settings()

# ============================================
# FASTAPI APP 설정
# ============================================

app = FastAPI(title="PetHealth+ Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)


# ============================================
# 기본 헬스체크
# ============================================

@app.get("/")
async def root():
    return {"status": "ok", "stubMode": settings.STUB_MODE}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/health")
async def api_health():
    return {"status": "ok", "stubMode": settings.STUB_MODE}

# ============================================
# AWS S3 클라이언트
# ============================================

def get_s3():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        return s3
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS Credential Error")


# ============================================
# Presigned URL 생성
# ============================================

@app.get("/api/s3/presign")
async def create_presigned_url(filename: str, filetype: str):
    s3 = get_s3()
    key = f"uploads/{uuid.uuid4()}_{filename}"

    try:
        presigned_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key, "ContentType": filetype},
            ExpiresIn=3600,
        )

        return {
            "url": presigned_url,
            "path": key,
            "view_url": f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Failed to generate URL")


# ============================================
# PDF 직접 업로드 (iOS multipart 방식)
# ============================================

@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    s3 = get_s3()

    file_bytes = await file.read()
    key = f"pdf/{uuid.uuid4()}_{file.filename}"

    try:
        s3.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=key,
            Body=file_bytes,
            ContentType="application/pdf",
        )
        return {
            "status": "uploaded",
            "key": key,
            "url": f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
        }
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="PDF Upload Failed")


# ============================================
# 이미지 업로드 + Gemini OCR 분석
# ============================================

@app.post("/api/upload/image-ocr")
async def upload_and_ocr(file: UploadFile = File(...)):
    s3 = get_s3()

    file_bytes = await file.read()
    key = f"images/{uuid.uuid4()}_{file.filename}"

    # 1) S3 저장
    try:
        s3.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=key,
            Body=file_bytes,
            ContentType=file.content_type,
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="Image Upload Failed")


    # 2) OCR 분석
    if not settings.GEMINI_ENABLED or not settings.GEMINI_API_KEY:
        return {
            "status": "uploaded_only",
            "ocr": None,
            "url": f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
        }

    # Gemini OCR 실행
    try:
        from google.generativeai import configure, GenerativeModel

        configure(api_key=settings.GEMINI_API_KEY)
        model = GenerativeModel("gemini-1.5-flash")

        result = model.generate_content(["Extract text:", file_bytes])
        ocr_text = result.text

    except Exception as e:
        logger.error(e)
        ocr_text = None

    return {
        "status": "ok",
        "ocr_text": ocr_text,
        "url": f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
    }


# ============================================
# 검사결과 / 증명서 LIST (앱 첫 로딩용)
# ============================================

@app.get("/api/labs")
async def get_labs():
    return {"items": []}

@app.get("/api/certificates")
async def get_certificates():
    return {"items": []}
