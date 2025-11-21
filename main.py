from __future__ import annotations

import os
import json
import io
import tempfile
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError
from pydantic_settings import BaseSettings


# ------------------------------------------
# SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # 서비스 계정 JSON "내용" 전체
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    GEMINI_ENABLED: str = "false"
    STUB_MODE: str = "false"

settings = Settings()


# ------------------------------------------
# S3 CLIENT
# ------------------------------------------

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


def upload_to_s3(file_obj, key: str, content_type: str) -> str:
    """파일 업로드 + presigned URL 반환"""
    try:
        s3_client.upload_fileobj(
            file_obj,
            settings.S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )

        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=7 * 24 * 3600,  # 7일
        )
        return url

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 인증 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {str(e)}")


# ------------------------------------------
# GOOGLE VISION OCR
# ------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    json_str = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not json_str:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS 환경변수가 비어있습니다.")

    try:
        info = json.loads(json_str)
        client = vision.ImageAnnotatorClient.from_service_account_info(info)
        return client
    except Exception as e:
        raise Exception(f"OCR 클라이언트 생성 실패: {e}")


def run_vision_ocr(image_path: str) -> str:
    client = get_vision_client()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"OCR 에러: {response.error.message}")

    texts = response.text_annotations
    if not texts:
        return ""

    return texts[0].description


# ------------------------------------------
# APP 기본 설정
# ------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "PetHealth+ Server Running"}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ------------------------------------------
# 1) 진료기록 OCR (영수증 이미지)
#    ※ iOS에서 실제 호출하는 경로에 맞게 수정 필요
#    일단 /receipt/upload /api/receipt/upload 으로 둠
# ------------------------------------------

@app.post("/receipt/upload")
@app.post("/api/receipt/upload")
async def upload_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    영수증 이미지 업로드 + Vision OCR
    - 이미지: receipts/{petId}/{id}.jpg
    - 응답: OCR 텍스트 포함
    """
    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        ext = ".jpg"

    key = f"receipts/{petId}/{rec_id}{ext}"

    data = await file.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    file_url = upload_to_s3(
        file_like,
        key,
        content_type=file.content_type or "image/jpeg",
    )

    # OCR용 임시 파일
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)

        ocr_text = run_vision_ocr(tmp_path)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vision OCR 사용 불가 (환경 설정 오류): {e}",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # iOS 쪽에서 뭐로 받을지는 나중에 맞춰야 하지만,
    # 일단 공통 필드 구조 맞춰서 반환
    return {
        "id": rec_id,
        "petId": petId,
        "title": "진료기록",
        "memo": ocr_text,
        "s3Url": file_url,
        "createdAt": created_at,
    }


# ------------------------------------------
# 2) 검사결과 PDF 업로드
#    /lab/upload-pdf, /api/lab/upload-pdf
#    → iOS가 기대하는 구조: LabRecord 한 개
# ------------------------------------------

@app.post("/lab/upload-pdf")
@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form("검사결과"),
    memo: str | None = Form(None),
    file: UploadFile = File(...),
):
    lab_id = str(uuid.uuid4())
    key = f"lab/{petId}/{lab_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")

    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # 🔥 iOS 팝업 raw에 찍힌 구조에 맞춤
    return {
        "id": lab_id,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at,
    }


# ------------------------------------------
# 3) 증명서 PDF 업로드
#    /cert/upload-pdf, /api/cert/upload-pdf
# ------------------------------------------

@app.post("/cert/upload-pdf")
@app.post("/api/cert/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form("증명서"),
    memo: str | None = Form(None),
    file: UploadFile = File(...),
):
    cert_id = str(uuid.uuid4())
    key = f"cert/{petId}/{cert_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")

    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "id": cert_id,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at,
    }


# ------------------------------------------
# 4) 검사결과 리스트
#    /lab/list, /api/lab/list  (단수 lab)
#    쿼리: ?petId=...
#    응답: [ { id, petId, title, memo, s3Url, createdAt }, ... ]
# ------------------------------------------

@app.get("/lab/list")
@app.get("/api/lab/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"

    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=prefix,
    )

    items = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]  # lab/{petId}/{id}.pdf
            if not key.endswith(".pdf"):
                continue

            file_id = os.path.splitext(key.split("/")[-1])[0]
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )
            created_at = obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%S")

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": "검사결과",
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at,
                }
            )

    # 🔥 iOS는 배열 자체를 기대하므로 items 만 반환
    return items


# ------------------------------------------
# 5) 증명서 리스트
#    /cert/list, /api/cert/list
#    쿼리: ?petId=...
# ------------------------------------------

@app.get("/cert/list")
@app.get("/api/cert/list")
def get_cert_list(petId: str = Query(...)):
    prefix = f"cert/{petId}/"

    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=prefix,
    )

    items = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]  # cert/{petId}/{id}.pdf
            if not key.endswith(".pdf"):
                continue

            file_id = os.path.splitext(key.split("/")[-1])[0]
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )
            created_at = obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%S")

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": "증명서",
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at,
                }
            )

    return items
