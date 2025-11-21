from __future__ import annotations

import os
import json
import io
import tempfile
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
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

    # 서비스 계정 JSON "내용" 전체를 넣어둘 환경변수
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    GEMINI_ENABLED: str = "false"
    STUB_MODE: str = "false"  # 필요하면 테스트용

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


def upload_to_s3(file_obj, key: str, content_type: str | None = None) -> str:
    """
    주어진 file-like 객체를 S3에 업로드하고 presigned URL 반환
    """
    if content_type is None:
        content_type = "application/octet-stream"

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
            ExpiresIn=3600,
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
    """
    Render 환경변수 GOOGLE_APPLICATION_CREDENTIALS 에
    서비스계정 JSON '문자열'이 들어있다는 전제
    """
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
    """
    Google Vision OCR 실행 후 전체 텍스트 반환
    """
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
# FASTAPI APP 기본 설정
# ------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 & 헬스체크
@app.get("/")
def root():
    return {"status": "ok", "message": "PetHealth+ Server Running"}

@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ------------------------------------------
# 1) 영수증 이미지 업로드 + OCR
#    /receipt/upload, /api/receipt/upload
# ------------------------------------------

@app.post("/receipt/upload")
@app.post("/api/receipt/upload")
async def upload_receipt(file: UploadFile = File(...)):
    """
    영수증 이미지 업로드 + Vision OCR 수행
    - 이미지 S3 업로드
    - /tmp 에 임시 파일 저장 후 OCR
    - ocrText 와 fileUrl 을 함께 반환
    """

    # 파일 확장자
    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".jpg"

    key = f"receipts/{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

    # 파일 내용을 메모리로 읽기
    data = await file.read()

    # 1) S3 업로드
    file_like = io.BytesIO(data)
    file_like.seek(0)
    file_url = upload_to_s3(
        file_like,
        key,
        content_type=file.content_type or "image/jpeg",
    )

    # 2) OCR용 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        ocr_text = run_vision_ocr(tmp_path)
    except Exception as e:
        # 여기서 에러 문구를 그대로 보내면 네가 지금 보는
        # "Vision OCR 사용 불가 (환경 설정 오류)" 같은 팝업이 뜨는 부분
        raise HTTPException(
            status_code=500,
            detail=f"Vision OCR 사용 불가 (환경 설정 오류): {e}",
        )
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

    return {
        "id": key,
        "fileUrl": file_url,
        "fileName": file.filename,
        "ocrText": ocr_text,
        "type": "receipt",
    }


# ------------------------------------------
# 2) 검사결과 PDF 업로드
#    /lab/upload-pdf, /api/lab/upload-pdf
# ------------------------------------------

@app.post("/lab/upload-pdf")
@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(file: UploadFile = File(...)):
    """
    검사결과 PDF 업로드
    """
    filename = f"labs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_like = file.file  # PDF는 스트림 그대로 사용
    file_url = upload_to_s3(file_like, filename, content_type="application/pdf")

    return {
        "id": filename,
        "fileUrl": file_url,
        "fileName": file.filename,
        "type": "lab",
    }


# ------------------------------------------
# 3) 증명서 PDF 업로드
#    /cert/upload-pdf, /api/cert/upload-pdf
# ------------------------------------------

@app.post("/cert/upload-pdf")
@app.post("/api/cert/upload-pdf")
async def upload_cert_pdf(file: UploadFile = File(...)):
    """
    증명서 PDF 업로드
    """
    filename = f"certs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file_like = file.file
    file_url = upload_to_s3(file_like, filename, content_type="application/pdf")

    return {
        "id": filename,
        "fileUrl": file_url,
        "fileName": file.filename,
        "type": "cert",
    }


# ------------------------------------------
# 4) 검사결과 리스트
#    /labs/list, /api/labs/list
# ------------------------------------------

@app.get("/labs/list")
@app.get("/api/labs/list")
def get_labs_list():
    """
    S3 labs/ 폴더의 PDF 목록 조회
    """
    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix="labs/",
    )

    items = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".pdf"):
                url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                    ExpiresIn=3600,
                )
                items.append(
                    {
                        "id": key,
                        "fileName": key.split("/")[-1],
                        "fileUrl": url,
                    }
                )

    return {"items": items}


# ------------------------------------------
# 5) 증명서 리스트
#    /cert/list, /api/cert/list
# ------------------------------------------

@app.get("/cert/list")
@app.get("/api/cert/list")
def get_cert_list():
    """
    S3 certs/ 폴더의 PDF 목록 조회
    """
    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix="certs/",
    )

    items = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".pdf"):
                url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                    ExpiresIn=3600,
                )
                items.append(
                    {
                        "id": key,
                        "fileName": key.split("/")[-1],
                        "fileUrl": url,
                    }
                )

    return {"items": items}
