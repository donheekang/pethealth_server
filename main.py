from __future__ import annotations

import os
import io
import json
import uuid
import tempfile
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware

# AWS S3
import boto3
from botocore.exceptions import NoCredentialsError

# Settings
from pydantic_settings import BaseSettings

# Gemini (Vertex AI Image + Text)
from google.genai import types
from google.genai import Client


# ------------------------------------------
# SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # GCP / Gemini
    GCP_PROJECT_ID: str
    GCP_LOCATION: str = "us-central1"
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    GOOGLE_APPLICATION_CREDENTIALS_JSON: str = ""

    STUB_MODE: str = "false"


settings = Settings()


# ------------------------------------------
# AWS S3 CLIENT
# ------------------------------------------

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


def upload_to_s3(file_obj, key: str, content_type: str) -> str:
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
            ExpiresIn=7 * 24 * 3600,
        )
        return url

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {str(e)}")


# ------------------------------------------
# GEMINI CLIENT (Vertex AI)
# ------------------------------------------

def get_gemini_client() -> Client:
    try:
        service_info = json.loads(settings.GOOGLE_APPLICATION_CREDENTIALS_JSON)
    except:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS_JSON 파싱 실패")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"
    with open("/tmp/gcp_key.json", "w") as f:
        f.write(json.dumps(service_info))

    client = Client(
        vertexai=True,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
    )
    return client


# ------------------------------------------
# Gemini로 영수증 이미지 직접 분석하기
# ------------------------------------------

def analyze_receipt_with_gemini(image_bytes: bytes) -> dict:
    client = get_gemini_client()

    prompt = """
당신은 한국 동물병원 영수증을 분석해 구조화하는 AI입니다.

절대 “동물병원”과 같이 일반명사로 요약하지 말고,
영수증에 적힌 병원명(예: 해랑동물병원, 펫앤아이동물병원 등)을
그대로 추출하세요.

JSON 형식으로만 출력하세요:

{
  "clinicName": string or null,
  "visitDate": string or null,
  "items": [
    { "name": string, "price": integer or null }
  ],
  "totalAmount": integer or null
}
"""

    result = client.models.generate_content(
        model=settings.GEMINI_MODEL_NAME,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    text = result.text.strip()

    # 코드블록 제거
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

    try:
        return json.loads(text)
    except:
        return {
            "clinicName": None,
            "visitDate": None,
            "items": [],
            "totalAmount": None
        }


# ------------------------------------------
# FASTAPI APP
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
# 1) 영수증 업로드 → S3 저장 + Gemini 분석
# ------------------------------------------

@app.post("/api/receipt/analyze")
@app.post("/api/receipts/analyze")
@app.post("/receipt/upload")
async def upload_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        ext = ".jpg"

    key = f"receipts/{petId}/{rec_id}{ext}"

    # 파일 읽기
    data = await file.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    # 1) S3 업로드
    file_url = upload_to_s3(
        file_like,
        key,
        content_type=file.content_type or "image/jpeg"
    )

    # 2) Gemini로 이미지 직접 분석
    ai_parsed = analyze_receipt_with_gemini(data)

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": ai_parsed,
    }


# ------------------------------------------
# 2) 검사결과 PDF 업로드
# ------------------------------------------

@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form("검사결과"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    lab_id = str(uuid.uuid4())

    original_base = os.path.splitext(file.filename or "")[0].strip() or "검사결과"
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")

    key = f"lab/{petId}/{safe_base}_{lab_id}.pdf"

    file_url = upload_to_s3(file.file, key, "application/pdf")

    created_at_dt = datetime.utcnow()
    created_at_iso = created_at_dt.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = created_at_dt.strftime("%Y-%m-%d")

    display_title = f"{original_base} ({date_str})"

    return {
        "id": lab_id,
        "petId": petId,
        "title": display_title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at_iso,
    }


# ------------------------------------------
# 3) 증명서 PDF 업로드
# ------------------------------------------

@app.post("/api/cert/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form("증명서"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    cert_id = str(uuid.uuid4())

    original_base = os.path.splitext(file.filename or "")[0].strip() or "증명서"
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")

    key = f"cert/{petId}/{safe_base}_{cert_id}.pdf"

    file_url = upload_to_s3(file.file, key, "application/pdf")

    created_at_dt = datetime.utcnow()
    created_at_iso = created_at_dt.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = created_at_dt.strftime("%Y-%m-%d")

    display_title = f"{original_base} ({date_str})"

    return {
        "id": cert_id,
        "petId": petId,
        "title": display_title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at_iso,
    }


# ------------------------------------------
# 4) 검사결과 리스트
# ------------------------------------------

@app.get("/api/labs/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"

    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=prefix,
    )

    items: List[Dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            name_no_ext = filename[:-4]

            base_title = "검사결과"
            file_id = name_no_ext

            if "_" in name_no_ext:
                safe_base, file_id = name_no_ext.split("_", 1)
                base_title = safe_base.replace("_", " ")

            created_dt = obj["LastModified"]
            created_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            display_title = f"{base_title} ({date_str})"

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )

            items.append({
                "id": file_id,
                "petId": petId,
                "title": display_title,
                "memo": None,
                "s3Url": url,
                "createdAt": created_iso,
            })

    return items


# ------------------------------------------
# 5) 증명서 리스트
# ------------------------------------------

@app.get("/api/certs/list")
def get_cert_list(petId: str = Query(...)):
    prefix = f"cert/{petId}/"

    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=prefix,
    )

    items: List[Dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            name_no_ext = filename[:-4]

            base_title = "증명서"
            file_id = name_no_ext

            if "_" in name_no_ext:
                safe_base, file_id = name_no_ext.split("_", 1)
                base_title = safe_base.replace("_", " ")

            created_dt = obj["LastModified"]
            created_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            display_title = f"{base_title} ({date_str})"

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )

            items.append({
                "id": file_id,
                "petId": petId,
                "title": display_title,
                "memo": None,
                "s3Url": url,
                "createdAt": created_iso,
            })

    return items
