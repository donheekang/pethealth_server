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

# AWS
import boto3
from botocore.exceptions import NoCredentialsError

# Settings
from pydantic_settings import BaseSettings

# Google Vision OCR
from google.cloud import vision

# Gemini (Vertex AI)
from google.genai import Client, types


# ------------------------------------------
# SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # Google credentials (JSON 전체 문자열)
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Vertex AI
    GCP_PROJECT_ID: str
    GCP_LOCATION: str = "us-central1"
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    # Gemini API key (키 기반 사용 시)
    GEMINI_API_KEY: str = ""

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
    """
    S3 업로드 + presigned URL 생성
    """
    try:
        s3_client.upload_fileobj(
            file_obj,
            settings.S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )

        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {str(e)}")


# ------------------------------------------
# GOOGLE VISION OCR
# ------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    """
    GOOGLE_APPLICATION_CREDENTIALS 에 JSON 문자열 또는 파일경로 입력 지원
    """
    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS 비어있음")

    try:
        info = json.loads(cred_value)
        return vision.ImageAnnotatorClient.from_service_account_info(info)
    except json.JSONDecodeError:
        if not os.path.exists(cred_value):
            raise Exception("GOOGLE_APPLICATION_CREDENTIALS 잘못됨")
        return vision.ImageAnnotatorClient.from_service_account_file(cred_value)


def run_vision_ocr(image_path: str) -> str:
    client = get_vision_client()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"OCR 에러: {response.error.message}")

    texts = response.text_annotations
    return texts[0].description if texts else ""


# ------------------------------------------
# 정규식 기반 Fallback Parser
# ------------------------------------------

def parse_receipt_kor(text: str) -> dict:
    import re
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 병원명 추정
    hospital = ""
    for line in lines[:8]:
        if "동물" in line and ("병원" in line or "의료" in line or "메디" in line):
            hospital = line.strip()
            break

    # 날짜
    dt_pattern = re.compile(r"(20\d{2})[.\-/년 ]+(\d{1,2})[.\-/월 ]+(\d{1,2})")
    visit = None
    for line in lines:
        m = dt_pattern.search(line)
        if m:
            y, mo, d = map(int, m.groups())
            visit = f"{y:04d}-{mo:02d}-{d:02d}"
            break

    # 금액 패턴
    amt_pattern = re.compile(r"(?:₩|￦)?\s*(\d{1,3}(?:,\d{3})|\d+)")
    items = []
    total_amount = 0

    for line in lines:
        m = amt_pattern.search(line)
        if m:
            amount = int(m.group(1).replace(",", ""))
            if any(k in line for k in ["합계", "총액", "총금액"]):
                total_amount = amount
                continue
            name = line[:m.start()].strip() or "항목"
            items.append({"name": name, "price": amount})

    if total_amount == 0:
        total_amount = sum(x["price"] for x in items)

    return {
        "clinicName": hospital or None,
        "visitDate": visit,
        "items": items,
        "totalAmount": total_amount,
    }


# ------------------------------------------
# GEMINI CLIENT (Vertex AI)
# ------------------------------------------

def get_gemini_client() -> Client:
    try:
        key_json = json.loads(settings.GOOGLE_APPLICATION_CREDENTIALS)
    except:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS JSON 파싱 실패")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"
    with open("/tmp/gcp_key.json", "w") as f:
        f.write(json.dumps(key_json))

    return Client(
        vertexai=True,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
    )


# ------------------------------------------
# Gemini로 영수증 이미지 OCR + 구조화
# ------------------------------------------

def analyze_receipt_with_gemini(image_bytes: bytes) -> Optional[dict]:
    client = get_gemini_client()

    prompt = """
당신은 한국 동물병원 영수증을 정확히 분석하는 AI입니다.

절대 "동물병원" 같은 일반명사로 요약하지 말고,
영수증에 기재된 병원명을 그대로 추출하세요.
예: 해랑동물병원, 펫앤아이동물병원, 24시스마트동물의료센터 등

아래 JSON 형식으로만 출력:

{
  "clinicName": string or null,
  "visitDate": string or null,
  "items": [
    { "name": string, "price": integer or null }
  ],
  "totalAmount": integer or null
}
"""

    try:
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

        # 코드 블록 제거
        if "```" in text:
            text = text[text.find("{"): text.rfind("}") + 1]

        data = json.loads(text)

        # 최소 키 검사
        if "clinicName" not in data:
            return None

        return data

    except:
        return None
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
# 영수증 업로드 + Gemini OCR + Vision fallback + regex fallback
# ------------------------------------------

@app.post("/api/receipt/analyze")
@app.post("/api/receipts/analyze")
@app.post("/receipt/upload")
async def upload_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    1) S3 업로드
    2) Gemini 2.5 Flash 이미지 OCR + 구조화
    3) Gemini 실패 → Vision OCR fallback
    4) Vision 실패 → 정규식 파서 fallback
    """

    # -------------------------------------
    # 파일 준비
    # -------------------------------------
    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or "")
    ext = ext if ext else ".jpg"

    key = f"receipts/{petId}/{rec_id}{ext}"

    data = await file.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    # -------------------------------------
    # 1) S3 업로드
    # -------------------------------------
    file_url = upload_to_s3(
        file_like,
        key,
        content_type=file.content_type or "image/jpeg"
    )

    # -------------------------------------
    # 2) Gemini 2.5 Flash OCR
    # -------------------------------------
    ai_parsed = analyze_receipt_with_gemini(data)

    # Gemini가 정확히 추출한 경우
    if ai_parsed and (
        ai_parsed.get("clinicName") or ai_parsed.get("items")
    ):
        return {
            "petId": petId,
            "s3Url": file_url,
            "parsed": ai_parsed,
            "notes": "gemini"
        }

    # -------------------------------------
    # 3) Vision OCR fallback
    # -------------------------------------
    text = ""
    tmp_path = None

    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)

        text = run_vision_ocr(tmp_path)

    except Exception:
        text = ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Vision이 정상 텍스트를 뽑았다면 정규식 파서 실행
    if text:
        fb = parse_receipt_kor(text)

        return {
            "petId": petId,
            "s3Url": file_url,
            "parsed": fb,
            "notes": text
        }

    # -------------------------------------
    # 4) 모든 단계 실패 → 빈값 처리
    # -------------------------------------
    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": {
            "clinicName": None,
            "visitDate": None,
            "items": [],
            "totalAmount": None
        },
        "notes": ""
    }
    # -------------------------------------------------------
# PDF 업로드 (검사결과 / 증명서 공통)
# -------------------------------------------------------

@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form(None),
    clinicName: str = Form(None),
    examDate: str = Form(None),   # yyyy-MM-dd 권장
    type: str = Form(None),
    memo: str = Form(None),
    file: UploadFile = File(...)
):
    """
    검사결과 PDF 업로드 → S3 저장 → DB 저장 없이 JSON으로 반환
    """
    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower() if ext else ".pdf"

    key = f"pdf/lab/{petId}/{rec_id}{ext}"

    data = await file.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    # PDF S3 업로드
    file_url = upload_to_s3(
        file_like,
        key,
        content_type="application/pdf"
    )

    # createdAt ISO8601
    created_at = datetime.utcnow().isoformat() + "Z"

    return {
        "id": rec_id,
        "petId": petId,
        "title": title,
        "clinicName": clinicName,
        "examDate": examDate,
        "type": type,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at,
    }



# -------------------------------------------------------
# 증명서 PDF 업로드
# -------------------------------------------------------

@app.post("/api/cert/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form(None),
    clinicName: str = Form(None),
    examDate: str = Form(None),
    type: str = Form(None),
    memo: str = Form(None),
    file: UploadFile = File(...)
):
    """
    증명서 업로드 → JSON 반환
    """
    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower() if ext else ".pdf"

    key = f"pdf/certificate/{petId}/{rec_id}{ext}"

    data = await file.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    file_url = upload_to_s3(
        file_like,
        key,
        content_type="application/pdf"
    )

    created_at = datetime.utcnow().isoformat() + "Z"

    return {
        "id": rec_id,
        "petId": petId,
        "title": title,
        "clinicName": clinicName,
        "examDate": examDate,
        "type": type,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at,
    }



# -------------------------------------------------------
# PDF 리스트 조회 (검사결과)
# -------------------------------------------------------

@app.get("/api/labs/list")
async def list_labs(petId: str):
    """
    DB가 없으므로 S3 prefix 스캔 방식
    단, Render는 리스트 제한 있으니 실제 서비스 시 DB 추천
    """

    prefix = f"pdf/lab/{petId}/"

    # 버킷 내 객체 리스트 불러오기
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

    records = []

    for obj in resp.get("Contents", []):
        key = obj["Key"]
        url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"

        # id 추출
        base = os.path.basename(key)
        rec_id, _ = os.path.splitext(base)

        # createdAt (S3 metadata)
        created_at = obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%SZ")

        records.append({
            "id": rec_id,
            "petId": petId,
            "title": None,
            "clinicName": None,
            "examDate": None,
            "type": None,
            "memo": None,
            "s3Url": url,
            "createdAt": created_at
        })

    return sorted(
        records,
        key=lambda x: x["createdAt"],
        reverse=True
    )



# -------------------------------------------------------
# PDF 리스트 조회 (증명서)
# -------------------------------------------------------

@app.get("/api/certs/list")
async def list_certs(petId: str):

    prefix = f"pdf/certificate/{petId}/"

    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

    records = []

    for obj in resp.get("Contents", []):
        key = obj["Key"]
        url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"

        base = os.path.basename(key)
        rec_id, _ = os.path.splitext(base)

        created_at = obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%SZ")

        records.append({
            "id": rec_id,
            "petId": petId,
            "title": None,
            "clinicName": None,
            "examDate": None,
            "type": None,
            "memo": None,
            "s3Url": url,
            "createdAt": created_at
        })

    return sorted(
        records,
        key=lambda x: x["createdAt"],
        reverse=True
    )
