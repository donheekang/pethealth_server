# main.py
import io
import os
import re
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import boto3
from botocore.client import Config
from PIL import Image

# -------------------------------
# Environment
# -------------------------------
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
GCV_ENABLED = os.getenv("GCV_ENABLED", "true").lower() == "true"
GCV_CREDENTIALS_JSON = os.getenv("GCV_CREDENTIALS_JSON", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------------
# AWS S3 Client
# -------------------------------
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    config=Config(signature_version="s3v4"),
)

# -------------------------------
# Google Vision (OCR)
# -------------------------------
vision_client = None
if GCV_ENABLED and GCV_CREDENTIALS_JSON:
    try:
        from google.cloud import vision
        from google.oauth2.service_account import Credentials

        creds_info = json.loads(GCV_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(creds_info)
        vision_client = vision.ImageAnnotatorClient(credentials=creds)
        print("[INFO] Google Vision ready ✅")
    except Exception as e:
        print("[WARN] Google Vision init failed:", e)
else:
    print("[INFO] Google Vision disabled")

# -------------------------------
# Gemini AI
# -------------------------------
try:
    import google.generativeai as gen
    if GEMINI_API_KEY:
        gen.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = gen.GenerativeModel("gemini-1.5-flash")
        print("[INFO] Gemini model ready ✅")
    else:
        GEMINI_MODEL = None
        print("[INFO] GEMINI_API_KEY missing")
except Exception as e:
    GEMINI_MODEL = None
    print("[WARN] Gemini init error:", e)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="PetHealth+ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Models
# -------------------------------
class MedicalItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    category: str = "others"
    price: Optional[int] = None

class OCRResult(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[datetime] = None
    items: List[MedicalItem] = []
    totalAmount: Optional[int] = None
    notes: Optional[str] = None

class UploadResponse(BaseModel):
    receipt_id: str

class OCRAnalyzeRequest(BaseModel):
    receipt_id: str

# -------------------------------
# Utility
# -------------------------------
MAX_IMAGE_BYTES = 15 * 1024 * 1024

def _validate_image(raw: bytes) -> bytes:
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="이미지 용량(15MB) 초과")
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 형식 오류")

def _put_s3(key: str, data: bytes, content_type: str):
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)

def _get_s3(key: str) -> bytes:
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()

def _extract_text_vision(image_bytes: bytes) -> str:
    if not vision_client:
        return ""
    from google.cloud import vision
    image = vision.Image(content=image_bytes)
    resp = vision_client.text_detection(image=image)
    if resp.error.message:
        print("[Vision error]", resp.error.message)
        return ""
    if resp.full_text_annotation:
        return resp.full_text_annotation.text
    return ""

def _parse_text_simple(text: str) -> OCRResult:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    result = OCRResult(items=[])
    if not lines:
        return result

    # 병원명
    for ln in lines:
        if "병원" in ln:
            result.clinicName = ln
            break
    # 날짜
    date_pat = re.compile(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})")
    for ln in lines:
        m = date_pat.search(ln)
        if m:
            try:
                result.visitDate = datetime.strptime(
                    m.group(1).replace(".", "-"), "%Y-%m-%d"
                )
            except:
                pass
            break
    # 합계
    total_pat = re.compile(r"(합계|총액)\s*[:\-]?\s*([\d,]+)")
    for ln in lines:
        m = total_pat.search(ln)
        if m:
            result.totalAmount = int(m.group(2).replace(",", ""))
            break
    # 항목
    for ln in lines:
        m = re.match(r"(.+?)\s+([\d,]+)$", ln)
        if m:
            name = m.group(1)
            price = int(m.group(2).replace(",", ""))
            if not any(k in name for k in ["합계", "총액"]):
                result.items.append(MedicalItem(name=name, price=price))
    return result

def _gemini_parse(text: str) -> Optional[OCRResult]:
    if not GEMINI_MODEL:
        return None
    try:
        prompt = f"다음 동물병원 영수증 텍스트를 JSON 형식으로 구조화해줘:\n{text}"
        resp = GEMINI_MODEL.generate_content(prompt)
        data = json.loads(resp.text)
        return OCRResult(**data)
    except Exception as e:
        print("[Gemini parse error]", e)
        return None

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "PetHealth+ 서버 연결 성공 ✅", "ocr": bool(vision_client)}

@app.get("/health")
def health():
    return {"ok": True, "service": "PetHealthPlus", "version": "1.0.0"}

@app.post("/ocr/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    raw = _validate_image(raw)
    key = f"receipts/{uuid.uuid4().hex}.jpg"
    _put_s3(key, raw, "image/jpeg")
    return UploadResponse(receipt_id=key)

@app.post("/ocr/analyze", response_model=OCRResult)
async def analyze(req: OCRAnalyzeRequest):
    if not req.receipt_id.startswith("receipts/"):
        raise HTTPException(status_code=400, detail="유효하지 않은 receipt_id")
    img = _get_s3(req.receipt_id)
    text = _extract_text_vision(img)
    if not text:
        raise HTTPException(status_code=500, detail="OCR 결과 없음")
    result = _gemini_parse(text) or _parse_text_simple(text)
    return result
