# main.py
import io
import os
import re
import json
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from botocore.client import Config
from PIL import Image

# ---------------------------
# Google Vision
# ---------------------------
GCV_ENABLED = os.getenv("GCV_ENABLED", "true").lower() == "true"
GCV_CREDENTIALS_JSON = os.getenv("GCV_CREDENTIALS_JSON", "")
vision_client = None

if GCV_ENABLED and GCV_CREDENTIALS_JSON:
    try:
        from google.cloud import vision
        from google.oauth2.service_account import Credentials
        creds_info = json.loads(GCV_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(creds_info)
        vision_client = vision.ImageAnnotatorClient(credentials=creds)
        print("[INFO] ✅ Google Vision ready")
    except Exception as e:
        print("[WARN] Vision init failed:", e)
else:
    print("[INFO] Google Vision disabled")

# ---------------------------
# AWS S3
# ---------------------------
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    config=Config(signature_version="s3v4"),
)

# ---------------------------
# FastAPI 기본 설정
# ---------------------------
app = FastAPI(title="PetHealth+ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 헬스체크 라우터 (절대 안 사라지게)
# ---------------------------
health_router = APIRouter()

@health_router.get("/health")
def health():
    """Render용 헬스체크 엔드포인트"""
    return {"ok": True, "service": "PetHealthPlus", "version": "1.0.0"}

app.include_router(health_router)

# ---------------------------
# 기본 루트
# ---------------------------
@app.get("/")
def root():
    return {"message": "PetHealth+ 서버 연결 성공 ✅", "ocr": bool(vision_client)}

# ---------------------------
# 데이터 모델
# ---------------------------
class MedicalItem(BaseModel):
    id: str = uuid.uuid4().hex
    name: str
    category: str = "others"
    price: Optional[int] = None

class OCRResult(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[datetime] = None
    items: List[MedicalItem] = []
    notes: Optional[str] = None
    totalAmount: Optional[int] = None

class UploadResponse(BaseModel):
    receipt_id: str

class OCRAnalyzeRequest(BaseModel):
    receipt_id: str

# ---------------------------
# 유틸리티
# ---------------------------
def _put_s3(key: str, data: bytes, content_type: str = "image/jpeg"):
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

def _get_s3(key: str) -> bytes:
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"S3 object not found: {e}")

def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def _extract_text_google_vision(image_bytes: bytes) -> str:
    if not vision_client:
        return ""
    image = vision.Image(content=image_bytes)
    resp = vision_client.text_detection(image=image)
    if resp.error.message:
        print("[Vision] error:", resp.error.message)
        return ""
    return resp.full_text_annotation.text if resp.full_text_annotation else ""

def _simple_parse(text: str) -> OCRResult:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    result = OCRResult(items=[])

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
            raw = m.group(1).replace("/", "-").replace(".", "-")
            try:
                result.visitDate = datetime.strptime(raw, "%Y-%m-%d")
            except:
                pass
            break
    # 총액
    total_pat = re.compile(r"(합계|총액|총\s*합계)\s*[:\-]?\s*([\d,]+)")
    for ln in lines:
        m = total_pat.search(ln)
        if m:
            result.totalAmount = int(m.group(2).replace(",", ""))
            break
    # 항목
    price_tail = re.compile(r"(.+?)\s+([\d,]{3,})$")
    for ln in lines:
        m = price_tail.match(ln)
        if m:
            name, price = m.groups()
            if not any(k in name for k in ["합계", "총액", "소계"]):
                result.items.append(MedicalItem(name=name, price=int(price.replace(",", ""))))
    return result

# ---------------------------
# 엔드포인트
# ---------------------------
@app.post("/ocr/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        raw = _pil_to_jpeg_bytes(img)
    except Exception:
        pass
    receipt_id = f"receipts/{uuid.uuid4().hex}.jpg"
    _put_s3(receipt_id, raw)
    return UploadResponse(receipt_id=receipt_id)

@app.post("/ocr/analyze", response_model=OCRResult)
async def analyze(req: OCRAnalyzeRequest):
    image_bytes = _get_s3(req.receipt_id)
    text = _extract_text_google_vision(image_bytes)
    if not text:
        raise HTTPException(status_code=400, detail="OCR 결과 없음")
    result = _simple_parse(text)
    return result

# ---------------------------
# 시작 시 라우트 확인
# ---------------------------
@app.on_event("startup")
async def _print_routes():
    print("[ROUTES]", [r.path for r in app.router.routes])
