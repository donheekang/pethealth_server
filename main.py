# main.py
import io
import os
import re
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import boto3
from botocore.client import Config
from PIL import Image

# ---------------------------
# 운영/개발 스위치
# ---------------------------
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"   # 프론트 연동/라우팅 테스트는 true
MAX_IMAGE_BYTES = 15 * 1024 * 1024

# ---------------------------
# Google Vision
# ---------------------------
GCV_ENABLED = os.getenv("GCV_ENABLED", "true").lower() == "true"
GCV_CREDENTIALS_JSON = os.getenv("GCV_CREDENTIALS_JSON", "")
vision_client = None
vision = None  # for typing/reference

if not STUB_MODE and GCV_ENABLED and GCV_CREDENTIALS_JSON:
    try:
        from google.cloud import vision as _vision
        from google.oauth2.service_account import Credentials
        creds_info = json.loads(GCV_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(creds_info)
        vision_client = _vision.ImageAnnotatorClient(credentials=creds)
        vision = _vision
        print("[INFO] ✅ Google Vision ready")
    except Exception as e:
        print("[WARN] Vision init failed:", e)
else:
    print("[INFO] Google Vision disabled or STUB_MODE on")

# ---------------------------
# AWS S3
# ---------------------------
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")

s3_client = None
if not STUB_MODE:
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        config=Config(signature_version="s3v4"),
    )

# ---------------------------
# FastAPI
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
# 헬스체크
# ---------------------------
health_router = APIRouter()

@health_router.get("/health")
def health():
    return {"ok": True, "service": "PetHealthPlus", "version": "1.0.0", "stub": STUB_MODE}

app.include_router(health_router)

# ---------------------------
# 기본 루트
# ---------------------------
@app.get("/")
def root():
    return {"message": "PetHealth+ 서버 연결 성공 ✅", "ocr": bool(vision_client), "stub": STUB_MODE}

# ---------------------------
# 데이터 모델
# ---------------------------
class MedicalItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    category: str = "others"  # exam/medication/vaccine/others
    price: Optional[int] = None

class OCRResult(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[datetime] = None
    items: List[MedicalItem] = Field(default_factory=list)
    notes: Optional[str] = None
    totalAmount: Optional[int] = None

class UploadResponse(BaseModel):
    receipt_id: str

class OCRAnalyzeRequest(BaseModel):
    receipt_id: str

class PetProfile(BaseModel):
    id: str
    name: str
    species: str
    breed: Optional[str] = None
    birthDate: Optional[datetime] = None
    allergies: List[str] = Field(default_factory=list)
    weightKg: Optional[float] = None

class MedicalRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    petId: str
    clinicName: str
    visitDate: datetime
    items: List[MedicalItem] = Field(default_factory=list)
    notes: Optional[str] = None
    totalAmount: Optional[int] = None

class RecommendRequest(BaseModel):
    profile: PetProfile
    recentRecords: List[MedicalRecord] = Field(default_factory=list)

class Recommendation(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: str  # food/supplement/insurance
    title: str
    subtitle: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    deeplink: Optional[str] = None

# 임시 인메모리 DB (데모)
RECORDS_DB: List[MedicalRecord] = []

# ---------------------------
# 유틸
# ---------------------------
def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def _validate_and_to_jpeg(raw: bytes) -> bytes:
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="이미지 용량(15MB) 초과")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return _pil_to_jpeg_bytes(img)
    except Exception:
        return raw  # 멀티파트 등 이미지가 아닐 수도 있으니 그대로

def _put_s3(key: str, data: bytes, content_type: str = "image/jpeg"):
    if STUB_MODE:
        return
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

def _get_s3(key: str) -> bytes:
    if STUB_MODE:
        return b"STUB_IMAGE_BYTES"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"S3 object not found: {e}")

def _extract_text_google_vision(image_bytes: bytes) -> str:
    if not vision_client:
        return ""
    image = vision.Image(content=image_bytes)
    resp = vision_client.text_detection(image=image)
    if getattr(resp, "error", None) and resp.error.message:
        print("[Vision] error:", resp.error.message)
        return ""
    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text
    return ""

def _simple_parse(text: str) -> OCRResult:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    result = OCRResult(items=[])

    # 병원명
    for ln in lines:
        if "동물병원" in ln or "병원" in ln:
            result.clinicName = ln
            break

    # 날짜
    date_pat = re.compile(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})")
    for ln in lines:
        m = date_pat.search(ln)
        if m:
            raw = m.group(1).replace("/", "-").replace(".", "-")
            try:
                result.visitDate = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass
            break

    # 총액
    total_pat = re.compile(r"(합계|총액|총\s*합계)\s*[:\-]?\s*([\d,]+)")
    for ln in lines:
        m = total_pat.search(ln)
        if m:
            result.totalAmount = int(m.group(2).replace(",", ""))
            break

    # 항목 (맨 뒤 금액 패턴)
    price_tail = re.compile(r"(.+?)\s+([\d,]{3,})$")
    for ln in lines:
        m = price_tail.match(ln)
        if m:
            name, price = m.groups()
            if not any(k in name for k in ["합계", "총액", "소계"]):
                result.items.append(MedicalItem(name=name, price=int(price.replace(",", ""))))

    if not result.visitDate:
        result.visitDate = datetime.now(timezone.utc)
    if not result.clinicName:
        result.clinicName = "동물병원"
    return result

# ---------------------------
# OCR 업로드/분석 (정식 플로우: S3 경유)
# ---------------------------
@app.post("/ocr/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if STUB_MODE:
        return UploadResponse(receipt_id="receipts/STUB.jpg")

    raw = await file.read()
    raw = _validate_and_to_jpeg(raw)
    receipt_id = f"receipts/{uuid.uuid4().hex}.jpg"
    _put_s3(receipt_id, raw)
    return UploadResponse(receipt_id=receipt_id)

@app.post("/ocr/analyze", response_model=OCRResult)
async def analyze(req: OCRAnalyzeRequest):
    if STUB_MODE:
        return OCRResult(
            clinicName="Stub Animal Hospital",
            visitDate=datetime.now(timezone.utc),
            items=[MedicalItem(name="기본 검진", category="exam", price=30000)],
            totalAmount=30000,
            notes="STUB MODE"
        )

    if not req.receipt_id:
        raise HTTPException(status_code=400, detail="receipt_id 누락")

    image_bytes = _get_s3(req.receipt_id)
    text = _extract_text_google_vision(image_bytes) if vision_client else ""
    if not text:
        return OCRResult(
            clinicName=None,
            visitDate=datetime.now(timezone.utc),
            items=[],
            totalAmount=None,
            notes="OCR 결과 없음"
        )
    return _simple_parse(text)

# ---------------------------
# Records (데모)
# ---------------------------
@app.get("/records", response_model=List[MedicalRecord])
def list_records():
    return RECORDS_DB

@app.post("/records", response_model=MedicalRecord)
def add_record(record: MedicalRecord):
    RECORDS_DB.insert(0, record)
    return record

# ---------------------------
# Recommend (간단 룰 데모)
# ---------------------------
@app.post("/recommend", response_model=List[Recommendation])
def recommend(req: RecommendRequest):
    recs: List[Recommendation] = []
    if req.profile.allergies:
        recs.append(Recommendation(
            type="food", title="저자극 단백질 사료",
            subtitle="알레르기 대응",
            reasons=["한정 원료", "곡물/옥수수/콩 배제"],
            tags=["limited-ingredient","allergenic-free"]
        ))
        recs.append(Recommendation(
            type="supplement", title="피부/피모 케어 영양제",
            reasons=["오메가-3/비오틴/아연", "가려움·각질 완화"],
            tags=["skin","itching"]
        ))
    else:
        recs.append(Recommendation(
            type="food", title="균형 잡힌 표준 사료",
            reasons=["오메가3/6 균형","표준 체중 유지"],
            tags=["balanced","adult"]
        ))
    return recs

# ---------------------------
# Diagnostics (Google Vision / 환경 확인)
# ---------------------------
@app.get("/ocr/diag")
def ocr_diag():
    return {
        "stub_mode": STUB_MODE,
        "gcv_enabled": GCV_ENABLED,
        "vision_client": bool(vision_client),
        "s3_bucket": S3_BUCKET,
        "aws_region": AWS_REGION,
    }

# S3를 거치지 않고 업로드 파일을 직접 Vision에 넣어보는 테스트
@app.post("/ocr/test")
async def ocr_test(file: UploadFile = File(...)):
    data = await file.read()
    data = _validate_and_to_jpeg(data)
    text = _extract_text_google_vision(data)
    return {"ok": bool(text), "len": len(text), "sample": text[:300]}

# ---------------------------
# 시작 시 라우트 로그
# ---------------------------
@app.on_event("startup")
async def _print_routes():
    paths = [r.path for r in app.router.routes]
    print("[ROUTES]", paths)
