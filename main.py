import io
import os
import re
import json
import uuid
import base64
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.client import Config
from PIL import Image

# ---- Google Vision (Service Account JSON via ENV) ----
GCV_ENABLED = os.getenv("GCV_ENABLED", "true").lower() == "true"
GCV_CREDENTIALS_JSON = os.getenv("GCV_CREDENTIALS_JSON", "")

if GCV_ENABLED and GCV_CREDENTIALS_JSON:
    from google.cloud import vision
    from google.oauth2.service_account import Credentials

    try:
        _svc_info = json.loads(GCV_CREDENTIALS_JSON)
        _creds = Credentials.from_service_account_info(_svc_info)
        vision_client = vision.ImageAnnotatorClient(credentials=_creds)
    except Exception as e:
        print("[WARN] Failed to init Google Vision client:", e)
        vision_client = None
else:
    vision_client = None
    print("[INFO] Google Vision disabled or missing creds")

# ---- AWS S3 ----
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")

if not (AWS_KEY and AWS_SECRET and S3_BUCKET):
    print("[WARN] Missing AWS/S3 environment variables")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    config=Config(signature_version="s3v4"),
)

# ---- FastAPI ----
app = FastAPI(title="PetHealthPlus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class MedicalItem(BaseModel):
    id: str = uuid.uuid4().hex
    name: str
    category: str = "others"  # exam/medication/vaccine/others
    quantity: Optional[float] = None
    unit: Optional[str] = None
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

class PetProfile(BaseModel):
    id: str
    name: str
    species: str
    breed: Optional[str] = None
    birthDate: Optional[datetime] = None
    allergies: List[str] = []
    weightKg: Optional[float] = None

class MedicalRecord(BaseModel):
    id: str = uuid.uuid4().hex
    petId: str
    clinicName: str
    visitDate: datetime
    items: List[MedicalItem] = []
    notes: Optional[str] = None
    totalAmount: Optional[int] = None

class RecommendRequest(BaseModel):
    profile: PetProfile
    recentRecords: List[MedicalRecord] = []

class Recommendation(BaseModel):
    id: str = uuid.uuid4().hex
    type: str  # food/supplement/insurance
    title: str
    subtitle: Optional[str] = None
    reasons: List[str] = []
    tags: List[str] = []
    deeplink: Optional[str] = None

# ---------- In-memory storage (데모) ----------
RECORDS_DB: List[MedicalRecord] = []


# ---------- Utils ----------
def _put_s3(key: str, data: bytes, content_type: str = "image/jpeg"):
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    except Exception as e:
        print("[S3] put_object error:", e)
        raise HTTPException(status_code=500, detail="S3 upload failed")

def _get_s3(key: str) -> bytes:
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception as e:
        print("[S3] get_object error:", e)
        raise HTTPException(status_code=404, detail="S3 object not found")

def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def _extract_text_google_vision(image_bytes: bytes) -> str:
    if not vision_client:
        # 안전장치: 비전 비활성 시 기본값
        return ""
    image = vision.Image(content=image_bytes)
    resp = vision_client.text_detection(image=image)
    if resp.error.message:
        print("[Vision] error:", resp.error.message)
        return ""
    return resp.full_text_annotation.text if resp.full_text_annotation and resp.full_text_annotation.text else (
        resp.text_annotations[0].description if resp.text_annotations else ""
    )

def _simple_parse(text: str) -> OCRResult:
    """
    매우 단순한 규칙 파서 (데모용)
    - 병원명: 첫 줄 혹은 '동물병원' 포함 줄
    - 날짜: YYYY.MM.DD / YYYY-MM-DD / YYYY/MM/DD
    - 총액: '합계|총액|총 합계' 숫자
    - 항목: 줄단위로 이름 + 금액
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result = OCRResult(items=[])

    # clinic
    for ln in lines[:5]:
        if ("동물병원" in ln) or ("병원" in ln):
            result.clinicName = ln
            break
    if not result.clinicName and lines:
        result.clinicName = lines[0]

    # date
    date_pat = re.compile(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})")
    for ln in lines:
        m = date_pat.search(ln)
        if m:
            raw = m.group(1).replace("/", "-").replace(".", "-")
            try:
                result.visitDate = datetime.strptime(raw, "%Y-%m-%d")
            except:
                try:
                    result.visitDate = datetime.strptime(raw, "%Y-%m-%d")
                except:
                    pass
            break

    # total amount
    won_pat = re.compile(r"(합계|총액|총\s*합계)\s*[:\-]?\s*([\d,]+)")
    for ln in lines[::-1]:
        m = won_pat.search(ln)
        if m:
            amt = int(m.group(2).replace(",", ""))
            result.totalAmount = amt
            break

    # items (아주 단순화: '... 12,000' 형식)
    price_tail = re.compile(r"(.+?)\s+([\d,]{2,})\s*$")
    for ln in lines:
        m = price_tail.match(ln)
        if m:
            name = m.group(1)
            price = int(m.group(2).replace(",", ""))
            # 총액/합계 라인은 제외
            if any(k in name for k in ["합계", "총", "총액"]):
                continue
            result.items.append(MedicalItem(name=name, price=price))

    # notes: 첫 4~6줄 정도를 묶어서 간단히
    if lines:
        head = "\n".join(lines[:6])
        if len(head) > 10:
            result.notes = head

    # 날짜 없으면 오늘
    if not result.visitDate:
        result.visitDate = datetime.utcnow()
    return result


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "service": "PetHealthPlus", "version": "1.0.0"}

@app.post("/ocr/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """
    이미지 업로드 → S3 저장 → receipt_id 반환
    """
    raw = await file.read()

    # 혹시 이미지가 HEIC/PNG 등일 경우 JPEG로 변환
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        raw = _pil_to_jpeg_bytes(img)
    except Exception:
        pass

    receipt_id = f"receipts/{uuid.uuid4().hex}.jpg"
    _put_s3(receipt_id, raw, "image/jpeg")
    return UploadResponse(receipt_id=receipt_id)

@app.post("/ocr/analyze", response_model=OCRResult)
async def analyze(req: OCRAnalyzeRequest):
    """
    S3에서 이미지 다운로드 → Google Vision OCR → 텍스트 파싱 → 구조화 결과 반환
    """
    image_bytes = _get_s3(req.receipt_id)
    text = _extract_text_google_vision(image_bytes) if GCV_ENABLED else ""
    if not text:
        # 비전 꺼져있거나 실패 시: 아주 간단한 fallback (빈 텍스트)
        text = ""

    result = _simple_parse(text)
    return result

@app.get("/records", response_model=List[MedicalRecord])
def list_records():
    return RECORDS_DB

@app.post("/records", response_model=MedicalRecord)
def add_record(record: MedicalRecord):
    RECORDS_DB.insert(0, record)
    return record

@app.post("/recommend", response_model=List[Recommendation])
def recommend(req: RecommendRequest):
    """
    매우 단순한 룰 기반 데모:
    - 알러지 키워드 있으면 사료/영양제 태그/이유 변경
    - 최근 항목에 '백신'/'접종' 있으면 예방관리 권고
    - 총액이 높으면 보험 추천
    """
    recs: List[Recommendation] = []

    allergy = set(a.lower() for a in req.profile.allergies)
    recent_text = " ".join([it.name for r in req.recentRecords for it in r.items])

    # Food
    if {"chicken", "beef", "lamb"} & allergy:
        recs.append(Recommendation(
            type="food",
            title="저자극 단백질 사료 추천",
            subtitle="피부염/단백질 알레르기 대응",
            reasons=[
                "단백질 제한(연어/곤충/오리 베이스)",
                "곡물·옥수수·콩 성분 없음",
                f"알러지: {', '.join(req.profile.allergies)}"
            ],
            tags=["limited-ingredient", "allergenic-free"],
            deeplink="https://your-commerce/food/low-irritation"
        ))
    else:
        recs.append(Recommendation(
            type="food",
            title="표준 성견용 균형 사료",
            reasons=["필수 아미노산/오메가3/6 균형", "표준 체중 유지"],
            tags=["balanced", "adult"]
        ))

    # Supplement
    if "피부" in recent_text or {"chicken", "beef"}.intersection(allergy):
        recs.append(Recommendation(
            type="supplement",
            title="피부/피모 케어 영양제",
            reasons=["오메가-3, 비오틴, 아연 포함", "가려움/각질 완화"],
            tags=["skin", "itching"],
            deeplink="https://your-commerce/supplement/skin"
        ))

    # Insurance
    big_total = False
    for r in req.recentRecords:
        if r.totalAmount and r.totalAmount >= 200000:
            big_total = True
            break
    if big_total:
        recs.append(Recommendation(
            type="insurance",
            title="수술/입원 집중형 보험",
            subtitle="고액 진료 대비",
            reasons=["최근 고액 진료 발생", "수술/입원 보장 강화 필요"],
            tags=["surgery", "inpatient"]
        ))

    return recs
