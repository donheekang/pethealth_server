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
from botocore.exceptions import ClientError
from PIL import Image

# -------------------------------
# Config: Environment
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
if not (AWS_KEY and AWS_SECRET and S3_BUCKET):
    print("[WARN] Missing AWS/S3 environment variables")

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

        _svc_info = json.loads(GCV_CREDENTIALS_JSON)
        _creds = Credentials.from_service_account_info(_svc_info)
        vision_client = vision.ImageAnnotatorClient(credentials=_creds)
        print("[INFO] Google Vision ready")
    except Exception as e:
        print("[WARN] Failed to init Google Vision:", e)
else:
    print("[INFO] Google Vision disabled or missing creds")

# -------------------------------
# Google AI (Gemini)
# -------------------------------
GEMINI_MODEL_ANALYZE = None
GEMINI_MODEL_RECO = None
try:
    import google.generativeai as gen
    if GEMINI_API_KEY:
        gen.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL_ANALYZE = gen.GenerativeModel("gemini-1.5-flash")
        GEMINI_MODEL_RECO = gen.GenerativeModel("gemini-1.5-flash")
        print("[INFO] Gemini models ready")
    else:
        print("[INFO] GEMINI_API_KEY not set; Google AI disabled.")
except Exception as e:
    print("[WARN] Gemini import/init failed:", e)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="PetHealthPlus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 도메인 화이트리스트 권장
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
    category: str = "others"  # exam/medication/vaccine/others
    quantity: Optional[float] = None
    unit: Optional[str] = None
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

class SimpleOK(BaseModel):
    ok: bool = True

class CertConfirmBody(BaseModel):
    key: str
    name: str
    size: int

# In-memory demo stores
RECORDS_DB: List[MedicalRecord] = []
CERTS_DB: List[CertConfirmBody] = []

# -------------------------------
# Utils
# -------------------------------
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB

def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def _validate_image_and_maybe_convert(raw: bytes) -> bytes:
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="이미지 용량(15MB) 초과: 사진 크기를 줄여 업로드 해주세요.")
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        return _pil_to_jpeg_bytes(img)
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 형식만 허용됩니다(JPEG/PNG 등).")

def _put_s3(key: str, data: bytes, content_type: str):
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

def _extract_text_google_vision(image_bytes: bytes) -> str:
    if not vision_client:
        return ""
    from google.cloud import vision  # local import for type
    image = vision.Image(content=image_bytes)
    resp = vision_client.text_detection(image=image)
    if resp.error.message:
        print("[Vision] error:", resp.error.message)
        return ""
    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text
    return resp.text_annotations[0].description if resp.text_annotations else ""

# ---------- Gemini prompts ----------
GEMINI_PARSE_SYS = """You are an expert medical receipt parser for veterinary receipts (Korean).
Return STRICT JSON only, no markdown, matching this schema:
{
  "clinicName": string|null,
  "visitDate": string|null,
  "totalAmount": number|null,
  "items": [
    {"id": string, "name": string, "category": "exam"|"medication"|"vaccine"|"others", "quantity": number|null, "unit": string|null, "price": number|null}
  ],
  "notes": string|null
}
Rules:
•⁠  ⁠Amounts are Korean won integers (no commas).
•⁠  ⁠If multiple dates, prefer visit/payment date on receipt.
•⁠  ⁠Category mapping:
  검사/진단/진료/혈검/엑스레이/초음파 -> exam
  주사/약/연고/처방/항생제 -> medication
  백신/접종/예방 -> vaccine
  기타 -> others
•⁠  ⁠If unknown, set null.
•⁠  ⁠Output JSON only.
"""

def _gemini_structured_parse(ocr_text: str) -> Optional[OCRResult]:
    if not GEMINI_MODEL_ANALYZE or not ocr_text.strip():
        return None
    try:
        prompt = f"Raw OCR text (Korean):\n⁠  \n{ocr_text}\n  ⁠"
        resp = GEMINI_MODEL_ANALYZE.generate_content([GEMINI_PARSE_SYS, prompt])
        data = json.loads(resp.text.strip())

        items: List[MedicalItem] = []
        for it in data.get("items", []):
            items.append(MedicalItem(
                id=it.get("id") or uuid.uuid4().hex,
                name=it.get("name") or "항목",
                category=it.get("category") or "others",
                quantity=it.get("quantity"),
                unit=it.get("unit"),
                price=it.get("price"),
            ))

        visit = None
        v = data.get("visitDate")
        if v:
            try:
                visit = datetime.fromisoformat(v)
            except Exception:
                pass

        return OCRResult(
            clinicName=data.get("clinicName"),
            visitDate=visit or datetime.now(timezone.utc),
            items=items,
            notes=data.get("notes"),
            totalAmount=data.get("totalAmount"),
        )
    except Exception as e:
        print("[Gemini] parse fail:", e)
        return None

GEMINI_RECO_SYS = """You are a pet nutrition assistant. Given a pet profile and recent vet records, recommend:
•⁠  ⁠dog food (\"food\"), and supplements (\"supplement\").
Rules: obey allergies strictly; do not invent brand names; provide 2-4 short reasons and 2-4 short tags.
Return STRICT JSON array of recommendations with {id,type,title,subtitle?,reasons[],tags[],deeplink?}. Output JSON only.
"""

def _gemini_recommend(profile: PetProfile, recent: List[MedicalRecord]) -> Optional[List[Recommendation]]:
    if not GEMINI_MODEL_RECO:
        return None
    try:
        payload = {
            "profile": profile.dict(),
            "recentRecords": [r.dict() for r in recent]
        }
        prompt = "Use this input:\n" + json.dumps(payload, ensure_ascii=False, default=str)
        resp = GEMINI_MODEL_RECO.generate_content([GEMINI_RECO_SYS, prompt])
        arr = json.loads(resp.text.strip())
        out: List[Recommendation] = []
        for r in arr:
            out.append(Recommendation(
                id=r.get("id") or uuid.uuid4().hex,
                type=r.get("type") or "food",
                title=r.get("title") or "추천",
                subtitle=r.get("subtitle"),
                reasons=r.get("reasons") or [],
                tags=r.get("tags") or [],
                deeplink=r.get("deeplink"),
            ))
        return out
    except Exception as e:
        print("[Gemini] recommend fail:", e)
        return None

# -------------------------------
# Basic endpoints
# -------------------------------
@app.get("/")
def root():
    return {"message": "PetHealth+ 서버 연결 성공 ✅", "ocr": bool(vision_client)}

@app.get("/health")
def health():
    return {"ok": True, "service": "PetHealthPlus", "version": "1.2.0"}

# -------------------------------
# OCR: upload & analyze
# -------------------------------
@app.post("/ocr/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    raw = _validate_image_and_maybe_convert(raw)
    key = f"receipts/{uuid.uuid4().hex}.jpg"
    _put_s3(key, raw, "image/jpeg")
    return UploadResponse(receipt_id=key)

@app.post("/ocr/analyze", response_model=OCRResult)
async def analyze(req: OCRAnalyzeRequest):
    if not req.receipt_id.startswith("receipts/"):
        raise HTTPException(status_code=400, detail="유효하지 않은 receipt_id 입니다.")
    image_bytes = _get_s3(req.receipt_id)

    text = _extract_text_google_vision(image_bytes) if vision_client else ""
    g_out = _gemini_structured_parse(text) if text else None
    if g_out:
        return g_out
    return _simple_parse(text or "")

# -------------------------------
# Records
# -------------------------------
@app.get("/records", response_model=List[MedicalRecord])
def list_records():
    return RECORDS_DB

@app.post("/records", response_model=MedicalRecord)
def add_record(record: MedicalRecord):
    RECORDS_DB.insert(0, record)
    return record

# -------------------------------
# Recommend
# -------------------------------
@app.post("/recommend", response_model=List[Recommendation])
def recommend(req: RecommendRequest):
    ai = _gemini_recommend(req.profile, req.recentRecords)
    if ai:
        return ai

    # Fallback rules
    recs: List[Recommendation] = []
    allergy = set(a.lower() for a in req.profile.allergies)
    recent_text = " ".join([it.name for r in req.recentRecords for it in r.items])

    if {"chicken","beef","lamb"} & allergy:
        recs.append(Recommendation(
            type="food",
            title="저자극 단백질 사료 추천",
            subtitle="피부/알레르기 관리",
            reasons=["한정 원료(연어/곤충/오리)", "곡물/옥수수/콩 배제", f"알러지: {', '.join(req.profile.allergies)}"],
            tags=["limited-ingredient","allergenic-free"]
        ))
    else:
        recs.append(Recommendation(
            type="food",
            title="균형 잡힌 표준 사료",
            reasons=["오메가3/6 균형", "표준 체중 유지"],
            tags=["balanced","adult"]
        ))

    if "피부" in recent_text or {"chicken","beef"} & allergy:
        recs.append(Recommendation(
            type="supplement",
            title="피부/피모 케어 영양제",
            reasons=["오메가-3/비오틴/아연", "가려움·각질 완화"],
            tags=["skin","itching"]
        ))
    return recs

# -------------------------------
# Certificates: presign & confirm
# -------------------------------
@app.get("/certs/presign")
def certs_presign(
    filename: str = Query(..., description="예: mydoc.pdf"),
    contentType: str = Query("application/pdf")
):
    # key 고정 prefix
    key = f"certs/{uuid.uuid4().hex}_{filename}"
    try:
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": S3_BUCKET, "Key": key, "ContentType": contentType},
            ExpiresIn=60 * 5,  # 5분
        )
        return {"url": url, "key": key}
    except ClientError as e:
        print("[S3] presign error:", e)
        raise HTTPException(status_code=500, detail="Presign failed")

@app.post("/certs/confirm", response_model=SimpleOK)
def certs_confirm(body: CertConfirmBody):
    # 최소 검증 (prefix)
    if not body.key.startswith("certs/"):
        raise HTTPException(status_code=400, detail="유효하지 않은 cert key")
    CERTS_DB.insert(0, body)
    return SimpleOK()

# -------------------------------
# Simple fallback parser
# -------------------------------
def _simple_parse(text: str) -> OCRResult:
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
                result.visitDate = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass
            break

    # total amount
    won_pat = re.compile(r"(합계|총액|총\s*합계)\s*[:\-]?\s*([\d,]+)")
    for ln in reversed(lines):
        m = won_pat.search(ln)
        if m:
            amt = int(m.group(2).replace(",", ""))
            result.totalAmount = amt
            break

    # items
    price_tail = re.compile(r"(.+?)\s+([\d,]{2,})\s*$")
    for ln in lines:
        m = price_tail.match(ln)
        if not m:
            continue
        name = m.group(1)
        if any(k in name for k in ["합계", "총", "총액"]):
            continue
        try:
            price = int(m.group(2).replace(",", ""))
        except ValueError:
            continue
        result.items.append(MedicalItem(name=name, price=price))

    if not result.visitDate:
        result.visitDate = datetime.now(timezone.utc)
    if lines:
        head = "\n".join(lines[:6])
        if len(head) > 10:
            result.notes = head
    return result
