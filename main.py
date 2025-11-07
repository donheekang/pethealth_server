from _future_ import annotations
import os, io, json, re, base64
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3

# --------- Google Cloud Vision 준비 (옵션) ----------
USE_GCV = os.getenv("GCV_ENABLED", "false").lower() == "true"
GCV_JSON = os.getenv("GCV_CREDENTIALS_JSON")

if USE_GCV and GCV_JSON:
    # Render에서는 파일이 없으니 JSON 문자열을 /tmp에 저장 후 경로로 지정
    gcv_path = "/tmp/gcv.json"
    Path(gcv_path).write_text(GCV_JSON, encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcv_path
    try:
        from google.cloud import vision  # type: ignore
        _gcv_client = vision.ImageAnnotatorClient()
    except Exception as e:
        print("[WARN] GCV import/init failed:", e)
        USE_GCV = False
        _gcv_client = None
else:
    _gcv_client = None

# --------- RapidOCR (백업용) ----------
try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
    _rapidocr = RapidOCR()
except Exception as e:
    print("[WARN] RapidOCR init failed:", e)
    _rapidocr = None

# --------- AWS S3 기본 설정 ----------
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")

if not all([AWS_KEY, AWS_SECRET, AWS_REGION, S3_BUCKET]):
    print("[WARN] Missing AWS envs. S3 features may fail.")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
)

# --------- FastAPI ----------
app = FastAPI(title="PetHealth+ API (GCV)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP 단계에서는 허용, 운영 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Models ----------
class MedicalItem(BaseModel):
    name: str
    category: str = "others"
    quantity: Optional[float] = None
    unit: Optional[str] = None
    price: Optional[int] = None

class OCRResult(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # iOS에서 ISO-8601로 파싱 가능 문자열이면 OK
    items: List[MedicalItem] = []
    notes: Optional[str] = None
    totalAmount: Optional[int] = None

class UploadResp(BaseModel):
    receipt_id: str

class OCRAnalyzeReq(BaseModel):
    receipt_id: str  # S3 객체 키

class RecommendReq(BaseModel):
    profile: Dict
    recentRecords: List[Dict]

# --------- Utils ----------
def s3_put_bytes(key: str, data: bytes, content_type: str = "image/jpeg") -> None:
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)

def s3_get_bytes(key: str) -> bytes:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()

def normalize_price(text: str) -> Optional[int]:
    # "12,300원" -> 12300
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)\s*원", text.replace(" ", ""))
    if not m:
        return None
    return int(m.group(1).replace(",", ""))

def gcv_ocr_lines(image_bytes: bytes) -> List[str]:
    """Google Cloud Vision으로 라인 텍스트 추출"""
    if not _gcv_client:
        return []
    from google.cloud import vision
    img = vision.Image(content=image_bytes)
    resp = _gcv_client.text_detection(image=img)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    # text_annotations[0]는 전체 문서, 이후는 요소들
    lines: List[str] = []
    if resp.text_annotations:
        # 전체 블록을 줄바꿈 기준으로 분리
        full = resp.text_annotations[0].description
        for ln in full.splitlines():
            t = ln.strip()
            if t:
                lines.append(t)
    return lines

def rapid_ocr_lines(image_bytes: bytes) -> List[str]:
    """RapidOCR로 라인 텍스트 추출(백업)"""
    if not _rapidocr:
        return []
    import cv2
    import numpy as np
    img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    res, _ = _rapidocr(img)
    lines = [r[1] for r in res] if res else []
    return lines

def parse_receipt(lines: List[str]) -> OCRResult:
    """
    매우 단순한 규칙 기반 파서 (MVP).
    병원명, 합계, 항목/가격을 라인에서 추출해 OCRResult로 반환.
    """
    result = OCRResult(items=[])

    # 병원명: '동물병원'/'병원' 단어가 포함된 첫 줄
    for ln in lines[:10]:
        if "동물병원" in ln or ln.endswith("병원") or "병원" in ln:
            result.clinicName = ln.strip()
            break

    # 총액
    for ln in reversed(lines):
        price = normalize_price(ln)
        if price:
            # "합계", "총액" 같은 단어가 있으면 더 신뢰
            if any(k in ln for k in ("합계", "총액", "총금액", "결제금액")):
                result.totalAmount = price
                break
            # 마지막 금액으로라도 채움
            if result.totalAmount is None:
                result.totalAmount = price

    # 간단 항목 파싱: "주사", "초음파", "X-ray" 등 키워드 체킹
    CATEGORIES = {
        "vaccine": ["백신", "예방접종", "DHPPL", "라비즈"],
        "exam": ["진찰", "초진", "재진", "X-ray", "엑스레이", "초음파", "혈액", "검사"],
        "medication": ["주사", "주사제", "항생제", "약", "처방"],
    }
    for ln in lines:
        price = normalize_price(ln)
        # 한 줄에서 이름 + 가격이 같이 잡히는 경우
        if price:
            name = ln
            cat = "others"
            for c, keys in CATEGORIES.items():
                if any(k in ln for k in keys):
                    cat = c
                    break
            result.items.append(MedicalItem(name=name, category=cat, price=price))

    # 메모 필드 (긴 문장 요약)
    if lines:
        sample = " / ".join(lines[:6])
        result.notes = sample[:300]

    return result

# --------- Endpoints ----------
@app.get("/")
def home():
    return {"message": "PetHealth+ 서버 연결 성공 ✅"}

@app.get("/health")
def health():
    return {"status": "ok", "gcv_enabled": USE_GCV}

# 영수증 업로드 -> S3 저장 -> receipt_id 반환
@app.post("/ocr/upload", response_model=UploadResp)
async def ocr_upload(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(400, "No file")
    data = await file.read()
    key = f"receipts/{os.urandom(8).hex()}_{file.filename or 'receipt.jpg'}"
    try:
        s3_put_bytes(key, data, content_type=file.content_type or "image/jpeg")
    except Exception as e:
        raise HTTPException(500, f"S3 upload failed: {e}")
    return UploadResp(receipt_id=key)

# 업로드된 영수증을 분석 -> OCRResult 반환
@app.post("/ocr/analyze", response_model=OCRResult)
def ocr_analyze(req: OCRAnalyzeReq):
    try:
        img = s3_get_bytes(req.receipt_id)
    except Exception as e:
        raise HTTPException(404, f"S3 object not found: {e}")

    lines: List[str] = []
    try:
        if USE_GCV:
            lines = gcv_ocr_lines(img)
        if not lines:  # 백업
            lines = rapid_ocr_lines(img)
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")

    if not lines:
        raise HTTPException(422, "텍스트를 인식하지 못했습니다.")

    return parse_receipt(lines)

# (MVP) 메모리 저장소 – 필요시 DB로 대체
_DB_RECORDS: List[Dict] = []

@app.get("/records")
def get_records():
    return _DB_RECORDS

@app.post("/records")
def add_record(record: Dict):
    _DB_RECORDS.insert(0, record)
    return {"ok": True, "count": len(_DB_RECORDS)}

@app.post("/recommend")
def recommend(req: RecommendReq):
    # 아주 단순한 데모 로직
    profile = req.profile
    allergies = set(map(str.lower, profile.get("allergies", [])))
    recs = []

    # 사료 추천
    if "chicken" in allergies or "beef" in allergies:
        recs.append({
            "type": "food",
            "title": "저자극 단백질 사료 (연어/오리)",
            "subtitle": "닭/소고기 알러지 회피",
            "reasons": ["알러지 프로필 기반", "단일 단백질", "피부 가려움 완화 기대"],
            "tags": ["그레인프리", "저자극", "연어/오리"],
        })
    else:
        recs.append({
            "type": "food",
            "title": "밸런스 고단백 사료",
            "subtitle": "활동량 높은 반려견",
            "reasons": ["체중/체형 고려", "기초 영양밸런스 적합"],
            "tags": ["고단백", "오메가3"],
        })

    # 영양제 추천
    recs.append({
        "type": "supplement",
        "title": "관절 케어 (글루코사민+MSM)",
        "reasons": ["소형견 슬개골 예방", "활동량↑"],
        "tags": ["관절", "MSM"],
    })

    # 보험 추천
    recs.append({
        "type": "insurance",
        "title": "통원형 70% 보장 플랜",
        "subtitle": "예상 진료비 분산",
        "reasons": ["검진/예방접종 히스토리 기반", "처치 항목 커버"],
        "tags": ["보장70%", "통원형"],
    })

    return recs
