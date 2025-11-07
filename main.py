# -- coding: utf-8 --
"""
PetHealth+ FastAPI MVP (Render 배포용)
•⁠  ⁠업로드: /ocr/upload (S3)
•⁠  ⁠분석:   /ocr/analyze (RapidOCR)
•⁠  ⁠기록:   /records (GET/POST)
•⁠  ⁠추천:   /recommend (룰 기반)
•⁠  ⁠상태:   /health
"""
import os, io, json, time, uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# ===== S3 =====
import boto3

# ===== OCR =====
from rapidocr_onnxruntime import RapidOCR

# ---------- 환경변수 ----------
AWS_KEY      = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET   = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION   = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET    = os.getenv("S3_BUCKET_NAME", "pethealthplus-files")

# 로컬 저장 위치(간이 DB)
DATA_DIR = Path(_file_).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
RECORDS_JSON = DATA_DIR / "records.json"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# ---------- S3 클라이언트 ----------
def s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )

# ---------- 모델 ----------
class MedicalItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str = "others"   # exam/medication/vaccine/others
    quantity: Optional[float] = None
    unit: Optional[str] = None
    price: Optional[int] = None

class MedicalRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    petId: str
    clinicName: str
    visitDate: str            # ISO date string
    items: List[MedicalItem] = []
    notes: Optional[str] = None
    totalAmount: Optional[int] = None

class OCRAnalyzeRequest(BaseModel):
    receipt_id: str

class OCRResult(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None
    items: List[MedicalItem] = []
    notes: Optional[str] = None
    totalAmount: Optional[int] = None

class RecommendRequest(BaseModel):
    profile: dict
    recentRecords: List[MedicalRecord] = []

class Recommendation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str   # food/supplement/insurance
    title: str
    subtitle: Optional[str] = None
    reasons: List[str] = []
    tags: List[str] = []
    deeplink: Optional[str] = None

class UploadResponse(BaseModel):
    receipt_id: str

# ---------- 간이 저장소 ----------
def load_records() -> List[dict]:
    if RECORDS_JSON.exists():
        return json.loads(RECORDS_JSON.read_text("utf-8"))
    return []

def save_records(lst: List[dict]):
    RECORDS_JSON.write_text(json.dumps(lst, ensure_ascii=False, indent=2), "utf-8")

# ---------- OCR ----------
_ocr = RapidOCR()

def run_ocr(image_bytes: bytes) -> List[str]:
    """RapidOCR 실행 후 줄 단위 텍스트 리스트 반환"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # RapidOCR -> (result, time)
    res, _ = _ocr(img)
    # res: [[text, score, [box]], ...]
    lines = [r[0] for r in res] if res else []
    return lines

def parse_receipt_to_ocrresult(lines: List[str]) -> OCRResult:
    """
    매우 단순 파서 (MVP): 
    - 병원명: '동물병원' 포함 줄
    - 날짜: yyyy-mm-dd / yyyy.mm.dd / yyyy/mm/dd
    - 금액: '원' 또는 숫자 4자리 이상
    - 항목: '주사','처방','진료','백신' 등 키워드 매칭
    """
    import re
    clinic = next((l for l in lines if "동물병원" in l or "동물 병원" in l or "애견병원" in l), None)

    date_pat = re.compile(r"(20\d{2}[-./](0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01]))")
    date = None
    for l in lines:
        m = date_pat.search(l)
        if m:
            date = m.group(1).replace(".", "-").replace("/", "-")
            break

    total = None
    for l in lines[::-1]:
        if "원" in l:
            nums = re.findall(r"\d{2,}", l.replace(",", ""))
            if nums:
                total = int(nums[-1])
                break

    items: List[MedicalItem] = []
    keywords = {
        "vaccine": ["백신", "접종"],
        "exam": ["검사", "진료"],
        "medication": ["처방", "주사", "약"],
    }
    for l in lines:
        for cat, keys in keywords.items():
            if any(k in l for k in keys):
                # 가격 추출
                price = None
                m2 = re.search(r"(\d[\d,]{2,})\s*원?", l)
                if m2:
                    price = int(m2.group(1).replace(",", ""))
                items.append(MedicalItem(name=l[:40], category=cat, price=price))
                break

    return OCRResult(
        clinicName=clinic or "동물병원",
        visitDate=date or time.strftime("%Y-%m-%d"),
        items=items or [MedicalItem(name="진료", category="exam", price=total)],
        notes=None,
        totalAmount=total
    )

# ---------- FastAPI ----------
app = FastAPI(title="PetHealth+ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 운영 시 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 엔드포인트 ----------
@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}

# 1) 업로드: S3 저장 후 receipt_id 반환
@app.post("/ocr/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "이미지 파일만 업로드해주세요.")
    data = await file.read()

    # S3 업로드
    rid = str(uuid.uuid4())
    key = f"uploads/{rid}.jpg"
    try:
        s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType="image/jpeg")
    except Exception as e:
        # 로컬에라도 저장(개발 테스트용)
        (UPLOADS_DIR / f"{rid}.jpg").write_bytes(data)
    return UploadResponse(receipt_id=rid)

# 2) 분석: S3에서 내려받아 OCR → 결과 구조화
@app.post("/ocr/analyze", response_model=OCRResult)
def analyze(req: OCRAnalyzeRequest):
    key = f"uploads/{req.receipt_id}.jpg"
    try:
        obj = s3_client().get_object(Bucket=S3_BUCKET, Key=key)
        img_bytes = obj["Body"].read()
    except Exception:
        # 로컬 백업에서 찾기
        local = UPLOADS_DIR / f"{req.receipt_id}.jpg"
        if not local.exists():
            raise HTTPException(404, "이미지를 찾을 수 없습니다.")
        img_bytes = local.read_bytes()

    lines = run_ocr(img_bytes)
    return parse_receipt_to_ocrresult(lines)

# 3) 기록: GET/POST (간이 JSON 저장)
@app.get("/records", response_model=List[MedicalRecord])
def list_records():
    return load_records()

@app.post("/records", response_model=MedicalRecord)
def add_record(rec: MedicalRecord):
    lst = load_records()
    lst.insert(0, rec.model_dump())
    save_records(lst)
    return rec

# 4) 추천: 간단 룰 기반(알러지, 항목, 금액 등)
@app.post("/recommend", response_model=List[Recommendation])
def recommend(req: RecommendRequest):
    pet = req.profile or {}
    allergies = set([a.lower() for a in pet.get("allergies", [])])

    has_skin = any("피부" in (i.name if isinstance(i, dict) else i.name) for r in req.recentRecords for i in r.items)
    high_total = any((r.totalAmount or 0) >= 100000 for r in req.recentRecords)

    recs: List[Recommendation] = []

    # 사료 추천 (알러지 고려)
    if "chicken" in allergies or "소고기" in allergies or "beef" in allergies:
        recs.append(Recommendation(
            type="food",
            title="저자극 단백질 사료",
            subtitle="알러지 성분 제외(치킨/소고기 Free)",
            reasons=["알러지 이력 기반", "소화기 부담↓", "피부 컨디션 개선"],
            tags=["그레인프리", "하이포알러제닉"],
            deeplink="https://store.example.com/food-hypo"
        ))
    else:
        recs.append(Recommendation(
            type="food",
            title="피모 케어 사료",
            subtitle="오메가3·비오틴 강화",
            reasons=["피부·모질 개선", "일반 체질에 적합"],
            tags=["오메가3", "비오틴"],
            deeplink="https://store.example.com/food-skin"
        ))

    # 영양제 추천 (피부/관절 등)
    if has_skin:
        recs.append(Recommendation(
            type="supplement",
            title="피부·모질 영양제",
            subtitle="오메가3 + 아연 + 비오틴",
            reasons=["피부 트러블 이력 감지", "피모 윤기 개선"],
            tags=["오메가3","아연","비오틴"],
            deeplink="https://store.example.com/supp-skin"
        ))

    # 보험 추천 (고액 진료 경험)
    if high_total:
        recs.append(Recommendation(
            type="insurance",
            title="실속형 반려동물 보험",
            subtitle="연간 보장 1,000만원 / 자기부담 20%",
            reasons=["최근 고액 진료 감지", "예상 의료비 리스크 완화"],
            tags=["자기부담 20%","연 1,000만원"],
            deeplink="https://store.example.com/ins-basic"
        ))

    if not recs:
        recs.append(Recommendation(
            type="food",
            title="균형 영양 사료",
            subtitle="활동량 보통, 표준 체중 가정",
            reasons=["기본 건강상태 가정", "가격대비 성능 우수"],
            tags=["표준형","가성비"],
            deeplink="https://store.example.com/food-basic"
        ))
    return recs
