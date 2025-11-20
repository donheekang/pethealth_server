from __future__ import annotations

import os
import uuid
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# OCR / LLM
from google.cloud import vision
import google.generativeai as genai


# ============================================================
#  Settings
# ============================================================

class Settings(BaseSettings):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "ap-northeast-2"
    s3_bucket_name: str

    gemini_api_key: Optional[str] = None
    gemini_enabled: bool = False

    google_application_credentials: Optional[str] = None

    class Config:
        env_prefix = ""
        env_file = ".env"
        case_sensitive = False


settings = Settings(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region=os.getenv("AWS_REGION", "ap-northeast-2"),
    s3_bucket_name=os.getenv("S3_BUCKET_NAME", ""),
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    gemini_enabled=os.getenv("GEMINI_ENABLED", "false").lower() == "true",
    google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
)

if not settings.s3_bucket_name:
    raise RuntimeError("S3_BUCKET_NAME 환경변수가 필요합니다.")


# ============================================================
#  S3 Client & Helper
# ============================================================

s3 = boto3.client(
    "s3",
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    region_name=settings.aws_region,
)


def upload_to_s3(key: str, file_bytes: bytes, content_type: str) -> str:
    s3.put_object(
        Bucket=settings.s3_bucket_name,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.s3_bucket_name, "Key": key},
        ExpiresIn=60 * 60 * 24 * 7,  # 7일
    )
    return url


# ============================================================
#  OCR / Gemini 초기화
# ============================================================

vision_client: Optional[vision.ImageAnnotatorClient] = None
if settings.google_application_credentials:
    # GOOGLE_APPLICATION_CREDENTIALS 는 환경변수에 이미 경로가 들어있어야 함
    vision_client = vision.ImageAnnotatorClient()

if settings.gemini_enabled and settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)


# ============================================================
#  Models (iOS DTO 맞춰서 정의)
# ============================================================

class PdfRecord(BaseModel):
    id: str
    petId: str
    title: str
    memo: Optional[str] = None
    s3Url: str
    createdAt: Optional[str] = None


class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsed(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None     # iOS: "yyyy-MM-dd"
    diseaseName: Optional[str] = None
    symptomsSummary: Optional[str] = None
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None


class ReceiptAnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    time: str
    gemini: bool
    ocr: bool


# ============================================================
#  In-memory "DB" (Render 재시작 시 초기화)
# ============================================================

LAB_DB: Dict[str, List[PdfRecord]] = {}
CERT_DB: Dict[str, List[PdfRecord]] = {}


# ============================================================
#  FastAPI App
# ============================================================

app = FastAPI(title="PetHealth+ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  Health / Root
# ============================================================

@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        time=datetime.now(timezone.utc).isoformat(),
        gemini=bool(settings.gemini_enabled and settings.gemini_api_key),
        ocr=vision_client is not None,
    )


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        time=datetime.now(timezone.utc).isoformat(),
        gemini=bool(settings.gemini_enabled and settings.gemini_api_key),
        ocr=vision_client is not None,
    )


# ============================================================
#  검사결과 (Labs)
# ============================================================

@app.get("/labs/list", response_model=List[PdfRecord])
@app.get("/api/labs/list", response_model=List[PdfRecord])
def list_labs(petId: str) -> List[PdfRecord]:
    return LAB_DB.get(petId, [])


@app.post("/lab/upload-pdf", response_model=PdfRecord)
@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form(...),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
) -> PdfRecord:
    try:
        content = await file.read()
        ext = os.path.splitext(file.filename or "")[1] or ".pdf"
        key = f"labs/{petId}/{uuid.uuid4()}{ext}"

        url = upload_to_s3(key, content, "application/pdf")

        record = PdfRecord(
            id=str(uuid.uuid4()),
            petId=petId,
            title=title,
            memo=memo,
            s3Url=url,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )
        LAB_DB.setdefault(petId, []).append(record)
        return record
    except Exception as e:
        print("upload_lab_pdf error:", e)
        raise HTTPException(status_code=500, detail="검사결과 업로드 실패")


# ============================================================
#  증명서 (Certificates)
# ============================================================

@app.get("/cert/list", response_model=List[PdfRecord])
@app.get("/api/cert/list", response_model=List[PdfRecord])
def list_certs(petId: str) -> List[PdfRecord]:
    return CERT_DB.get(petId, [])


@app.post("/cert/upload-pdf", response_model=PdfRecord)
@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form(...),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
) -> PdfRecord:
    try:
        content = await file.read()
        ext = os.path.splitext(file.filename or "")[1] or ".pdf"
        key = f"certs/{petId}/{uuid.uuid4()}{ext}"

        url = upload_to_s3(key, content, "application/pdf")

        record = PdfRecord(
            id=str(uuid.uuid4()),
            petId=petId,
            title=title,
            memo=memo,
            s3Url=url,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )
        CERT_DB.setdefault(petId, []).append(record)
        return record
    except Exception as e:
        print("upload_cert_pdf error:", e)
        raise HTTPException(status_code=500, detail="증명서 업로드 실패")


# ============================================================
#  OCR + Gemini Helpers
# ============================================================

def run_vision_ocr(image_bytes: bytes) -> str:
    if vision_client is None:
        raise HTTPException(status_code=500, detail="OCR 사용 불가 (GOOGLE_APPLICATION_CREDENTIALS 확인)")

    img = vision.Image(content=image_bytes)
    res = vision_client.text_detection(image=img)
    if res.error.message:
        print("vision error:", res.error.message)
        raise HTTPException(status_code=500, detail="OCR 실패")

    text = res.full_text_annotation.text or ""
    return text


PROMPT = """
너는 동물병원 영수증 텍스트를 구조화하는 AI다.
입력된 OCR 텍스트에서 다음 정보만 JSON으로 추출해라.

필수:
•⁠  ⁠clinicName: 병원 이름
•⁠  ⁠visitDate: yyyy-MM-dd 형식 날짜 (없으면 null)
•⁠  ⁠diseaseName: 진료 이름 또는 영수증 상 대표 진료명 (없으면 null)
•⁠  ⁠symptomsSummary: 한글로 간단한 요약 (없으면 null)
•⁠  ⁠items: [
    { "name": 항목명, "price": 숫자 or null }
]
•⁠  ⁠totalAmount: 총 금액 (숫자 or null)

주어지지 않은 값은 null.
반드시 JSON 문자열만 출력해라. 설명/말붙임 금지.
"""


def parse_with_gemini(text: str) -> ReceiptParsed:
    if not (settings.gemini_enabled and settings.gemini_api_key):
        raise HTTPException(status_code=500, detail="Gemini 사용 불가 (환경 설정 확인)")

    model = genai.GenerativeModel("gemini-1.5-flash")
    res = model.generate_content(f"{PROMPT}\n\n===== OCR TEXT =====\n{text}")

    raw = res.text
    try:
        data = json.loads(raw)
        return ReceiptParsed(**data)
    except Exception as e:
        print("Gemini JSON parse error:", e, "raw:", raw)
        raise HTTPException(status_code=500, detail="Gemini JSON 파싱 실패")


def fallback_parse(text: str) -> ReceiptParsed:
    """Gemini 실패 시 최소한 병원명/날짜/합계 정도 추출하기 위한 단순 파서."""
    import re

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    clinic = None
    for ln in lines:
        if "동물병원" in ln:
            clinic = ln
            break

    date = None
    m = re.search(r"(20\d{2}[./-]\d{1,2}[./-]\d{1,2})", text)
    if m:
        raw = m.group(1).replace(".", "-").replace("/", "-")
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d")
            date = dt.strftime("%Y-%m-%d")
        except:
            pass

    total = None
    mt = re.search(r"(\d{1,3}(,\d{3})+)\s*원", text)
    if mt:
        total = int(mt.group(1).replace(",", ""))

    items: List[ReceiptItem] = []
    for ln in lines:
        if any(k in ln for k in ["진료", "주사", "백신", "약", "검사"]):
            items.append(ReceiptItem(name=ln, price=None))

    return ReceiptParsed(
        clinicName=clinic,
        visitDate=date,
        diseaseName=None,
        symptomsSummary="Gemini 실패로 단순 규칙 기반 파싱을 사용했습니다.",
        items=items,
        totalAmount=total,
    )


# ============================================================
#  병원 영수증 분석 API  (/api/receipt/analyze)
# ============================================================

@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str = Form(...),
    image: UploadFile = File(...),
) -> ReceiptAnalyzeResponse:

    if image.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="image/jpeg 또는 image/png만 허용됩니다.")

    try:
        img_bytes = await image.read()

        # 1) 영수증 이미지 S3 업로드 → iOS에서 "영수증 이미지 보기"에 사용
        ext = ".jpg"
        if image.filename and image.filename.lower().endswith(".png"):
            ext = ".png"
        key = f"receipts/{petId}/{uuid.uuid4()}{ext}"
        s3_url = upload_to_s3(key, img_bytes, image.content_type or "image/jpeg")

        # 2) Vision OCR → 텍스트
        text = run_vision_ocr(img_bytes)
        if not text.strip():
            raise HTTPException(status_code=500, detail="OCR 결과 텍스트 없음")

        # 3) Gemini로 구조화 파싱, 실패하면 fallback
        try:
            parsed = parse_with_gemini(text)
            notes = "Gemini 파싱 완료"
        except HTTPException as e:
            # Gemini 설정 문제라면 그대로 에러 내보냄
            if "Gemini 사용 불가" in str(e.detail):
                raise
            # 나머지는 fallback
            parsed = fallback_parse(text)
            notes = "Gemini 실패 → fallback 파싱"

        resp = ReceiptAnalyzeResponse(
            petId=petId,
            s3Url=s3_url,
            parsed=parsed,
            notes=notes,
        )
        return resp

    except HTTPException:
        raise
    except Exception as e:
        print("analyze_receipt error:", e)
        raise HTTPException(status_code=500, detail="영수증 분석 실패")
