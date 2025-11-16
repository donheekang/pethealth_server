"""
PetHealth+ 서버 (Render / FastAPI)

•⁠  ⁠/health                      : 헬스체크
•⁠  ⁠/api/ocr/receipt             : 영수증 이미지 업로드 + S3 저장 + GCV OCR + 간단 파싱
•⁠  ⁠/api/upload/pdf/{kind}       : 검사결과 / 증명서 PDF S3 업로드
•⁠  ⁠/api/ai/health-insight       : 체중, 진료 텍스트 등 보내면 Gemini로 리포트 생성

환경변수(렌더에 이미 설정해 둔 값 사용):
•⁠  ⁠AWS_ACCESS_KEY_ID
•⁠  ⁠AWS_SECRET_ACCESS_KEY
•⁠  ⁠AWS_REGION
•⁠  ⁠S3_BUCKET_NAME
•⁠  ⁠GCV_CREDENTIALS_JSON  (Google Vision 서비스계정 JSON 전문)
•⁠  ⁠GCV_ENABLED           ("true"/"false")
•⁠  ⁠GEMINI_API_KEY        (AI Studio에서 받은 키)
•⁠  ⁠STUB_MODE             ("true" 이면 외부 API 대신 더미 응답)
"""

from _future_ import annotations

import io
import json
import os
import re
import uuid
from datetime import datetime
from typing import List, Optional

import boto3
from botocore.client import Config
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Google Cloud Vision ---
from google.cloud import vision
from google.oauth2 import service_account

# --- Gemini (Google GenAI SDK) ---
from google import genai  # google-genai 패키지

# ---------------------------
# 환경변수 / 전역 설정
# ---------------------------
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

GCV_ENABLED = os.getenv("GCV_ENABLED", "true").lower() == "true"
GCV_CREDENTIALS_JSON = os.getenv("GCV_CREDENTIALS_JSON")

# ---------------------------
# S3 클라이언트
# ---------------------------
s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME:
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

# ---------------------------
# Google Vision 클라이언트
# ---------------------------
vision_client: Optional[vision.ImageAnnotatorClient] = None
if not STUB_MODE and GCV_ENABLED and GCV_CREDENTIALS_JSON:
    try:
        info = json.loads(GCV_CREDENTIALS_JSON)
        creds = service_account.Credentials.from_service_account_info(info)
        vision_client = vision.ImageAnnotatorClient(credentials=creds)
    except Exception as e:
        # 초기화 실패 시 서버는 계속 뜨고, 요청 시에만 에러 반환
        print("[WARN] Vision client init failed:", e)

# ---------------------------
# Gemini 클라이언트
# ---------------------------
gemini_client: Optional[genai.Client] = None
if not STUB_MODE:
    try:
        # GEMINI_API_KEY 는 SDK가 자동으로 읽음 (환경변수) :contentReference[oaicite:0]{index=0}
        gemini_client = genai.Client()
    except Exception as e:
        print("[WARN] Gemini client init failed:", e)


# ============================================================
# Pydantic 모델 (iOS ↔️ 서버 통신용)
# ============================================================
class OCRMedicalItem(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptOCRResponse(BaseModel):
    pet_id: Optional[str] = None
    s3_url: str
    clinic_name: Optional[str] = None
    visit_date: Optional[str] = None  # "2025-11-16"
    total_amount: Optional[int] = None
    items: List[OCRMedicalItem] = []
    raw_text: str


class PDFUploadResponse(BaseModel):
    kind: str  # "lab" or "certificate"
    s3_url: str
    filename: str


class WeightPoint(BaseModel):
    date: str  # ISO date string
    weight: float


class HealthInsightRequest(BaseModel):
    pet_name: str
    species: str
    birth_date: Optional[str] = None
    allergies: List[str] = []
    weight_history: List[WeightPoint] = Field(default_factory=list)

    # 서버에서 OCR한 텍스트들 (영수증)
    receipt_texts: List[str] = Field(default_factory=list)
    # 검사결과 / 증명서에 대한 요약 텍스트
    lab_summaries: List[str] = Field(default_factory=list)
    certificate_summaries: List[str] = Field(default_factory=list)


class HealthInsightResponse(BaseModel):
    summary: str  # 전체 요약
    # 나중에 프론트에서 bullet 로 나누기 쉽게 별도 필드
    bullets: List[str] = Field(default_factory=list)


# ============================================================
# 유틸 함수
# ============================================================
def upload_to_s3(folder: str, data: bytes, content_type: str) -> str:
    if not s3_client or not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3 설정이 되어 있지 않습니다.")

    ext = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "application/pdf": "pdf",
    }.get(content_type, "bin")

    key = f"{folder}/{uuid.uuid4().hex}.{ext}"

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
    except Exception as e:
        print("[ERROR] S3 upload failed:", e)
        raise HTTPException(status_code=500, detail="파일 업로드에 실패했습니다.")

    # 일반적인 S3 퍼블릭 URL 패턴 (버킷 퍼블릭/프리사인 고려해서 필요시 변경)
    url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return url


def parse_receipt_text(text: str):
    """
    아주 간단한 영수증 파서 (한국 동물병원 영수증 기준 대충 추출)
    - 클리닉 이름: 첫 줄 또는 '동물병원' 포함된 줄
    - 날짜: YYYY-MM-DD, YYYY.MM.DD, YY/MM/DD 형태
    - 총 금액: '합계', '총액' 이 있는 줄의 숫자
    - 항목: 금액이 포함된 줄들을 name/price 로 분리 (완전 정확하진 않음)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clinic_name = None
    visit_date = None
    total_amount = None
    items: List[OCRMedicalItem] = []

    # 클리닉 이름
    for ln in lines[:5]:
        if "동물병원" in ln or "애견" in ln or "동물 메디컬" in ln:
            clinic_name = ln
            break
    if clinic_name is None and lines:
        clinic_name = lines[0]

    # 날짜
    date_patterns = [
        r"(\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})",
        r"(\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})",
    ]
    for ln in lines:
        for pat in date_patterns:
            m = re.search(pat, ln)
            if m:
                raw = m.group(1)
                visit_date = normalize_date(raw)
                break
        if visit_date:
            break

    # 총액
    for ln in lines[::-1]:
        if any(k in ln for k in ["합계", "총액", "카드", "현금"]):
            nums = re.findall(r"[\d,]+", ln)
            if nums:
                try:
                    total_amount = int(nums[-1].replace(",", ""))
                    break
                except ValueError:
                    continue

    # 항목 (굉장히 러프한 규칙)
    for ln in lines:
        nums = re.findall(r"[\d,]{3,}", ln)
        if not nums:
            continue
        try:
            price = int(nums[-1].replace(",", ""))
        except ValueError:
            continue
        # 숫자를 제거한 나머지 문자열을 이름으로
        name = re.sub(r"[\d,]+", "", ln).replace("원", "").strip()
        if not name:
            name = "항목"
        items.append(OCRMedicalItem(name=name, price=price))

    return clinic_name, visit_date, total_amount, items


def normalize_date(raw: str) -> Optional[str]:
    raw = raw.replace(".", "-").replace("/", "-")
    parts = raw.split("-")
    try:
        if len(parts[0]) == 2:  # 예: 25-11-16
            year = 2000 + int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
        else:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def call_gcv_ocr(image_bytes: bytes) -> str:
    if STUB_MODE or not GCV_ENABLED or not vision_client:
        # 개발용 더미 응답
        return "토리동물병원\n2025-11-16\n진료 15,000원\n피부약 22,000원\n합계 37,000원"

    try:
        img = vision.Image(content=image_bytes)
        resp = vision_client.text_detection(image=img)
        if resp.error.message:
            print("[ERROR] GCV:", resp.error.message)
            raise HTTPException(status_code=500, detail="OCR 중 오류가 발생했습니다.")
        if resp.full_text_annotation and resp.full_text_annotation.text:
            return resp.full_text_annotation.text
        if resp.text_annotations:
            return resp.text_annotations[0].description
        return ""
    except HTTPException:
        raise
    except Exception as e:
        print("[ERROR] GCV exception:", e)
        raise HTTPException(status_code=500, detail="OCR 호출에 실패했습니다.")


def call_gemini_health_insight(req: HealthInsightRequest) -> HealthInsightResponse:
    if STUB_MODE or not gemini_client:
        # 앱 UI만 먼저 붙여보고 싶을 때 쓸 수 있는 더미 응답
        dummy = (
            f"{req.pet_name}의 최근 체중과 진료기록을 기준으로 보면, "
            "전반적인 건강 상태는 안정적이지만, 피부/알러지 관리에 조금 더 신경 쓰면 좋겠다는 정도의 수준입니다."
        )
        return HealthInsightResponse(
            summary=dummy,
            bullets=[
                "체중 변화가 크지 않아 비만/급격한 체중 감소 위험은 낮은 편입니다.",
                "최근 진료 내역에 피부 관련 처치가 있으므로 가려움/피부염 재발 여부를 관찰해 주세요.",
                "예방접종/심장사상충 스케줄을 꾸준히 유지하면 전반적인 리스크를 더 줄일 수 있습니다.",
            ],
        )

    # Gemini에 보낼 프롬프트 구성 (텍스트 위주)
    parts = []

    parts.append(
        f"너는 한국 반려동물 보호자를 위한 수의사 보조 AI야. "
        f"설명은 모두 한국어로, 보호자가 이해하기 쉽게 써줘."
    )
    parts.append(f"반려동물 이름: {req.pet_name}")
    parts.append(f"종: {req.species}")
    if req.birth_date:
        parts.append(f"생년월일: {req.birth_date}")
    if req.allergies:
        parts.append("알러지: " + ", ".join(req.allergies))

    if req.weight_history:
        parts.append("\n[체중 기록]")
        for w in req.weight_history:
            parts.append(f"- {w.date}: {w.weight:.1f} kg")

    if req.receipt_texts:
        parts.append("\n[진료 영수증 OCR 텍스트]")
        for i, txt in enumerate(req.receipt_texts, start=1):
            parts.append(f"--- 영수증 {i} ---\n{txt}")

    if req.lab_summaries:
        parts.append("\n[검사결과 요약]")
        for s in req.lab_summaries:
            parts.append(f"- {s}")

    if req.certificate_summaries:
        parts.append("\n[증명서/진단서 요약]")
        for s in req.certificate_summaries:
            parts.append(f"- {s}")

    parts.append(
        "\n위 정보를 바탕으로 아래 형식의 JSON을 만들어줘. "
        '키는 "summary" 와 "bullets" 두 가지만 사용해. '
        '"summary"는 전체 요약(한 단락), "bullets"는 3~5개의 핵심 포인트 배열이야.'
    )

    prompt = "\n".join(parts)

    try:
        res = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = res.text or ""
    except Exception as e:
        print("[ERROR] Gemini call failed:", e)
        raise HTTPException(status_code=500, detail="Gemini 분석 호출에 실패했습니다.")

    # JSON 파싱 시도
    summary = text
    bullets: List[str] = []
    try:
        # 응답 안에서 첫 번째 { ... } 추출
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            summary = obj.get("summary", summary)
            bullets = obj.get("bullets") or []
            if not isinstance(bullets, list):
                bullets = [str(bullets)]
    except Exception:
        # JSON 형식이 아니면 그대로 summary 로 사용
        pass

    return HealthInsightResponse(summary=summary, bullets=bullets)


# ============================================================
# FastAPI 앱 생성
# ============================================================
app = FastAPI(title="PetHealth+ API", version="1.0.0")

# 모바일 앱은 CORS 크게 상관 없지만, 웹 클라이언트 고려해서 넉넉하게 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# 헬스 체크
# ---------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "stub_mode": STUB_MODE,
        "gcv_enabled": bool(vision_client) and GCV_ENABLED,
        "gemini_enabled": bool(gemini_client),
        "s3_enabled": bool(s3_client),
    }


# ---------------------------
# 1) 영수증 OCR + S3 업로드
# ---------------------------
@app.post("/api/ocr/receipt", response_model=ReceiptOCRResponse)
async def ocr_receipt(
    file: UploadFile = File(...),
    pet_id: Optional[str] = None,
):
    if file.content_type not in ("image/jpeg", "image/png", "image/heic", "image/heif"):
        raise HTTPException(status_code=400, detail="이미지(jpg, png, heic)만 업로드 가능합니다.")

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="이미지 용량이 너무 큽니다. (최대 15MB)")

    # HEIC/HEIF 는 S3 ContentType 만 image/heic 로 저장 (실제 변환은 iOS 쪽에서 JPG 로 보내는 게 더 안전)
    content_type = "image/jpeg" if file.content_type.startswith("image/") else file.content_type

    s3_url = upload_to_s3("receipts", data, content_type)

    ocr_text = call_gcv_ocr(data)
    clinic_name, visit_date, total_amount, items = parse_receipt_text(ocr_text)

    return ReceiptOCRResponse(
        pet_id=pet_id,
        s3_url=s3_url,
        clinic_name=clinic_name,
        visit_date=visit_date,
        total_amount=total_amount,
        items=items,
        raw_text=ocr_text,
    )


# ---------------------------
# 2) 검사결과 / 증명서 PDF 업로드
# ---------------------------
@app.post("/api/upload/pdf/{kind}", response_model=PDFUploadResponse)
async def upload_pdf(
    kind: str,
    file: UploadFile = File(...),
):
    kind = kind.lower()
    if kind not in ("lab", "certificate"):
        raise HTTPException(status_code=400, detail="kind는 lab 또는 certificate 만 가능합니다.")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF 용량이 너무 큽니다. (최대 30MB)")

    folder = "labs" if kind == "lab" else "certificates"
    s3_url = upload_to_s3(folder, data, "application/pdf")

    return PDFUploadResponse(kind=kind, s3_url=s3_url, filename=file.filename or "")


# ---------------------------
# 3) Gemini 건강 인사이트
# ---------------------------
@app.post("/api/ai/health-insight", response_model=HealthInsightResponse)
async def health_insight(req: HealthInsightRequest):
    """
    iOS 쪽에서:
    - 펫 기본정보
    - 체중 히스토리
    - 서버 OCR 텍스트 (영수증 raw_text)
    - 검사결과/증명서 텍스트 요약(있으면)
    을 보내면, Gemini 가 전체 건강 리포트를 만들어줌.
    """
    return call_gemini_health_insight(req)
