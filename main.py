from __future__ import annotations

import os
import io
import json
import uuid
import tempfile
import re
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from condition_tags import CONDITION_TAGS  # ConditionTagConfig / CONDITION_TAGS 사용


# =========================
#  설정
# =========================

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    GEMINI_ENABLED: str = "false"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    STUB_MODE: str = "false"

    class Config:
        env_file = ".env"


settings = Settings()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


# =========================
#  공통 유틸
# =========================

def upload_to_s3(file_obj, key: str, content_type: str) -> str:
    try:
        s3_client.upload_fileobj(
            file_obj,
            settings.S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
        return url
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 인증 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")


def get_vision_client() -> vision.ImageAnnotatorClient:
    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS 환경변수가 비어있습니다.")

    try:
        info = json.loads(cred_value)
        return vision.ImageAnnotatorClient.from_service_account_info(info)
    except json.JSONDecodeError:
        if not os.path.exists(cred_value):
            raise Exception(
                "GOOGLE_APPLICATION_CREDENTIALS가 JSON도 아니고, "
                f"파일 경로({cred_value})도 아닙니다."
            )
        return vision.ImageAnnotatorClient.from_service_account_file(cred_value)
    except Exception as e:
        raise Exception(f"OCR 클라이언트 생성 실패: {e}")


def run_vision_ocr(image_path: str) -> str:
    client = get_vision_client()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"OCR 에러: {response.error.message}")

    texts = response.text_annotations
    if not texts:
        return ""

    return texts[0].description


# =========================
#  영수증 파서 (기존 버전)
# =========================

def guess_hospital_name(lines: List[str]) -> str:
    keywords = [
        "동물병원", "동물 병원", "동물의료", "동물메디컬", "동물 메디컬",
        "동물클리닉", "동물 클리닉",
        "애견병원", "애완동물병원", "펫병원", "펫 병원",
        "종합동물병원", "동물의원", "동물병의원",
    ]

    best_line = None
    best_score = -1

    for idx, line in enumerate(lines):
        score = 0
        text = line.replace(" ", "")

        if any(k in text for k in keywords):
            score += 5

        if idx <= 4:
            score += 2

        if any(x in line for x in ["TEL", "전화", "FAX", "팩스", "도로명"]):
            score -= 2

        digit_count = sum(c.isdigit() for c in line)
        if digit_count >= 8:
            score -= 1

        if len(line) < 2 or len(line) > 25:
            score -= 1

        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None and lines:
        return lines[0]
    return best_line or ""


def parse_receipt_kor(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 병원명
    hospital_name = guess_hospital_name(lines)

    # 날짜
    visit_at = None
    dt_pattern = re.compile(
        r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2}).*?(\d{1,2}):(\d{2})"
    )
    for line in lines:
        m = dt_pattern.search(line)
        if m:
            y, mo, d, h, mi = map(int, m.groups())
            visit_at = datetime(y, mo, d, h, mi).strftime("%Y-%m-%d %H:%M")
            break

    if not visit_at:
        dt_pattern_short = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})")
        for line in lines:
            m = dt_pattern_short.search(line)
            if m:
                y, mo, d = map(int, m.groups())
                visit_at = datetime(y, mo, d).strftime("%Y-%m-%d")
                break

    # 금액
    amt_pattern_total = re.compile(r"(?:₩|￦)?\s*(\d{1,3}(?:,\d{3})+|\d+)\s*(원)?\s*$")
    candidate_totals: List[int] = []
    for line in lines:
        m = amt_pattern_total.search(line)
        if not m:
            continue
        amount_str = m.group(1).replace(",", "")
        try:
            amount = int(amount_str)
        except ValueError:
            continue

        lowered = line.replace(" ", "")
        if any(k in lowered for k in ["합계", "총액", "총금액", "합계금액", "결제요청"]):
            candidate_totals.append(amount)

    # 항목
    start_idx = None
    for i, line in enumerate(lines):
        if "[날짜" in line:
            start_idx = i + 1
            break
        if ("진료" in line and "내역" in line) or ("진료 및" in line and "내역" in line):
            start_idx = i + 1

    if start_idx is None:
        start_idx = 0

    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if any(k in lines[i] for k in ["소 계", "소계", "합계", "결제요청"]):
            end_idx = i
            break

    item_block = lines[start_idx:end_idx]

    names: List[str] = []
    prices: List[int] = []

    for line in item_block:
        if any(k in line for k in ["동물명", "항목", "단가", "수량", "금액"]):
            continue

        if line.startswith("*"):
            name = line.lstrip("*").strip().strip(".")
            if name:
                names.append(name)
            continue

        if re.fullmatch(r"[0-9,\s]+", line):
            m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", line)
            if m:
                amt = int(m.group(1).replace(",", ""))
                if amt > 0:
                    prices.append(amt)
            continue

        m = re.search(r"(.+?)\s+(\d{1,3}(?:,\d{3})+|\d+)", line)
        if m and ":" not in line and "[" not in line:
            name = m.group(1).strip()
            amt = int(m.group(2).replace(",", ""))
            if name:
                names.append(name)
                prices.append(amt)

    items: List[Dict[str, Any]] = []
    pair_count = min(len(names), len(prices))
    for i in range(pair_count):
        items.append({"name": names[i], "amount": prices[i]})

    if candidate_totals:
        total_amount = max(candidate_totals)
    elif prices:
        total_amount = sum(prices)
    else:
        total_amount = 0

    return {
        "hospitalName": hospital_name,
        "visitAt": visit_at,
        "items": items,
        "totalAmount": total_amount,
    }


# =========================
#  Pydantic DTOs (AI 케어)
# =========================

class CamelBase(BaseModel):
    class Config:
        allow_population_by_field_name = True
        orm_mode = True
        extra = "ignore"


class PetProfileDTO(CamelBase):
    name: str
    species: str = "dog"
    age_text: Optional[str] = Field(None, alias="ageText")
    weight_current: Optional[float] = Field(None, alias="weightCurrent")
    allergies: List[str] = Field(default_factory=list)


class WeightLogDTO(CamelBase):
    date: str
    weight: Optional[float] = None


class MedicalHistoryDTO(CamelBase):
    visit_date: Optional[str] = Field(None, alias="visitDate")
    clinic_name: Optional[str] = Field(None, alias="clinicName")
    item_count: Optional[int] = Field(0, alias="itemCount")
    diagnosis: Optional[str] = None


class ScheduleDTO(CamelBase):
    title: str
    date: Optional[str] = None
    is_upcoming: Optional[bool] = Field(None, alias="isUpcoming")


class AICareRequest(CamelBase):
    request_date: Optional[str] = Field(None, alias="requestDate")
    profile: PetProfileDTO
    recent_weights: List[WeightLogDTO] = Field(default_factory=list, alias="recentWeights")
    medical_history: List[MedicalHistoryDTO] = Field(default_factory=list, alias="medicalHistory")
    schedules: List[ScheduleDTO] = Field(default_factory=list)


# =========================
#  FastAPI 앱
# =========================

app = FastAPI(title="PetHealth+ Server", version="1.2.0")

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


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "stub_mode": settings.STUB_MODE,
    }


# =========================
#  영수증 업로드 엔드포인트
# =========================

@app.post("/api/receipts/upload")
@app.post("/api/receipt/upload")
async def upload_receipt(
    petId: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    upload: Optional[UploadFile] = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="no file or image field")

    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(upload.filename or "")
    if not ext:
        ext = ".jpg"

    key = f"receipts/{petId}/{rec_id}{ext}"

    data = await upload.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    file_url = upload_to_s3(
        file_like,
        key,
        content_type=upload.content_type or "image/jpeg",
    )

    ocr_text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as tmp:
            tmp.write(data)
            tmp.flush()
            ocr_text = run_vision_ocr(tmp.name)
    except Exception as e:
        print("OCR error:", e)
        ocr_text = ""

    fallback = (
        parse_receipt_kor(ocr_text)
        if ocr_text
        else {"hospitalName": "", "visitAt": None, "items": [], "totalAmount": 0}
    )

    dto_items = [
        {"name": it.get("name", "항목"), "price": it.get("amount") or 0}
        for it in fallback.get("items", [])
    ]

    parsed_for_dto = {
        "clinicName": fallback.get("hospitalName"),
        "visitDate": fallback.get("visitAt"),
        "diseaseName": None,
        "symptomsSummary": None,
        "items": dto_items,
        "totalAmount": fallback.get("totalAmount"),
    }

    clinic_name = (parsed_for_dto.get("clinicName") or "").strip()
    clinic_name = re.sub(r"^원\s*명[:：]?\s*", "", clinic_name)
    parsed_for_dto["clinicName"] = clinic_name

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,
    }


# =========================
#  AI 케어: 태그 통계
# =========================

def _parse_visit_date(s: str | None) -> date | None:
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        try:
            part = s.split()[0]
            return datetime.strptime(part, "%Y-%m-%d").date()
        except Exception:
            return None


def _build_tag_stats(
    medical_history: List[MedicalHistoryDTO],
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    today = date.today()

    agg: Dict[str, Dict[str, Any]] = {}
    period_stats: Dict[str, Dict[str, int]] = {
        "1m": {},
        "3m": {},
        "1y": {},
    }

    for mh in medical_history:
        visit_str = mh.visit_date or ""
        diag = mh.diagnosis or ""
        clinic = mh.clinic_name or ""

        base_text = f"{diag} {clinic}".strip()
        if not base_text:
            continue

        text_lower = base_text.lower()
        visit_dt = _parse_visit_date(visit_str)
        visit_date_str = visit_dt.isoformat() if visit_dt else None

        for cfg in CONDITION_TAGS.values():
            keyword_hit = False

            # 코드 자체(ortho_patella 등)도 키워드로 인정
            if cfg.code.lower() in text_lower:
                keyword_hit = True
            else:
                for kw in cfg.keywords:
                    if kw.lower() in text_lower:
                        keyword_hit = True
                        break

            if not keyword_hit:
                continue

            stat = agg.setdefault(
                cfg.code,
                {"tag": cfg.code, "label": cfg.label, "count": 0, "recentDates": []},
            )
            stat["count"] += 1
            if visit_date_str:
                stat["recentDates"].append(visit_date_str)

            if visit_dt:
                days = (today - visit_dt).days
                if days <= 365:
                    period_stats["1y"][cfg.code] = period_stats["1y"].get(cfg.code, 0) + 1
                if days <= 90:
                    period_stats["3m"][cfg.code] = period_stats["3m"].get(cfg.code, 0) + 1
                if days <= 30:
                    period_stats["1m"][cfg.code] = period_stats["1m"].get(cfg.code, 0) + 1

    for stat in agg.values():
        stat["recentDates"] = sorted(stat["recentDates"], reverse=True)

    tags = sorted(agg.values(), key=lambda x: x["count"], reverse=True)
    return tags, period_stats


DEFAULT_CARE_GUIDE: Dict[str, List[str]] = {
    "ortho_patella": [
        "미끄럽지 않은 매트를 깔아주세요.",
        "계단이나 높은 점프는 최대한 피하는 것이 좋아요.",
        "관절 영양제를 꾸준히 급여하는 것을 보호자와 상의해 보세요.",
    ],
    "skin_atopy": [
        "정기적인 목욕과 빗질로 피부를 깨끗하게 유지해 주세요.",
        "간식이나 사료를 바꾼 후 증상이 심해졌는지 함께 체크해 주세요.",
    ],
    "prevent_vaccine_comprehensive": [
        "정기적인 종합백신 접종 스케줄을 캘린더에 기록해 두면 좋아요.",
    ],
}


@app.post("/api/ai/analyze")
async def analyze_pet_health(req: AICareRequest):
    print(f"[AI] analyze_pet_health called for {req.profile.name}")
    print(f"[AI] medical history count = {len(req.medical_history or [])}")

    tags, period_stats = _build_tag_stats(req.medical_history or [])

    has_history = len(req.medical_history or []) > 0

    if not has_history:
        summary = (
            f"{req.profile.name}의 진료 기록이 없어서 현재 상태에 대한 "
            "구체적인 조언을 드리기 어렵습니다. 진단명이 포함된 영수증을 "
            "조금 더 기록해 주시면 통계를 만들어 드릴 수 있어요."
        )
    elif not tags:
        summary = (
            f"{req.profile.name}의 진료 기록은 있지만, 아직 슬개골·피부·관절 같은 "
            "특정 컨디션 태그로 분류할 수 있는 단서가 부족해요. "
            "영수증에 진단명이나 증상이 보이도록 기록하면 태그 통계를 만들어 드릴게요."
        )
    else:
        top = tags[0]
        summary = (
            f"최근 진료에서 '{top['label']}' 관련 기록이 {top['count']}회 확인됐어요. "
            "기간별 통계를 바탕으로 관리 포인트를 정리해 드렸어요."
        )

    care_guide: Dict[str, List[str]] = {}
    for t in tags:
        code = t["tag"]
        if code in DEFAULT_CARE_GUIDE:
            care_guide[code] = DEFAULT_CARE_GUIDE[code]

    response = {
        "summary": summary,
        "tags": tags,
        "periodStats": period_stats,
        "careGuide": care_guide,
    }

    print(f"[AI] response tags={len(tags)}")
    return response
