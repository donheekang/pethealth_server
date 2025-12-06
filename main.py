# main.py

import os
import io
import json
import uuid
import tempfile
import re
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from condition_tags import CONDITION_TAGS  # 질환/컨디션 태그 정의


# ------------------------------------------------
# 1. 설정
# ------------------------------------------------

try:
    from condition_tags import CONDITION_TAGS
except ImportError:
    CONDITION_TAGS = {}
    print("Warning: condition_tags.py not found. AI tagging will be limited.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google.generativeai not installed. Gemini features disabled.")


class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # Google Vision
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Gemini
    GEMINI_ENABLED: str = "false"        # "true" / "false"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"   # 콘솔에서 쓰는 모델명

    # 디버그용 스텁 모드
    STUB_MODE: str = "false"

    class Config:
        env_file = ".env"


settings = Settings()


# ------------------------------------------------
# 2. S3 클라이언트
# ------------------------------------------------

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


def upload_to_s3(file_obj, key: str, content_type: str) -> str:
    """
    파일을 S3에 올리고, 7일짜리 presigned URL을 돌려준다.
    """
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


# ------------------------------------------------
# 3. Google Vision OCR
# ------------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS 환경변수가 비어있습니다.")

    try:
        # JSON 문자열로 넘어온 경우
        info = json.loads(cred_value)
        return vision.ImageAnnotatorClient.from_service_account_info(info)
    except json.JSONDecodeError:
        # 파일 경로로 넘어온 경우
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


# ------------------------------------------------
# 4. 영수증 파서 (기존 Kor 파서 + AI 파서)
# ------------------------------------------------

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
    """
    OCR 텍스트를 한국 동물병원 영수증 형태라고 가정하고
    병원명 / 날짜 / 항목 / 합계를 최대한 맞춰보는 파서.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1) 병원명 추정
    hospital_name = guess_hospital_name(lines)

    # 2) 날짜 추정
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

    # 3) 금액 추정
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

    # 4) 항목 블록 추출
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


def parse_receipt_ai(raw_text: str) -> Optional[dict]:
    """
    Gemini를 써서 영수증을 파싱하는 버전.
    실패하면 None을 돌려준다.
    """
    if settings.GEMINI_ENABLED.lower() != "true":
        return None
    if not settings.GEMINI_API_KEY:
        return None
    if genai is None:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        prompt = f"""
        너는 한국 동물병원 영수증을 구조화된 JSON으로 정리하는 어시스턴트야.
        다음은 OCR로 읽은 영수증 텍스트야:

        \"\"\"{raw_text}\"\"\"

        이 텍스트를 분석해서 아래 형식의 JSON만 돌려줘.

        {{
          "clinicName": string or null,
          "visitDate": string or null,
          "diseaseName": string or null,
          "symptomsSummary": string or null,
          "items": [
            {{
              "name": string,
              "price": integer or null
            }}
          ],
          "totalAmount": integer or null
        }}
        """

        resp = model.generate_content(prompt)

        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts)

        text = (text or "").strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        data = json.loads(text)

        for key in ["clinicName", "visitDate", "items", "totalAmount"]:
            if key not in data:
                return None

        if not isinstance(data.get("items"), list):
            data["items"] = []

        fixed_items = []
        for it in data["items"]:
            if isinstance(it, dict):
                name = it.get("name", "항목")
                price = it.get("price") or 0
                fixed_items.append(
                    {"name": str(name), "price": int(price)}
                )
        data["items"] = fixed_items

        return data

    except Exception as e:
        print("parse_receipt_ai error:", e)
        return None


# ------------------------------------------------
# 5. (참고용) AI 케어 Request DTO – 현재 엔드포인트에서는 직접 사용 안 함
# ------------------------------------------------

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


# ------------------------------------------------
# 6. FASTAPI APP SETUP
# ------------------------------------------------

app = FastAPI(title="PetHealth+ Server", version="1.1.0")

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


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_model": settings.GEMINI_MODEL_NAME,
        "gemini_enabled": settings.GEMINI_ENABLED,
        "stub_mode": settings.STUB_MODE,
    }


# ------------------------------------------------
# 7. ENDPOINTS – 영수증 / PDF
# ------------------------------------------------

# (1) 영수증 업로드 & 분석
@app.post("/receipts/upload")
@app.post("/api/receipt/upload")
@app.post("/api/receipts/upload")
@app.post("/api/receipt/analyze")
@app.post("/api/receipts/analyze")
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

    # 파일 데이터 읽기
    data = await upload.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    # 1) S3 업로드
    file_url = upload_to_s3(
        file_like,
        key,
        content_type=upload.content_type or "image/jpeg",
    )

    # 2) OCR 실행
    ocr_text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as tmp:
            tmp.write(data)
            tmp.flush()
            ocr_text = run_vision_ocr(tmp.name)
    except Exception as e:
        print("OCR error:", e)
        ocr_text = ""

    # 3) AI 파싱 시도 → 결과가 비정상이면 정규식 파서로 Fallback
    ai_parsed = parse_receipt_ai(ocr_text) if ocr_text else None

    use_ai = False
    if ai_parsed:
        ai_items = ai_parsed.get("items") or []
        ai_total = ai_parsed.get("totalAmount") or 0
        if len(ai_items) > 0 and ai_total > 0:
            use_ai = True

    if use_ai:
        parsed_for_dto = ai_parsed
    else:
        fallback = (
            parse_receipt_kor(ocr_text)
            if ocr_text
            else {"hospitalName": "", "visitAt": None, "items": [], "totalAmount": 0}
        )

        dto_items = []
        for it in fallback.get("items", []):
            dto_items.append(
                {
                    "name": it.get("name", "항목"),
                    "price": it.get("amount") or 0,
                }
            )

        parsed_for_dto = {
            "clinicName": fallback.get("hospitalName"),
            "visitDate": fallback.get("visitAt"),
            "diseaseName": None,
            "symptomsSummary": None,
            "items": dto_items,
            "totalAmount": fallback.get("totalAmount"),
        }

    # 병원명 앞의 '원 명:' 같은 접두어 제거
    clinic_name = (parsed_for_dto.get("clinicName") or "").strip()
    clinic_name = re.sub(r"^원\s*명[:：]?\s*", "", clinic_name)
    parsed_for_dto["clinicName"] = clinic_name

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,
    }


# (2) PDF 업로드 (검사/증명서)
@app.post("/lab/upload-pdf")
@app.post("/labs/upload-pdf")
@app.post("/api/lab/upload-pdf")
@app.post("/api/labs/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form("검사결과"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    original_base = os.path.splitext(file.filename or "")[0].strip() or "검사결과"
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")
    key = f"lab/{petId}/{safe_base}{uuid.uuid4()}.pdf"

    url = upload_to_s3(file.file, key, "application/pdf")
    created_at_iso = datetime.utcnow().isoformat()

    return {
        "id": key.split("/")[-1],
        "petId": petId,
        "title": original_base,
        "memo": memo,
        "s3Url": url,
        "createdAt": created_at_iso,
    }


@app.post("/cert/upload-pdf")
@app.post("/certs/upload-pdf")
@app.post("/api/cert/upload-pdf")
@app.post("/api/certs/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form("증명서"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    original_base = os.path.splitext(file.filename or "")[0].strip() or "증명서"
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")
    key = f"cert/{petId}/{safe_base}{uuid.uuid4()}.pdf"

    url = upload_to_s3(file.file, key, "application/pdf")
    created_at_iso = datetime.utcnow().isoformat()

    return {
        "id": key.split("/")[-1],
        "petId": petId,
        "title": original_base,
        "memo": memo,
        "s3Url": url,
        "createdAt": created_at_iso,
    }


# (3) 검사/증명서 리스트 조회
@app.get("/lab/list")
@app.get("/labs/list")
@app.get("/api/lab/list")
@app.get("/api/labs/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"
    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME, Prefix=prefix
    )

    items = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            base_name, _ = os.path.splitext(filename)

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=604800,
            )

            items.append(
                {
                    "id": base_name,
                    "petId": petId,
                    "title": f"검사결과 ({date_str})",
                    "s3Url": url,
                    "createdAt": created_at_iso,
                }
            )

    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


@app.get("/cert/list")
@app.get("/certs/list")
@app.get("/api/cert/list")
@app.get("/api/certs/list")
def get_cert_list(petId: str = Query(...)):
    prefix = f"cert/{petId}/"
    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME, Prefix=prefix
    )

    items = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            base_name, _ = os.path.splitext(filename)

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=604800,
            )

            items.append(
                {
                    "id": base_name,
                    "petId": petId,
                    "title": f"증명서 ({date_str})",
                    "s3Url": url,
                    "createdAt": created_at_iso,
                }
            )

    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


# ------------------------------------------------
# 8. AI 케어 – 태그 통계 & 케어 가이드
# ------------------------------------------------

from datetime import datetime, date
from typing import Any, Dict, List, Tuple

def _parse_visit_date(s: str | None) -> date | None:
    """'2025-12-03' 또는 '2025-12-03 10:30' 같이 들어오는 날짜 문자열 파싱."""
    if not s:
        return None
    s = s.strip()
    # 1) ISO 포맷 우선
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        pass
    # 2) 앞부분만 YYYY-MM-DD 로 시도
    try:
        part = s.split()[0]
        return datetime.strptime(part, "%Y-%m-%d").date()
    except Exception:
        return None


def _get_field(obj: Any, *candidates: str) -> str:
    """
    medical_history 아이템이 dict 로 오든 Pydantic 모델로 오든
    안전하게 값을 꺼내는 헬퍼.
    """
    # dict
    if isinstance(obj, dict):
        for key in candidates:
            if key in obj and obj[key]:
                return str(obj[key])
        return ""
    # pydantic 모델 / 일반 객체
    for key in candidates:
        value = getattr(obj, key, None)
        if value:
            return str(value)
    return ""

def _build_tag_stats(medical_history: list) -> tuple[list[dict], dict[str, dict[str, int]]]:
    # 디버그: 현재 로딩된 태그 코드 확인
    print("[AI] CONDITION_TAGS keys =", list(CONDITION_TAGS.keys()))
    
def _build_tag_stats(
    medical_history: List[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    진료 이력 리스트에서 CONDITION_TAGS를 기준으로
    - tags: [{tag, label, count, recentDates}]
    - periodStats: {"1m": {...}, "3m": {...}, "1y": {...}}
    를 만들어서 반환.
    dict / Pydantic 둘 다 처리 가능하게 구현.
    """
    today = date.today()

    # tag_code -> 집계
    agg: Dict[str, Dict[str, Any]] = {}

    # 기간별 통계
    period_stats: Dict[str, Dict[str, int]] = {
        "1m": {},
        "3m": {},
        "1y": {},
    }

    for mh in medical_history or []:
        # 필드 꺼내기 (snake / camel 모두 지원)
        visit_str = _get_field(mh, "visit_date", "visitDate")
        diag = _get_field(mh, "diagnosis")
        clinic = _get_field(mh, "clinic_name", "clinicName")

        base_text = f"{diag} {clinic}".strip()
        if not base_text:
            continue

        visit_dt = _parse_visit_date(visit_str)
        visit_date_str = visit_dt.isoformat() if visit_dt else None
        text_lower = base_text.lower()

        # CONDITION_TAGS 키워드 매칭
        for cfg in CONDITION_TAGS.values():
            keyword_hit = False

            # 1) 코드 자체
            if cfg.code.lower() in text_lower:
                keyword_hit = True
            else:
                # 2) 키워드 목록
                for kw in cfg.keywords:
                    if kw.lower() in text_lower:
                        keyword_hit = True
                        break

            if not keyword_hit:
                continue

            stat = agg.setdefault(
                cfg.code,
                {
                    "tag": cfg.code,
                    "label": cfg.label,
                    "count": 0,
                    "recentDates": [],
                },
            )
            stat["count"] += 1
            if visit_date_str:
                stat["recentDates"].append(visit_date_str)

            # 기간별 통계 집계
            if visit_dt:
                days = (today - visit_dt).days
                if days <= 365:
                    period_stats["1y"][cfg.code] = period_stats["1y"].get(cfg.code, 0) + 1
                if days <= 90:
                    period_stats["3m"][cfg.code] = period_stats["3m"].get(cfg.code, 0) + 1
                if days <= 30:
                    period_stats["1m"][cfg.code] = period_stats["1m"].get(cfg.code, 0) + 1

    # 날짜 정렬 + count 기준 정렬
    for stat in agg.values():
        stat["recentDates"] = sorted(stat["recentDates"], reverse=True)

    tags = sorted(agg.values(), key=lambda x: x["count"], reverse=True)

    # 디버그용 로그
    print(f"[AI] _build_tag_stats result: tags={tags}, period_stats={period_stats}")

    return tags, period_stats

# 태그 코드별 기본 케어 가이드
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
    "prevent_vaccine_corona": [
        "접종 후 1~2일 동안은 기력, 식욕 변화를 잘 관찰해 주세요.",
    ],
    # 필요하면 계속 추가 가능
}


@app.post("/api/ai/analyze")
async def analyze_pet_health(body: Dict[str, Any]):
    """
    PetHealth+ AI 케어: iOS에서 보내는 raw JSON을 그대로 받아
    진료 태그 기반 요약/통계 리포트를 생성.
    """

    # 디버그용 로그
    try:
        print("[AI] raw body =", json.dumps(body, ensure_ascii=False))
    except Exception:
        print("[AI] raw body (repr) =", repr(body))

    profile = body.get("profile") or {}
    pet_name = profile.get("name") or "반려동물"

    medical_history = body.get("medicalHistory") or []
    has_history = len(medical_history) > 0

    # 1) 태그 집계
    tags, period_stats = _build_tag_stats(medical_history)

    # 2) 요약 문구
    if not has_history:
        summary = (
            f"{pet_name}의 진료 기록이 없어서 현재 상태에 대한 "
            "구체적인 조언을 드리기 어렵습니다. 진단명이 포함된 영수증을 "
            "조금 더 기록해 주시면 통계를 만들어 드릴 수 있어요."
        )
    elif not tags:
        summary = (
            f"{pet_name}의 진료 기록은 있지만, 아직 슬개골·피부·관절 같은 "
            "특정 컨디션 태그로 분류할 수 있는 단서가 부족해요. "
            "영수증에 진단명이나 증상이 보이도록 기록하면 태그 통계를 만들어 드릴게요."
        )
    else:
        top = tags[0]
        summary = (
            f"최근 진료에서 '{top['label']}' 관련 기록이 {top['count']}회 확인됐어요. "
            "기간별 통계를 바탕으로 관리 포인트를 정리해 드렸어요."
        )

    # 3) 케어 가이드
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
