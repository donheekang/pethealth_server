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
from botocore.exceptions import NoCredentialsError, ClientError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ------------------------------------------------
# 1. 설정 / 외부 모듈
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
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

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


def delete_from_s3(key: str) -> None:
    """
    S3 객체 삭제.
    - 존재 확인(head_object) 후 delete_object 실행
    - 없으면 404 반환
    """
    try:
        s3_client.head_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        s3_client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=key)

    except ClientError as e:
        err = e.response.get("Error") or {}
        code = err.get("Code", "")

        if code in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

        raise HTTPException(status_code=500, detail=f"S3 삭제 실패: {e}")

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 인증 실패")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 삭제 실패: {e}")


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
# 4. 영수증 파서 (Kor 파서 + AI 파서)
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
                fixed_items.append({"name": str(name), "price": int(price)})
        data["items"] = fixed_items

        return data

    except Exception as e:
        print("parse_receipt_ai error:", e)
        return None


# ------------------------------------------------
# 5. DTO 정의 (현재는 참고용)
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
    tags: List[str] = Field(default_factory=list)


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


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_model": settings.GEMINI_MODEL_NAME,
        "gemini_enabled": settings.GEMINI_ENABLED,
        "stub_mode": settings.STUB_MODE,
        "keyword_tag_fallback": "disabled",  # ✅ 추정 태그(키워드 매칭) 완전 비활성
    }


# ------------------------------------------------
# 7. ENDPOINTS – 영수증 / PDF
# ------------------------------------------------

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

    # 3) AI 파싱 시도 → 실패 시 정규식 파서로 Fallback
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

    clinic_name = (parsed_for_dto.get("clinicName") or "").strip()
    clinic_name = re.sub(r"^원\s*명[:：]?\s*", "", clinic_name)
    parsed_for_dto["clinicName"] = clinic_name

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,
        "objectKey": key,
    }


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

    filename = key.split("/")[-1]
    base_name, _ = os.path.splitext(filename)

    return {
        "id": base_name,
        "petId": petId,
        "title": original_base,
        "memo": memo,
        "s3Url": url,
        "createdAt": created_at_iso,
        "objectKey": key,
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

    filename = key.split("/")[-1]
    base_name, _ = os.path.splitext(filename)

    return {
        "id": base_name,
        "petId": petId,
        "title": original_base,
        "memo": memo,
        "s3Url": url,
        "createdAt": created_at_iso,
        "objectKey": key,
    }


@app.delete("/lab/delete")
@app.delete("/labs/delete")
@app.delete("/api/lab/delete")
@app.delete("/api/labs/delete")
def delete_lab_pdf(
    petId: str = Query(...),
    id: str = Query(...),
):
    object_id = (id or "").strip()
    if not object_id:
        raise HTTPException(status_code=400, detail="id is required")

    filename = object_id if object_id.endswith(".pdf") else f"{object_id}.pdf"
    key = f"lab/{petId}/{filename}"

    delete_from_s3(key)
    return {"ok": True, "deletedKey": key}


@app.delete("/cert/delete")
@app.delete("/certs/delete")
@app.delete("/api/cert/delete")
@app.delete("/api/certs/delete")
def delete_cert_pdf(
    petId: str = Query(...),
    id: str = Query(...),
):
    object_id = (id or "").strip()
    if not object_id:
        raise HTTPException(status_code=400, detail="id is required")

    filename = object_id if object_id.endswith(".pdf") else f"{object_id}.pdf"
    key = f"cert/{petId}/{filename}"

    delete_from_s3(key)
    return {"ok": True, "deletedKey": key}


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
                    "objectKey": key,
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
                    "objectKey": key,
                }
            )

    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


# ------------------------------------------------
# 8. AI 케어 – 태그 통계 & 케어 가이드
# ------------------------------------------------

def _parse_visit_date(s: Optional[str]) -> Optional[date]:
    """'2025-12-03' 또는 '2025-12-03 10:30' 형식 날짜 문자열 파싱."""
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


def _canonicalize_tag_code(raw_code: str) -> Optional[str]:
    """
    입력된 태그 코드(raw)가 alias/canonical 어떤 형태든,
    CONDITION_TAGS에서 찾은 cfg.code(=canonical snake_case)로 정규화.
    못 찾으면 None.
    """
    if not raw_code:
        return None
    code = raw_code.strip()
    if not code:
        return None

    cfg = CONDITION_TAGS.get(code)
    if cfg:
        return getattr(cfg, "code", code)

    # 약간의 방어(문자 변형 케이스)
    code2 = code.replace("-", "_").strip()
    cfg = CONDITION_TAGS.get(code2)
    if cfg:
        return getattr(cfg, "code", code2)

    return None


def _build_tag_stats(
    medical_history: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    ✅ 안전 모드(프리미엄/신뢰 우선):
    - 1순위: iOS에서 넘어온 record["tags"]만 사용
    - ❌ 2순위(추정): diagnosis/clinic_name 키워드 매칭은 완전 비활성

    반환:
    - tags: [{tag, label, count, recentDates}]
    - periodStats: {"1m": {...}, "3m": {...}, "1y": {...}}
    """
    today = date.today()

    agg: Dict[str, Dict[str, Any]] = {}
    period_stats: Dict[str, Dict[str, int]] = {"1m": {}, "3m": {}, "1y": {}}

    for mh in medical_history:
        visit_str = mh.get("visitDate") or mh.get("visit_date") or ""
        visit_dt = _parse_visit_date(visit_str)
        visit_date_str = visit_dt.isoformat() if visit_dt else None

        record_tags: List[str] = mh.get("tags") or []
        used_codes: set[str] = set()

        # ✅ tags만 사용 (추정 매칭 OFF)
        for raw in record_tags:
            canonical = _canonicalize_tag_code(raw)
            if not canonical:
                continue

            cfg = CONDITION_TAGS.get(canonical)
            if not cfg:
                # canonical 키가 dict에 없을 수도 있어 방어
                # (하지만 네 condition_tags.py 구조면 canonical 키는 반드시 있음)
                continue

            if canonical in used_codes:
                continue
            used_codes.add(canonical)

            stat = agg.setdefault(
                canonical,
                {
                    "tag": canonical,
                    "label": getattr(cfg, "label", canonical),
                    "count": 0,
                    "recentDates": [],
                },
            )
            stat["count"] += 1
            if visit_date_str:
                stat["recentDates"].append(visit_date_str)

            if visit_dt:
                days = (today - visit_dt).days
                if days <= 365:
                    period_stats["1y"][canonical] = period_stats["1y"].get(canonical, 0) + 1
                if days <= 90:
                    period_stats["3m"][canonical] = period_stats["3m"].get(canonical, 0) + 1
                if days <= 30:
                    period_stats["1m"][canonical] = period_stats["1m"].get(canonical, 0) + 1

    for stat in agg.values():
        stat["recentDates"] = sorted(stat["recentDates"], reverse=True)

    tags = sorted(agg.values(), key=lambda x: x["count"], reverse=True)
    return tags, period_stats


def _summarize_recent_weights(body: Dict[str, Any]) -> List[str]:
    """
    recentWeights를 사람이 읽을 수 있는 한 줄 요약 리스트로 변환.
    예: ["2025-12-03: 8.1kg", "2025-12-19: 8.2kg"]
    """
    rows = body.get("recentWeights") or body.get("recent_weights") or []
    parsed: List[Tuple[datetime, float]] = []
    for r in rows:
        try:
            ds = (r.get("date") or "").strip()
            w = r.get("weight")
            if not ds or w is None:
                continue
            dt = datetime.strptime(ds, "%Y-%m-%d")
            parsed.append((dt, float(w)))
        except Exception:
            continue

    parsed.sort(key=lambda x: x[0])
    return [f"{dt.strftime('%Y-%m-%d')}: {w:.1f}kg" for dt, w in parsed[-6:]]


def _summarize_schedules(body: Dict[str, Any]) -> List[str]:
    """
    schedules에서 isUpcoming=True 위주로 가까운 일정 몇 개를 요약.
    예: ["2025-12-24: 종합백신", "2026-01-09: 심장사상충"]
    """
    rows = body.get("schedules") or []
    parsed: List[Tuple[datetime, str]] = []
    for r in rows:
        try:
            title = (r.get("title") or "").strip() or "일정"
            ds = (r.get("date") or "").strip()
            if not ds:
                continue
            is_upcoming = bool(r.get("isUpcoming", False))
            dt = datetime.strptime(ds, "%Y-%m-%d")
            if is_upcoming:
                parsed.append((dt, title))
        except Exception:
            continue

    parsed.sort(key=lambda x: x[0])
    return [f"{dt.strftime('%Y-%m-%d')}: {title}" for dt, title in parsed[:5]]


# ------------------------------------------------
# 9. Gemini 기반 AI 요약 생성
# ------------------------------------------------

def _build_gemini_prompt(
    pet_name: str,
    tags: List[Dict[str, Any]],
    period_stats: Dict[str, Dict[str, int]],
    body: Dict[str, Any],
) -> str:
    profile = body.get("profile") or {}
    species = profile.get("species", "dog")
    age_text = profile.get("ageText") or profile.get("age_text") or ""
    weight = profile.get("weightCurrent") or profile.get("weight_current")

    mh_list = body.get("medicalHistory") or body.get("medical_history") or []
    mh_summary_lines = []
    for mh in mh_list[:5]:
        clinic = mh.get("clinicName") or mh.get("clinic_name") or ""
        diag = mh.get("diagnosis") or ""
        visit = mh.get("visitDate") or mh.get("visit_date") or ""
        # diagnosis가 비어도 괜찮게
        mh_summary_lines.append(f"- {visit} / {clinic} / {diag}")

    tag_lines = []
    for t in tags:
        code = t.get("tag")
        cfg = CONDITION_TAGS.get(code) if code else None
        group = getattr(cfg, "group", "") if cfg else ""
        recent_dates = ", ".join(t.get("recentDates", [])[:3])
        tag_lines.append(
            f"- {t.get('label','')} ({group}) : {t.get('count',0)}회 (최근 기록일: {recent_dates or '정보 없음'})"
        )

    weight_lines = _summarize_recent_weights(body)
    schedule_lines = _summarize_schedules(body)

    prompt = f"""
당신은 반려동물 건강관리 전문가입니다.
아래 제공된 정보만을 근거로 보호자에게 한국어로 3~6문장 정도의 간단한 설명을 해주세요.

[반려동물 기본 정보]
이름: {pet_name}
종: {species}
나이 정보: {age_text or '정보 없음'}
현재 체중: {weight if weight is not None else '정보 없음'} kg

[최근 체중 기록(최대 6개)]
{os.linesep.join(weight_lines) if weight_lines else '체중 기록 없음'}

[다가오는 일정(최대 5개)]
{os.linesep.join(schedule_lines) if schedule_lines else '다가오는 일정 없음'}

[최근 진료 태그 통계]
{os.linesep.join(tag_lines) if tag_lines else '태그 통계 없음'}

[최근 진료 이력 요약(최대 5개)]
{os.linesep.join(mh_summary_lines) if mh_summary_lines else '진료 내역 없음'}

반드시 지켜야 할 규칙:
1) 제공된 태그/진료이력/체중/일정 정보에 없는 질환명, 검사명, 진단을 새로 추측하거나 만들어내지 마세요.
2) 태그 그룹 해석:
   - group이 exam/medication/procedure인 태그는 '검사/처치/처방 기록'입니다. 이것만으로 질환을 추론하지 마세요.
   - group이 preventive인 태그는 '예방 관리'입니다. 일정/주기 관리 관점으로만 안내하세요.
   - dermatology/orthopedics/cardiology/wellness 등은 '관리 포인트'로 부드럽게 정리하세요.
3) 너무 무섭게 말하지 말고, 안심시키면서 현실적인 관리 조언을 1~2개만 제안하세요.
4) 출력은 마크다운 없이 문장만 출력하세요. 불릿/번호/따옴표/코드블록을 쓰지 마세요.
"""
    return prompt.strip()


def _generate_gemini_summary(
    pet_name: str,
    tags: List[Dict[str, Any]],
    period_stats: Dict[str, Dict[str, int]],
    body: Dict[str, Any],
) -> Optional[str]:
    if settings.GEMINI_ENABLED.lower() != "true":
        return None
    if not settings.GEMINI_API_KEY:
        return None
    if genai is None:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        prompt = _build_gemini_prompt(pet_name, tags, period_stats, body)
        resp = model.generate_content(prompt)

        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts)

        summary = (text or "").strip()
        if not summary:
            return None

        summary = summary.strip("`").strip()
        return summary

    except Exception as e:
        print("[AI] Gemini summary error:", e)
        return None


# ------------------------------------------------
# 10. AI 케어 분석 엔드포인트
# ------------------------------------------------

@app.post("/api/ai/analyze")
async def analyze_pet_health(body: Dict[str, Any]):
    """
    PetHealth+ AI 케어: iOS에서 보내는 raw JSON을 그대로 받아
    ✅ record.tags 기반으로만 통계/요약 리포트를 생성 (추정 태그 OFF)
    """
    try:
        print("[AI] raw body =", json.dumps(body, ensure_ascii=False))
    except Exception:
        print("[AI] raw body (repr) =", repr(body))

    profile = body.get("profile") or {}
    pet_name = profile.get("name") or "반려동물"

    medical_history = body.get("medicalHistory") or body.get("medical_history") or []
    has_history = len(medical_history) > 0

    tags, period_stats = _build_tag_stats(medical_history)

    # tags 기반 케어 가이드: condition_tags.py의 guide 사용
    care_guide: Dict[str, List[str]] = {}
    for t in tags:
        code = t.get("tag")
        cfg = CONDITION_TAGS.get(code) if code else None
        guide = getattr(cfg, "guide", None) if cfg else None
        if code and guide:
            care_guide[code] = guide

    # 기본 summary (Gemini 없거나 데이터 부족 시)
    if not has_history:
        summary = (
            f"{pet_name}의 진료 기록이 아직 많지 않아요. "
            "진료기록을 남길 때 태그를 함께 선택해 주시면, "
            "다음부터는 컨디션별로 더 정확하게 정리해 드릴게요."
        )
    elif not tags:
        summary = (
            f"{pet_name}의 진료 기록은 있지만, 태그가 비어 있어 통계를 만들기 어려워요. "
            "기록 저장 시 컨디션/예방/검사 태그를 선택해 주시면 "
            "홈과 분석 화면에서 더 정확히 정리해 드릴게요."
        )
    else:
        top = tags[0]
        summary = (
            f"최근 기록에서 '{top.get('label','')}' 관련 태그가 {top.get('count',0)}회 확인됐어요. "
            "기록을 바탕으로 관리 포인트를 정리해 드렸어요."
        )

    # Gemini 요약(있으면 덮어씀) - 단, 규칙에 따라 '추측/환각' 최소화
    ai_summary = _generate_gemini_summary(pet_name, tags, period_stats, body)
    if ai_summary:
        summary = ai_summary

    response = {
        "summary": summary,
        "tags": tags,
        "periodStats": period_stats,
        "careGuide": care_guide,
        "tagFallback": "disabled",  # 디버그/신뢰 표시
    }

    print(f"[AI] response tags={len(tags)}")
    return response
