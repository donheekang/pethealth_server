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

    hospital_name = guess_hospital_name(lines)

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
""".strip()

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
# 8. AI 케어 – 태그 통계 & 케어 가이드 (+ 체중/일정 신호 추가)
# ------------------------------------------------

def _parse_visit_date(s: Optional[str]) -> Optional[date]:
    """'2025-12-03' 또는 '2025-12-03 10:30' 형식 날짜 문자열 파싱."""
    if not s:
        return None
    s = str(s).strip()
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        try:
            part = s.split()[0]
            return datetime.strptime(part, "%Y-%m-%d").date()
        except Exception:
            return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _is_exam_like_tag(code: str, cfg: Any = None) -> bool:
    """
    ✅ 검사/영상/패널 같은 'exam' 태그는
    - 컨디션(질환)처럼 요약/조언 대상으로 쓰지 않기 위해 분리한다.
    """
    c = (code or "").strip().lower()

    if c.startswith(("exam_", "test_", "lab_", "imaging_")):
        return True

    grp = getattr(cfg, "group", None)
    if isinstance(grp, str) and grp.lower() in ("exam", "test", "lab", "imaging"):
        return True

    # 보조 휴리스틱: label에 검사성 단어가 있으면 exam 취급
    label = getattr(cfg, "label", None)
    if isinstance(label, str):
        if any(k in label for k in ["검사", "엑스레이", "X-ray", "초음파", "혈액", "패널", "영상"]):
            return True

    return False


def _split_tag_stats(
    tags: List[Dict[str, Any]],
    period_stats: Dict[str, Dict[str, int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """
    tags/periodStats 를
    - condition(컨디션/질환/예방/약 등)
    - exam(검사/영상)
    으로 분리한다.
    """
    cond: List[Dict[str, Any]] = []
    exam: List[Dict[str, Any]] = []

    for t in tags:
        code = (t.get("tag") or "").strip()
        cfg = CONDITION_TAGS.get(code)
        if _is_exam_like_tag(code, cfg):
            exam.append(t)
        else:
            cond.append(t)

    cond_codes = {t.get("tag") for t in cond}
    exam_codes = {t.get("tag") for t in exam}

    cond_period: Dict[str, Dict[str, int]] = {}
    exam_period: Dict[str, Dict[str, int]] = {}

    for period, d in (period_stats or {}).items():
        d = d or {}
        cond_period[period] = {k: v for k, v in d.items() if k in cond_codes}
        exam_period[period] = {k: v for k, v in d.items() if k in exam_codes}

    return cond, exam, cond_period, exam_period


def _build_weight_stats(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    recentWeights + profile.weightCurrent 를 이용해 체중 흐름을 '팩트'로 만든다.
    (LLM이 추론하지 않아도 되게)
    """
    profile = body.get("profile") or {}
    recent = body.get("recentWeights") or body.get("recent_weights") or []

    entries: List[Tuple[date, float]] = []
    for it in recent:
        if not isinstance(it, dict):
            continue
        d = _parse_visit_date(it.get("date"))
        w = _safe_float(it.get("weight"))
        if d and w is not None:
            entries.append((d, w))

    entries.sort(key=lambda x: x[0])

    profile_w = _safe_float(profile.get("weightCurrent") or profile.get("weight_current"))

    if len(entries) == 0:
        if profile_w is None:
            return {
                "status": "none",
                "latest": None,
                "previous": None,
                "deltaKg": None,
                "message": "체중 기록이 아직 없어요.",
            }
        return {
            "status": "single",
            "latest": {"date": None, "weight": profile_w},
            "previous": None,
            "deltaKg": None,
            "message": f"현재 체중은 약 {profile_w:.1f}kg이에요.",
        }

    latest_d, latest_w = entries[-1]
    prev = entries[-2] if len(entries) >= 2 else None

    if not prev:
        return {
            "status": "single",
            "latest": {"date": latest_d.isoformat(), "weight": latest_w},
            "previous": None,
            "deltaKg": None,
            "message": f"최근 체중 기록은 {latest_w:.1f}kg이에요.",
        }

    prev_d, prev_w = prev
    delta = latest_w - prev_w
    abs_delta = abs(delta)

    if abs_delta < 0.05:
        status = "stable"
        msg = "최근 체중은 안정적인 편이에요."
    elif delta > 0:
        status = "up"
        msg = f"최근 체중이 약 {abs_delta:.1f}kg 늘었어요."
    else:
        status = "down"
        msg = f"최근 체중이 약 {abs_delta:.1f}kg 줄었어요."

    return {
        "status": status,
        "latest": {"date": latest_d.isoformat(), "weight": latest_w},
        "previous": {"date": prev_d.isoformat(), "weight": prev_w},
        "deltaKg": round(delta, 2),
        "message": msg,
    }


def _build_schedule_stats(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    schedules 를 이용해 '다가오는 약속'을 팩트로 만든다.
    """
    schedules = body.get("schedules") or body.get("schedule") or []
    today = date.today()

    upcoming: List[Tuple[date, str]] = []
    for it in schedules:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip() or "약속"
        d = _parse_visit_date(it.get("date"))
        if not d:
            continue

        is_upcoming = it.get("isUpcoming")
        if is_upcoming is None:
            is_upcoming = d >= today

        if is_upcoming and d >= today:
            upcoming.append((d, title))

    upcoming.sort(key=lambda x: x[0])

    if not upcoming:
        return {
            "upcomingCount": 0,
            "next": None,
            "message": "다가오는 약속이 아직 없어요.",
        }

    next_d, next_title = upcoming[0]
    days = (next_d - today).days
    if days == 0:
        when = "오늘"
    elif days == 1:
        when = "내일"
    else:
        when = f"{days}일 뒤"

    return {
        "upcomingCount": len(upcoming),
        "next": {"date": next_d.isoformat(), "title": next_title, "daysUntil": days},
        "message": f"다가오는 약속이 {len(upcoming)}개 있어요. 가장 가까운 약속은 {when} ‘{next_title}’예요.",
    }


def _build_tag_stats(
    medical_history: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    진료 이력 리스트에서 CONDITION_TAGS 를 기준으로

    - tags: [{tag, label, count, recentDates}]
    - periodStats: {"1m": {...}, "3m": {...}, "1y": {...}}

    를 만들어서 반환.

    1순위: iOS 에서 넘어온 record["tags"] 사용
    2순위: tags 가 비어 있을 때만 diagnosis/clinic_name 에서 키워드 검색

    ✅ 변경:
    - keyword 매칭(2순위)에서는 exam/test류 태그를 '추론'하지 않는다.
      (검사는 질환이 아니라서, 기록만으로 AI가 판별하면 오해 가능성이 큼)
    """
    today = date.today()

    agg: Dict[str, Dict[str, Any]] = {}

    period_stats: Dict[str, Dict[str, int]] = {
        "1m": {},
        "3m": {},
        "1y": {},
    }

    for mh in medical_history:
        visit_str = mh.get("visitDate") or mh.get("visit_date") or ""
        visit_dt = _parse_visit_date(visit_str)
        visit_date_str = visit_dt.isoformat() if visit_dt else None

        record_tags: List[str] = mh.get("tags") or []
        used_codes: set[str] = set()

        # 1) tags 우선 사용
        if record_tags:
            for code in record_tags:
                code = (code or "").strip()
                cfg = CONDITION_TAGS.get(code)
                if not cfg:
                    continue
                used_codes.add(code)

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

                if visit_dt:
                    days = (today - visit_dt).days
                    if days <= 365:
                        period_stats["1y"][cfg.code] = period_stats["1y"].get(cfg.code, 0) + 1
                    if days <= 90:
                        period_stats["3m"][cfg.code] = period_stats["3m"].get(cfg.code, 0) + 1
                    if days <= 30:
                        period_stats["1m"][cfg.code] = period_stats["1m"].get(cfg.code, 0) + 1

        # 2) tags 없을 때만 diagnosis/clinic_name 키워드 매칭
        if not record_tags:
            diag = mh.get("diagnosis") or ""
            clinic = mh.get("clinicName") or mh.get("clinic_name") or ""
            base_text = f"{diag} {clinic}".strip()
            if not base_text:
                continue

            text_lower = base_text.lower()

            for cfg in CONDITION_TAGS.values():
                # ✅ exam/test류는 추론(키워드 매칭) 대상에서 제외
                if _is_exam_like_tag(cfg.code, cfg):
                    continue

                code_lower = cfg.code.lower()
                keyword_hit = False

                if code_lower in text_lower:
                    keyword_hit = True
                else:
                    for kw in getattr(cfg, "keywords", []) or []:
                        if str(kw).lower() in text_lower:
                            keyword_hit = True
                            break

                if not keyword_hit:
                    continue

                if cfg.code in used_codes:
                    continue
                used_codes.add(cfg.code)

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
    "prevent_vaccine_corona": [
        "접종 후 1~2일 동안은 기력, 식욕 변화를 잘 관찰해 주세요.",
    ],
}


# ------------------------------------------------
# 9. Gemini 기반 AI 요약 생성 (✅ 체중/일정 포함 + 검사태그 추론 금지)
# ------------------------------------------------

def _build_gemini_prompt(
    pet_name: str,
    condition_tags: List[Dict[str, Any]],
    exam_tags: List[Dict[str, Any]],
    period_stats: Dict[str, Dict[str, int]],
    weight_stats: Dict[str, Any],
    schedule_stats: Dict[str, Any],
    body: Dict[str, Any],
    max_condition_tags: int = 4,
    max_exam_tags: int = 3,
) -> str:
    profile = body.get("profile") or {}
    species = profile.get("species", "dog")
    age_text = profile.get("ageText") or profile.get("age_text") or ""
    weight_current = profile.get("weightCurrent") or profile.get("weight_current")
    allergies = profile.get("allergies") or []

    # 최근 진료 요약 (최대 4개)
    mh_list = body.get("medicalHistory") or body.get("medical_history") or []
    mh_summary_lines = []
    for mh in mh_list[:4]:
        clinic = mh.get("clinicName") or mh.get("clinic_name") or ""
        diag = mh.get("diagnosis") or ""
        visit = mh.get("visitDate") or mh.get("visit_date") or ""
        mh_summary_lines.append(f"- {visit} / {clinic} / {diag}".strip())

    # 컨디션 태그 라인(Top N)
    cond_lines = []
    for t in (condition_tags or [])[:max_condition_tags]:
        recent_dates = ", ".join((t.get("recentDates") or [])[:2])
        cond_lines.append(
            f"- {t.get('label')} ({t.get('tag')}) : {t.get('count')}회 (최근: {recent_dates or '정보 없음'})"
        )

    # 검사 태그 라인(Top N)
    exam_lines = []
    for t in (exam_tags or [])[:max_exam_tags]:
        recent_dates = ", ".join((t.get("recentDates") or [])[:2])
        exam_lines.append(
            f"- {t.get('label')} ({t.get('tag')}) : {t.get('count')}회 (최근: {recent_dates or '정보 없음'})"
        )

    # 체중/일정 팩트 라인
    w_line = weight_stats.get("message") if isinstance(weight_stats, dict) else ""
    s_line = schedule_stats.get("message") if isinstance(schedule_stats, dict) else ""

    prompt = f"""
당신은 반려동물 건강관리 전문가입니다.
아래 '사실 정보'만을 바탕으로 보호자에게 한국어로 3~5문장 정도의 따뜻하고 차분한 요약을 해주세요.

[반려동물 기본 정보]
•⁠  ⁠이름: {pet_name}
•⁠  ⁠종: {species}
•⁠  ⁠나이 정보: {age_text or '정보 없음'}
•⁠  ⁠현재 체중(프로필): {weight_current if weight_current is not None else '정보 없음'} kg
•⁠  ⁠알러지: {", ".join(allergies) if allergies else "정보 없음"}

[체중 기록 요약(팩트)]
{w_line or "체중 정보가 아직 부족해요."}

[다가오는 약속 요약(팩트)]
{s_line or "약속 정보가 아직 없어요."}

[컨디션/예방 태그 통계(기록 기반)]
{os.linesep.join(cond_lines) if cond_lines else "해석 가능한 컨디션 태그가 아직 없어요."}

[검사/영상 태그 통계(참고)]
{os.linesep.join(exam_lines) if exam_lines else "최근 검사 태그 정보 없음"}

[최근 진료 이력 요약(최대 4개)]
{os.linesep.join(mh_summary_lines) if mh_summary_lines else "진료 내역 없음"}

작성 가이드:
1) 보호자에게 말하듯이 존댓말로, 과하게 겁주지 말고 안정감을 주세요.
2) '컨디션/예방 태그'는 기록 카테고리이며, 태그만으로 질병을 확정하지 마세요. (단정 금지)
3) '검사/영상 태그'는 '진단'이 아닙니다. 검사 결과를 추론하거나 병명을 만들어내지 마세요.
4) 데이터가 부족하면 솔직히 "아직 기록이 부족해요"라고 말하고, 다음에 기록하면 좋은 항목을 1개만 제안해 주세요.
5) 출력은 마크다운 없이 문장만 출력합니다. 불릿/번호/따옴표/코드블록을 쓰지 마세요.
""".strip()

    return prompt


def _generate_gemini_summary(
    pet_name: str,
    condition_tags: List[Dict[str, Any]],
    exam_tags: List[Dict[str, Any]],
    period_stats: Dict[str, Dict[str, int]],
    weight_stats: Dict[str, Any],
    schedule_stats: Dict[str, Any],
    body: Dict[str, Any],
) -> Optional[str]:
    if settings.GEMINI_ENABLED.lower() != "true":
        return None
    if not settings.GEMINI_API_KEY:
        return None
    if genai is None:
        return None

    # 기록이 너무 빈약하면 Gemini 대신 fallback이 더 안전
    has_any_signal = bool(condition_tags) or bool(exam_tags) or bool(weight_stats) or bool(schedule_stats)
    if not has_any_signal:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        prompt = _build_gemini_prompt(
            pet_name=pet_name,
            condition_tags=condition_tags,
            exam_tags=exam_tags,
            period_stats=period_stats,
            weight_stats=weight_stats,
            schedule_stats=schedule_stats,
            body=body,
        )
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


def _fallback_summary(
    pet_name: str,
    has_history: bool,
    condition_tags: List[Dict[str, Any]],
    exam_tags: List[Dict[str, Any]],
    weight_stats: Dict[str, Any],
    schedule_stats: Dict[str, Any],
) -> str:
    """
    ✅ Gemini가 꺼져 있거나, 방어적으로 fallback이 필요할 때 쓰는 요약.
    (따뜻한 톤 + 과잉해석 방지)
    """
    pieces: List[str] = []

    w_msg = (weight_stats or {}).get("message")
    s_msg = (schedule_stats or {}).get("message")

    if w_msg:
        pieces.append(w_msg)
    if s_msg:
        pieces.append(s_msg)

    if has_history and condition_tags:
        top = condition_tags[0]
        pieces.append(f"최근 기록에는 ‘{top.get('label')}’ 관련 항목이 {top.get('count')}회 있었어요.")
    elif has_history and not condition_tags:
        pieces.append("진료 기록은 있지만, 아직 컨디션으로 정리할 단서가 조금 더 필요해요.")
    else:
        pieces.append(f"{pet_name}의 기록이 아직 많지 않아요. 천천히 쌓이면 더 잘 정리해 드릴게요.")

    # 검사 태그는 '사실'로만 언급
    if exam_tags:
        top_exam = exam_tags[0]
        pieces.append(f"참고로 최근에 ‘{top_exam.get('label')}’ 같은 검사 기록도 있었어요.")

    # 마지막은 안심 문장
    pieces.append("오늘은 이 정도만 확인해도 충분해요. 필요한 순간에만 조금씩 기록해 주세요.")

    # 3~5문장 정도로 정리
    return " ".join(pieces[:5])


# ------------------------------------------------
# 10. AI 케어 분석 엔드포인트 (✅ 태그+체중+일정 반영)
# ------------------------------------------------

@app.post("/api/ai/analyze")
async def analyze_pet_health(body: Dict[str, Any]):
    """
    PetHealth+ AI 케어:
    - medicalHistory.tags 기반 통계(우선)
    - 체중 흐름(recentWeights)
    - 다가오는 약속(schedules)
    를 함께 반영해 요약을 만든다.

    ✅ 핵심 변경
    1) weights/schedules를 실제로 읽어서 summary에 반영
    2) exam_* 태그는 '질환/컨디션'으로 해석하지 않게 분리
    3) Gemini 프롬프트에서 검사 결과/병명 추론 금지
    """
    try:
        print("[AI] raw body =", json.dumps(body, ensure_ascii=False))
    except Exception:
        print("[AI] raw body (repr) =", repr(body))

    profile = body.get("profile") or {}
    pet_name = profile.get("name") or "반려동물"

    medical_history = body.get("medicalHistory") or body.get("medical_history") or []
    has_history = len(medical_history) > 0

    # ✅ 팩트 신호(체중/일정)
    weight_stats = _build_weight_stats(body)
    schedule_stats = _build_schedule_stats(body)

    # ✅ 태그 통계
    all_tags, all_period_stats = _build_tag_stats(medical_history)

    # ✅ 컨디션 태그 vs 검사 태그 분리 + periodStats도 분리
    condition_tags, exam_tags, period_stats, exam_period_stats = _split_tag_stats(all_tags, all_period_stats)

    # ✅ careGuide는 컨디션 태그에만
    care_guide: Dict[str, List[str]] = {}
    for t in condition_tags:
        code = t.get("tag")
        if code in DEFAULT_CARE_GUIDE:
            care_guide[code] = DEFAULT_CARE_GUIDE[code]

    # ✅ 요약 생성
    summary: str

    if settings.STUB_MODE.lower() == "true":
        summary = (
            f"{pet_name}의 기록을 차분히 정리해 봤어요. "
            f"{weight_stats.get('message', '')} "
            f"{schedule_stats.get('message', '')} "
            "오늘은 이 정도만 확인해도 충분해요."
        ).strip()
    else:
        # 1) 기록이 거의 없으면 기본 문구
        if not has_history and (weight_stats.get("status") in ("none",) and schedule_stats.get("upcomingCount") == 0):
            summary = (
                f"{pet_name}의 기록이 아직 많지 않아요. "
                "필요할 때 한 줄씩만 남겨도, 다음부터는 더 잘 정리해 드릴게요."
            )
        else:
            # 2) Gemini 시도 → 실패 시 fallback
            ai_summary = _generate_gemini_summary(
                pet_name=pet_name,
                condition_tags=condition_tags,
                exam_tags=exam_tags,
                period_stats=period_stats,
                weight_stats=weight_stats,
                schedule_stats=schedule_stats,
                body=body,
            )
            if ai_summary:
                summary = ai_summary
            else:
                summary = _fallback_summary(
                    pet_name=pet_name,
                    has_history=has_history,
                    condition_tags=condition_tags,
                    exam_tags=exam_tags,
                    weight_stats=weight_stats,
                    schedule_stats=schedule_stats,
                )

    response = {
        # 기존 호환
        "summary": summary,
        "tags": condition_tags,
        "periodStats": period_stats,
        "careGuide": care_guide,

        # ✅ 추가 필드(기존 iOS에서 몰라도 무시됨)
        "examTags": exam_tags,
        "examPeriodStats": exam_period_stats,
        "weightStats": weight_stats,
        "scheduleStats": schedule_stats,
    }

    print(f"[AI] response condition_tags={len(condition_tags)} exam_tags={len(exam_tags)}")
    return response
