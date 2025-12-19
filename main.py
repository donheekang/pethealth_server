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
# 4. 영수증 파서 (✅ 최소 추출: 병원명/날짜만)
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


def parse_receipt_minimal(text: str) -> dict:
    """
    ✅ OCR 텍스트에서 "병원명/방문일"만 안전하게 추출.
    - items / totalAmount 추출하지 않음 (앱에서 수기 입력 & 자동합계가 더 안전)
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1) 병원명 추정
    hospital_name = guess_hospital_name(lines)

    # 2) 날짜 추정 (시간 포함 우선)
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

    return {
        "hospitalName": hospital_name,
        "visitAt": visit_at,
    }


def parse_receipt_ai(raw_text: str) -> Optional[dict]:
    """
    (유지) Gemini를 써서 영수증을 파싱하는 버전.
    ⚠️ 현재는 "최소 입력" 정책으로 receipt에서 사용하지 않음.
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

    # ✅ 3) 최소 파싱: 병원명/날짜만
    minimal = parse_receipt_minimal(ocr_text) if ocr_text else {"hospitalName": "", "visitAt": None}

    parsed_for_dto = {
        "clinicName": (minimal.get("hospitalName") or "").strip(),
        "visitDate": minimal.get("visitAt"),
        # ✅ 아래는 "자동 입력" 안 함 (앱에서 수기 입력/자동 합계)
        "diseaseName": None,
        "symptomsSummary": None,
        "items": [],
        "totalAmount": None,
    }

    # 병원명 앞의 '원 명:' 같은 접두어 제거
    clinic_name = (parsed_for_dto.get("clinicName") or "").strip()
    clinic_name = re.sub(r"^원\s*명[:：]?\s*", "", clinic_name)
    parsed_for_dto["clinicName"] = clinic_name

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,     # (선택) 디버그/재파싱용
        "objectKey": key,
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


# (2-1) PDF 삭제
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


def _build_tag_stats(
    medical_history: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    ✅ record.tags(서버 코드)만 사용해서 통계 생성.
    ❌ diagnosis/clinic_name 기반 "키워드 추정"은 완전 제거 (안전/신뢰 우선)

    - tags: [{tag, label, group, count, recentDates}]
    - periodStats: {"1m": {...}, "3m": {...}, "1y": {...}}
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
        if not record_tags:
            continue

        used_codes: set[str] = set()

        for code in record_tags:
            cfg = CONDITION_TAGS.get(code)
            if not cfg:
                continue

            canonical_code = cfg.code  # alias가 들어와도 canonical code로 집계
            if canonical_code in used_codes:
                continue
            used_codes.add(canonical_code)

            stat = agg.setdefault(
                canonical_code,
                {
                    "tag": canonical_code,
                    "label": cfg.label,
                    "group": getattr(cfg, "group", ""),
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
                    period_stats["1y"][canonical_code] = period_stats["1y"].get(canonical_code, 0) + 1
                if days <= 90:
                    period_stats["3m"][canonical_code] = period_stats["3m"].get(canonical_code, 0) + 1
                if days <= 30:
                    period_stats["1m"][canonical_code] = period_stats["1m"].get(canonical_code, 0) + 1

    for stat in agg.values():
        stat["recentDates"] = sorted(stat["recentDates"], reverse=True)

    tags = sorted(agg.values(), key=lambda x: x["count"], reverse=True)
    return tags, period_stats


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

    mh_list = body.get("medicalHistory") or []
    mh_summary_lines = []
    for mh in mh_list[:8]:
        clinic = mh.get("clinicName") or mh.get("clinic_name") or ""
        visit = mh.get("visitDate") or mh.get("visit_date") or ""
        tag_list = mh.get("tags") or []
        mh_summary_lines.append(f"- {visit} / {clinic} / tags={tag_list}")

    tag_lines = []
    for t in tags:
        recent_dates = ", ".join(t.get("recentDates", [])[:3])
        group = t.get("group") or ""
        tag_lines.append(
            f"- {t['label']} ({t['tag']}, group={group}) : {t['count']}회 (최근: {recent_dates or '정보 없음'})"
        )

    prompt = f"""
당신은 반려동물 건강 기록을 '정리/요약'해 주는 어시스턴트입니다.
아래 입력은 보호자가 앱에서 직접 기록한 태그(확정 정보)입니다.

[중요 규칙]
1) 아래 '태그'에 없는 질환/컨디션을 새로 추정하거나 만들어내지 마세요.
2) group=exam/medication/procedure 태그는 '질환 진단'이 아니라 검사/처방/시술 기록입니다. 이를 근거로 질환을 단정하지 마세요.
3) 의료적 진단/판단을 하지 말고, 보호자 관점의 정리/안심/현실적인 관리 팁만 제공하세요.
4) 출력은 마크다운 없이 '문장만' 출력합니다. 불릿/번호/따옴표/코드블록 금지.

[반려동물 기본 정보]
이름: {pet_name}
종: {species}
나이: {age_text or '정보 없음'}
현재 체중: {weight if weight is not None else '정보 없음'} kg

[확정 태그 통계]
{os.linesep.join(tag_lines) if tag_lines else '태그 없음'}

[최근 진료 이력 요약(최대 8개)]
{os.linesep.join(mh_summary_lines) if mh_summary_lines else '진료 내역 없음'}

요약은 3~5문장으로 짧게 작성해 주세요.
너무 무섭게 말하지 말고, "잘 관리되고 있다"는 안정감을 주는 톤으로 작성해 주세요.
""".strip()

    return prompt


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

    # ✅ tags가 비어도(기록이 없더라도) 요약은 가능하도록 열어둠
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
    PetHealth+ AI 케어:
    - ✅ record.tags(확정 태그)만 기반으로 통계/요약 생성
    - ❌ diagnosis/clinic_name 키워드 추정 OFF
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

    # 케어 가이드: condition_tags.py의 guide를 그대로 사용
    care_guide: Dict[str, List[str]] = {}
    for t in tags:
        code = t["tag"]
        cfg = CONDITION_TAGS.get(code)
        if cfg and getattr(cfg, "guide", None):
            care_guide[code] = list(cfg.guide)

    # 요약 문장 (Gemini 우선, 없으면 규칙 기반)
    ai_summary = _generate_gemini_summary(pet_name, tags, period_stats, body)

    if ai_summary:
        summary = ai_summary
    else:
        if not has_history:
            summary = (
                f"{pet_name}의 병원 기록이 아직 많지 않아요. "
                "필요할 때 한 줄씩만 남겨도, 다음부터는 더 잘 정리해 드릴게요."
            )
        elif not tags:
            summary = (
                f"{pet_name}의 병원 기록은 있지만, "
                "아직 확정 태그가 저장된 기록이 없어서 컨디션별 통계를 만들기 어려워요. "
                "기록할 때 태그가 함께 저장되면 더 정확하게 정리해 드릴게요."
            )
        else:
            top = tags[0]
            summary = (
                f"최근 기록에서 '{top['label']}' 관련 태그가 {top['count']}회 확인됐어요. "
                "기록을 바탕으로 관리 포인트를 차분히 정리해 드렸어요."
            )

    response = {
        "summary": summary,
        "tags": tags,
        "periodStats": period_stats,
        "careGuide": care_guide,
    }

    print(f"[AI] response tags={len(tags)}")
    return response
