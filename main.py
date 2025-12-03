from __future__ import annotations

import os
import io
import json
import uuid
import tempfile
import re
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ------------------------------------------
# 0. Optional: condition_tags / Gemini
# ------------------------------------------
try:
    from condition_tags import CONDITION_TAGS
except ImportError:
    CONDITION_TAGS = {}
    print("Warning: condition_tags.py not found. AI tagging will be limited.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google.generativeai not installed. GEMINI 기능 비활성화.")


# ------------------------------------------
# 1. SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # Google Vision OCR
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Gemini 사용 여부 + API Key
    GEMINI_ENABLED: str = "false"
    GEMINI_API_KEY: str = ""
    # ✅ 최신 모델 이름로 기본값 변경
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest"

    # AI 대신 더미 데이터만 돌리는 모드
    STUB_MODE: str = "false"

    class Config:
        env_file = ".env"


settings = Settings()


# ------------------------------------------
# 2. AWS S3 CLIENT
# ------------------------------------------

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


def upload_to_s3(file_obj, key: str, content_type: str) -> str:
    """
    file-like 객체를 S3에 업로드하고 presigned URL 반환
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
            ExpiresIn=7 * 24 * 3600,  # 7일
        )
        return url

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 인증 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {str(e)}")


# ------------------------------------------
# 3. GOOGLE VISION OCR
# ------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS 환경변수가 비어있습니다.")

    # 1) JSON 내용 시도
    try:
        info = json.loads(cred_value)
        return vision.ImageAnnotatorClient.from_service_account_info(info)
    except json.JSONDecodeError:
        # 2) JSON이 아니면 경로로 간주
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


# ------------------------------------------
# 4. 영수증 파싱 로직 (정규식 Fallback)
# ------------------------------------------

def guess_hospital_name(lines: List[str]) -> str:
    """
    병원명 추론: 키워드 + 위치 + 형태 기반으로 대략 고르기
    """
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

        # 1) 키워드 점수
        if any(k in text for k in keywords):
            score += 5

        # 2) 위치 점수 (위쪽일수록 가산점)
        if idx <= 4:
            score += 2

        # 3) 주소/전화번호처럼 보이면 감점
        if any(x in line for x in ["TEL", "전화", "FAX", "팩스", "도로명"]):
            score -= 2

        # 4) 숫자 많으면 감점 (사업자번호 등)
        digit_count = sum(c.isdigit() for c in line)
        if digit_count >= 8:
            score -= 1

        # 5) 길이 너무 짧거나 너무 길면 감점
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
    한국 동물병원 영수증 OCR 텍스트를 구조화.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1) 병원명
    hospital_name = guess_hospital_name(lines)

    # 2) 방문일시
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

    # 시간 없는 날짜만 있는 경우
    if not visit_at:
        dt_pattern_short = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})")
        for line in lines:
            m = dt_pattern_short.search(line)
            if m:
                y, mo, d = map(int, m.groups())
                visit_at = datetime(y, mo, d).strftime("%Y-%m-%d")
                break

    # 3) 전체 라인에서 합계 후보 금액
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

    # 4) 진료 항목 영역 추출
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
        # 헤더/설명 줄 스킵
        if any(k in line for k in ["동물명", "항목", "단가", "수량", "금액"]):
            continue

        # (1) *로 시작하는 줄 → 항목 이름
        if line.startswith("*"):
            name = line.lstrip("*").strip().strip(".")
            if name:
                names.append(name)
            continue

        # (2) 숫자/콤마/공백만 있는 줄 → 금액
        if re.fullmatch(r"[0-9,\s]+", line):
            m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", line)
            if m:
                amt = int(m.group(1).replace(",", ""))
                if amt > 0:
                    prices.append(amt)
            continue

        # (3) 텍스트 + 숫자가 같이 있는 줄
        m = re.search(r"(.+?)\s+(\d{1,3}(?:,\d{3})+|\d+)", line)
        if m and ":" not in line and "[" not in line:
            name = m.group(1).strip()
            amt = int(m.group(2).replace(",", ""))
            if name:
                names.append(name)
                prices.append(amt)

    # 5) 이름-금액 매칭
    items: List[Dict] = []
    pair_count = min(len(names), len(prices))
    for i in range(pair_count):
        items.append({"name": names[i], "amount": prices[i]})

    # 6) totalAmount 결정
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
    Gemini를 이용한 영수증 AI 파싱
    """
    if settings.GEMINI_ENABLED.lower() != "true":
        return None
    if not settings.GEMINI_API_KEY:
        return None
    if genai is None:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # ✅ 최신 모델명 사용
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        prompt = f"""
        너는 한국 동물병원 영수증을 구조화된 JSON으로 정리하는 어시스턴트야.
        다음은 OCR로 읽은 영수증 텍스트야:

        \"\"\"{raw_text}\"\"\"

        이 텍스트를 분석해서 아래 형식의 JSON만 돌려줘.
        키 이름은 반드시 아래와 같아야 해.

        형식:
        {{
          "clinicName": string or null,
          "visitDate": string or null,   // "YYYY-MM-DD" 또는 "YYYY-MM-DD HH:MM"
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
        JSON만 출력하고, 설명 문장은 쓰지 마.
        """

        resp = model.generate_content(prompt)

        # google.generativeai 응답에서 텍스트 추출
        text = getattr(resp, "text", None)
        if not text and resp.candidates:
            parts = resp.candidates[0].content.parts
            text = "".join(p.text for p in parts if hasattr(p, "text"))
        text = (text or "").strip()

        # 코드블록 안에 있을 경우 정리
        if "⁠  " in text:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        data = json.loads(text)

        # 필수 키 검증
        for key in ["clinicName", "visitDate", "items", "totalAmount"]:
            if key not in data:
                return None

        # items 정규화
        if not isinstance(data.get("items"), list):
            data["items"] = []

        fixed_items = []
        for it in data["items"]:
            if isinstance(it, dict):
                name = it.get("name", "항목")
                price = it.get("price", 0)
                fixed_items.append(
                    {"name": str(name), "price": int(price) if price else 0}
                )
        data["items"] = fixed_items

        return data

    except Exception as e:
        print("parse_receipt_ai error:", e)
        return None


# ------------------------------------------
# 5. AI Care (태그 헬퍼 & DTO)
# ------------------------------------------

def get_tags_definition_for_prompt() -> str:
    """Gemini 프롬프트용 태그 목록 생성"""
    if not CONDITION_TAGS:
        return "태그 정의 파일이 없습니다."

    lines = []
    lines.append("[가능한 질환/예방 태그 목록]")
    for code, config in CONDITION_TAGS.items():
        keywords_str = ", ".join(config.keywords[:3])
        line = f"- {code}: {config.label} (관련어: {keywords_str})"
        lines.append(line)
    return "\n".join(lines)


class CamelBase(BaseModel):
    """camelCase JSON 을 받아주고 내보내는 공통 설정"""
    class Config:
        allow_population_by_field_name = True
        orm_mode = True
        extra = "ignore"   # iOS에서 추가 필드가 와도 무시


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


class AICareResponse(CamelBase):
    summary: str
    detail_analysis: str = Field(..., alias="detailAnalysis")
    weight_trend_status: str = Field(..., alias="weightTrendStatus")
    risk_factors: List[str] = Field(..., alias="riskFactors")
    action_guide: List[str] = Field(..., alias="actionGuide")
    health_score: int = Field(..., alias="healthScore")
    condition_tags: List[str] = Field(default_factory=list, alias="conditionTags")


# ------------------------------------------
# 6. FASTAPI APP SETUP
# ------------------------------------------

app = FastAPI(title="PetHealth+ Server", version="1.0.0")

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


# ------------------------------------------
# 7. ENDPOINTS
# ------------------------------------------

# (1) 영수증 업로드 & 분석
@app.post("/receipt/upload")
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

    # 🔧 병원명 앞의 '원 명:' 같은 접두어 제거
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


# (3) 리스트 조회 (단순 버전)
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


# (4) AI 종합 분석
@app.post("/api/ai/analyze", response_model=AICareResponse)
async def analyze_pet_health(req: AICareRequest):
    """
    PetHealth+ AI 케어: 종합 건강 리포트 생성
    """

    # 0. STUB_MODE 이면 항상 더미 리포트 반환 (Gemini 없이도 UI 확인용)
    if settings.STUB_MODE.lower() == "true":
        name = req.profile.name
        return AICareResponse(
            summary=f"{name}의 기본 상태를 정리했어요.",
            detail_analysis=(
                f"{name}의 최근 체중·스케줄·진료 기록을 바탕으로 전체적인 건강 상태를 가볍게 점검한 "
                "샘플 리포트입니다. 실제 AI 분석이 활성화되면 보다 정교한 맞춤 케어 가이드를 제공할 예정이에요."
            ),
            weight_trend_status="데이터 수집 중",
            risk_factors=["AI 분석 기능 준비 중"],
            action_guide=[
                "체중을 주기적으로 기록해 주세요.",
                "예방 접종 및 검진 일정을 스케줄에 등록해 주세요.",
            ],
            health_score=75,
            condition_tags=[],
        )

    # 1. Gemini 비활성화 시 Fallback
    if settings.GEMINI_ENABLED.lower() != "true" or not settings.GEMINI_API_KEY or genai is None:
        return AICareResponse(
            summary="AI 설정이 필요해요.",
            detail_analysis="서버 환경변수 GEMINI_API_KEY와 GEMINI_ENABLED를 확인해주세요.",
            weight_trend_status="데이터 없음",
            risk_factors=[],
            action_guide=["서버 점검 필요"],
            health_score=0,
            condition_tags=[],
        )

    # 2. Gemini 호출
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        tags_context = get_tags_definition_for_prompt()

        prompt = f"""
        당신은 'PetHealth+' 앱의 수의학 AI 파트너입니다.
        반려동물 데이터를 분석해 보호자에게 따뜻하고 정확한 조언을 주세요.

        [반려동물 정보]
        - 이름/종: {req.profile.name} ({req.profile.species})
        - 나이: {req.profile.age_text}
        - 현재 체중: {req.profile.weight_current}kg
        - 알러지: {", ".join(req.profile.allergies) if req.profile.allergies else "없음"}

        [최근 체중 기록]
        {req.recent_weights}

        [진료 이력]
        {req.medical_history}

        [스케줄]
        {req.schedules}

        {tags_context}

        [분석 요청사항]
        1. 체중: 최근 변화 추세(증가/감소/유지)를 0.1kg 단위로 민감하게 체크하세요.
        2. 리스크: 노령 여부, 체중 급변, 빈번한 병원 방문 등을 고려해 위험 요소를 찾으세요.
        3. 액션: 구체적이고 실천 가능한 행동을 제안하세요.
        4. 태그: 위 태그 목록 중, 이 동물의 '현재 상태', '최근 치료', '예방 접종 이력'에 해당하는 코드(code)를 모두 고르세요.
           - "음성(Negative)"이거나 "정상"인 질환은 선택하지 마세요.
        5. 점수: 0~100점 (건강할수록 높은 점수)

        [출력 포맷 (JSON Only)]
        {{
            "summary": "홈 화면 카드용 40자 이내 핵심 요약",
            "detail_analysis": "전체적인 건강 상태 상세 분석 (줄바꿈 없이 3~5문장)",
            "weight_trend_status": "체중 상태 (예: 안정적, 급격한 증가, 감소 주의)",
            "risk_factors": ["위험 요소1", "위험 요소2"],
            "action_guide": ["추천 행동1", "추천 행동2"],
            "health_score": 85,
            "condition_tags": ["code1", "code2"]
        }}
        JSON만 출력하고, 다른 문장은 쓰지 마.
        """

        resp = model.generate_content(prompt)

        text = getattr(resp, "text", None)
        if not text and resp.candidates:
            parts = resp.candidates[0].content.parts
            text = "".join(p.text for p in parts if hasattr(p, "text"))
        text = (text or "").strip()

        if "  ⁠" in text:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        data = json.loads(text)

        return AICareResponse(
            summary=data.get("summary", "건강 분석을 완료했어요."),
            detail_analysis=data.get("detail_analysis", data.get("detailAnalysis", "상세 분석 데이터가 없습니다.")),
            weight_trend_status=data.get("weight_trend_status", data.get("weightTrendStatus", "-")),
            risk_factors=data.get("risk_factors", data.get("riskFactors", [])),
            action_guide=data.get("action_guide", data.get("actionGuide", [])),
            health_score=data.get("health_score", data.get("healthScore", 50)),
            condition_tags=data.get("condition_tags", data.get("conditionTags", [])),
        )

    except Exception as e:
        print(f"AI Analyze Error: {e}")
        return AICareResponse(
            summary="잠시 후 다시 시도해주세요.",
            detail_analysis=f"AI 분석 중 오류가 발생했습니다: {str(e)}",
            weight_trend_status="-",
            risk_factors=[],
            action_guide=[],
            health_score=0,
            condition_tags=[],
        )
