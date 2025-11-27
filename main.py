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

# Gemini (google-generativeai) - 설치 안 되어 있어도 서버가 죽지 않게 try/except
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ------------------------------------------
# 1. SETTINGS (환경 변수 연동)
# ------------------------------------------

class Settings(BaseSettings):
    # AWS S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # Google Vision OCR (JSON 내용 or 파일 경로)
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Google Gemini AI
    GEMINI_ENABLED: str = "true"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"  # 환경변수 없을 시 기본값

    # 기타
    STUB_MODE: str = "false"

    class Config:
        env_file = ".env"  # .env 파일이 있다면 로드

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
    """
    GOOGLE_APPLICATION_CREDENTIALS:
      - 서비스 계정 JSON '내용'일 수도 있고
      - JSON 파일 경로일 수도 있음
    """
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
    """
    Google Vision OCR로 텍스트 추출
    """
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
# 4. 영수증 OCR 파싱 로직 (Regex Fallback)
# ------------------------------------------

def guess_hospital_name(lines: List[str]) -> str:
    keywords = [
        "동물병원", "동물 병원", "동물의료", "동물메디컬", "동물 메디컬",
        "동물클리닉", "동물 클리닉", "애견병원", "애완동물병원", "펫병원",
        "종합동물병원", "동물의원", "동물병의원"
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
        if any(x in line for x in ["TEL", "전화", "FAX", "도로명"]):
            score -= 2
        if sum(c.isdigit() for c in line) >= 8:
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
    한국 동물병원 영수증 OCR 텍스트 구조화 (정규식 기반 Fallback)
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hospital_name = guess_hospital_name(lines)

    # 날짜/시간
    visit_at = None
    dt_pattern = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2}).*?(\d{1,2}):(\d{2})")
    
    # 시간 없는 날짜 패턴 추가
    dt_pattern_short = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})")

    for line in lines:
        m = dt_pattern.search(line)
        if m:
            y, mo, d, h, mi = map(int, m.groups())
            visit_at = datetime(y, mo, d, h, mi).strftime("%Y-%m-%d %H:%M")
            break
        
        # 시간 없으면 날짜만
        m2 = dt_pattern_short.search(line)
        if not visit_at and m2:
             y, mo, d = map(int, m2.groups())
             visit_at = datetime(y, mo, d).strftime("%Y-%m-%d")

    # 금액
    amt_pattern = re.compile(r"(?:₩|￦)?\s*(\d{1,3}(?:,\d{3})|\d+)\s(원)?\s*$")
    items: List[Dict] = []
    candidate_totals: List[int] = []

    for line in lines:
        m = amt_pattern.search(line)
        if not m:
            continue
        amount_str = m.group(1).replace(",", "")
        try:
            amount = int(amount_str)
        except ValueError:
            continue
        
        name = line[:m.start()].strip()
        lowered = name.replace(" ", "")
        
        if any(k in lowered for k in ["합계", "총액", "총금액", "합계금액"]):
            candidate_totals.append(amount)
            continue
        
        if not name:
            name = "항목"
        items.append({"name": name, "amount": amount})

    if candidate_totals:
        total_amount = max(candidate_totals)
    elif items:
        total_amount = sum(i["amount"] for i in items)
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
    Gemini LLM을 이용한 정밀 파싱
    """
    if settings.GEMINI_ENABLED.lower() != "true" or not settings.GEMINI_API_KEY or not genai:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        prompt = f"""
        너는 한국 동물병원 영수증을 구조화된 JSON으로 정리하는 어시스턴트야.
        다음은 OCR로 읽은 영수증 텍스트야:

        \"\"\"{raw_text}\"\"\"

        이 텍스트를 분석해서 아래 JSON 형식으로만 답해줘. 추가 설명 금지.

        형식:
        {{
          "clinicName": string or null,
          "visitDate": string or null,   // "YYYY-MM-DD" 또는 "YYYY-MM-DD HH:MM"
          "diseaseName": string or null,
          "symptomsSummary": string or null,
          "items": [ {{ "name": string, "price": integer or null }} ],
          "totalAmount": integer or null
        }}
        """

        resp = model.generate_content(prompt)
        text = resp.text.strip()

        # Markdown Strip
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
        fixed_items = []
        raw_items = data.get("items")
        if isinstance(raw_items, list):
            for it in raw_items:
                if isinstance(it, dict):
                    fixed_items.append({"name": it.get("name", "항목"), "price": it.get("price", 0)})
        data["items"] = fixed_items

        return data

    except Exception:
        return None


# ------------------------------------------
# 5. AI Care Models (DTO)
# ------------------------------------------

class PetProfileDTO(BaseModel):
    name: str
    species: str
    age_text: str = Field(..., alias="age_text")
    weight_current: Optional[float] = Field(None, alias="weight_current")
    allergies: List[str] = []

class WeightLogDTO(BaseModel):
    date: str
    weight: float

class MedicalHistoryDTO(BaseModel):
    visit_date: str = Field(..., alias="visit_date")
    clinic_name: str = Field(..., alias="clinic_name")
    item_count: int = Field(..., alias="item_count")

class ScheduleDTO(BaseModel):
    title: str
    date: str
    is_upcoming: bool = Field(..., alias="is_upcoming")

class AICareRequest(BaseModel):
    request_date: str = Field(..., alias="request_date")
    profile: PetProfileDTO
    recent_weights: List[WeightLogDTO] = Field(..., alias="recent_weights")
    medical_history: List[MedicalHistoryDTO] = Field(..., alias="medical_history")
    schedules: List[ScheduleDTO]

class AICareResponse(BaseModel):
    summary: str
    detail_analysis: str
    weight_trend_status: str
    risk_factors: List[str]
    action_guide: List[str]
    health_score: int


# ------------------------------------------
# 6. FASTAPI APP SETUP
# ------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "PetHealth+ Server Running (v1.2)"}

@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "gemini_model": settings.GEMINI_MODEL_NAME}


# ------------------------------------------
# 7. ENDPOINTS
# ------------------------------------------

# (1) 영수증 업로드 & 분석 (OCR + AI)

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

    # 파일 읽기
    data = await upload.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    # 1) S3 업로드
    file_url = upload_to_s3(
        file_like,
        key,
        content_type=upload.content_type or "image/jpeg",
    )

    # 2) OCR
    ocr_text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as tmp:
            tmp.write(data)
            tmp.flush()
            ocr_text = run_vision_ocr(tmp.name)
    except Exception as e:
        print(f"OCR Fail: {e}")
        ocr_text = ""

    # 3) AI 파싱 -> 실패시 Fallback
    ai_parsed = parse_receipt_ai(ocr_text) if ocr_text else None
    
    if ai_parsed:
        parsed_for_dto = ai_parsed
    else:
        # 정규식 Fallback
        fallback = parse_receipt_kor(ocr_text) if ocr_text else {
            "hospitalName": "", "visitAt": None, "items": [], "totalAmount": 0
        }
        
        items_dto = []
        for it in fallback.get("items", []):
            items_dto.append({"name": it.get("name"), "price": it.get("amount")})

        parsed_for_dto = {
            "clinicName": fallback.get("hospitalName"),
            "visitDate": fallback.get("visitAt"),
            "diseaseName": None,
            "symptomsSummary": None,
            "items": items_dto,
            "totalAmount": fallback.get("totalAmount"),
        }

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,
    }


# (2) PDF 업로드 (검사결과 / 증명서)

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
    lab_id = str(uuid.uuid4())
    original_base = os.path.splitext(file.filename or "")[0].strip() or "검사결과"
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")
    key = f"lab/{petId}/{safe_base}__{lab_id}.pdf"  # 구분자 __ 사용

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")
    created_at_iso = datetime.utcnow().isoformat()

    return {
        "id": lab_id,
        "petId": petId,
        "title": original_base,
        "memo": memo,
        "s3Url": file_url,
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
    cert_id = str(uuid.uuid4())
    original_base = os.path.splitext(file.filename or "")[0].strip() or "증명서"
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")
    key = f"cert/{petId}/{safe_base}__{cert_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")
    created_at_iso = datetime.utcnow().isoformat()

    return {
        "id": cert_id,
        "petId": petId,
        "title": original_base,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at_iso,
    }


# (3) 리스트 조회 (원본 로직 복원)

@app.get("/lab/list")
@app.get("/labs/list")
@app.get("/api/lab/list")
@app.get("/api/labs/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"
    response = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)

    items: List[Dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            # 파일명 파싱 (safe_base__uuid.pdf)
            filename = key.split("/")[-1]
            name_no_ext = os.path.splitext(filename)[0]

            base_title = "검사결과"
            file_id = name_no_ext

            if "__" in name_no_ext:
                safe_base, file_id = name_no_ext.rsplit("__", 1)
                base_title = safe_base.replace("_", " ")
            elif "" in name_no_ext and len(name_no_ext) > 36: # 예전 포맷 대비
                 safe_base, file_id = name_no_ext.rsplit("", 1)
                 base_title = safe_base.replace("_", " ")

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            display_title = f"{base_title} ({date_str})"

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )

            items.append({
                "id": file_id,
                "petId": petId,
                "title": display_title,
                "memo": None,
                "s3Url": url,
                "createdAt": created_at_iso,
            })

    return items


@app.get("/cert/list")
@app.get("/certs/list")
@app.get("/api/cert/list")
@app.get("/api/certs/list")
def get_cert_list(petId: str = Query(...)):
    prefix = f"cert/{petId}/"
    response = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)

    items: List[Dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            name_no_ext = os.path.splitext(filename)[0]

            base_title = "증명서"
            file_id = name_no_ext

            if "__" in name_no_ext:
                safe_base, file_id = name_no_ext.rsplit("__", 1)
                base_title = safe_base.replace("_", " ")

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            display_title = f"{base_title} ({date_str})"

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )

            items.append({
                "id": file_id,
                "petId": petId,
                "title": display_title,
                "memo": None,
                "s3Url": url,
                "createdAt": created_at_iso,
            })

    return items


# (4) AI Care 분석 (New)

@app.post("/api/ai/analyze", response_model=AICareResponse)
async def analyze_pet_health(req: AICareRequest):
    # Gemini 사용 불가 시 Fallback
    if settings.GEMINI_ENABLED.lower() != "true" or not settings.GEMINI_API_KEY:
        return AICareResponse(
            summary="AI 설정이 필요해요.",
            detail_analysis="서버 환경변수를 확인해주세요.",
            weight_trend_status="데이터 없음",
            risk_factors=[],
            action_guide=["서버 점검 필요"],
            health_score=0
        )

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        prompt = f"""
        당신은 'PetHealth+' 앱의 수의학 AI 파트너입니다.
        반려동물 데이터를 분석해 보호자에게 따뜻하고 정확한 조언을 주세요.

        [반려동물 정보]
        - 이름/종: {req.profile.name} ({req.profile.species})
        - 나이: {req.profile.age_text}
        - 현재 체중: {req.profile.weight_current}kg
        - 알러지: {", ".join(req.profile.allergies) if req.profile.allergies else "없음"}

        [최근 데이터]
        - 체중 기록(최신순): {req.recent_weights}
        - 진료 이력: {req.medical_history}
        - 스케줄: {req.schedules}

        [분석 요청사항]
        1. 체중: 최근 변화 추세(증가/감소/유지)를 0.1kg 단위로 민감하게 체크하세요.
        2. 리스크: 노령견/묘 여부, 체중 급변, 빈번한 병원 방문 등을 고려해 위험 요소를 찾으세요.
        3. 액션: 구체적이고 실천 가능한 행동을 제안하세요. (예: "간식 줄이기", "관절 영양제 고려")
        4. 점수: 0~100점 (건강할수록 높은 점수)

        [출력 포맷 (JSON Only)]
        {{
            "summary": "홈 화면 카드용 40자 이내 핵심 요약 (예: '체중이 0.2kg 줄었어요, 사료 양을 체크해보세요!')",
            "detail_analysis": "전체적인 건강 상태 상세 분석 (줄바꿈 없이 3~5문장)",
            "weight_trend_status": "체중 상태 (예: 안정적, 급격한 증가, 감소 주의)",
            "risk_factors": ["위험 요소1", "위험 요소2"],
            "action_guide": ["추천 행동1", "추천 행동2"],
            "health_score": 85
        }}
        """

        resp = model.generate_content(prompt)
        text = resp.text.strip()
        
        if "  ⁠" in text:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end+1]

        data = json.loads(text)

        return AICareResponse(
            summary=data.get("summary", "건강 분석을 완료했어요."),
            detail_analysis=data.get("detail_analysis", "상세 분석 데이터가 없습니다."),
            weight_trend_status=data.get("weight_trend_status", "-"),
            risk_factors=data.get("risk_factors", []),
            action_guide=data.get("action_guide", []),
            health_score=data.get("health_score", 50)
        )

    except Exception as e:
        print(f"AI Analyze Error: {e}")
        return AICareResponse(
            summary="잠시 후 다시 시도해주세요.",
            detail_analysis=f"오류가 발생했습니다: {str(e)}",
            weight_trend_status="-",
            risk_factors=[],
            action_guide=[],
            health_score=0
        )
