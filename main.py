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

# 🔥 작성하신 태그 파일 Import (같은 폴더에 condition_tags.py가 있어야 함)
try:
    from condition_tags import CONDITION_TAGS
except ImportError:
    CONDITION_TAGS = {}
    print("Warning: condition_tags.py not found. AI tagging will be limited.")

# Gemini Import
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
            ExpiresIn=7 * 24 * 3600,  # 7일 유효
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
# 4. 영수증 OCR 파싱 로직 (Regex Fallback)
# ------------------------------------------

def parse_receipt_kor(text: str) -> dict:
    """
    한국 동물병원 영수증 OCR 텍스트 구조화 (정규식 기반 Fallback)
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    # 병원명 추론 (간이)
    hospital_name = lines[0] if lines else ""
    for line in lines[:5]:
        if any(x in line for x in ["병원", "메디컬", "의료", "클리닉"]):
            hospital_name = line
            break

    # 날짜/시간
    visit_at = None
    dt_pattern = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})")
    for line in lines:
        m = dt_pattern.search(line)
        if m:
            y, mo, d = map(int, m.groups())
            # 시간 정보가 없으면 기본 날짜 포맷
            visit_at = f"{y:04d}-{mo:02d}-{d:02d}"
            break

    # 금액
    amt_pattern = re.compile(r"(\d{1,3}(?:,\d{3})|\d+)")
    items: List[Dict] = []
    candidate_totals: List[int] = []

    for line in lines:
        # 금액이 포함된 줄 찾기
        m = amt_pattern.search(line)
        if not m:
            continue
            
        try:
            amount = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        
        # 합계/총액 키워드가 있으면 후보군에 추가
        if any(k in line for k in ["합계", "총액", "결제", "청구"]):
            candidate_totals.append(amount)
            continue
        
        # 일반 항목으로 간주
        name = line[:m.start()].strip()
        if not name: name = "항목"
        items.append({"name": name, "amount": amount})

    # 총액 결정 (후보 중 최대값, 없으면 합산)
    if candidate_totals:
        total_amount = max(candidate_totals)
    elif items:
        total_amount = sum(i["amount"] for i in items)
    else:
        total_amount = 0

    return {
        "hospitalName": hospital_name,
        "visitAt": visit_at,
        "items": [], # Regex로는 항목 디테일을 완벽히 뽑기 어려워 생략 (총액 위주)
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
        너는 한국 동물병원 영수증을 구조화된 JSON으로 정리하는 AI야.
        OCR 텍스트: \"\"\"{raw_text}\"\"\"

        아래 JSON 형식으로만 답해줘. 추가 설명 금지.
        {{
          "clinicName": string or null,
          "visitDate": string or null,   // "YYYY-MM-DD"
          "diseaseName": string or null,
          "symptomsSummary": string or null,
          "items": [ {{ "name": string, "price": integer }} ],
          "totalAmount": integer
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
        
        # items 정규화
        safe_items = []
        for it in data.get("items", []):
            if isinstance(it, dict):
                safe_items.append({"name": str(it.get("name","")), "price": int(it.get("price") or 0)})
        data["items"] = safe_items

        return data

    except Exception:
        return None


# ------------------------------------------
# 5. AI HELPERS (Tagging)
# ------------------------------------------

def get_tags_definition_for_prompt() -> str:
    """
    Gemini에게 알려줄 태그 목록 문자열 생성
    포맷: - 코드 : 라벨 (키워드 예시)
    """
    if not CONDITION_TAGS:
        return "태그 정의 파일이 없습니다."

    lines = []
    lines.append("[가능한 질환/예방 태그 목록]")
    
    for code, config in CONDITION_TAGS.items():
        # 토큰 절약을 위해 키워드는 3개까지만
        keywords_str = ", ".join(config.keywords[:3])
        line = f"- {code}: {config.label} (관련어: {keywords_str})"
        lines.append(line)
        
    return "\n".join(lines)


# ------------------------------------------
# 6. DTO MODELS
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
    diagnosis: Optional[str] = None 

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
    # 🔥 AI가 선택한 태그 리스트
    condition_tags: List[str] = []


# ------------------------------------------
# 7. FASTAPI SETUP
# ------------------------------------------

app = FastAPI(title="PetHealth+ Server", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "PetHealth+ Server is Running"}

@app.get("/health")
def health():
    return {"status": "ok", "gemini_model": settings.GEMINI_MODEL_NAME}


# ------------------------------------------
# 8. ENDPOINTS
# ------------------------------------------

# (1) AI 종합 분석 (핵심 기능)
@app.post("/api/ai/analyze", response_model=AICareResponse)
async def analyze_pet_health(req: AICareRequest):
    if settings.GEMINI_ENABLED.lower() != "true" or not settings.GEMINI_API_KEY:
        return AICareResponse(
            summary="AI 설정이 필요해요.",
            detail_analysis="서버 환경변수 GEMINI_API_KEY를 확인해주세요.",
            weight_trend_status="-",
            risk_factors=[],
            action_guide=["서버 점검 필요"],
            health_score=0,
            condition_tags=[]
        )

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        # 태그 목록 텍스트 생성
        tags_context = get_tags_definition_for_prompt()

        prompt = f"""
        당신은 수의학 지식을 갖춘 'PetHealth+' AI 파트너입니다.
        데이터를 분석해 보호자에게 따뜻하고 정확한 조언을 주세요.

        [반려동물 정보]
        - {req.profile.name} ({req.profile.species}, {req.profile.age_text})
        - 체중: {req.profile.weight_current}kg
        - 알러지: {", ".join(req.profile.allergies) or "없음"}

        [건강 기록]
        - 최근 체중 변화: {req.recent_weights}
        - 최근 병원 방문: {req.medical_history}
        - 예정된 스케줄: {req.schedules}

        {tags_context}

        [분석 지시사항]
        1. **체중**: 0.1kg 단위 변화도 민감하게 체크하여 추세(증가/감소/유지)를 판단하세요.
        2. **태그 선택**: 위 태그 목록 중, 이 동물의 '현재 상태', '최근 치료', '예방 접종 이력'에 해당하는 코드(code)를 모두 고르세요.
           - 주의: "심장사상충 음성(정상)"인 경우 '심장사상충 질환(heart_heartworm)' 태그를 선택하지 마세요. "예방약 처방"인 경우 '예방(prevent_heartworm)' 태그를 선택하세요.
           - 광견병, 종합백신, 코로나 등 백신 종류를 정확히 구별해서 태그를 선택하세요.
        3. **액션**: 구체적인 행동 가이드를 2~3개 제안하세요.
        4. **점수**: 0~100점 사이 건강 점수.

        [출력 JSON]
        {{
            "summary": "40자 이내 홈 화면 요약 (친절하게)",
            "detail_analysis": "3~5문장의 상세 분석",
            "weight_trend_status": "체중 상태 요약",
            "risk_factors": ["위험요소1", "위험요소2"],
            "action_guide": ["행동가이드1", "행동가이드2"],
            "health_score": 85,
            "condition_tags": ["code1", "code2"]
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
            summary=data.get("summary", "분석 완료"),
            detail_analysis=data.get("detail_analysis", ""),
            weight_trend_status=data.get("weight_trend_status", "-"),
            risk_factors=data.get("risk_factors", []),
            action_guide=data.get("action_guide", []),
            health_score=data.get("health_score", 50),
            condition_tags=data.get("condition_tags", [])
        )

    except Exception as e:
        print(f"AI Analyze Error: {e}")
        return AICareResponse(
            summary="분석 중 오류가 발생했어요.",
            detail_analysis=f"Error: {str(e)}",
            weight_trend_status="-",
            risk_factors=[],
            action_guide=[],
            health_score=0,
            condition_tags=[]
        )


# (2) 영수증 업로드 (OCR)
@app.post("/api/receipt/analyze")
async def analyze_receipt_endpoint(
    petId: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    upload = file or image
    if not upload:
        raise HTTPException(400, "파일 누락")

    rec_id = str(uuid.uuid4())
    ext = os.path.splitext(upload.filename or "")[1] or ".jpg"
    key = f"receipts/{petId}/{rec_id}{ext}"

    data = await upload.read()
    
    # S3 Upload
    s3_url = upload_to_s3(io.BytesIO(data), key, upload.content_type or "image/jpeg")

    # Vision OCR
    ocr_text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as tmp:
            tmp.write(data)
            tmp.flush()
            ocr_text = run_vision_ocr(tmp.name)
    except Exception as e:
        print(f"OCR Error: {e}")

    # Parse
    parsed = parse_receipt_ai(ocr_text)
    if not parsed:
        fallback = parse_receipt_kor(ocr_text)
        items = [{"name": "항목", "price": fallback["totalAmount"]}] if fallback["totalAmount"] else []
        parsed = {
            "clinicName": fallback["hospitalName"],
            "visitDate": fallback["visitAt"],
            "diseaseName": None,
            "symptomsSummary": None,
            "items": items,
            "totalAmount": fallback["totalAmount"]
        }

    return {
        "petId": petId,
        "s3Url": s3_url,
        "parsed": parsed,
        "notes": ocr_text
    }


# (3) PDF 업로드 (검사/증명서)
@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(petId: str = Form(...), title: str = Form(...), memo: str = Form(None), file: UploadFile = File(...)):
    # 파일명 보존 로직 (제목으로 사용)
    original = os.path.splitext(file.filename or "")[0].strip() or "검사결과"
    safe = original.replace(" ", "_")
    # 파일명에 구분자(__)를 넣어 저장 -> 리스트 조회시 파싱
    key = f"lab/{petId}/{safe}__{uuid.uuid4()}.pdf"
    
    url = upload_to_s3(file.file, key, "application/pdf")
    return {"s3Url": url, "createdAt": datetime.now().isoformat(), "title": original}

@app.post("/api/cert/upload-pdf")
async def upload_cert_pdf(petId: str = Form(...), title: str = Form(...), memo: str = Form(None), file: UploadFile = File(...)):
    original = os.path.splitext(file.filename or "")[0].strip() or "증명서"
    safe = original.replace(" ", "_")
    key = f"cert/{petId}/{safe}__{uuid.uuid4()}.pdf"
    
    url = upload_to_s3(file.file, key, "application/pdf")
    return {"s3Url": url, "createdAt": datetime.now().isoformat(), "title": original}


# (4) 리스트 조회 (파일명 파싱 복원)
@app.get("/api/lab/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"
    res = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)
    items = []
    
    if "Contents" in res:
        for obj in res["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"): continue
            
            # Key: lab/petId/Filename__UUID.pdf
            fname = key.split("/")[-1]
            base, _ = os.path.splitext(fname)
            
            display_title = "검사결과"
            file_id = base
            
            # 구분자(__)가 있으면 제목과 UUID 분리
            if "__" in base:
                safe_name, file_id = base.rsplit("__", 1)
                display_title = safe_name.replace("_", " ")
            elif len(base) > 36: # 구분자 없는 Legacy 데이터 호환
                file_id = base
                display_title = "검사결과"
                
            dt_str = obj["LastModified"].strftime("%Y-%m-%d")
            url = s3_client.generate_presigned_url("get_object", Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key}, ExpiresIn=604800)
            
            items.append({
                "id": file_id,
                "petId": petId,
                "title": f"{display_title} ({dt_str})",
                "s3Url": url,
                "createdAt": obj["LastModified"].isoformat()
            })
    
    # 최신순 정렬
    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items

@app.get("/api/cert/list")
def get_cert_list(petId: str = Query(...)):
    prefix = f"cert/{petId}/"
    res = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)
    items = []
    
    if "Contents" in res:
        for obj in res["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"): continue
            
            fname = key.split("/")[-1]
            base, _ = os.path.splitext(fname)
            
            display_title = "증명서"
            file_id = base
            
            if "__" in base:
                safe_name, file_id = base.rsplit("__", 1)
                display_title = safe_name.replace("_", " ")
            
            dt_str = obj["LastModified"].strftime("%Y-%m-%d")
            url = s3_client.generate_presigned_url("get_object", Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key}, ExpiresIn=604800)
            
            items.append({
                "id": file_id,
                "petId": petId,
                "title": f"{display_title} ({dt_str})",
                "s3Url": url,
                "createdAt": obj["LastModified"].isoformat()
            })
            
    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items
