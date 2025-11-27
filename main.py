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
# 기존 Google Vision (백업용)
from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
# 🔥 이미지 처리를 위한 라이브러리 (추가됨)
from PIL import Image

# 🔥 새로 만든 태그 파일 Import
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
# 1. SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    GEMINI_ENABLED: str = "true"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 Upload Error: {str(e)}")


# ------------------------------------------
# 3. GEMINI VISION (이미지 직접 분석 - NEW)
# ------------------------------------------

def analyze_receipt_image_with_gemini(image_bytes: bytes) -> Optional[dict]:
    """
    이미지 바이너리를 Gemini에게 직접 보내서 영수증 정보를 추출
    (OCR 텍스트 추출 과정을 건너뛰고 이미지 자체를 이해함)
    """
    if settings.GEMINI_ENABLED.lower() != "true" or not settings.GEMINI_API_KEY:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        # Bytes -> PIL Image 변환
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return None 

        prompt = """
        이 이미지는 한국 동물병원 영수증이야.
        이미지를 분석해서 다음 정보를 정확한 JSON으로 추출해줘.
        
        1. 병원명 (clinicName): 상단에 있는 병원 이름
        2. 방문일자 (visitDate): 날짜 (YYYY-MM-DD 형식). 시간은 제외.
        3. 진료항목 (items): 품목명(name)과 금액(price). 
           - '합계', '부가세', '총액', '카드', '현금', '면세' 같은 결제 정보 줄은 제외해.
           - 순수 진료/처방 항목만 추출해.
           - 금액에 '원'이나 콤마(,)는 제거하고 정수형(Integer)으로 줘.
        4. 총결제금액 (totalAmount): 최종 합계 금액.

        [출력 JSON 형식]
        {
          "clinicName": "OO동물병원",
          "visitDate": "2023-10-25",
          "items": [
            {"name": "초진 진찰료", "price": 5000},
            {"name": "종합백신", "price": 25000}
          ],
          "totalAmount": 30000
        }
        
        오직 JSON만 출력해. 마크다운이나 설명은 쓰지 마.
        """

        response = model.generate_content([prompt, img])
        text = response.text.strip()

        if "⁠  " in text:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end+1]

        data = json.loads(text)
        
        # 데이터 정제
        safe_items = []
        if "items" in data and isinstance(data["items"], list):
            for it in data["items"]:
                if isinstance(it, dict):
                    price_val = str(it.get("price", "0")).replace(",", "").replace("원", "").strip()
                    try:
                        final_price = int(float(price_val))
                    except:
                        final_price = 0
                        
                    safe_items.append({
                        "name": str(it.get("name", "항목")),
                        "price": final_price
                    })
        data["items"] = safe_items
        
        return data

    except Exception as e:
        print(f"Gemini Vision Error: {e}")
        return None


# ------------------------------------------
# 4. GOOGLE VISION OCR (Legacy)
# ------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS missing")
    try:
        info = json.loads(cred_value)
        return vision.ImageAnnotatorClient.from_service_account_info(info)
    except json.JSONDecodeError:
        if not os.path.exists(cred_value):
            raise Exception(f"Credential file not found: {cred_value}")
        return vision.ImageAnnotatorClient.from_service_account_file(cred_value)
    except Exception as e:
        raise Exception(f"Vision Client Error: {e}")

def run_vision_ocr(image_path: str) -> str:
    try:
        client = get_vision_client()
        with open(image_path, "rb") as f:
            content = f.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        return texts[0].description if texts else ""
    except Exception:
        return ""


# ------------------------------------------
# 5. 기존 파서 (Regex + Text AI) - 백업용 (보내주신 코드 복원)
# ------------------------------------------

def guess_hospital_name(lines: List[str]) -> str:
    keywords = [
        "동물병원", "동물 병원", "동물의료", "동물메디컬", "동물 메디컬",
        "동물클리닉", "동물 클리닉",
        "애견병원", "애완동물병원", "펫병원", "펫 병원",
        "종합동물병원", "동물의원", "동물병의원"
    ]
    best_line = None
    best_score = -1

    for idx, line in enumerate(lines):
        score = 0
        text = line.replace(" ", "")
        if any(k in text for k in keywords): score += 5
        if idx <= 4: score += 2
        if any(x in line for x in ["TEL", "전화", "FAX", "팩스", "도로명"]): score -= 2
        digit_count = sum(c.isdigit() for c in line)
        if digit_count >= 8: score -= 1
        if len(line) < 2 or len(line) > 25: score -= 1

        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None and lines: return lines[0]
    return best_line or ""

def parse_receipt_kor(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hospital_name = guess_hospital_name(lines)
    
    visit_at = None
    dt_pattern = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2}).*?(\d{1,2}):(\d{2})")
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
    
    amt_pattern = re.compile(r"(?:₩|￦)?\s*(\d{1,3}(?:,\d{3})|\d+)\s(원)?\s*$")
    items = []
    candidate_totals = []

    for line in lines:
        m = amt_pattern.search(line)
        if not m: continue
        amount_str = m.group(1).replace(",", "")
        try:
            amount = int(amount_str)
        except ValueError: continue
        
        name = line[:m.start()].strip()
        lowered = name.replace(" ", "")
        
        if any(k in lowered for k in ["합계", "총액", "총금액", "합계금액", "결제금액"]):
            candidate_totals.append(amount)
            continue
        
        if not name: name = "항목"
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
    # 기존 Text 기반 Gemini 파서 (백업용)
    if settings.GEMINI_ENABLED.lower() != "true" or not settings.GEMINI_API_KEY:
        return None
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        prompt = f"""
        한국 동물병원 영수증 OCR 텍스트를 JSON으로 변환해줘.
        [OCR] {raw_text}
        [Format] {{ "clinicName": str, "visitDate": "YYYY-MM-DD", "items": [{{"name": str, "price": int}}], "totalAmount": int }}
        """
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        if "  ⁠" in text:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1: text = text[start:end+1]
        
        data = json.loads(text)
        # items 안전장치
        safe_items = []
        for it in data.get("items", []):
            if isinstance(it, dict):
                safe_items.append({"name": str(it.get("name","")), "price": int(it.get("price") or 0)})
        data["items"] = safe_items
        return data
    except:
        return None


# ------------------------------------------
# 6. AI HELPERS (Tagging) - NEW
# ------------------------------------------

def get_tags_definition_for_prompt() -> str:
    if not CONDITION_TAGS:
        return "태그 정의 파일이 없습니다."
    lines = ["[가능한 질환/예방 태그 목록]"]
    for code, config in CONDITION_TAGS.items():
        keywords_str = ", ".join(config.keywords[:3])
        line = f"- {code}: {config.label} (관련어: {keywords_str})"
        lines.append(line)
    return "\n".join(lines)


# ------------------------------------------
# 7. DTO MODELS
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
    condition_tags: List[str] = []


# ------------------------------------------
# 8. FASTAPI APP SETUP
# ------------------------------------------

app = FastAPI(title="PetHealth+ Server", version="2.1.0")

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
def health():
    return {"status": "ok", "gemini_model": settings.GEMINI_MODEL_NAME}


# ------------------------------------------
# 9. ENDPOINTS
# ------------------------------------------

# (1) AI 종합 분석 (헬스케어 리포트) - NEW
@app.post("/api/ai/analyze", response_model=AICareResponse)
async def analyze_pet_health(req: AICareRequest):
    if settings.GEMINI_ENABLED.lower() != "true" or not settings.GEMINI_API_KEY:
        return AICareResponse(
            summary="AI 설정이 필요해요.", detail_analysis="API KEY 확인 필요",
            weight_trend_status="-", risk_factors=[], action_guide=[], health_score=0, condition_tags=[]
        )

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        tags_context = get_tags_definition_for_prompt()

        prompt = f"""
        당신은 'PetHealth+' 앱의 수의학 AI 파트너입니다.
        데이터를 분석해 보호자에게 따뜻하고 정확한 조언을 주세요.

        [반려동물] {req.profile.name} ({req.profile.species}, {req.profile.age_text})
        [체중] {req.profile.weight_current}kg
        [기록]
        - 체중 변화: {req.recent_weights}
        - 병원 방문: {req.medical_history}
        - 스케줄: {req.schedules}

        {tags_context}

        [지시사항]
        1. 체중: 0.1kg 단위 변화도 민감하게 체크하세요.
        2. 태그: 위 목록에서 '현재 상태', '최근 치료', '예방 접종'에 해당하는 코드를 고르세요. ("음성" 제외)
        3. 액션: 구체적인 행동 제안 2~3개.
        4. 점수: 0~100점.

        [출력 JSON]
        {{
            "summary": "40자 이내 요약",
            "detail_analysis": "상세 분석 3~5문장",
            "weight_trend_status": "체중 상태",
            "risk_factors": ["위험1", "위험2"],
            "action_guide": ["행동1", "행동2"],
            "health_score": 85,
            "condition_tags": ["code1"]
        }}
        """

        resp = model.generate_content(prompt)
        text = resp.text.strip()
        if "```" in text:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1: text = text[start:end+1]

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
        print(f"AI Analysis Error: {e}")
        return AICareResponse(
            summary="오류가 발생했어요.", detail_analysis=str(e),
            weight_trend_status="-", risk_factors=[], action_guide=[], health_score=0, condition_tags=[]
        )


# (2) 영수증 분석 (Vision 우선 -> Fallback) - IMPROVED
@app.post("/api/receipt/analyze")
async def analyze_receipt_endpoint(
    petId: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    upload = file or image
    if not upload: raise HTTPException(400, "No file provided")

    # S3 키 생성
    rec_id = str(uuid.uuid4())
    ext = os.path.splitext(upload.filename or "")[1] or ".jpg"
    key = f"receipts/{petId}/{rec_id}{ext}"

    # 파일 읽기 (Bytes)
    data = await upload.read()
    
    # S3 Upload
    s3_url = upload_to_s3(io.BytesIO(data), key, upload.content_type or "image/jpeg")

    # 🔥 1순위: Gemini Vision으로 이미지 직접 분석
    parsed_data = analyze_receipt_image_with_gemini(data)
    
    notes = "AI Vision 분석 완료"

    # 🔥 2순위: Vision 실패 시 기존 OCR + Text AI 방식 (백업)
    if not parsed_data:
        print("Vision failed, fallback to OCR")
        try:
            with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as tmp:
                tmp.write(data)
                tmp.flush()
                ocr_text = run_vision_ocr(tmp.name)
            
            parsed_data = parse_receipt_text_ai(ocr_text)
            
            # 🔥 3순위: AI Text도 실패 시 정규식 (백업의 백업)
            if not parsed_data:
                fallback = parse_receipt_kor(ocr_text)
                items = [{"name": "항목", "price": fallback["totalAmount"]}] if fallback["totalAmount"] else []
                parsed_data = {
                    "clinicName": fallback["hospitalName"],
                    "visitDate": fallback["visitAt"],
                    "diseaseName": None,
                    "symptomsSummary": None,
                    "items": items,
                    "totalAmount": fallback["totalAmount"]
                }
            notes = "OCR 분석 (Vision 실패)"
        except Exception as e:
            print(f"Fallback Error: {e}")
            parsed_data = {
                "clinicName": "", "visitDate": "", "items": [], "totalAmount": 0
            }
            notes = "분석 실패"

    return {
        "petId": petId,
        "s3Url": s3_url,
        "parsed": parsed_data,
        "notes": notes
    }


# (3) PDF 업로드 (검사/증명서) - 기존 로직 유지
@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(petId: str = Form(...), title: str = Form(...), memo: str = Form(None), file: UploadFile = File(...)):
    original = os.path.splitext(file.filename or "")[0].strip() or "검사결과"
    safe = original.replace(" ", "_")
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


# (4) 리스트 조회 (파일명 파싱 지원) - 기존 로직 유지
@app.get("/api/lab/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"
    res = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)
    items = []
    if "Contents" in res:
        for obj in res["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"): continue
            fname = key.split("/")[-1]
            base, _ = os.path.splitext(fname)
            
            display = "검사결과"
            fid = base
            if "__" in base:
                safe, fid = base.rsplit("__", 1)
                display = safe.replace("_", " ")
            elif len(base) > 36: # Legacy
                fid = base
            
            dt = obj["LastModified"].strftime("%Y-%m-%d")
            url = s3_client.generate_presigned_url("get_object", Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key}, ExpiresIn=604800)
            items.append({"id": fid, "petId": petId, "title": f"{display} ({dt})", "s3Url": url, "createdAt": obj["LastModified"].isoformat()})
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
            display = "증명서"
            fid = base
            if "__" in base:
                safe, fid = base.rsplit("__", 1)
                display = safe.replace("_", " ")
            
            dt = obj["LastModified"].strftime("%Y-%m-%d")
            url = s3_client.generate_presigned_url("get_object", Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key}, ExpiresIn=604800)
            items.append({"id": fid, "petId": petId, "title": f"{display} ({dt})", "s3Url": url, "createdAt": obj["LastModified"].isoformat()})
    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items
