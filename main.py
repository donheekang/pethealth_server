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
# Google Vision (Legacy Support)
from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
# 이미지 처리를 위한 라이브러리 (Gemini Vision용) - pip install Pillow 필요
from PIL import Image

# 🔥 새로 만든 태그 파일 Import (없을 경우 대비하여 예외처리)
try:
    from condition_tags import CONDITION_TAGS
except ImportError:
    CONDITION_TAGS = {}
    print("Warning: condition_tags.py not found. AI tagging will be limited.")

# Gemini (google-generativeai) Import
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

    # Google Vision OCR
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Gemini 사용 여부 + API Key
    GEMINI_ENABLED: str = "false"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash" # 기본값

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
            return None # 이미지 파일이 아님

        prompt = """
        이 이미지는 한국 동물병원 영수증이야.
        이미지를 분석해서 다음 정보를 정확한 JSON으로 추출해줘.
        
        1. 병원명 (clinicName): 상단에 있는 병원 이름
        2. 방문일자 (visitDate): 날짜 (YYYY-MM-DD 형식). 시간은 제외.
        3. 진료항목 (items): 품목명(name)과 금액(price). 
           - '합계', '부가세', '총액', '카드', '현금' 같은 결제 정보는 제외하고 순수 진료 항목만 추출해.
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

        # 이미지와 프롬프트를 함께 전송
        response = model.generate_content([prompt, img])
        text = response.text.strip()

        # Markdown 처리
        if "⁠  " in text:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end+1]

        data = json.loads(text)
        
        # 데이터 정제 (안전장치)
        safe_items = []
        if "items" in data and isinstance(data["items"], list):
            for it in data["items"]:
                if isinstance(it, dict):
                    # 금액 정제
                    price_val = str(it.get("price", "0")).replace(",", "").replace("원", "").strip()
                    try:
                        final_price = int(float(price_val)) # 1000.0 같은 경우 대비
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
# 4. GOOGLE VISION OCR (Legacy Support)
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
# 5. 영수증 파싱 로직 (기존 코드 유지)
# ------------------------------------------

def guess_hospital_name(lines: List[str]) -> str:
    """
    병원명 추론: 키워드 + 위치 + 형태 기반으로 대략 고르기
    """
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
    한국 동물병원 영수증 OCR 텍스트를 구조화 (정규식 Fallback)
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1) 병원명
    hospital_name = guess_hospital_name(lines)

    # 2) 날짜/시간
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
    
    # 시간 없는 날짜 패턴 추가 (보완)
    if not visit_at:
        dt_pattern_short = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})")
        for line in lines:
            m = dt_pattern_short.search(line)
            if m:
                y, mo, d = map(int, m.groups())
                visit_at = datetime(y, mo, d).strftime("%Y-%m-%d")
                break

    # 3) 금액 패턴
    amt_pattern = re.compile(
        r"(?:₩|￦)?\s*(\d{1,3}(?:,\d{3})|\d+)\s(원)?\s*$"
    )

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

        # 합계/총액 줄은 total 후보
        if any(k in lowered for k in ["합계", "총액", "총금액", "합계금액", "결제금액"]):
            candidate_totals.append(amount)
            continue

        if not name:
            name = "항목"

        items.append({"name": name, "amount": amount})

    # 4) totalAmount 결정
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
    Gemini를 이용한 영수증 AI 파싱 (텍스트 기반 - 백업용)
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
        """

        resp = model.generate_content(prompt)
        text = resp.text.strip()

        # Markdown json 태그 제거
        if "  ⁠" in text:
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
                fixed_items.append({"name": str(name), "price": int(price) if price else 0})
        data["items"] = fixed_items

        return data

    except Exception:
        return None


# ------------------------------------------
# 6. AI Care (태그 헬퍼 & DTO) - 🔥 새로 추가된 부분
# ------------------------------------------

def get_tags_definition_for_prompt() -> str:
    """Gemini 프롬프트용 태그 목록 생성"""
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


# DTO Models for AI Analysis
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
# 7. FASTAPI APP SETUP
# ------------------------------------------

app = FastAPI(title="PetHealth+ Server", version="2.0.0")

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
    return {"status": "ok", "gemini_model": settings.GEMINI_MODEL_NAME}


# ------------------------------------------
# 8. ENDPOINTS
# ------------------------------------------

# (1) 영수증 업로드 & 분석 (Gemini Vision 적용 - 개선된 로직)
@app.post("/receipt/upload")
@app.post("/receipts/upload")
@app.post("/api/receipt/upload")
@app.post("/api/receipts/upload")
@app.post("/api/receipt/analyze")   # iOS에서 쓰는 엔드포인트
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

    # 🔥 1순위: Gemini Vision으로 이미지 직접 분석 (OCR 불량 해결)
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
            
            # Text AI 시도
            parsed_data = parse_receipt_ai(ocr_text)
            
            # Text AI도 실패하면 정규식
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
        "s3Url": file_url,
        "parsed": parsed_data,
        "notes": notes
    }


# (2) PDF 업로드 (검사/증명서) - 기존 로직 유지 (파일명 처리)
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
    # 파일명을 key에 포함 (리스트 조회 시 복원용) -> 구분자 __ 사용
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")
    key = f"lab/{petId}/{safe_base}__{uuid.uuid4()}.pdf"

    url = upload_to_s3(file.file, key, "application/pdf")
    created_at_iso = datetime.utcnow().isoformat()

    return {
        "id": key.split("/")[-1], # ID는 파일명으로 대체 가능
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
    key = f"cert/{petId}/{safe_base}__{uuid.uuid4()}.pdf"

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


# (3) 리스트 조회 - 기존 로직 복원 (파일명 파싱)
@app.get("/lab/list")
@app.get("/labs/list")
@app.get("/api/lab/list")
@app.get("/api/labs/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"
    response = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)

    items = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"): continue

            # Key format: lab/petId/Filename__UUID.pdf
            filename = key.split("/")[-1]
            base_name, _ = os.path.splitext(filename)

            display_title = "검사결과"
            file_id = base_name

            # 구분자(__)가 있으면 제목과 UUID 분리
            if "__" in base_name:
                safe_name, file_id = base_name.rsplit("__", 1)
                display_title = safe_name.replace("_", " ")
            elif len(base_name) > 36: # 구분자 없는 Legacy 데이터 호환
                file_id = base_name
                # 기존 레거시는 제목 복원이 어려우니 기본값 사용

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=604800,
            )

            items.append({
                "id": file_id,
                "petId": petId,
                "title": f"{display_title} ({date_str})",
                "s3Url": url,
                "createdAt": created_at_iso,
            })
    
    # 최신순 정렬
    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


@app.get("/cert/list")
@app.get("/certs/list")
@app.get("/api/cert/list")
@app.get("/api/certs/list")
def get_cert_list(petId: str = Query(...)):
    prefix = f"cert/{petId}/"
    response = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)

    items = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"): continue

            filename = key.split("/")[-1]
            base_name, _ = os.path.splitext(filename)

            display_title = "증명서"
            file_id = base_name

            if "__" in base_name:
                safe_name, file_id = base_name.rsplit("__", 1)
                display_title = safe_name.replace("_", " ")

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=604800,
            )

            items.append({
                "id": file_id,
                "petId": petId,
                "title": f"{display_title} ({date_str})",
                "s3Url": url,
                "createdAt": created_at_iso,
            })

    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


# (4) AI 종합 분석 (🔥 새로 추가된 핵심 기능)
@app.post("/api/ai/analyze", response_model=AICareResponse)
async def analyze_pet_health(req: AICareRequest):
    """
    PetHealth+ AI 케어: 종합 건강 리포트 생성
    """
    # 1. Gemini 비활성화 시 Fallback
    if settings.GEMINI_ENABLED.lower() != "true" or not settings.GEMINI_API_KEY:
        return AICareResponse(
            summary="AI 설정이 필요해요.",
            detail_analysis="서버 환경변수 GEMINI_API_KEY를 확인해주세요.",
            weight_trend_status="데이터 없음",
            risk_factors=[],
            action_guide=["서버 점검 필요"],
            health_score=0,
            condition_tags=[]
        )

    # 2. Gemini 호출
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        # 태그 목록 텍스트 생성
        tags_context = get_tags_definition_for_prompt()

        # 프롬프트 구성
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

        {tags_context}

        [분석 요청사항]
        1. 체중: 최근 변화 추세(증가/감소/유지)를 0.1kg 단위로 민감하게 체크하세요.
        2. 리스크: 노령견/묘 여부, 체중 급변, 빈번한 병원 방문 등을 고려해 위험 요소를 찾으세요.
        3. 액션: 구체적이고 실천 가능한 행동을 제안하세요. (예: "간식 줄이기", "관절 영양제 고려")
        4. 태그: 위 태그 목록 중, 이 동물의 '현재 상태', '최근 치료', '예방 접종 이력'에 해당하는 코드(code)를 모두 고르세요. 
           - "음성(Negative)"이거나 "정상"인 질환은 절대 선택하지 마세요.
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
        """

        resp = model.generate_content(prompt)
        text = resp.text.strip()
        
        if "```" in text:
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
            health_score=data.get("health_score", 50),
            condition_tags=data.get("condition_tags", [])
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
            condition_tags=[]
        )
