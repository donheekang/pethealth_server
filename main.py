from __future__ import annotations

import os
import io
import json
import uuid
import tempfile
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware

# Vision OCR (선택적으로 사용)
try:
    from google.cloud import vision  # google-cloud-vision
except ImportError:
    vision = None

# (나중에 증상 태그 등 쓸 때를 대비해서 남겨둠)
try:
    from condition_tags import CONDITION_TAGS, ConditionTagConfig  # noqa: F401
except ImportError:
    CONDITION_TAGS = []
    ConditionTagConfig = object

import boto3
from botocore.exceptions import NoCredentialsError
from pydantic_settings import BaseSettings

# Gemini (google-generativeai)
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ------------------------------------------
# SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # Vision OCR용 서비스 계정 JSON 내용 또는 파일 경로
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Gemini 사용 여부 + API Key
    GEMINI_ENABLED: str = "false"  # "true" 이면 사용
    GEMINI_API_KEY: str = ""

    # 개발용 스텁
    STUB_MODE: str = "false"

    class Config:
        env_file = ".env"
        extra = "ignore"   # GCP_PROJECT_ID 같은 추가 env 있어도 무시


settings = Settings()


# ------------------------------------------
# AWS S3 CLIENT
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
# GOOGLE VISION OCR (옵션)
# ------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    """
    GOOGLE_APPLICATION_CREDENTIALS:
      - 서비스 계정 JSON '내용'일 수도 있고
      - JSON 파일 경로일 수도 있음
    둘 다 지원
    """
    if vision is None:
        raise RuntimeError("VISION_DISABLED")

    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        # 환경변수 비어 있으면 Vision 안 쓰고 건너뜀
        raise RuntimeError("VISION_DISABLED")

    # 1) JSON 내용 시도
    try:
        info = json.loads(cred_value)
        client = vision.ImageAnnotatorClient.from_service_account_info(info)
        return client
    except json.JSONDecodeError:
        # 2) JSON이 아니면 경로로 간주
        if not os.path.exists(cred_value):
            raise RuntimeError(
                "VISION_DISABLED"
            )
        client = vision.ImageAnnotatorClient.from_service_account_file(cred_value)
        return client
    except Exception as e:
        raise RuntimeError(f"OCR 클라이언트 생성 실패: {e}")


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
        raise RuntimeError(f"OCR 에러: {response.error.message}")

    texts = response.text_annotations
    if not texts:
        return ""

    return texts[0].description


# ------------------------------------------
# 영수증 OCR 결과 파싱 (정규식 Fallback)
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
        digit_count = sum(c.isdigit() for c in line)
        if digit_count >= 8:
            score -= 1

        # 4) 길이 너무 짧거나 너무 길면 감점
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
    한국 동물병원 영수증 OCR 텍스트를 구조화:
    - hospitalName
    - visitAt  (YYYY-MM-DD HH:MM 또는 None)
    - items    [{ name, amount }]
    - totalAmount
    """
    import re

    # 공백 줄 제거
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1) 병원명
    hospital_name = guess_hospital_name(lines)

    # 2) 날짜/시간: 2025.11.20 12:51, 2025-11-20 12:51, 2025년 11월 20일 12:51 등
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

    # 3) 금액 패턴: 끝에 오는 숫자 (30,000 / 81000 / ￦30,000 / 30,000원)
    amt_pattern = re.compile(
        r"(?:₩|￦)?\s*(\d{1,3}(?:,\d{3})|\d+)\s*(?:원)?\s*$"
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
        if any(k in lowered for k in ["합계", "총액", "총금액", "합계금액"]):
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


# ------------------------------------------
# Gemini 2.5 Flash-Lite 로 AI 파싱
# ------------------------------------------

def parse_receipt_ai(raw_text: str) -> Optional[dict]:
    """
    Gemini로 영수증 텍스트를 파싱해서
    ReceiptParsedDTO 에 맞는 dict 리턴:
    {
      "clinicName": str | null,
      "visitDate": "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" | null,
      "diseaseName": str | null,
      "symptomsSummary": str | null,
      "items": [ { "name": str, "price": int | null }, ... ],
      "totalAmount": int | null
    }
    실패하면 None 리턴 (fallback 은 정규식 파서)
    """
    if settings.GEMINI_ENABLED.lower() != "true":
        return None
    if not settings.GEMINI_API_KEY:
        return None
    if genai is None:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Google AI Studio 키를 쓴다는 전제 → 모델명만 사용
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        prompt = f"""
너는 한국 동물병원 영수증을 구조화된 JSON으로 정리하는 어시스턴트야.

다음은 OCR로 읽은 영수증 원문이야:

\"\"\"{raw_text}\"\"\"


요구사항:

1.⁠ ⁠병원명(clinicName)
   - '동물병원', '동물 병원' 같은 너무 일반적인 이름만 있으면
     가능한 한 전체 상호명을 그대로 써줘.
     예) '해랑동물병원', '행복한우리동물의료센터' 등.
   - 진짜 상호명이 보이지 않으면 null 로 둬도 된다.

2.⁠ ⁠visitDate
   - "YYYY-MM-DD" 형식 또는 "YYYY-MM-DD HH:MM" 형식 중 하나로.
   - 영수증에 날짜·시간이 여러 개 있으면
     "진료일/방문일" 의미에 가장 가까운 것을 선택해.
   - 아무 정보가 없으면 null.

3.⁠ ⁠items
   - 실제 진료/약/예방접종 등 과금된 항목을 한 줄씩 넣어줘.
   - 예: 종합백신, 코로나, 넥스가드, 처치료 등.
   - 영수증의 메타 정보(사업자번호, 전화번호, 합계, 부가세 등)는
     items 에 넣지 마.
   - price 는 정수(원 단위)로, 금액이 없으면 null.

4.⁠ ⁠totalAmount
   - '합계', '총액', '결제요청' 등으로 보이는 최종 결제 금액.
   - 찾지 못하면 null.

5.⁠ ⁠diseaseName / symptomsSummary
   - 영수증에 병명이나 증상 요약이 보이면 간단히 채우고,
   - 없으면 null 로 둬.

반드시 아래 형식의 JSON "한 개"만, 추가 설명 없이 순수 JSON으로만 출력해.

형식:
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
        text = resp.text.strip()

        # ⁠ json ...  ⁠ 같은 마크다운 감싸기 제거
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        data = json.loads(text)

        # 최소 키 검증
        for key in ["clinicName", "visitDate", "items", "totalAmount"]:
            if key not in data:
                return None

        # items 정규화
        if not isinstance(data.get("items"), list):
            data["items"] = []

        fixed_items = []
        for it in data["items"]:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            price = it.get("price")
            fixed_items.append({"name": name, "price": price})
        data["items"] = fixed_items

        return data

    except Exception:
        # Gemini 에러 나면 조용히 Fallback
        return None


# ------------------------------------------
# FASTAPI APP
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
    return {"status": "ok", "message": "PetHealth+ Server Running"}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ------------------------------------------
# 1) 진료기록 OCR (영수증 업로드)
# ------------------------------------------

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
    """
    영수증 이미지 업로드 + Vision OCR + (옵션) Gemini AI 분석
    S3 key: receipts/{petId}/{id}.jpg
    iOS가 file 이나 image 어떤 이름으로 보내도 처리.
    """
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

    # 2) OCR 실행 (Vision 없으면 그냥 빈 문자열)
    ocr_text = ""
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)

        try:
            ocr_text = run_vision_ocr(tmp_path)
        except RuntimeError:
            # Vision 비활성화/에러 → OCR 텍스트 없이 Gemini에게 바로 이미지 안 넘기고,
            # (현재 구조상 텍스트만 쓰므로, 이 경우엔 AI 파싱도 스킵되고 fallback 으로 감)
            ocr_text = ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3) AI 파싱 시도 → 실패하면 정규식 fallback
    ai_parsed = parse_receipt_ai(ocr_text) if ocr_text else None
    if ai_parsed:
        parsed_for_dto = ai_parsed
    else:
        # OCR 텍스트가 없거나, Gemini 파싱 실패하면 정규식 파서
        fallback = parse_receipt_kor(ocr_text) if ocr_text else {
            "hospitalName": "",
            "visitAt": None,
            "items": [],
            "totalAmount": 0,
        }

        visit_at = fallback.get("visitAt")
        visit_date: Optional[str] = None
        if visit_at:
            # "YYYY-MM-DD HH:MM" 그대로 사용 (iOS에서 길이 보고 처리)
            visit_date = visit_at

        dto_items: List[Dict] = []
        for it in fallback.get("items", []):
            dto_items.append(
                {
                    "name": it.get("name"),
                    "price": it.get("amount"),
                }
            )

        parsed_for_dto = {
            "clinicName": fallback.get("hospitalName"),
            "visitDate": visit_date,
            "diseaseName": None,
            "symptomsSummary": None,
            "items": dto_items,
            "totalAmount": fallback.get("totalAmount"),
        }

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,
    }


# ------------------------------------------
# 2) 검사결과 PDF 업로드
#    - iOS: POST /api/lab/upload-pdf
#    - 응답: PdfRecord 1개 (키: s3Url)
#    - 제목: "원본파일명 (YYYY-MM-DD)"
# ------------------------------------------

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

    # 원본 파일명(확장자 제거)
    original_base = os.path.splitext(file.filename or "")[0].strip() or "검사결과"

    # S3 key 에도 파일명을 일부 포함 (safe_base__uuid 형식)
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")
    key = f"lab/{petId}/{safe_base}__{lab_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")

    created_at_dt = datetime.utcnow()
    created_at_iso = created_at_dt.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = created_at_dt.strftime("%Y-%m-%d")

    display_title = f"{original_base} ({date_str})"

    return {
        "id": lab_id,
        "petId": petId,
        "title": display_title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at_iso,
    }


# ------------------------------------------
# 3) 증명서 PDF 업로드
#    - iOS: POST /api/cert/upload-pdf
# ------------------------------------------

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

    created_at_dt = datetime.utcnow()
    created_at_iso = created_at_dt.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = created_at_dt.strftime("%Y-%m-%d")

    display_title = f"{original_base} ({date_str})"

    return {
        "id": cert_id,
        "petId": petId,
        "title": display_title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at_iso,
    }


# ------------------------------------------
# 4) 검사결과 리스트
#    - iOS: GET /api/labs/list?petId=...
# ------------------------------------------

@app.get("/lab/list")
@app.get("/labs/list")
@app.get("/api/lab/list")
@app.get("/api/labs/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"

    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=prefix,
    )

    items: List[Dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]  # lab/{petId}/{safe_base}__{id}.pdf 또는 예전 {id}.pdf
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            name_no_ext = os.path.splitext(filename)[0]

            base_title = "검사결과"
            file_id = name_no_ext

            # 새 패턴: safe_base__uuid
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

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": display_title,
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at_iso,
                }
            )

    return items


# ------------------------------------------
# 5) 증명서 리스트
#    - iOS: GET /api/certs/list?petId=...
# ------------------------------------------

@app.get("/cert/list")
@app.get("/certs/list")
@app.get("/api/cert/list")
@app.get("/api/certs/list")
def get_cert_list(petId: str = Query(...)):
    prefix = f"cert/{petId}/"

    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=prefix,
    )

    items: List[Dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]  # cert/{petId}/{safe_base}__{id}.pdf 또는 예전 {id}.pdf
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

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": display_title,
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at_iso,
                }
            )

    return items
