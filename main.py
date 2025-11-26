from __future__ import annotations

import os
import io
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.exceptions import NoCredentialsError
from pydantic_settings import BaseSettings

# condition_tags 가 없어도 서버 깨지지 않도록 방어
try:
    from condition_tags import CONDITION_TAGS, ConditionTagConfig  # noqa: F401
except ImportError:
    CONDITION_TAGS = []
    ConditionTagConfig = None

# Gemini (google-generativeai)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

import re


# ------------------------------------------
# SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    # AWS S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # Gemini
    GEMINI_ENABLED: str = "true"        # "true" / "false"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"

    # 선택: Vertex용 값이지만, 현재 코드는 직접 사용 안 함
    GCP_PROJECT_ID: str = ""
    GCP_LOCATION: str = "us-central1"

    # 로컬 테스트용 (외부 API 안 부르고 더미 응답)
    STUB_MODE: str = "false"


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
# Gemini 기반 OCR (이미지 → 텍스트)
# ------------------------------------------

def extract_receipt_text_ai(image_bytes: bytes) -> str:
    """
    Gemini 2.5 Flash-Lite 로 영수증 이미지에서 텍스트를 추출
    """
    if settings.STUB_MODE.lower() == "true":
        # 개발용: 실제 호출 안 하고 더미 리턴
        return ""

    if settings.GEMINI_ENABLED.lower() != "true":
        return ""
    if not settings.GEMINI_API_KEY:
        return ""
    if genai is None:
        return ""

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model_name = settings.GEMINI_MODEL_NAME or "gemini-2.5-flash-lite"
    model = genai.GenerativeModel(model_name)

    prompt = """
다음 이미지는 한국 동물병원 영수증입니다.
이미지 안에 보이는 텍스트를 가능한 한 그대로 줄 단위로 써 주세요.

•⁠  ⁠항목 순서는 영수증에 나온 순서대로 유지해 주세요.
•⁠  ⁠숫자, 쉼표, '원' 같은 표기도 그대로 적어 주세요.
•⁠  ⁠불필요한 설명은 쓰지 말고, 텍스트만 출력하세요.
"""

    try:
        resp = model.generate_content(
            [
                prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes,
                },
            ]
        )
        text = (resp.text or "").strip()
        return text
    except Exception as e:
        print("extract_receipt_text_ai error:", e)
        return ""


# ------------------------------------------
# 영수증 텍스트 파싱 보조: 병원명 추론 / 정규식 파서
# (Gemini 실패 시 fallback 용)
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
    한국 동물병원 영수증 텍스트를 구조화:
    - hospitalName
    - visitAt  (YYYY-MM-DD HH:MM 또는 None)
    - items    [{ name, amount }]
    - totalAmount
    """
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

    # 3) 금액 패턴: 끝에 오는 숫자 (30,000 / 81000 / ₩30,000 / 30,000원)
    amt_pattern = re.compile(
        r"(?:₩|￦)?\s*(\d{1,3}(?:,\d{3})|\d+)\s*(원)?\s*$"
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
# Gemini 기반 구조화 파싱 (텍스트 → DTO)
# ------------------------------------------

JUNK_ITEM_KEYWORDS = [
    "사업자", "등록번호", "전화", "tel", "fax", "팩스",
    "serial", "영수증", "영 수 증",
    "합계", "총액", "총 금액", "총금액", "합계금액", "소계",
    "부가세", "비과세", "과세공급가액",
    "결제요청", "요청금액", "결제일", "발행일",
    "카드", "현금", "승인", "매입", "가맹점", "전표번호",
]


def is_junk_item(name: Optional[str], price: Optional[int]) -> bool:
    """
    Gemini가 만들어 준 items 중에서
    영수증 메타데이터/쓰레기 줄은 걸러내기 위한 필터
    """
    if not name:
        return True

    text = name.strip()
    if not text:
        return True

    # '항목' 같은 의미 없는 이름은 버리기
    if text in ["항목", "상품", "비고", "내역"]:
        return True

    # 한글/영문이 하나도 없고 숫자/쉼표/원만 있으면 버리기 (예: '30,000')
    if re.fullmatch(r"[0-9,\s]+원?", text):
        return True

    # 메타데이터 키워드 포함하면 버리기
    for kw in JUNK_ITEM_KEYWORDS:
        if kw.replace(" ", "").lower() in text.replace(" ", "").lower():
            return True

    # 금액이 0 또는 음수인 경우 버리기
    if price is not None and price <= 0:
        return True

    return False


def parse_receipt_ai(raw_text: str) -> Optional[dict]:
    """
    Gemini로 영수증 텍스트를 파싱해서
    ReceiptParsedDTO 에 맞는 dict 리턴:
    {
      "clinicName": str | null,
      "visitDate": "YYYY-MM-DD" | null,
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

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model_name = settings.GEMINI_MODEL_NAME or "gemini-2.5-flash-lite"
    model = genai.GenerativeModel(model_name)

    prompt = f"""
너는 한국 '동물병원 영수증'만 다루는 전문가야.

아래 텍스트는 OCR로 읽은 영수증 전문이야. 이 텍스트를 분석해서
'진짜 진료 항목'만 골라서 JSON으로 정리해.

영수증 텍스트:

\"\"\"{raw_text}\"\"\"


### 1. clinicName
•⁠  ⁠'동물병원', '동물 메디컬', '동물클리닉' 등으로 끝나는 상호를 찾고,
•⁠  ⁠가장 병원명에 가까운 한 줄만 선택해.
•⁠  ⁠없으면 null.

### 2. visitDate
•⁠  ⁠실제 진료를 받은 날짜 1개만 골라.
•⁠  ⁠'발행일', '결제일' 말고 '진료일' 또는 문맥상 방문 날짜로 보이는 것.
•⁠  ⁠형식은 반드시 "YYYY-MM-DD" 로만 출력.

### 3. items
•⁠  ⁠아래 조건을 모두 만족하는 줄들만 항목으로 만든다:
  - 백신/약/검사/처방 등 의료 서비스나 약품명
    예: "DHPPi", "Corona", "Nexgard Spectra 7.5~15kg"
  - '사업자등록번호', '전화번호', '발행일', '결제요청', '합계', '총액', '부가세',
    '과세공급가액', '카드', '현금' 같은 메타 정보는 절대로 항목으로 넣지 않는다.
  - '항목', '상품', '비고', '내역' 같은 일반 단어만 있는 줄도 버린다.

•⁠  ⁠각 항목의 name:
  - 영수증 안에 적힌 설명을 그대로 사용한다.
  - 예: "DHPPi", "Corona", "Nexgard Spectra 7.5~15kg"

•⁠  ⁠각 항목의 price:
  - 한 줄에 '단가 30,000  수량 1  금액 30,000'처럼 쓰여 있으면
    '금액'에 해당하는 최종 금액(여기서는 30000)을 사용한다.
  - 숫자 형식: 쉼표 제거 후 정수. 예: 30,000원 → 30000
  - 금액을 확실히 알 수 없으면 null.

### 4. totalAmount
•⁠  ⁠'합계', '총액', '총 금액', '총금액'에 해당하는 값을 찾아서 정수로 넣어라.
•⁠  ⁠없다면 items 의 price 들을 합산해서 넣어라.
•⁠  ⁠예: 81,000원 → 81000

### 5. 출력 형식
반드시 아래 형식의 JSON "한 개"만, 설명 없이 출력해.

{{
  "clinicName": string or null,
  "visitDate": string or null,
  "diseaseName": null,
  "symptomsSummary": null,
  "items": [
    {{
      "name": string,
      "price": integer or null
    }}
  ],
  "totalAmount": integer or null
}}
"""

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()

        if "```" in text:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        data = json.loads(text)

        for key in ["clinicName", "visitDate", "items", "totalAmount"]:
            if key not in data:
                return None

        raw_items = data.get("items") or []
        cleaned_items: List[dict] = []

        for it in raw_items:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            price = it.get("price")
            if isinstance(price, str):
                try:
                    price = int(price.replace(",", ""))
                except Exception:
                    price = None

            if is_junk_item(name, price):
                continue

            cleaned_items.append({"name": name, "price": price})

        data["items"] = cleaned_items

        # totalAmount 재계산 (AI가 이상치 준 경우 대비)
        if (data.get("totalAmount") is None or data["totalAmount"] == 0) and cleaned_items:
            data["totalAmount"] = sum((p["price"] or 0) for p in cleaned_items)

        return data

    except Exception as e:
        print("parse_receipt_ai error:", e)
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
# 1) 영수증 업로드 + Gemini 분석
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
    영수증 이미지 업로드 + S3 저장 + Gemini OCR + Gemini 파싱
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

    # STUB_MODE 이면 여기서 바로 빈 결과 리턴
    if settings.STUB_MODE.lower() == "true":
        parsed_for_dto = {
            "clinicName": None,
            "visitDate": None,
            "diseaseName": None,
            "symptomsSummary": None,
            "items": [],
            "totalAmount": None,
        }
        return {
            "petId": petId,
            "s3Url": file_url,
            "parsed": parsed_for_dto,
            "notes": "",
        }

    # 2) Gemini로 텍스트 추출
    ocr_text = extract_receipt_text_ai(data)

    # 3) Gemini AI 파싱 시도 → 실패하면 정규식 fallback
    ai_parsed = parse_receipt_ai(ocr_text) if ocr_text else None
    if ai_parsed:
        parsed_for_dto = ai_parsed
    else:
        fallback = parse_receipt_kor(ocr_text) if ocr_text else {
            "hospitalName": "",
            "visitAt": None,
            "items": [],
            "totalAmount": 0,
        }

        visit_at = fallback.get("visitAt")
        visit_date: Optional[str] = None
        if visit_at:
            visit_date = visit_at  # "YYYY-MM-DD HH:MM" 그대로

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
    title: str = Form("검사결과"),   # iOS에서 파일명 보내지만, 서버 쪽에서 다시 계산
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    lab_id = str(uuid.uuid4())

    # 원본 파일명(확장자 제거)
    original_base = os.path.splitext(file.filename or "")[0].strip() or "검사결과"

    # S3 key 에도 파일명을 일부 포함 (나중에 리스트에서 꺼내 쓰기 위함)
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
#    - 제목: "원본파일명 (YYYY-MM-DD)"
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
