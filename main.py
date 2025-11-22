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
from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError
from pydantic_settings import BaseSettings

# Gemini (google-generativeai) - 설치 안 되어 있어도 서버가 죽지 않게 try/except
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ------------------------------------------
# SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # 서비스 계정 JSON 내용 또는 JSON 파일 경로 (Vision OCR)
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Gemini 사용 여부 + API Key
    GEMINI_ENABLED: str = "false"
    GEMINI_API_KEY: str = ""

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
# GOOGLE VISION OCR
# ------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    """
    GOOGLE_APPLICATION_CREDENTIALS:
      - 서비스 계정 JSON '내용'일 수도 있고
      - JSON 파일 경로일 수도 있음
    둘 다 지원
    """
    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS 환경변수가 비어있습니다.")

    # 1) JSON 내용 시도
    try:
        info = json.loads(cred_value)
        client = vision.ImageAnnotatorClient.from_service_account_info(info)
        return client
    except json.JSONDecodeError:
        # 2) JSON이 아니면 경로로 간주
        if not os.path.exists(cred_value):
            raise Exception(
                "GOOGLE_APPLICATION_CREDENTIALS가 JSON도 아니고, "
                f"파일 경로({cred_value})도 아닙니다."
            )
        client = vision.ImageAnnotatorClient.from_service_account_file(cred_value)
        return client
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
        if any(x in line for x in ["TEL", "전화", "FAX", "팩스", "도로명", "주소"]):
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
    한국 동물병원 영수증 파서 (강화 버전)
    - '진료/미용 내역' 이후 구간만 항목 후보로 사용
    - 사업자등록번호/전화번호/날짜/발행일/결제요청/소계/합계/카드/과세/비과세 등은 항목에서 제외
    - 금액이 1,000원 미만인 줄은 무시
    - '이름 줄 + 다음 줄에 금액만 있는 경우'를 하나의 항목으로 결합
    """
    import re

    # 공백 줄 제거
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1) 병원명
    hospital_name = guess_hospital_name(lines)

    # 2) 날짜 후보들 수집 → 가장 이른 날짜 선택
    date_regex = re.compile(r"(20\d{2})[.\-/년 ]+(\d{1,2})[.\-/월 ]+(\d{1,2})")
    date_candidates: List[datetime] = []

    for line in lines:
        m = date_regex.search(line)
        if m:
            y, mo, d = map(int, m.groups())
            try:
                date_candidates.append(datetime(y, mo, d))
            except ValueError:
                pass

    visit_at: Optional[str] = None
    if date_candidates:
        first_date = sorted(date_candidates)[0]
        visit_at = first_date.strftime("%Y-%m-%d %H:%M")

    # 3) "진료/미용 내역" 이후 구간을 항목 후보로 설정
    start_idx = 0
    for idx, line in enumerate(lines):
        compact = line.replace(" ", "")
        if any(k in compact for k in ["진료및미용", "진료및미용내역", "진료내역", "진료및미용내역"]):
            start_idx = idx + 1
            break

    # 금액 패턴: 끝에 오는 숫자 (30,000 / 81000 / ￦30,000 / 30,000원)
    amt_pattern = re.compile(r"(?:₩|￦)?\s*(\d{1,3}(?:,\d{3})|\d+)\s(원)?$")

    # 항목으로 쓰지 않을 키워드
    skip_keywords = [
        "사업자등록번호", "사업자등록", "등록번호",
        "전화번호", "TEL", "팩스", "FAX",
        "고객명", "고객이름", "고객", "주소",
        "날짜", "일자", "발행일", "결제일", "기간",
        "합계", "소계", "총액", "총금액",
        "과세", "비과세", "부가세",
        "결제요청", "카드", "현금", "영수증",
    ]

    items: List[Dict] = []
    candidate_totals: List[int] = []

    current_name: Optional[str] = None

    for idx, raw_line in enumerate(lines):
        if idx < start_idx:
            # 진료/미용 내역 이전은 항목으로 보지 않음 (단, total 후보는 나중에 따로 처리)
            continue

        line = raw_line.lstrip("*").strip()
        if not line:
            continue

        # 먼저 금액 패턴 찾기
        m = amt_pattern.search(line)
        if not m:
            # 금액이 없는 줄인데, '항목 이름'일 수 있음
            # 숫자가 거의 없고, skip 키워드도 없으면 다음 줄 금액과 묶기 위해 저장
            has_digit = any(c.isdigit() for c in line)
            compact = line.replace(" ", "")
            if (not has_digit) and (not any(k in compact for k in skip_keywords)):
                current_name = line
            continue

        # 금액 추출
        try:
            amount = int(m.group(1).replace(",", ""))
        except ValueError:
            continue

        # 너무 작은 금액(예: 1원, 11원 등)은 OCR 잡음으로 보고 무시
        if amount < 1000:
            continue

        # 이름 부분
        name_part = line[:m.start()].strip()
        name = name_part or current_name or ""

        compact_name = name.replace(" ", "")

        # 소계/합계/결제요청/카드 등은 total 후보로만 사용
        if any(k in compact_name for k in skip_keywords):
            candidate_totals.append(amount)
            current_name = None
            continue

        if not name:
            name = "기타"

        items.append({"name": name, "amount": amount})
        current_name = None

    # total 계산
    if candidate_totals:
        total_amount = max(candidate_totals)
    else:
        total_amount = sum(i["amount"] for i in items) if items else 0

    return {
        "hospitalName": hospital_name,
        "visitAt": visit_at,
        "items": items,
        "totalAmount": total_amount,
    }


# ------------------------------------------
# Gemini LLM 를 이용한 AI 파싱
# ------------------------------------------

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
    실패하면 None 리턴
    """
    if settings.GEMINI_ENABLED.lower() != "true":
        return None
    if not settings.GEMINI_API_KEY:
        return None
    if genai is None:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)

        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
너는 한국 동물병원 영수증을 구조화된 JSON으로 정리하는 어시스턴트야.

다음은 OCR로 읽은 영수증 텍스트야:

\"\"\"{raw_text}\"\"\"


이 텍스트를 분석해서 아래 형식의 JSON만 돌려줘.
한국어로 되어 있어도 상관 없지만, 키 이름은 반드시 아래와 같아야 해.

반드시 이 JSON "한 개"만, 추가 설명 없이 순수 JSON으로만 출력해.

형식:
{{
  "clinicName": string or null,
  "visitDate": string or null,   // 형식: "YYYY-MM-DD"
  "diseaseName": string or null,
  "symptomsSummary": string or null,
  "items": [
    {{
      "name": string,
      "price": integer or null
    }},
    ...
  ],
  "totalAmount": integer or null
}}
"""

        resp = model.generate_content(prompt)
        text = resp.text.strip()

        # ⁠ json ...  ⁠ 같은 마크다운이 섞여 있을 수 있으니 정리
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
            if isinstance(it, dict):
                name = it.get("name")
                price = it.get("price")
            else:
                name = None
                price = None
            fixed_items.append({"name": name, "price": price})
        data["items"] = fixed_items

        return data

    except Exception:
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

    # 2) OCR 실행
    ocr_text = ""
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)

        ocr_text = run_vision_ocr(tmp_path)

    except Exception:
        ocr_text = ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3) Fallback 정규식 파서 (항목/금액은 여기 것이 기준)
    if ocr_text:
        fallback = parse_receipt_kor(ocr_text)
    else:
        fallback = {
            "hospitalName": "",
            "visitAt": None,
            "items": [],
            "totalAmount": 0,
        }

    # 4) Gemini AI 파싱 (병원명/날짜 위주로만 활용)
    ai_parsed = parse_receipt_ai(ocr_text) if ocr_text else None

    # clinicName
    clinic_name = None
    if ai_parsed and ai_parsed.get("clinicName"):
        clinic_name = ai_parsed["clinicName"]
    else:
        clinic_name = fallback.get("hospitalName") or "동물병원"

    # visitDate (YYYY-MM-DD)
    visit_date: Optional[str] = None
    if ai_parsed and ai_parsed.get("visitDate"):
        visit_date = ai_parsed["visitDate"]
    else:
        visit_at = fallback.get("visitAt")
        if visit_at:
            visit_date = visit_at.split(" ")[0]

    # disease / symptoms 는 Gemini 있으면 사용, 없으면 None
    disease_name = ai_parsed.get("diseaseName") if ai_parsed else None
    symptoms_summary = ai_parsed.get("symptomsSummary") if ai_parsed else None

    # 항목/금액은 항상 fallback 정규식 결과 사용
    dto_items: List[Dict] = []
    for it in fallback.get("items", []):
        dto_items.append(
            {
                "name": it.get("name"),
                "price": it.get("amount"),
            }
        )

    total_amount = fallback.get("totalAmount")

    parsed_for_dto = {
        "clinicName": clinic_name,
        "visitDate": visit_date,
        "diseaseName": disease_name,
        "symptomsSummary": symptoms_summary,
        "items": dto_items,
        "totalAmount": total_amount,
    }

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,
    }


# ------------------------------------------
# 2) 검사결과 PDF 업로드
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
    key = f"lab/{petId}/{lab_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "id": lab_id,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": file_url,          # iOS PdfRecord.s3Url
        "createdAt": created_at,
    }


# ------------------------------------------
# 3) 증명서 PDF 업로드
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
    key = f"cert/{petId}/{cert_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "id": cert_id,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": file_url,          # iOS PdfRecord.s3Url
        "createdAt": created_at,
    }


# ------------------------------------------
# 4) 검사결과 리스트
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
            key = obj["Key"]  # lab/{petId}/{id}.pdf
            if not key.endswith(".pdf"):
                continue

            file_id = os.path.splitext(key.split("/")[-1])[0]

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )
            created_at = obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%S")

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": "검사결과",
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at,
                }
            )

    return items


# ------------------------------------------
# 5) 증명서 리스트
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
            key = obj["Key"]  # cert/{petId}/{id}.pdf
            if not key.endswith(".pdf"):
                continue

            file_id = os.path.splitext(key.split("/")[-1])[0]

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )
            created_at = obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%S")

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": "증명서",
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at,
                }
            )

    return items
