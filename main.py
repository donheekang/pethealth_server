import os
import uuid
import json
import re
import logging
from datetime import datetime
from io import BytesIO
from typing import List, Optional

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Google Cloud / Gemini
from google.cloud import vision
from google.oauth2 import service_account
import google.generativeai as genai

# -----------------------------------------------------------------------------
# Logging 설정
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 환경변수
# -----------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Google Vision (서비스계정 JSON 을 ENV 로 넣었다고 가정)
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not S3_BUCKET_NAME:
    logger.warning("환경변수 S3_BUCKET_NAME 이 설정되어 있지 않습니다.")

# -----------------------------------------------------------------------------
# S3 클라이언트
# -----------------------------------------------------------------------------
s3_client = boto3.client("s3", region_name=AWS_REGION)

# -----------------------------------------------------------------------------
# Google Vision 클라이언트 초기화
# -----------------------------------------------------------------------------
if GOOGLE_CREDENTIALS_JSON:
    try:
        info = json.loads(GOOGLE_CREDENTIALS_JSON)
        creds = service_account.Credentials.from_service_account_info(info)
        vision_client = vision.ImageAnnotatorClient(credentials=creds)
        logger.info("Vision client initialized with explicit credentials.")
    except Exception as e:
        logger.exception("Failed to init Vision client with JSON env, fallback default: %s", e)
        vision_client = vision.ImageAnnotatorClient()
else:
    vision_client = vision.ImageAnnotatorClient()
    logger.info("Vision client initialized with default credentials.")

# -----------------------------------------------------------------------------
# Gemini 초기화
# -----------------------------------------------------------------------------
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    logger.info("Gemini model initialized.")
else:
    gemini_model = None
    logger.warning("GEMINI_API_KEY 가 없어 Gemini 를 사용하지 않습니다. (나이브 파서 사용)")

# -----------------------------------------------------------------------------
# FastAPI 앱
# -----------------------------------------------------------------------------
app = FastAPI(title="PetHealth+ Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic 모델
# -----------------------------------------------------------------------------
class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsed(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # "YYYY-MM-DD"
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None


class ReceiptAnalyzeResponseDTO(BaseModel):
    parsed: ReceiptParsed
    rawText: str


class PdfRecord(BaseModel):
    id: str
    petId: str
    title: Optional[str] = None
    memo: Optional[str] = None
    fileUrl: str


# -----------------------------------------------------------------------------
# 공통 유틸 – S3 업로드
# -----------------------------------------------------------------------------
def upload_to_s3(file_bytes: bytes, key: str, content_type: str) -> str:
    if not S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME 이 설정되어 있지 않습니다.")

    logger.info("Uploading to S3 bucket=%s key=%s", S3_BUCKET_NAME, key)

    fileobj = BytesIO(file_bytes)
    s3_client.upload_fileobj(
        fileobj,
        S3_BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": content_type},
    )

    url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return url


# -----------------------------------------------------------------------------
# OCR + Gemini 파싱 유틸
# -----------------------------------------------------------------------------
def run_ocr(image_bytes: bytes) -> str:
    """Google Vision 으로 텍스트 추출."""
    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)

    if response.error.message:
        logger.error("Vision OCR error: %s", response.error.message)
        raise RuntimeError(response.error.message)

    text = response.full_text_annotation.text or ""
    logger.info("OCR text length=%d", len(text))
    return text


def naive_parse_receipt(text: str) -> ReceiptParsed:
    """Gemini 안쓰는 경우를 위한 단순 파서 (백업용)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clinic_name = lines[0] if lines else None

    visit_date = None
    for ln in lines:
        for token in ln.split():
            if len(token) == 10 and token[4] == "-" and token[7] == "-":
                visit_date = token
                break
        if visit_date:
            break

    return ReceiptParsed(
        clinicName=clinic_name,
        visitDate=visit_date,
        items=[],
        totalAmount=None,
    )


def clean_json_from_markdown(s: str) -> str:
    """ ⁠ json ...  ⁠ 같은 마크다운 코드 블록 제거."""
    s = s.strip()
    # ⁠ json ...  ⁠ 제거
    s = re.sub(r"⁠  json\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("  ⁠", "")
    return s.strip()


def parse_receipt_with_gemini(ocr_text: str) -> ReceiptParsed:
    """Gemini 로 영수증 텍스트를 구조화된 JSON 으로 파싱."""
    if not gemini_model:
        logger.info("Gemini 미설정 → naive parser 사용")
        return naive_parse_receipt(ocr_text)

    prompt = f"""
너는 동물병원 영수증을 분석하는 어시스턴트야.
아래 OCR 텍스트를 보고, JSON 으로만 답해.

요구 포맷 (꼭 그대로, 키 이름/구조 바꾸지 말기):

{{
  "clinicName": "병원 이름 (모르면 null)",
  "visitDate": "YYYY-MM-DD 형식 (모르면 null)",
  "items": [
    {{
      "name": "항목 이름",
      "price": 12000   // 금액 없으면 null
    }}
  ],
  "totalAmount": 34000 // 총액 없으면 null
}}

규칙:
•⁠  ⁠날짜는 YYYY-MM-DD 형식 하나만 넣어.
•⁠  ⁠금액은 숫자만 (쉼표, '원' 제거).
•⁠  ⁠알 수 없는 값은 null 로.
•⁠  ⁠설명 문장 쓰지 말고, JSON 만 반환.

OCR 텍스트:
\"\"\"{ocr_text}\"\"\"
"""

    try:
        resp = gemini_model.generate_content(prompt)
        text = resp.text or ""
        text = clean_json_from_markdown(text)
        logger.info("Gemini raw response: %s", text[:200])

        data = json.loads(text)

        items = [
            ReceiptItem(
                name=item.get("name", ""),
                price=item.get("price"),
            )
            for item in data.get("items", [])
            if item.get("name")
        ]

        parsed = ReceiptParsed(
            clinicName=data.get("clinicName"),
            visitDate=data.get("visitDate"),
            items=items,
            totalAmount=data.get("totalAmount"),
        )
        return parsed
    except Exception as e:
        logger.exception("Gemini parse 실패, naive parser 로 fallback: %s", e)
        return naive_parse_receipt(ocr_text)


# -----------------------------------------------------------------------------
# 헬스 체크
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# -----------------------------------------------------------------------------
# 1) 영수증 이미지 분석 엔드포인트
#    iOS: APIClient.shared.analyzeReceipt(petId:image:)
# -----------------------------------------------------------------------------
@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponseDTO)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    data = await file.read()

    # 1) 영수증 이미지를 S3 에 저장 (원하면 나중에 목록/조회에 활용 가능)
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    key = f"receipts/{petId}/{uuid.uuid4().hex}{ext}"
    try:
        s3_url = upload_to_s3(data, key, file.content_type)
    except Exception as e:
        logger.exception("S3 업로드 실패")
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")

    # 2) OCR 실행
    try:
        ocr_text = run_ocr(data)
    except Exception as e:
        logger.exception("OCR 실패")
        raise HTTPException(status_code=500, detail=f"OCR 실패: {e}")

    # 3) Gemini 로 구조화 파싱
    parsed = parse_receipt_with_gemini(ocr_text)

    # rawText 에는 OCR 결과 + S3 URL 같이 넣어줌
    raw_text = f"[S3] {s3_url}\n\n{ocr_text}"

    return ReceiptAnalyzeResponseDTO(parsed=parsed, rawText=raw_text)


# -----------------------------------------------------------------------------
# 2) 검사결과 PDF 업로드
#    iOS: /api/lab/upload-pdf
# -----------------------------------------------------------------------------
@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: Optional[str] = Form(None),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    data = await file.read()
    key = f"labs/{petId}/{uuid.uuid4().hex}.pdf"

    try:
        url = upload_to_s3(data, key, "application/pdf")
    except Exception as e:
        logger.exception("S3 업로드 실패")
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")

    return PdfRecord(
        id=uuid.uuid4().hex,
        petId=petId,
        title=title,
        memo=memo,
        fileUrl=url,
    )


# -----------------------------------------------------------------------------
# 3) 증명서 PDF 업로드
#    iOS: /api/cert/upload-pdf
# -----------------------------------------------------------------------------
@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: Optional[str] = Form(None),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    data = await file.read()
    key = f"certs/{petId}/{uuid.uuid4().hex}.pdf"

    try:
        url = upload_to_s3(data, key, "application/pdf")
    except Exception as e:
        logger.exception("S3 업로드 실패")
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")

    return PdfRecord(
        id=uuid.uuid4().hex,
        petId=petId,
        title=title,
        memo=memo,
        fileUrl=url,
    )


# -----------------------------------------------------------------------------
# 로컬 개발용 진입점
# Render 에서는 "main:app" 으로 실행되기 때문에 이 부분은 무시됨
# -----------------------------------------------------------------------------
if _name_ == "_main_":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
