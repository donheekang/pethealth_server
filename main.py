import os
import io
import uuid
import logging
from datetime import datetime
from typing import List, Optional

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.cloud import vision

# ------------------------------------------------------------------------------
# 기본 설정 & 로거
# ------------------------------------------------------------------------------
logger = logging.getLogger(_name_)  # ✅ _name_ 오타 주의
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# 환경변수 (Render 대시보드에서 설정한 값과 매칭)
# ------------------------------------------------------------------------------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# GOOGLE_APPLICATION_CREDENTIALS = "/opt/render/project/src/google-cred.json" 처럼
# "컨테이너 안의 파일 경로" 여야 해.
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

# ------------------------------------------------------------------------------
# S3 클라이언트 생성
# ------------------------------------------------------------------------------
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME]):
    logger.warning("⚠️ AWS/S3 환경변수가 일부 비어 있습니다. S3 업로드가 실패할 수 있습니다.")

session = boto3.session.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)
s3_client = session.client("s3")

# ------------------------------------------------------------------------------
# Vision 클라이언트 생성
# ------------------------------------------------------------------------------
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

try:
    vision_client = vision.ImageAnnotatorClient()
    logger.info("✅ Google Vision client initialized")
except Exception as e:
    logger.error(f"Google Vision 초기화 실패: {e}")
    vision_client = None

# ------------------------------------------------------------------------------
# FastAPI 앱
# ------------------------------------------------------------------------------
app = FastAPI(title="PetHealth+ Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 iOS 도메인/포트만 남기기
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Pydantic 모델 - iOS에서 쓰는 DTO와 맞추기
# ------------------------------------------------------------------------------
class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ParsedReceipt(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # "YYYY-MM-DD"
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None


class ReceiptAnalyzeResponseDTO(BaseModel):
    parsed: ParsedReceipt
    rawText: str


class PdfRecord(BaseModel):
    id: str
    petId: str
    title: Optional[str] = None
    memo: Optional[str] = None
    s3Url: str
    createdAt: datetime


# 메모리용 저장 (나중에 DB 붙이면 대체)
LAB_PDFS: dict[str, List[PdfRecord]] = {}
CERT_PDFS: dict[str, List[PdfRecord]] = {}

# ------------------------------------------------------------------------------
# 유틸: S3 업로드
# ------------------------------------------------------------------------------
def upload_to_s3(file_bytes: bytes, key: str, content_type: str) -> str:
    if STUB_MODE:
        # STUB 모드면 그냥 가짜 URL 리턴 (테스트용)
        return f"https://example.com/{key}"

    if not S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME 환경변수가 비어 있습니다.")

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )

    url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return url


# ------------------------------------------------------------------------------
# 유틸: OCR + 간단 파싱 (Google Vision만 사용)
# ------------------------------------------------------------------------------
def ocr_receipt_image(image_bytes: bytes) -> str:
    """
    영수증 이미지에서 텍스트만 뽑아오는 함수
    """
    if STUB_MODE or vision_client is None:
        # 테스트용 더미 텍스트
        return """펫케어동물병원
2025-11-18
진찰료 20000
X-ray 50000
혈액검사 80000
합계 150000"""

    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    return response.full_text_annotation.text


def parse_receipt_text(text: str) -> ParsedReceipt:
    """
    OCR 결과 텍스트를 매우 단순하게 파싱.
    - 첫 줄: 병원 이름 가정
    - 첫 번째 날짜 패턴: 방문일
    - '원' 이 들어간 줄: 항목/금액 추출
    """
    import re

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ParsedReceipt(items=[])

    clinic_name = lines[0]

    # 날짜 찾기
    date_pattern = re.compile(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})")
    visit_date = None
    for ln in lines:
        m = date_pattern.search(ln)
        if m:
            y, mth, d = m.groups()
            visit_date = f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
            break

    items: List[ReceiptItem] = []
    total_amount = None

    money_pattern = re.compile(r"([0-9,]+)\s*원")

    for ln in lines[1:]:
        m = money_pattern.search(ln)
        if not m:
            continue
        amount_str = m.group(1).replace(",", "")
        try:
            amount = int(amount_str)
        except ValueError:
            continue

        if "합계" in ln or "총액" in ln:
            total_amount = amount
        else:
            # 매우 단순하게: 숫자 앞 부분을 항목 이름으로 사용
            name = ln[: m.start()].strip() or "항목"
            items.append(ReceiptItem(name=name, price=amount))

    # 총액이 없으면 항목 가격 합으로 대체
    if total_amount is None and items:
        total_amount = sum(i.price or 0 for i in items)

    return ParsedReceipt(
        clinicName=clinic_name,
        visitDate=visit_date,
        items=items,
        totalAmount=total_amount,
    )


# ------------------------------------------------------------------------------
# API 라우트
# ------------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "stubMode": STUB_MODE}


# ------------------------- 영수증 분석 (진료기록) ------------------------------
@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponseDTO)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    iOS: APIClient.shared.analyzeReceipt(petId:image:) 가 호출하는 엔드포인트
    - multipart/form-data 로 petId + file(이미지) 받음
    - Vision OCR -> 텍스트 -> 간단 파싱 -> DTO 리턴
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        raw_text = ocr_receipt_image(content)
        parsed = parse_receipt_text(raw_text)

        return ReceiptAnalyzeResponseDTO(parsed=parsed, rawText=raw_text)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("영수증 분석 실패")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------- PDF 업로드 (검사결과 / 증명서) ----------------------
@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: Optional[str] = Form(None),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    iOS: APIClient.uploadLabPDF(petId:title:memo:fileURL:)
    """
    return await _handle_pdf_upload(petId, title, memo, file, LAB_PDFS, folder="lab")


@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: Optional[str] = Form(None),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    iOS: APIClient.uploadCertPDF(petId:title:memo:fileURL:)
    """
    return await _handle_pdf_upload(petId, title, memo, file, CERT_PDFS, folder="cert")


async def _handle_pdf_upload(
    pet_id: str,
    title: Optional[str],
    memo: Optional[str],
    file: UploadFile,
    store: dict[str, List[PdfRecord]],
    folder: str,
) -> PdfRecord:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    ext = ".pdf"
    object_id = str(uuid.uuid4())
    key = f"{folder}/{pet_id}/{object_id}{ext}"

    try:
        s3_url = upload_to_s3(content, key, "application/pdf")
    except Exception as e:
        logger.exception("S3 업로드 실패")
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")

    record = PdfRecord(
        id=object_id,
        petId=pet_id,
        title=title,
        memo=memo,
        s3Url=s3_url,
        createdAt=datetime.utcnow(),
    )

    store.setdefault(pet_id, []).append(record)
    return record


# --------------------- PDF 목록 조회 (선택) ------------------------------------
@app.get("/api/lab/list", response_model=List[PdfRecord])
async def list_lab_pdfs(petId: str):
    return LAB_PDFS.get(petId, [])


@app.get("/api/cert/list", response_model=List[PdfRecord])
async def list_cert_pdfs(petId: str):
    return CERT_PDFS.get(petId, [])


# ------------------------------------------------------------------------------
# Render: Start command 에서 ⁠ uvicorn main:app --host 0.0.0.0 --port $PORT ⁠
# 를 쓰고 있으므로, 여기서는 따로 if _name_ == "_main_": 없어도 됨.
# ------------------------------------------------------------------------------
