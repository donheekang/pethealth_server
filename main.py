import os
import uuid
import logging
from datetime import datetime
from io import BytesIO
from typing import List, Optional

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Logging 설정
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# -----------------------------------------------------------------------------
# 환경 변수
# -----------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not S3_BUCKET_NAME:
    logger.warning("환경변수 S3_BUCKET_NAME 이 설정되어 있지 않습니다.")

# boto3 S3 클라이언트 (자격 증명은 Render 환경변수에서 자동으로 읽음)
s3_client = boto3.client("s3", region_name=AWS_REGION)

# -----------------------------------------------------------------------------
# FastAPI 앱
# -----------------------------------------------------------------------------
app = FastAPI(title="PetHealth+ Server", version="1.0.0")

# CORS (앱에서 호출할 수 있도록 전체 허용)
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
# 유틸 함수 (S3 업로드)
# -----------------------------------------------------------------------------
def upload_to_s3(file_bytes: bytes, key: str, content_type: str) -> str:
    """
    S3 에 파일 업로드 후, 접근 가능한 URL 을 반환합니다.
    """
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

    # 지역 기반 public URL (필요하면 여기 형식만 바꿔주면 됨)
    url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return url


def naive_parse_receipt(text: str) -> ReceiptParsed:
    """
    아직 Google OCR / Gemini 안 붙이고, 임시로 대충 파싱하는 함수.
    나중에 OCR + Gemini 붙일 때 이 함수 안만 교체하면 됨.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clinic_name = lines[0] if lines else None

    # YYYY-MM-DD 패턴 간단 탐색
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


# -----------------------------------------------------------------------------
# 헬스 체크
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# -----------------------------------------------------------------------------
# 1) 영수증 이미지 분석 (현재는 OCR 더미 + S3 업로드)
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

    # 1) S3 업로드
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    key = f"receipts/{petId}/{uuid.uuid4().hex}{ext}"
    try:
        s3_url = upload_to_s3(data, key, file.content_type)
    except Exception as e:
        logger.exception("S3 업로드 실패")
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")

    # 2) 아직 OCR 안 붙였으니까, rawText 에 파일 정보만 넣어둠
    raw_text = f"petId={petId}, filename={file.filename}, s3Url={s3_url}"

    parsed = naive_parse_receipt(raw_text)

    return ReceiptAnalyzeResponseDTO(parsed=parsed, rawText=raw_text)


# -----------------------------------------------------------------------------
# 2) 검사결과 PDF 업로드
#    iOS: /api/lab/upload-pdf 로 맞춰뒀음
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
#    iOS: /api/cert/upload-pdf 로 맞춰뒀음
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
# 로컬 개발용 진입점 (Render 에서는 main:app 으로 실행됨)
# -----------------------------------------------------------------------------
if _name_ == "_main_":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
