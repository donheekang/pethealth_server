import os
import uuid
import datetime
import logging
from typing import List, Optional

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# 기본 로거 설정
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# =========================
# AWS S3 설정
# =========================
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "")


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def upload_to_s3(data: bytes, key: str, content_type: str) -> str:
    """
    S3에 업로드하고 public URL 리턴
    """
    if not AWS_S3_BUCKET:
        raise RuntimeError("환경변수 AWS_S3_BUCKET 이 설정되어 있지 않습니다.")

    s3 = get_s3_client()
    s3.put_object(
        Bucket=AWS_S3_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


# =========================
# Pydantic 모델들
# =========================
class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ParsedReceipt(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # "YYYY-MM-DD"
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None


class ReceiptAnalyzeResponse(BaseModel):
    parsed: ParsedReceipt


class PdfRecord(BaseModel):
    id: str
    petId: str
    title: Optional[str] = None
    memo: Optional[str] = None
    s3Url: str
    uploadedAt: str  # ISO8601 문자열


# =========================
# FastAPI 앱 생성
# =========================
app = FastAPI(title="PetHealth+ Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 특정 도메인으로 좁혀도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# 헬스 체크
# -------------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}


# =========================
# 1) 영수증 OCR 엔드포인트 (이미지)
# =========================
@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    iOS: multipart/form-data
      - petId: string
      - file: image/* (영수증 사진)
    """

    logger.info(f"[analyze_receipt] petId={petId}, filename={file.filename}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="비어 있는 파일입니다.")

    # S3에 원본 이미지 업로드 (선택)
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    key = f"receipts/{petId}/{uuid.uuid4()}{ext}"

    try:
        image_url = upload_to_s3(data, key, file.content_type or "image/jpeg")
        logger.info(f"[analyze_receipt] uploaded to {image_url}")
    except Exception as e:
        logger.exception("S3 업로드 실패")
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")

    # TODO: Google Vision + Gemini 연동 (현재는 더미 데이터)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    parsed = ParsedReceipt(
        clinicName="미분류 동물병원",
        visitDate=today_str,
        items=[],
        totalAmount=None,
    )

    return ReceiptAnalyzeResponse(parsed=parsed)


# =========================
# 2) PDF 업로드 공통 함수
# =========================
async def handle_pdf_upload(
    category: str,
    petId: str,
    title: str,
    memo: str,
    file: UploadFile,
) -> PdfRecord:
    """
    category: 'lab' 또는 'cert' 등
    """

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="비어 있는 파일입니다.")

    key = f"{category}/{petId}/{uuid.uuid4()}.pdf"

    try:
        url = upload_to_s3(data, key, "application/pdf")
        logger.info(f"[handle_pdf_upload] {category} pdf uploaded: {url}")
    except Exception as e:
        logger.exception("S3 PDF 업로드 실패")
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")

    now_iso = datetime.datetime.utcnow().isoformat()

    record = PdfRecord(
        id=str(uuid.uuid4()),
        petId=petId,
        title=title or None,
        memo=memo or None,
        s3Url=url,
        uploadedAt=now_iso,
    )
    return record


# =========================
# 3) 검사결과 PDF 업로드
# =========================
@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form(""),
    memo: str = Form(""),
    file: UploadFile = File(...),
):
    """
    iOS:
      - path: /api/lab/upload-pdf
      - fields: petId, title, memo, file(PDF)
    """
    return await handle_pdf_upload("lab", petId, title, memo, file)


# =========================
# 4) 증명서 PDF 업로드
# =========================
@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form(""),
    memo: str = Form(""),
    file: UploadFile = File(...),
):
    """
    iOS:
      - path: /api/cert/upload-pdf
      - fields: petId, title, memo, file(PDF)
    """
    return await handle_pdf_upload("cert", petId, title, memo, file)


# =========================
# 로컬 실행용 (Render는 uvicorn main:app ... 사용)
# =========================
if _name_ == "_main_":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
