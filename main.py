# main.py
import os
import io
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import boto3
from botocore.client import Config

# ------------------------------------------------------
# 환경 스위치
# ------------------------------------------------------
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"
MAX_IMAGE_BYTES = 15 * 1024 * 1024

# ------------------------------------------------------
# S3 설정
# ------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE")  # 예: https://my-bucket.s3.ap-northeast-2.amazonaws.com

s3_client = None
if not STUB_MODE and S3_BUCKET:
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4")
    )

# ------------------------------------------------------
# FastAPI 앱
# ------------------------------------------------------
app = FastAPI(title="PetHealth+ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def build_s3_url(key: str) -> str:
    if S3_PUBLIC_BASE:
        return f"{S3_PUBLIC_BASE.rstrip('/')}/{key}"
    # 기본 형식
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

def upload_bytes_to_s3(data: bytes, key: str, content_type: str) -> str:
    """
    S3에 업로드 후 공개 URL 반환.
    STUB_MODE면 가짜 URL만 리턴.
    """
    if STUB_MODE or not s3_client or not S3_BUCKET:
        return f"https://stub.local/{key}"

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
        ACL="public-read",  # 필요에 따라 조정
    )
    return build_s3_url(key)

# ------------------------------------------------------
# 모델 정의 (Swift DTO에 맞춰 설계)
# ------------------------------------------------------

# 1) 영수증 분석용 DTO -----------------------------------
class ReceiptItemDTO(BaseModel):
    id: str
    name: str
    price: Optional[int] = None

class ReceiptParsedDTO(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # "yyyy-MM-dd"
    items: List[ReceiptItemDTO] = Field(default_factory=list)
    totalAmount: Optional[int] = None

class ReceiptAnalyzeResponseDTO(BaseModel):
    id: str
    petId: str
    createdAt: str
    parsed: ReceiptParsedDTO

# 2) PDF 업로드용 DTO ------------------------------------
class PdfRecord(BaseModel):
    id: str
    petId: str
    title: str
    memo: Optional[str] = None
    uploadedAt: str     # ISO8601 문자열
    fileURL: str        # S3 또는 스텁 URL

# ------------------------------------------------------
# 헬퍼: 영수증 분석 (Stub 버전)
# ------------------------------------------------------
def analyze_receipt_stub(pet_id: str) -> ReceiptAnalyzeResponseDTO:
    """
    실제로는 Google Vision + Gemini 로직이 들어갈 자리.
    지금은 iOS 개발용 stub만 내려줌.
    """
    now = now_iso()
    items = [
        ReceiptItemDTO(
            id=str(uuid.uuid4()),
            name="진찰료",
            price=15000,
        ),
        ReceiptItemDTO(
            id=str(uuid.uuid4()),
            name="예방접종",
            price=35000,
        ),
    ]
    parsed = ReceiptParsedDTO(
        clinicName="PetHealth 동물병원",
        visitDate=now[:10],   # "YYYY-MM-DD"
        items=items,
        totalAmount=sum([i.price or 0 for i in items])
    )
    return ReceiptAnalyzeResponseDTO(
        id=str(uuid.uuid4()),
        petId=pet_id,
        createdAt=now,
        parsed=parsed,
    )

# ------------------------------------------------------
# 라우트: 헬스체크
# ------------------------------------------------------
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "stubMode": STUB_MODE}

# ------------------------------------------------------
# 라우트: 영수증 분석 (이미지 OCR)
# ------------------------------------------------------
@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponseDTO)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...)
):
    """
    iOS에서 영수증 이미지를 올릴 때 호출.
    - petId: 반려동물 ID
    - file: 이미지 (jpeg/png 등)
    현재는 STUB로 동작하고, 나중에 Vision+Gemini 붙이면 됨.
    """
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="이미지 용량이 너무 큽니다.")

    # 필요하다면 이미지 S3 업로드 (선택사항)
    _ = upload_bytes_to_s3(
        data=data,
        key=f"receipt-images/{petId}/{uuid.uuid4()}.jpg",
        content_type=file.content_type or "image/jpeg"
    )

    # 실제 분석 대신 Stub 호출
    result = analyze_receipt_stub(pet_id=petId)
    return result

# ------------------------------------------------------
# 라우트: 검사결과 PDF 업로드
# ------------------------------------------------------
@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form(""),
    memo: str = Form(""),
    file: UploadFile = File(...)
):
    """
    검사결과 PDF 업로드
    Swift: APIClient.shared.uploadLabPDF(...)
    경로: /api/lab/upload-pdf
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # 일부 브라우저는 octet-stream 으로 보내기도 함
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일은 업로드할 수 없습니다.")

    record_id = str(uuid.uuid4())
    key = f"lab/{petId}/{record_id}.pdf"

    url = upload_bytes_to_s3(data=data, key=key, content_type="application/pdf")

    rec = PdfRecord(
        id=record_id,
        petId=petId,
        title=title or "검사결과",
        memo=memo or None,
        uploadedAt=now_iso(),
        fileURL=url,
    )
    return rec

# ------------------------------------------------------
# 라우트: 증명서 PDF 업로드
# ------------------------------------------------------
@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form(""),
    memo: str = Form(""),
    file: UploadFile = File(...)
):
    """
    증명서 PDF 업로드
    Swift: APIClient.shared.uploadCertPDF(...)
    경로: /api/cert/upload-pdf
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일은 업로드할 수 없습니다.")

    record_id = str(uuid.uuid4())
    key = f"cert/{petId}/{record_id}.pdf"

    url = upload_bytes_to_s3(data=data, key=key, content_type="application/pdf")

    rec = PdfRecord(
        id=record_id,
        petId=petId,
        title=title or "증명서",
        memo=memo or None,
        uploadedAt=now_iso(),
        fileURL=url,
    )
    return rec

# ------------------------------------------------------
# (선택) Uvicorn 실행용
# ------------------------------------------------------
if _name_ == "_main_":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
