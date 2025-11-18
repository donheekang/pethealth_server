import os
import uuid
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import boto3

# =========================
# 로깅 설정
# =========================
logger = logging.getLogger(_name_)
logging.basicConfig(level=logging.INFO)

# =========================
# FastAPI 앱
# =========================
app = FastAPI(title="PetHealth+ Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 필요하면 iOS / 웹 도메인만 넣어도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# S3 설정
# =========================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not S3_BUCKET_NAME:
    logger.warning("⚠️ 환경변수 S3_BUCKET_NAME 이 설정되어 있지 않습니다.")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# =========================
# DTO 정의 (iOS와 맞추기)
# =========================

class ReceiptItemDTO(BaseModel):
    name: str
    price: int | None = None


class ParsedReceiptDTO(BaseModel):
    clinicName: str | None = None
    visitDate: str | None = None  # "yyyy-MM-dd"
    items: list[ReceiptItemDTO] = []
    totalAmount: int | None = None


class ReceiptAnalyzeResponseDTO(BaseModel):
    parsed: ParsedReceiptDTO


class PdfUploadResponse(BaseModel):
    id: str
    s3Url: str


# =========================
# 유틸 함수
# =========================

def s3_upload_fileobj(file_obj, key: str, content_type: str) -> str:
    """
    S3에 파일 업로드 후 public url 반환
    """
    if not S3_BUCKET_NAME:
        raise RuntimeError("S3_BUCKET_NAME is not configured")

    s3_client.upload_fileobj(
        file_obj,
        S3_BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": content_type}
    )

    # 퍼블릭 URL (버킷 정책에 따라 다를 수 있음)
    url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return url


# =========================
# 헬스 체크
# =========================
@app.get("/health")
async def health():
    return {"status": "ok"}


# =========================
# 1) 영수증 분석 (임시 더미 버전)
#    iOS: APIClient.shared.analyzeReceipt(petId:image:) 에서 호출
#    경로는 필요에 따라 /api/receipt/analyze 로 맞춰 사용
# =========================
@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponseDTO)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    영수증 이미지 분석 엔드포인트 (현재는 OCR 없이 더미 데이터 반환)
    나중에 Google Vision + Gemini 로직을 여기에 붙이면 됨.
    """
    logger.info(f"📥 receipt analyze requested. petId={petId}, filename={file.filename}")

    # 파일은 지금은 그냥 읽기만 하고 사용 안 함 (OCR 연동 예정)
    _ = await file.read()

    # 오늘 날짜를 yyyy-MM-dd 로
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    parsed = ParsedReceiptDTO(
        clinicName="동물병원",
        visitDate=today_str,
        items=[],          # 실제 OCR 로직 붙이면 항목 채우기
        totalAmount=None,  # 마찬가지
    )

    return ReceiptAnalyzeResponseDTO(parsed=parsed)


# =========================
# 2) PDF 업로드 공통 핸들러
# =========================
async def _handle_pdf_upload(
    prefix: str,
    petId: str,
    title: str | None,
    memo: str | None,
    file: UploadFile,
) -> PdfUploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    # 고유 ID + 키 생성
    record_id = str(uuid.uuid4())
    ext = ".pdf"
    key = f"{prefix}/{petId}/{record_id}{ext}"

    logger.info(f"📤 uploading PDF to s3. key={key}")

    # 파일을 바이트로 읽어서 S3 업로드
    file_bytes = await file.read()
    from io import BytesIO
    file_obj = BytesIO(file_bytes)

    try:
        url = s3_upload_fileobj(file_obj, key, "application/pdf")
    except Exception as e:
        logger.exception("S3 업로드 실패")
        raise HTTPException(status_code=500, detail="S3 업로드 중 오류가 발생했습니다.") from e

    logger.info(f"✅ PDF uploaded. url={url}")

    # iOS 쪽 PdfUploadResponse(id, s3Url)에 맞춰 반환
    return PdfUploadResponse(id=record_id, s3Url=url)


# =========================
# 2-1) 검사결과 PDF 업로드
#     iOS: uploadLabPDF(...) path: "/api/lab/upload-pdf"
# =========================
@app.post("/api/lab/upload-pdf", response_model=PdfUploadResponse)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str | None = Form(None),
    memo: str | None = Form(None),
    file: UploadFile = File(...),
):
    """
    검사결과 PDF 업로드
    """
    return await _handle_pdf_upload(
        prefix="lab",
        petId=petId,
        title=title,
        memo=memo,
        file=file,
    )


# =========================
# 2-2) 증명서 PDF 업로드
#     iOS: uploadCertPDF(...) path: "/api/cert/upload-pdf"
# =========================
@app.post("/api/cert/upload-pdf", response_model=PdfUploadResponse)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str | None = Form(None),
    memo: str | None = Form(None),
    file: UploadFile = File(...),
):
    """
    증명서 PDF 업로드
    """
    return await _handle_pdf_upload(
        prefix="cert",
        petId=petId,
        title=title,
        memo=memo,
        file=file,
    )


# =========================
# 로컬 테스트용 진입점 (Render에서는 uvicorn main:app 사용)
# =========================
if _name_ == "_main_":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
