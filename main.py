from _future_ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# =========================
# 설정
# =========================

class Settings(BaseSettings):
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "ap-northeast-2"
    s3_bucket_name: str

    gemini_api_key: str | None = None
    gemini_enabled: bool = False

    stub_mode: bool = True  # True면 더미 데이터/로직 사용

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Render 환경변수 이름과 맞추기
        fields = {
            "aws_access_key_id": {"env": "AWS_ACCESS_KEY_ID"},
            "aws_secret_access_key": {"env": "AWS_SECRET_ACCESS_KEY"},
            "aws_region": {"env": "AWS_REGION"},
            "s3_bucket_name": {"env": "S3_BUCKET_NAME"},
            "gemini_api_key": {"env": "GEMINI_API_KEY"},
            "gemini_enabled": {"env": "GEMINI_ENABLED"},
            "stub_mode": {"env": "STUB_MODE"},
        }


settings = Settings()

# =========================
# AWS S3 클라이언트
# =========================

session = boto3.session.Session(
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    region_name=settings.aws_region,
)
s3 = session.client("s3")


def upload_file_to_s3(prefix: str, file: UploadFile, content_type: str) -> str:
    """
    파일을 S3에 올리고 object key를 반환
    """
    ext = os.path.splitext(file.filename or "")[1] or ".bin"
    key = f"{prefix}/{uuid.uuid4().hex}{ext}"

    try:
        s3.upload_fileobj(
            file.file,
            settings.s3_bucket_name,
            key,
            ExtraArgs={"ContentType": content_type},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")

    return key


def create_presigned_url(key: str, expires_in: int = 60 * 60 * 24 * 7) -> str:
    """
    S3 object에 접근 가능한 서명 URL 생성
    """
    try:
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": settings.s3_bucket_name, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 URL 생성 실패: {e}")


# =========================
# Pydantic 모델 (iOS DTO와 맞춤)
# =========================

class ReceiptItemDTO(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsedDTO(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # "yyyy-MM-dd"
    diseaseName: Optional[str] = None
    symptomsSummary: Optional[str] = None
    items: List[ReceiptItemDTO] = []
    totalAmount: Optional[int] = None


class ReceiptAnalyzeResponseDTO(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsedDTO
    notes: Optional[str] = None


class PdfRecord(BaseModel):
    id: str
    petId: str
    title: str
    memo: Optional[str] = None
    url: str          # iOS의 PdfRecord.s3Url 로 매핑
    createdAt: Optional[str] = None  # ISO 문자열


# =========================
# FastAPI 앱
# =========================

app = FastAPI(title="PetHealth+ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 기본/헬스 체크
# =========================

@app.get("/")
def root():
    return {"message": "PetHealth+ API running"}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# =========================
# 검사결과 / 증명서 리스트 (STUB)
# 실제로는 DB 붙이면 됨
# =========================

# 간단한 인메모리 스토리지 (STUB_MODE용)
LAB_STORE: list[PdfRecord] = []
CERT_STORE: list[PdfRecord] = []


@app.get("/labs/list", response_model=list[PdfRecord])
@app.get("/api/labs/list", response_model=list[PdfRecord])
def list_labs(petId: Optional[str] = None):
    """
    검사결과 리스트
    petId가 오면 해당 펫만 필터링
    """
    if petId:
        return [r for r in LAB_STORE if r.petId == petId]
    return LAB_STORE


@app.get("/cert/list", response_model=list[PdfRecord])
@app.get("/api/cert/list", response_model=list[PdfRecord])
def list_certs(petId: Optional[str] = None):
    """
    증명서 리스트
    """
    if petId:
        return [r for r in CERT_STORE if r.petId == petId]
    return CERT_STORE


# =========================
# PDF 업로드 (검사결과 / 증명서)
# =========================

def _build_pdf_record(
    store: list[PdfRecord],
    prefix: str,
    petId: str,
    title: str,
    memo: Optional[str],
    file: UploadFile,
) -> PdfRecord:
    key = upload_file_to_s3(prefix=prefix, file=file, content_type="application/pdf")
    url = create_presigned_url(key)
    now = datetime.now(timezone.utc).isoformat()

    record = PdfRecord(
        id=uuid.uuid4().hex,
        petId=petId,
        title=title,
        memo=memo,
        url=url,
        createdAt=now,
    )
    store.append(record)
    return record


@app.post("/lab/upload-pdf", response_model=PdfRecord)
@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form(...),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    검사결과 PDF 업로드
    """
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    return _build_pdf_record(
        store=LAB_STORE,
        prefix="labs",
        petId=petId,
        title=title,
        memo=memo,
        file=file,
    )


@app.post("/cert/upload-pdf", response_model=PdfRecord)
@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form(...),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    증명서 PDF 업로드
    """
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    return _build_pdf_record(
        store=CERT_STORE,
        prefix="certs",
        petId=petId,
        title=title,
        memo=memo,
        file=file,
    )


# =========================
# 병원 영수증 OCR → 진료기록 DTO
# (현재는 STUB: OCR/제미니 없이 더미값 반환)
# =========================

@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponseDTO)
async def analyze_receipt(
    petId: str = Form(...),
    image: UploadFile = File(...),
):
    """
    병원 영수증 이미지 업로드 + 분석
    - 이미지 원본은 S3에 저장
    - 현재는 STUB_MODE 기준으로 더미 분석 결과 반환
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/heic", "image/heif"):
        # iOS에서 heic로 올 수 있으니 넉넉히 허용
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    # 1) S3에 영수증 이미지 저장
    key = upload_file_to_s3(prefix=f"receipts/{petId}", file=image, content_type=image.content_type)
    s3_url = create_presigned_url(key)

    # 2) 실제라면 여기서 Google OCR + Gemini 호출해서 파싱
    if settings.stub_mode or not settings.gemini_enabled:
        # 더미 분석 결과
        parsed = ReceiptParsedDTO(
            clinicName="해랑동물병원",
            visitDate=datetime.now().strftime("%Y-%m-%d"),
            diseaseName="예방접종",
            symptomsSummary="정기 예방 접종 방문",
            items=[
                ReceiptItemDTO(name="DHPPI", price=30000),
                ReceiptItemDTO(name="Corona", price=25000),
                ReceiptItemDTO(name="Nexgard Spectra 7.5~15kg", price=26000),
            ],
            totalAmount=81000,
        )
        notes = "STUB_MODE: 실제 OCR 없이 더미 데이터입니다."
    else:
        # TODO: Google Vision / Gemini 연동 로직 구현
        # 지금은 구조만 맞춰두고, 나중에 진짜 분석 붙이면 됨
        parsed = ReceiptParsedDTO(
            clinicName=None,
            visitDate=None,
            diseaseName=None,
            symptomsSummary=None,
            items=[],
            totalAmount=None,
        )
        notes = "분석 로직 미구현 상태입니다."

    return ReceiptAnalyzeResponseDTO(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=notes,
    )
