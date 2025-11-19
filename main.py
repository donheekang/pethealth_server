import os
import uuid
import datetime
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import boto3


# =========================================
# 설정
# =========================================

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "ap-northeast-2"
    S3_BUCKET_NAME: str | None = None

    # STUB_MODE = True 이면 S3 안 쓰고 가짜 URL만 리턴
    STUB_MODE: bool = False


settings = Settings()

logger = logging.getLogger("pethealthplus")
logger.setLevel(logging.INFO)

if not settings.STUB_MODE:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )
else:
    s3_client = None


# =========================================
# FastAPI 기본 세팅
# =========================================

app = FastAPI(title="PetHealth+ Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# =========================================
# 모델 (검사결과 PDF / 증명서 PDF)
# =========================================

class LabDocument(BaseModel):
    id: str
    petId: str
    s3Url: str
    createdAt: datetime.datetime


class CertificateDocument(BaseModel):
    id: str
    petId: str
    s3Url: str
    createdAt: datetime.datetime


# 간단 인메모리 저장소 (배포 서버 재시작 시 초기화)
labs_db: dict[str, list[LabDocument]] = {}
cert_db: dict[str, list[CertificateDocument]] = {}


# =========================================
# 헬스 체크
# =========================================

@app.get("/")
async def root():
    """Render 브라우저에서 확인용 루트 엔드포인트"""
    return {"status": "ok", "stubMode": settings.STUB_MODE}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "stubMode": settings.STUB_MODE}


# =========================================
# 공통: S3 업로드 유틸
# =========================================

def _upload_to_s3(prefix: str, pet_id: str, file: UploadFile) -> str:
    """
    prefix: "labs" 또는 "certificates"
    pet_id: 반려동물 ID
    file  : 업로드된 PDF
    """
    # STUB_MODE면 그냥 가짜 URL
    if settings.STUB_MODE:
        key = f"{prefix}/{pet_id}/{uuid.uuid4()}.pdf"
        fake_url = f"https://stub-s3/{key}"
        logger.info(f"[STUB] upload_to_s3 -> {fake_url}")
        return fake_url

    if not settings.S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3 bucket not configured")

    key = f"{prefix}/{pet_id}/{uuid.uuid4()}.pdf"

    s3_client.upload_fileobj(
        file.file,
        settings.S3_BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": file.content_type or "application/pdf"},
    )

    url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
    logger.info(f"Uploaded to S3: {url}")
    return url


# =========================================
# 응답 직렬화 helper
#  - Swift 쪽에서 fileUrl / url 을 기대할 수도 있으니
#    s3Url 과 별도로 fileUrl 키도 같이 내려줌
# =========================================

def serialize_lab(doc: LabDocument) -> dict:
    return {
        "id": doc.id,
        "petId": doc.petId,
        "fileUrl": doc.s3Url,     # Swift 가 기대할 가능성 큰 키
        "s3Url": doc.s3Url,       # 실제 S3 URL
        "createdAt": doc.createdAt,
    }


def serialize_cert(doc: CertificateDocument) -> dict:
    return {
        "id": doc.id,
        "petId": doc.petId,
        "fileUrl": doc.s3Url,
        "s3Url": doc.s3Url,
        "createdAt": doc.createdAt,
    }


# =========================================
# 검사결과(Lab) 리스트 조회
# =========================================

def _list_labs(pet_id: str | None):
    if pet_id:
        docs = labs_db.get(pet_id, [])
    else:
        docs: list[LabDocument] = []
        for d in labs_db.values():
            docs.extend(d)
    return [serialize_lab(d) for d in docs]


@app.get("/labs")
@app.get("/labs/list")
@app.get("/api/labs")
@app.get("/api/labs/list")
async def list_labs(petId: str | None = None):
    return _list_labs(petId)


# iOS 앱에서 실제 호출하는 경로
@app.get("/api/lab/list")
async def api_lab_list(petId: str | None = None):
    return _list_labs(petId)


# =========================================
# 증명서(Certificate) 리스트 조회
# =========================================

def _list_certs(pet_id: str | None):
    if pet_id:
        docs = cert_db.get(pet_id, [])
    else:
        docs: list[CertificateDocument] = []
        for d in cert_db.values():
            docs.extend(d)
    return [serialize_cert(d) for d in docs]


@app.get("/certificates")
@app.get("/api/certificates")
async def list_certificates(petId: str | None = None):
    return _list_certs(petId)


@app.get("/api/cert/list")
async def api_cert_list(petId: str | None = None):
    return _list_certs(petId)


# =========================================
# 검사결과(Lab) PDF 업로드
# =========================================

@app.post("/labs/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    url = _upload_to_s3("labs", petId, file)

    doc = LabDocument(
        id=str(uuid.uuid4()),
        petId=petId,
        s3Url=url,
        createdAt=datetime.datetime.utcnow(),
    )
    labs_db.setdefault(petId, []).append(doc)
    return serialize_lab(doc)


# iOS 앱에서 실제 호출하는 경로
@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf_alias(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    return await upload_lab_pdf(petId=petId, file=file)


# =========================================
# 증명서(Certificate) PDF 업로드 (필요 시)
# =========================================

@app.post("/certificates/upload-pdf")
async def upload_certificate_pdf(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    url = _upload_to_s3("certificates", petId, file)

    doc = CertificateDocument(
        id=str(uuid.uuid4()),
        petId=petId,
        s3Url=url,
        createdAt=datetime.datetime.utcnow(),
    )
    cert_db.setdefault(petId, []).append(doc)
    return serialize_cert(doc)
