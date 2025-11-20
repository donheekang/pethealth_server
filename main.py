from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

import boto3
from uuid import uuid4

# -------------------------------------------------------------------
# 설정 / 환경변수
# -------------------------------------------------------------------


class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "ap-northeast-2"
    S3_BUCKET_NAME: str | None = None

    GEMINI_API_KEY: str | None = None
    GEMINI_ENABLED: bool = False

    # 서버 테스트용 플래그 (S3 / Gemini 안 쓰고 가짜 데이터만)
    STUB_MODE: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
logger = logging.getLogger("pethealthplus")
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------------
# S3 클라이언트 (STUB_MODE 이거나 설정이 없으면 None)
# -------------------------------------------------------------------

s3_client = None
if (
    not settings.STUB_MODE
    and settings.AWS_ACCESS_KEY_ID
    and settings.AWS_SECRET_ACCESS_KEY
    and settings.S3_BUCKET_NAME
):
    session = boto3.session.Session(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )
    s3_client = session.client("s3")
    logger.info("S3 client initialized")
else:
    logger.info("S3 client not initialized (stub mode or missing credentials)")

# -------------------------------------------------------------------
# (선택) Gemini / OCR – 설치 안 되어 있으면 그냥 None
# -------------------------------------------------------------------

try:
    import google.generativeai as genai

    if settings.GEMINI_API_KEY and settings.GEMINI_ENABLED:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        logger.info("Gemini client configured")
    else:
        genai = None
        logger.info("Gemini not enabled")
except ImportError:
    genai = None
    logger.info("google.generativeai is not installed, Gemini disabled")

# -------------------------------------------------------------------
# 공통 모델
# -------------------------------------------------------------------


class DocumentRecord(BaseModel):
    """
    iOS 쪽에서 검사결과/증명서 둘 다 이런 형태의 JSON을 디코딩한다고 가정.
    (keyNotFound 방지용으로 넉넉하게 필드 넣어 둠)
    """

    id: str
    petId: str
    fileName: str
    fileUrl: str  # iOS struct 에서 fileUrl 로 쓸 수 있게
    url: str      # 혹시 url 로만 받는 경우도 대비
    uploadedAt: datetime
    createdAt: datetime
    category: str  # "lab" or "cert"


# petId별로 메모리상에 저장 (DB 대신 임시 저장소)
LAB_STORE: Dict[str, List[DocumentRecord]] = {}
CERT_STORE: Dict[str, List[DocumentRecord]] = {}


def _get_store(category: str) -> Dict[str, List[DocumentRecord]]:
    if category == "lab":
        return LAB_STORE
    elif category == "cert":
        return CERT_STORE
    else:
        raise ValueError(f"Unknown category: {category}")


def _make_document_record(
    *,
    category: str,
    pet_id: str,
    filename: str,
    url: str,
) -> DocumentRecord:
    now = datetime.now(timezone.utc)
    return DocumentRecord(
        id=str(uuid4()),
        petId=pet_id,
        fileName=filename,
        fileUrl=url,
        url=url,
        uploadedAt=now,
        createdAt=now,
        category=category,
    )


# -------------------------------------------------------------------
# S3 업로드 헬퍼
# -------------------------------------------------------------------


async def save_pdf_to_s3(category: str, pet_id: str, file: UploadFile) -> str:
    """
    PDF 를 S3 에 저장하고, iOS 에서 바로 열 수 있는 presigned URL 을 리턴.
    STUB_MODE 이거나 S3 설정이 없으면 가짜 URL 리턴.
    """
    if settings.STUB_MODE or s3_client is None or not settings.S3_BUCKET_NAME:
        # 서버/앱 연동 테스트용 가짜 URL
        fake_url = f"https://example.com/{category}/{pet_id}/{uuid4()}.pdf"
        logger.info(f"[STUB] return fake url: {fake_url}")
        return fake_url

    content = await file.read()
    key = f"{category}/{pet_id}/{uuid4()}_{file.filename}"

    logger.info(f"Uploading to S3: bucket={settings.S3_BUCKET_NAME}, key={key}")
    s3_client.put_object(
        Bucket=settings.S3_BUCKET_NAME,
        Key=key,
        Body=content,
        ContentType=file.content_type or "application/pdf",
    )

    # presigned URL 생성 (예: 7일 유효)
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
        ExpiresIn=7 * 24 * 3600,
    )
    logger.info(f"Generated presigned url: {url}")
    return url


# -------------------------------------------------------------------
# FastAPI 앱 초기화
# -------------------------------------------------------------------

app = FastAPI(title="PetHealth+ Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # iOS 앱/시뮬레이터에서 자유롭게 호출
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# 헬스 체크 / 기본 라우트
# -------------------------------------------------------------------


@app.get("/")
async def root():
    """Render 브라우저 및 기본 서버 동작 확인용"""
    return {"status": "ok", "stubMode": settings.STUB_MODE}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "stubMode": settings.STUB_MODE}


# -------------------------------------------------------------------
# 검사결과(Lab) 리스트 조회
#  - iOS 에서 호출하는 엔드포인트: GET /api/lab/list?petId=...
#  - 브라우저 테스트용 엔드포인트도 같이 묶어둠
# -------------------------------------------------------------------


@app.get(
    "/api/lab/list",
    response_model=list[DocumentRecord],
)
@app.get("/labs", response_model=list[DocumentRecord])
@app.get("/labs/list", response_model=list[DocumentRecord])
@app.get("/api/labs", response_model=list[DocumentRecord])
@app.get("/api/labs/list", response_model=list[DocumentRecord])
async def get_lab_list(petId: str = Query(..., alias="petId")):
    """
    petId 기준으로 검사결과 PDF 리스트 리턴.
    """
    store = _get_store("lab")
    return store.get(petId, [])


# -------------------------------------------------------------------
# 증명서(Cert) 리스트 조회
#  - iOS: GET /api/cert/list?petId=...
# -------------------------------------------------------------------


@app.get(
    "/api/cert/list",
    response_model=list[DocumentRecord],
)
@app.get("/certificates", response_model=list[DocumentRecord])
@app.get("/api/certificates", response_model=list[DocumentRecord])
async def get_cert_list(petId: str = Query(..., alias="petId")):
    store = _get_store("cert")
    return store.get(petId, [])


# -------------------------------------------------------------------
# 검사결과(Lab) PDF 업로드
#  - iOS: POST /api/lab/upload-pdf (multipart/form-data)
#  - Form 필드: petId, file
#  - 응답: 업로드된 DocumentRecord (iOS struct 와 맞춤)
# -------------------------------------------------------------------


@app.post("/api/lab/upload-pdf", response_model=DocumentRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    if file.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    url = await save_pdf_to_s3(category="lab", pet_id=petId, file=file)
    record = _make_document_record(
        category="lab",
        pet_id=petId,
        filename=file.filename,
        url=url,
    )

    store = _get_store("lab")
    store.setdefault(petId, []).append(record)

    return record


# -------------------------------------------------------------------
# 증명서(Cert) PDF 업로드
#  - iOS: POST /api/cert/upload-pdf (multipart/form-data)
#  - Form 필드: petId, file
#  - 응답: DocumentRecord (Lab 과 동일한 구조)
# -------------------------------------------------------------------


@app.post("/api/cert/upload-pdf", response_model=DocumentRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    if file.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    url = await save_pdf_to_s3(category="cert", pet_id=petId, file=file)
    record = _make_document_record(
        category="cert",
        pet_id=petId,
        filename=file.filename,
        url=url,
    )

    store = _get_store("cert")
    store.setdefault(petId, []).append(record)

    return record


# -------------------------------------------------------------------
# (선택) OCR + Gemini 분석 엔드포인트 (추가 기능용, 안 쓰면 무시)
#  - iOS 에서 나중에 쓸 수 있게 기본 형태만 둠
# -------------------------------------------------------------------


class AnalysisRequest(BaseModel):
    petId: str
    s3Url: str | None = None
    text: str | None = None


class AnalysisResponse(BaseModel):
    summary: str
    weight: float | None = None
    memo: str | None = None


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_document(payload: AnalysisRequest):
    """
    OCR/Gemini 사용해서 진료/검사 내용을 요약하고,
    체중 등의 정보를 추출하는 엔드포인트(기본 골격만).
    """
    if not genai or not settings.GEMINI_ENABLED:
        # Gemini 비활성화 시에는 간단한 더미 응답
        return AnalysisResponse(
            summary="AI 분석 기능이 비활성화된 상태입니다.",
            weight=None,
            memo=None,
        )

    prompt = (
        "아래 텍스트는 반려동물 진료/검사 결과입니다. "
        "飼い主에게 설명하듯이 한국어로 짧게 요약해 주세요.\n\n"
    )
    text = payload.text or payload.s3Url or ""
    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content(prompt + text)
    summary = result.text if hasattr(result, "text") else "요약 생성 실패"

    return AnalysisResponse(summary=summary)


# -------------------------------------------------------------------
# Uvicorn 로컬 실행용
# -------------------------------------------------------------------

if __name__ == "_main_":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
