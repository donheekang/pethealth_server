import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

import boto3
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings

# Google / Gemini
from google.cloud import vision
from google.cloud import storage
import google.generativeai as genai


# ============================================================
# 설정
# ============================================================

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-northeast-2"
    S3_BUCKET_NAME: str

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_ENABLED: bool = False

    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

    # True 이면 실제 외부 연동 대신 더미 데이터 리턴
    STUB_MODE: bool = False

    class Config:
        env_file = ".env"


settings = Settings()

# 로거
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 외부 클라이언트 (S3 / Vision / Gemini)
# ============================================================

# S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)

# Google Vision
vision_client: Optional[vision.ImageAnnotatorClient] = None
storage_client: Optional[storage.Client] = None

if settings.GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
    try:
        vision_client = vision.ImageAnnotatorClient()
        storage_client = storage.Client()
        logger.info("Google Cloud Vision / Storage 클라이언트 초기화 완료")
    except Exception as e:
        logger.exception("Google Cloud 클라이언트 초기화 실패: %s", e)
        vision_client = None
        storage_client = None
else:
    logger.warning("GOOGLE_APPLICATION_CREDENTIALS 미설정 - OCR 기능 비활성화")


# Gemini
if settings.GEMINI_API_KEY:
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        logger.info("Gemini 클라이언트 초기화 완료")
    except Exception as e:
        logger.exception("Gemini 초기화 실패: %s", e)
else:
    logger.warning("GEMINI_API_KEY 미설정 - Gemini 분석 기능 비활성화")


# ============================================================
# FastAPI 앱
# ============================================================

app = FastAPI(title="PetHealth+ Server", version="1.0.0")

# iOS 앱에서 호출하니 CORS 넉넉하게 열어둠
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 앱 도메인으로 좁혀도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 공통 모델
# ============================================================

class UploadResponse(BaseModel):
    s3_key: str
    file_url: str
    content_type: Optional[str] = None


class AnalyzeRequest(BaseModel):
    s3_key: str       # S3에 저장된 파일 키
    doc_type: str     # "receipt" | "lab" | "certificate" 등 자유롭게


class AnalyzeResponse(BaseModel):
    ocr_text: str
    summary: str
    doc_type: str


# ============================================================
# 헬퍼 함수
# ============================================================

def upload_to_s3(file: UploadFile, prefix: str) -> tuple[str, str]:
    """
    S3에 파일 업로드 후 (key, presigned_url) 리턴
    """
    if settings.STUB_MODE:
        # 실제로는 업로드하지 않고 테스트용 값만 리턴
        fake_key = f"{prefix}/stub-{uuid.uuid4()}.pdf"
        fake_url = f"https://example.com/{fake_key}"
        logger.info("[STUB] upload_to_s3 호출: %s", fake_key)
        return fake_key, fake_url

    content_type = file.content_type or "application/octet-stream"
    ext = ".pdf"
    key = f"{prefix}/{uuid.uuid4().hex}{ext}"

    logger.info("S3 업로드 시작: bucket=%s, key=%s", settings.S3_BUCKET_NAME, key)

    try:
        s3_client.upload_fileobj(
            file.file,
            settings.S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )
    except Exception as e:
        logger.exception("S3 업로드 실패: %s", e)
        raise HTTPException(status_code=500, detail="S3 업로드 실패")

    # 7일짜리 presigned URL
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=int(timedelta(days=7).total_seconds()),
        )
    except Exception as e:
        logger.exception("S3 presigned URL 생성 실패: %s", e)
        raise HTTPException(status_code=500, detail="S3 URL 생성 실패")

    return key, url


def run_ocr_from_s3(s3_key: str) -> str:
    """
    S3에서 파일을 읽어서 Google Vision OCR 수행.
    """
    if settings.STUB_MODE or not vision_client:
        logger.warning("OCR STUB 동작 (실제 Vision 호출 안 함)")
        return f"[STUB OCR] s3://{settings.S3_BUCKET_NAME}/{s3_key}"

    try:
        obj = s3_client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        content = obj["Body"].read()
    except Exception as e:
        logger.exception("S3 객체 읽기 실패: %s", e)
        raise HTTPException(status_code=500, detail="S3 파일 읽기 실패")

    try:
        image = vision.Image(content=content)
        response = vision_client.document_text_detection(image=image)

        if response.error.message:
            logger.error("Vision API 오류: %s", response.error.message)
            raise HTTPException(status_code=500, detail="OCR 처리 오류")

        text = response.full_text_annotation.text or ""
        return text.strip()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Vision OCR 실패: %s", e)
        raise HTTPException(status_code=500, detail="OCR 처리 실패")


def run_gemini_analysis(ocr_text: str, doc_type: str) -> str:
    """
    OCR 텍스트를 기반으로 Gemini에게 간단 요약 요청.
    """
    if settings.STUB_MODE or not settings.GEMINI_API_KEY:
        logger.warning("Gemini STUB 동작 (실제 분석 안 함)")
        return f"[STUB SUMMARY] type={doc_type}, length={len(ocr_text)}"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
다음은 반려동물 관련 문서({doc_type})의 OCR 결과입니다.
주요 정보(날짜, 동물 이름, 병원명, 검사명/진료내용, 금액 등)를 짧게 bullet 형식으로 요약해 주세요.

텍스트:
{ocr_text}
"""
        resp = model.generate_content(prompt)
        return resp.text.strip() if resp.text else ""
    except Exception as e:
        logger.exception("Gemini 분석 실패: %s", e)
        raise HTTPException(status_code=500, detail="Gemini 분석 실패")


# ============================================================
# 헬스체크 / 기본 라우트
# ============================================================

@app.get("/")
async def root():
    """
    루트 확인용 (Render 브라우저에서 보는 용도)
    """
    return {"status": "ok", "stubMode": settings.STUB_MODE}


@app.get("/health")
async def health():
    """
    옛날 헬스체크 (간단 버전)
    """
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    """
    앱에서 사용하는 헬스체크 (stubMode 포함)
    """
    return {"status": "ok", "stubMode": settings.STUB_MODE}


# ============================================================
# 리스트 API (지금은 DB 없어서 더미 구현)
# iOS 쪽에서 []만 잘 받으면 되기 때문에 우선 비어있는 리스트 리턴
# ============================================================

@app.get("/api/lab/list")
async def get_lab_list():
    """
    검사결과 리스트.
    추후 DB 붙이면 실제 데이터 리턴하도록 수정.
    """
    return []


@app.get("/api/cert/list")
async def get_cert_list():
    """
    증명서 리스트.
    추후 DB 붙이면 실제 데이터 리턴하도록 수정.
    """
    return []


# ============================================================
# 업로드 API (PDF)
#   - /api/lab/upload
#   - /api/cert/upload
#   - /lab/upload  (레거시, 앱이 이 경로를 부를 수도 있음)
#   - /cert/upload (레거시)
# ============================================================

@app.post("/api/lab/upload", response_model=UploadResponse)
async def upload_lab_result(file: UploadFile = File(...)):
    """
    검사결과 PDF 업로드
    """
    if not file.content_type or not file.content_type.lower().startswith("application/pdf"):
        logger.warning("검사결과 업로드: PDF가 아닌 content_type=%s", file.content_type)

    key, url = upload_to_s3(file, "lab_results")
    logger.info("Lab result uploaded: %s", key)
    return UploadResponse(s3_key=key, file_url=url, content_type=file.content_type)


@app.post("/api/cert/upload", response_model=UploadResponse)
async def upload_certificate(file: UploadFile = File(...)):
    """
    증명서 PDF 업로드
    """
    if not file.content_type or not file.content_type.lower().startswith("application/pdf"):
        logger.warning("증명서 업로드: PDF가 아닌 content_type=%s", file.content_type)

    key, url = upload_to_s3(file, "certificates")
    logger.info("Certificate uploaded: %s", key)
    return UploadResponse(s3_key=key, file_url=url, content_type=file.content_type)


# ---- 레거시 경로 (앱이 /lab/upload, /cert/upload 를 사용할 때 대비) ----

@app.post("/lab/upload", response_model=UploadResponse)
async def upload_lab_result_legacy(file: UploadFile = File(...)):
    """
    레거시: /lab/upload → /api/lab/upload 와 동일 로직
    """
    return await upload_lab_result(file)


@app.post("/cert/upload", response_model=UploadResponse)
async def upload_certificate_legacy(file: UploadFile = File(...)):
    """
    레거시: /cert/upload → /api/cert/upload 와 동일 로직
    """
    return await upload_certificate(file)


# ============================================================
# OCR + Gemini 분석 API
#   - S3에 이미 올라간 파일 기준으로 분석
# ============================================================

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_document(req: AnalyzeRequest):
    """
    S3에 저장된 PDF/이미지를 OCR & Gemini 로 분석.
    - s3_key: upload_to_s3()에서 받은 key
    - doc_type: "receipt", "lab", "certificate" 등 자유롭게 전달
    """
    logger.info("문서 분석 요청: s3_key=%s, doc_type=%s", req.s3_key, req.doc_type)

    ocr_text = run_ocr_from_s3(req.s3_key)
    summary = run_gemini_analysis(ocr_text, req.doc_type)

    return AnalyzeResponse(
        ocr_text=ocr_text,
        summary=summary,
        doc_type=req.doc_type,
    )


# ============================================================
# 로컬 실행용
# ============================================================

if __name__ == "_main_":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
