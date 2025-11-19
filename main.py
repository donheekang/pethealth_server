# main.py
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import uuid4

import boto3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# 환경 설정
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

if not S3_BUCKET_NAME:
    logger.warning("S3_BUCKET_NAME 환경변수가 설정되지 않았습니다. 업로드 시 오류가 날 수 있습니다.")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ============================================================
# FastAPI 앱 설정
# ============================================================

app = FastAPI(title="PetHealth+ Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 앱 도메인만 허용하도록 나중에 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Pydantic 모델
# ============================================================


class UploadResponse(BaseModel):
    s3_key: str
    file_url: str
    content_type: Optional[str] = None


class LabResult(BaseModel):
    id: str
    pet_id: Optional[str] = None
    title: Optional[str] = None
    file_url: str
    created_at: datetime


class Certificate(BaseModel):
    id: str
    pet_id: Optional[str] = None
    title: Optional[str] = None
    file_url: str
    created_at: datetime


# ============================================================
# S3 헬퍼 함수
# ============================================================


def _build_s3_key(prefix: str, filename: str) -> str:
    safe_name = filename.replace(" ", "_")
    return f"{prefix}/{uuid4().hex}_{safe_name}"


def upload_to_s3(file: UploadFile, prefix: str) -> Tuple[str, str]:
    """
    파일을 S3에 업로드하고, presigned URL을 반환.
    """
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME not configured")

    key = _build_s3_key(prefix, file.filename)
    try:
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": file.content_type or "application/octet-stream"},
        )
    except Exception as e:
        logger.exception("S3 업로드 실패")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": key},
            ExpiresIn=60 * 60 * 24 * 7,  # 7일
        )
    except Exception as e:
        logger.exception("Presigned URL 생성 실패")
        raise HTTPException(status_code=500, detail=f"Failed to create presigned URL: {e}")

    return key, url


# ============================================================
# 기본 헬스체크 / 상태 확인
# ============================================================


@app.get("/")
async def root():
    """
    Render 브라우저에서 확인용 루트 엔드포인트.
    """
    return {"status": "ok", "stubMode": STUB_MODE}


@app.get("/health")
async def health():
    """
    단순 헬스체크 (텍스트)
    """
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    """
    앱에서 사용하는 JSON 헬스체크
    """
    return {"status": "ok", "stubMode": STUB_MODE}


# ============================================================
# 업로드 엔드포인트
#  - 영수증(이미지)
#  - 검사결과(PDF)
#  - 증명서(PDF)
# ============================================================


@app.post("/api/receipt/upload", response_model=UploadResponse)
async def upload_receipt(file: UploadFile = File(...)):
    """
    병원 영수증 (이미지) 업로드용 엔드포인트.
    """
    key, url = upload_to_s3(file, "receipts")
    logger.info("Receipt uploaded: %s", key)
    return UploadResponse(s3_key=key, file_url=url, content_type=file.content_type)


@app.post("/api/lab/upload", response_model=UploadResponse)
async def upload_lab_result(file: UploadFile = File(...)):
    """
    검사결과 (PDF) 업로드용 엔드포인트.
    """
    if not file.content_type or not file.content_type.lower().startswith("application/pdf"):
        logger.warning("검사결과 업로드 시 PDF가 아님: %s", file.content_type)
    key, url = upload_to_s3(file, "lab_results")
    logger.info("Lab result uploaded: %s", key)
    return UploadResponse(s3_key=key, file_url=url, content_type=file.content_type)


@app.post("/api/cert/upload", response_model=UploadResponse)
async def upload_certificate(file: UploadFile = File(...)):
    """
    증명서 (PDF) 업로드용 엔드포인트.
    """
    if not file.content_type or not file.content_type.lower().startswith("application/pdf"):
        logger.warning("증명서 업로드 시 PDF가 아님: %s", file.content_type)
    key, url = upload_to_s3(file, "certificates")
    logger.info("Certificate uploaded: %s", key)
    return UploadResponse(s3_key=key, file_url=url, content_type=file.content_type)


# ============================================================
# 검사결과 / 증명서 목록 조회 (앱 진입 시 404 방지용)
# 지금은 DB를 안 쓰고, 일단 빈 리스트만 반환.
# 나중에 RDS / DynamoDB 붙일 때 여기서 실제 데이터 리턴하면 됨.
# ============================================================


@app.get("/api/lab/list", response_model=List[LabResult])
async def list_lab_results() -> List[LabResult]:
    """
    검사결과 탭 진입 시 호출되는 목록 API.

    현재는 DB가 없으므로, 404 대신 항상 빈 배열([])을 반환해서
    앱에서 "서버 오류" 팝업이 뜨지 않도록만 처리.
    """
    return []


@app.get("/api/cert/list", response_model=List[Certificate])
async def list_cert_results() -> List[Certificate]:
    """
    증명서 탭 진입 시 호출되는 목록 API (위와 동일한 개념).
    """
    return []


# 혹시 예전 버전 앱이 /api 없이 호출하고 있을 경우를 대비한 레거시 경로
@app.get("/lab/list", response_model=List[LabResult])
async def list_lab_results_legacy() -> List[LabResult]:
    return []


@app.get("/cert/list", response_model=List[Certificate])
async def list_cert_results_legacy() -> List[Certificate]:
    return []


# ============================================================
# 선택: OCR + Gemini 분석용 엔드포인트 (추후 연동)
# 현재 앱이 안 쓰고 있다면 그냥 두어도 되고,
# 나중에 연동할 때 이 부분만 확장하면 됨.
# ============================================================


class AnalyzeRequest(BaseModel):
    """
    나중에 사용할 AI 분석 요청용 모델.
    - text: 이미 추출된 텍스트
    - s3_key: S3에 올려둔 파일 키 (필요 시)
    """
    text: Optional[str] = None
    s3_key: Optional[str] = None


class AnalyzeResponse(BaseModel):
    summary: str
    raw_text: Optional[str] = None


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_with_gemini(payload: AnalyzeRequest):
    """
    (선택) 제미니를 이용한 진료/검사결과 요약 API.
    지금 당장 안 써도 되고, 시험용으로만 사용해도 됨.
    """
    if STUB_MODE or not GEMINI_ENABLED or not GEMINI_API_KEY:
        # 개발용 / 장애 시에도 에러 대신 기본 응답 주기
        return AnalyzeResponse(
            summary="STUB MODE: 실제 Gemini 분석 대신 더미 응답을 반환했습니다.",
            raw_text=payload.text or "",
        )

    if not payload.text:
        raise HTTPException(status_code=400, detail="text 또는 s3_key 중 하나는 필요합니다.")

    try:
        import google.generativeai as genai
    except Exception:
        logger.exception("google.generativeai import 실패")
        raise HTTPException(status_code=500, detail="Gemini 모듈이 설치되어 있지 않습니다.")

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "아래 텍스트는 반려동물의 진료기록/검사결과/증명서 내용입니다.\n"
            "飼い主(반려인)가 이해하기 쉽도록 한국어로 핵심 요약을 3~5줄로 정리해 주세요.\n\n"
            f"{payload.text}"
        )
        response = model.generate_content(prompt)
        summary_text = response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        logger.exception("Gemini 호출 실패")
        raise HTTPException(status_code=500, detail=f"Gemini 호출 실패: {e}")

    return AnalyzeResponse(summary=summary_text, raw_text=payload.text or "")
