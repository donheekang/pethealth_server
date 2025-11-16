# main.py
import io
import os
import uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.client import Config
from PIL import Image

# =========================
# 설정 값 / 상수
# =========================

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB 제한

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")          # 필수: S3 버킷명
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")  # 선택: 커스텀 엔드포인트
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL")  # 선택: CloudFront 등

# 구글 OCR / Gemini 나중에 붙일 때 사용할 환경변수 (지금은 안 씀)
GCV_ENABLED = os.getenv("GCV_ENABLED", "false").lower() == "true"
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"

# =========================
# S3 클라이언트 준비
# =========================

s3_client = None
if AWS_S3_BUCKET:
    session = boto3.session.Session()
    s3_client = session.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4"),
        endpoint_url=AWS_S3_ENDPOINT_URL or None,
    )

def build_public_url(key: str) -> str:
    """
    업로드된 S3 객체에 접근할 수 있는 URL을 생성.
    - AWS_S3_PUBLIC_BASE_URL 있으면 그걸 기준으로
    - 없으면 일반 S3 URL 형식으로
    """
    if AWS_S3_PUBLIC_BASE_URL:
        return f"{AWS_S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    if AWS_S3_ENDPOINT_URL:
        return f"{AWS_S3_ENDPOINT_URL.rstrip('/')}/{key}"
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload_image_to_s3(file_bytes: bytes, filename: str) -> str:
    """
    이미지를 S3에 업로드하고, 접근 URL을 돌려준다.
    S3가 설정 안 돼 있으면 예외를 던진다.
    """
    if not s3_client or not AWS_S3_BUCKET:
        raise RuntimeError("S3가 설정되어 있지 않습니다. AWS_S3_BUCKET 환경변수를 확인하세요.")

    # 확장자 추출 (없으면 jpg)
    ext = "jpg"
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()

    key = f"receipts/{uuid.uuid4().hex}.{ext}"

    s3_client.put_object(
        Bucket=AWS_S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=f"image/{ext}",
        ACL="private",  # 필요에 따라 public-read 등으로 변경 가능
    )

    return build_public_url(key)

# =========================
# Pydantic 응답 모델 정의
# =========================

class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None

class ReceiptParsed(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None        # "YYYY-MM-DD"
    diseaseName: Optional[str] = None
    symptomsSummary: Optional[str] = None
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None

class ReceiptAnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: Optional[str] = None

# =========================
# FastAPI 앱 생성
# =========================

app = FastAPI(
    title="PetHealth+ Backend",
    description="반려동물 영수증 분석 / 기록 저장용 API (Stub 버전)",
    version="0.1.0",
)

# CORS 설정 (iOS / 로컬 개발 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요하면 도메인 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 헬스체크 / 루트
# =========================

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 연결 성공 ✅",
        "ocr": False,              # 나중에 구글 OCR 붙이면 True 로 변경
        "gemini": False,           # 나중에 Gemini 붙이면 True 로 변경
        "s3": bool(AWS_S3_BUCKET), # S3 설정 여부
    }

# =========================
# 영수증 분석 엔드포인트 (Stub)
# =========================

@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    ⚠️ 완전 STUB 버전
    - 파일은 그냥 길이만 체크하고
    - S3 / Pillow / OCR 아무 것도 안 쓰고
    - 무조건 더미 데이터로 200 OK 응답만 보냄
    """

    # 1) 파일 읽기
    raw = await file.read()
    if len(raw) == 0:
        # 이건 400으로 돌려보내는 정상 에러 (앱에서 "빈 파일입니다" 같은 메시지 나옴)
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    # 2) STUB 파싱 결과 (그냥 예시 데이터)
    parsed = ReceiptParsed(
        clinicName="테스트동물병원",
        visitDate="2025-11-17",
        diseaseName="피부염",
        symptomsSummary="가려움, 붉은 발진",
        items=[
            ReceiptItem(name="진료비", price=20000),
            ReceiptItem(name="피부약", price=15000),
        ],
        totalAmount=35000,
    )

    # 3) S3 URL 도 그냥 가짜값
    dummy_url = f"https://dummy-s3.pethealthplus/{petId}/{file.filename}"

    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=dummy_url,
        parsed=parsed,
        notes="이 응답은 S3/OCR 없이 STUB 로 생성된 데이터입니다."
    )
