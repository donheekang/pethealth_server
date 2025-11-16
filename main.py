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

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")                 # 필수: S3 버킷명
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")     # 기본 서울 리전
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")     # 선택: 커스텀 엔드포인트
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL")  # 선택: CloudFront 등 퍼블릭 도메인

# (지금은 아직 안 쓰지만, 나중에 Gemini 붙일 때 쓸 예정)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================
# S3 클라이언트 준비
# =========================

def get_s3_client():
    """
    S3 사용 시 쓸 클라이언트 생성.
    환경변수에 버킷이 없으면 None 리턴해서 'S3 없이도 앱이 떠 있게' 만든다.
    """
    if not AWS_S3_BUCKET:
        return None

    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4"),
        endpoint_url=AWS_S3_ENDPOINT_URL or None,
    )
    return client


s3_client = get_s3_client()


def build_public_url(key: str) -> str:
    """
    업로드된 S3 객체에 접근 가능한 URL 생성
    - AWS_S3_PUBLIC_BASE_URL 있으면 그걸 기준으로
    - 없으면 일반 S3 URL 형식으로
    """
    if AWS_S3_PUBLIC_BASE_URL:
        return f"{AWS_S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"

    if AWS_S3_ENDPOINT_URL:
        # 예: Minio 같은 커스텀 엔드포인트
        return f"{AWS_S3_ENDPOINT_URL.rstrip('/')}/{key}"

    # 기본 S3 퍼블릭 URL
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload_image_to_s3(file_bytes: bytes, filename: str) -> str:
    """
    이미지를 S3에 업로드하고, 접근 URL을 돌려준다.
    S3가 설정 안 돼 있으면 예외를 던지기보다는 "더미 URL"을 반환한다.
    """
    if not s3_client or not AWS_S3_BUCKET:
        # S3 설정이 아직 안 되어 있어도 서버는 떠야 하니까,
        # 임시 dummy URL을 반환한다.
        dummy_key = f"dummy/{uuid.uuid4().hex}_{filename}"
        return f"https://dummy-s3.pethealthplus/{dummy_key}"

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
        ACL="private",  # 필요에 따라 public-read 로 변경 가능
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
    version="0.2.0",
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
    return {
        "status": "ok",
        "s3": bool(AWS_S3_BUCKET),
        "gemini_key": bool(GEMINI_API_KEY),
    }


@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 연결 성공 ✅",
        "hint": "현재 영수증 분석은 STUB 데이터(테스트용)로 응답합니다.",
        "s3": bool(AWS_S3_BUCKET),
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
    iOS에서 업로드한 영수증 이미지를 받아서:
    - (현재) S3 업로드 시도 + 더미 파싱 결과 반환
    - (향후) Gemini 멀티모달로 실제 텍스트/질병/항목 분석 예정
    """

    # 1) 파일 읽기
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="이미지 용량이 너무 큽니다. (15MB 이하)")

    # 2) 이미지인지 간단히 검증 + JPEG로 정규화
    try:
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        image_bytes = buf.read()
    except Exception:
        # 이미지가 아니거나 변환에 실패한 경우: 원본 그대로 업로드
        image_bytes = raw

    # 3) S3 업로드 (실패하면 더미 URL 사용)
    try:
        s3_url = upload_image_to_s3(image_bytes, file.filename or "receipt.jpg")
    except Exception as e:
        print("S3 업로드 실패:", e)
        s3_url = f"https://dummy-s3.pethealthplus/{petId}/{uuid.uuid4().hex}.jpg"

    # 4) STUB 파싱 결과 (테스트용)
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

    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=(
            "⚠️ 현재는 STUB(테스트) 데이터입니다. "
            "나중에 Gemini 분석 결과로 교체될 예정입니다."
        ),
    )
