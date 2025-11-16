# main.py
import io
import os
import uuid
import json
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.client import Config
from PIL import Image

# =========================
# Gemini (Google Generative AI)
# =========================
import google.generativeai as genai

GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

if GEMINI_ENABLED and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
else:
    gemini_model = None

# =========================
# 설정 값 / 상수
# =========================

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB 제한

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")          # 필수: S3 버킷명
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")  # 선택: 커스텀 엔드포인트
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL")  # 선택: CloudFront 등

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
# Gemini 분석 헬퍼
# =========================

def analyze_receipt_with_gemini(image_bytes: bytes) -> ReceiptParsed:
    """
    Gemini Vision 을 사용해 영수증 이미지를 분석하고
    ReceiptParsed 형태로 변환한다.
    문제가 생기면 예외를 던진다.
    """
    if not gemini_model:
        raise RuntimeError("Gemini 모델이 활성화되어 있지 않습니다.")

    prompt = """
너는 반려동물 병원 영수증을 분석하는 전문가야.

아래 영수증 사진을 보고, 다음 정보를 한국어로 추출해.
반드시 *순수 JSON* 만 출력하고, 주석이나 설명은 절대 붙이지 마.

JSON 스키마는 다음과 같아:

{
  "clinicName": "병원 이름(모르면 null)",
  "visitDate": "YYYY-MM-DD 형식 (모르면 null)",
  "diseaseName": "진단명 또는 증상명 (모르면 null)",
  "symptomsSummary": "증상 요약 (모르면 null)",
  "items": [
    {
      "name": "항목명 (예: 진료비, X-ray, 혈액검사)",
      "price": 12345   // 숫자, 원 단위. 모르면 null
    }
  ],
  "totalAmount": 123456  // 최종 결제 금액. 모르면 null
}

규칙:
•⁠  ⁠가격은 쉼표 없이 정수로만 써.
•⁠  ⁠날짜는 YYYY-MM-DD 형식으로 통일해. (모를 경우 null)
•⁠  ⁠JSON 이 파이썬에서 json.loads 로 바로 파싱 가능해야 한다.
"""

    response = gemini_model.generate_content(
        [
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            },
        ]
    )

    text = response.text.strip()

    # 혹시 ⁠ json ...  ⁠ 로 감싸져 있으면 제거
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    data = json.loads(text)

    # JSON -> ReceiptParsed 변환
    items = [
        ReceiptItem(
            name=item.get("name", ""),
            price=item.get("price"),
        )
        for item in (data.get("items") or [])
        if item.get("name")
    ]

    parsed = ReceiptParsed(
        clinicName=data.get("clinicName"),
        visitDate=data.get("visitDate"),
        diseaseName=data.get("diseaseName"),
        symptomsSummary=data.get("symptomsSummary"),
        items=items,
        totalAmount=data.get("totalAmount"),
    )

    return parsed

# =========================
# FastAPI 앱 생성
# =========================

app = FastAPI(
    title="PetHealth+ Backend",
    description="반려동물 영수증 분석 / 기록 저장용 API (Gemini 연동 버전)",
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
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 연결 성공 ✅",
        "gemini_enabled": bool(gemini_model),
        "s3": bool(AWS_S3_BUCKET),
    }

# =========================
# 영수증 분석 엔드포인트
# =========================


@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    iOS에서 업로드한 영수증 이미지를 받아서:
    - S3 업로드
    - Gemini Vision 으로 실제 분석 (가능하면)
    - 분석 결과 + S3 URL 반환
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
        # 이미지가 아니면 원본 그대로 업로드 시도 & Gemini 분석도 원본 사용
        image_bytes = raw

    # 3) S3 업로드 시도 (실패하면 더미 URL 사용)
    try:
        s3_url = upload_image_to_s3(image_bytes, file.filename)
    except Exception as e:
        print("S3 업로드 실패:", e)
        s3_url = f"https://dummy-s3.pethealthplus/{petId}/{file.filename}"

    # 4) Gemini로 실제 분석 시도
    parsed: Optional[ReceiptParsed] = None
    notes = None

    if gemini_model:
        try:
            parsed = analyze_receipt_with_gemini(image_bytes)
            notes = "Gemini Vision 으로 분석한 결과입니다."
        except Exception as e:
            print("Gemini 분석 실패:", e)
            notes = f"Gemini 분석 실패: {e}. STUB 데이터로 대체했습니다."

    # 5) Gemini가 실패했거나 비활성화면 STUB 데이터 사용
    if not parsed:
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
        if not notes:
            notes = "Gemini 비활성화로 STUB 데이터가 반환되었습니다."

    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=notes,
    )
