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

# Google APIs
from google.cloud import vision
import google.generativeai as genai

# =========================
# 설정 값 / 상수
# =========================

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB 제한

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL")

# Google Vision / Gemini
GCV_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 사용할 Gemini 모델 (콘솔에 나온 것 기준)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

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
    업로드된 S3 객체 접근 URL 생성
    """
    if AWS_S3_PUBLIC_BASE_URL:
        return f"{AWS_S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    if AWS_S3_ENDPOINT_URL:
        return f"{AWS_S3_ENDPOINT_URL.rstrip('/')}/{key}"
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload_image_to_s3(file_bytes: bytes, filename: str) -> str:
    """
    이미지를 S3에 업로드하고 접근 URL 반환
    """
    if not s3_client or not AWS_S3_BUCKET:
        raise RuntimeError("S3 설정이 되어 있지 않습니다. AWS_S3_BUCKET 환경변수를 확인하세요.")

    ext = "jpg"
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()

    key = f"receipts/{uuid.uuid4().hex}.{ext}"

    s3_client.put_object(
        Bucket=AWS_S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=f"image/{ext}",
        ACL="private",  # 필요하면 public-read 등으로 변경
    )

    return build_public_url(key)


# =========================
# Pydantic 응답 모델
# =========================

class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsed(BaseModel):
    clinicName: Optional[str] = None          # 병원명
    visitDate: Optional[str] = None           # "YYYY-MM-DD HH:MM" 형식 권장
    diseaseName: Optional[str] = None         # 진단명 / 질환명
    symptomsSummary: Optional[str] = None     # 증상 요약 / 진료 내용 요약
    items: List[ReceiptItem] = []             # 영수증 항목 리스트
    totalAmount: Optional[int] = None         # 총 금액 (숫자, 원 단위)


class ReceiptAnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: Optional[str] = None               # 분석 경고/메모


# =========================
# Google Vision OCR
# =========================

def run_google_ocr(image_bytes: bytes) -> str:
    """
    Google Cloud Vision OCR로 영수증 텍스트 추출
    """
    if not GCV_CREDENTIALS_JSON:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되지 않았습니다.")

    try:
        creds_dict = json.loads(GCV_CREDENTIALS_JSON)
    except json.JSONDecodeError:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS 값이 올바른 JSON 형식이 아닙니다.")

    client = vision.ImageAnnotatorClient.from_service_account_info(creds_dict)

    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API 오류: {response.error.message}")

    texts = response.text_annotations
    if not texts:
        return ""

    # 첫 번째 element가 전체 블록 텍스트
    return texts[0].description


# =========================
# Gemini로 구조화 파싱
# =========================

def parse_receipt_with_gemini(ocr_text: str) -> (ReceiptParsed, str):
    """
    OCR 텍스트를 Gemini에게 넘겨서 구조화된 영수증 정보로 변환
    반환값: (ReceiptParsed, notes)
    """
    if not GEMINI_API_KEY:
        # 키가 없으면 OCR 텍스트만 요약해서 basic 구조로 반환
        parsed = ReceiptParsed(
            clinicName=None,
            visitDate=None,
            diseaseName=None,
            symptomsSummary=ocr_text[:500] if ocr_text else "텍스트를 인식하지 못했습니다.",
            items=[],
            totalAmount=None,
        )
        return parsed, "GEMINI_API_KEY 미설정 – OCR 텍스트만 그대로 저장했습니다."

    # Gemini 클라이언트 설정 (한 번만 설정해도 되지만, idempotent 하므로 여기서 호출)
    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    system_prompt = """
너는 한국의 동물병원 영수증 OCR 텍스트를 구조화해서 JSON으로만 출력하는 도우미야.

다음 형식의 JSON만 출력해라 (설명 글, 문장 절대 금지):

{
  "clinicName": "병원 이름 또는 null",
  "visitDate": "YYYY-MM-DD HH:MM 형식 또는 날짜만 있는 경우 YYYY-MM-DD",
  "diseaseName": "진단명/질환명 또는 null",
  "symptomsSummary": "진료 내용과 주요 증상 요약 (한국어 문장)",
  "items": [
    { "name": "항목명", "price": 20000 },
    { "name": "피부약", "price": 15000 }
  ],
  "totalAmount": 35000
}

규칙:
•⁠  ⁠금액은 숫자만 사용하고 '원'이나 콤마(,)는 빼라. 예: 25,000원 -> 25000
•⁠  ⁠항목 금액을 모르면 price에 null을 넣어라.
•⁠  ⁠총 금액(totalAmount)을 찾을 수 없으면 null.
•⁠  ⁠진단명이 명확하지 않으면 diseaseName은 null로 두고, symptomsSummary에 최대한 자세히 적어라.
"""

    user_prompt = f"OCR로 인식된 영수증 전체 텍스트:\n\n{ocr_text}\n\n위 내용을 기준으로 지정한 JSON 형식만 반환해줘."

    try:
        response = model.generate_content(system_prompt + "\n\n" + user_prompt)
        raw_text = response.text.strip()

        # 혹시 ⁠ json ...  ⁠ 형식으로 줄 수도 있으니 제거
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            # "json\n{...}" 같은 형태 케이스 제거
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)

        parsed = ReceiptParsed(
            clinicName=data.get("clinicName"),
            visitDate=data.get("visitDate"),
            diseaseName=data.get("diseaseName"),
            symptomsSummary=data.get("symptomsSummary"),
            items=[
                ReceiptItem(name=item.get("name", ""), price=item.get("price"))
                for item in data.get("items", []) if item.get("name")
            ],
            totalAmount=data.get("totalAmount"),
        )

        return parsed, "Google Vision OCR + Gemini 분석 결과입니다."

    except Exception as e:
        # Gemini / JSON 파싱 실패 시, OCR 텍스트만 저장
        fallback = ReceiptParsed(
            clinicName=None,
            visitDate=None,
            diseaseName=None,
            symptomsSummary=ocr_text[:500] if ocr_text else "텍스트를 인식하지 못했습니다.",
            items=[],
            totalAmount=None,
        )
        return fallback, f"Gemini 분석에 실패하여 OCR 텍스트만 저장했습니다: {e}"


# =========================
# FastAPI 앱
# =========================

app = FastAPI(
    title="PetHealth+ Backend",
    description="반려동물 영수증 분석 / 기록 저장용 API (Vision OCR + Gemini)",
    version="0.2.0",
)

# CORS (iOS 앱 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 도메인 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "s3": bool(AWS_S3_BUCKET),
        "vision": bool(GCV_CREDENTIALS_JSON),
        "gemini": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL_NAME if GEMINI_API_KEY else None,
    }


@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 연결 성공 ✅",
        "detail": "영수증 이미지를 업로드하면 Google Vision OCR + Gemini로 분석해줍니다.",
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
    1) 이미지 정규화 (JPEG)
    2) S3 업로드
    3) Google Vision OCR로 텍스트 추출
    4) Gemini로 병원명/날짜/진단/항목/총액 파싱
    5) 구조화된 JSON 응답
    """

    # 1) 파일 읽기 + 용량 체크
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="이미지 용량이 너무 큽니다. (15MB 이하)")

    # 2) 이미지인지 검증 후 JPEG로 정규화
    try:
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        image_bytes = buf.read()
    except Exception:
        # 이미지가 아니면 원본 그대로 사용 (그래도 Vision이 시도는 해볼 수 있음)
        image_bytes = raw

    # 3) S3 업로드 (실패해도 OCR/분석은 시도)
    try:
        s3_url = upload_image_to_s3(image_bytes, file.filename)
    except Exception as e:
        print("S3 업로드 실패:", e)
        s3_url = f"https://dummy-s3.pethealthplus/{petId}/{file.filename}"

    # 4) Vision OCR
    try:
        ocr_text = run_google_ocr(image_bytes)
    except Exception as e:
        # Vision 자체가 안 되면 더 이상 진행 불가
        raise HTTPException(
            status_code=500,
            detail=f"Google Vision OCR 오류: {e}",
        )

    # 5) Gemini로 구조화 파싱
    parsed, notes = parse_receipt_with_gemini(ocr_text)

    # 6) 최종 응답
    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=notes,
    )
