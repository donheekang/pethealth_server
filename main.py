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

import google.generativeai as genai

# =========================
# 환경변수 / 설정
# =========================

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")          # S3 버킷명 (필수)
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")          # 선택
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL")    # 선택 (CloudFront 등)

# STUB 모드 (True면 무조건 더미 데이터 리턴)
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

# Gemini 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

USE_GEMINI = bool(GEMINI_API_KEY) and not STUB_MODE

gemini_model = None
if USE_GEMINI:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        # 설정 실패 시 그냥 STUB 모드처럼 동작하게
        print("Gemini 초기화 실패:", e)
        gemini_model = None
        USE_GEMINI = False

# =========================
# S3 클라이언트
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
    업로드된 S3 객체에 접근할 수 있는 URL 생성
    """
    if AWS_S3_PUBLIC_BASE_URL:
        return f"{AWS_S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    if AWS_S3_ENDPOINT_URL:
        return f"{AWS_S3_ENDPOINT_URL.rstrip('/')}/{key}"
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload_image_to_s3(file_bytes: bytes, filename: str) -> str:
    """
    이미지를 S3에 업로드하고 접근 URL을 리턴
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
        ACL="private",
    )

    return build_public_url(key)

# =========================
# Pydantic 모델
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
# Gemini 분석 로직
# =========================


def _clean_json_text(text: str) -> str:
    """
    ⁠ json ...  ⁠ 같은 코드 블록이 붙어 있으면 안쪽만 추출
    """
    text = text.strip()
    if text.startswith("⁠  "):
        #  ⁠json\n ... \n⁠  형태 제거
        parts = text.split("  ⁠")
        # parts[1]이나 2 사이에 내용이 들어있음
        for part in parts:
            part = part.strip()
            if part and not part.lower().startswith("json"):
                return part
    return text


def analyze_with_gemini(image_bytes: bytes) -> (ReceiptParsed, str):
    """
    Gemini로 영수증 이미지를 분석해서 ReceiptParsed 객체와 notes를 리턴.
    실패하면 예외를 던진다.
    """
    if not gemini_model:
        raise RuntimeError("Gemini 모델이 초기화되지 않았습니다.")

    prompt = """
다음 이미지는 반려동물 병원 영수증입니다.

아래 JSON 스키마에 정확히 맞게, *순수 JSON만* 반환하세요. 불필요한 설명, 코드블록, 주석, 자연어는 절대 포함하지 마세요.

{
  "clinicName": "병원 이름 (string, 없으면 null)",
  "visitDate": "내원일 (YYYY-MM-DD, 없으면 null)",
  "diseaseName": "질병명 또는 진료명 요약 (string, 없으면 null)",
  "symptomsSummary": "반려동물 증상/메모 요약 (string, 없으면 null)",
  "items": [
    {
      "name": "항목명 (예: 진료비, 혈액검사, 약제비 등)",
      "price": 20000   // 숫자, 원 단위. 모르면 null
    }
  ],
  "totalAmount": 35000   // 최종 결제금액, 숫자. 모르면 null
}

※ price, totalAmount 는 쉼표(,) 없는 정수로만 적어주세요.
※ 값이 확실하지 않으면 null 로 두세요.
"""

    # 이미지 + 텍스트 프롬프트 같이 전달
    parts = [
        {"mime_type": "image/jpeg", "data": image_bytes},
        prompt,
    ]

    response = gemini_model.generate_content(parts)
    text = response.text or ""
    cleaned = _clean_json_text(text)

    data = json.loads(cleaned)

    items: List[ReceiptItem] = []
    for it in data.get("items", []):
        if not isinstance(it, dict):
            continue
        items.append(
            ReceiptItem(
                name=str(it.get("name", "")).strip(),
                price=it.get("price"),
            )
        )

    parsed = ReceiptParsed(
        clinicName=data.get("clinicName"),
        visitDate=data.get("visitDate"),
        diseaseName=data.get("diseaseName"),
        symptomsSummary=data.get("symptomsSummary"),
        items=items,
        totalAmount=data.get("totalAmount"),
    )

    return parsed, "Gemini 분석 결과입니다."


def build_stub_parsed() -> ReceiptParsed:
    """
    Gemini 미사용/실패 시 사용할 더미 데이터
    """
    return ReceiptParsed(
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

# =========================
# FastAPI 앱
# =========================


app = FastAPI(
    title="PetHealth+ Backend",
    description="반려동물 영수증 분석 / 기록 저장용 API",
    version="0.2.0",
)

# CORS (iOS 앱에서 접근)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요시 도메인 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "s3": bool(AWS_S3_BUCKET),
        "gemini_enabled": USE_GEMINI,
        "model": GEMINI_MODEL_NAME if USE_GEMINI else None,
    }


@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 연결 성공 ✅",
        "stub": STUB_MODE,
        "s3": bool(AWS_S3_BUCKET),
        "gemini_enabled": USE_GEMINI,
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
    iOS에서 업로드한 영수증 이미지를 받아:
    1) S3에 원본 이미지 업로드
    2) (가능하면) Gemini로 실제 분석
    3) 실패/미설정 시 STUB 데이터 반환
    """

    # 1) 파일 읽기
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="이미지 용량이 너무 큽니다. (15MB 이하)")

    # 2) 이미지 가공 없이 그대로 사용
    image_bytes = raw

    # 3) S3 업로드
    try:
        s3_url = upload_image_to_s3(image_bytes, file.filename)
    except Exception as e:
        # S3 실패해도 분석은 계속 진행할 수 있게 더미 URL 사용
        print("S3 업로드 실패:", e)
        s3_url = f"https://dummy-s3.pethealthplus/{petId}/{file.filename}"

    # 4) 분석 (Gemini 또는 STUB)
    if USE_GEMINI:
        try:
            parsed, notes = analyze_with_gemini(image_bytes)
        except Exception as e:
            print("Gemini 분석 실패:", e)
            parsed = build_stub_parsed()
            notes = f"Gemini 분석 실패로 STUB 데이터를 반환했습니다: {e}"
    else:
        parsed = build_stub_parsed()
        notes = "Gemini API 미설정 또는 STUB_MODE 활성화로 더미 데이터를 반환했습니다."

    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=notes,
    )
