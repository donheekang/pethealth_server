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

# ✅ 새로운 Gemini SDK
from google import genai
from google.genai import types as genai_types

# =========================
# 설정 값 / 상수
# =========================

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB 제한

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")          # 필수: S3 버킷명
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")  # 선택: 커스텀 엔드포인트
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL")  # 선택: CloudFront 등

# ✅ Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
# Gemini 클라이언트 준비
# =========================

gemini_client = None
GEMINI_ENABLED = False

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
        print("✅ Gemini client 초기화 성공")
    except Exception as e:
        print("❌ Gemini client 초기화 실패:", repr(e))
else:
    print("⚠️ GEMINI_API_KEY 환경변수가 없습니다. STUB 모드로 동작합니다.")


def analyze_receipt_with_gemini(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> Optional[dict]:
    """
    Gemini 2.5 Flash 로 영수증 이미지를 분석해서
    우리가 원하는 JSON 스키마(dict)로 돌려준다.
    실패하면 None 반환.
    """
    if not GEMINI_ENABLED or not gemini_client:
        return None

    try:
        prompt = (
            "이 이미지는 동물병원 영수증입니다. "
            "다음 JSON 스키마에 맞춰 한국어로 값을 채워서 'JSON만' 반환해 주세요.\n\n"
            "{\n"
            '  \"clinicName\": \"병원 이름 (string)\",\n'
            '  \"visitDate\": \"방문일 (YYYY-MM-DD 형식, 없으면 null)\",\n'
            '  \"diseaseName\": \"진단명 또는 주요 질환명 (string 또는 null)\",\n'
            '  \"symptomsSummary\": \"증상 및 메모 요약 (string 또는 null)\",\n'
            '  \"items\": [\n'
            '    { \"name\": \"항목 이름\", \"price\": 금액(원, integer 또는 null) }\n'
            "  ],\n"
            '  \"totalAmount\": 총합 금액(원, integer 또는 null)\n'
            "}\n\n"
            "설명 문장은 쓰지 말고, 반드시 위 형태의 JSON만 출력하세요."
        )

        image_part = genai_types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type or "image/jpeg",
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        text = response.text or ""
        data = json.loads(text)  # JSON 파싱 실패 시 예외 → except 로 감

        # 최소한 dict 인지만 체크
        if not isinstance(data, dict):
            raise ValueError("Gemini 응답이 dict 형식이 아닙니다.")

        return data

    except Exception as e:
        print("❌ Gemini 분석 실패:", repr(e))
        return None

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
    description="반려동물 영수증 분석 / 기록 저장용 API (Gemini 2.5 연결 버전)",
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
        "gemini": GEMINI_ENABLED,
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
    - Gemini 2.5 Flash로 실제 분석 시도
    - 실패 시 STUB 데이터 반환
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
        mime_type = "image/jpeg"
    except Exception:
        # 이미지가 아니면 원본 그대로 업로드, Gemini 분석도 skip
        image_bytes = raw
        mime_type = file.content_type or "application/octet-stream"

    # 3) S3 업로드 시도 (실패하면 더미 URL 사용)
    try:
        s3_url = upload_image_to_s3(image_bytes, file.filename)
    except Exception as e:
        print("S3 업로드 실패:", e)
        s3_url = f"https://dummy-s3.pethealthplus/{petId}/{file.filename}"

    # 4) 기본 STUB 파싱 결과 (fallback)
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
    notes = "현재는 STUB 데이터입니다."

    # 5) Gemini 실제 분석 시도
    gemini_info = analyze_receipt_with_gemini(image_bytes, mime_type=mime_type)

    if gemini_info:
        try:
            parsed = ReceiptParsed(
                clinicName=gemini_info.get("clinicName"),
                visitDate=gemini_info.get("visitDate"),
                diseaseName=gemini_info.get("diseaseName"),
                symptomsSummary=gemini_info.get("symptomsSummary"),
                items=[
                    ReceiptItem(
                        name=item.get("name", ""),
                        price=item.get("price"),
                    )
                    for item in gemini_info.get("items", [])
                    if isinstance(item, dict)
                ],
                totalAmount=gemini_info.get("totalAmount"),
            )
            notes = "Gemini 2.5 Flash 실제 분석 결과입니다."
        except Exception as e:
            print("Gemini 결과 파싱 중 오류:", repr(e))
            notes = "Gemini 분석은 호출되었으나, 파싱 오류로 STUB 데이터를 사용했습니다."

    elif GEMINI_ENABLED:
        notes = "Gemini 호출에 실패하여 STUB 데이터를 사용했습니다. (서버 로그 참고)"
    else:
        notes = "Gemini 비활성화 상태입니다. GEMINI_API_KEY 환경변수를 확인하세요."

    # 6) 최종 응답
    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=notes,
    )

# =========================
# 간단 Gemini 디버그용 엔드포인트
# =========================

@app.get("/debug/gemini")
async def debug_gemini():
    """
    브라우저에서 /debug/gemini 열어서
    Gemini 호출이 되는지 빠르게 확인할 수 있는 엔드포인트
    """
    if not GEMINI_ENABLED or not gemini_client:
        return {
            "enabled": False,
            "message": "Gemini 비활성화. GEMINI_API_KEY 환경변수를 확인하세요.",
        }

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="짧게 한 문장으로만 대답해줘. PetHealth+ 서버에서 Gemini 테스트 중이야.",
        )
        return {
            "enabled": True,
            "text": resp.text,
        }
    except Exception as e:
        return {
            "enabled": True,
            "error": str(e),
        }
