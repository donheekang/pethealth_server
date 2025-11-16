import io
import os
import json
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.client import Config
from PIL import Image

# Gemini
import google.generativeai as genai

# ---------------------------
# 환경 변수
# ---------------------------
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"

if GEMINI_ENABLED and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------
# S3 클라이언트
# ---------------------------
def get_s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4")
    )

# ---------------------------
# 모델 정의
# ---------------------------
class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None

class ReceiptParsed(BaseModel):
    clinicName: Optional[str]
    visitDate: Optional[str]
    diseaseName: Optional[str]
    symptomsSummary: Optional[str]
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None

class ReceiptAnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: Optional[str] = None


# ---------------------------
# Gemini 분석 함수
# ---------------------------
def analyze_receipt_with_gemini(image_bytes: bytes) -> ReceiptParsed:
    if not (GEMINI_ENABLED and GEMINI_API_KEY):
        raise RuntimeError("Gemini 비활성화")

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = """
    너는 동물병원 영수증을 분석하는 한국어 비서야.
    아래 영수증 이미지를 보고 아래 JSON 형식으로만 출력해.

    {
      "clinicName": "...",
      "visitDate": "YYYY-MM-DD 또는 null",
      "diseaseName": "... 또는 null",
      "symptomsSummary": "... 또는 null",
      "items": [
        {"name": "...", "price": 숫자 또는 null}
      ],
      "totalAmount": 숫자 또는 null
    }

    반드시 JSON만 반환하고 설명 문장을 쓰지 마.
    금액은 숫자만 (원, 콤마 제거).
    """

    response = model.generate_content([
        prompt,
        {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
    ])

    text = response.text.strip()
    data = json.loads(text)

    items = []
    for it in data.get("items", []):
        if it.get("name"):
            items.append(
                ReceiptItem(
                    name=it.get("name"),
                    price=it.get("price")
                )
            )

    return ReceiptParsed(
        clinicName=data.get("clinicName"),
        visitDate=data.get("visitDate"),
        diseaseName=data.get("diseaseName"),
        symptomsSummary=data.get("symptomsSummary"),
        items=items,
        totalAmount=data.get("totalAmount")
    )

# ---------------------------
# S3 업로드 함수
# ---------------------------
def upload_image_to_s3(image_bytes: bytes, filename: str) -> str:
    s3 = get_s3_client()

    key = f"receipts/{uuid.uuid4()}.jpg"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType="image/jpeg",
        ACL="public-read"
    )

    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 영수증 분석 API
# ---------------------------
@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str,
    file: UploadFile = File(...)
):

    # 파일 읽기
    content = await file.read()

    # JPEG 변환
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
    except:
        raise HTTPException(status_code=400, detail="이미지 처리 실패")

    # S3 업로드
    try:
        s3_url = upload_image_to_s3(image_bytes, file.filename)
    except Exception as e:
        print("S3 업로드 실패:", e)
        s3_url = f"https://dummy-s3/{uuid.uuid4()}.jpg"

    # AI 분석
    try:
        if GEMINI_ENABLED and GEMINI_API_KEY:
            parsed = analyze_receipt_with_gemini(image_bytes)
            notes = "Gemini 분석 성공"
        else:
            raise RuntimeError("Gemini 비활성화")

    except Exception as e:
        print("Gemini 오류, STUB 사용:", e)
        parsed = ReceiptParsed(
            clinicName="(테스트) PetHealth+ 클리닉",
            visitDate=None,
            diseaseName=None,
            symptomsSummary="AI 분석 실패로 기본값 저장됨",
            items=[ReceiptItem(name="진료비", price=None)],
            totalAmount=None
        )
        notes = "STUB 결과"

    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=notes
    )
