# main.py
import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.client import Config

import google.generativeai as genai

# ==============================
# 환경 변수
# ==============================
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ==============================
# FastAPI 앱 & CORS
# ==============================
app = FastAPI(title="PetHealth+ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 나중에 도메인 제한하고 싶으면 여기 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# 공통 모델 (iOS에서 그대로 받기 좋게)
# ==============================
class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsed(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None      # "YYYY-MM-DD" 또는 null
    diseaseName: Optional[str] = None
    symptomsSummary: Optional[str] = None
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None


class ReceiptResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: Optional[str] = None          # STUB / 분석 설명


class FileAnalysisResponse(BaseModel):
    petId: Optional[str] = None
    kind: str                            # "lab" / "certificate"
    s3Url: str
    summary: str
    recommendation: Optional[str] = None


# ==============================
# S3 유틸
# ==============================
def get_s3_client():
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME):
        return None
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def upload_to_s3(folder: str, filename: str, data: bytes, content_type: str) -> str:
    """
    S3에 업로드하고, 접근 가능한 URL을 리턴.
    STUB_MODE일 때는 실제 업로드 없이 예시 URL 리턴.
    """
    if STUB_MODE or not S3_BUCKET_NAME:
        return f"https://example.com/{folder}/{filename}"

    client = get_s3_client()
    if client is None:
        raise HTTPException(status_code=500, detail="S3 설정이 잘못되었습니다.")

    key = f"{folder}/{filename}"
    client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )

    return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"


# ==============================
# Gemini 유틸
# ==============================
def _clean_json_text(text: str) -> str:
    """
    ⁠ json ...  ⁠ 같은 코드블록을 제거하고 순수 JSON만 남긴다.
    """
    cleaned = text.strip()
    if cleaned.startswith("⁠  "):
        parts = cleaned.split("  ⁠")
        if len(parts) >= 2:
            cleaned = parts[1]
    cleaned = cleaned.replace("⁠  json", "").replace("  ⁠", "")
    return cleaned.strip()


def analyze_receipt_with_gemini(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    영수증 이미지를 Gemini로 분석해서 구조화된 JSON을 돌려받는다.
    clinicName / visitDate / diseaseName / symptomsSummary / items[] / totalAmount
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key가 설정되지 않았습니다.")

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = """
당신은 동물병원 영수증을 분석하는 어시스턴트입니다.
이미지를 보고 아래 정보를 한국어로 JSON 형식으로만 반환하세요.

목표:
•⁠  ⁠보호자가 "어떤 병 때문에, 어떤 치료를 받고, 얼마 썼는지" 한눈에 볼 수 있도록 정리합니다.

규칙:
•⁠  ⁠JSON 이외의 텍스트는 절대 쓰지 마세요.
•⁠  ⁠금액은 '원' 같은 단위를 제거한 정수 숫자로만 적으세요.
•⁠  ⁠날짜가 영수증에 보이면 YYYY-MM-DD 형식으로 적고, 없으면 null 로 적으세요.
•⁠  ⁠질병명이 명확하면 질병명을 쓰고, 애매하면 "기타 피부질환", "위장관 질환 추정"과 같이 카테고리 수준으로 적어도 됩니다.
•⁠  ⁠증상 요약은 1~2문장으로, 보호자가 이해하기 쉽게 적으세요.

반환 JSON 스키마는 다음과 같습니다:

{
  "clinicName": "병원 이름 (모르면 null)",
  "visitDate": "YYYY-MM-DD 또는 null",
  "diseaseName": "주된 질환/진료 카테고리 (예: 피부염, 위장염, 관절질환 등) 또는 null",
  "symptomsSummary": "증상/내원 사유 요약 (한두 문장)",
  "items": [
    { "name": "항목명", "price": 15000 }
  ],
  "totalAmount": 37000
}
"""

    result = model.generate_content(
        [
            prompt,
            {"mime_type": mime_type, "data": image_bytes},
        ]
    )

    cleaned = _clean_json_text(result.text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 파싱 실패 시 최소 구조 반환
        return {
            "clinicName": None,
            "visitDate": None,
            "diseaseName": None,
            "symptomsSummary": None,
            "items": [],
            "totalAmount": None,
        }


def analyze_pdf_with_gemini(pdf_bytes: bytes, kind: str) -> Dict[str, str]:
    """
    검사결과/증명서 PDF를 Gemini로 요약.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key가 설정되지 않았습니다.")

    model = genai.GenerativeModel("gemini-1.5-flash")

    role = "수의학 검사결과" if kind == "lab" else "동물 관련 증명서(예: 예방접종 증명서 등)"

    prompt = f"""
다음 PDF는 반려동물의 {role} 입니다.
보호자가 이해하기 쉽게 핵심 내용을 요약해 주세요.

1) '한줄 요약' (한두 문장)
2) '검사/내용 핵심 포인트'
3) '보호자가 앞으로 어떻게 관리하면 좋은지' 간단한 행동 가이드

응답은 아래 JSON 형식으로만 해주세요.

{{
  "summary": "핵심 요약 한두 문장",
  "recommendation": "보호자 행동 가이드"
}}
"""

    result = model.generate_content(
        [
            prompt,
            {"mime_type": "application/pdf", "data": pdf_bytes},
        ]
    )

    cleaned = _clean_json_text(result.text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {
            "summary": "PDF 내용을 요약하는 데 실패했습니다.",
            "recommendation": "수의사와 상의하여 해석을 받는 것이 좋습니다.",
        }
    return data


# ==============================
# STUB 헬퍼 (개발/테스트용)
# ==============================
def stub_receipt_parsed() -> Dict[str, Any]:
    today = datetime.now().date().isoformat()
    return {
        "clinicName": "토리동물병원",
        "visitDate": today,
        "diseaseName": "피부염",
        "symptomsSummary": "가려움과 피부 발진으로 내원한 기록입니다.",
        "items": [
            {"name": "진찰료", "price": 15000},
            {"name": "피부약", "price": 22000},
        ],
        "totalAmount": 37000,
    }


def stub_file_summary(kind: str) -> Dict[str, str]:
    if kind == "lab":
        return {
            "summary": "혈액 검사 결과 전반적으로 양호하나, 간 수치가 약간 높게 나왔습니다.",
            "recommendation": "기름진 간식을 줄이고 1~2개월 후 재검사를 권장합니다.",
        }
    else:
        return {
            "summary": "예방접종이 일정에 맞게 잘 완료되었음을 확인했습니다.",
            "recommendation": "다음 접종 일정에 맞춰 병원을 방문해 주세요.",
        }


# ==============================
# 기본 라우트
# ==============================
@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 연결 성공 ✅",
        "stubMode": STUB_MODE,
        "usesGemini": bool(GEMINI_API_KEY),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc),
        "stubMode": STUB_MODE,
    }


# ==============================
# 1) 영수증 이미지 분석
# ==============================
@app.post("/api/receipt/analyze", response_model=ReceiptResponse)
async def upload_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    # 파일 이름 & 확장자
    ext = "jpg"
    if file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()

    key = f"{uuid.uuid4()}.{ext}"
    s3_url = upload_to_s3("receipts", key, data, file.content_type)

    # 분석
    if STUB_MODE:
        parsed_dict = stub_receipt_parsed()
        notes = "STUB_MODE: 실제 Gemini 대신 예시 데이터가 반환되었습니다."
    else:
        parsed_dict = analyze_receipt_with_gemini(data, file.content_type)
        notes = "Gemini로 영수증 내용을 분석했습니다."

    parsed = ReceiptParsed(
        clinicName=parsed_dict.get("clinicName"),
        visitDate=parsed_dict.get("visitDate"),
        diseaseName=parsed_dict.get("diseaseName"),
        symptomsSummary=parsed_dict.get("symptomsSummary"),
        items=[ReceiptItem(**it) for it in parsed_dict.get("items", [])],
        totalAmount=parsed_dict.get("totalAmount"),
    )

    return ReceiptResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=notes,
    )


# ==============================
# 2) 검사결과/증명서 PDF 분석
# ==============================
@app.post("/api/file/analyze", response_model=FileAnalysisResponse)
async def upload_pdf(
    kind: str = Form(...),        # "lab" or "certificate"
    petId: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    kind = kind.lower()
    if kind not in ("lab", "certificate"):
        raise HTTPException(status_code=400, detail="kind는 'lab' 또는 'certificate' 여야 합니다.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    key = f"{uuid.uuid4()}.pdf"
    s3_url = upload_to_s3(f"{kind}s", key, data, "application/pdf")

    if STUB_MODE:
        analysis = stub_file_summary(kind)
    else:
        analysis = analyze_pdf_with_gemini(data, kind)

    return FileAnalysisResponse(
        petId=petId,
        kind=kind,
        s3Url=s3_url,
        summary=analysis.get("summary", ""),
        recommendation=analysis.get("recommendation"),
    )
