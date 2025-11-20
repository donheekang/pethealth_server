from __future__ import annotations
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import uuid
from google.cloud import vision
import google.generativeai as genai

# =========================
# 설정 로딩
# =========================
USE_GEMINI = os.getenv("GEMINI_ENABLED", "false").lower() == "true"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if USE_GEMINI and GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

if GOOGLE_CREDENTIALS:
    vision_client = vision.ImageAnnotatorClient()

# =========================
# FastAPI 객체
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# OCR + 파싱 모델
# =========================

class ReceiptItem(BaseModel):
    name: str
    price: int | None

class ReceiptParsed(BaseModel):
    clinicName: str | None
    visitDate: str | None     # 2025-11-20 형태
    visitTime: str | None     # 12:01 형태
    items: list[ReceiptItem]
    totalAmount: int | None

class AnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: str | None = None

# =========================
# OCR → 텍스트 추출 (Vision OCR)
# =========================
def run_vision_ocr(image_bytes: bytes) -> str:
    img = vision.Image(content=image_bytes)
    res = vision_client.text_detection(image=img)
    if res.error.message:
        return ""
    return res.full_text_annotation.text


# =========================
# Gemini로 영수증 파싱
# =========================
PROMPT = """
너는 동물병원 영수증 텍스트를 구조화하는 AI다.
입력된 OCR 텍스트에서 다음 정보만 JSON으로 추출해라.

필수:
•⁠  ⁠clinicName: 병원 이름
•⁠  ⁠visitDate: yyyy-MM-dd 형식으로 날짜 (없으면 null)
•⁠  ⁠visitTime: HH:mm 형식으로 시간 (없으면 null)
•⁠  ⁠items: [
    { "name": 항목명, "price": 숫자 or null }
]
•⁠  ⁠totalAmount: 총 금액 (숫자 or null)

항목이 여러 줄이라도 정확히 리스트로 만들어라.
주어지지 않은 값은 null.
반드시 JSON만 출력.
"""

def parse_with_gemini(text: str) -> ReceiptParsed:
    model = genai.GenerativeModel("gemini-1.5-flash")
    res = model.generate_content(f"{PROMPT}\n===== OCR TEXT =====\n{text}")

    import json
    try:
        data = json.loads(res.text)
        return ReceiptParsed(**data)
    except:
        raise HTTPException(status_code=500, detail="Gemini JSON 파싱 실패")


# =========================
# Fallback 단순 파싱 (Gemini 실패 시)
# =========================
def fallback_parse(text: str) -> ReceiptParsed:

    import re

    clinic = None
    date = None
    time = None
    items = []
    total = None

    lines = [x.strip() for x in text.split("\n") if x.strip()]

    # 병원명 후보
    for line in lines:
        if "동물병원" in line:
            clinic = line
            break

    # 날짜
    m = re.search(r"(20\d{2}[./-]\d{1,2}[./-]\d{1,2})", text)
    if m:
        raw = m.group(1).replace(".", "-").replace("/", "-")
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d")
            date = dt.strftime("%Y-%m-%d")
        except:
            pass

    # 시간
    m = re.search(r"(\d{1,2}[:시]\d{1,2})", text)
    if m:
        t = m.group(1).replace("시", ":")
        try:
            tm = datetime.strptime(t, "%H:%M")
            time = tm.strftime("%H:%M")
        except:
            pass

    # 항목
    for line in lines:
        if any(x in line for x in ["진료", "주사", "백신", "약", "검사"]):
            items.append(ReceiptItem(name=line, price=None))

    # 총 금액
    mt = re.search(r"(\d{1,3}(,\d{3})+)\s*원", text)
    if mt:
        total = int(mt.group(1).replace(",", ""))

    return ReceiptParsed(
        clinicName=clinic,
        visitDate=date,
        visitTime=time,
        items=items,
        totalAmount=total
    )


# =========================
# 📌 최종 API: 영수증 분석
# =========================
@app.post("/api/receipt/analyze")
async def analyze_receipt(
    petId: str = Form(...),
    image: UploadFile = Form(...)
):
    try:
        img_bytes = await image.read()
    except:
        raise HTTPException(status_code=400, detail="이미지 읽기 실패")

    # 1) Vision OCR 먼저
    text = run_vision_ocr(img_bytes)

    if not text.strip():
        raise HTTPException(status_code=500, detail="OCR 텍스트 없음")

    # 2) Gemini 파싱 시도
    if USE_GEMINI and GEMINI_KEY:
        try:
            parsed = parse_with_gemini(text)
        except Exception as e:
            print("Gemini 실패 → fallback 전환", e)
            parsed = fallback_parse(text)
    else:
        parsed = fallback_parse(text)

    # 3) 영수증 이미지 S3 업로드 (너 기존 S3 코드 그대로 연결)
    s3_url = f"https://dummy-s3/{uuid.uuid4()}.jpg"

    return AnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes="OK"
    )
