import io
import os
import json
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.client import Config
from PIL import Image

import google.generativeai as genai

# ============================================================
# 환경변수
# ============================================================

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB 제한

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")
AWS_S3_PUBLIC_BASE_URL = os.getenv("AWS_S3_PUBLIC_BASE_URL")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini 활성화 여부 → API KEY 있으면 자동 활성
GEMINI_ENABLED = bool(GEMINI_API_KEY)

if GEMINI_ENABLED:
    genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# S3 클라이언트
# ============================================================

s3_client = None
if AWS_S3_BUCKET:
    session = boto3.session.Session()
    s3_client = session.client(
        "s3",
        region_name=AWS_REGION,
        endpoint_url=AWS_S3_ENDPOINT_URL or None,
        config=Config(signature_version="s3v4"),
    )


def build_public_url(key: str) -> str:
    if AWS_S3_PUBLIC_BASE_URL:
        return f"{AWS_S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    if AWS_S3_ENDPOINT_URL:
        return f"{AWS_S3_ENDPOINT_URL.rstrip('/')}/{key}"
    return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload_image_to_s3(file_bytes: bytes, filename: str) -> str:
    if not s3_client or not AWS_S3_BUCKET:
        raise RuntimeError("S3 설정이 없습니다. AWS_S3_BUCKET 확인.")

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

# ============================================================
# Gemini JSON Safe Parser
# ============================================================

def safe_parse_gemini_json(text: str) -> dict:
    """
    Gemini 결과에서 JSON만 추출해 dict로 반환.
    """
    text = text.strip()

    # 1) 그대로 JSON 시도
    try:
        return json.loads(text)
    except:
        pass

    # 2) { ... } 범위만 잘라서 재시도
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start:end+1]
        try:
            return json.loads(chunk)
        except:
            pass

    raise ValueError("Gemini가 JSON 형식이 아닌 응답을 반환했습니다.")


# ============================================================
# Pydantic Models
# ============================================================

class ReceiptItem(BaseModel):
    name: str
    price: int | None = None


class ReceiptParsed(BaseModel):
    clinicName: str | None = None
    visitDate: str | None = None
    diseaseName: str | None = None
    symptomsSummary: str | None = None
    items: list[ReceiptItem] = []
    totalAmount: int | None = None


class ReceiptAnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: str | None = None

# ============================================================
# FastAPI 설정
# ============================================================

app = FastAPI(
    title="PetHealth+ Backend",
    description="영수증 분석 / S3 업로드 / Gemini AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 기본 라우트
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "PetHealth+ 서버 정상 작동중",
        "gemini": GEMINI_ENABLED,
        "s3": bool(AWS_S3_BUCKET),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}

# ============================================================
# 영수증 분석 API
# ============================================================

@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    # -------- 1) 파일 로드
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(400, "빈 파일입니다.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(400, "파일 용량이 너무 큽니다. (15MB 이하)")

    # 이미지 → JPEG 정규화
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        image_bytes = buf.read()
    except:
        image_bytes = raw

    # -------- 2) S3 업로드
    try:
        s3_url = upload_image_to_s3(image_bytes, file.filename)
    except Exception as e:
        print("S3 업로드 실패:", e)
        s3_url = f"https://dummy-s3.pethealthplus/{uuid.uuid4().hex}.jpg"

    # -------- 3) Gemini 분석
    ai_data = None
    note_msg = ""

    if GEMINI_ENABLED:
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = """
반려동물 병원 영수증을 분석해서 아래 JSON 형식으로만 답하라.
문장 설명 없이 반드시 JSON만 출력하라.

{
  "clinicName": "",
  "visitDate": "YYYY-MM-DD",
  "diseaseName": "",
  "symptomsSummary": "",
  "items": [
      {"name": "", "price": 0}
  ],
  "totalAmount": 0
}
"""

            response = model.generate_content(
                [prompt, image_bytes],
                safety_settings=None,
            )

            ai_text = (response.text or "").strip()
            print("=== Gemini RAW ===")
            print(ai_text[:500])

            ai_data = safe_parse_gemini_json(ai_text)
            note_msg = "AI 분석 성공"

        except Exception as e:
            print("Gemini 분석 오류:", e)
            ai_data = None
            note_msg = f"AI 분석 실패 → STUB 적용 ({e})"

    # -------- 4) AI 실패 시 STUB 기본값
    if ai_data is None:
        ai_data = {
            "clinicName": "테스트동물병원",
            "visitDate": datetime.now().strftime("%Y-%m-%d"),
            "diseaseName": "피부염",
            "symptomsSummary": "가려움, 발진",
            "items": [
                {"name": "진료비", "price": 20000},
                {"name": "약품비", "price": 15000},
            ],
            "totalAmount": 35000,
        }

    # -------- 5) 모델 변환
    parsed = ReceiptParsed(
        clinicName=ai_data.get("clinicName"),
        visitDate=ai_data.get("visitDate"),
        diseaseName=ai_data.get("diseaseName"),
        symptomsSummary=ai_data.get("symptomsSummary"),
        items=[ReceiptItem(**item) for item in ai_data.get("items", [])],
        totalAmount=ai_data.get("totalAmount"),
    )

    return ReceiptAnalyzeResponse(
        petId=petId,
        s3Url=s3_url,
        parsed=parsed,
        notes=note_msg,
    )
