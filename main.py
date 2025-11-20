from __future__ import annotations

import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict

import boto3
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse

# -------- Settings --------


class Settings(BaseSettings):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "ap-northeast-2"
    s3_bucket_name: str

    gemini_api_key: Optional[str] = None
    gemini_enabled: bool = False

    stub_mode: bool = False  # True면 OCR 결과를 더미로 반환

    class Config:
        env_prefix = ""
        env_file = ".env"
        case_sensitive = False


settings = Settings(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region=os.getenv("AWS_REGION", "ap-northeast-2"),
    s3_bucket_name=os.getenv("S3_BUCKET_NAME", ""),
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    gemini_enabled=os.getenv("GEMINI_ENABLED", "false").lower() == "true",
    stub_mode=os.getenv("STUB_MODE", "false").lower() == "true",
)

# -------- S3 Client --------

if not settings.s3_bucket_name:
    raise RuntimeError("S3_BUCKET_NAME 환경변수가 필요합니다.")

s3 = boto3.client(
    "s3",
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    region_name=settings.aws_region,
)


def upload_to_s3(key: str, file_bytes: bytes, content_type: str) -> str:
    """
    S3에 업로드하고 presigned URL을 반환
    """
    s3.put_object(
        Bucket=settings.s3_bucket_name,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.s3_bucket_name, "Key": key},
        ExpiresIn=60 * 60 * 24 * 7,  # 7일짜리 URL
    )
    return url


# -------- Pydantic Models --------


class PdfRecord(BaseModel):
    id: str
    petId: str
    title: str
    memo: Optional[str] = None
    s3Url: str
    createdAt: Optional[str] = None


class ReceiptItem(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsed(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # "yyyy-MM-dd HH:mm"
    diseaseName: Optional[str] = None
    symptomsSummary: Optional[str] = None
    items: List[ReceiptItem] = []
    totalAmount: Optional[int] = None


class ReceiptAnalyzeResponse(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsed
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    time: str
    stubMode: bool


# -------- In-memory DB (간단히 메모리에 유지) --------

LAB_DB: Dict[str, List[PdfRecord]] = {}
CERT_DB: Dict[str, List[PdfRecord]] = {}

# -------- Gemini OCR Helper --------

_gemini_model = None

def get_gemini_model():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    if not (settings.gemini_enabled and settings.gemini_api_key):
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        return _gemini_model
    except Exception as e:
        # Gemini 사용불가 → None
        print("Gemini 초기화 실패:", e)
        return None


RECEIPT_PROMPT = """
너는 동물병원 영수증을 분석하는 AI야.
이미지 내용을 보고 아래 JSON 형식으로만 답해.

{
  "clinicName": "병원 이름 (모르면 null)",
  "visitDate": "2025-11-19 08:26",
  "diseaseName": "진료명 또는 질병명 (모르면 null)",
  "symptomsSummary": "증상 요약 (모르면 null)",
  "items": [
    { "name": "항목명", "price": 30000 }
  ],
  "totalAmount": 81000
}

규칙:
•⁠  ⁠금액은 쉼표 없이 숫자만(원 단위)으로.
•⁠  ⁠진료 항목이 여러 줄이면 items 배열에 모두 넣어.
•⁠  ⁠totalAmount 가 보이면 그대로 숫자로 넣고, 없으면 항목들의 합계를 넣어.
•⁠  ⁠모르는 값은 null 로 넣어.
JSON 이외의 설명 텍스트는 절대 쓰지 마.
"""


def analyze_receipt_with_gemini(image_bytes: bytes) -> ReceiptParsed:
    """
    이미지 바이트 → Gemini 호출 → ReceiptParsed 생성
    """
    if settings.stub_mode:
        # 더미 값 반환 (테스트용)
        return ReceiptParsed(
            clinicName="테스트동물병원",
            visitDate=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            diseaseName=None,
            symptomsSummary="더미 OCR 결과",
            items=[
                ReceiptItem(name="DHPPI", price=30000),
                ReceiptItem(name="Corona", price=25000),
                ReceiptItem(name="Nexgard Spectra 7.5~15kg", price=26000),
            ],
            totalAmount=81000,
        )

    model = get_gemini_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Gemini 사용 불가 (환경 설정 확인)")

    try:
        from PIL import Image
        from io import BytesIO

        img = Image.open(BytesIO(image_bytes))
        response = model.generate_content(
            [RECEIPT_PROMPT, img],
            request_options={"timeout": 90},
        )

        text = response.text.strip()
        # Gemini가 ⁠ json ...  ⁠ 형식으로 돌려 줄 수도 있으므로 처리
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]  # "json" 제거

        data = json.loads(text)
        return ReceiptParsed(
            clinicName=data.get("clinicName"),
            visitDate=data.get("visitDate"),
            diseaseName=data.get("diseaseName"),
            symptomsSummary=data.get("symptomsSummary"),
            items=[
                ReceiptItem(name=i.get("name", ""), price=i.get("price"))
                for i in data.get("items", [])
                if i.get("name")
            ],
            totalAmount=data.get("totalAmount"),
        )
    except Exception as e:
        print("Gemini 분석 실패:", e)
        # 실패 시 최소한의 정보라도 채워서 반환
        return ReceiptParsed(
            clinicName=None,
            visitDate=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            diseaseName=None,
            symptomsSummary="영수증 분석 중 오류가 발생했습니다.",
            items=[],
            totalAmount=None,
        )


# -------- FastAPI app --------

app = FastAPI(title="PetHealth+ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Health & Root --------

@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        time=datetime.now(timezone.utc).isoformat(),
        stubMode=settings.stub_mode,
    )


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        time=datetime.now(timezone.utc).isoformat(),
        stubMode=settings.stub_mode,
    )


# -------- Labs (검사결과 PDF) --------

@app.get("/labs/list", response_model=List[PdfRecord])
@app.get("/api/labs/list", response_model=List[PdfRecord])
def list_labs(petId: str) -> List[PdfRecord]:
    return LAB_DB.get(petId, [])


@app.post("/lab/upload-pdf", response_model=PdfRecord)
@app.post("/api/lab/upload-pdf", response_model=PdfRecord)
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form(...),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
) -> PdfRecord:
    try:
        content = await file.read()
        ext = os.path.splitext(file.filename or "")[1] or ".pdf"
        key = f"labs/{petId}/{uuid.uuid4()}{ext}"

        url = upload_to_s3(key, content, "application/pdf")

        record = PdfRecord(
            id=str(uuid.uuid4()),
            petId=petId,
            title=title,
            memo=memo,
            s3Url=url,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )
        LAB_DB.setdefault(petId, []).append(record)
        return record
    except Exception as e:
        print("upload_lab_pdf error:", e)
        raise HTTPException(status_code=500, detail="검사결과 업로드 실패")


# -------- Certificates (증명서 PDF) --------

@app.get("/cert/list", response_model=List[PdfRecord])
@app.get("/api/cert/list", response_model=List[PdfRecord])
def list_certs(petId: str) -> List[PdfRecord]:
    return CERT_DB.get(petId, [])


@app.post("/cert/upload-pdf", response_model=PdfRecord)
@app.post("/api/cert/upload-pdf", response_model=PdfRecord)
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form(...),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
) -> PdfRecord:
    try:
        content = await file.read()
        ext = os.path.splitext(file.filename or "")[1] or ".pdf"
        key = f"certs/{petId}/{uuid.uuid4()}{ext}"

        url = upload_to_s3(key, content, "application/pdf")

        record = PdfRecord(
            id=str(uuid.uuid4()),
            petId=petId,
            title=title,
            memo=memo,
            s3Url=url,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )
        CERT_DB.setdefault(petId, []).append(record)
        return record
    except Exception as e:
        print("upload_cert_pdf error:", e)
        raise HTTPException(status_code=500, detail="증명서 업로드 실패")


# -------- 영수증 OCR: /api/receipt/analyze --------

@app.post("/api/receipt/analyze", response_model=ReceiptAnalyzeResponse)
async def analyze_receipt_endpoint(
    petId: str = Form(...),
    image: UploadFile = File(...),
) -> ReceiptAnalyzeResponse:
    """
    - iOS: multipart/form-data 로 petId, image(jpeg) 전송
    - 서버:
        1) 이미지 S3 업로드
        2) Gemini(OCR)로 내용 분석
        3) ReceiptAnalyzeResponse 반환
    """

    if image.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="image/jpeg 또는 image/png만 허용됩니다.")

    try:
        content = await image.read()

        # 1) 영수증 이미지 S3 업로드
        ext = ".jpg"
        if image.filename and image.filename.lower().endswith(".png"):
            ext = ".png"

        key = f"receipts/{petId}/{uuid.uuid4()}{ext}"
        s3_url = upload_to_s3(key, content, image.content_type or "image/jpeg")

        # 2) Gemini / Stub 으로 OCR 분석
        parsed = analyze_receipt_with_gemini(content)

        # 3) Response 생성
        resp = ReceiptAnalyzeResponse(
            petId=petId,
            s3Url=s3_url,
            parsed=parsed,
            notes=None,
        )
        return resp

    except HTTPException:
        raise
    except Exception as e:
        print("analyze_receipt_endpoint error:", e)
        raise HTTPException(status_code=500, detail="영수증 분석 실패")
