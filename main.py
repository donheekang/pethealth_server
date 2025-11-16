# main.py
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.client import Config

# 선택: 설치 안 돼 있으면 그냥 None로
try:
    from google.cloud import vision
except ImportError:
    vision = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# =========================
# 환경 변수 / 기본 설정
# =========================
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

GCV_ENABLED = os.getenv("GCV_ENABLED", "false").lower() == "true"
GCV_CREDENTIALS_JSON = os.getenv("GCV_CREDENTIALS_JSON")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =========================
# S3 클라이언트
# =========================
s3_client = None
if not STUB_MODE and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME:
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

# =========================
# Google Vision 클라이언트
# =========================
vision_client = None
if not STUB_MODE and GCV_ENABLED and GCV_CREDENTIALS_JSON and vision is not None:
    # Render 환경변수에 JSON 문자열로 넣어둔 걸 임시파일로 저장
    cred_path = "/tmp/gcv_credentials.json"
    with open(cred_path, "w", encoding="utf-8") as f:
        f.write(GCV_CREDENTIALS_JSON)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    vision_client = vision.ImageAnnotatorClient()

# =========================
# Gemini 클라이언트
# =========================
gemini_model = None
if not STUB_MODE and GEMINI_API_KEY and genai is not None:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# =========================
# Pydantic 모델 (iOS와 맞춤)
# =========================
class MedicalItem(BaseModel):
    id: str
    name: str
    price: Optional[int] = None


class MedicalRecord(BaseModel):
    id: str
    petId: str
    clinicName: str
    visitDate: datetime
    items: List[MedicalItem]
    totalAmount: Optional[int] = None
    imageUrl: Optional[str] = None
    rawText: Optional[str] = None
    summary: Optional[str] = None  # Gemini 요약 (옵션)


class UploadedFileInfo(BaseModel):
    id: str
    kind: str            # "lab" / "certificate"
    petId: Optional[str] = None
    fileUrl: str
    originalFilename: str
    createdAt: datetime


# =========================
# FastAPI 앱 & CORS
# =========================
app = FastAPI(title="PetHealth+ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 나중에 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 유틸 함수
# =========================
def upload_to_s3(folder: str, filename: str, data: bytes, content_type: str) -> str:
    """S3 업로드 후 presigned URL 반환 (STUB_MODE면 dummy URL)."""
    if STUB_MODE or s3_client is None or not S3_BUCKET_NAME:
        return f"https://example.com/{folder}/{filename}"

    key = f"{folder}/{filename}"
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )

    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": key},
        ExpiresIn=60 * 60 * 24 * 7,  # 7일
    )
    return url


def run_vision_ocr(image_bytes: bytes) -> str:
    """Google Vision OCR. 실패 시 빈 문자열 / STUB_MODE면 예시 텍스트."""
    if STUB_MODE or vision_client is None:
        return "토리동물병원\n진료 15,000원\n피부약 22,000원\n합계 37,000원"

    try:
        from google.cloud import vision as _vision

        img = _vision.Image(content=image_bytes)
        resp = vision_client.document_text_detection(image=img)
        if resp.error.message:
            print("GCV error:", resp.error.message)
            return ""
        if resp.full_text_annotation and resp.full_text_annotation.text:
            return resp.full_text_annotation.text
        if resp.text_annotations:
            return "\n".join([a.description for a in resp.text_annotations])
    except Exception as e:
        print("GCV exception:", e)
        return ""
    return ""


def gemini_summary(text: str) -> Optional[str]:
    if STUB_MODE or not text.strip() or gemini_model is None:
        return None
    try:
        prompt = (
            "다음은 반려동물 병원 영수증 OCR 텍스트입니다. "
            "병원명, 날짜, 주요 진료/처치, 비용 특징을 2~3줄로 한국어 요약해 주세요.\n\n"
            f"{text}"
        )
        res = gemini_model.generate_content(prompt)
        return (res.text or "").strip()
    except Exception as e:
        print("Gemini error:", e)
        return None


def parse_receipt(text: str):
    """
    OCR 텍스트에서 병원명 / 합계 금액 정도만 간단히 파싱.
    나중에 정교하게 바꿀 수 있음.
    """
    if not text:
        now = datetime.now(timezone.utc)
        items = [MedicalItem(id=str(uuid.uuid4()), name="진료", price=None)]
        return "미확인 동물병원", now, items, None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clinic_name = lines[0] if lines else "미확인 동물병원"
    visit_date = datetime.now(timezone.utc)

    import re

    amount_pattern = re.compile(r"([0-9][0-9,]{2,})\s*원")
    candidates = []
    for ln in lines:
        for m in amount_pattern.finditer(ln):
            try:
                candidates.append(int(m.group(1).replace(",", "")))
            except ValueError:
                pass

    total_amount = max(candidates) if candidates else None

    items: List[MedicalItem] = []
    if total_amount is not None:
        items.append(
            MedicalItem(
                id=str(uuid.uuid4()),
                name="진료/처치",
                price=total_amount,
            )
        )
    else:
        items.append(MedicalItem(id=str(uuid.uuid4()), name="진료", price=None))

    return clinic_name, visit_date, items, total_amount


# =========================
# 라우트 정의
# =========================

@app.get("/")
async def root():
    """브라우저에서 열면 보이는 기본 응답."""
    return {
        "message": "PetHealth+ 서버 연결 성공 ✅",
        "ocr": bool(vision_client) or STUB_MODE,
        "stubMode": STUB_MODE,
    }


@app.get("/health")
async def health():
    """헬스 체크 (iOS / Render)."""
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "stubMode": STUB_MODE,
    }


@app.post("/api/receipt/analyze", response_model=MedicalRecord)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    iOS: 영수증 이미지 업로드 → MedicalRecord 응답.

    - URL: POST /api/receipt/analyze
    - Form fields:
        - petId: String
        - file: image/jpeg or image/png
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    # 1) S3 업로드
    ext = "jpg"
    if file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1]
    rec_id = str(uuid.uuid4())
    filename = f"{rec_id}.{ext}"

    image_url = upload_to_s3(
        folder="receipts",
        filename=filename,
        data=data,
        content_type=file.content_type or "image/jpeg",
    )

    # 2) OCR
    raw_text = run_vision_ocr(data)

    # 3) 파싱
    clinic_name, visit_date, items, total_amount = parse_receipt(raw_text)

    # 4) Gemini 요약 (옵션)
    summary = gemini_summary(raw_text)

    return MedicalRecord(
        id=rec_id,
        petId=petId,
        clinicName=clinic_name,
        visitDate=visit_date,
        items=items,
        totalAmount=total_amount,
        imageUrl=image_url,
        rawText=raw_text or None,
        summary=summary,
    )


@app.post("/api/lab/upload", response_model=UploadedFileInfo)
async def upload_lab(
    petId: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """검사결과 PDF S3 업로드."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="PDF만 업로드 가능합니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    rec_id = str(uuid.uuid4())
    filename = f"{rec_id}.pdf"
    url = upload_to_s3("labs", filename, data, "application/pdf")

    return UploadedFileInfo(
        id=rec_id,
        kind="lab",
        petId=petId,
        fileUrl=url,
        originalFilename=file.filename or filename,
        createdAt=datetime.now(timezone.utc),
    )


@app.post("/api/certificate/upload", response_model=UploadedFileInfo)
async def upload_certificate(
    petId: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """증명서 PDF S3 업로드."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="PDF만 업로드 가능합니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    rec_id = str(uuid.uuid4())
    filename = f"{rec_id}.pdf"
    url = upload_to_s3("certificates", filename, data, "application/pdf")

    return UploadedFileInfo(
        id=rec_id,
        kind="certificate",
        petId=petId,
        fileUrl=url,
        originalFilename=file.filename or filename,
        createdAt=datetime.now(timezone.utc),
    )
