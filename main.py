# main.py
import io
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from botocore.client import Config
import boto3

# 외부 서비스(옵션)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from google.cloud import vision
except ImportError:
    vision = None

# ---------------------------
# 환경변수 / 설정
# ---------------------------
STUB_MODE = os.getenv("STUB_MODE", "false").lower() == "true"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

GCV_ENABLED = os.getenv("GCV_ENABLED", "false").lower() == "true"
GCV_CREDENTIALS_JSON = os.getenv("GCV_CREDENTIALS_JSON")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------
# AWS S3 클라이언트
# ---------------------------
s3_client = None
if not STUB_MODE and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME:
    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

# ---------------------------
# Google Vision 클라이언트
# ---------------------------
vision_client = None
if not STUB_MODE and GCV_ENABLED and GCV_CREDENTIALS_JSON and vision is not None:
    # Render 환경변수에 저장한 JSON 문자열을 임시 파일로 저장
    cred_path = "/tmp/gcv_credentials.json"
    with open(cred_path, "w", encoding="utf-8") as f:
        f.write(GCV_CREDENTIALS_JSON)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    vision_client = vision.ImageAnnotatorClient()

# ---------------------------
# Gemini 클라이언트
# ---------------------------
if not STUB_MODE and GEMINI_API_KEY and genai is not None:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

# ---------------------------
# Pydantic 모델 정의
# ---------------------------

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
    summary: Optional[str] = None  # Gemini 요약 (iOS 쪽에서는 없어도 디코딩됨)


class UploadedFileInfo(BaseModel):
    id: str
    kind: str          # "lab" or "certificate"
    petId: Optional[str] = None
    fileUrl: str
    originalFilename: str
    createdAt: datetime


# ---------------------------
# FastAPI 앱 생성 + CORS
# ---------------------------
app = FastAPI(title="PetHealth+ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요하면 iOS 도메인으로 좁혀도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# 유틸 함수들
# ---------------------------

def upload_to_s3_and_get_url(folder: str, filename: str, data: bytes, content_type: str) -> str:
    """
    S3에 업로드하고 presigned URL 반환.
    버킷이 private여도 iOS에서 이 URL로 접근 가능 (기본 7일 유효).
    """
    if STUB_MODE or s3_client is None or not S3_BUCKET_NAME:
        # 개발/테스트용 Mock URL
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


def run_ocr_on_image_bytes(image_bytes: bytes) -> str:
    """Google Vision으로 OCR 수행. 실패 시 빈 문자열."""
    if STUB_MODE or vision_client is None:
        # 개발 모드에서는 고정 텍스트 리턴
        return "토리동물병원\n진료 15,000원\n피부약 22,000원\n합계 37,000원"

    try:
        image = vision.Image(content=image_bytes)
        resp = vision_client.document_text_detection(image=image)
        if resp.error.message:
            print("GCV error:", resp.error.message)
            return ""
        if resp.full_text_annotation and resp.full_text_annotation.text:
            return resp.full_text_annotation.text
        if resp.text_annotations:
            return "\n".join([ann.description for ann in resp.text_annotations])
    except Exception as e:
        print("GCV exception:", e)
    return ""


def summarize_with_gemini(text: str) -> Optional[str]:
    """OCR 텍스트를 Gemini로 요약 (선택)."""
    if STUB_MODE or not text.strip() or gemini_model is None:
        return None

    try:
        prompt = (
            "다음은 반려동물 병원 영수증 OCR 텍스트입니다. "
            "병원명, 진료 날짜, 주요 진료/처치 항목, 비용 특징을 2~3줄 한국어로 간단히 정리해 주세요.\n\n"
            f"{text}"
        )
        result = gemini_model.generate_content(prompt)
        return (result.text or "").strip()
    except Exception as e:
        print("Gemini error:", e)
        return None


def parse_receipt_text(text: str):
    """
    OCR 텍스트에서 병원명, 합계 금액 등을 아주 단순하게 파싱.
    나중에 정교하게 바꿔도 됨. (지금은 기본값 위주)
    """
    if not text:
        now = datetime.now(timezone.utc)
        items = [
            MedicalItem(id=str(uuid.uuid4()), name="진료", price=None)
        ]
        return "미확인 동물병원", now, items, None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clinic_name = lines[0] if lines else "미확인 동물병원"

    # 날짜는 일단 지금 시간 사용 (추후 정규식으로 추출 가능)
    visit_date = datetime.now(timezone.utc)

    import re

    # 금액 후보 (예: 37,000원, 15000원 등)
    amount_pattern = re.compile(r"([0-9][0-9,]{2,})\s*원")
    candidates = []
    for ln in lines:
        for m in amount_pattern.finditer(ln):
            try:
                val = int(m.group(1).replace(",", ""))
                candidates.append(val)
            except ValueError:
                continue

    total_amount = max(candidates) if candidates else None

    # 간단히 두 개 항목으로 나누기
    items: List[MedicalItem] = []
    if total_amount is not None and len(candidates) >= 2:
        # 가장 큰 값 제외한 나머지 중 첫 번째를 진료비, 나머지를 기타로
        sorted_vals = sorted(candidates)
        main_fee = sorted_vals[-2]
        items.append(
            MedicalItem(
                id=str(uuid.uuid4()),
                name="진료",
                price=main_fee,
            )
        )
        # 나머지는 한 번에 묶기
        others = total_amount - main_fee
        if others > 0:
            items.append(
                MedicalItem(
                    id=str(uuid.uuid4()),
                    name="기타 처치/약",
                    price=others,
                )
            )
    else:
        items.append(MedicalItem(id=str(uuid.uuid4()), name="진료", price=total_amount))

    return clinic_name, visit_date, items, total_amount


# ---------------------------
# 엔드포인트
# ---------------------------

@app.get("/health")
async def health_check():
    """Render / iOS에서 상태 확인용."""
    return {
        "status": "ok",
        "stubMode": STUB_MODE,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/receipt/analyze", response_model=MedicalRecord)
async def analyze_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    영수증 이미지 업로드 → S3 저장 → Google OCR → 파싱 → MedicalRecord 반환.

    iOS에서:
    - multipart/form-data
    - field: petId (String)
    - field: file (이미지)
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다.")

    try:
        content = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="파일을 읽을 수 없습니다.")

    if not content:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    # 1) S3 업로드
    ext = "jpg"
    if file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1]
    object_id = str(uuid.uuid4())
    filename = f"{object_id}.{ext}"
    image_url = upload_to_s3_and_get_url("receipts", filename, content, file.content_type or "image/jpeg")

    # 2) OCR
    raw_text = run_ocr_on_image_bytes(content)

    # 3) 파싱
    clinic_name, visit_date, items, total_amount = parse_receipt_text(raw_text)

    # 4) (선택) Gemini 요약
    summary = summarize_with_gemini(raw_text)

    record = MedicalRecord(
        id=object_id,
        petId=petId,
        clinicName=clinic_name,
        visitDate=visit_date,
        items=items,
        totalAmount=total_amount,
        imageUrl=image_url,
        rawText=raw_text or None,
        summary=summary,
    )

    return record


@app.post("/api/lab/upload", response_model=UploadedFileInfo)
async def upload_lab_result(
    petId: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    검사결과 PDF 업로드 → S3 저장 (지금은 텍스트 추출 없이 URL만 반환).
    나중에 Gemini 분석용으로 raw 데이터/텍스트를 추가 엔드포인트에서 사용할 수 있음.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        data = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="파일을 읽을 수 없습니다.")

    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    ext = "pdf"
    object_id = str(uuid.uuid4())
    filename = f"{object_id}.{ext}"
    url = upload_to_s3_and_get_url("labs", filename, data, "application/pdf")

    info = UploadedFileInfo(
        id=object_id,
        kind="lab",
        petId=petId,
        fileUrl=url,
        originalFilename=file.filename or filename,
        createdAt=datetime.now(timezone.utc),
    )
    return info


@app.post("/api/certificate/upload", response_model=UploadedFileInfo)
async def upload_certificate(
    petId: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    증명서 PDF 업로드 → S3 저장.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        data = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="파일을 읽을 수 없습니다.")

    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    ext = "pdf"
    object_id = str(uuid.uuid4())
    filename = f"{object_id}.{ext}"
    url = upload_to_s3_and_get_url("certificates", filename, data, "application/pdf")

    info = UploadedFileInfo(
        id=object_id,
        kind="certificate",
        petId=petId,
        fileUrl=url,
        originalFilename=file.filename or filename,
        createdAt=datetime.now(timezone.utc),
    )
    return info
