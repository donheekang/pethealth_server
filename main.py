from __future__ import annotations

import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError


# ============================================================
#  설정
# ============================================================

class Settings(BaseSettings):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "ap-northeast-2"
    s3_bucket_name: str

    gemini_api_key: Optional[str] = None
    gemini_enabled: bool = False

    google_application_credentials: Optional[str] = None

    stub_mode: bool = False  # True면 OCR/LLM 안 쓰고 더미 데이터 반환

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ============================================================
#  앱 & CORS
# ============================================================

app = FastAPI(title="PetHealth+ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ============================================================
#  S3 클라이언트
# ============================================================

def get_s3_client():
    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        raise RuntimeError("AWS 자격 증명이 설정되지 않았습니다.")
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
        config=Config(s3={"addressing_style": "path"}),
    )


def upload_bytes_to_s3(data: bytes, key: str, content_type: str, metadata: Optional[dict] = None) -> str:
    try:
        s3 = get_s3_client()
        extra = {"ContentType": content_type}
        if metadata:
            extra["Metadata"] = metadata
        s3.put_object(
            Bucket=settings.s3_bucket_name,
            Key=key,
            Body=data,
            **extra,
        )

        # presigned URL (7일)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket_name, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
        return url
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")


def list_pdf_records_from_s3(prefix: str, pet_id: Optional[str]) -> List["PdfRecord"]:
    try:
        s3 = get_s3_client()
    except RuntimeError:
        # S3 설정이 안 되어 있으면 빈 리스트
        return []

    results: List[PdfRecord] = []

    effective_prefix = prefix.rstrip("/") + "/"
    if pet_id:
        effective_prefix += f"{pet_id}/"

    try:
        resp = s3.list_objects_v2(
            Bucket=settings.s3_bucket_name,
            Prefix=effective_prefix,
        )
    except (BotoCoreError, ClientError) as e:
        print("⚠️ S3 list_objects 실패:", e)
        return []

    contents = resp.get("Contents", [])
    for obj in contents:
        key = obj["Key"]
        # key 형식: labs/{petId}/{id}.pdf 또는 certs/{petId}/{id}.pdf
        parts = key.split("/")
        if len(parts) < 3:
            continue
        _folder, petId, file_name = parts[0], parts[1], parts[2]
        obj_id = file_name.rsplit(".", 1)[0]

        # 메타데이터 읽기
        try:
            head = s3.head_object(Bucket=settings.s3_bucket_name, Key=key)
            meta = head.get("Metadata", {})
        except (BotoCoreError, ClientError):
            meta = {}

        title = meta.get("title", file_name)
        memo = meta.get("memo") or None
        created = head.get("LastModified", obj.get("LastModified", datetime.now(timezone.utc)))
        created_at = created.astimezone(timezone.utc).isoformat()

        # presigned url
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket_name, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )

        results.append(
            PdfRecord(
                id=obj_id,
                petId=petId,
                title=title,
                memo=memo,
                s3Url=url,
                createdAt=created_at,
            )
        )

    # 최신순 정렬
    results.sort(key=lambda r: r.createdAt or "", reverse=True)
    return results


# ============================================================
#  OCR / Gemini 초기화 (예외 발생해도 서버는 죽지 않게)
# ============================================================

try:
    from google.cloud import vision  # type: ignore
except Exception as e:
    print("⚠️ google.cloud.vision import 실패:", e)
    vision = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except Exception as e:
    print("⚠️ google.generativeai import 실패:", e)
    genai = None  # type: ignore

vision_client: Optional["vision.ImageAnnotatorClient"] = None
if "vision" in globals() and vision is not None:
    try:
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        vision_client = vision.ImageAnnotatorClient()
        print("✅ Vision OCR client 초기화 완료")
    except Exception as e:
        print("⚠️ Vision OCR 초기화 실패:", e)
        vision_client = None
else:
    vision_client = None

if settings.gemini_enabled and settings.gemini_api_key and "genai" in globals() and genai is not None:
    try:
        genai.configure(api_key=settings.gemini_api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini 초기화 완료")
    except Exception as e:
        print("⚠️ Gemini 초기화 실패:", e)
        settings.gemini_enabled = False
        gemini_model = None
else:
    gemini_model = None
    settings.gemini_enabled = False

# ============================================================
#  Pydantic 모델 (iOS DTO와 동일 구조)
# ============================================================

class ReceiptItemDTO(BaseModel):
    name: str
    price: Optional[int] = None


class ReceiptParsedDTO(BaseModel):
    clinicName: Optional[str] = None
    visitDate: Optional[str] = None  # yyyy-MM-dd
    diseaseName: Optional[str] = None
    symptomsSummary: Optional[str] = None
    items: List[ReceiptItemDTO] = []
    totalAmount: Optional[int] = None


class ReceiptAnalyzeResponseDTO(BaseModel):
    petId: str
    s3Url: str
    parsed: ReceiptParsedDTO
    notes: Optional[str] = None


class PdfRecord(BaseModel):
    id: str
    petId: str
    title: str
    memo: Optional[str] = None
    s3Url: str
    createdAt: Optional[str] = None


# ============================================================
#  유틸: OCR + Gemini 파이프라인
# ============================================================

def run_vision_ocr_bytes(image_bytes: bytes) -> str:
    """Vision OCR로 텍스트 추출 (안되면 예외 던짐)."""
    global vision_client
    if vision_client is None:
        raise RuntimeError("Vision OCR 클라이언트가 초기화되지 않았습니다.")

    image = vision.Image(content=image_bytes)  # type: ignore
    response = vision_client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision OCR 오류: {response.error.message}")
    text = response.full_text_annotation.text or ""
    return text.strip()


def parse_receipt_with_gemini(ocr_text: str) -> ReceiptParsedDTO:
    """Gemini에게 영수증 텍스트를 구조화 JSON으로 파싱 요청."""
    global gemini_model
    if not settings.gemini_enabled or gemini_model is None:
        raise RuntimeError("Gemini 사용 불가 (환경 설정 확인)")

    prompt = f"""
다음은 반려동물 병원 영수증 OCR 텍스트입니다.

이 텍스트를 기반으로 아래 JSON 스키마에 맞게만 답변해 주세요.
추가 설명 없이, JSON 하나만 출력합니다.

스키마:
{{
  "clinicName": string | null,
  "visitDate": "yyyy-MM-dd" | null,
  "diseaseName": string | null,
  "symptomsSummary": string | null,
  "items": [
    {{"name": string, "price": number|null}},
    ...
  ],
  "totalAmount": number|null
}}

텍스트:
```TEXT
{ocr_text}
