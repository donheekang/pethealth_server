from __future__ import annotations

import os
import io
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.exceptions import NoCredentialsError
from pydantic_settings import BaseSettings

# Vertex AI / Gemini
import vertexai
from vertexai.generative_models import GenerativeModel, Part


# ------------------------------------------
# SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    # AWS S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # GCP / Vertex AI
    # 프로젝트 ID가 없으면 서버는 뜨되 Gemini 기능만 비활성화
    GCP_PROJECT_ID: Optional[str] = None
    GCP_LOCATION: str = "us-central1"
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"

    # 기타
    STUB_MODE: str = "false"

    class Config:
        env_file = ".env"


settings = Settings()


# ------------------------------------------
# AWS S3 CLIENT
# ------------------------------------------

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


def upload_to_s3(file_obj, key: str, content_type: str) -> str:
    """
    file-like 객체를 S3에 업로드하고 presigned URL 반환
    """
    try:
        s3_client.upload_fileobj(
            file_obj,
            settings.S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )

        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=7 * 24 * 3600,  # 7일
        )
        return url

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 인증 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {str(e)}")


# ------------------------------------------
# Vertex AI / Gemini 초기화
# ------------------------------------------

gemini_model: Optional[GenerativeModel] = None

try:
    if settings.GCP_PROJECT_ID:
        vertexai.init(
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )
        gemini_model = GenerativeModel(settings.GEMINI_MODEL_NAME)
        print(f"[Vertex AI] Gemini 모델 초기화 완료: {settings.GEMINI_MODEL_NAME}")
    else:
        print("[Vertex AI] GCP_PROJECT_ID 가 없어 Gemini 비활성화 상태입니다.")
except Exception as e:
    print(f"[Vertex AI] 초기화 실패: {e}")
    gemini_model = None


def _to_int_safe(value) -> Optional[int]:
    """
    문자열/숫자에서 숫자만 골라서 int로 변환.
    '81,000원' -> 81000
    변환 실패시 None
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def parse_receipt_with_gemini(image_bytes: bytes) -> dict:
    """
    영수증 이미지 1장을 Gemini 2.5 Flash-Lite로 OCR + 구조화해서 반환.

    반환 스키마:
    {
      "clinicName": str | null,
      "visitDate": "YYYY-MM-DD" | null,
      "visitTime": "HH:MM" | null,
      "items": [
        {"name": str, "price": int | null},
        ...
      ],
      "totalAmount": int | null,
      "paymentMethod": str | null
    }
    """
    if gemini_model is None:
        raise RuntimeError("Gemini 모델이 초기화되지 않았습니다.")

    image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    prompt = """
다음 이미지는 한국 반려동물 병원 영수증입니다.
이미지에서 직접 OCR을 수행하고 영수증 내용을 분석해 아래 JSON 형식으로만 출력하세요.

형식 (키 이름 반드시 동일해야 합니다):

{
  "clinicName": string or null,
  "visitDate": string or null,   // 방문 날짜, 형식: "YYYY-MM-DD"
  "visitTime": string or null,   // 방문 시각, 형식: "HH:MM" (24시간제, 앞에 0 포함)
  "items": [
    {
      "name": string,
      "price": integer or null   // 항목 금액(원 단위, 쉼표 제거한 정수)
    }
  ],
  "totalAmount": integer or null,   // 총 결제 금액(원 단위, 정수)
  "paymentMethod": string or null   // "카드", "현금", "계좌이체" 등
}

규칙:
•⁠  ⁠금액은 모두 '원' 단위의 정수로만 적어주세요. (예: 30000, 81000)
•⁠  ⁠날짜가 여러 번 나와도 실제 진료/결제일을 선택하세요.
•⁠  ⁠항목 이름은 가능하면 영수증 상 품명 그대로 적으세요. (예: "DHPPi", "Corona", "Nexgard Spectra 7.5~15kg")
•⁠  ⁠정보가 없으면 null 을 넣으세요.
•⁠  ⁠JSON 이외의 설명 텍스트는 절대 추가하지 마세요.
"""

    response = gemini_model.generate_content(
        [prompt, image_part],
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    )

    text = response.text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 혹시 ⁠ json  ⁠ 같은 마크다운이 감싸져 있을 경우 대비
        if "{" in text and "}" in text:
            text = text[text.find("{"): text.rfind("}") + 1]
            data = json.loads(text)
        else:
            raise

    # 최소 필드 보정
    data.setdefault("clinicName", None)
    data.setdefault("visitDate", None)
    data.setdefault("visitTime", None)
    data.setdefault("items", [])
    data.setdefault("totalAmount", None)
    data.setdefault("paymentMethod", None)

    # items 정규화
    fixed_items: List[Dict] = []
    if isinstance(data.get("items"), list):
        for it in data["items"]:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            price = _to_int_safe(it.get("price"))
            fixed_items.append(
                {
                    "name": name,
                    "price": price,
                }
            )
    data["items"] = fixed_items

    # totalAmount 정수 보정
    data["totalAmount"] = _to_int_safe(data.get("totalAmount"))

    return data


# ------------------------------------------
# FASTAPI APP
# ------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "PetHealth+ Server Running"}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ------------------------------------------
# 1) 진료기록 OCR (영수증 업로드)
# ------------------------------------------

@app.post("/receipt/upload")
@app.post("/receipts/upload")
@app.post("/api/receipt/upload")
@app.post("/api/receipts/upload")
@app.post("/api/receipt/analyze")   # iOS에서 쓰는 엔드포인트
@app.post("/api/receipts/analyze")
async def upload_receipt(
    petId: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    """
    영수증 이미지 업로드 + Gemini 2.5 Flash-Lite로 OCR + 구조화.
    S3 key: receipts/{petId}/{id}.jpg
    iOS가 file 이나 image 어떤 이름으로 보내도 처리.
    """
    upload: Optional[UploadFile] = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="no file or image field")

    # 파일 읽기
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    # S3 업로드
    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(upload.filename or "")
    if not ext:
        ext = ".jpg"
    key = f"receipts/{petId}/{rec_id}{ext}"

    file_like = io.BytesIO(data)
    file_like.seek(0)

    file_url = upload_to_s3(
        file_like,
        key,
        content_type=upload.content_type or "image/jpeg",
    )

    # Gemini로 OCR + 구조화
    if gemini_model is None:
        # Gemini 미초기화시 비어 있는 기본값으로 응답
        ai_parsed = {
            "clinicName": None,
            "visitDate": None,
            "visitTime": None,
            "items": [],
            "totalAmount": None,
            "paymentMethod": None,
        }
    else:
        try:
            ai_parsed = parse_receipt_with_gemini(data)
        except Exception as e:
            print(f"[Gemini] 영수증 파싱 실패: {e}")
            ai_parsed = {
                "clinicName": None,
                "visitDate": None,
                "visitTime": None,
                "items": [],
                "totalAmount": None,
                "paymentMethod": None,
            }

    # iOS DTO 형식으로 변환
    visit_date = ai_parsed.get("visitDate")
    visit_time = ai_parsed.get("visitTime")
    visit_date_for_dto: Optional[str] = None
    if visit_date and visit_time:
        visit_date_for_dto = f"{visit_date} {visit_time}"
    elif visit_date:
        visit_date_for_dto = visit_date

    dto_items: List[Dict] = []
    for it in ai_parsed.get("items", []):
        dto_items.append(
            {
                "name": it.get("name"),
                "price": it.get("price"),
            }
        )

    parsed_for_dto = {
        "clinicName": ai_parsed.get("clinicName"),
        "visitDate": visit_date_for_dto,
        "diseaseName": None,
        "symptomsSummary": None,
        "items": dto_items,
        "totalAmount": ai_parsed.get("totalAmount"),
        "paymentMethod": ai_parsed.get("paymentMethod"),
    }

    return {
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": None,
    }


# ------------------------------------------
# 2) 검사결과 PDF 업로드
# ------------------------------------------

@app.post("/lab/upload-pdf")
@app.post("/labs/upload-pdf")
@app.post("/api/lab/upload-pdf")
@app.post("/api/labs/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form("검사결과"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    lab_id = str(uuid.uuid4())

    original_base = os.path.splitext(file.filename or "")[0].strip() or "검사결과"
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")
    key = f"lab/{petId}/{safe_base}__{lab_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")

    created_at_dt = datetime.utcnow()
    created_at_iso = created_at_dt.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = created_at_dt.strftime("%Y-%m-%d")

    display_title = f"{original_base} ({date_str})"

    return {
        "id": lab_id,
        "petId": petId,
        "title": display_title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at_iso,
    }


# ------------------------------------------
# 3) 증명서 PDF 업로드
# ------------------------------------------

@app.post("/cert/upload-pdf")
@app.post("/certs/upload-pdf")
@app.post("/api/cert/upload-pdf")
@app.post("/api/certs/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form("증명서"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    cert_id = str(uuid.uuid4())

    original_base = os.path.splitext(file.filename or "")[0].strip() or "증명서"
    safe_base = original_base.replace("/", "").replace("\\", "").replace(" ", "_")
    key = f"cert/{petId}/{safe_base}__{cert_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")

    created_at_dt = datetime.utcnow()
    created_at_iso = created_at_dt.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = created_at_dt.strftime("%Y-%m-%d")

    display_title = f"{original_base} ({date_str})"

    return {
        "id": cert_id,
        "petId": petId,
        "title": display_title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at_iso,
    }


# ------------------------------------------
# 4) 검사결과 리스트
# ------------------------------------------

@app.get("/lab/list")
@app.get("/labs/list")
@app.get("/api/lab/list")
@app.get("/api/labs/list")
def get_lab_list(petId: str = Query(...)):
    prefix = f"lab/{petId}/"

    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=prefix,
    )

    items: List[Dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            name_no_ext = os.path.splitext(filename)[0]

            base_title = "검사결과"
            file_id = name_no_ext

            if "__" in name_no_ext:
                safe_base, file_id = name_no_ext.rsplit("__", 1)
                base_title = safe_base.replace("_", " ")

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            display_title = f"{base_title} ({date_str})"

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": display_title,
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at_iso,
                }
            )

    return items


# ------------------------------------------
# 5) 증명서 리스트
# ------------------------------------------

@app.get("/cert/list")
@app.get("/certs/list")
@app.get("/api/cert/list")
@app.get("/api/certs/list")
def get_cert_list(petId: str = Query(...)):
    prefix = f"cert/{petId}/"

    response = s3_client.list_objects_v2(
        Bucket=settings.S3_BUCKET_NAME,
        Prefix=prefix,
    )

    items: List[Dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            name_no_ext = os.path.splitext(filename)[0]

            base_title = "증명서"
            file_id = name_no_ext

            if "__" in name_no_ext:
                safe_base, file_id = name_no_ext.rsplit("__", 1)
                base_title = safe_base.replace("_", " ")

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            display_title = f"{base_title} ({date_str})"

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": display_title,
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at_iso,
                }
            )

    return items
