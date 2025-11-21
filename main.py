from __future__ import annotations

import os
import io
import json
import uuid
import tempfile
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError
from pydantic_settings import BaseSettings


# ------------------------------------------
# SETTINGS
# ------------------------------------------

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # 서비스 계정 JSON 내용 또는 파일 경로
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    GEMINI_ENABLED: str = "false"
    STUB_MODE: str = "false"   # 필요하면 테스트용으로 사용

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
# GOOGLE VISION OCR
# ------------------------------------------

def get_vision_client() -> vision.ImageAnnotatorClient:
    """
    GOOGLE_APPLICATION_CREDENTIALS:
      - 서비스 계정 JSON '내용'일 수도 있고
      - JSON 파일 경로일 수도 있음
    둘 다 지원하게 구성
    """
    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS 환경변수가 비어있습니다.")

    # 1) JSON 내용으로 시도
    try:
        info = json.loads(cred_value)
        client = vision.ImageAnnotatorClient.from_service_account_info(info)
        return client
    except json.JSONDecodeError:
        # 2) JSON이 아니면 '파일 경로'로 간주
        if not os.path.exists(cred_value):
            raise Exception(
                "GOOGLE_APPLICATION_CREDENTIALS가 JSON도 아니고, "
                f"파일 경로({cred_value})도 아닙니다."
            )
        client = vision.ImageAnnotatorClient.from_service_account_file(cred_value)
        return client
    except Exception as e:
        raise Exception(f"OCR 클라이언트 생성 실패: {e}")


def run_vision_ocr(image_path: str) -> str:
    """
    Google Vision OCR로 텍스트 추출
    """
    client = get_vision_client()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"OCR 에러: {response.error.message}")

    texts = response.text_annotations
    if not texts:
        return ""

    return texts[0].description


# ------------------------------------------
# 영수증 OCR 결과 파싱 (병원명 / 시간 / 항목 / 금액)
# ------------------------------------------

def parse_receipt_kor(text: str) -> dict:
    """
    한국 동물병원 영수증 OCR 텍스트를
    - 병원명
    - 방문시간
    - 항목 리스트(이름+금액)
    - 총액
    으로 대략 파싱
    (완벽한 건 아니고, 기본 구조 뽑기용)
    """
    import re

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    hospital_name = lines[0] if lines else ""

    # 날짜/시간 (2025-11-19 08:26 / 2025.11.19 08:26 / 2025년 11월 19일 08:26 등)
    visit_at = None
    dt_pattern = re.compile(
        r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})[^\d]{0,3}(\d{1,2}):(\d{2})"
    )
    for line in lines:
        m = dt_pattern.search(line)
        if m:
            y, mo, d, h, mi = map(int, m.groups())
            visit_at = datetime(y, mo, d, h, mi).strftime("%Y-%m-%dT%H:%M:%S")
            break

    # 금액 패턴 (숫자+원)
    amt_pattern = re.compile(r"(\d{1,3}(,\d{3})*)\s*원")

    items = []
    total_amount = 0

    for line in lines:
        m = amt_pattern.search(line)
        if not m:
            continue
        amount_str = m.group(1).replace(",", "")
        try:
            amount = int(amount_str)
        except ValueError:
            continue

        name = line[:m.start()].strip()
        if not name:
            name = "항목"

        items.append({"name": name, "amount": amount})

    if items:
        total_amount = sum(i["amount"] for i in items)

    return {
        "hospitalName": hospital_name,
        "visitAt": visit_at,
        "items": items,
        "totalAmount": total_amount,
    }


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
#    - 여러 경로를 동시에 열어둠:
#      /receipt/upload
#      /receipts/upload
#      /api/receipt/upload
#      /api/receipts/upload
# ------------------------------------------

@app.post("/receipt/upload")
@app.post("/receipts/upload")
@app.post("/api/receipt/upload")
@app.post("/api/receipts/upload")
async def upload_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
):
    """
    영수증 이미지 업로드 + Vision OCR 실행
    - S3 key: receipts/{petId}/{id}.jpg
    - 응답: 병원명 / 시간 / 항목 / 금액 + 원본 OCR 텍스트 + S3 URL
    """
    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        ext = ".jpg"

    key = f"receipts/{petId}/{rec_id}{ext}"

    # 파일 데이터 읽기
    data = await file.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    # 1) S3 업로드
    file_url = upload_to_s3(
        file_like,
        key,
        content_type=file.content_type or "image/jpeg",
    )

    # 2) OCR용 임시 파일
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)

        ocr_text = run_vision_ocr(tmp_path)

    except Exception as e:
        # 여기서 예외 메시지 전체를 내보내면 팝업으로 그대로 보임
        raise HTTPException(
            status_code=500,
            detail=f"Vision OCR 사용 불가 (환경 설정 오류): {e}",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    parsed = parse_receipt_kor(ocr_text)
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # 👉 iOS 진료기록용 응답 구조
    return {
        "id": rec_id,
        "petId": petId,
        "hospitalName": parsed["hospitalName"],
        "visitAt": parsed["visitAt"],
        "items": parsed["items"],          # [ { name, amount }, ... ]
        "totalAmount": parsed["totalAmount"],
        "s3Url": file_url,
        "rawText": ocr_text,
        "createdAt": created_at,
    }


# ------------------------------------------
# 2) 검사결과 PDF 업로드
#    - 여러 경로 지원:
#      /lab/upload-pdf, /labs/upload-pdf,
#      /api/lab/upload-pdf, /api/labs/upload-pdf
#    - iOS 구조: PdfRecord 1개
# ------------------------------------------

@app.post("/lab/upload-pdf")
@app.post("/labs/upload-pdf")
@app.post("/api/lab/upload-pdf")
@app.post("/api/labs/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form("검사결과"),
    memo: str | None = Form(None),
    file: UploadFile = File(...),
):
    lab_id = str(uuid.uuid4())
    key = f"lab/{petId}/{lab_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # 👉 PdfRecord와 맞는 구조
    return {
        "id": lab_id,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at,
    }


# ------------------------------------------
# 3) 증명서 PDF 업로드
#    - /cert/upload-pdf, /certs/upload-pdf,
#      /api/cert/upload-pdf, /api/certs/upload-pdf
# ------------------------------------------

@app.post("/cert/upload-pdf")
@app.post("/certs/upload-pdf")
@app.post("/api/cert/upload-pdf")
@app.post("/api/certs/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form("증명서"),
    memo: str | None = Form(None),
    file: UploadFile = File(...),
):
    cert_id = str(uuid.uuid4())
    key = f"cert/{petId}/{cert_id}.pdf"

    file_url = upload_to_s3(file.file, key, content_type="application/pdf")
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # 👉 PdfRecord와 동일 구조
    return {
        "id": cert_id,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": file_url,
        "createdAt": created_at,
    }


# ------------------------------------------
# 4) 검사결과 리스트
#    - iOS: GET /api/labs/list?petId=...
#    - 여분: /lab/list, /labs/list, /api/lab/list, /api/labs/list 모두 열어둠
#    - 응답: [ PdfRecord ]
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

    items: list[dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]  # lab/{petId}/{id}.pdf
            if not key.endswith(".pdf"):
                continue

            file_id = os.path.splitext(key.split("/")[-1])[0]

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )
            created_at = obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%S")

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": "검사결과",
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at,
                }
            )

    # 👉 [PdfRecord] 형태로 바로 디코딩 가능
    return items


# ------------------------------------------
# 5) 증명서 리스트
#    - iOS: GET /api/cert/list?petId=...
#    - 여분: /cert/list, /certs/list, /api/cert/list, /api/certs/list
#    - 응답: [ PdfRecord ]
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

    items: list[dict] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]  # cert/{petId}/{id}.pdf
            if not key.endswith(".pdf"):
                continue

            file_id = os.path.splitext(key.split("/")[-1])[0]

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )
            created_at = obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%S")

            items.append(
                {
                    "id": file_id,
                    "petId": petId,
                    "title": "증명서",
                    "memo": None,
                    "s3Url": url,
                    "createdAt": created_at,
                }
            )

    return items
