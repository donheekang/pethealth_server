import os, uuid, time
from typing import Optional, List, Dict
from datetime import datetime
from fastapi import FastAPI, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from botocore.client import Config

# ----------------------------
# Env (Render "Environment"에 이미 넣은 값들 자동 읽음)
# ----------------------------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
    # Render Logs에서 보라고 메시지 남겨두기
    print("[WARN] AWS env 미설정: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / S3_BUCKET_NAME 확인 필요")

# ----------------------------
# S3 client (v4 서명)
# ----------------------------
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
)

# ----------------------------
# FastAPI 기본 세팅
# ----------------------------
app = FastAPI(title="PetHealth+ API", version="0.2.0")

# MVP: 어디서든 호출 허용 (운영 시 특정 도메인으로 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# In-memory 저장소 (MVP)
# 운영 전환 시 DB로 교체 (PostgreSQL/Supabase)
# ----------------------------
_CERTS: List[Dict] = []  # {id, file_key, file_name, hospital, issued_at}
_RECORDS: List[Dict] = [
    {"date": "2025-11-01", "hospital": "행복동물병원", "title": "예방접종(DHPPL) 4차"},
    {"date": "2025-10-20", "hospital": "다온동물병원", "title": "정기검진"},
]

# ----------------------------
# 모델
# ----------------------------
class CertOut(BaseModel):
    id: str
    hospital: str
    file_name: str
    file_key: str
    issued_at: str   # ISO string
    download_url: Optional[str] = None  # presign GET (요청 시 생성)

# ----------------------------
# 헬스 체크
# ----------------------------
@app.get("/")
def home():
    return {"message": "PetHealth+ 서버 연결 성공 ✅", "time": int(time.time())}

# ----------------------------
# 1) Records (샘플)
# ----------------------------
@app.get("/api/records")
def list_records():
    return _RECORDS

# ----------------------------
# 2) S3 업로드용 Presigned URL 발급
# ----------------------------
@app.get("/api/upload/presign")
def presign_upload(
    file_name: str = Query(..., description="원본 파일명, 예: vaccine_2025-09-20.pdf"),
    content_type: str = Query("application/pdf", description="기본값 PDF"),
):
    """
    iOS 앱이 업로드 전에 호출:
    1) GET /api/upload/presign?file_name=xxx.pdf
    2) 반환된 upload_url 로 'PUT' 업로드
    3) 업로드 완료 후 /api/certificates 로 등록
    """
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3 bucket 미설정")

    # 버킷 내부 키 규칙 (중복 방지)
    unique = uuid.uuid4().hex[:12]
    file_key = f"certificates/{unique}_{file_name}"

    try:
        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": S3_BUCKET_NAME,
                "Key": file_key,
                "ContentType": content_type,
            },
            ExpiresIn=300,  # 5분
        )
        return {
            "upload_url": url,
            "file_key": file_key,
            "expires_in": 300,
            "bucket": S3_BUCKET_NAME,
            "region": AWS_REGION,
        }
    except Exception as e:
        print("[ERROR] presign_upload:", e)
        raise HTTPException(status_code=500, detail="presign 생성 실패")

# ----------------------------
# 3) 업로드 완료 후, 서버에 '등록'
# ----------------------------
@app.post("/api/certificates")
def register_certificate(
    file_key: str = Form(..., description="S3 객체 키 (presign 응답의 file_key)"),
    file_name: str = Form(..., description="표시용 파일명"),
    hospital: str = Form(..., description="발급 병원명"),
):
    """
    iOS 앱 흐름:
      - presign 응답 수신 → PUT 업로드 성공
      - 그 다음 이 API로 메타데이터 등록
    """
    cert_id = uuid.uuid4().hex
    item = {
        "id": cert_id,
        "hospital": hospital,
        "file_name": file_name,
        "file_key": file_key,
        "issued_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _CERTS.append(item)
    return {"status": "ok", "id": cert_id}

# ----------------------------
# 4) 증명서 목록
# ----------------------------
@app.get("/api/certificates", response_model=List[CertOut])
def list_certificates(with_download_url: bool = False, expires_in: int = 300):
    """
    with_download_url=true 로 호출하면, 각 항목에 presigned GET URL 추가
    """
    result: List[CertOut] = []
    for c in _CERTS:
        download_url = None
        if with_download_url:
            try:
                download_url = s3.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": S3_BUCKET_NAME, "Key": c["file_key"]},
                    ExpiresIn=expires_in,
                )
            except Exception as e:
                print("[WARN] presign_get 실패:", e)
                download_url = None
        result.append(CertOut(**c, download_url=download_url))
    return result

# ----------------------------
# 5) 단건 다운로드 URL (옵션)
# ----------------------------
@app.get("/api/files/presign_get")
def presign_get(file_key: str, expires_in: int = 300):
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3 bucket 미설정")
    try:
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": file_key},
            ExpiresIn=expires_in,
        )
        return {"download_url": url, "expires_in": expires_in}
    except Exception as e:
        print("[ERROR] presign_get:", e)
        raise HTTPException(status_code=500, detail="download url 생성 실패")
