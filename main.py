# main.py — PetHealth+ MVP API (CRUD + Certificates)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

app = FastAPI(title="PetHealth+ API")

# (MVP용) 어디서든 접근 허용 — 운영 전 도메인 제한 권장
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=[""], allow_headers=[""],
)

# -------------------------------
# 0) 헬스체크
# -------------------------------
@app.get("/")
def home():
    return {"message": "PetHealth+ 서버 연결 성공 ✅"}

# -------------------------------
# 1) Records (GET 완료 + POST 추가)
# -------------------------------
class RecordIn(BaseModel):
    date: str        # "2025-11-01"
    hospital: str
    title: str
    type: str = "exam"   # exam/vaccine/...
    notes: str = ""

# 임시 메모리 (DB 붙이기 전)
RECORDS_MEM: List[RecordIn] = []

@app.get("/api/records")
def get_records():
    seed = [
        {"date": "2025-11-01", "hospital": "행복동물병원", "title": "예방접종(DHPPL) 4차"},
        {"date": "2025-10-20", "hospital": "다온동물병원", "title": "정기검진"},
    ]
    return seed + [r.dict() for r in RECORDS_MEM]

@app.post("/api/records")
def add_record(item: RecordIn):
    RECORDS_MEM.append(item)
    return {"ok": True, "count": len(RECORDS_MEM)}

# -------------------------------
# 2) Certificates (presign → PUT → register → list)
# -------------------------------
class PresignReq(BaseModel):
    filename: str
    content_type: str = "application/pdf"

class PresignRes(BaseModel):
    uploadUrl: str
    fileUrl: str

# 증명서 등록용
class CertificateIn(BaseModel):
    petId: str
    hospital: str
    kind: str = "예방접종증명서"  # vaccine/prescription/diagnosis 텍스트
    fileUrl: str
    issuedAt: Optional[str] = None  # "YYYY-MM-DD"

# 임시 메모리
CERTS: List[dict] = []

@app.post("/api/upload/presign", response_model=PresignRes)
def presign(req: PresignReq):
    # (MVP) 파일 저장 대신 업로드 에코 엔드포인트 사용
    # 실제 운영 시: boto3로 S3 presigned_url 생성해서 반환
    doc_id = str(uuid.uuid4())
    fake_upload = f"https://httpbin.org/put?key={doc_id}"   # PUT 허용(테스트용)
    file_url = f"https://files.pethealth.plus/{doc_id}/{req.filename}"
    return PresignRes(uploadUrl=fake_upload, fileUrl=file_url)

@app.post("/api/certificates")
def add_certificate(item: CertificateIn):
    doc = item.dict()
    doc["id"] = str(uuid.uuid4())
    if not doc.get("issuedAt"):
        doc["issuedAt"] = datetime.utcnow().strftime("%Y-%m-%d")
    CERTS.append(doc)
    return {"ok": True, "id": doc["id"]}

@app.get("/api/certificates")
def list_certificates(petId: Optional[str] = None):
    data = CERTS if petId is None else [c for c in CERTS if c.get("petId") == petId]
    return sorted(data, key=lambda x: x.get("issuedAt", ""), reverse=True)
