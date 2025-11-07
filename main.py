# main.py (라우팅 검증용 최소 버전)
from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PetHealth+ routing-check")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=[""], allow_headers=[""],
)

# 1) 반드시 살아있는 라우트들
@app.get("/")
def root():
    return {"message": "UP"}

@app.get("/health")
def health():
    return {"ok": True, "service": "PetHealthPlus", "version": "routing-check-1"}

# 2) 문제되던 네 개 경로를 stub으로 고정
api = APIRouter()

@api.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...)):
    return {"receipt_id": "receipts/STUB.jpg"}

@api.post("/ocr/analyze")
async def ocr_analyze(payload: dict):
    # payload: {"receipt_id":"..."}
    return {
        "clinicName": "Stub Animal Hospital",
        "visitDate": "2025-11-07T00:00:00Z",
        "items": [{"id":"1","name":"검진","category":"exam","price":30000}],
        "totalAmount": 30000,
        "notes": "stub response"
    }

@api.get("/records")
async def list_records():
    return []

@api.post("/records")
async def add_record(record: dict):
    return record

@api.post("/recommend")
async def recommend(body: dict):
    return [
        {"id":"r1","type":"food","title":"stub food","reasons":["reason1"],"tags":["tag1"]},
        {"id":"r2","type":"supplement","title":"stub supplement","reasons":["reason2"],"tags":["tag2"]}
    ]

app.include_router(api)

# 배포 시 라우트 목록 로그로 확인
@app.on_event("startup")
async def _print_routes():
    print("[ROUTES]", [r.path for r in app.router.routes])
