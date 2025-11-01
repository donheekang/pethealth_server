from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PetHealth+ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "PetHealth+ 서버 연결 성공 ✅"}

@app.get("/api/records")
def records():
    return [
        {"date": "2025-11-01", "hospital": "행복동물병원", "title": "예방접종(DHPPL) 4차"},
        {"date": "2025-10-20", "hospital": "다온동물병원", "title": "정기검진"},
    ]