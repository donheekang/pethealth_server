# main.py
import os, base64, re, json, datetime
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

# ========= FastAPI 기본 =========
app = FastAPI(title="PetHealth+ OCR/Analyze API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 전 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")  # ★ Render 환경변수에 넣기

@app.get("/")
def home():
    return {"message": "PetHealth+ 서버 연결 성공 ✅", "ocr": bool(GOOGLE_VISION_API_KEY)}

# ========= 유틸: 간단 파서/정규화 =========

VACCINE_MAP = {
    "dhppl": "종합백신(DHPPL)", "dhpp": "종합백신(DHPP)", "dhlpp": "종합백신(DHLPP)",
    "ra": "광견병백신", "rabies": "광견병백신",
    "corona": "코로나백신", "influenza": "켄넬코프/기관지염 백신",
}
KEYWORDS = {
    "surgery": ["수술", "중성화", "슬개골", "제거술", "봉합", "절제"],
    "vaccine": ["백신", "접종", "dhppl", "dhpp", "dhlpp", "ra", "rabies", "코로나", "인플루엔자", "켄넬"],
    "exam": ["진찰", "진료", "검진", "재진", "초진", "상담", "처치"],
    "lab": ["검사", "혈액", "x-ray", "엑스레이", "초음파", "feca", "대변", "소변"],
    "drug": ["약", "항생제", "소염제", "진통제", "연고", "정", "mL", "mg"],
    "treatment": ["주사", "수액", "처치", "처방", "치료"],
}

def normalize_date(text: str) -> Optional[str]:
    """
    한국형 날짜 포맷을 YYYY-MM-DD로 정규화
    """
    # 2025.09.24 / 2025-09-24
    m = re.search(r"(20\d{2})[.\-\/년 ]\s*(\d{1,2})[.\-\/월 ]\s*(\d{1,2})", text)
    if m:
        y, mon, d = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime.date(int(y), int(mon), int(d))
            return dt.isoformat()
        except:
            pass
    return None

def detect_price(line: str) -> Optional[int]:
    """
    금액 숫자 추출: 30,000원 / 15000 / 15,000 등
    """
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)\s*원?", line.replace(" ", ""))
    if m:
        val = m.group(1).replace(",", "")
        try:
            return int(val)
        except:
            return None
    return None

def classify_item(line: str) -> str:
    low = line.lower()
    # 백신 정규화(키워드 우선)
    for k, v in VACCINE_MAP.items():
        if k in low or v.replace("백신", "").lower() in low:
            return "vaccine"
    # 카테고리 매칭
    for t, keys in KEYWORDS.items():
        for kw in keys:
            if kw.lower() in low:
                return t
    # 기본값
    return "other"

def normalize_item_name(line: str) -> str:
    low = line.lower()
    # 백신 약어 매핑
    for k, v in VACCINE_MAP.items():
        if k in low or v.replace("백신","").lower() in low:
            return v
    # 흔한 축약/영문 대체
    repl = {
        "x-ray": "X-ray",
        "dhppl": "종합백신(DHPPL)",
        "dhpp": "종합백신(DHPP)",
        "dhlpp": "종합백신(DHLPP)",
        "ra": "광견병백신",
        "rabies": "광견병백신",
    }
    for k, v in repl.items():
        if k in low:
            return v
    # 기타는 원문 trim
    return line.strip()

def extract_hospital(text: str) -> Optional[str]:
    # ‘동물병원’, ‘의료센터’ 등 키워드
    for line in text.splitlines():
        if any(w in line for w in ["동물병원", "동물 메디컬", "애니멀", "의료센터", "동물병원원"]):
            return line.strip()
    # 상단 첫 줄에 병원명 있는 경우도 꽤 있음
    top = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return top[0] if top else None

def parse_receipt_text(ocr_text: str) -> Dict[str, Any]:
    """
    OCR 텍스트 → 우리 스키마로 구조화
    """
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    joined = "\n".join(lines)

    hospital = extract_hospital(joined)
    visit_date = normalize_date(joined)

    items: List[Dict[str, Any]] = []
    total_price: Optional[int] = None

    for ln in lines:
        if any(k in ln for k in ["합계", "총액", "총합계", "Total", "TOTAL"]):
            total_price = detect_price(ln)
            continue
        # 진료 항목 후보 라인
        if any(kw in ln for kws in KEYWORDS.values() for kw in kws) or detect_price(ln):
            items.append({
                "type": classify_item(ln),
                "name": normalize_item_name(ln),
                "price": detect_price(ln)
            })

    result = {
        "hospital": hospital,
        "visit_date": visit_date,
        "items": items,
        "total": total_price,
        "notes": None  # 필요하면 LLM 요약으로 보강 가능
    }
    return result

# ========= 추천 엔진 (MVP: 룰기반) =========

def recommend_from_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    항목 키워드 기반 사료/영양제/보험 간단 추천 (MVP 룰)
    """
    low_text = " ".join([i.get("name","") or "" for i in items]).lower()
    rec_food, rec_supp, rec_ins = [], [], []

    # 수술/관절
    if any(k in low_text for k in ["수술", "슬개골", "봉합", "절제"]):
        rec_supp += ["관절: 글루코사민", "관절: 콘드로이틴", "관절: MSM"]
        rec_food += ["회복기 고단백 저지방 포뮬러"]
    # 피부/알러지
    if any(k in low_text for k in ["피부", "알러지", "소양", "피부염"]):
        rec_supp += ["오메가3(DHA/EPA)", "비오틴"]
        rec_food += ["가수분해 단백질 사료", "단일 단백질 레시피(닭 제외 가능)"]
    # 장트러블
    if any(k in low_text for k in ["설사", "장염", "구토"]):
        rec_supp += ["프로바이오틱스", "프리바이오틱스"]
        rec_food += ["저지방 장케어 포뮬러"]
    # 백신/예방
    if "백신" in low_text or "접종" in low_text:
        rec_ins += ["예방형 플랜(자기부담 20%) 추천", "노령 전 특약 최소화"]

    # 기본 추천 (아무것도 매칭되지 않을 때)
    if not rec_food:
        rec_food = ["기본 소화기 건강 포뮬러"]
    if not rec_supp:
        rec_supp = ["기본 종합영양(오메가3 권장)"]
    if not rec_ins:
        rec_ins = ["표준 플랜(자기부담 20%, 연간보장 1,000만원 가정)"]

    return {
        "foods": list(dict.fromkeys(rec_food))[:3],
        "supplements": list(dict.fromkeys(rec_supp))[:3],
        "insurance": list(dict.fromkeys(rec_ins))[:3]
    }

# ========= Google OCR 호출 & 전체 파이프라인 =========

def google_ocr_bytes(image_bytes: bytes, mime: str) -> str:
    if not GOOGLE_VISION_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY not set")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    body = {
        "requests": [{
            "image": {"content": b64},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }
    headers = {"Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json=body, timeout=20)

    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Vision API error: {res.text[:200]}")

    data = res.json()
    try:
        text = data["responses"][0]["fullTextAnnotation"]["text"]
    except KeyError:
        text = data["responses"][0].get("textAnnotations", [{}])[0].get("description", "")
    return text or ""

@app.post("/api/ocr_analyze")
async def ocr_analyze(file: UploadFile = File(...)):
    """
    보호자가 영수증/명세서 사진을 업로드하면:
    1) Google OCR로 텍스트 추출
    2) 텍스트 → 진료 JSON 구조화
    3) JSON 기반 추천(사료/영양제/보험) 반환
    """
    content = await file.read()
    mime = file.content_type or "image/jpeg"

    # (PDF도 Vision API가 처리 가능하지만, 첫 MVP는 이미지 권장)
    ocr_text = google_ocr_bytes(content, mime)

    record = parse_receipt_text(ocr_text)
    recos = recommend_from_items(record["items"])

    return {
        "ocr_text": ocr_text,    # 디버깅/확인용
        "record": record,        # 진료 JSON
        "recommendations": recos # 추천 JSON
    }
