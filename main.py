# main.py (PetHealth+ Server) - Firebase Storage + Receipt Redaction + Migration (MVP)
import os
import io
import json
import uuid
import re
import base64
import hashlib
import secrets
import tempfile
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel, Field
from pydantic import ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pillow (image processing)
from PIL import Image, ImageDraw

# Google Vision
from google.cloud import vision

# Firebase Admin
import firebase_admin
from firebase_admin import credentials as fb_credentials
from firebase_admin import auth as fb_auth
from firebase_admin import storage as fb_storage

# PostgreSQL
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
import psycopg2.extras

try:
    from condition_tags import CONDITION_TAGS, ConditionTagConfig
except ImportError:
    CONDITION_TAGS = {}
    ConditionTagConfig = Any  # type: ignore
    print("Warning: condition_tags.py not found. AI tagging will be limited.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google.generativeai not installed. Gemini features disabled.")


# =========================================================
# Settings
# =========================================================
class Settings(BaseSettings):
    # --- Google Vision ---
    GOOGLE_APPLICATION_CREDENTIALS: str = ""  # JSON string or file path

    # --- Firebase Auth / Storage ---
    AUTH_REQUIRED: str = "true"
    STUB_MODE: str = "false"

    FIREBASE_ADMIN_SA_JSON: str = ""  # service account JSON string
    FIREBASE_ADMIN_SA_B64: str = ""   # base64-encoded JSON string (optional)
    FIREBASE_STORAGE_BUCKET: str = "" # e.g. "<project-id>.appspot.com"

    # --- Receipt image pipeline ---
    RECEIPT_MAX_WIDTH: int = 1024
    RECEIPT_WEBP_QUALITY: int = 85

    # Migration token TTL / processing
    MIGRATION_TOKEN_TTL_SECONDS: int = 10 * 60
    MIGRATION_PROCESSING_STALE_SECONDS: int = 5 * 60

    # --- Tag inference ---
    TAG_INFERENCE_ENABLED: str = "true"
    TAG_INFERENCE_ALLOWED_GROUPS: str = "exam,medication,procedure,preventive,wellness"
    TAG_INFERENCE_MIN_SCORE: int = 170
    TAG_INFERENCE_MAX_TAGS: int = 6

    # --- Gemini ---
    GEMINI_ENABLED: str = "false"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    # --- Postgres ---
    DB_ENABLED: str = "true"
    DATABASE_URL: str = ""
    DB_POOL_MIN: int = 1
    DB_POOL_MAX: int = 5
    DB_AUTO_UPSERT_USER: str = "true"

    # --- Admin ---
    ADMIN_UIDS: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# =========================================================
# Common UUID helpers
# =========================================================
def _uuid_or_400(value: Any, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except Exception:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a valid UUID")


def _uuid_or_new(value: Optional[str], field_name: str) -> uuid.UUID:
    v = (value or "").strip()
    if not v:
        return uuid.uuid4()
    return _uuid_or_400(v, field_name)


# =========================================================
# DB (PostgreSQL) helpers
# =========================================================
_db_pool: Optional[ThreadedConnectionPool] = None


def _normalize_db_url(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("postgres://"):
        u = "postgresql://" + u[len("postgres://"):]
    return u


def init_db_pool() -> None:
    global _db_pool
    if _db_pool is not None:
        return
    if (settings.DB_ENABLED or "").lower() != "true":
        print("[DB] DB_ENABLED=false. Skipping DB init.")
        return
    if not settings.DATABASE_URL:
        print("[DB] DATABASE_URL is empty. Skipping DB init.")
        return

    dsn = _normalize_db_url(settings.DATABASE_URL)
    try:
        psycopg2.extras.register_uuid()
        _db_pool = ThreadedConnectionPool(
            minconn=int(settings.DB_POOL_MIN),
            maxconn=int(settings.DB_POOL_MAX),
            dsn=dsn,
        )
        print("[DB] Postgres pool initialized.")
    except Exception as e:
        _db_pool = None
        print("[DB] Postgres pool init failed:", e)


def _require_db() -> None:
    if (settings.DB_ENABLED or "").lower() != "true":
        raise HTTPException(status_code=503, detail="DB is disabled (DB_ENABLED=false)")
    if not settings.DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not set")
    if _db_pool is None:
        init_db_pool()
    if _db_pool is None:
        raise HTTPException(status_code=503, detail="DB connection pool is not ready")


@contextmanager
def db_conn():
    _require_db()
    assert _db_pool is not None
    conn = _db_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _db_pool.putconn(conn)


def db_fetchone(sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None


def db_fetchall(sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    with db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]


def db_execute(sql: str, params: Tuple[Any, ...] = ()) -> int:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount


def db_touch_user(firebase_uid: str) -> Dict[str, Any]:
    uid = (firebase_uid or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="firebase_uid is empty")

    with db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO public.users (firebase_uid)
                VALUES (%s)
                ON CONFLICT (firebase_uid) DO UPDATE SET
                    last_seen_at = now()
                RETURNING firebase_uid, pet_count, created_at, updated_at, last_seen_at
                """,
                (uid,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=500, detail="Failed to upsert user")

            # DAU (optional)
            try:
                cur.execute(
                    """
                    INSERT INTO public.user_daily_active (day, firebase_uid)
                    VALUES (CURRENT_DATE, %s)
                    ON CONFLICT (day, firebase_uid) DO NOTHING
                    """,
                    (uid,),
                )
            except Exception as e:
                print("[DB] user_daily_active insert failed (ignored):", e)

            return dict(row)


def _clean_tags(tags: Any) -> List[str]:
    if not tags or not isinstance(tags, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for t in tags:
        s = str(t).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# =========================================================
# Firebase Admin init & Auth dependency
# =========================================================
_firebase_initialized = False
auth_scheme = HTTPBearer(auto_error=False)


def _normalize_private_key_newlines(info: dict) -> dict:
    if not isinstance(info, dict):
        return info
    pk = info.get("private_key")
    if isinstance(pk, str) and "\\n" in pk:
        info["private_key"] = pk.replace("\\n", "\n")
    return info


def _load_firebase_service_account() -> Optional[dict]:
    if settings.FIREBASE_ADMIN_SA_JSON:
        try:
            info = json.loads(settings.FIREBASE_ADMIN_SA_JSON)
        except Exception as e:
            raise RuntimeError(f"FIREBASE_ADMIN_SA_JSON JSON 파싱 실패: {e}")
        return _normalize_private_key_newlines(info)

    if settings.FIREBASE_ADMIN_SA_B64:
        try:
            raw = base64.b64decode(settings.FIREBASE_ADMIN_SA_B64).decode("utf-8")
            info = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"FIREBASE_ADMIN_SA_B64 디코드/파싱 실패: {e}")
        return _normalize_private_key_newlines(info)

    return None


def init_firebase_admin() -> None:
    global _firebase_initialized
    if _firebase_initialized:
        return

    if settings.STUB_MODE.lower() == "true" or settings.AUTH_REQUIRED.lower() != "true":
        print("[Auth] Firebase init skipped (STUB_MODE or AUTH_REQUIRED=false).")
        _firebase_initialized = True
        return

    info = _load_firebase_service_account()
    if not info:
        raise RuntimeError(
            "AUTH_REQUIRED=true 인데 Firebase service account가 비어있습니다. "
            "FIREBASE_ADMIN_SA_JSON 또는 FIREBASE_ADMIN_SA_B64를 설정하세요."
        )

    if not settings.FIREBASE_STORAGE_BUCKET:
        raise RuntimeError("FIREBASE_STORAGE_BUCKET is required for Firebase Storage")

    try:
        cred = fb_credentials.Certificate(info)
        firebase_admin.initialize_app(cred, {"storageBucket": settings.FIREBASE_STORAGE_BUCKET})
        _firebase_initialized = True
        print("[Auth] Firebase Admin initialized.")
    except Exception as e:
        raise RuntimeError(f"Firebase Admin initialize 실패: {e}")


def _maybe_auto_upsert_user(uid: str) -> None:
    if (settings.DB_AUTO_UPSERT_USER or "").lower() != "true":
        return
    try:
        if uid:
            db_touch_user(uid)
    except Exception as e:
        print("[DB] auto upsert user failed:", e)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
) -> Dict[str, Any]:
    if settings.STUB_MODE.lower() == "true" or settings.AUTH_REQUIRED.lower() != "true":
        return {"uid": "dev", "email": "dev@example.com"}

    init_firebase_admin()

    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    token = credentials.credentials
    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get("uid") or ""
        _maybe_auto_upsert_user(uid)
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Firebase token: {e}")


def _parse_admin_uids() -> Set[str]:
    raw = (settings.ADMIN_UIDS or "").strip()
    if not raw:
        return set()
    return set([p.strip() for p in raw.split(",") if p.strip()])


_ADMIN_UID_SET: Set[str] = _parse_admin_uids()


def get_admin_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user.get("uid") or ""
    if not _ADMIN_UID_SET:
        raise HTTPException(status_code=403, detail="ADMIN_UIDS is not configured")
    if uid not in _ADMIN_UID_SET:
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# =========================================================
# Firebase Storage helpers
# =========================================================
def _require_storage() -> None:
    if settings.STUB_MODE.lower() == "true":
        # dev stub
        return
    if not settings.FIREBASE_STORAGE_BUCKET:
        raise HTTPException(status_code=503, detail="FIREBASE_STORAGE_BUCKET is not set")


def get_bucket():
    init_firebase_admin()
    _require_storage()
    return fb_storage.bucket()


def upload_bytes_to_storage(path: str, data: bytes, content_type: str) -> str:
    """
    Upload bytes to Firebase Storage at 'path'.
    Returns the same relative path (no URL).
    """
    b = get_bucket()
    blob = b.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    return path


def download_json_from_storage(path: str) -> Dict[str, Any]:
    b = get_bucket()
    blob = b.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="object not found")
    raw = blob.download_as_bytes()
    return json.loads(raw.decode("utf-8"))


def upload_json_to_storage(path: str, obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return upload_bytes_to_storage(path, raw, "application/json")


def delete_storage_object_if_exists(path: str) -> bool:
    b = get_bucket()
    blob = b.blob(path)
    if not blob.exists():
        return False
    blob.delete()
    return True


def list_storage_objects(prefix: str) -> List[Dict[str, Any]]:
    b = get_bucket()
    out: List[Dict[str, Any]] = []
    for blob in b.list_blobs(prefix=prefix):
        out.append(
            {
                "name": blob.name,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "size": int(blob.size or 0),
                "contentType": getattr(blob, "content_type", None),
            }
        )
    return out


# =========================================================
# Path helpers (fixed structure)
# =========================================================
def _user_prefix(uid: str, pet_id: str) -> str:
    return f"users/{uid}/pets/{pet_id}"


def _backup_prefix(uid: str) -> str:
    return f"users/{uid}/backups"


def _receipt_path(uid: str, pet_id: str, record_id: str) -> str:
    return f"{_user_prefix(uid, pet_id)}/receipts/{record_id}.webp"


def _lab_pdf_path(uid: str, pet_id: str, doc_id: str) -> str:
    return f"{_user_prefix(uid, pet_id)}/lab/{doc_id}.pdf"


def _lab_meta_path(uid: str, pet_id: str, doc_id: str) -> str:
    return f"{_user_prefix(uid, pet_id)}/lab/{doc_id}.json"


def _cert_pdf_path(uid: str, pet_id: str, doc_id: str) -> str:
    return f"{_user_prefix(uid, pet_id)}/cert/{doc_id}.pdf"


def _cert_meta_path(uid: str, pet_id: str, doc_id: str) -> str:
    return f"{_user_prefix(uid, pet_id)}/cert/{doc_id}.json"


# =========================================================
# Google Vision OCR
# =========================================================
_vision_client: Optional[vision.ImageAnnotatorClient] = None


def get_vision_client() -> vision.ImageAnnotatorClient:
    global _vision_client
    if _vision_client is not None:
        return _vision_client

    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is empty")

    try:
        info = json.loads(cred_value)
        if isinstance(info, dict) and isinstance(info.get("private_key"), str):
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        _vision_client = vision.ImageAnnotatorClient.from_service_account_info(info)
        return _vision_client
    except json.JSONDecodeError:
        if not os.path.exists(cred_value):
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is neither JSON nor file path")
        _vision_client = vision.ImageAnnotatorClient.from_service_account_file(cred_value)
        return _vision_client


def run_vision_ocr_words(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Returns word-ish annotations with bounding boxes.
    Never store raw OCR text in DB/logs.
    """
    client = get_vision_client()
    img = vision.Image(content=image_bytes)
    resp = client.text_detection(image=img)
    if resp.error.message:
        raise RuntimeError(f"OCR error: {resp.error.message}")

    anns = resp.text_annotations or []
    out: List[Dict[str, Any]] = []

    # anns[0] is full text. skip it.
    for a in anns[1:]:
        desc = (a.description or "").strip()
        if not desc:
            continue
        vs = a.bounding_poly.vertices
        xs = [v.x for v in vs if v.x is not None]
        ys = [v.y for v in vs if v.y is not None]
        if not xs or not ys:
            continue
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        out.append({"text": desc, "bbox": (x1, y1, x2, y2)})
    return out


# =========================================================
# Aggressive PII Redaction (lines + footer padding)
# =========================================================
_RE_PHONE = re.compile(r"(01[016789][\-\s]?\d{3,4}[\-\s]?\d{4})|(0\d{1,2}[\-\s]?\d{3,4}[\-\s]?\d{4})")
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_RE_BIZNO = re.compile(r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{5}\b")
_RE_CARDLIKE = re.compile(r"(?:\d[\-\s]?){13,19}")
_RE_APPROVAL = re.compile(r"(승인|승인번호|approval|auth)[^\d]{0,8}\d{4,}")
_RE_ZIP = re.compile(r"\b\d{5}\b")  # KR postal 5 digits (heuristic)

_ADDR_KEYWORDS = ["주소", "도로명", "지번", "우편", "시", "군", "구", "읍", "면", "동", "로", "길", "번길"]
_PII_LINE_KEYWORDS = [
    "tel", "전화", "연락", "휴대", "phone",
    "주소", "도로명", "지번", "우편",
    "사업자", "사업자번호", "대표", "대표자",
    "카드", "card", "승인", "approval", "auth",
    "성명", "이름", "보호자", "고객", "owner", "name",
]

_MONEY_HINT = re.compile(r"(₩|원|krw)", re.IGNORECASE)
_MONEY_NUM = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")


def _group_words_into_lines(words: List[Dict[str, Any]], img_h: int) -> List[Dict[str, Any]]:
    """
    Cluster by y center. Returns lines with words sorted by x.
    """
    if not words:
        return []

    items = []
    for w in words:
        x1, y1, x2, y2 = w["bbox"]
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        items.append({**w, "cy": cy, "cx": cx})

    items.sort(key=lambda d: d["cy"])
    threshold = max(10, int(img_h * 0.012))  # ~1.2% height

    lines: List[Dict[str, Any]] = []
    for it in items:
        placed = False
        for ln in lines:
            if abs(it["cy"] - ln["cy"]) <= threshold:
                ln["words"].append(it)
                # update avg cy
                ln["cy"] = (ln["cy"] * (len(ln["words"]) - 1) + it["cy"]) / len(ln["words"])
                placed = True
                break
        if not placed:
            lines.append({"cy": it["cy"], "words": [it]})

    # finalize
    out = []
    for ln in lines:
        ws = ln["words"]
        ws.sort(key=lambda d: d["cx"])
        text = " ".join([w["text"] for w in ws]).strip()
        xs1 = [w["bbox"][0] for w in ws]
        ys1 = [w["bbox"][1] for w in ws]
        xs2 = [w["bbox"][2] for w in ws]
        ys2 = [w["bbox"][3] for w in ws]
        bbox = (min(xs1), min(ys1), max(xs2), max(ys2))
        out.append({"text": text, "bbox": bbox, "words": ws})
    return out


def _looks_like_address_line(line_text: str) -> bool:
    t = (line_text or "").strip()
    if not t:
        return False
    # keyword + digit heuristic
    has_kw = any(k in t for k in _ADDR_KEYWORDS)
    if has_kw and re.search(r"\d", t):
        return True
    # postal + kw
    if _RE_ZIP.search(t) and has_kw:
        return True
    return False


def _line_contains_pii_trigger(line_text: str) -> bool:
    t = (line_text or "")
    if not t.strip():
        return False

    low = t.lower()
    if any(k in low for k in _PII_LINE_KEYWORDS):
        return True

    if _RE_EMAIL.search(t):
        return True
    if _RE_PHONE.search(t):
        return True
    if _RE_BIZNO.search(t):
        return True
    if _RE_APPROVAL.search(t):
        return True
    if _RE_CARDLIKE.search(t):
        return True
    if _looks_like_address_line(t):
        return True

    # suspicious long digit blobs, excluding money-ish lines
    digit_runs = re.findall(r"\d{6,}", re.sub(r"[,\s\-]", "", t))
    if digit_runs:
        # allow if clearly money line
        if _MONEY_HINT.search(t):
            return False
        # if it has big numbers but no currency hint, treat as suspicious
        return True

    return False


def _line_is_money_item_candidate(line_text: str) -> bool:
    t = (line_text or "").strip()
    if not t:
        return False
    # must have a number candidate
    if not _MONEY_NUM.search(t):
        return False
    # prefer money hint, but allow if trailing number looks like amount
    if _MONEY_HINT.search(t):
        return True
    # fallback: last token is 3+ digits
    m = re.search(r"(\d{3,})(?!.*\d)", re.sub(r"[,\s]", "", t))
    return bool(m)


def _compute_redaction_boxes(lines: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    """
    Aggressive:
    - if a line triggers PII -> mask the entire line bbox with padding
    - footer padding: if address/phone triggers appear near bottom -> mask from that line upward padding to bottom
    """
    boxes: List[Tuple[int, int, int, int]] = []
    footer_trigger_y: Optional[int] = None

    # padding values (pixels)
    pad_x = max(12, int(img_w * 0.01))
    pad_y = max(8, int(img_h * 0.008))

    for ln in lines:
        text = ln["text"]
        x1, y1, x2, y2 = ln["bbox"]

        if _line_contains_pii_trigger(text):
            rx1 = max(0, x1 - pad_x)
            ry1 = max(0, y1 - pad_y)
            rx2 = min(img_w, x2 + pad_x)
            ry2 = min(img_h, y2 + pad_y)
            boxes.append((rx1, ry1, rx2, ry2))

            # footer detection: if trigger line is in bottom 35% OR looks like address/phone
            if (y1 > img_h * 0.65) or _looks_like_address_line(text) or _RE_PHONE.search(text):
                footer_trigger_y = y1 if footer_trigger_y is None else min(footer_trigger_y, y1)

    # footer padding mask: from (footer_trigger_y - extra_pad) to bottom
    if footer_trigger_y is not None:
        extra = max(40, int(img_h * 0.03))
        y = max(0, footer_trigger_y - extra)
        boxes.append((0, y, img_w, img_h))

    # merge-ish (simple): return as is (overlaps ok)
    return boxes


def _apply_redaction(img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    if not boxes:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    return out


def _resize_to_width(img: Image.Image, max_w: int) -> Image.Image:
    w, h = img.size
    if w <= max_w:
        return img
    ratio = max_w / float(w)
    nh = int(h * ratio)
    return img.resize((max_w, nh), Image.LANCZOS)


def _encode_webp(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    # method=6 gives better compression (slower, but MVP ok)
    img.save(buf, format="WEBP", quality=int(quality), method=6)
    return buf.getvalue()


# =========================================================
# Receipt parsing (best-effort, PII lines excluded)
# =========================================================
def _parse_visit_date_from_text(text: str) -> Optional[date]:
    t = (text or "").strip()
    if not t:
        return None
    # 2026-01-26 / 2026.01.26 / 2026년 1월 26일
    m = re.search(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})", t)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return date(y, mo, d)
    except Exception:
        return None


def _guess_hospital_name_from_lines(line_texts: List[str]) -> str:
    keywords = [
        "동물병원", "동물 병원", "동물의료", "동물메디컬", "동물 메디컬",
        "동물클리닉", "동물 클리닉",
        "애견병원", "애완동물병원", "펫병원", "펫 병원",
        "종합동물병원", "동물의원", "동물병의원",
    ]
    best_line = ""
    best_score = -999
    for idx, line in enumerate(line_texts[:30]):
        s = line.strip()
        if not s:
            continue
        score = 0
        compact = s.replace(" ", "")
        if any(k.replace(" ", "") in compact for k in keywords):
            score += 5
        if idx <= 4:
            score += 2
        if any(x in s for x in ["TEL", "전화", "FAX", "팩스", "도로명", "주소"]):
            score -= 2
        digit_count = sum(c.isdigit() for c in s)
        if digit_count >= 8:
            score -= 1
        if len(s) < 2 or len(s) > 30:
            score -= 1
        if score > best_score:
            best_score = score
            best_line = s
    return best_line.strip()


def _extract_total_amount(lines: List[str]) -> Optional[int]:
    keys = ["합계", "총액", "총 금액", "결제", "청구", "TOTAL", "AMOUNT"]
    for ln in lines:
        t = ln.upper()
        if any(k.upper() in t for k in keys):
            # find last number
            nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d{3,}", t)
            if nums:
                n = nums[-1].replace(",", "")
                try:
                    return int(n)
                except Exception:
                    pass
    return None


def _extract_items_from_lines(lines: List[str], max_items: int = 60) -> List[Dict[str, Any]]:
    """
    Very heuristic:
    - pick lines that look like money item candidates
    - extract last amount as price
    - remaining text -> item_name
    """
    out: List[Dict[str, Any]] = []
    for ln in lines:
        if not _line_is_money_item_candidate(ln):
            continue
        # extract amount
        nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d{3,}", ln)
        if not nums:
            continue
        price_raw = nums[-1].replace(",", "")
        try:
            price = int(price_raw)
        except Exception:
            price = None

        # remove that last number occurrence
        item_name = ln
        item_name = re.sub(re.escape(nums[-1]), "", item_name, count=1).strip()
        item_name = re.sub(r"(₩|원|KRW)", "", item_name, flags=re.IGNORECASE).strip()

        # avoid empty / too short
        if len(item_name) < 1:
            continue

        out.append(
            {
                "itemName": item_name[:200],
                "price": price,
                "categoryTag": None,
            }
        )
        if len(out) >= max_items:
            break
    return out


def process_receipt_image_and_parse(raw_image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    Returns (redacted_webp_bytes, parsed_structured_dict)
    - parsed_structured contains only non-PII: hospitalName, visitDate, totalAmount, items[]
    """
    # load image
    img = Image.open(io.BytesIO(raw_image_bytes))
    img = img.convert("RGB")  # strip alpha/exif

    w, h = img.size

    # OCR on original (for bbox accuracy), but never store raw OCR text
    words = run_vision_ocr_words(raw_image_bytes)
    lines = _group_words_into_lines(words, img_h=h)
    line_texts = [ln["text"] for ln in lines if ln.get("text")]

    # compute redaction boxes
    boxes = _compute_redaction_boxes(lines, img_w=w, img_h=h)
    redacted = _apply_redaction(img, boxes)

    # resize to 1024 width (no upscale)
    redacted_small = _resize_to_width(redacted, int(settings.RECEIPT_MAX_WIDTH))

    # encode webp quality 85 fixed
    webp = _encode_webp(redacted_small, int(settings.RECEIPT_WEBP_QUALITY))

    # Parse structured data from NON-PII candidate lines:
    # We exclude lines that triggered PII (conservatively)
    safe_lines: List[str] = []
    for ln in line_texts:
        if _line_contains_pii_trigger(ln):
            continue
        safe_lines.append(ln)

    hospital_name = _guess_hospital_name_from_lines(safe_lines)
    visit_date: Optional[date] = None
    for ln in safe_lines:
        vd = _parse_visit_date_from_text(ln)
        if vd:
            visit_date = vd
            break

    total_amount = _extract_total_amount(safe_lines)
    items = _extract_items_from_lines(safe_lines)

    parsed = {
        "hospitalName": hospital_name or None,
        "visitDate": visit_date.isoformat() if visit_date else None,
        "totalAmount": total_amount,
        "items": items,  # itemName/price/categoryTag only
    }
    return webp, parsed


# =========================================================
# DTOs
# =========================================================
class BackupUploadRequest(BaseModel):
    snapshot: Any
    clientTime: Optional[str] = None
    appVersion: Optional[str] = None
    device: Optional[str] = None
    note: Optional[str] = None


class BackupUploadResponse(BaseModel):
    ok: bool
    uid: str
    backupId: str
    objectPath: str
    createdAt: str


class PetUpsertRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str] = None
    name: str
    species: str = "dog"
    breed: Optional[str] = None
    birthday: Optional[date] = None
    weight_kg: Optional[float] = Field(default=None, alias="weightKg")

    gender: Optional[str] = Field(default=None, description="M/F/U or null")
    neutered: Optional[str] = Field(default=None, description="Y/N/U or null")

    has_no_allergy: Optional[bool] = Field(default=None, alias="hasNoAllergy")
    allergy_tags: List[str] = Field(default_factory=list, alias="allergyTags")


class HealthItemInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    item_name: str = Field(alias="itemName")
    price: Optional[int] = None
    category_tag: Optional[str] = Field(default=None, alias="categoryTag")


class HealthRecordUpsertRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str] = None
    pet_id: str = Field(alias="petId")
    visit_date: date = Field(alias="visitDate")
    hospital_name: Optional[str] = Field(default=None, alias="hospitalName")
    hospital_mgmt_no: Optional[str] = Field(default=None, alias="hospitalMgmtNo")
    total_amount: Optional[int] = Field(default=0, alias="totalAmount")
    pet_weight_at_visit: Optional[float] = Field(default=None, alias="petWeightAtVisit")
    tags: List[str] = Field(default_factory=list)
    items: Optional[List[HealthItemInput]] = None


class MigrationPrepareResponse(BaseModel):
    oldUid: str
    migrationCode: str
    expiresAt: str


class MigrationExecuteRequest(BaseModel):
    migrationCode: str


# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(title="PetHealth+ Server", version="2.1.0-fbstorage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    if settings.AUTH_REQUIRED.lower() == "true" and settings.STUB_MODE.lower() != "true":
        try:
            init_firebase_admin()
        except Exception as e:
            print("[Startup] Firebase init failed:", e)
    else:
        print("[Startup] Auth not required or STUB_MODE. Skipping Firebase init.")

    init_db_pool()


@app.on_event("shutdown")
def _shutdown():
    global _db_pool
    if _db_pool is not None:
        try:
            _db_pool.closeall()
            print("[DB] Pool closed.")
        except Exception as e:
            print("[DB] Pool close error:", e)
        _db_pool = None


@app.get("/")
def root():
    return {"status": "ok", "message": "PetHealth+ Server Running (Firebase Storage)"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "storage": "firebase",
        "storage_bucket": settings.FIREBASE_STORAGE_BUCKET,
        "receipt_webp_quality": settings.RECEIPT_WEBP_QUALITY,
        "receipt_max_width": settings.RECEIPT_MAX_WIDTH,
        "stub_mode": settings.STUB_MODE,
        "auth_required": settings.AUTH_REQUIRED,
        "db_enabled": settings.DB_ENABLED,
        "db_configured": bool(settings.DATABASE_URL),
    }


@app.get("/api/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return {"uid": user.get("uid"), "email": user.get("email")}


# =========================================================
# DB APIs
# =========================================================
@app.post("/api/db/user/upsert")
def api_user_upsert(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    row = db_touch_user(uid)
    return jsonable_encoder(row)


@app.post("/api/db/pets/upsert")
def api_pet_upsert(req: PetUpsertRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    pet_uuid = _uuid_or_new(req.id, "id")
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    species = (req.species or "dog").strip().lower()
    if species not in ("dog", "cat", "etc"):
        raise HTTPException(status_code=400, detail="species must be one of 'dog','cat','etc'")

    breed = req.breed.strip() if isinstance(req.breed, str) and req.breed.strip() else None
    birthday = req.birthday
    weight_kg = float(req.weight_kg) if req.weight_kg is not None else None

    gender_raw = (req.gender or "").strip().upper()
    gender: Optional[str] = None if not gender_raw else gender_raw
    if gender is not None and gender not in ("M", "F", "U"):
        raise HTTPException(status_code=400, detail="gender must be one of 'M','F','U' or null")

    neutered_raw = (req.neutered or "").strip().upper()
    neutered: Optional[str] = None if not neutered_raw else neutered_raw
    if neutered is not None and neutered not in ("Y", "N", "U"):
        raise HTTPException(status_code=400, detail="neutered must be one of 'Y','N','U' or null")

    # allergy policy (schema chk_allergy_consistency)
    has_no_allergy: Optional[bool] = req.has_no_allergy
    allergy_tags = [str(x).strip() for x in (req.allergy_tags or []) if str(x).strip()]

    # if tags exist but has_no_allergy is None => infer "FALSE"
    if allergy_tags and has_no_allergy is None:
        has_no_allergy = False

    # if has_no_allergy is True OR None => tags must be empty (force empty to satisfy constraint)
    if has_no_allergy is True or has_no_allergy is None:
        allergy_tags = []

    db_touch_user(uid)

    row = db_fetchone(
        """
        INSERT INTO public.pets
            (id, user_uid, name, species, breed, birthday, weight_kg, gender, neutered, has_no_allergy, allergy_tags)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            species = EXCLUDED.species,
            breed = EXCLUDED.breed,
            birthday = EXCLUDED.birthday,
            weight_kg = EXCLUDED.weight_kg,
            gender = EXCLUDED.gender,
            neutered = EXCLUDED.neutered,
            has_no_allergy = EXCLUDED.has_no_allergy,
            allergy_tags = EXCLUDED.allergy_tags
        WHERE public.pets.user_uid = EXCLUDED.user_uid
        RETURNING
            id, user_uid, name, species, breed, birthday, weight_kg, gender, neutered, has_no_allergy, allergy_tags,
            created_at, updated_at
        """,
        (pet_uuid, uid, name, species, breed, birthday, weight_kg, gender, neutered, has_no_allergy, allergy_tags),
    )

    if row:
        return jsonable_encoder(row)

    exists = db_fetchone("SELECT id, user_uid FROM public.pets WHERE id=%s", (pet_uuid,))
    if exists and exists.get("user_uid") != uid:
        raise HTTPException(status_code=403, detail="You do not own this pet id")
    raise HTTPException(status_code=500, detail="Failed to upsert pet")


@app.get("/api/db/pets/list")
def api_pets_list(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    rows = db_fetchall(
        """
        SELECT
            id, user_uid, name, species, breed, birthday, weight_kg, gender, neutered,
            has_no_allergy, allergy_tags,
            created_at, updated_at
        FROM public.pets
        WHERE user_uid=%s
        ORDER BY created_at DESC
        """,
        (uid,),
    )
    return jsonable_encoder(rows)


@app.post("/api/db/records/upsert")
def api_record_upsert(req: HealthRecordUpsertRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    db_touch_user(uid)

    record_uuid = _uuid_or_new(req.id, "id")
    pet_uuid = _uuid_or_400(req.pet_id, "petId")

    visit_date = req.visit_date
    hospital_name = req.hospital_name.strip() if isinstance(req.hospital_name, str) and req.hospital_name.strip() else None
    hospital_mgmt_no = req.hospital_mgmt_no.strip() if isinstance(req.hospital_mgmt_no, str) and req.hospital_mgmt_no.strip() else None

    total_amount = int(req.total_amount or 0)
    if total_amount < 0:
        raise HTTPException(status_code=400, detail="total_amount must be >= 0")

    pet_weight_at_visit = float(req.pet_weight_at_visit) if req.pet_weight_at_visit is not None else None
    tags = _clean_tags(req.tags)

    with db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="pet not found")

            cur.execute(
                """
                INSERT INTO public.health_records
                    (id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    pet_id = EXCLUDED.pet_id,
                    hospital_mgmt_no = EXCLUDED.hospital_mgmt_no,
                    hospital_name = EXCLUDED.hospital_name,
                    visit_date = EXCLUDED.visit_date,
                    total_amount = EXCLUDED.total_amount,
                    pet_weight_at_visit = EXCLUDED.pet_weight_at_visit,
                    tags = EXCLUDED.tags
                WHERE EXISTS (
                    SELECT 1
                    FROM public.pets p
                    WHERE p.id = public.health_records.pet_id
                      AND p.user_uid = %s
                )
                RETURNING
                    id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags,
                    receipt_image_path, receipt_meta_path,
                    created_at, updated_at
                """,
                (record_uuid, pet_uuid, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags, uid),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=500, detail="Failed to upsert record")

            if req.items is not None:
                cur.execute("DELETE FROM public.health_items WHERE record_id=%s", (record_uuid,))
                for it in req.items:
                    item_name = (it.item_name or "").strip()
                    if not item_name:
                        raise HTTPException(status_code=400, detail="itemName is required")

                    price = it.price
                    if price is not None:
                        price = int(price)
                        if price < 0:
                            raise HTTPException(status_code=400, detail="item price must be >= 0")

                    category_tag = it.category_tag.strip() if isinstance(it.category_tag, str) and it.category_tag.strip() else None
                    cur.execute(
                        """
                        INSERT INTO public.health_items (record_id, item_name, price, category_tag)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (record_uuid, item_name, price, category_tag),
                    )

            cur.execute(
                """
                SELECT id, record_id, item_name, price, category_tag, created_at, updated_at
                FROM public.health_items
                WHERE record_id=%s
                ORDER BY created_at ASC
                """,
                (record_uuid,),
            )
            items_rows = cur.fetchall() or []
            payload = dict(row)
            payload["items"] = [dict(x) for x in items_rows]
            return jsonable_encoder(payload)


@app.get("/api/db/records/list")
def api_records_list(
    petId: Optional[str] = Query(None),
    includeItems: bool = Query(False),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    if petId:
        pet_uuid = _uuid_or_400(petId, "petId")
        rows = db_fetchall(
            """
            SELECT
                r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount,
                r.pet_weight_at_visit, r.tags,
                r.receipt_image_path, r.receipt_meta_path,
                r.created_at, r.updated_at
            FROM public.health_records r
            JOIN public.pets p ON p.id = r.pet_id
            WHERE p.user_uid=%s AND p.id=%s
            ORDER BY r.visit_date DESC, r.created_at DESC
            """,
            (uid, pet_uuid),
        )
    else:
        rows = db_fetchall(
            """
            SELECT
                r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount,
                r.pet_weight_at_visit, r.tags,
                r.receipt_image_path, r.receipt_meta_path,
                r.created_at, r.updated_at
            FROM public.health_records r
            JOIN public.pets p ON p.id = r.pet_id
            WHERE p.user_uid=%s
            ORDER BY r.visit_date DESC, r.created_at DESC
            """,
            (uid,),
        )

    if not includeItems:
        return jsonable_encoder(rows)

    record_ids = [str(r["id"]) for r in rows if r.get("id")]
    if not record_ids:
        return jsonable_encoder(rows)

    items = db_fetchall(
        """
        SELECT id, record_id, item_name, price, category_tag, created_at, updated_at
        FROM public.health_items
        WHERE record_id = ANY(%s::uuid[])
        ORDER BY created_at ASC
        """,
        (record_ids,),
    )

    by_record: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        rid = str(it.get("record_id"))
        by_record.setdefault(rid, []).append(it)

    out = []
    for r in rows:
        rr = dict(r)
        rr["items"] = by_record.get(str(r.get("id")), [])
        out.append(rr)
    return jsonable_encoder(out)


@app.get("/api/db/records/get")
def api_record_get(
    recordId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or ""
    rid = (recordId or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="recordId is required")

    record_uuid = _uuid_or_400(rid, "recordId")

    row = db_fetchone(
        """
        SELECT
            r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount,
            r.pet_weight_at_visit, r.tags,
            r.receipt_image_path, r.receipt_meta_path,
            r.created_at, r.updated_at
        FROM public.health_records r
        JOIN public.pets p ON p.id = r.pet_id
        WHERE p.user_uid=%s AND r.id=%s
        """,
        (uid, record_uuid),
    )
    if not row:
        raise HTTPException(status_code=404, detail="record not found")

    items = db_fetchall(
        """
        SELECT id, record_id, item_name, price, category_tag, created_at, updated_at
        FROM public.health_items
        WHERE record_id=%s
        ORDER BY created_at ASC
        """,
        (record_uuid,),
    )
    row["items"] = items
    return jsonable_encoder(row)


# =========================================================
# Receipt endpoint (server-only upload)
# =========================================================
@app.post("/api/receipts/process")
async def api_receipts_process(
    petId: str = Form(...),
    recordId: Optional[str] = Form(None),
    replaceItems: bool = Form(True),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="file/image is required")

    uid = user.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    pet_uuid = _uuid_or_400(petId, "petId")
    record_uuid = _uuid_or_new(recordId, "recordId")

    # ownership check
    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    # 1) process: OCR -> redact -> resize -> webp
    try:
        webp_bytes, parsed = process_receipt_image_and_parse(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"receipt processing failed: {e}")

    # 2) upload to Firebase Storage (relative path)
    receipt_path = _receipt_path(uid, str(pet_uuid), str(record_uuid))
    try:
        upload_bytes_to_storage(receipt_path, webp_bytes, "image/webp")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"storage upload failed: {e}")

    # 3) upsert DB record + items (best-effort)
    visit_date = parsed.get("visitDate")
    hospital_name = parsed.get("hospitalName")
    total_amount = parsed.get("totalAmount")

    vd: date = date.today()
    if isinstance(visit_date, str) and visit_date:
        try:
            vd = datetime.strptime(visit_date, "%Y-%m-%d").date()
        except Exception:
            vd = date.today()

    hn = hospital_name.strip() if isinstance(hospital_name, str) and hospital_name.strip() else None
    ta = int(total_amount) if isinstance(total_amount, int) and total_amount >= 0 else 0

    extracted_items = parsed.get("items") if isinstance(parsed.get("items"), list) else []
    # sanitize extracted items
    safe_items: List[Dict[str, Any]] = []
    for it in extracted_items[:80]:
        if not isinstance(it, dict):
            continue
        nm = (it.get("itemName") or "").strip()
        if not nm:
            continue
        pr = it.get("price")
        if pr is not None:
            try:
                pr = int(pr)
                if pr < 0:
                    pr = None
            except Exception:
                pr = None
        safe_items.append({"itemName": nm[:200], "price": pr, "categoryTag": None})

    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # record upsert (receipt_image_path set by server only)
                cur.execute(
                    """
                    INSERT INTO public.health_records
                      (id, pet_id, hospital_name, visit_date, total_amount, tags, receipt_image_path)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                      pet_id = EXCLUDED.pet_id,
                      hospital_name = EXCLUDED.hospital_name,
                      visit_date = EXCLUDED.visit_date,
                      total_amount = EXCLUDED.total_amount,
                      receipt_image_path = EXCLUDED.receipt_image_path
                    WHERE EXISTS (
                      SELECT 1 FROM public.pets p
                      WHERE p.id = EXCLUDED.pet_id AND p.user_uid = %s
                    )
                    RETURNING
                      id, pet_id, hospital_name, visit_date, total_amount,
                      receipt_image_path, receipt_meta_path,
                      created_at, updated_at
                    """,
                    (record_uuid, pet_uuid, hn, vd, ta, [], receipt_path, uid),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=500, detail="Failed to upsert health_record")

                if replaceItems:
                    cur.execute("DELETE FROM public.health_items WHERE record_id=%s", (record_uuid,))
                    for it in safe_items:
                        cur.execute(
                            """
                            INSERT INTO public.health_items (record_id, item_name, price, category_tag)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (record_uuid, it["itemName"], it["price"], None),
                        )

                cur.execute(
                    """
                    SELECT id, record_id, item_name, price, category_tag, created_at, updated_at
                    FROM public.health_items
                    WHERE record_id=%s
                    ORDER BY created_at ASC
                    """,
                    (record_uuid,),
                )
                items_rows = cur.fetchall() or []

                payload = dict(row)
                payload["items"] = [dict(x) for x in items_rows]
                return jsonable_encoder(payload)

    except Exception as e:
        # DB 실패 시 업로드한 파일 정리 시도(최대한)
        try:
            delete_storage_object_if_exists(receipt_path)
        except Exception:
            pass
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"db upsert failed: {e}")


# =========================================================
# Lab / Cert (server-only write, client read)
# =========================================================
@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form("검사결과"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or ""
    pet_uuid = _uuid_or_400(petId, "petId")

    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty pdf")

    doc_id = uuid.uuid4().hex
    pdf_path = _lab_pdf_path(uid, str(pet_uuid), doc_id)
    meta_path = _lab_meta_path(uid, str(pet_uuid), doc_id)

    upload_bytes_to_storage(pdf_path, data, "application/pdf")
    meta = {
        "id": doc_id,
        "uid": uid,
        "petId": str(pet_uuid),
        "kind": "lab",
        "title": title,
        "memo": memo,
        "originalFilename": file.filename,
        "createdAt": datetime.utcnow().isoformat(),
        "objectPath": pdf_path,
    }
    upload_json_to_storage(meta_path, meta)

    return {"id": doc_id, "objectPath": pdf_path, "metaPath": meta_path, "createdAt": meta["createdAt"]}


@app.get("/api/lab/list")
def list_lab(petId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    pet_uuid = _uuid_or_400(petId, "petId")

    base = _user_prefix(uid, str(pet_uuid)) + "/lab/"
    objects = list_storage_objects(base)

    pdfs = [o for o in objects if o["name"].endswith(".pdf")]
    out = []
    for o in sorted(pdfs, key=lambda x: x.get("updated") or "", reverse=True):
        name = o["name"]
        doc_id = os.path.splitext(os.path.basename(name))[0]
        mp = _lab_meta_path(uid, str(pet_uuid), doc_id)
        meta = None
        try:
            meta = download_json_from_storage(mp)
        except Exception:
            meta = None
        out.append(
            {
                "id": doc_id,
                "title": (meta or {}).get("title") or "검사결과",
                "memo": (meta or {}).get("memo"),
                "objectPath": name,
                "createdAt": (meta or {}).get("createdAt") or (o.get("updated")),
            }
        )
    return out


@app.delete("/api/lab/delete")
def delete_lab(petId: str = Query(...), id: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    pet_uuid = _uuid_or_400(petId, "petId")
    doc_id = (id or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="id required")

    pdf_path = _lab_pdf_path(uid, str(pet_uuid), doc_id)
    meta_path = _lab_meta_path(uid, str(pet_uuid), doc_id)

    deleted_pdf = delete_storage_object_if_exists(pdf_path)
    delete_storage_object_if_exists(meta_path)

    if not deleted_pdf:
        raise HTTPException(status_code=404, detail="file not found")
    return {"ok": True, "deleted": [pdf_path, meta_path]}


@app.post("/api/cert/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form("접종증명서"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or ""
    pet_uuid = _uuid_or_400(petId, "petId")

    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty pdf")

    doc_id = uuid.uuid4().hex
    pdf_path = _cert_pdf_path(uid, str(pet_uuid), doc_id)
    meta_path = _cert_meta_path(uid, str(pet_uuid), doc_id)

    upload_bytes_to_storage(pdf_path, data, "application/pdf")
    meta = {
        "id": doc_id,
        "uid": uid,
        "petId": str(pet_uuid),
        "kind": "cert",
        "title": title,
        "memo": memo,
        "originalFilename": file.filename,
        "createdAt": datetime.utcnow().isoformat(),
        "objectPath": pdf_path,
    }
    upload_json_to_storage(meta_path, meta)
    return {"id": doc_id, "objectPath": pdf_path, "metaPath": meta_path, "createdAt": meta["createdAt"]}


@app.get("/api/cert/list")
def list_cert(petId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    pet_uuid = _uuid_or_400(petId, "petId")

    base = _user_prefix(uid, str(pet_uuid)) + "/cert/"
    objects = list_storage_objects(base)
    pdfs = [o for o in objects if o["name"].endswith(".pdf")]

    out = []
    for o in sorted(pdfs, key=lambda x: x.get("updated") or "", reverse=True):
        name = o["name"]
        doc_id = os.path.splitext(os.path.basename(name))[0]
        mp = _cert_meta_path(uid, str(pet_uuid), doc_id)
        meta = None
        try:
            meta = download_json_from_storage(mp)
        except Exception:
            meta = None
        out.append(
            {
                "id": doc_id,
                "title": (meta or {}).get("title") or "접종증명서",
                "memo": (meta or {}).get("memo"),
                "objectPath": name,
                "createdAt": (meta or {}).get("createdAt") or (o.get("updated")),
            }
        )
    return out


@app.delete("/api/cert/delete")
def delete_cert(petId: str = Query(...), id: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    pet_uuid = _uuid_or_400(petId, "petId")
    doc_id = (id or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="id required")

    pdf_path = _cert_pdf_path(uid, str(pet_uuid), doc_id)
    meta_path = _cert_meta_path(uid, str(pet_uuid), doc_id)

    deleted_pdf = delete_storage_object_if_exists(pdf_path)
    delete_storage_object_if_exists(meta_path)

    if not deleted_pdf:
        raise HTTPException(status_code=404, detail="file not found")
    return {"ok": True, "deleted": [pdf_path, meta_path]}


# =========================================================
# Backup endpoints (Firebase Storage)
# =========================================================
@app.post("/api/backup/upload", response_model=BackupUploadResponse)
async def backup_upload(req: BackupUploadRequest = Body(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or "unknown"
    backup_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat()

    doc = {
        "meta": {
            "uid": uid,
            "backupId": backup_id,
            "createdAt": created_at,
            "clientTime": req.clientTime,
            "appVersion": req.appVersion,
            "device": req.device,
            "note": req.note,
        },
        "snapshot": req.snapshot,
    }

    path = f"{_backup_prefix(uid)}/{backup_id}.json"
    upload_json_to_storage(path, doc)

    return {"ok": True, "uid": uid, "backupId": backup_id, "objectPath": path, "createdAt": created_at}


@app.get("/api/backup/list")
def backup_list(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or "unknown"
    prefix = _backup_prefix(uid) + "/"
    objs = list_storage_objects(prefix)
    out = []
    for o in objs:
        if not o["name"].endswith(".json"):
            continue
        bid = os.path.splitext(os.path.basename(o["name"]))[0]
        out.append(
            {
                "backupId": bid,
                "objectPath": o["name"],
                "lastModified": o.get("updated"),
                "size": o.get("size", 0),
            }
        )
    out.sort(key=lambda x: x.get("lastModified") or "", reverse=True)
    return out


@app.get("/api/backup/get")
def backup_get(backupId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or "unknown"
    bid = (backupId or "").strip()
    if not bid:
        raise HTTPException(status_code=400, detail="backupId is required")
    path = f"{_backup_prefix(uid)}/{bid}.json"
    return download_json_from_storage(path)


@app.get("/api/backup/latest")
def backup_latest(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or "unknown"
    prefix = _backup_prefix(uid) + "/"
    objs = list_storage_objects(prefix)
    jsons = [o for o in objs if o["name"].endswith(".json")]
    if not jsons:
        raise HTTPException(status_code=404, detail="no backups")
    latest = max(jsons, key=lambda o: o.get("updated") or "")
    return download_json_from_storage(latest["name"])


# =========================================================
# Migration tokens + migration execution
# =========================================================
def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _generate_migration_code() -> str:
    return secrets.token_urlsafe(32)


@app.post("/api/migration/prepare", response_model=MigrationPrepareResponse)
def migration_prepare(user: Dict[str, Any] = Depends(get_current_user)):
    old_uid = user.get("uid") or ""
    if not old_uid:
        raise HTTPException(status_code=401, detail="missing uid")

    db_touch_user(old_uid)

    code = _generate_migration_code()
    code_hash = _hash_code(code)
    expires_at = datetime.utcnow() + timedelta(seconds=int(settings.MIGRATION_TOKEN_TTL_SECONDS))

    # store token
    try:
        db_execute(
            """
            INSERT INTO public.migration_tokens (old_uid, code_hash, expires_at, status)
            VALUES (%s, %s, %s, 'issued')
            """,
            (old_uid, code_hash, expires_at),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to create migration token: {e}")

    return {"oldUid": old_uid, "migrationCode": code, "expiresAt": expires_at.isoformat() + "Z"}


def _copy_prefix(old_uid: str, new_uid: str) -> int:
    """
    Copy all objects under users/{old_uid}/ to users/{new_uid}/
    Skip if destination already exists to avoid overwriting new data.
    """
    b = get_bucket()
    src_prefix = f"users/{old_uid}/"
    dst_prefix = f"users/{new_uid}/"

    copied = 0
    for blob in b.list_blobs(prefix=src_prefix):
        src_name = blob.name
        dst_name = dst_prefix + src_name[len(src_prefix):]

        dst_blob = b.blob(dst_name)
        # skip overwrite
        if dst_blob.exists():
            continue

        b.copy_blob(blob, b, dst_name)
        copied += 1
    return copied


def _delete_prefix(old_uid: str) -> int:
    b = get_bucket()
    src_prefix = f"users/{old_uid}/"
    deleted = 0
    # list then delete
    blobs = list(b.list_blobs(prefix=src_prefix))
    for blob in blobs:
        blob.delete()
        deleted += 1
    return deleted


def _run_db_user_migration(old_uid: str, new_uid: str) -> None:
    # transactional DB move (see section 3)
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.users(firebase_uid)
                VALUES (%s)
                ON CONFLICT (firebase_uid) DO UPDATE SET last_seen_at = now()
                """,
                (new_uid,),
            )
            cur.execute("UPDATE public.pets SET user_uid=%s WHERE user_uid=%s", (new_uid, old_uid))

            cur.execute(
                """
                INSERT INTO public.user_daily_active(day, firebase_uid, first_seen_at, device_os, app_version)
                SELECT day, %s, first_seen_at, device_os, app_version
                FROM public.user_daily_active
                WHERE firebase_uid = %s
                ON CONFLICT (day, firebase_uid) DO UPDATE
                SET
                  first_seen_at = LEAST(public.user_daily_active.first_seen_at, EXCLUDED.first_seen_at),
                  device_os = COALESCE(public.user_daily_active.device_os, EXCLUDED.device_os),
                  app_version = COALESCE(public.user_daily_active.app_version, EXCLUDED.app_version)
                """,
                (new_uid, old_uid),
            )
            cur.execute("DELETE FROM public.user_daily_active WHERE firebase_uid=%s", (old_uid,))

            cur.execute(
                """
                UPDATE public.health_records
                SET
                  receipt_image_path = CASE
                    WHEN receipt_image_path IS NULL THEN NULL
                    ELSE regexp_replace(receipt_image_path, '^users/' || %s || '/', 'users/' || %s || '/')
                  END,
                  receipt_meta_path = CASE
                    WHEN receipt_meta_path IS NULL THEN NULL
                    ELSE regexp_replace(receipt_meta_path, '^users/' || %s || '/', 'users/' || %s || '/')
                  END
                WHERE
                  (receipt_image_path LIKE ('users/' || %s || '/%'))
                  OR
                  (receipt_meta_path  LIKE ('users/' || %s || '/%'))
                """,
                (old_uid, new_uid, old_uid, new_uid, old_uid, old_uid),
            )

            cur.execute(
                """
                DELETE FROM public.users u
                WHERE u.firebase_uid = %s
                  AND NOT EXISTS (SELECT 1 FROM public.pets p WHERE p.user_uid = %s)
                """,
                (old_uid, old_uid),
            )


@app.post("/api/migration/execute")
def migration_execute(req: MigrationExecuteRequest, user: Dict[str, Any] = Depends(get_current_user)):
    new_uid = user.get("uid") or ""
    if not new_uid:
        raise HTTPException(status_code=401, detail="missing uid")

    code = (req.migrationCode or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="migrationCode is required")
    code_hash = _hash_code(code)

    now = datetime.utcnow()

    # 1) lock token row (processing stale handling)
    token_row = None
    with db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM public.migration_tokens
                WHERE code_hash=%s
                FOR UPDATE
                """,
                (code_hash,),
            )
            token_row = cur.fetchone()
            if not token_row:
                raise HTTPException(status_code=404, detail="migration token not found")

            status = token_row.get("status")
            expires_at = token_row.get("expires_at")
            used_at = token_row.get("used_at")
            old_uid = token_row.get("old_uid")

            if used_at is not None or status == "completed":
                raise HTTPException(status_code=409, detail="migration token already used")

            if expires_at is not None and expires_at < now:
                raise HTTPException(status_code=410, detail="migration token expired")

            if not old_uid:
                raise HTTPException(status_code=500, detail="invalid token row (old_uid)")

            if old_uid == new_uid:
                # same uid: nothing to migrate
                cur.execute(
                    """
                    UPDATE public.migration_tokens
                    SET status='completed', used_at=now(), new_uid=%s, last_error=NULL
                    WHERE code_hash=%s
                    """,
                    (new_uid, code_hash),
                )
                return {"ok": True, "oldUid": old_uid, "newUid": new_uid, "copied": 0, "deleted": 0, "dbUpdated": False, "warnings": ["oldUid == newUid (no-op)"]}

            # handle stale processing
            if status == "processing":
                ps = token_row.get("processing_started_at")
                if ps and (now - ps).total_seconds() < int(settings.MIGRATION_PROCESSING_STALE_SECONDS):
                    raise HTTPException(status_code=409, detail="migration is already processing")
                # stale -> allow retry

            cur.execute(
                """
                UPDATE public.migration_tokens
                SET status='processing', processing_started_at=now(), new_uid=%s,
                    attempt_count = attempt_count + 1,
                    last_error = NULL
                WHERE code_hash=%s
                """,
                (new_uid, code_hash),
            )

    old_uid = token_row["old_uid"]

    # 2) Storage Copy
    copied = 0
    try:
        copied = _copy_prefix(old_uid, new_uid)
    except Exception as e:
        db_execute(
            "UPDATE public.migration_tokens SET status='failed', last_error=%s WHERE code_hash=%s",
            (f"copy failed: {e}", code_hash),
        )
        raise HTTPException(status_code=500, detail=f"storage copy failed: {e}")

    # 3) DB transactional update
    try:
        _run_db_user_migration(old_uid, new_uid)
    except Exception as e:
        db_execute(
            "UPDATE public.migration_tokens SET status='failed', last_error=%s WHERE code_hash=%s",
            (f"db migration failed: {e}", code_hash),
        )
        raise HTTPException(status_code=500, detail=f"db migration failed: {e}")

    # 4) Storage Delete old prefix
    deleted = 0
    warnings: List[str] = []
    try:
        deleted = _delete_prefix(old_uid)
    except Exception as e:
        warnings.append(f"delete failed (manual cleanup may be needed): {e}")

    # 5) finalize token
    db_execute(
        """
        UPDATE public.migration_tokens
        SET status='completed', used_at=now(), last_error=NULL
        WHERE code_hash=%s
        """,
        (code_hash,),
    )

    return {"ok": True, "oldUid": old_uid, "newUid": new_uid, "copied": copied, "deleted": deleted, "dbUpdated": True, "warnings": warnings}


# =========================================================
# Admin overview (unchanged-ish)
# =========================================================
@app.get("/api/admin/overview")
def admin_overview(admin: Dict[str, Any] = Depends(get_admin_user)):
    users_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.users") or {"c": 0}
    pets_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.pets") or {"c": 0}
    records_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.health_records") or {"c": 0}
    items_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.health_items") or {"c": 0}
    total_amount = db_fetchone("SELECT COALESCE(SUM(total_amount),0)::bigint AS s FROM public.health_records") or {"s": 0}

    return {
        "users": int(users_cnt["c"]),
        "pets": int(pets_cnt["c"]),
        "records": int(records_cnt["c"]),
        "items": int(items_cnt["c"]),
        "totalAmountSum": int(total_amount["s"]),
        "updatedAt": datetime.utcnow().isoformat() + "Z",
    }


# =========================================================
# AI analyze (기존 로직이 있으면 여기 아래에 그대로 두면 됨)
# - 이 main.py는 스토리지/마이그레이션/영수증 파이프라인이 핵심이므로
#   기존 AI 코드/태그 코드가 있다면 그대로 이어붙이세요.
# =========================================================


