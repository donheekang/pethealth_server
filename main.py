# main.py (PetHealth+ Server) - Firebase Storage + Receipt Redaction + Migration + Hospitals + PDF Vault
# Schema: v2.1.20-final (users/pets/hospitals/health_records/health_items/pet_documents/user_daily_active/migration_tokens)
import os
import io
import json
import uuid
import re
import base64
import hashlib
import secrets
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

    # (dev) local stub storage dir when STUB_MODE=true
    STUB_STORAGE_DIR: str = "./_stub_storage"

    # --- Receipt image pipeline ---
    RECEIPT_MAX_WIDTH: int = 1024
    RECEIPT_WEBP_QUALITY: int = 85  # fixed

    # Migration token TTL / processing
    MIGRATION_TOKEN_TTL_SECONDS: int = 10 * 60
    MIGRATION_PROCESSING_STALE_SECONDS: int = 5 * 60

    # --- Tier feature gating (business) ---
    # (표 기반) Guest: 영수증 3개까지 / PDF 불가
    # Member: PDF 3개까지 (맛보기) / Premium: 무제한
    GUEST_MAX_RECORDS: int = 3
    MEMBER_MAX_DOCS: int = 3

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
    """
    v2.1.20-final users upsert:
    - create row if not exists
    - bump last_seen_at
    - insert DAU row (optional)
    """
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
                RETURNING
                    firebase_uid, membership_tier, premium_until,
                    pet_count, record_count, doc_count, total_storage_bytes,
                    created_at, updated_at, last_seen_at
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


def db_get_user_quota_usage(uid: str) -> Dict[str, Any]:
    """
    Returns membership_tier, premium_until, record_count, doc_count, total_storage_bytes, quota_limit
    """
    row = db_fetchone(
        """
        SELECT
            firebase_uid,
            membership_tier,
            premium_until,
            pet_count,
            record_count,
            doc_count,
            total_storage_bytes,
            public.get_tier_quota(membership_tier) AS quota_limit
        FROM public.users
        WHERE firebase_uid=%s
        """,
        (uid,),
    )
    if not row:
        # ensure row exists
        row = db_touch_user(uid)
        # re-fetch quota using db function
        row = db_fetchone(
            """
            SELECT
                firebase_uid,
                membership_tier,
                premium_until,
                pet_count,
                record_count,
                doc_count,
                total_storage_bytes,
                public.get_tier_quota(membership_tier) AS quota_limit
            FROM public.users
            WHERE firebase_uid=%s
            """,
            (uid,),
        )
        if not row:
            raise HTTPException(status_code=500, detail="Failed to load user profile")
    return row


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
# Storage helpers (Firebase or local stub)
# =========================================================
def _stub_rel_to_abs(rel_path: str) -> str:
    # prevent path traversal
    safe = rel_path.strip().lstrip("/").replace("..", "__")
    return os.path.join(settings.STUB_STORAGE_DIR, safe)


def _stub_ensure_dir(abs_path: str) -> None:
    d = os.path.dirname(abs_path)
    os.makedirs(d, exist_ok=True)


def _stub_list(prefix: str) -> List[Dict[str, Any]]:
    base = settings.STUB_STORAGE_DIR
    if not os.path.exists(base):
        return []

    out: List[Dict[str, Any]] = []
    prefix_norm = prefix.strip().lstrip("/")
    for root, _, files in os.walk(base):
        for f in files:
            abs_path = os.path.join(root, f)
            rel = os.path.relpath(abs_path, base).replace("\\", "/")
            if not rel.startswith(prefix_norm):
                continue
            st = os.stat(abs_path)
            out.append(
                {
                    "name": rel,
                    "updated": datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
                    "size": int(st.st_size),
                    "contentType": None,
                }
            )
    return out


def _require_storage() -> None:
    if settings.STUB_MODE.lower() == "true":
        os.makedirs(settings.STUB_STORAGE_DIR, exist_ok=True)
        return
    if not settings.FIREBASE_STORAGE_BUCKET:
        raise HTTPException(status_code=503, detail="FIREBASE_STORAGE_BUCKET is not set")


def get_bucket():
    if settings.STUB_MODE.lower() == "true":
        return None
    init_firebase_admin()
    _require_storage()
    return fb_storage.bucket()


def upload_bytes_to_storage(path: str, data: bytes, content_type: str) -> str:
    """
    Upload bytes to Firebase Storage at 'path'.
    Returns the same relative path (no URL).
    """
    _require_storage()

    if settings.STUB_MODE.lower() == "true":
        abs_path = _stub_rel_to_abs(path)
        _stub_ensure_dir(abs_path)
        with open(abs_path, "wb") as f:
            f.write(data)
        return path

    b = get_bucket()
    assert b is not None
    blob = b.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    return path


def download_json_from_storage(path: str) -> Dict[str, Any]:
    _require_storage()

    if settings.STUB_MODE.lower() == "true":
        abs_path = _stub_rel_to_abs(path)
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail="object not found")
        with open(abs_path, "rb") as f:
            raw = f.read()
        return json.loads(raw.decode("utf-8"))

    b = get_bucket()
    assert b is not None
    blob = b.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="object not found")
    raw = blob.download_as_bytes()
    return json.loads(raw.decode("utf-8"))


def upload_json_to_storage(path: str, obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return upload_bytes_to_storage(path, raw, "application/json")


def delete_storage_object_if_exists(path: str) -> bool:
    _require_storage()

    if settings.STUB_MODE.lower() == "true":
        abs_path = _stub_rel_to_abs(path)
        if not os.path.exists(abs_path):
            return False
        os.remove(abs_path)
        return True

    b = get_bucket()
    assert b is not None
    blob = b.blob(path)
    if not blob.exists():
        return False
    blob.delete()
    return True


def list_storage_objects(prefix: str) -> List[Dict[str, Any]]:
    _require_storage()

    if settings.STUB_MODE.lower() == "true":
        return _stub_list(prefix)

    b = get_bucket()
    assert b is not None
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


def _doc_pdf_path(uid: str, pet_id: str, doc_type: str, doc_id: str) -> str:
    # doc_type: lab | cert
    return f"{_user_prefix(uid, pet_id)}/{doc_type}/{doc_id}.pdf"


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
                ln["cy"] = (ln["cy"] * (len(ln["words"]) - 1) + it["cy"]) / len(ln["words"])
                placed = True
                break
        if not placed:
            lines.append({"cy": it["cy"], "words": [it]})

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
    has_kw = any(k in t for k in _ADDR_KEYWORDS)
    if has_kw and re.search(r"\d", t):
        return True
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

    digit_runs = re.findall(r"\d{6,}", re.sub(r"[,\s\-]", "", t))
    if digit_runs:
        if _MONEY_HINT.search(t):
            return False
        return True

    return False


def _line_is_money_item_candidate(line_text: str) -> bool:
    t = (line_text or "").strip()
    if not t:
        return False
    if not _MONEY_NUM.search(t):
        return False
    if _MONEY_HINT.search(t):
        return True
    m = re.search(r"(\d{3,})(?!.*\d)", re.sub(r"[,\s]", "", t))
    return bool(m)


def _compute_redaction_boxes(lines: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    footer_trigger_y: Optional[int] = None

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

            if (y1 > img_h * 0.65) or _looks_like_address_line(text) or _RE_PHONE.search(text):
                footer_trigger_y = y1 if footer_trigger_y is None else min(footer_trigger_y, y1)

    if footer_trigger_y is not None:
        extra = max(40, int(img_h * 0.03))
        y = max(0, footer_trigger_y - extra)
        boxes.append((0, y, img_w, img_h))

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
    img.save(buf, format="WEBP", quality=int(quality), method=6)
    return buf.getvalue()


# =========================================================
# Receipt parsing (best-effort, PII lines excluded)
# =========================================================
def _parse_visit_date_from_text(text: str) -> Optional[date]:
    t = (text or "").strip()
    if not t:
        return None
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
            nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d{3,}", t)
            if nums:
                n = nums[-1].replace(",", "")
                try:
                    return int(n)
                except Exception:
                    pass
    return None


def _extract_items_from_lines(lines: List[str], max_items: int = 60) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ln in lines:
        if not _line_is_money_item_candidate(ln):
            continue
        nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d{3,}", ln)
        if not nums:
            continue
        price_raw = nums[-1].replace(",", "")
        try:
            price = int(price_raw)
        except Exception:
            price = None

        item_name = re.sub(re.escape(nums[-1]), "", ln, count=1).strip()
        item_name = re.sub(r"(₩|원|KRW)", "", item_name, flags=re.IGNORECASE).strip()

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
    img = Image.open(io.BytesIO(raw_image_bytes))
    img = img.convert("RGB")  # strip alpha/exif

    w, h = img.size

    words = run_vision_ocr_words(raw_image_bytes)
    lines = _group_words_into_lines(words, img_h=h)
    line_texts = [ln["text"] for ln in lines if ln.get("text")]

    boxes = _compute_redaction_boxes(lines, img_w=w, img_h=h)
    redacted = _apply_redaction(img, boxes)

    redacted_small = _resize_to_width(redacted, int(settings.RECEIPT_MAX_WIDTH))
    webp = _encode_webp(redacted_small, int(settings.RECEIPT_WEBP_QUALITY))

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
        "items": items,
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


class HospitalCustomCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    name: str
    road_address: str = Field(alias="roadAddress")
    lng: Optional[float] = None
    lat: Optional[float] = None


class MigrationPrepareResponse(BaseModel):
    oldUid: str
    migrationCode: str
    expiresAt: str


class MigrationExecuteRequest(BaseModel):
    migrationCode: str


# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(title="PetHealth+ Server", version="2.1.20-final")

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
    if settings.STUB_MODE.lower() == "true":
        os.makedirs(settings.STUB_STORAGE_DIR, exist_ok=True)


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
    return {"status": "ok", "message": "PetHealth+ Server Running (Firebase Storage, v2.1.20-final)"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "storage": "firebase" if settings.STUB_MODE.lower() != "true" else "stub",
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
    uid = user.get("uid") or ""
    prof = db_touch_user(uid) if uid else None
    if not uid:
        return {"uid": None}
    quota = db_get_user_quota_usage(uid)
    return {
        "uid": uid,
        "email": user.get("email"),
        "membershipTier": quota.get("membership_tier"),
        "premiumUntil": quota.get("premium_until"),
        "petCount": quota.get("pet_count"),
        "recordCount": quota.get("record_count"),
        "docCount": quota.get("doc_count"),
        "totalStorageBytes": quota.get("total_storage_bytes"),
        "quotaLimitBytes": quota.get("quota_limit"),
        "updatedAt": (prof or {}).get("updated_at"),
    }


# =========================================================
# Tier/Quota helper (server-side precheck)
# =========================================================
def _raise_quota_exceeded(uid: str, current: int, delta: int, limit: int):
    raise HTTPException(
        status_code=403,
        detail={
            "message": "Quota exceeded",
            "uid": uid,
            "currentBytes": current,
            "deltaBytes": delta,
            "limitBytes": limit,
        },
    )


def _precheck_quota(uid: str, delta_bytes: int) -> Dict[str, Any]:
    """
    Pre-check to avoid wasting upload/OCR cost.
    DB trigger is still final authority.
    """
    info = db_get_user_quota_usage(uid)
    current = int(info.get("total_storage_bytes") or 0)
    limit = int(info.get("quota_limit") or 0)
    if delta_bytes > 0 and (current + int(delta_bytes)) > limit:
        _raise_quota_exceeded(uid, current, int(delta_bytes), limit)
    return info


def _enforce_business_gating_for_receipt(user_info: Dict[str, Any], is_new_record: bool):
    tier = (user_info.get("membership_tier") or "guest").lower()
    if tier == "guest" and is_new_record:
        max_records = int(settings.GUEST_MAX_RECORDS)
        if max_records > 0:
            rc = int(user_info.get("record_count") or 0)
            if rc >= max_records:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "message": "Guest receipt limit reached",
                        "tier": tier,
                        "recordCount": rc,
                        "maxRecords": max_records,
                    },
                )


def _enforce_business_gating_for_pdf(user_info: Dict[str, Any]):
    tier = (user_info.get("membership_tier") or "guest").lower()
    if tier == "guest":
        raise HTTPException(
            status_code=403,
            detail={
                "message": "PDF vault is not available for Guest tier",
                "tier": tier,
            },
        )
    if tier == "member":
        max_docs = int(settings.MEMBER_MAX_DOCS)
        if max_docs > 0:
            dc = int(user_info.get("doc_count") or 0)
            if dc >= max_docs:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "message": "Member PDF limit reached (upgrade required)",
                        "tier": tier,
                        "docCount": dc,
                        "maxDocs": max_docs,
                    },
                )


# =========================================================
# Hospitals APIs (gov master + private custom)
# =========================================================
def _hospital_accessible_to_user(uid: str, hospital_mgmt_no: str) -> Optional[Dict[str, Any]]:
    return db_fetchone(
        """
        SELECT hospital_mgmt_no, name, road_address, lng, lat, is_custom_entry, created_by_uid
        FROM public.hospitals
        WHERE hospital_mgmt_no=%s
          AND (
            is_custom_entry = false
            OR created_by_uid = %s
          )
        """,
        (hospital_mgmt_no, uid),
    )


@app.get("/api/hospitals/search")
def hospitals_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=50),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    db_touch_user(uid)

    query = (q or "").strip()
    if not query:
        return []

    # pg_trgm: use similarity operator (%) + fallback ILIKE
    rows = db_fetchall(
        """
        SELECT
          hospital_mgmt_no, name, road_address, lng, lat, is_custom_entry
        FROM public.hospitals
        WHERE
          (is_custom_entry = false OR created_by_uid = %s)
          AND (
            search_vector ILIKE ('%%' || %s || '%%')
            OR search_vector %% %s
          )
        ORDER BY
          (CASE WHEN search_vector ILIKE ('%%' || %s || '%%') THEN 0 ELSE 1 END) ASC,
          similarity(search_vector, %s) DESC
        LIMIT %s
        """,
        (uid, query, query, query, query, limit),
    )
    return jsonable_encoder(rows)


@app.post("/api/hospitals/custom/create")
def hospitals_custom_create(
    req: HospitalCustomCreateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    db_touch_user(uid)

    name = (req.name or "").strip()
    road_address = (req.road_address or "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not road_address:
        raise HTTPException(status_code=400, detail="roadAddress is required")

    # (optional) dedupe within user's custom entries
    existing = db_fetchone(
        """
        SELECT hospital_mgmt_no, name, road_address, lng, lat, is_custom_entry
        FROM public.hospitals
        WHERE is_custom_entry = true
          AND created_by_uid = %s
          AND name = %s
          AND road_address = %s
        """,
        (uid, name, road_address),
    )
    if existing:
        return jsonable_encoder(existing)

    custom_no = "CUSTOM_" + uuid.uuid4().hex

    row = db_fetchone(
        """
        INSERT INTO public.hospitals
          (hospital_mgmt_no, name, road_address, lng, lat, is_custom_entry, created_by_uid)
        VALUES
          (%s, %s, %s, %s, %s, true, %s)
        RETURNING hospital_mgmt_no, name, road_address, lng, lat, is_custom_entry
        """,
        (custom_no, name, road_address, req.lng, req.lat, uid),
    )
    if not row:
        raise HTTPException(status_code=500, detail="failed to create custom hospital")
    return jsonable_encoder(row)


# =========================================================
# DB APIs
# =========================================================
@app.post("/api/db/user/upsert")
def api_user_upsert(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    row = db_touch_user(uid)
    quota = db_get_user_quota_usage(uid)
    row.update({"quota_limit": quota.get("quota_limit")})
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

    # if has_no_allergy is TRUE => tags must be empty
    if has_no_allergy is True:
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


@app.delete("/api/db/pets/delete")
def api_pets_delete(
    petId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Deletes pet row (cascade deletes records/docs/items),
    AND server tries to delete storage objects under users/{uid}/pets/{petId}/...
    """
    uid = user.get("uid") or ""
    pet_uuid = _uuid_or_400(petId, "petId")

    # ensure ownership
    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    # collect storage paths BEFORE delete (because cascade will remove rows)
    rec_paths = db_fetchall(
        """
        SELECT receipt_image_path AS path
        FROM public.health_records r
        WHERE r.pet_id=%s AND receipt_image_path IS NOT NULL
        """,
        (pet_uuid,),
    )
    doc_paths = db_fetchall(
        """
        SELECT file_path AS path
        FROM public.pet_documents d
        WHERE d.pet_id=%s
        """,
        (pet_uuid,),
    )
    paths = [r["path"] for r in rec_paths if r.get("path")] + [d["path"] for d in doc_paths if d.get("path")]

    # delete DB pet (cascade)
    db_execute("DELETE FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))

    # delete storage objects best-effort
    deleted: List[str] = []
    failed: List[str] = []
    for p in paths:
        try:
            ok = delete_storage_object_if_exists(p)
            if ok:
                deleted.append(p)
            else:
                # not found = ignore
                pass
        except Exception:
            failed.append(p)

    return {"ok": True, "petId": str(pet_uuid), "deletedStorage": deleted, "failedStorage": failed}


@app.post("/api/db/records/upsert")
def api_record_upsert(req: HealthRecordUpsertRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """
    Manual record upsert:
    - Does NOT allow client to set receipt_image_path/file_size_bytes (server-only)
    """
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
            # ownership check
            cur.execute("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="pet not found")

            # optional: validate hospital scope (better error than FK)
            if hospital_mgmt_no:
                hosp = _hospital_accessible_to_user(uid, hospital_mgmt_no)
                if not hosp:
                    raise HTTPException(status_code=404, detail="hospital not found or not accessible")

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
                    receipt_image_path, file_size_bytes,
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
                r.receipt_image_path, r.file_size_bytes,
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
                r.receipt_image_path, r.file_size_bytes,
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
    record_uuid = _uuid_or_400(recordId, "recordId")

    row = db_fetchone(
        """
        SELECT
            r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount,
            r.pet_weight_at_visit, r.tags,
            r.receipt_image_path, r.file_size_bytes,
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


@app.delete("/api/db/records/delete")
def api_record_delete(
    recordId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Delete record row AND delete receipt file if exists.
    """
    uid = user.get("uid") or ""
    record_uuid = _uuid_or_400(recordId, "recordId")

    rec = db_fetchone(
        """
        SELECT r.id, r.receipt_image_path
        FROM public.health_records r
        JOIN public.pets p ON p.id = r.pet_id
        WHERE p.user_uid=%s AND r.id=%s
        """,
        (uid, record_uuid),
    )
    if not rec:
        raise HTTPException(status_code=404, detail="record not found")

    path = rec.get("receipt_image_path")

    # delete storage first (best effort)
    warnings: List[str] = []
    if path:
        try:
            ok = delete_storage_object_if_exists(path)
            if not ok:
                warnings.append("receipt file not found in storage (already deleted?)")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"storage delete failed: {e}")

    db_execute(
        """
        DELETE FROM public.health_records
        WHERE id=%s AND EXISTS (
          SELECT 1
          FROM public.pets p
          WHERE p.id = public.health_records.pet_id AND p.user_uid=%s
        )
        """,
        (record_uuid, uid),
    )

    return {"ok": True, "recordId": str(record_uuid), "deletedReceiptPath": path, "warnings": warnings}


# =========================================================
# Receipt endpoint (server-only upload)
# =========================================================
@app.post("/api/receipts/process")
async def api_receipts_process(
    petId: str = Form(...),
    recordId: Optional[str] = Form(None),
    replaceItems: bool = Form(True),
    hospitalMgmtNo: Optional[str] = Form(None),  # optional (user selected)
    visitDateOverride: Optional[str] = Form(None),  # optional YYYY-MM-DD
    totalAmountOverride: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Pipeline:
      1) OCR (in-memory) -> aggressive redaction -> resize -> webp (quality 85 fixed)
      2) quota precheck (reduce wasted cost)
      3) upload redacted image to Firebase Storage (relative path only)
      4) upsert health_records.receipt_image_path + file_size_bytes
      5) upsert extracted health_items (optional replace)
    """
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="file/image is required")

    uid = user.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    db_touch_user(uid)

    pet_uuid = _uuid_or_400(petId, "petId")
    record_uuid = _uuid_or_new(recordId, "recordId")

    # ownership check
    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    # record exists?
    existing = db_fetchone(
        """
        SELECT r.id
        FROM public.health_records r
        JOIN public.pets p ON p.id = r.pet_id
        WHERE r.id=%s AND p.user_uid=%s
        """,
        (record_uuid, uid),
    )
    is_new_record = existing is None

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    # 1) process: OCR -> redact -> resize -> webp
    try:
        webp_bytes, parsed = process_receipt_image_and_parse(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"receipt processing failed: {e}")

    size_bytes = len(webp_bytes)

    # load user usage/tier + business gating + quota precheck
    user_info = db_get_user_quota_usage(uid)
    _enforce_business_gating_for_receipt(user_info, is_new_record=is_new_record)
    _precheck_quota(uid, size_bytes)

    # optional hospital validate
    hosp_no = (hospitalMgmtNo or "").strip() or None
    if hosp_no:
        hosp = _hospital_accessible_to_user(uid, hosp_no)
        if not hosp:
            raise HTTPException(status_code=404, detail="hospital not found or not accessible")

    # 2) upload to Storage (relative path)
    receipt_path = _receipt_path(uid, str(pet_uuid), str(record_uuid))
    try:
        upload_bytes_to_storage(receipt_path, webp_bytes, "image/webp")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"storage upload failed: {e}")

    # 3) Build DB fields from parsed + overrides
    parsed_visit = parsed.get("visitDate")
    parsed_hname = parsed.get("hospitalName")
    parsed_total = parsed.get("totalAmount")

    # visit date
    vd: date = date.today()
    if isinstance(visitDateOverride, str) and visitDateOverride.strip():
        try:
            vd = datetime.strptime(visitDateOverride.strip(), "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="visitDateOverride must be YYYY-MM-DD")
    else:
        if isinstance(parsed_visit, str) and parsed_visit:
            try:
                vd = datetime.strptime(parsed_visit, "%Y-%m-%d").date()
            except Exception:
                vd = date.today()

    # hospital name
    hn = parsed_hname.strip() if isinstance(parsed_hname, str) and parsed_hname.strip() else None

    # total amount
    if totalAmountOverride is not None:
        try:
            ta = int(totalAmountOverride)
            if ta < 0:
                raise ValueError()
        except Exception:
            raise HTTPException(status_code=400, detail="totalAmountOverride must be >= 0")
    else:
        ta = int(parsed_total) if isinstance(parsed_total, int) and parsed_total >= 0 else 0

    # items sanitize
    extracted_items = parsed.get("items") if isinstance(parsed.get("items"), list) else []
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

    # 4) upsert DB record + items (DB guard is final authority)
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO public.health_records
                      (id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, tags,
                       receipt_image_path, file_size_bytes)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s,
                       %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                      pet_id = EXCLUDED.pet_id,
                      hospital_mgmt_no = EXCLUDED.hospital_mgmt_no,
                      hospital_name = EXCLUDED.hospital_name,
                      visit_date = EXCLUDED.visit_date,
                      total_amount = EXCLUDED.total_amount,
                      tags = EXCLUDED.tags,
                      receipt_image_path = EXCLUDED.receipt_image_path,
                      file_size_bytes = EXCLUDED.file_size_bytes
                    WHERE EXISTS (
                      SELECT 1 FROM public.pets p
                      WHERE p.id = EXCLUDED.pet_id AND p.user_uid = %s
                    )
                    RETURNING
                      id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount,
                      receipt_image_path, file_size_bytes,
                      created_at, updated_at
                    """,
                    (record_uuid, pet_uuid, hosp_no, hn, vd, ta, [], receipt_path, size_bytes, uid),
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
        # DB 실패 시 업로드한 파일 정리
        try:
            delete_storage_object_if_exists(receipt_path)
        except Exception:
            pass

        # quota/guard 위반은 메시지를 더 친절히
        msg = str(e)
        if "Quota exceeded" in msg or "quota" in msg.lower():
            raise HTTPException(status_code=403, detail={"message": "Quota exceeded (DB guard)", "raw": msg})
        if "Ownership mismatch" in msg:
            raise HTTPException(status_code=400, detail={"message": "ownership/path mismatch", "raw": msg})
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"db upsert failed: {e}")


# =========================================================
# PDF Vault (lab/cert) -> pet_documents table
# =========================================================
async def _upload_document_pdf(
    doc_type: str,
    petId: str,
    file: UploadFile,
    displayName: Optional[str],
    title: Optional[str],
    user: Dict[str, Any],
):
    uid = user.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    db_touch_user(uid)

    pet_uuid = _uuid_or_400(petId, "petId")

    # ownership
    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    dt = (doc_type or "").strip().lower()
    if dt not in ("lab", "cert"):
        raise HTTPException(status_code=400, detail="doc_type must be lab or cert")

    # read
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty pdf")

    # basic pdf guard (content-type can be wrong in clients, so soft check)
    if file.content_type and "pdf" not in file.content_type.lower():
        # allow, but warn in response (no hard block)
        pass

    size_bytes = len(data)

    # tier gating + quota precheck
    user_info = db_get_user_quota_usage(uid)
    _enforce_business_gating_for_pdf(user_info)
    _precheck_quota(uid, size_bytes)

    doc_uuid = uuid.uuid4()
    doc_id = str(doc_uuid)
    file_path = _doc_pdf_path(uid, str(pet_uuid), dt, doc_id)

    # upload first
    try:
        upload_bytes_to_storage(file_path, data, "application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"storage upload failed: {e}")

    # insert DB (guard trigger will enforce ownership/quota final)
    disp = (displayName or title or file.filename or "").strip()
    if not disp:
        disp = "document.pdf"

    try:
        row = db_fetchone(
            """
            INSERT INTO public.pet_documents
              (id, pet_id, doc_type, display_name, file_path, file_size_bytes)
            VALUES
              (%s, %s, %s, %s, %s, %s)
            RETURNING
              id, pet_id, doc_type, display_name, file_path, file_size_bytes, created_at, updated_at
            """,
            (doc_uuid, pet_uuid, dt, disp[:200], file_path, size_bytes),
        )
        if not row:
            raise HTTPException(status_code=500, detail="failed to insert pet_documents")
        return jsonable_encoder(row)

    except Exception as e:
        # cleanup storage if db fails
        try:
            delete_storage_object_if_exists(file_path)
        except Exception:
            pass

        msg = str(e)
        if "Quota exceeded" in msg or "quota" in msg.lower():
            raise HTTPException(status_code=403, detail={"message": "Quota exceeded (DB guard)", "raw": msg})
        if "Ownership mismatch" in msg:
            raise HTTPException(status_code=400, detail={"message": "ownership/path mismatch", "raw": msg})
        raise HTTPException(status_code=500, detail=f"db insert failed: {e}")


@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    displayName: Optional[str] = Form(None),
    title: Optional[str] = Form(None),  # backward compatible
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    return await _upload_document_pdf("lab", petId, file, displayName, title, user)


@app.post("/api/cert/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    displayName: Optional[str] = Form(None),
    title: Optional[str] = Form(None),  # backward compatible
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    return await _upload_document_pdf("cert", petId, file, displayName, title, user)


@app.get("/api/documents/list")
def list_documents(
    petId: str = Query(...),
    docType: Optional[str] = Query(None),  # lab/cert
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or ""
    pet_uuid = _uuid_or_400(petId, "petId")

    # ownership
    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    dt = (docType or "").strip().lower()
    if dt and dt not in ("lab", "cert"):
        raise HTTPException(status_code=400, detail="docType must be lab or cert")

    if dt:
        rows = db_fetchall(
            """
            SELECT id, pet_id, doc_type, display_name, file_path, file_size_bytes, created_at, updated_at
            FROM public.pet_documents
            WHERE pet_id=%s AND doc_type=%s
            ORDER BY created_at DESC
            """,
            (pet_uuid, dt),
        )
    else:
        rows = db_fetchall(
            """
            SELECT id, pet_id, doc_type, display_name, file_path, file_size_bytes, created_at, updated_at
            FROM public.pet_documents
            WHERE pet_id=%s
            ORDER BY created_at DESC
            """,
            (pet_uuid,),
        )
    return jsonable_encoder(rows)


@app.get("/api/lab/list")
def list_lab(petId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    return list_documents(petId=petId, docType="lab", user=user)


@app.get("/api/cert/list")
def list_cert(petId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    return list_documents(petId=petId, docType="cert", user=user)


@app.delete("/api/documents/delete")
def delete_document(
    docId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or ""
    doc_uuid = _uuid_or_400(docId, "docId")

    # load + ownership check via pet
    row = db_fetchone(
        """
        SELECT d.id, d.file_path, d.pet_id
        FROM public.pet_documents d
        JOIN public.pets p ON p.id = d.pet_id
        WHERE d.id=%s AND p.user_uid=%s
        """,
        (doc_uuid, uid),
    )
    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    path = row["file_path"]

    # delete storage first
    warnings: List[str] = []
    try:
        ok = delete_storage_object_if_exists(path)
        if not ok:
            warnings.append("file not found in storage (already deleted?)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"storage delete failed: {e}")

    # delete db row
    db_execute(
        """
        DELETE FROM public.pet_documents
        WHERE id=%s AND EXISTS (
          SELECT 1
          FROM public.pets p
          WHERE p.id = public.pet_documents.pet_id AND p.user_uid=%s
        )
        """,
        (doc_uuid, uid),
    )

    return {"ok": True, "docId": str(doc_uuid), "deletedPath": path, "warnings": warnings}


@app.delete("/api/lab/delete")
def delete_lab(id: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    return delete_document(docId=id, user=user)


@app.delete("/api/cert/delete")
def delete_cert(id: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    return delete_document(docId=id, user=user)


# =========================================================
# Backup endpoints (Firebase Storage) - allowed client write in rules
# =========================================================
@app.post("/api/backup/upload", response_model=BackupUploadResponse)
async def backup_upload(req: BackupUploadRequest = Body(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or "unknown"
    backup_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat() + "Z"

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
# Migration tokens + migration execution (Storage Copy + DB migrate_user_data + Storage Delete)
# =========================================================
def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _generate_migration_code() -> str:
    return secrets.token_urlsafe(32)


def _copy_prefix(old_uid: str, new_uid: str) -> int:
    """
    Copy all objects under users/{old_uid}/ to users/{new_uid}/
    Skip overwrite if destination exists.
    """
    src_prefix = f"users/{old_uid}/"
    dst_prefix = f"users/{new_uid}/"

    if settings.STUB_MODE.lower() == "true":
        # local copy
        base = settings.STUB_STORAGE_DIR
        copied = 0
        objs = _stub_list(src_prefix)
        for o in objs:
            src_name = o["name"]
            dst_name = dst_prefix + src_name[len(src_prefix):]
            src_abs = _stub_rel_to_abs(src_name)
            dst_abs = _stub_rel_to_abs(dst_name)
            if os.path.exists(dst_abs):
                continue
            _stub_ensure_dir(dst_abs)
            with open(src_abs, "rb") as fsrc:
                raw = fsrc.read()
            with open(dst_abs, "wb") as fdst:
                fdst.write(raw)
            copied += 1
        return copied

    b = get_bucket()
    assert b is not None

    copied = 0
    for blob in b.list_blobs(prefix=src_prefix):
        src_name = blob.name
        dst_name = dst_prefix + src_name[len(src_prefix):]

        dst_blob = b.blob(dst_name)
        if dst_blob.exists():
            continue

        b.copy_blob(blob, b, dst_name)
        copied += 1
    return copied


def _delete_prefix(old_uid: str) -> int:
    src_prefix = f"users/{old_uid}/"

    if settings.STUB_MODE.lower() == "true":
        objs = _stub_list(src_prefix)
        deleted = 0
        for o in objs:
            abs_path = _stub_rel_to_abs(o["name"])
            if os.path.exists(abs_path):
                os.remove(abs_path)
                deleted += 1
        return deleted

    b = get_bucket()
    assert b is not None

    deleted = 0
    blobs = list(b.list_blobs(prefix=src_prefix))
    for blob in blobs:
        blob.delete()
        deleted += 1
    return deleted


def _run_db_user_migration(old_uid: str, new_uid: str) -> List[Dict[str, Any]]:
    """
    Calls DB function public.migrate_user_data(old_uid, new_uid)
    Returns rows: (step, value)
    """
    with db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT step, value FROM public.migrate_user_data(%s, %s)", (old_uid, new_uid))
            rows = cur.fetchall() or []
            return [dict(r) for r in rows]


@app.post("/api/migration/prepare", response_model=MigrationPrepareResponse)
def migration_prepare(user: Dict[str, Any] = Depends(get_current_user)):
    old_uid = user.get("uid") or ""
    if not old_uid:
        raise HTTPException(status_code=401, detail="missing uid")

    db_touch_user(old_uid)

    code = _generate_migration_code()
    code_hash = _hash_code(code)
    expires_at = datetime.utcnow() + timedelta(seconds=int(settings.MIGRATION_TOKEN_TTL_SECONDS))

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


@app.post("/api/migration/execute")
def migration_execute(req: MigrationExecuteRequest, user: Dict[str, Any] = Depends(get_current_user)):
    new_uid = user.get("uid") or ""
    if not new_uid:
        raise HTTPException(status_code=401, detail="missing uid")

    db_touch_user(new_uid)

    code = (req.migrationCode or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="migrationCode is required")
    code_hash = _hash_code(code)

    now = datetime.utcnow()

    # 1) lock token row + mark processing
    token_row: Optional[Dict[str, Any]] = None
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
                cur.execute(
                    """
                    UPDATE public.migration_tokens
                    SET status='completed', used_at=now(), new_uid=%s, last_error=NULL
                    WHERE code_hash=%s
                    """,
                    (new_uid, code_hash),
                )
                return {
                    "ok": True,
                    "oldUid": old_uid,
                    "newUid": new_uid,
                    "copied": 0,
                    "deleted": 0,
                    "dbUpdated": False,
                    "stats": [],
                    "warnings": ["oldUid == newUid (no-op)"],
                }

            # handle stale processing
            if status == "processing":
                ps = token_row.get("processing_started_at")
                if ps and (now - ps).total_seconds() < int(settings.MIGRATION_PROCESSING_STALE_SECONDS):
                    raise HTTPException(status_code=409, detail="migration is already processing")

            cur.execute(
                """
                UPDATE public.migration_tokens
                SET status='processing',
                    processing_started_at=now(),
                    new_uid=%s,
                    attempt_count = attempt_count + 1,
                    last_error = NULL
                WHERE code_hash=%s
                """,
                (new_uid, code_hash),
            )

    assert token_row is not None
    old_uid = token_row["old_uid"]

    # 1.5) Optional precheck: call recompute_user_stats to reduce wasted copy when target quota will fail
    try:
        db_execute("SELECT public.recompute_user_stats(%s)", (old_uid,))
        db_execute("SELECT public.recompute_user_stats(%s)", (new_uid,))
        chk = db_fetchone(
            """
            SELECT
              (SELECT total_storage_bytes FROM public.users WHERE firebase_uid=%s) AS old_usage,
              (SELECT total_storage_bytes FROM public.users WHERE firebase_uid=%s) AS new_usage,
              (SELECT membership_tier FROM public.users WHERE firebase_uid=%s) AS new_tier,
              (SELECT public.get_tier_quota(membership_tier) FROM public.users WHERE firebase_uid=%s) AS new_limit
            """,
            (old_uid, new_uid, new_uid, new_uid),
        )
        if chk:
            old_usage = int(chk.get("old_usage") or 0)
            new_usage = int(chk.get("new_usage") or 0)
            new_limit = int(chk.get("new_limit") or 0)
            if (new_usage + old_usage) > new_limit:
                db_execute(
                    "UPDATE public.migration_tokens SET status='failed', last_error=%s WHERE code_hash=%s",
                    (f"quota precheck failed: new_usage({new_usage}) + moving({old_usage}) > limit({new_limit})", code_hash),
                )
                raise HTTPException(
                    status_code=403,
                    detail={
                        "message": "Quota exceeded on target account (precheck)",
                        "oldUsage": old_usage,
                        "newUsage": new_usage,
                        "limit": new_limit,
                    },
                )
    except HTTPException:
        raise
    except Exception as e:
        # precheck failure should not block migration; DB function still has authoritative check
        print("[Migration] precheck warning:", e)

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

    # 3) DB migrate_user_data
    stats_rows: List[Dict[str, Any]] = []
    try:
        stats_rows = _run_db_user_migration(old_uid, new_uid)
    except Exception as e:
        db_execute(
            "UPDATE public.migration_tokens SET status='failed', last_error=%s WHERE code_hash=%s",
            (f"db migrate_user_data failed: {e}", code_hash),
        )
        msg = str(e)
        if "Quota exceeded" in msg or "Quota exceeded on target" in msg or "quota" in msg.lower():
            raise HTTPException(status_code=403, detail={"message": "Quota exceeded (DB migration)", "raw": msg})
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

    return {
        "ok": True,
        "oldUid": old_uid,
        "newUid": new_uid,
        "copied": copied,
        "deleted": deleted,
        "dbUpdated": True,
        "stats": stats_rows,
        "warnings": warnings,
    }


# =========================================================
# Admin overview
# =========================================================
@app.get("/api/admin/overview")
def admin_overview(admin: Dict[str, Any] = Depends(get_admin_user)):
    users_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.users") or {"c": 0}
    pets_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.pets") or {"c": 0}
    records_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.health_records") or {"c": 0}
    docs_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.pet_documents") or {"c": 0}
    items_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.health_items") or {"c": 0}
    total_amount = db_fetchone("SELECT COALESCE(SUM(total_amount),0)::bigint AS s FROM public.health_records") or {"s": 0}
    total_storage = db_fetchone("SELECT COALESCE(SUM(total_storage_bytes),0)::bigint AS s FROM public.users") or {"s": 0}

    return {
        "users": int(users_cnt["c"]),
        "pets": int(pets_cnt["c"]),
        "records": int(records_cnt["c"]),
        "documents": int(docs_cnt["c"]),
        "items": int(items_cnt["c"]),
        "totalAmountSum": int(total_amount["s"]),
        "totalStorageBytesSum": int(total_storage["s"]),
        "updatedAt": datetime.utcnow().isoformat() + "Z",
    }


