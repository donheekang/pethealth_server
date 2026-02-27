# main.py (PetHealth+ Server) — v2.3.0 (DB-aligned)
# Firebase Storage + Signed URL + Migration
# + DB schema aligned with:
#   - Soft delete SoT (deleted_at => path NULL + bytes=0) via DB triggers
#   - No undelete (enforced by DB)
#   - Quota enforcement via SELECT ... FOR UPDATE on users row (DB triggers call fn_guard_quota_for_user)
#   - Accounting via triggers (users.total_storage_bytes, counts)
#   - Storage delete jobs enqueued by DB triggers on soft-delete transition
#
# v2.3.0 additions:
#   - diagnosis docType support (pet_documents)
#   - Document meta update (clinic_name, memo, visit_date)
#   - Insurance claims CRUD
#   - Prevent schedules CRUD
#
# Architecture:
#   main.py     : API + DB I/O + pipeline wiring
#   ocr_policy  : OCR + redaction + receipt parsing -> items/meta (NO tag decision)
#   tag_policy  : items/text -> standard tag codes (SoT is your ReceiptTag codes)

from __future__ import annotations

import os
import json
import uuid
import re
import base64
import hashlib
import secrets
import threading
import logging
from contextlib import contextmanager
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple, Set, Iterable, Iterator

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Depends, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

# Firebase Admin
import firebase_admin
from firebase_admin import credentials as fb_credentials
from firebase_admin import auth as fb_auth
from firebase_admin import storage as fb_storage

# PostgreSQL
import psycopg2
from psycopg2.pool import ThreadedConnectionPool, PoolError
from psycopg2.extras import RealDictCursor
import psycopg2.extras


# =========================================================
# Logging
# =========================================================
logger = logging.getLogger("pethealth.server")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# =========================================================
# Settings
# =========================================================
class Settings(BaseSettings):
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    OCR_TIMEOUT_SECONDS: int = 12
    OCR_MAX_CONCURRENCY: int = 4
    OCR_SEMA_ACQUIRE_TIMEOUT_SECONDS: float = 1.0

    AUTH_REQUIRED: bool = True
    STUB_MODE: bool = False

    FIREBASE_ADMIN_SA_JSON: str = ""
    FIREBASE_ADMIN_SA_B64: str = ""
    FIREBASE_STORAGE_BUCKET: str = ""

    RECEIPT_MAX_WIDTH: int = 2048          # ✅ 1024→2048: OCR 해상도 향상
    RECEIPT_WEBP_QUALITY: int = 85

    MAX_RECEIPT_IMAGE_BYTES: int = 10 * 1024 * 1024
    MAX_PDF_BYTES: int = 20 * 1024 * 1024
    MAX_BACKUP_BYTES: int = 5 * 1024 * 1024
    IMAGE_MAX_PIXELS: int = 20_000_000

    SIGNED_URL_DEFAULT_TTL_SECONDS: int = 600
    SIGNED_URL_MAX_TTL_SECONDS: int = 3600

    MIGRATION_TOKEN_TTL_SECONDS: int = 10 * 60

    OCR_HOSPITAL_CANDIDATE_LIMIT: int = 3

    GEMINI_ENABLED: bool = True           # ✅ False→True: Gemini OCR 기본 활성화
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-3-flash-preview"  # ✅ 2.5-flash→3-flash: OCR 정확도 대폭 향상
    GEMINI_TIMEOUT_SECONDS: int = 60      # ✅ 10→60: Gemini 3 이미지 thinking 시간 충분히 확보

    TAG_RECORD_THRESHOLD: int = 90
    TAG_ITEM_THRESHOLD: int = 90

    DB_ENABLED: bool = True
    DATABASE_URL: str = ""
    DB_POOL_MIN: int = 2
    DB_POOL_MAX: int = 15
    DB_AUTO_UPSERT_USER: bool = True

    USER_TOUCH_THROTTLE_SECONDS: int = 300

    EXPOSE_ERROR_DETAILS: bool = False

    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = False

    ADMIN_UIDS: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


# Optional policy modules
ocr_policy = None
ocr_policy_import_error: str | None = None
try:
    import ocr_policy  # type: ignore
except Exception as e:
    ocr_policy_import_error = repr(e)
    logger.exception("[Import] ocr_policy import failed: %r", e)
    ocr_policy = None

tag_policy = None
tag_policy_import_error: str | None = None
try:
    import tag_policy  # type: ignore
except Exception as e:
    tag_policy_import_error = repr(e)
    logger.exception("[Import] tag_policy import failed: %r", e)
    tag_policy = None

def _require_module(mod, name: str):
    if mod is None:
        raise HTTPException(
            status_code=503,
            detail=f"{name} module is not installed. Upload {name}.py next to main.py",
        )


# =========================================================
# Common helpers
# =========================================================
def _new_error_id() -> str:
    return uuid.uuid4().hex[:12]


def _sanitize_for_log(s: str, max_len: int = 800) -> str:
    t = (s or "")
    t = t.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if len(t) > max_len:
        t = t[:max_len] + "...(truncated)"
    return t


def _internal_detail(msg: str, *, kind: str = "Internal Server Error") -> str:
    eid = _new_error_id()
    safe = _sanitize_for_log(msg)
    logger.error("[ERR:%s] %s: %s", eid, kind, safe)
    if settings.EXPOSE_ERROR_DETAILS:
        return f"{kind}: {msg} (errorId={eid})"
    return f"{kind} (errorId={eid})"


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


def _effective_tier_from_row(membership_tier: Optional[str], premium_until: Any) -> str:
    tier = (membership_tier or "guest").strip().lower()
    if tier not in ("guest", "member", "premium"):
        tier = "guest"
    try:
        if premium_until is not None:
            if hasattr(premium_until, "tzinfo"):
                now = datetime.now(tz=premium_until.tzinfo)
            else:
                now = datetime.utcnow()
            if premium_until > now:
                return "premium"
    except Exception:
        pass
    return tier


# =========================================================
# DB helpers
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
    if not settings.DB_ENABLED:
        logger.info("[DB] DB_ENABLED=false. Skipping DB init.")
        return
    if not settings.DATABASE_URL:
        logger.info("[DB] DATABASE_URL is empty. Skipping DB init.")
        return

    dsn = _normalize_db_url(settings.DATABASE_URL)
    try:
        psycopg2.extras.register_uuid()
        _db_pool = ThreadedConnectionPool(
            minconn=int(settings.DB_POOL_MIN),
            maxconn=int(settings.DB_POOL_MAX),
            dsn=dsn,
        )
        logger.info("[DB] Postgres pool initialized.")
    except Exception as e:
        _db_pool = None
        logger.exception("[DB] Postgres pool init failed: %s", _sanitize_for_log(repr(e)))


def _require_db() -> None:
    if not settings.DB_ENABLED:
        raise HTTPException(status_code=503, detail="DB is disabled (DB_ENABLED=false)")
    if not settings.DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not set")
    if _db_pool is None:
        init_db_pool()
    if _db_pool is None:
        raise HTTPException(status_code=503, detail="DB connection pool is not ready")


@contextmanager
def db_conn() -> Iterator[psycopg2.extensions.connection]:
    _require_db()
    assert _db_pool is not None

    conn = None
    try:
        try:
            conn = _db_pool.getconn()
        except PoolError as e:
            raise HTTPException(
                status_code=503,
                detail=_internal_detail(str(e), kind="DB pool exhausted"),
            )
        yield conn
        conn.commit()
    except Exception:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn is not None:
            try:
                _db_pool.putconn(conn)
            except Exception as e:
                logger.warning("[DB] putconn failed: %s", _sanitize_for_log(repr(e)))


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


def _pg_message(e: Exception) -> str:
    if isinstance(e, psycopg2.Error):
        try:
            if getattr(e, "diag", None) and getattr(e.diag, "message_primary", None):
                return str(e.diag.message_primary)
        except Exception:
            pass
        try:
            if getattr(e, "pgerror", None):
                return str(e.pgerror)
        except Exception:
            pass
    return str(e)


def _raise_mapped_db_error(e: Exception) -> None:
    msg = _pg_message(e)

    if "function public.get_effective_tier" in msg and "does not exist" in msg:
        raise HTTPException(status_code=503, detail="DB schema mismatch: get_effective_tier() is missing")
    if "users_membership_tier_check" in msg or "membership_tier" in msg and "check constraint" in msg:
        raise HTTPException(status_code=500, detail="Server tier mapping bug: membership_tier check failed")

    if "Quota exceeded" in msg:
        raise HTTPException(status_code=409, detail=msg)

    # v2.3.0 — structured quota error
    if "QUOTA_EXCEEDED:" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "Undelete is not allowed" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "Ownership mismatch" in msg or "Hospital access denied" in msg:
        raise HTTPException(status_code=403, detail=msg)

    if "Candidates not allowed" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "Hard DELETE is blocked" in msg:
        raise HTTPException(status_code=409, detail="Hard DELETE is blocked. Use soft delete.")

    # v2.3.0 — custom repeat constraint
    if "schedules_custom_fields_check" in msg:
        raise HTTPException(status_code=400, detail="custom repeat rule requires interval and unit")

    if "violates foreign key constraint" in msg:
        raise HTTPException(status_code=400, detail="Invalid reference (foreign key)")

    if "duplicate key value violates unique constraint" in msg:
        raise HTTPException(status_code=409, detail="Duplicate value (unique constraint)")

    raise HTTPException(status_code=500, detail=_internal_detail(msg, kind="DB error"))


# =========================================================
# Auth / membership helpers
# =========================================================
def _infer_membership_tier_from_token(decoded: Dict[str, Any]) -> Optional[str]:
    fb = decoded.get("firebase") or {}
    if isinstance(fb, dict):
        provider = (fb.get("sign_in_provider") or "").lower()
        if provider == "anonymous":
            return "guest"
        if provider:
            return "member"
    return None


def db_touch_user(firebase_uid: str, desired_tier: Optional[str] = None) -> Dict[str, Any]:
    uid = (firebase_uid or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="firebase_uid is empty")

    desired = (desired_tier or "").strip().lower() if desired_tier else None
    if desired not in (None, "guest", "member"):
        desired = None

    throttle = int(settings.USER_TOUCH_THROTTLE_SECONDS or 300)
    throttle = max(10, min(throttle, 3600))

    with db_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO public.users (firebase_uid, membership_tier)
                VALUES (%s, COALESCE(%s, 'guest'))
                ON CONFLICT (firebase_uid) DO UPDATE SET
                    last_seen_at = CASE
                        WHEN COALESCE(public.users.last_seen_at, 'epoch'::timestamptz)
                             < now() - (%s * INTERVAL '1 second')
                        THEN now()
                        ELSE public.users.last_seen_at
                    END,
                    membership_tier = CASE
                        WHEN public.users.membership_tier = 'guest'
                             AND COALESCE(%s,'guest') = 'member' THEN 'member'
                        ELSE public.users.membership_tier
                    END
                WHERE
                    (public.users.membership_tier = 'guest' AND COALESCE(%s,'guest') = 'member')
                    OR (COALESCE(public.users.last_seen_at, 'epoch'::timestamptz)
                        < now() - (%s * INTERVAL '1 second'))
                RETURNING
                    firebase_uid, membership_tier, premium_until,
                    pet_count, record_count, doc_count, total_storage_bytes,
                    created_at, updated_at, last_seen_at
                """,
                (uid, desired, throttle, desired, desired, throttle),
            )
            row = cur.fetchone()

            if not row:
                cur.execute(
                    """
                    SELECT
                        firebase_uid, membership_tier, premium_until,
                        pet_count, record_count, doc_count, total_storage_bytes,
                        created_at, updated_at, last_seen_at
                    FROM public.users
                    WHERE firebase_uid=%s
                    """,
                    (uid,),
                )
                row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=500, detail=_internal_detail("Failed to upsert user", kind="DB error"))

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
                logger.info("[DB] user_daily_active insert failed (ignored): %s", _sanitize_for_log(_pg_message(e)))

            return dict(row)


# =========================================================
# Firebase Admin init
# =========================================================
_firebase_initialized = False
auth_scheme = HTTPBearer(auto_error=False)

_stub_objects: Dict[str, Dict[str, Any]] = {}
_stub_lock = threading.RLock()


class _StubBlob:
    def __init__(self, bucket: "_StubBucket", name: str):
        self._bucket = bucket
        self.name = name

    def exists(self) -> bool:
        with _stub_lock:
            return self.name in self._bucket._objects

    def upload_from_string(self, data: bytes, content_type: str = "application/octet-stream") -> None:
        with _stub_lock:
            self._bucket._objects[self.name] = {
                "data": data or b"",
                "content_type": content_type,
                "updated": datetime.utcnow(),
                "size": int(len(data or b"")),
            }

    def download_as_bytes(self) -> bytes:
        with _stub_lock:
            obj = self._bucket._objects.get(self.name)
            if not obj:
                raise FileNotFoundError(self.name)
            return obj["data"]

    def delete(self) -> None:
        with _stub_lock:
            self._bucket._objects.pop(self.name, None)

    @property
    def updated(self) -> Optional[datetime]:
        with _stub_lock:
            obj = self._bucket._objects.get(self.name)
            return obj.get("updated") if obj else None

    @property
    def size(self) -> int:
        with _stub_lock:
            obj = self._bucket._objects.get(self.name)
            return int(obj.get("size") or 0) if obj else 0

    def generate_signed_url(self, *args, **kwargs) -> str:
        return f"/_stub/{self.name}"


class _StubBucket:
    def __init__(self, objects: Dict[str, Dict[str, Any]]):
        self._objects = objects

    def blob(self, name: str) -> _StubBlob:
        return _StubBlob(self, name)

    def list_blobs(self, prefix: str = "") -> Iterable[_StubBlob]:
        with _stub_lock:
            keys = sorted([k for k in self._objects.keys() if k.startswith(prefix)])
        for k in keys:
            yield _StubBlob(self, k)

    def copy_blob(self, blob: _StubBlob, destination_bucket: "_StubBucket", new_name: str) -> None:
        with _stub_lock:
            src = self._objects.get(blob.name)
            if not src:
                return
            if new_name in destination_bucket._objects:
                return
            destination_bucket._objects[new_name] = {
                "data": src["data"],
                "content_type": src.get("content_type") or "application/octet-stream",
                "updated": datetime.utcnow(),
                "size": int(src.get("size") or len(src["data"])),
            }


_stub_bucket_singleton = _StubBucket(_stub_objects)


def _normalize_private_key_newlines(info: dict) -> dict:
    if not isinstance(info, dict):
        return info
    pk = info.get("private_key")
    if isinstance(pk, str) and "\\n" in pk:
        info["private_key"] = pk.replace("\\n", "\n")
    return info


def _load_firebase_service_account() -> Optional[dict]:
    if settings.FIREBASE_ADMIN_SA_JSON:
        info = json.loads(settings.FIREBASE_ADMIN_SA_JSON)
        return _normalize_private_key_newlines(info)
    if settings.FIREBASE_ADMIN_SA_B64:
        raw = base64.b64decode(settings.FIREBASE_ADMIN_SA_B64).decode("utf-8")
        info = json.loads(raw)
        return _normalize_private_key_newlines(info)
    return None


def init_firebase_admin(*, require_init: bool = False) -> None:
    global _firebase_initialized

    if settings.STUB_MODE:
        _firebase_initialized = True
        return

    if firebase_admin._apps:
        _firebase_initialized = True
        return

    if _firebase_initialized and not firebase_admin._apps:
        _firebase_initialized = False

    info = _load_firebase_service_account()
    if not info:
        if require_init:
            raise RuntimeError("Firebase service account missing (set FIREBASE_ADMIN_SA_JSON or FIREBASE_ADMIN_SA_B64)")
        return

    bucket = (settings.FIREBASE_STORAGE_BUCKET or "").strip()

    try:
        cred = fb_credentials.Certificate(info)
        if bucket:
            firebase_admin.initialize_app(cred, {"storageBucket": bucket})
            logger.info("[Firebase] Admin initialized (with bucket).")
        else:
            firebase_admin.initialize_app(cred)
            logger.info("[Firebase] Admin initialized (no bucket).")
        _firebase_initialized = True
    except Exception as e:
        if require_init:
            raise RuntimeError(f"Firebase Admin initialize failed: {e}")
        logger.warning("[Firebase] init failed (ignored): %s", _sanitize_for_log(repr(e)))
        return


def _maybe_auto_upsert_user(decoded: Dict[str, Any]) -> None:
    if not settings.DB_AUTO_UPSERT_USER:
        return
    uid = (decoded.get("uid") or "").strip()
    if not uid:
        return
    try:
        desired = _infer_membership_tier_from_token(decoded)
        db_touch_user(uid, desired_tier=desired)
    except Exception as e:
        logger.info("[DB] auto upsert user failed (ignored): %s", _sanitize_for_log(_pg_message(e)))


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
) -> Dict[str, Any]:
    if settings.STUB_MODE or (not settings.AUTH_REQUIRED):
        return {"uid": "dev", "email": "dev@example.com", "firebase": {"sign_in_provider": "password"}}

    try:
        init_firebase_admin(require_init=True)
    except Exception as e:
        raise HTTPException(status_code=503, detail=_internal_detail(str(e), kind="Firebase init error"))

    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    token = credentials.credentials
    try:
        decoded = fb_auth.verify_id_token(token)
        _maybe_auto_upsert_user(decoded)
        return decoded
    except Exception as e:
        if settings.EXPOSE_ERROR_DETAILS:
            raise HTTPException(status_code=401, detail=f"Invalid Firebase token: {e}")
        raise HTTPException(status_code=401, detail=_internal_detail(str(e), kind="Invalid token"))


def _parse_admin_uids() -> Set[str]:
    raw = (settings.ADMIN_UIDS or "").strip()
    if not raw:
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}


_ADMIN_UID_SET: Set[str] = _parse_admin_uids()


def get_admin_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = (user.get("uid") or "").strip()
    if not _ADMIN_UID_SET:
        raise HTTPException(status_code=403, detail="ADMIN_UIDS is not configured")
    if uid not in _ADMIN_UID_SET:
        raise HTTPException(status_code=403, detail="Admin only")
    return user


# =========================================================
# Firebase Storage helpers
# =========================================================
def get_bucket():
    if settings.STUB_MODE:
        return _stub_bucket_singleton

    try:
        init_firebase_admin(require_init=True)
    except Exception as e:
        raise HTTPException(status_code=503, detail=_internal_detail(str(e), kind="Firebase init error"))

    bucket = (settings.FIREBASE_STORAGE_BUCKET or "").strip()
    if not bucket:
        raise HTTPException(status_code=503, detail=_internal_detail("FIREBASE_STORAGE_BUCKET is not set", kind="Firebase init error"))

    try:
        return fb_storage.bucket(bucket)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))


def upload_bytes_to_storage(path: str, data: bytes, content_type: str) -> str:
    b = get_bucket()
    blob = b.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    return path


def delete_storage_object_if_exists(path: str) -> bool:
    b = get_bucket()
    blob = b.blob(path)
    if not blob.exists():
        return False
    blob.delete()
    return True


# =========================================================
# Path helpers
# =========================================================
def _user_prefix(uid: str, pet_id: str) -> str:
    return f"users/{uid}/pets/{pet_id}"


def _backup_prefix(uid: str) -> str:
    return f"users/{uid}/backups"


def _receipt_path(uid: str, pet_id: str, record_id: str) -> str:
    """마스킹본 경로 (앱 표시용, 기본)"""
    return f"{_user_prefix(uid, pet_id)}/receipts/{record_id}.webp"


def _receipt_original_path(uid: str, pet_id: str, record_id: str) -> str:
    """원본 경로 (보험 청구 PDF용)"""
    return f"{_user_prefix(uid, pet_id)}/receipts/{record_id}_original.webp"


def _receipt_draft_path(uid: str, pet_id: str, draft_id: str) -> str:
    """드래프트 마스킹본 경로 (OCR 후 커밋 전)"""
    return f"{_user_prefix(uid, pet_id)}/draft_receipts/{draft_id}.webp"


def _receipt_draft_original_path(uid: str, pet_id: str, draft_id: str) -> str:
    """드래프트 원본 경로"""
    return f"{_user_prefix(uid, pet_id)}/draft_receipts/{draft_id}_original.webp"


def _doc_pdf_path(uid: str, pet_id: str, doc_type: str, doc_id: str) -> str:
    return f"{_user_prefix(uid, pet_id)}/{doc_type}/{doc_id}.pdf"


def _doc_file_path(uid: str, pet_id: str, doc_type: str, doc_id: str, ext: str) -> str:
    """확장자를 지정할 수 있는 문서 경로 (pdf, jpg, png, webp 등)"""
    return f"{_user_prefix(uid, pet_id)}/{doc_type}/{doc_id}.{ext}"


def _read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    try:
        upload.file.seek(0)
    except Exception:
        pass
    data = upload.file.read(int(max_bytes) + 1)
    if data is None:
        data = b""
    if len(data) > int(max_bytes):
        raise HTTPException(status_code=413, detail="file too large")
    return data


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
    visit_date: str = Field(alias="visitDate")  # ✅ "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM"
    hospital_name: Optional[str] = Field(default=None, alias="hospitalName")
    hospital_mgmt_no: Optional[str] = Field(default=None, alias="hospitalMgmtNo")
    total_amount: Optional[int] = Field(default=None, alias="totalAmount")
    pet_weight_at_visit: Optional[float] = Field(default=None, alias="petWeightAtVisit")
    receipt_image_path: Optional[str] = Field(default=None, alias="receiptImagePath")
    tags: List[str] = Field(default_factory=list)
    items: Optional[List[HealthItemInput]] = None


class HealthRecordConfirmHospitalRequest(BaseModel):
    recordId: str
    hospitalMgmtNo: str


class MigrationPrepareResponse(BaseModel):
    oldUid: str
    migrationCode: str
    expiresAt: str


class MigrationExecuteRequest(BaseModel):
    migrationCode: str


class SignedUrlResponse(BaseModel):
    path: str
    url: str
    expiresAt: str


class MeSummaryResponse(BaseModel):
    uid: str
    membership_tier: str
    effective_tier: str
    premium_until: Optional[str]
    used_bytes: int
    quota_bytes: int
    remaining_bytes: int
    pet_count: int
    record_count: int
    doc_count: int
    claim_count: int
    schedule_count: int
    ai_usage_count: int
    ai_usage_limit: Optional[int]  # None = unlimited


class DocumentUploadResponse(BaseModel):
    id: str
    petId: str
    docType: str
    displayName: str
    filePath: str
    fileSizeBytes: int
    createdAt: str
    updatedAt: str


class DocumentUpdateMetaRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    doc_id: str = Field(alias="docId")
    clinic_name: Optional[str] = Field(default=None, alias="clinicName")
    memo: Optional[str] = None
    visit_date: Optional[date] = Field(default=None, alias="visitDate")
    display_name: Optional[str] = Field(default=None, alias="displayName")


class InsuranceClaimUpsertRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str] = None
    pet_id: str = Field(alias="petId")
    claim_title: Optional[str] = Field(default=None, alias="claimTitle")
    insurance_company: Optional[str] = Field(default=None, alias="insuranceCompany")
    claim_date: Optional[date] = Field(default=None, alias="claimDate")
    date_range_start: Optional[date] = Field(default=None, alias="dateRangeStart")
    date_range_end: Optional[date] = Field(default=None, alias="dateRangeEnd")
    total_amount: Optional[int] = Field(default=None, alias="totalAmount")
    attached_documents: Optional[List[Dict[str, Any]]] = Field(default=None, alias="attachedDocuments")
    merged_pdf_path: Optional[str] = Field(default=None, alias="mergedPdfPath")
    merged_pdf_bytes: Optional[int] = Field(default=None, alias="mergedPdfBytes")
    memo: Optional[str] = None


class PreventScheduleUpsertRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str] = None
    pet_id: str = Field(alias="petId")
    title: str
    schedule_kind: str = Field(default="medical", alias="scheduleKind")
    event_date: str = Field(alias="eventDate")
    alarm_enabled: bool = Field(default=True, alias="alarmEnabled")
    repeat_rule: str = Field(default="none", alias="repeatRule")
    repeat_interval: Optional[int] = Field(default=None, alias="repeatInterval")
    repeat_unit: Optional[str] = Field(default=None, alias="repeatUnit")
    memo: Optional[str] = None


class OCRItemMapUpsertRequest(BaseModel):
    ocrItemName: str
    canonicalName: str
    isActive: Optional[bool] = True


class OCRItemMapDeactivateRequest(BaseModel):
    ocrItemName: str


# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(title="PetHealth+ Server", version="2.3.2")

_origins = [o.strip() for o in (settings.CORS_ALLOW_ORIGINS or "*").split(",") if o.strip()]
_allow_credentials = bool(settings.CORS_ALLOW_CREDENTIALS)
if "*" in _origins and _allow_credentials:
    _allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(psycopg2.Error)
async def _pg_error_handler(request: Request, exc: psycopg2.Error):
    try:
        _raise_mapped_db_error(exc)
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    return JSONResponse(status_code=500, content={"detail": _internal_detail(_pg_message(exc), kind="DB error")})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    msg = _sanitize_for_log(repr(exc), max_len=800)
    return JSONResponse(status_code=500, content={"detail": _internal_detail(msg, kind="Internal Server Error")})


# ── 🔍 OCR/Gemini 진단 엔드포인트 ──
@app.get("/api/debug/ocr-status")
async def debug_ocr_status():
    """
    브라우저에서 /api/debug/ocr-status 접속하면
    Gemini 설정 상태를 바로 확인할 수 있음
    """
    has_key = bool(settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 5)
    key_preview = (settings.GEMINI_API_KEY[:6] + "***") if has_key else "(비어있음)"

    # ocr_policy 모듈 상태 확인
    ocr_ok = False
    try:
        if ocr_policy is not None:
            ocr_ok = True
    except Exception:
        pass

    return {
        "gemini_enabled": bool(settings.GEMINI_ENABLED),
        "gemini_api_key_set": has_key,
        "gemini_api_key_preview": key_preview,
        "gemini_model": str(settings.GEMINI_MODEL_NAME),
        "gemini_timeout": int(settings.GEMINI_TIMEOUT_SECONDS),
        "receipt_max_width": int(settings.RECEIPT_MAX_WIDTH),
        "ocr_module_loaded": ocr_ok,
        "verdict": (
            "✅ Gemini 정상 — AI OCR 활성화됨"
            if (bool(settings.GEMINI_ENABLED) and has_key and ocr_ok)
            else "❌ Gemini 비활성 — "
                 + ("API 키 없음" if not has_key else "")
                 + ("GEMINI_ENABLED=False" if not bool(settings.GEMINI_ENABLED) else "")
                 + ("ocr_policy 로드 실패" if not ocr_ok else "")
        ),
    }


@app.get("/api/debug/gemini-test")
async def debug_gemini_test():
    """
    실제로 Gemini API를 호출해서 응답이 오는지 테스트.
    브라우저에서 /api/debug/gemini-test 접속
    """
    import time, traceback, httpx
    api_key = str(settings.GEMINI_API_KEY or "")
    model = str(settings.GEMINI_MODEL_NAME or "gemini-3-flash-preview")

    if not api_key:
        return {"success": False, "error": "GEMINI_API_KEY가 비어있음"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": "Say 'hello' in Korean. Reply only one word."}]}],
        "generationConfig": {"maxOutputTokens": 1024, "thinkingConfig": {"thinkingBudget": 256}},
    }

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=payload)
        elapsed = round(time.time() - t0, 2)
        body = resp.json()

        if resp.status_code == 200:
            # 원본 응답 구조를 그대로 반환해서 디버깅
            raw_parts = []
            try:
                raw_parts = body["candidates"][0]["content"]["parts"]
            except Exception:
                pass
            return {
                "success": True,
                "model": model,
                "status_code": resp.status_code,
                "raw_parts": raw_parts,
                "raw_body_preview": str(body)[:2000],
                "elapsed_sec": elapsed,
            }
        else:
            err_msg = body.get("error", {}).get("message", str(body)[:300])
            return {
                "success": False,
                "model": model,
                "status_code": resp.status_code,
                "error": err_msg,
                "elapsed_sec": elapsed,
                "verdict": f"❌ Gemini 오류 {resp.status_code}: {err_msg[:200]}",
            }
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        return {
            "success": False,
            "model": model,
            "error": str(e)[:300],
            "traceback": traceback.format_exc()[-500:],
            "elapsed_sec": elapsed,
            "verdict": f"❌ 호출 실패: {str(e)[:200]}",
        }


@app.post("/api/debug/ocr-test")
async def debug_ocr_test(request: Request):
    """
    영수증 이미지를 직접 보내서 OCR 파이프라인 전체 결과를 확인.
    curl -X POST -F "file=@receipt.jpg" https://pethealthplus.onrender.com/api/debug/ocr-test
    """
    import time as _time
    body = await request.body()
    if not body:
        return {"error": "이미지를 보내주세요 (raw body 또는 multipart)"}

    # multipart인 경우 파일 추출
    content_type = request.headers.get("content-type", "")
    raw = body
    if "multipart" in content_type:
        form = await request.form()
        f = form.get("file")
        if f:
            raw = await f.read()

    t0 = _time.time()
    try:
        _require_module(ocr_policy, "ocr_policy")
        result = ocr_policy.process_receipt_image(
            raw,
            timeout=int(settings.OCR_TIMEOUT_SECONDS),
            max_concurrency=int(settings.OCR_MAX_CONCURRENCY),
            sema_timeout=float(settings.OCR_SEMA_ACQUIRE_TIMEOUT_SECONDS),
            max_pixels=int(settings.IMAGE_MAX_PIXELS),
            receipt_max_width=int(settings.RECEIPT_MAX_WIDTH),
            receipt_webp_quality=int(settings.RECEIPT_WEBP_QUALITY),
            gemini_enabled=bool(settings.GEMINI_ENABLED),
            gemini_api_key=str(settings.GEMINI_API_KEY),
            gemini_model_name=str(settings.GEMINI_MODEL_NAME),
            gemini_timeout=int(settings.GEMINI_TIMEOUT_SECONDS),
        )
        elapsed = round(_time.time() - t0, 2)

        items = result.get("items") or []
        meta = result.get("meta") or {}
        ocr_text = result.get("ocr_text") or ""

        return {
            "success": True,
            "elapsed_sec": elapsed,
            "pipeline": meta.get("pipeline", "unknown"),
            "gemini_used": meta.get("geminiUsed", False),
            "gemini_error": meta.get("geminiError", None),
            "vision_error": meta.get("visionError", None),
            "ocr_engine": meta.get("ocrEngine", "unknown"),
            "item_count": len(items),
            "items": items[:30],
            "hospital_name": meta.get("hospital_name"),
            "visit_date": meta.get("visit_date"),
            "total_amount": result.get("total_amount"),
            "ocr_text_preview": ocr_text[:1000],
            "all_meta": {k: v for k, v in meta.items() if k != "tags"},
        }
    except Exception as e:
        import traceback
        elapsed = round(_time.time() - t0, 2)
        return {
            "success": False,
            "elapsed_sec": elapsed,
            "error": str(e)[:500],
            "traceback": traceback.format_exc()[-800:],
        }


@app.get("/api/debug/tag-test")
async def debug_tag_test(item: str = "검사 - 혈압측정"):
    """
    브라우저에서 /api/debug/tag-test?item=검사 - 혈압측정
    태그 매칭 결과를 확인
    """
    import traceback as _tb
    result = {"item": item, "TAG_ITEM_THRESHOLD": int(settings.TAG_ITEM_THRESHOLD)}
    try:
        if tag_policy is None:
            result["error"] = "tag_policy 모듈이 로드되지 않음"
            return result

        # _generate_variants 존재 여부 확인
        has_variants = hasattr(tag_policy, "_generate_variants")
        has_strip = hasattr(tag_policy, "_strip_prefixes")
        result["has_generate_variants"] = has_variants
        result["has_strip_prefixes"] = has_strip

        if has_variants:
            result["variants"] = tag_policy._generate_variants(item)

        # classify_item 호출 (실제 설정 threshold 사용)
        actual_threshold = int(settings.TAG_ITEM_THRESHOLD)
        tag_code = tag_policy.classify_item(item, threshold=actual_threshold)
        result["tag_code"] = tag_code
        result["actual_threshold_used"] = actual_threshold
        result["success"] = tag_code is not None
        # threshold=90으로도 테스트
        tag_code_90 = tag_policy.classify_item(item, threshold=90)
        result["tag_code_with_90"] = tag_code_90

        # 상위 5개 후보 점수
        candidates = []
        for tag in tag_policy.TAG_CATALOG:
            s, ev = tag_policy._match_score(tag, item)
            if s > 50:
                candidates.append({"code": tag["code"], "score": s, "why": ev.get("why", [])[:3]})
        candidates.sort(key=lambda x: x["score"], reverse=True)
        result["top_candidates"] = candidates[:5]

    except Exception as e:
        result["error"] = str(e)[:500]
        result["traceback"] = _tb.format_exc()[-500:]

    return result


@app.on_event("startup")
def _startup():
    init_db_pool()
    if settings.AUTH_REQUIRED and (not settings.STUB_MODE):
        try:
            init_firebase_admin(require_init=True)
        except Exception as e:
            logger.warning("[Startup] Firebase init failed: %s", _sanitize_for_log(repr(e)))
    # --- auto migration: 할인 항목 음수 price 허용 ---
    if settings.DB_ENABLED and settings.DATABASE_URL:
        try:
            db_execute("""
                ALTER TABLE public.health_items DROP CONSTRAINT IF EXISTS health_items_price_nonneg
            """)
            logger.info("[Startup] Dropped health_items_price_nonneg constraint (allow negative prices for discounts)")
        except Exception as e:
            logger.warning("[Startup] Could not drop price constraint: %s", _sanitize_for_log(repr(e)))


@app.on_event("shutdown")
def _shutdown():
    global _db_pool
    if _db_pool is not None:
        try:
            _db_pool.closeall()
            logger.info("[DB] Pool closed.")
        except Exception as e:
            logger.warning("[DB] Pool close error: %s", _sanitize_for_log(repr(e)))
        _db_pool = None


@app.get("/")
def root():
    return {"status": "ok", "message": "PetHealth+ Server Running (v2.3.1)"}


@app.get("/api/health")
def health():
    db_checks: Dict[str, Any] = {"ok": None, "missing": []}
    if settings.DB_ENABLED and settings.DATABASE_URL:
        try:
            checks = [
                ("users", "public.users"),
                ("pets", "public.pets"),
                ("hospitals", "public.hospitals"),
                ("health_records", "public.health_records"),
                ("health_items", "public.health_items"),
                ("pet_documents", "public.pet_documents"),
                ("candidates", "public.health_record_hospital_candidates"),
                ("ocr_item_maps", "public.ocr_item_name_maps"),
                ("migration_tokens", "public.migration_tokens"),
                ("storage_delete_jobs", "public.storage_delete_jobs"),
                ("insurance_claims", "public.insurance_claims"),
                ("prevent_schedules", "public.prevent_schedules"),
            ]
            missing = []
            for _, reg in checks:
                r = db_fetchone("SELECT to_regclass(%s) AS c", (reg,))
                if not r or not r.get("c"):
                    missing.append(reg)
            db_checks["ok"] = (len(missing) == 0)
            db_checks["missing"] = missing
        except Exception as e:
            db_checks["ok"] = False
            db_checks["error"] = _pg_message(e)

    fb_check: Dict[str, Any] = {
        "stubMode": bool(settings.STUB_MODE),
        "bucketProvided": bool((settings.FIREBASE_STORAGE_BUCKET or "").strip()),
        "serviceAccountProvided": bool((settings.FIREBASE_ADMIN_SA_JSON or "").strip() or (settings.FIREBASE_ADMIN_SA_B64 or "").strip()),
        "appInitialized": bool(firebase_admin._apps),
    }

    return {
        "status": "ok",
        "version": "2.3.1",
        "storage": "stub" if settings.STUB_MODE else "firebase",
        "db_enabled": bool(settings.DB_ENABLED),
        "db_configured": bool(settings.DATABASE_URL),
        "db_schema": db_checks,
        "firebase": fb_check,
        "cors": {"origins": _origins, "allowCredentials": _allow_credentials},
        "modules": {
            "ocr_policy": bool(ocr_policy is not None),
            "tag_policy": bool(tag_policy is not None),
            "ocr_policy_error": (ocr_policy_import_error if settings.EXPOSE_ERROR_DETAILS else None),
            "tag_policy_error": (tag_policy_import_error if settings.EXPOSE_ERROR_DETAILS else None),
        },
    }


@app.get("/api/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid")
    desired = _infer_membership_tier_from_token(user)
    row = None
    try:
        if uid:
            row = db_touch_user(uid, desired_tier=desired)
    except Exception:
        row = None
    return {"uid": uid, "email": user.get("email"), "user": row}


# ─── 계정 삭제 (App Store 5.1.1(v) 대응) ───
@app.delete("/api/me/delete")
def me_delete(user: Dict[str, Any] = Depends(get_current_user)):
    """
    사용자 계정 및 관련 데이터를 모두 soft-delete 합니다.
    - prevent_schedules, insurance_claims, pet_documents, health_records → soft delete
    - health_items → hard delete (record 종속)
    - pets → hard delete
    - users → hard delete
    - Storage(GCS) → 유저 폴더 전체 삭제
    """
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    summary = {}
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            # 1) 스케줄 soft delete
            cur.execute(
                "UPDATE public.prevent_schedules SET deleted_at = now() WHERE user_uid=%s AND deleted_at IS NULL",
                (uid,),
            )
            summary["schedules_deleted"] = cur.rowcount

            # 2) 보험청구 soft delete
            cur.execute(
                "UPDATE public.insurance_claims SET deleted_at = now() WHERE user_uid=%s AND deleted_at IS NULL",
                (uid,),
            )
            summary["claims_deleted"] = cur.rowcount

            # 3) 문서 soft delete
            cur.execute(
                """UPDATE public.pet_documents d SET deleted_at = now()
                   WHERE d.deleted_at IS NULL
                     AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = d.pet_id AND p.user_uid = %s)""",
                (uid,),
            )
            summary["documents_deleted"] = cur.rowcount

            # 4) health_items hard delete (record 종속)
            cur.execute(
                """DELETE FROM public.health_items
                   WHERE record_id IN (
                       SELECT r.id FROM public.health_records r
                       JOIN public.pets p ON p.id = r.pet_id
                       WHERE p.user_uid = %s
                   )""",
                (uid,),
            )
            summary["health_items_deleted"] = cur.rowcount

            # 5) 진료기록 soft delete
            cur.execute(
                """UPDATE public.health_records r SET deleted_at = now()
                   WHERE r.deleted_at IS NULL
                     AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = r.pet_id AND p.user_uid = %s)""",
                (uid,),
            )
            summary["records_deleted"] = cur.rowcount

            # 6) 펫 삭제
            cur.execute("DELETE FROM public.pets WHERE user_uid=%s", (uid,))
            summary["pets_deleted"] = cur.rowcount

            # 7) 유저 삭제
            cur.execute("DELETE FROM public.users WHERE firebase_uid=%s", (uid,))
            summary["user_deleted"] = cur.rowcount

        conn.commit()
    except Exception as e:
        logger.error("[me/delete] DB error: %s", repr(e))
        raise HTTPException(status_code=500, detail="account deletion failed")

    # 8) Storage 파일 삭제 (실패해도 DB 삭제는 유지)
    storage_deleted = 0
    try:
        storage_deleted = _delete_prefix(uid)
    except Exception as e:
        logger.warning("[me/delete] storage cleanup error: %s", repr(e))
    summary["storage_files_deleted"] = storage_deleted

    logger.info("[me/delete] uid=%s summary=%s", uid, summary)
    return {"ok": True, "uid": uid, "summary": summary}


@app.get("/api/me/summary", response_model=MeSummaryResponse)
def me_summary(user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    desired = _infer_membership_tier_from_token(user)
    _ = db_touch_user(uid, desired_tier=desired)

    row = db_fetchone(
        """
        SELECT
            firebase_uid,
            membership_tier,
            premium_until,
            pet_count,
            record_count,
            doc_count,
            COALESCE(claim_count, 0) AS claim_count,
            COALESCE(schedule_count, 0) AS schedule_count,
            COALESCE(ai_usage_count, 0) AS ai_usage_count,
            total_storage_bytes AS used_bytes
        FROM public.users
        WHERE firebase_uid = %s
        """,
        (uid,),
    )
    if not row:
        raise HTTPException(status_code=500, detail=_internal_detail("user row missing", kind="DB error"))

    membership_tier = str(row.get("membership_tier") or "guest")
    premium_until_raw = row.get("premium_until")
    effective_tier = _effective_tier_from_row(membership_tier, premium_until_raw)

    quota_row = db_fetchone("SELECT public.get_tier_quota(%s) AS quota_bytes", (effective_tier,))
    quota = int((quota_row or {}).get("quota_bytes") or 0)

    used = int(row.get("used_bytes") or 0)
    remaining = max(0, quota - used)

    pu = premium_until_raw
    premium_until = pu.isoformat() if hasattr(pu, "isoformat") and pu is not None else None

    # ✅ claim_count 실시간 동기화: 실제 건수를 세서 반환 + DB 업데이트
    actual_claim_count = 0
    try:
        cnt_row = db_fetchone(
            "SELECT COUNT(*) AS c FROM public.insurance_claims WHERE user_uid=%s AND deleted_at IS NULL",
            (uid,),
        )
        actual_claim_count = int(cnt_row["c"]) if cnt_row else 0
        stored_claim_count = int(row.get("claim_count") or 0)
        if actual_claim_count != stored_claim_count:
            db_execute("UPDATE public.users SET claim_count = %s WHERE firebase_uid = %s", (actual_claim_count, uid))
    except Exception:
        actual_claim_count = int(row.get("claim_count") or 0)

    return {
        "uid": uid,
        "membership_tier": membership_tier,
        "effective_tier": effective_tier,
        "premium_until": premium_until,
        "used_bytes": used,
        "quota_bytes": quota,
        "remaining_bytes": int(remaining),
        "pet_count": int(row.get("pet_count") or 0),
        "record_count": int(row.get("record_count") or 0),
        "doc_count": int(row.get("doc_count") or 0),
        "claim_count": actual_claim_count,
        "schedule_count": int(row.get("schedule_count") or 0),
        "ai_usage_count": int(row.get("ai_usage_count") or 0),
        "ai_usage_limit": None if effective_tier == "premium" else 3,
    }


# =========================================================
# Users
# =========================================================
@app.post("/api/db/user/upsert")
def api_user_upsert(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    desired = _infer_membership_tier_from_token(user)
    row = db_touch_user(uid, desired_tier=desired)
    return jsonable_encoder(row)


# =========================================================
# Pets
# =========================================================
@app.post("/api/db/pets/upsert")
def api_pet_upsert(req: PetUpsertRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)

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

    has_no_allergy: Optional[bool] = req.has_no_allergy
    allergy_tags = [str(x).strip() for x in (req.allergy_tags or []) if str(x).strip()]
    if has_no_allergy is True:
        allergy_tags = []

    try:
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
        raise HTTPException(status_code=500, detail=_internal_detail("Failed to upsert pet", kind="DB error"))
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.get("/api/db/pets/list")
def api_pets_list(user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
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


# =========================================================
# Hospitals
# =========================================================
def _validate_lat_lng(lat: Optional[float], lng: Optional[float]) -> None:
    if lat is not None:
        try:
            v = float(lat)
        except Exception:
            raise HTTPException(status_code=400, detail="lat must be a number")
        if v < -90 or v > 90:
            raise HTTPException(status_code=400, detail="lat must be between -90 and 90")
    if lng is not None:
        try:
            v = float(lng)
        except Exception:
            raise HTTPException(status_code=400, detail="lng must be a number")
        if v < -180 or v > 180:
            raise HTTPException(status_code=400, detail="lng must be between -180 and 180")


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    import math
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


@app.get("/api/hospitals/search")
def hospitals_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)
    query = (q or "").strip()
    like = f"%{query}%"
    rows = db_fetchall(
        "SELECT hospital_mgmt_no, name, road_address, jibun_address, lng, lat, is_custom_entry FROM public.hospitals WHERE (is_custom_entry = false OR created_by_uid = %s) AND search_vector ILIKE %s ORDER BY similarity(search_vector, %s) DESC LIMIT %s",
        (uid, like, query, limit),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({"hospitalMgmtNo": r["hospital_mgmt_no"], "name": r["name"], "roadAddress": r.get("road_address"), "jibunAddress": r.get("jibun_address"), "lng": r.get("lng"), "lat": r.get("lat"), "isCustomEntry": bool(r.get("is_custom_entry"))})
    return out


@app.get("/api/hospitals/nearby")
def hospitals_nearby(lat: float = Query(...), lng: float = Query(...), radiusM: int = Query(3000, ge=200, le=20000), limit: int = Query(50, ge=1, le=200), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)
    _validate_lat_lng(lat, lng)
    import math
    lat_delta = radiusM / 111000.0
    lng_delta = radiusM / (111000.0 * max(0.2, math.cos(math.radians(lat))))
    min_lat, max_lat = lat - lat_delta, lat + lat_delta
    min_lng, max_lng = lng - lng_delta, lng + lng_delta
    rows = db_fetchall(
        "SELECT hospital_mgmt_no, name, road_address, jibun_address, lat, lng, is_custom_entry FROM public.hospitals WHERE (is_custom_entry = false OR created_by_uid = %s) AND lat IS NOT NULL AND lng IS NOT NULL AND lat BETWEEN %s AND %s AND lng BETWEEN %s AND %s LIMIT %s",
        (uid, min_lat, max_lat, min_lng, max_lng, limit * 8),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = _haversine_m(lat, lng, r["lat"], r["lng"])
        if d <= radiusM:
            out.append({"hospitalMgmtNo": r["hospital_mgmt_no"], "name": r["name"], "roadAddress": r.get("road_address"), "jibunAddress": r.get("jibun_address"), "lat": r["lat"], "lng": r["lng"], "distanceM": int(d), "isCustomEntry": bool(r.get("is_custom_entry"))})
    out.sort(key=lambda x: x["distanceM"])
    return out[:limit]


class HospitalCustomCreateRequest(BaseModel):
    name: str
    roadAddress: Optional[str] = None
    jibunAddress: Optional[str] = None
    lng: Optional[float] = None
    lat: Optional[float] = None


@app.post("/api/hospitals/custom/create")
def hospitals_custom_create(req: HospitalCustomCreateRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    road = (req.roadAddress or "").strip() if isinstance(req.roadAddress, str) else ""
    jibun = (req.jibunAddress or "").strip() if isinstance(req.jibunAddress, str) else ""
    road_val = road if road else None
    jibun_val = jibun if jibun else None
    _validate_lat_lng(req.lat, req.lng)
    mgmt_no = "CUSTOM_" + uuid.uuid4().hex
    try:
        row = db_fetchone(
            "INSERT INTO public.hospitals (hospital_mgmt_no, name, road_address, jibun_address, lng, lat, is_custom_entry, created_by_uid) VALUES (%s, %s, %s, %s, %s, %s, true, %s) RETURNING hospital_mgmt_no, name, road_address, jibun_address, lng, lat, is_custom_entry",
            (mgmt_no, name, road_val, jibun_val, req.lng, req.lat, uid),
        )
        if not row:
            raise HTTPException(status_code=500, detail=_internal_detail("failed to create hospital", kind="DB error"))
        return {"hospitalMgmtNo": row["hospital_mgmt_no"], "name": row["name"], "roadAddress": row.get("road_address"), "jibunAddress": row.get("jibun_address"), "lng": row.get("lng"), "lat": row.get("lat"), "isCustomEntry": True}
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.delete("/api/hospitals/custom/delete")
def hospitals_custom_delete(hospitalMgmtNo: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)
    mg = (hospitalMgmtNo or "").strip()
    if not mg:
        raise HTTPException(status_code=400, detail="hospitalMgmtNo is required")
    row = db_fetchone("SELECT hospital_mgmt_no FROM public.hospitals WHERE hospital_mgmt_no=%s AND is_custom_entry=true AND created_by_uid=%s", (mg, uid))
    if not row:
        raise HTTPException(status_code=404, detail="custom hospital not found")
    try:
        n = db_execute("DELETE FROM public.hospitals WHERE hospital_mgmt_no=%s", (mg,))
        if n <= 0:
            raise HTTPException(status_code=404, detail="not found")
        return {"ok": True, "deleted": mg}
    except HTTPException:
        raise
    except Exception as e:
        msg = _pg_message(e)
        if "violates foreign key constraint" in msg:
            raise HTTPException(status_code=409, detail="hospital is in use by records; detach it first")
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Records + items
# =========================================================
@app.post("/api/db/records/upsert")
def api_record_upsert(req: HealthRecordUpsertRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)
    record_uuid = _uuid_or_new(req.id, "id")
    pet_uuid = _uuid_or_400(req.pet_id, "petId")
    # ✅ visit_date: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM" → datetime
    from datetime import datetime as _dt
    _vd_raw = (req.visit_date or "").strip()
    try:
        if "T" in _vd_raw and len(_vd_raw) >= 16:
            visit_date = _dt.fromisoformat(_vd_raw)
        else:
            visit_date = _dt.combine(date.fromisoformat(_vd_raw[:10]), _dt.min.time())
    except Exception:
        visit_date = _dt.now()
    hospital_name = req.hospital_name.strip() if isinstance(req.hospital_name, str) and req.hospital_name.strip() else None
    hospital_mgmt_no = req.hospital_mgmt_no.strip() if isinstance(req.hospital_mgmt_no, str) and req.hospital_mgmt_no.strip() else None
    total_amount_in: Optional[int] = None
    if req.total_amount is not None:
        try:
            total_amount_in = int(req.total_amount)
        except Exception:
            raise HTTPException(status_code=400, detail="totalAmount must be a number")
        if total_amount_in < 0:
            raise HTTPException(status_code=400, detail="totalAmount must be >= 0")
    pet_weight_in: Optional[float] = None
    if req.pet_weight_at_visit is not None:
        try:
            pet_weight_in = float(req.pet_weight_at_visit)
        except Exception:
            raise HTTPException(status_code=400, detail="petWeightAtVisit must be a number")
        if pet_weight_in <= 0:
            raise HTTPException(status_code=400, detail="petWeightAtVisit must be > 0")
    items_sum = 0
    has_any_price = False
    if req.items is not None:
        for it in req.items:
            if it.price is not None:
                try:
                    p = int(it.price)
                except Exception:
                    raise HTTPException(status_code=400, detail="item price must be a number")
                # 음수 가격 허용 (절사할인, 쿠폰할인 등 할인 항목)
                has_any_price = True
                items_sum += p
    tags = _clean_tags(req.tags)
    # ✅ receipt_image_path: 편집 시 이미지 경로 저장 지원
    receipt_image_path_in = req.receipt_image_path.strip() if isinstance(req.receipt_image_path, str) and req.receipt_image_path.strip() else None
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, weight_kg FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
                pet_row = cur.fetchone()
                if not pet_row:
                    raise HTTPException(status_code=404, detail="pet not found")
                pet_weight_at_visit = pet_weight_in
                if pet_weight_at_visit is None:
                    try:
                        wkg = pet_row.get("weight_kg")
                        pet_weight_at_visit = float(wkg) if wkg is not None else None
                    except Exception:
                        pet_weight_at_visit = None
                total_amount = total_amount_in
                if total_amount is None or total_amount <= 0:
                    if has_any_price and items_sum > 0:
                        total_amount = items_sum
                    else:
                        total_amount = 0
                # ✅ receipt_image_path가 있으면 INSERT/UPDATE에 포함
                #    file_size_bytes도 함께 포함해야 health_records_file_presence_check 제약 조건 통과
                if receipt_image_path_in:
                    # 기존 레코드에서 file_size_bytes 조회 (upsert 시 보존)
                    existing_fsb = db_fetchone(
                        "SELECT file_size_bytes FROM public.health_records WHERE id=%s",
                        (record_uuid,),
                    )
                    file_size_val = existing_fsb["file_size_bytes"] if existing_fsb and existing_fsb.get("file_size_bytes") else 0
                    cur.execute(
                        """
                        INSERT INTO public.health_records
                            (id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags, receipt_image_path, file_size_bytes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            pet_id = EXCLUDED.pet_id, hospital_mgmt_no = EXCLUDED.hospital_mgmt_no,
                            hospital_name = EXCLUDED.hospital_name, visit_date = EXCLUDED.visit_date,
                            total_amount = EXCLUDED.total_amount, pet_weight_at_visit = EXCLUDED.pet_weight_at_visit,
                            tags = EXCLUDED.tags, receipt_image_path = EXCLUDED.receipt_image_path,
                            file_size_bytes = EXCLUDED.file_size_bytes
                        WHERE public.health_records.deleted_at IS NULL
                            AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = public.health_records.pet_id AND p.user_uid = %s)
                        RETURNING id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags,
                            receipt_image_path, file_size_bytes, created_at, updated_at
                        """,
                        (record_uuid, pet_uuid, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags, receipt_image_path_in, file_size_val, uid),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO public.health_records
                            (id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            pet_id = EXCLUDED.pet_id, hospital_mgmt_no = EXCLUDED.hospital_mgmt_no,
                            hospital_name = EXCLUDED.hospital_name, visit_date = EXCLUDED.visit_date,
                            total_amount = EXCLUDED.total_amount, pet_weight_at_visit = EXCLUDED.pet_weight_at_visit,
                            tags = EXCLUDED.tags
                        WHERE public.health_records.deleted_at IS NULL
                            AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = public.health_records.pet_id AND p.user_uid = %s)
                        RETURNING id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags,
                            receipt_image_path, file_size_bytes, created_at, updated_at
                        """,
                        (record_uuid, pet_uuid, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags, uid),
                    )
                row = cur.fetchone()
                if not row:
                    exists = db_fetchone("SELECT id FROM public.health_records WHERE id=%s", (record_uuid,))
                    if exists:
                        raise HTTPException(status_code=409, detail="record is deleted or not editable")
                    raise HTTPException(status_code=500, detail=_internal_detail("Failed to upsert record", kind="DB error"))
                if req.items is not None:
                    cur.execute("DELETE FROM public.health_items WHERE record_id=%s", (record_uuid,))
                    for it in req.items:
                        item_name = (it.item_name or "").strip()
                        if not item_name:
                            continue
                        price = it.price
                        if price is not None:
                            price = int(price)
                            # 음수 가격 허용 (절사할인, 쿠폰할인 등 할인 항목)
                        category_tag = it.category_tag.strip() if isinstance(it.category_tag, str) and it.category_tag.strip() else None
                        cur.execute("INSERT INTO public.health_items (record_id, item_name, price, category_tag) VALUES (%s, %s, %s, %s)", (record_uuid, item_name, price, category_tag))
                cur.execute("SELECT id, record_id, item_name, price, category_tag, created_at, updated_at FROM public.health_items WHERE record_id=%s ORDER BY created_at ASC", (record_uuid,))
                items_rows = cur.fetchall() or []
                payload = dict(row)
                payload["items"] = [dict(x) for x in items_rows]
                # pet 정보 추가 (종, 이름)
                pet_info = db_fetchone("SELECT name, species FROM public.pets WHERE id=%s", (pet_uuid,))
                if pet_info:
                    payload["pet_name"] = pet_info.get("name")
                    payload["pet_species"] = pet_info.get("species")
                return jsonable_encoder(payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "[UPSERT-ERROR] record_id=%s, pet_id=%s, hospital=%s, total=%s, "
            "receipt_path=%s, items_count=%s, error=%s: %s",
            str(record_uuid), str(pet_uuid), hospital_name, total_amount_in,
            (req.receipt_image_path or "")[:80],
            len(req.items) if req.items else 0,
            type(e).__name__, _sanitize_for_log(str(e)),
        )
        _raise_mapped_db_error(e)
        raise


@app.get("/api/db/records/list")
def api_records_list(petId: Optional[str] = Query(None), includeItems: bool = Query(False), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    try:
        desired = _infer_membership_tier_from_token(user)
        db_touch_user(uid, desired_tier=desired)
    except Exception:
        pass
    try:
        if petId:
            pet_uuid = _uuid_or_400(petId, "petId")
            rows = db_fetchall(
                "SELECT r.id, r.pet_id, p.name AS pet_name, p.species AS pet_species, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount, r.pet_weight_at_visit, r.tags, r.receipt_image_path, r.file_size_bytes, r.created_at, r.updated_at FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND p.id=%s AND r.deleted_at IS NULL ORDER BY r.visit_date DESC, r.created_at DESC",
                (uid, pet_uuid),
            )
        else:
            rows = db_fetchall(
                "SELECT r.id, r.pet_id, p.name AS pet_name, p.species AS pet_species, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount, r.pet_weight_at_visit, r.tags, r.receipt_image_path, r.file_size_bytes, r.created_at, r.updated_at FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND r.deleted_at IS NULL ORDER BY r.visit_date DESC, r.created_at DESC",
                (uid,),
            )
        if not includeItems:
            return jsonable_encoder(rows)
        record_ids = [r["id"] for r in rows if r.get("id")]
        if not record_ids:
            return jsonable_encoder(rows)
        items = db_fetchall("SELECT id, record_id, item_name, price, category_tag, created_at, updated_at FROM public.health_items WHERE record_id = ANY(%s::uuid[]) ORDER BY created_at ASC", (record_ids,))
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
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.get("/api/db/records/get")
def api_record_get(recordId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    rid = (recordId or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="recordId is required")
    try:
        record_uuid = _uuid_or_400(rid, "recordId")
        row = db_fetchone(
            "SELECT r.id, r.pet_id, p.name AS pet_name, p.species AS pet_species, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount, r.pet_weight_at_visit, r.tags, r.receipt_image_path, r.file_size_bytes, r.created_at, r.updated_at FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND r.id=%s AND r.deleted_at IS NULL",
            (uid, record_uuid),
        )
        if not row:
            raise HTTPException(status_code=404, detail="record not found")
        items = db_fetchall("SELECT id, record_id, item_name, price, category_tag, created_at, updated_at FROM public.health_items WHERE record_id=%s ORDER BY created_at ASC", (record_uuid,))
        row["items"] = items
        return jsonable_encoder(row)
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.delete("/api/db/records/delete")
def api_record_delete(recordId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    record_uuid = _uuid_or_400(recordId, "recordId")
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT r.id, r.receipt_image_path FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND r.id=%s AND r.deleted_at IS NULL", (uid, record_uuid))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="record not found (or already deleted)")
                old_path = row.get("receipt_image_path")
                cur.execute("UPDATE public.health_records r SET deleted_at = now() WHERE r.id=%s AND r.deleted_at IS NULL AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = r.pet_id AND p.user_uid = %s)", (record_uuid, uid))
                if (cur.rowcount or 0) <= 0:
                    raise HTTPException(status_code=404, detail="record not found (or already deleted)")
        return {"ok": True, "recordId": str(record_uuid), "softDeleted": True, "oldReceiptPath": old_path, "storageDeletion": "queued_by_trigger"}
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Record -> hospital candidates + confirm
# =========================================================
@app.get("/api/db/records/hospital-candidates")
def api_record_hospital_candidates(recordId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    record_uuid = _uuid_or_400(recordId, "recordId")
    rows = db_fetchall(
        "SELECT c.rank, c.score, h.hospital_mgmt_no, h.name, h.road_address, h.jibun_address, h.lat, h.lng, h.is_custom_entry FROM public.health_record_hospital_candidates c JOIN public.hospitals h ON h.hospital_mgmt_no = c.hospital_mgmt_no JOIN public.health_records r ON r.id = c.record_id JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid = %s AND r.id = %s AND r.deleted_at IS NULL ORDER BY c.rank ASC",
        (uid, record_uuid),
    )
    out = []
    for r in rows:
        out.append({"rank": int(r["rank"]), "score": float(r["score"]) if r.get("score") is not None else None, "hospitalMgmtNo": r["hospital_mgmt_no"], "name": r["name"], "roadAddress": r.get("road_address"), "jibunAddress": r.get("jibun_address"), "lat": r.get("lat"), "lng": r.get("lng"), "isCustomEntry": bool(r.get("is_custom_entry"))})
    return out


@app.post("/api/db/records/confirm-hospital")
def api_record_confirm_hospital(req: HealthRecordConfirmHospitalRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    record_uuid = _uuid_or_400(req.recordId, "recordId")
    mgmt = (req.hospitalMgmtNo or "").strip()
    if not mgmt:
        raise HTTPException(status_code=400, detail="hospitalMgmtNo is required")
    rec = db_fetchone("SELECT r.id FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid = %s AND r.id = %s AND r.deleted_at IS NULL", (uid, record_uuid))
    if not rec:
        raise HTTPException(status_code=404, detail="record not found")
    hosp = db_fetchone("SELECT hospital_mgmt_no, is_custom_entry, created_by_uid FROM public.hospitals WHERE hospital_mgmt_no=%s", (mgmt,))
    if not hosp:
        raise HTTPException(status_code=400, detail="Invalid hospitalMgmtNo")
    if hosp.get("is_custom_entry") and (hosp.get("created_by_uid") or "") != uid:
        raise HTTPException(status_code=403, detail="custom hospital belongs to another user")
    try:
        row = db_fetchone(
            "UPDATE public.health_records r SET hospital_mgmt_no = %s WHERE r.id = %s AND r.deleted_at IS NULL AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = r.pet_id AND p.user_uid = %s) RETURNING r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount, r.pet_weight_at_visit, r.tags, r.receipt_image_path, r.file_size_bytes, r.created_at, r.updated_at",
            (mgmt, record_uuid, uid),
        )
        if not row:
            raise HTTPException(status_code=500, detail=_internal_detail("failed to confirm hospital", kind="DB error"))
        items = db_fetchall("SELECT id, record_id, item_name, price, category_tag, created_at, updated_at FROM public.health_items WHERE record_id=%s ORDER BY created_at ASC", (record_uuid,))
        row["items"] = items
        row["hospitalCandidatesCleared"] = True
        return jsonable_encoder(row)
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Fallback tag classification (when tag_policy.py is not available)
# =========================================================
_TAG_KEYWORDS: Dict[str, List[str]] = {
    "vaccine": ["백신", "vaccination", "vaccine", "접종", "rabies", "광견병", "dhppl", "dhlpp",
                 "종합백신", "코로나장염", "켄넬코프", "인플루엔자", "보르데텔라"],
    "surgery": ["수술", "surgery", "중성화", "거세", "spay", "neuter", "마취", "절개",
                "봉합", "발치", "슬개골", "십자인대", "종양제거"],
    "dental": ["치석", "스케일링", "dental", "scaling", "치과", "치아", "발치", "치주",
               "구강", "잇몸"],
    "checkup": ["건강검진", "종합검진", "checkup", "check-up", "신체검사", "정기검진",
                "혈액검사", "혈검", "blood test", "cbc"],
    "lab": ["혈액", "blood", "소변", "urine", "x-ray", "xray", "초음파", "ultrasound",
            "방사선", "ct", "mri", "내시경", "심전도", "검사", "분석", "배양"],
    "medicine": ["약", "처방", "prescription", "drug", "medicine", "항생제", "소염제",
                 "진통제", "위장약", "심장사상충", "구충제", "외구충", "내구충",
                 "넥스가드", "브라벡토", "레볼루션", "하트가드", "프론트라인",
                 "팔라디아", "팔리디아", "palladia", "항암", "빈크리스틴", "독소루비신"],
    "hospitalization": ["입원", "hospitalization", "icu", "중환자"],
    "emergency": ["응급", "emergency", "야간진료"],
}


def _fallback_classify_item(item_name: str) -> Optional[str]:
    """Keyword-based item classification fallback."""
    if not item_name:
        return None
    low = item_name.lower()
    for tag, keywords in _TAG_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                return tag
    return None


def _fallback_classify_record(items: List[Dict[str, Any]], ocr_text: str) -> List[str]:
    """Keyword-based record tag classification fallback."""
    tags: List[str] = []
    seen: set = set()

    # From items' categoryTags
    for it in (items or []):
        ct = (it.get("categoryTag") or "").strip()
        if ct and ct not in seen:
            seen.add(ct)
            tags.append(ct)

    # From item names
    for it in (items or []):
        nm = (it.get("itemName") or "").lower()
        for tag, keywords in _TAG_KEYWORDS.items():
            if tag in seen:
                continue
            for kw in keywords:
                if kw in nm:
                    seen.add(tag)
                    tags.append(tag)
                    break

    # From OCR text
    text_low = (ocr_text or "").lower()
    for tag, keywords in _TAG_KEYWORDS.items():
        if tag in seen:
            continue
        for kw in keywords:
            if kw in text_low:
                seen.add(tag)
                tags.append(tag)
                break

    return tags[:10]


# =========================================================
# Receipt processing
# =========================================================
@app.post("/api/receipts/process")
def process_receipt(
    petId: str = Form(...),
    file: UploadFile = File(...),
    original_file: Optional[UploadFile] = File(None),  # ✅ v2.8.0: iOS에서 보내는 원본 컬러 이미지
    visitDate: Optional[str] = Form(None),
    hospitalName: Optional[str] = Form(None),
    hospitalMgmtNo: Optional[str] = Form(None),
    existingRecordId: Optional[str] = Form(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
    _require_module(ocr_policy, "ocr_policy")

    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)

    pet_uuid = _uuid_or_400(petId, "petId")
    pet = db_fetchone("SELECT id, name, species, weight_kg FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")
    selected_pet_name = (pet.get("name") or "").strip()

    raw = _read_upload_limited(file, int(settings.MAX_RECEIPT_IMAGE_BYTES))
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")

    # ✅ v2.8.0: iOS가 보낸 원본 컬러 이미지 (보험청구용)
    raw_original: Optional[bytes] = None
    if original_file is not None:
        try:
            raw_original = _read_upload_limited(original_file, int(settings.MAX_RECEIPT_IMAGE_BYTES))
            if raw_original:
                logger.info("[receipt-draft] original_file received from iOS: %d bytes", len(raw_original))
            else:
                raw_original = None
        except Exception as _orig_err:
            logger.warning("[receipt-draft] original_file read failed (ignored): %s", _orig_err)
            raw_original = None

    # --- OCR pipeline ---
    _require_module(ocr_policy, "ocr_policy")

    try:
        result = ocr_policy.process_receipt_image(
            raw,
            timeout=int(settings.OCR_TIMEOUT_SECONDS),
            max_concurrency=int(settings.OCR_MAX_CONCURRENCY),
            sema_timeout=float(settings.OCR_SEMA_ACQUIRE_TIMEOUT_SECONDS),
            max_pixels=int(settings.IMAGE_MAX_PIXELS),
            receipt_max_width=int(settings.RECEIPT_MAX_WIDTH),
            receipt_webp_quality=int(settings.RECEIPT_WEBP_QUALITY),
            gemini_enabled=bool(settings.GEMINI_ENABLED),
            gemini_api_key=str(settings.GEMINI_API_KEY),
            gemini_model_name=str(settings.GEMINI_MODEL_NAME),
            gemini_timeout=int(settings.GEMINI_TIMEOUT_SECONDS),
        )
    except getattr(ocr_policy, "OCRTimeoutError", type(None)) as e:
        raise HTTPException(status_code=504, detail=f"OCR timeout: {e}")
    except getattr(ocr_policy, "OCRConcurrencyError", type(None)) as e:
        raise HTTPException(status_code=429, detail=f"OCR busy: {e}")
    except getattr(ocr_policy, "OCRImageError", type(None)) as e:
        raise HTTPException(status_code=400, detail=f"Image error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="OCR error"))

    ocr_text = result.get("ocr_text") or ""
    items_raw = result.get("items") or []
    meta = result.get("meta") or {}
    # 🔍 디버그: Gemini 파이프라인 상태 로깅
    import logging as _logging
    _dlog = _logging.getLogger("receipt_debug")
    _dlog.warning(f"[OCR-RESULT] pipeline={meta.get('pipeline')}, geminiUsed={meta.get('geminiUsed')}, geminiError={meta.get('geminiError')}, items={len(items_raw)}, ocrEngine={meta.get('ocrEngine')}")
    webp_bytes = result.get("webp_bytes") or raw          # 마스킹본
    original_webp = result.get("original_webp_bytes")      # 원본
    content_type = result.get("content_type") or "image/webp"

    ocr_hospital_name_raw = meta.get("hospital_name") or ""
    ocr_visit_date_raw = meta.get("visit_date") or ""
    ocr_visit_time_raw = meta.get("visit_time") or ""  # ✅ "HH:MM" or ""

    # Gemini may have already provided tags via ocr_policy
    gemini_tags: List[str] = meta.get("tags") or []

    # --- Apply tag_policy / item mapping ---
    record_tags: List[str] = []
    final_items: List[Dict[str, Any]] = []

    item_maps = db_fetchall(
        "SELECT ocr_item_name, canonical_name FROM public.ocr_item_name_maps WHERE is_active = true AND (is_custom_entry = false OR created_by_uid = %s) ORDER BY is_custom_entry DESC, created_at DESC",
        (uid,),
    )
    canon_map: Dict[str, str] = {}
    for m in item_maps:
        key = (m.get("ocr_item_name") or "").strip().lower()
        if key and key not in canon_map:
            canon_map[key] = m.get("canonical_name") or ""

    for it in items_raw:
        name_raw = (it.get("name") or "").strip()
        price = it.get("price")
        mapped = canon_map.get(name_raw.lower(), name_raw)

        # Gemini may have already provided standard tag code + Korean name
        cat_tag = (it.get("categoryTag") or "").strip() or None
        std_name = (it.get("standardName") or "").strip() or None

        # Fallback chain if Gemini didn't provide
        if not cat_tag:
            if tag_policy is not None:
                try:
                    cat_tag = tag_policy.classify_item(mapped, threshold=int(settings.TAG_ITEM_THRESHOLD))
                    _dlog.warning(f"[TAG] classify_item('{mapped}', thresh={settings.TAG_ITEM_THRESHOLD}) → {cat_tag}")
                except Exception as _tag_err:
                    _dlog.warning(f"[TAG] classify_item ERROR for '{mapped}': {_tag_err}")
            else:
                _dlog.warning(f"[TAG] tag_policy is None, skipping classify_item for '{mapped}'")
        else:
            _dlog.warning(f"[TAG] Gemini already set categoryTag='{cat_tag}' for '{mapped}'")
        if not cat_tag:
            cat_tag = _fallback_classify_item(mapped)
            _dlog.warning(f"[TAG] fallback_classify_item('{mapped}') → {cat_tag}")

        # 할인 태그 항목은 가격을 음수로 강제 (Gemini가 양수로 반환하는 경우 대비)
        _DISCOUNT_TAGS = {"etc_discount"}
        _DISCOUNT_KEYWORDS = {"할인", "절사", "감면", "환급", "조정", "discount", "쿠폰"}
        is_discount_item = (cat_tag in _DISCOUNT_TAGS) or any(kw in mapped for kw in _DISCOUNT_KEYWORDS)
        if is_discount_item and price is not None and price > 0:
            price = -price
            _dlog.warning(f"[DISCOUNT] forced negative price for '{mapped}': {price}")

        final_items.append({
            "itemName": mapped,
            "price": price,
            "categoryTag": cat_tag,
            "standardName": std_name,
        })

    # Tag priority: Gemini tags → tag_policy → fallback
    if gemini_tags:
        record_tags = gemini_tags
    elif tag_policy is not None:
        try:
            record_tags = tag_policy.classify_record(
                items=[{"name": fi["itemName"], "price": fi.get("price")} for fi in final_items],
                ocr_text=ocr_text,
                threshold=int(settings.TAG_RECORD_THRESHOLD),
            )
        except Exception:
            record_tags = []

    if not record_tags:
        record_tags = _fallback_classify_record(final_items, ocr_text)

    # --- parse visit date + time ---
    from datetime import datetime as _dt
    vd: Optional[_dt] = None
    if visitDate:
        try:
            vd = _dt.combine(date.fromisoformat(visitDate), _dt.min.time())
        except Exception:
            pass
    if vd is None and ocr_visit_date_raw:
        try:
            vd = _dt.combine(date.fromisoformat(ocr_visit_date_raw), _dt.min.time())
        except Exception:
            pass
    if vd is None:
        vd = _dt.combine(date.today(), _dt.min.time())
    # ✅ visitTime이 있으면 합치기
    if ocr_visit_time_raw:
        try:
            _parts = ocr_visit_time_raw.split(":")
            _h, _m = int(_parts[0]), int(_parts[1])
            if 0 <= _h <= 23 and 0 <= _m <= 59:
                vd = vd.replace(hour=_h, minute=_m)
        except Exception:
            pass

    # --- hospital matching ---
    hosp_name = hospitalName.strip() if isinstance(hospitalName, str) and hospitalName.strip() else None
    hosp_mgmt = hospitalMgmtNo.strip() if isinstance(hospitalMgmtNo, str) and hospitalMgmtNo.strip() else None

    if not hosp_name and ocr_hospital_name_raw:
        hosp_name = ocr_hospital_name_raw.strip()

    # --- draft (DB 저장 없이 OCR 결과만 반환) ---
    draft_id = str(uuid.uuid4())

    # 할인 항목(음수 가격)도 합산하여 실제 결제 금액 계산
    items_sum = sum(fi.get("price") or 0 for fi in final_items if fi.get("price") is not None)

    # Gemini가 반환한 totalAmount와 항목 합계 비교 검증
    gemini_total = meta.get("total_amount")
    if gemini_total and gemini_total > 0:
        total_amount = gemini_total  # Gemini totalAmount 우선 사용 (영수증의 합계/청구금액)
        diff = abs(gemini_total - items_sum)
        diff_pct = (diff / gemini_total * 100) if gemini_total else 0
        if diff_pct > 5:
            _dlog.warning(f"[VALIDATION] 항목 합계({items_sum})와 totalAmount({gemini_total}) 차이: {diff}원 ({diff_pct:.1f}%) — 항목 누락 가능성")
        else:
            _dlog.info(f"[VALIDATION] 항목 합계({items_sum}) ≈ totalAmount({gemini_total}) OK (차이 {diff_pct:.1f}%)")
    else:
        total_amount = items_sum if items_sum > 0 else 0

    pet_weight = None
    try:
        wkg = pet.get("weight_kg")
        pet_weight = float(wkg) if wkg is not None else None
    except Exception:
        pass

    # 드래프트 경로 (/draft_receipts/) — iOS isDraftReceiptPath() 판별용
    file_path = _receipt_draft_path(uid, str(pet_uuid), draft_id)
    file_size = int(len(webp_bytes))

    # ✅ v2.8.0: iOS가 원본 컬러를 별도로 보냈으면 그것을 _original로 저장
    #   → iOS original_file(컬러) > OCR 파이프라인 original_webp(전처리된 것일 수 있음)
    effective_original = None
    effective_original_ct = content_type
    if raw_original:
        # iOS가 보낸 원본 컬러 → WebP로 변환하여 저장
        try:
            from PIL import Image as _PILImage
            import io as _io
            _orig_img = _PILImage.open(_io.BytesIO(raw_original))
            _orig_img = _orig_img.convert("RGB")
            _orig_buf = _io.BytesIO()
            _rw = int(settings.RECEIPT_MAX_WIDTH) if hasattr(settings, "RECEIPT_MAX_WIDTH") else 2048
            if max(_orig_img.size) > _rw:
                ratio = _rw / max(_orig_img.size)
                new_size = (int(_orig_img.width * ratio), int(_orig_img.height * ratio))
                _orig_img = _orig_img.resize(new_size, _PILImage.LANCZOS)
            _rq = int(settings.RECEIPT_WEBP_QUALITY) if hasattr(settings, "RECEIPT_WEBP_QUALITY") else 85
            _orig_img.save(_orig_buf, format="WEBP", quality=_rq)
            effective_original = _orig_buf.getvalue()
            effective_original_ct = "image/webp"
            logger.info("[receipt-draft] iOS original_file → WebP: %d bytes (color preserved)", len(effective_original))
        except Exception as _conv_err:
            logger.warning("[receipt-draft] iOS original_file WebP conversion failed, using raw: %s", _conv_err)
            effective_original = raw_original
            effective_original_ct = "image/jpeg"
    elif original_webp:
        effective_original = original_webp
        effective_original_ct = content_type

    logger.info("[receipt-draft] effective_original present=%s, size=%s, webp_bytes size=%s, from_ios=%s",
                effective_original is not None,
                len(effective_original) if effective_original else 0,
                len(webp_bytes),
                raw_original is not None)

    try:
        # 마스킹본 (앱 표시용)
        upload_bytes_to_storage(file_path, webp_bytes, content_type)
        # 원본 컬러 (보험 청구 PDF용)
        if effective_original:
            orig_path = _receipt_draft_original_path(uid, str(pet_uuid), draft_id)
            logger.info("[receipt-draft] uploading original to %s (%d bytes, from_ios=%s)", orig_path, len(effective_original), raw_original is not None)
            try:
                upload_bytes_to_storage(orig_path, effective_original, effective_original_ct)
            except Exception as orig_err:
                logger.error("[receipt-draft] original upload FAILED: %s", orig_err)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))

    # --- hospital candidate 검색 (DB 저장 없이 조회만) ---
    resp_candidates = []
    auto_hosp_mgmt = hosp_mgmt
    try:
        candidate_limit = int(settings.OCR_HOSPITAL_CANDIDATE_LIMIT or 3)
        if hosp_name and not hosp_mgmt and candidate_limit > 0:
            like = f"%{hosp_name}%"
            clean_name = hosp_name.replace(" ", "")
            like_clean = f"%{clean_name}%"
            cands = db_fetchall(
                """SELECT hospital_mgmt_no, name, road_address, jibun_address, lat, lng, is_custom_entry,
                   similarity(name, %s) AS score
                FROM public.hospitals
                WHERE (is_custom_entry = false OR created_by_uid = %s)
                  AND (search_vector ILIKE %s
                       OR REPLACE(name, ' ', '') ILIKE %s
                       OR REPLACE(%s, ' ', '') ILIKE '%%' || REPLACE(name, ' ', '') || '%%')
                ORDER BY score DESC
                LIMIT %s""",
                (hosp_name, uid, like, like_clean, hosp_name, candidate_limit),
            )
            for rank, c in enumerate(cands or [], start=1):
                resp_candidates.append({
                    "rank": rank,
                    "score": float(c.get("score") or 0),
                    "hospitalMgmtNo": c["hospital_mgmt_no"],
                    "name": c["name"],
                    "roadAddress": c.get("road_address"),
                    "jibunAddress": c.get("jibun_address"),
                    "lat": c.get("lat"),
                    "lng": c.get("lng"),
                    "isCustomEntry": bool(c.get("is_custom_entry")),
                })
            if resp_candidates:
                auto_hosp_mgmt = resp_candidates[0]["hospitalMgmtNo"]
    except Exception as cand_err:
        logger.warning("[receipt-draft] hospital candidate search failed (ignored): %s", _sanitize_for_log(repr(cand_err)))

    # ── Response: iOS ReceiptDraftResponseDTO 포맷 (DB 저장 없음) ──
    resp_items = []
    for fi in final_items:
        resp_items.append({
            "itemName": fi.get("itemName") or "",
            "price": fi.get("price"),
            "categoryTag": fi.get("categoryTag"),
            "standardName": fi.get("standardName"),
        })

    draft_response = {
        "mode": "draft",
        "draftId": draft_id,
        "petId": str(pet_uuid),
        "petName": selected_pet_name or None,
        "petSpecies": (pet.get("species") or "").strip() or None,
        "draftReceiptPath": file_path,
        "fileSizeBytes": file_size,
        "visitDate": vd.isoformat() if hasattr(vd, "isoformat") else str(vd),
        "visitTime": ocr_visit_time_raw if ocr_visit_time_raw else None,
        "hospitalName": hosp_name,
        "hospitalMgmtNo": auto_hosp_mgmt,
        "totalAmount": int(total_amount),
        "petWeightAtVisit": pet_weight,
        "tags": record_tags,
        "items": resp_items,
        "hospitalCandidates": resp_candidates if resp_candidates else None,
        "hospitalCandidateCount": len(resp_candidates),
        "hospitalConfirmed": bool(auto_hosp_mgmt),
        "tagEvidence": None,
        "ocrText": ocr_text[:2000],
        "ocrMeta": meta,
    }

    return draft_response


# =========================================================
# Receipt: Commit draft → DB save
# =========================================================
class ReceiptCommitItemDTO(BaseModel):
    itemName: str
    price: Optional[int] = None
    categoryTag: Optional[str] = None
    standardName: Optional[str] = None

class ReceiptCommitRequestDTO(BaseModel):
    petId: str
    draftReceiptPath: str
    visitDate: str
    hospitalName: Optional[str] = None
    hospitalMgmtNo: Optional[str] = None
    totalAmount: Optional[int] = None
    petWeightAtVisit: Optional[float] = None
    recordId: Optional[str] = None  # ✅ 기존 레코드 업데이트 시 사용
    tags: List[str] = []
    items: List[ReceiptCommitItemDTO] = []
    replaceItems: bool = True


@app.post("/api/receipts/commit")
def commit_receipt(
    req: ReceiptCommitRequestDTO,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """드래프트를 확정하여 DB에 저장하고 이미지를 최종 경로로 이동"""
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)

    pet_uuid = _uuid_or_400(req.petId, "petId")
    pet = db_fetchone("SELECT id, name, species, weight_kg FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    draft_path = (req.draftReceiptPath or "").strip()
    if not draft_path:
        raise HTTPException(status_code=400, detail="draftReceiptPath is required")

    # visit date + time
    from datetime import datetime as _dt
    import logging as _logging
    _clog = _logging.getLogger("commit_receipt")
    vd = None
    _clog.warning(f"[COMMIT] req.visitDate raw = '{req.visitDate}'")
    if req.visitDate:
        try:
            if "T" in req.visitDate and len(req.visitDate) >= 16:
                vd = _dt.fromisoformat(req.visitDate)
                _clog.warning(f"[COMMIT] parsed datetime with T: {vd}")
            else:
                vd = _dt.combine(date.fromisoformat(req.visitDate[:10]), _dt.min.time())
                _clog.warning(f"[COMMIT] parsed date only: {vd}")
        except Exception as _e:
            _clog.warning(f"[COMMIT] parse FAILED: {_e}")
    if vd is None:
        vd = _dt.combine(date.today(), _dt.min.time())
        _clog.warning(f"[COMMIT] FALLBACK to today: {vd}")

    hosp_name = req.hospitalName.strip() if isinstance(req.hospitalName, str) and req.hospitalName.strip() else None
    hosp_mgmt = req.hospitalMgmtNo.strip() if isinstance(req.hospitalMgmtNo, str) and req.hospitalMgmtNo.strip() else None

    total_amount = req.totalAmount if req.totalAmount is not None else 0
    if total_amount <= 0:
        total_amount = sum(it.price or 0 for it in req.items if it.price is not None)

    pet_weight = req.petWeightAtVisit
    if pet_weight is None:
        try:
            wkg = pet.get("weight_kg")
            pet_weight = float(wkg) if wkg is not None else None
        except Exception:
            pet_weight = None

    tags = _clean_tags(req.tags)

    # ✅ 기존 recordId가 있으면 업데이트, 없으면 새 UUID 생성
    existing_record = False
    if req.recordId:
        try:
            record_uuid = uuid.UUID(req.recordId)
            # 기존 레코드가 실제로 존재하는지 확인
            _existing = db_fetchone(
                "SELECT r.id FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id "
                "WHERE r.id=%s AND p.user_uid=%s AND r.deleted_at IS NULL",
                (record_uuid, uid)
            )
            if _existing:
                existing_record = True
        except (ValueError, AttributeError):
            record_uuid = uuid.uuid4()
    else:
        record_uuid = uuid.uuid4()

    # 드래프트 이미지를 최종 경로로 복사(이동)
    final_path = _receipt_path(uid, str(pet_uuid), str(record_uuid))
    file_size = 0
    try:
        b = get_bucket()
        src_blob = b.blob(draft_path)
        if src_blob.exists():
            b.copy_blob(src_blob, b, final_path)
            src_blob.delete()
            dest_blob = b.blob(final_path)
            dest_blob.reload()
            file_size = dest_blob.size or 0
            logger.info("[commit] moved draft %s → %s (%d bytes)", draft_path, final_path, file_size)

            # 원본도 이동 (있으면)
            draft_orig = draft_path.replace(".webp", "_original.webp")
            final_orig = _receipt_original_path(uid, str(pet_uuid), str(record_uuid))
            src_orig = b.blob(draft_orig)
            if src_orig.exists():
                b.copy_blob(src_orig, b, final_orig)
                src_orig.delete()
                logger.info("[commit] moved draft original %s → %s", draft_orig, final_orig)
        else:
            logger.warning("[commit] draft blob not found: %s — proceeding without image", draft_path)
    except Exception as e:
        logger.error("[commit] storage move failed: %s", _sanitize_for_log(repr(e)))
        # 이미지 이동 실패해도 레코드는 저장 진행

    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if existing_record:
                    # ✅ 기존 레코드 업데이트 (편집 중 이미지 추가)
                    cur.execute(
                        """
                        UPDATE public.health_records SET
                            hospital_mgmt_no=%s, hospital_name=%s, visit_date=%s,
                            total_amount=%s, pet_weight_at_visit=%s, tags=%s,
                            receipt_image_path=%s, file_size_bytes=%s
                        WHERE id=%s AND deleted_at IS NULL
                        RETURNING id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags,
                            receipt_image_path, file_size_bytes, created_at, updated_at
                        """,
                        (hosp_mgmt, hosp_name, vd, total_amount, pet_weight, tags, final_path, file_size, record_uuid),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO public.health_records
                            (id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags, receipt_image_path, file_size_bytes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags,
                            receipt_image_path, file_size_bytes, created_at, updated_at
                        """,
                        (record_uuid, pet_uuid, hosp_mgmt, hosp_name, vd, total_amount, pet_weight, tags, final_path, file_size),
                    )
                rec_row = cur.fetchone()
                if not rec_row:
                    raise HTTPException(status_code=500, detail=_internal_detail("commit record insert failed", kind="DB error"))

                # ✅ 기존 레코드면 기존 items 삭제 후 재삽입
                if existing_record and req.replaceItems:
                    cur.execute("DELETE FROM public.health_items WHERE record_id=%s", (record_uuid,))

                # items 저장
                for it in req.items:
                    iname = (it.itemName or "").strip()
                    if not iname:
                        continue
                    iprice = int(it.price) if it.price is not None else None
                    cat_tag = it.categoryTag.strip() if isinstance(it.categoryTag, str) and it.categoryTag.strip() else None
                    cur.execute("INSERT INTO public.health_items (record_id, item_name, price, category_tag) VALUES (%s, %s, %s, %s)", (record_uuid, iname, iprice, cat_tag))

                cur.execute("SELECT id, record_id, item_name, price, category_tag, created_at, updated_at FROM public.health_items WHERE record_id=%s ORDER BY created_at ASC", (record_uuid,))
                items_rows = cur.fetchall() or []

                # hospital candidates
                candidate_limit = int(settings.OCR_HOSPITAL_CANDIDATE_LIMIT or 3)
                if hosp_name and not hosp_mgmt and candidate_limit > 0:
                    like = f"%{hosp_name}%"
                    clean_name = hosp_name.replace(" ", "")
                    like_clean = f"%{clean_name}%"
                    cur.execute(
                        """SELECT hospital_mgmt_no, name, road_address, jibun_address, lat, lng, is_custom_entry,
                           similarity(name, %s) AS score
                        FROM public.hospitals
                        WHERE (is_custom_entry = false OR created_by_uid = %s)
                          AND (search_vector ILIKE %s
                               OR REPLACE(name, ' ', '') ILIKE %s
                               OR REPLACE(%s, ' ', '') ILIKE '%%' || REPLACE(name, ' ', '') || '%%')
                        ORDER BY score DESC
                        LIMIT %s""",
                        (hosp_name, uid, like, like_clean, hosp_name, candidate_limit),
                    )
                    cands = cur.fetchall() or []
                    if cands:
                        for rank, c in enumerate(cands, start=1):
                            try:
                                cur.execute(
                                    "INSERT INTO public.health_record_hospital_candidates (record_id, hospital_mgmt_no, rank, score) VALUES (%s, %s, %s, %s) ON CONFLICT (record_id, hospital_mgmt_no) DO UPDATE SET rank = EXCLUDED.rank, score = EXCLUDED.score",
                                    (record_uuid, c["hospital_mgmt_no"], rank, float(c.get("score") or 0)),
                                )
                            except Exception as ce:
                                logger.info("[DB] candidate insert failed (ignored): %s", _sanitize_for_log(_pg_message(ce)))
                        # Auto-select rank 1 candidate
                        top_mgmt_no = cands[0]["hospital_mgmt_no"]
                        try:
                            cur.execute(
                                "UPDATE public.health_records SET hospital_mgmt_no = %s WHERE id = %s AND deleted_at IS NULL",
                                (top_mgmt_no, record_uuid),
                            )
                            rec_row = dict(rec_row)
                            rec_row["hospital_mgmt_no"] = top_mgmt_no
                        except Exception as am_err:
                            logger.warning("[commit] Auto-match failed (ignored): %s", _sanitize_for_log(_pg_message(am_err)))

                # HealthRecordRowDTO 포맷 응답
                payload = dict(rec_row)
                payload["items"] = [dict(x) for x in items_rows]
                # pet 정보 추가
                payload["pet_name"] = pet.get("name")
                payload["pet_species"] = pet.get("species")
                return jsonable_encoder(payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "[COMMIT-ERROR] record_id=%s, pet_id=%s, draft_path=%s, "
            "existing=%s, error=%s: %s",
            str(record_uuid), str(pet_uuid), (draft_path or "")[:80],
            existing_record, type(e).__name__, _sanitize_for_log(str(e)),
        )
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Receipt: Delete draft (storage only)
# =========================================================
@app.delete("/api/receipts/draft/delete")
def delete_receipt_draft(
    draftReceiptPath: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """드래프트 이미지를 스토리지에서 삭제 (DB 레코드 없음)"""
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    path = (draftReceiptPath or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="draftReceiptPath is required")

    # 보안: 본인 경로인지 확인
    if not path.startswith(f"users/{uid}/"):
        raise HTTPException(status_code=403, detail="not your draft")

    deleted = False
    try:
        deleted = delete_storage_object_if_exists(path)
        # 원본도 삭제
        orig_path = path.replace(".webp", "_original.webp")
        delete_storage_object_if_exists(orig_path)
        logger.info("[draft-delete] deleted=%s path=%s", deleted, path)
    except Exception as e:
        logger.warning("[draft-delete] error: %s", _sanitize_for_log(repr(e)))

    return {"ok": True, "deleted": deleted, "draftReceiptPath": path}


# =========================================================
# Documents (PDF / Image) — v2.3.0: diagnosis support + image upload
# =========================================================
def _is_pdf_bytes(data: bytes) -> bool:
    return bool(data) and data[:5] == b"%PDF-"


def _is_image_bytes(data: bytes) -> Optional[str]:
    """Return mime type if recognized image, else None."""
    if not data:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_ext(mime: str) -> str:
    return {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(mime, "jpg")


@app.post("/api/docs/upload", response_model=DocumentUploadResponse)
def upload_document(
    petId: str = Form(...),
    docType: str = Form(...),
    displayName: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Upload PDF or image (jpg/png/webp) document."""
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)
    pet_uuid = _uuid_or_400(petId, "petId")
    dt = (docType or "").strip().lower()
    if dt not in ("lab", "cert", "diagnosis"):
        raise HTTPException(status_code=400, detail="docType must be 'lab', 'cert', or 'diagnosis'")
    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")
    data = _read_upload_limited(file, int(settings.MAX_PDF_BYTES))
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    # Detect file type: PDF or image
    doc_uuid = uuid.uuid4()
    if _is_pdf_bytes(data):
        content_type = "application/pdf"
        file_path = _doc_file_path(uid, str(pet_uuid), dt, str(doc_uuid), "pdf")
    else:
        img_mime = _is_image_bytes(data)
        if not img_mime:
            raise HTTPException(status_code=400, detail="file must be PDF or image (jpg/png/webp)")
        content_type = img_mime
        ext = _image_ext(img_mime)
        file_path = _doc_file_path(uid, str(pet_uuid), dt, str(doc_uuid), ext)

    name = (displayName or "").strip()
    if not name:
        name = (file.filename or "").strip() or f"{dt}.{'pdf' if content_type == 'application/pdf' else _image_ext(content_type)}"
    file_size_bytes = int(len(data))
    try:
        upload_bytes_to_storage(file_path, data, content_type)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))
    try:
        row = db_fetchone(
            "INSERT INTO public.pet_documents (id, pet_id, doc_type, display_name, file_path, file_size_bytes) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id, pet_id, doc_type, display_name, file_path, file_size_bytes, created_at, updated_at",
            (doc_uuid, pet_uuid, dt, name, file_path, file_size_bytes),
        )
        if not row:
            raise HTTPException(status_code=500, detail=_internal_detail("failed to insert pet_document", kind="DB error"))
        return {
            "id": str(row["id"]), "petId": str(row["pet_id"]), "docType": row["doc_type"],
            "displayName": row["display_name"], "filePath": row["file_path"],
            "fileSizeBytes": int(row["file_size_bytes"]),
            "createdAt": row["created_at"].isoformat() if row.get("created_at") else datetime.utcnow().isoformat() + "Z",
            "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else datetime.utcnow().isoformat() + "Z",
        }
    except HTTPException:
        try:
            delete_storage_object_if_exists(file_path)
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            delete_storage_object_if_exists(file_path)
        except Exception:
            pass
        _raise_mapped_db_error(e)
        raise


# Legacy alias: keep old endpoint working
@app.post("/api/docs/upload-pdf", response_model=DocumentUploadResponse)
def upload_pdf_document(
    petId: str = Form(...),
    docType: str = Form(...),
    displayName: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Legacy endpoint — delegates to new /api/docs/upload."""
    return upload_document(petId=petId, docType=docType, displayName=displayName, file=file, user=user)


@app.get("/api/docs/list")
def list_pdf_documents(petId: str = Query(...), docType: Optional[str] = Query(None), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    try:
        pet_uuid = _uuid_or_400(petId, "petId")
        dt = (docType or "").strip().lower() if docType else None
        if dt is not None:
            alias = {"vaccine": "cert", "vacc": "cert", "vaccination": "cert", "diag": "diagnosis"}
            dt = alias.get(dt, dt)
        if dt is not None and dt not in ("lab", "cert", "diagnosis"):
            raise HTTPException(status_code=400, detail="docType must be 'lab', 'cert', or 'diagnosis'")
        if dt:
            rows = db_fetchall("SELECT d.* FROM public.pet_documents d JOIN public.pets p ON p.id = d.pet_id WHERE p.user_uid=%s AND p.id=%s AND d.doc_type=%s AND d.deleted_at IS NULL ORDER BY d.created_at DESC", (uid, pet_uuid, dt))
        else:
            rows = db_fetchall("SELECT d.* FROM public.pet_documents d JOIN public.pets p ON p.id = d.pet_id WHERE p.user_uid=%s AND p.id=%s AND d.deleted_at IS NULL ORDER BY d.created_at DESC", (uid, pet_uuid))
        out = []
        for r in rows:
            out.append({
                "id": str(r["id"]), "petId": str(r["pet_id"]), "docType": r["doc_type"],
                "displayName": r["display_name"], "filePath": r["file_path"],
                "fileSizeBytes": int(r["file_size_bytes"]),
                "clinicName": r.get("clinic_name"),
                "memo": r.get("memo"),
                "visitDate": r["visit_date"].isoformat() if r.get("visit_date") else None,
                "createdAt": r["created_at"].isoformat() if r.get("created_at") else None,
                "updatedAt": r["updated_at"].isoformat() if r.get("updated_at") else None,
            })
        return out
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.put("/api/docs/update-meta")
def update_document_meta(req: DocumentUpdateMetaRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    doc_uuid = _uuid_or_400(req.doc_id, "docId")
    sets: List[str] = []
    params: List[Any] = []
    if req.clinic_name is not None:
        sets.append("clinic_name = %s")
        params.append(req.clinic_name.strip() if req.clinic_name.strip() else None)
    if req.memo is not None:
        sets.append("memo = %s")
        params.append(req.memo.strip() if req.memo.strip() else None)
    if req.visit_date is not None:
        sets.append("visit_date = %s")
        params.append(req.visit_date)
    if req.display_name is not None:
        dn = req.display_name.strip()
        if not dn:
            raise HTTPException(status_code=400, detail="displayName cannot be empty")
        sets.append("display_name = %s")
        params.append(dn)
    if not sets:
        raise HTTPException(status_code=400, detail="no fields to update")
    params.extend([doc_uuid, uid])
    try:
        row = db_fetchone(
            f"UPDATE public.pet_documents d SET {', '.join(sets)} WHERE d.id = %s AND d.deleted_at IS NULL AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = d.pet_id AND p.user_uid = %s) RETURNING id, pet_id, doc_type, display_name, file_path, file_size_bytes, clinic_name, memo, visit_date, created_at, updated_at",
            tuple(params),
        )
        if not row:
            raise HTTPException(status_code=404, detail="document not found")
        return {
            "id": str(row["id"]), "petId": str(row["pet_id"]), "docType": row["doc_type"],
            "displayName": row["display_name"], "filePath": row["file_path"],
            "fileSizeBytes": int(row["file_size_bytes"]),
            "clinicName": row.get("clinic_name"), "memo": row.get("memo"),
            "visitDate": row["visit_date"].isoformat() if row.get("visit_date") else None,
            "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
            "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.delete("/api/docs/delete")
def delete_pdf_document(docId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    doc_uuid = _uuid_or_400(docId, "docId")
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT d.id, d.file_path FROM public.pet_documents d JOIN public.pets p ON p.id = d.pet_id WHERE p.user_uid=%s AND d.id=%s AND d.deleted_at IS NULL", (uid, doc_uuid))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="document not found (or already deleted)")
                old_path = row.get("file_path")
                cur.execute("UPDATE public.pet_documents d SET deleted_at = now() WHERE d.id=%s AND d.deleted_at IS NULL AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = d.pet_id AND p.user_uid = %s)", (doc_uuid, uid))
                if (cur.rowcount or 0) <= 0:
                    raise HTTPException(status_code=404, detail="document not found (or already deleted)")
        return {"ok": True, "docId": str(doc_uuid), "softDeleted": True, "oldFilePath": old_path, "storageDeletion": "queued_by_trigger"}
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# compat wrappers
@app.post("/api/lab/upload-pdf", response_model=DocumentUploadResponse)
def upload_lab_pdf_compat(petId: str = Form(...), displayName: Optional[str] = Form(None), file: UploadFile = File(...), user: Dict[str, Any] = Depends(get_current_user)):
    return upload_pdf_document(petId=petId, docType="lab", displayName=displayName, file=file, user=user)


@app.post("/api/cert/upload-pdf", response_model=DocumentUploadResponse)
def upload_cert_pdf_compat(petId: str = Form(...), displayName: Optional[str] = Form(None), file: UploadFile = File(...), user: Dict[str, Any] = Depends(get_current_user)):
    return upload_pdf_document(petId=petId, docType="cert", displayName=displayName, file=file, user=user)


@app.post("/api/diagnosis/upload-pdf", response_model=DocumentUploadResponse)
def upload_diagnosis_pdf_compat(petId: str = Form(...), displayName: Optional[str] = Form(None), file: UploadFile = File(...), user: Dict[str, Any] = Depends(get_current_user)):
    return upload_pdf_document(petId=petId, docType="diagnosis", displayName=displayName, file=file, user=user)


# =========================================================
# Signed URL endpoint
# =========================================================
def _validate_storage_path(path: str, uid: str) -> str:
    p = (path or "").strip().lstrip("/")
    if not p:
        raise HTTPException(status_code=400, detail="path is required")
    if ".." in p:
        raise HTTPException(status_code=400, detail="invalid path")
    if not p.startswith(f"users/{uid}/"):
        raise HTTPException(status_code=403, detail="path is not under your user prefix")
    return p


def _assert_path_owned(uid: str, path: str) -> Dict[str, Any]:
    if path.startswith(f"users/{uid}/backups/") and path.endswith(".json"):
        return {"kind": "backup"}
    # ✅ draft_receipts는 아직 DB에 저장 전이므로 경로 소유권만 확인
    if "/draft_receipts/" in path and path.startswith(f"users/{uid}/"):
        return {"kind": "draft_receipt"}
    # Check exact match first
    rec = db_fetchone("SELECT 1 FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND r.deleted_at IS NULL AND r.receipt_image_path=%s LIMIT 1", (uid, path))
    if rec:
        return {"kind": "receipt"}
    # ✅ _original.webp → 마스킹본 경로로 변환하여 소유권 확인
    if "_original.webp" in path:
        base_path = path.replace("_original.webp", ".webp")
        rec2 = db_fetchone("SELECT 1 FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND r.deleted_at IS NULL AND r.receipt_image_path=%s LIMIT 1", (uid, base_path))
        if rec2:
            return {"kind": "receipt_original"}
    doc = db_fetchone("SELECT 1 FROM public.pet_documents d JOIN public.pets p ON p.id = d.pet_id WHERE p.user_uid=%s AND d.deleted_at IS NULL AND d.file_path=%s LIMIT 1", (uid, path))
    if doc:
        return {"kind": "document"}
    raise HTTPException(status_code=404, detail="file not found or not accessible")


def _generate_signed_url(path: str, ttl_seconds: int, filename: Optional[str] = None) -> Tuple[str, str]:
    if settings.STUB_MODE:
        expires_at = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat() + "Z"
        return f"/_stub/{path}", expires_at
    b = get_bucket()
    blob = b.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="object not found in storage")
    ttl_seconds = int(ttl_seconds)
    expires_at_dt = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    expires_at = expires_at_dt.isoformat() + "Z"
    response_disposition = None
    if filename:
        safe = re.sub(r'[\/\\\:\?\%\*\|"<>\n\r\t]', " ", filename).strip()
        if safe:
            response_disposition = f'inline; filename="{safe}"'
    try:
        url = blob.generate_signed_url(version="v4", expiration=timedelta(seconds=ttl_seconds), method="GET", response_disposition=response_disposition)
        return url, expires_at
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Signed URL error"))


@app.get("/api/storage/signed-url", response_model=SignedUrlResponse)
def storage_signed_url(path: str = Query(...), ttl: int = Query(None, ge=60), filename: Optional[str] = Query(None), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)
    ttl_final = int(ttl or settings.SIGNED_URL_DEFAULT_TTL_SECONDS)
    ttl_final = max(60, min(ttl_final, int(settings.SIGNED_URL_MAX_TTL_SECONDS)))
    p = _validate_storage_path(path, uid)
    _ = _assert_path_owned(uid, p)
    url, expires_at = _generate_signed_url(p, ttl_final, filename=filename)
    return {"path": p, "url": url, "expiresAt": expires_at}


# =========================================================
# Backup endpoints
# =========================================================
@app.post("/api/backup/upload", response_model=BackupUploadResponse)
def backup_upload(req: BackupUploadRequest = Body(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "unknown").strip()
    desired = _infer_membership_tier_from_token(user)
    if uid and uid != "unknown":
        try:
            db_touch_user(uid, desired_tier=desired)
        except Exception:
            pass
    payload_bytes = json.dumps(req.snapshot, ensure_ascii=False).encode("utf-8")
    if len(payload_bytes) > int(settings.MAX_BACKUP_BYTES):
        raise HTTPException(status_code=413, detail="backup too large")
    backup_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat() + "Z"
    doc = {"meta": {"uid": uid, "backupId": backup_id, "createdAt": created_at, "clientTime": req.clientTime, "appVersion": req.appVersion, "device": req.device, "note": req.note}, "snapshot": req.snapshot}
    path = f"{_backup_prefix(uid)}/{backup_id}.json"
    try:
        upload_bytes_to_storage(path, json.dumps(doc, ensure_ascii=False).encode("utf-8"), "application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))
    return {"ok": True, "uid": uid, "backupId": backup_id, "objectPath": path, "createdAt": created_at}


@app.get("/api/backup/list")
def backup_list(user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "unknown").strip()
    prefix = _backup_prefix(uid) + "/"
    b = get_bucket()
    out = []
    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".json"):
            continue
        bid = os.path.splitext(os.path.basename(blob.name))[0]
        out.append({"backupId": bid, "objectPath": blob.name, "lastModified": blob.updated.isoformat() if getattr(blob, "updated", None) else None, "size": int(getattr(blob, "size", 0) or 0)})
    out.sort(key=lambda x: x.get("lastModified") or "", reverse=True)
    return out


@app.get("/api/backup/get")
def backup_get(backupId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "unknown").strip()
    bid = (backupId or "").strip()
    if not bid:
        raise HTTPException(status_code=400, detail="backupId is required")
    path = f"{_backup_prefix(uid)}/{bid}.json"
    b = get_bucket()
    blob = b.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="backup not found")
    raw = blob.download_as_bytes()
    return json.loads(raw.decode("utf-8"))


@app.get("/api/backup/latest")
def backup_latest(user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "unknown").strip()
    prefix = _backup_prefix(uid) + "/"
    b = get_bucket()
    jsons = []
    for blob in b.list_blobs(prefix=prefix):
        if blob.name.endswith(".json"):
            jsons.append(blob)
    if not jsons:
        raise HTTPException(status_code=404, detail="no backups")
    latest = max(jsons, key=lambda o: getattr(o, "updated", None) or datetime.min)
    raw = latest.download_as_bytes()
    return json.loads(raw.decode("utf-8"))


# =========================================================
# Migration tokens
# =========================================================
def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

def _generate_migration_code() -> str:
    return secrets.token_urlsafe(32)


@app.post("/api/migration/prepare", response_model=MigrationPrepareResponse)
def migration_prepare(user: Dict[str, Any] = Depends(get_current_user)):
    old_uid = (user.get("uid") or "").strip()
    if not old_uid:
        raise HTTPException(status_code=401, detail="missing uid")
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(old_uid, desired_tier=desired)
    code = _generate_migration_code()
    code_hash = _hash_code(code)
    expires_at = datetime.utcnow() + timedelta(seconds=int(settings.MIGRATION_TOKEN_TTL_SECONDS))
    try:
        db_execute("INSERT INTO public.migration_tokens (old_uid, code_hash, expires_at, status) VALUES (%s, %s, %s, 'issued')", (old_uid, code_hash, expires_at))
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise
    return {"oldUid": old_uid, "migrationCode": code, "expiresAt": expires_at.isoformat() + "Z"}


def _copy_prefix(old_uid: str, new_uid: str) -> int:
    b = get_bucket()
    src_prefix = f"users/{old_uid}/"
    dst_prefix = f"users/{new_uid}/"
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
    b = get_bucket()
    src_prefix = f"users/{old_uid}/"
    deleted = 0
    blobs = list(b.list_blobs(prefix=src_prefix))
    for blob in blobs:
        blob.delete()
        deleted += 1
    return deleted


@app.post("/api/migration/execute")
def migration_execute(req: MigrationExecuteRequest, user: Dict[str, Any] = Depends(get_current_user)):
    new_uid = (user.get("uid") or "").strip()
    if not new_uid:
        raise HTTPException(status_code=401, detail="missing uid")
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(new_uid, desired_tier=desired)
    code = (req.migrationCode or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="migrationCode is required")
    code_hash = _hash_code(code)
    now = datetime.utcnow()
    token_row = db_fetchone("SELECT * FROM public.migration_tokens WHERE code_hash=%s", (code_hash,))
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
        raise HTTPException(status_code=500, detail=_internal_detail("invalid token row (old_uid)", kind="DB error"))
    if old_uid == new_uid:
        db_execute("UPDATE public.migration_tokens SET status='completed', used_at=now(), new_uid=%s WHERE code_hash=%s", (new_uid, code_hash))
        return {"ok": True, "oldUid": old_uid, "newUid": new_uid, "copied": 0, "deleted": 0, "dbUpdated": False, "warnings": ["oldUid == newUid (no-op)"]}
    db_execute("UPDATE public.migration_tokens SET status='processing', new_uid=%s WHERE code_hash=%s", (new_uid, code_hash))
    try:
        copied = _copy_prefix(old_uid, new_uid)
    except Exception as e:
        db_execute("UPDATE public.migration_tokens SET status='failed' WHERE code_hash=%s", (code_hash,))
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage copy failed"))
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM public.migrate_user_data(%s, %s)", (old_uid, new_uid))
                steps = cur.fetchall() or []
    except HTTPException:
        db_execute("UPDATE public.migration_tokens SET status='failed' WHERE code_hash=%s", (code_hash,))
        raise
    except Exception as e:
        db_execute("UPDATE public.migration_tokens SET status='failed' WHERE code_hash=%s", (code_hash,))
        _raise_mapped_db_error(e)
        raise
    deleted = 0
    warnings: List[str] = []
    try:
        deleted = _delete_prefix(old_uid)
    except Exception as e:
        warnings.append(_internal_detail(str(e), kind="Cleanup error"))
    db_execute("UPDATE public.migration_tokens SET status='completed', used_at=now() WHERE code_hash=%s", (code_hash,))
    return {"ok": True, "oldUid": old_uid, "newUid": new_uid, "copied": copied, "deleted": deleted, "dbUpdated": True, "steps": [dict(s) for s in steps], "warnings": warnings}


# =========================================================
# OCR ItemName Map APIs
# =========================================================
@app.get("/api/ocr/item-maps/list")
def list_item_maps(includeInactive: bool = Query(False), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    rows = db_fetchall(
        "SELECT id, ocr_item_name, canonical_name, is_active, is_custom_entry, created_by_uid, created_at, updated_at FROM public.ocr_item_name_maps WHERE (is_custom_entry = false OR created_by_uid = %s) AND (%s OR is_active = true) ORDER BY is_custom_entry DESC, created_at DESC",
        (uid, includeInactive),
    )
    out = []
    for r in rows:
        out.append({"id": str(r["id"]), "ocrItemName": r["ocr_item_name"], "canonicalName": r["canonical_name"], "isActive": bool(r["is_active"]), "isCustomEntry": bool(r["is_custom_entry"]), "createdAt": r["created_at"].isoformat() if r.get("created_at") else None, "updatedAt": r["updated_at"].isoformat() if r.get("updated_at") else None})
    return out


@app.post("/api/ocr/item-maps/custom/upsert")
def upsert_custom_item_map(req: OCRItemMapUpsertRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    ocr_name = (req.ocrItemName or "").strip()
    canonical = (req.canonicalName or "").strip()
    is_active = bool(req.isActive) if req.isActive is not None else True
    if not ocr_name:
        raise HTTPException(status_code=400, detail="ocrItemName is required")
    if not canonical:
        raise HTTPException(status_code=400, detail="canonicalName is required")
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("UPDATE public.ocr_item_name_maps SET canonical_name = %s, is_active = %s WHERE is_custom_entry = true AND created_by_uid = %s AND lower(ocr_item_name) = lower(%s) AND is_active = true RETURNING id, ocr_item_name, canonical_name, is_active, is_custom_entry, created_at, updated_at", (canonical, is_active, uid, ocr_name))
                row = cur.fetchone()
                if row:
                    return {"id": str(row["id"]), "ocrItemName": row["ocr_item_name"], "canonicalName": row["canonical_name"], "isActive": bool(row["is_active"]), "isCustomEntry": True, "createdAt": row["created_at"].isoformat() if row.get("created_at") else None, "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None}
                cur.execute("SELECT id FROM public.ocr_item_name_maps WHERE is_custom_entry = true AND created_by_uid = %s AND lower(ocr_item_name) = lower(%s) AND is_active = false ORDER BY created_at DESC LIMIT 1", (uid, ocr_name))
                inactive = cur.fetchone()
                if inactive:
                    cur.execute("UPDATE public.ocr_item_name_maps SET canonical_name = %s, is_active = %s WHERE id = %s RETURNING id, ocr_item_name, canonical_name, is_active, is_custom_entry, created_at, updated_at", (canonical, is_active, inactive["id"]))
                    row = cur.fetchone()
                    if row:
                        return {"id": str(row["id"]), "ocrItemName": row["ocr_item_name"], "canonicalName": row["canonical_name"], "isActive": bool(row["is_active"]), "isCustomEntry": True, "createdAt": row["created_at"].isoformat() if row.get("created_at") else None, "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None}
                cur.execute("INSERT INTO public.ocr_item_name_maps (ocr_item_name, canonical_name, is_active, is_custom_entry, created_by_uid) VALUES (%s, %s, %s, true, %s) RETURNING id, ocr_item_name, canonical_name, is_active, is_custom_entry, created_at, updated_at", (ocr_name, canonical, is_active, uid))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=500, detail=_internal_detail("failed to upsert item map", kind="DB error"))
                return {"id": str(row["id"]), "ocrItemName": row["ocr_item_name"], "canonicalName": row["canonical_name"], "isActive": bool(row["is_active"]), "isCustomEntry": True, "createdAt": row["created_at"].isoformat() if row.get("created_at") else None, "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None}
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.post("/api/ocr/item-maps/custom/deactivate")
def deactivate_custom_item_map(req: OCRItemMapDeactivateRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    ocr_name = (req.ocrItemName or "").strip()
    if not ocr_name:
        raise HTTPException(status_code=400, detail="ocrItemName is required")
    try:
        row = db_fetchone("UPDATE public.ocr_item_name_maps SET is_active = false WHERE is_custom_entry = true AND created_by_uid = %s AND lower(ocr_item_name) = lower(%s) AND is_active = true RETURNING id, ocr_item_name, canonical_name, is_active, created_at, updated_at", (uid, ocr_name))
        if not row:
            raise HTTPException(status_code=404, detail="active mapping not found")
        return {"ok": True, "id": str(row["id"]), "ocrItemName": row["ocr_item_name"], "canonicalName": row["canonical_name"], "isActive": bool(row["is_active"]), "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None}
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.post("/api/admin/ocr/item-maps/global/upsert")
def admin_upsert_global_item_map(req: OCRItemMapUpsertRequest, admin: Dict[str, Any] = Depends(get_admin_user)):
    ocr_name = (req.ocrItemName or "").strip()
    canonical = (req.canonicalName or "").strip()
    is_active = bool(req.isActive) if req.isActive is not None else True
    if not ocr_name:
        raise HTTPException(status_code=400, detail="ocrItemName is required")
    if not canonical:
        raise HTTPException(status_code=400, detail="canonicalName is required")
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("UPDATE public.ocr_item_name_maps SET canonical_name=%s, is_active=%s WHERE is_custom_entry=false AND is_active=true AND lower(ocr_item_name)=lower(%s) RETURNING id, ocr_item_name, canonical_name, is_active, created_at, updated_at", (canonical, is_active, ocr_name))
                row = cur.fetchone()
                if row:
                    return {"id": str(row["id"]), "ocrItemName": row["ocr_item_name"], "canonicalName": row["canonical_name"], "isActive": bool(row["is_active"]), "isCustomEntry": False, "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None}
                cur.execute("SELECT id FROM public.ocr_item_name_maps WHERE is_custom_entry=false AND is_active=false AND lower(ocr_item_name)=lower(%s) ORDER BY created_at DESC LIMIT 1", (ocr_name,))
                inactive = cur.fetchone()
                if inactive:
                    cur.execute("UPDATE public.ocr_item_name_maps SET canonical_name=%s, is_active=%s WHERE id=%s RETURNING id, ocr_item_name, canonical_name, is_active, created_at, updated_at", (canonical, is_active, inactive["id"]))
                    row = cur.fetchone()
                    if row:
                        return {"id": str(row["id"]), "ocrItemName": row["ocr_item_name"], "canonicalName": row["canonical_name"], "isActive": bool(row["is_active"]), "isCustomEntry": False, "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None}
                cur.execute("INSERT INTO public.ocr_item_name_maps (ocr_item_name, canonical_name, is_active, is_custom_entry, created_by_uid) VALUES (%s, %s, %s, false, NULL) RETURNING id, ocr_item_name, canonical_name, is_active, created_at, updated_at", (ocr_name, canonical, is_active))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=500, detail=_internal_detail("failed to upsert global item map", kind="DB error"))
                return {"id": str(row["id"]), "ocrItemName": row["ocr_item_name"], "canonicalName": row["canonical_name"], "isActive": bool(row["is_active"]), "isCustomEntry": False, "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None}
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Insurance Claims (v2.3.0)
# =========================================================
@app.post("/api/claims/upsert")
def api_claim_upsert(req: InsuranceClaimUpsertRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    pet_uuid = _uuid_or_400(req.pet_id, "petId")
    claim_uuid = _uuid_or_new(req.id, "id")
    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")
    claim_date = req.claim_date or date.today()
    total_amount = int(req.total_amount) if req.total_amount is not None else 0
    if total_amount < 0:
        raise HTTPException(status_code=400, detail="totalAmount must be >= 0")
    attached = json.dumps(req.attached_documents or [], ensure_ascii=False)
    try:
        row = db_fetchone(
            """
            INSERT INTO public.insurance_claims
                (id, pet_id, user_uid, claim_title, insurance_company, claim_date,
                 date_range_start, date_range_end, total_amount,
                 attached_documents, merged_pdf_path, merged_pdf_bytes, memo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                claim_title = EXCLUDED.claim_title,
                insurance_company = EXCLUDED.insurance_company,
                claim_date = EXCLUDED.claim_date,
                date_range_start = EXCLUDED.date_range_start,
                date_range_end = EXCLUDED.date_range_end,
                total_amount = EXCLUDED.total_amount,
                attached_documents = EXCLUDED.attached_documents,
                merged_pdf_path = EXCLUDED.merged_pdf_path,
                merged_pdf_bytes = EXCLUDED.merged_pdf_bytes,
                memo = EXCLUDED.memo
            WHERE public.insurance_claims.user_uid = %s AND public.insurance_claims.deleted_at IS NULL
            RETURNING *
            """,
            (claim_uuid, pet_uuid, uid, req.claim_title, req.insurance_company, claim_date,
             req.date_range_start, req.date_range_end, total_amount,
             attached, req.merged_pdf_path, req.merged_pdf_bytes or 0, req.memo, uid),
        )
        if not row:
            raise HTTPException(status_code=409, detail="claim not found or not editable")

        # ✅ claim_count 갱신: insurance_claims 테이블에서 실제 건수를 세서 반영
        try:
            cnt = db_fetchone(
                "SELECT COUNT(*) AS c FROM public.insurance_claims WHERE user_uid=%s AND deleted_at IS NULL",
                (uid,),
            )
            if cnt:
                db_execute(
                    "UPDATE public.users SET claim_count = %s WHERE firebase_uid = %s",
                    (int(cnt["c"]), uid),
                )
        except Exception as _cnt_err:
            logger.warning("[claim-upsert] claim_count update failed (ignored): %s", _cnt_err)

        return jsonable_encoder(row)
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.get("/api/claims/list")
def api_claims_list(petId: Optional[str] = Query(None), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    if petId:
        pet_uuid = _uuid_or_400(petId, "petId")
        rows = db_fetchall("SELECT * FROM public.insurance_claims WHERE user_uid=%s AND pet_id=%s AND deleted_at IS NULL ORDER BY claim_date DESC, created_at DESC", (uid, pet_uuid))
    else:
        rows = db_fetchall("SELECT * FROM public.insurance_claims WHERE user_uid=%s AND deleted_at IS NULL ORDER BY claim_date DESC, created_at DESC", (uid,))
    return jsonable_encoder(rows)


@app.get("/api/claims/get")
def api_claim_get(claimId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    claim_uuid = _uuid_or_400(claimId, "claimId")
    row = db_fetchone("SELECT * FROM public.insurance_claims WHERE id=%s AND user_uid=%s AND deleted_at IS NULL", (claim_uuid, uid))
    if not row:
        raise HTTPException(status_code=404, detail="claim not found")
    return jsonable_encoder(row)


@app.delete("/api/claims/delete")
def api_claim_delete(claimId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    claim_uuid = _uuid_or_400(claimId, "claimId")
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("UPDATE public.insurance_claims SET deleted_at = now() WHERE id=%s AND user_uid=%s AND deleted_at IS NULL", (claim_uuid, uid))
                if (cur.rowcount or 0) <= 0:
                    raise HTTPException(status_code=404, detail="claim not found (or already deleted)")
        return {"ok": True, "claimId": str(claim_uuid), "softDeleted": True}
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Prevent Schedules (v2.3.0)
# =========================================================
@app.post("/api/schedules/upsert")
def api_schedule_upsert(req: PreventScheduleUpsertRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    pet_uuid = _uuid_or_400(req.pet_id, "petId")
    sched_uuid = _uuid_or_new(req.id, "id")
    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    kind = (req.schedule_kind or "medical").strip().lower()
    if kind not in ("medical", "life", "other"):
        raise HTTPException(status_code=400, detail="scheduleKind must be 'medical', 'life', or 'other'")
    try:
        event_dt = datetime.fromisoformat(req.event_date.replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="eventDate must be ISO format")
    rule = (req.repeat_rule or "none").strip().lower()
    if rule not in ("none", "monthly", "every3m", "yearly", "custom"):
        raise HTTPException(status_code=400, detail="repeatRule invalid")
    interval = req.repeat_interval
    unit = (req.repeat_unit or "").strip().lower() or None
    if rule == "custom":
        if interval is None or interval <= 0:
            raise HTTPException(status_code=400, detail="repeatInterval must be > 0 for custom rule")
        if unit not in ("day", "week", "month"):
            raise HTTPException(status_code=400, detail="repeatUnit must be day/week/month for custom rule")
    else:
        interval = None
        unit = None
    try:
        row = db_fetchone(
            """
            INSERT INTO public.prevent_schedules
                (id, pet_id, user_uid, title, schedule_kind, event_date, alarm_enabled,
                 repeat_rule, repeat_interval, repeat_unit, memo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                schedule_kind = EXCLUDED.schedule_kind,
                event_date = EXCLUDED.event_date,
                alarm_enabled = EXCLUDED.alarm_enabled,
                repeat_rule = EXCLUDED.repeat_rule,
                repeat_interval = EXCLUDED.repeat_interval,
                repeat_unit = EXCLUDED.repeat_unit,
                memo = EXCLUDED.memo
            WHERE public.prevent_schedules.user_uid = %s AND public.prevent_schedules.deleted_at IS NULL
            RETURNING *
            """,
            (sched_uuid, pet_uuid, uid, title, kind, event_dt, req.alarm_enabled,
             rule, interval, unit, req.memo, uid),
        )
        if not row:
            raise HTTPException(status_code=409, detail="schedule not found or not editable")
        return jsonable_encoder(row)
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.get("/api/schedules/list")
def api_schedules_list(petId: Optional[str] = Query(None), scheduleKind: Optional[str] = Query(None), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")
    conditions = ["user_uid=%s", "deleted_at IS NULL"]
    params: List[Any] = [uid]
    if petId:
        pet_uuid = _uuid_or_400(petId, "petId")
        conditions.append("pet_id=%s")
        params.append(pet_uuid)
    if scheduleKind:
        sk = scheduleKind.strip().lower()
        if sk in ("medical", "life", "other"):
            conditions.append("schedule_kind=%s")
            params.append(sk)
    where = " AND ".join(conditions)
    rows = db_fetchall(f"SELECT * FROM public.prevent_schedules WHERE {where} ORDER BY event_date ASC, created_at DESC", tuple(params))
    return jsonable_encoder(rows)


@app.get("/api/schedules/get")
def api_schedule_get(scheduleId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    sched_uuid = _uuid_or_400(scheduleId, "scheduleId")
    row = db_fetchone("SELECT * FROM public.prevent_schedules WHERE id=%s AND user_uid=%s AND deleted_at IS NULL", (sched_uuid, uid))
    if not row:
        raise HTTPException(status_code=404, detail="schedule not found")
    return jsonable_encoder(row)


@app.delete("/api/schedules/delete")
def api_schedule_delete(scheduleId: str = Query(...), user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    sched_uuid = _uuid_or_400(scheduleId, "scheduleId")
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("UPDATE public.prevent_schedules SET deleted_at = now() WHERE id=%s AND user_uid=%s AND deleted_at IS NULL", (sched_uuid, uid))
                if (cur.rowcount or 0) <= 0:
                    raise HTTPException(status_code=404, detail="schedule not found (or already deleted)")
        return {"ok": True, "scheduleId": str(sched_uuid), "softDeleted": True}
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Admin overview
# =========================================================


# =========================================================
# AI Care Analysis (v2.3.0) — premium only
# =========================================================
class AICareAnalyzeRequest(BaseModel):
    request_date: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    recent_weights: Optional[List[Dict[str, Any]]] = Field(default=None, alias="recentWeights")
    medical_history: Optional[List[Dict[str, Any]]] = Field(default=None, alias="medicalHistory")
    schedules: Optional[List[Dict[str, Any]]] = None
    force_refresh: Optional[bool] = Field(default=False, alias="forceRefresh")

    class Config:
        populate_by_name = True


# ── AI 분석 결과 캐시 (메모리, 인스턴스 단위) ──
import hashlib as _hashlib
_ai_cache: Dict[str, Dict[str, Any]] = {}  # key: uid+data_hash → result
_AI_CACHE_MAX = 200  # 최대 캐시 항목수


def _make_ai_cache_key(uid: str, req: AICareAnalyzeRequest) -> str:
    """진료 데이터의 해시를 생성. 같은 데이터면 같은 키."""
    raw = json.dumps({
        "profile": req.profile or {},
        "history_count": len(req.medical_history or []),
        "history_tags": [
            (h.get("date", ""), h.get("tags", []))
            for h in (req.medical_history or [])[:20]
        ],
        "schedules_count": len(req.schedules or []),
        "weights_count": len(req.recent_weights or []),
    }, sort_keys=True, ensure_ascii=False)
    h = _hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{uid}:{h}"


@app.post("/api/ai/analyze")
def api_ai_analyze(req: AICareAnalyzeRequest, user: Dict[str, Any] = Depends(get_current_user)):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    # ── 캐시 확인 (force_refresh=false일 때) ──
    cache_key = _make_ai_cache_key(uid, req)
    if not req.force_refresh and cache_key in _ai_cache:
        logger.info("[AI Care] Cache HIT for %s", cache_key)
        cached = _ai_cache[cache_key]
        # ✅ 캐시 HIT 시 현재 사용량 정보도 함께 반환
        try:
            urow = db_fetchone("SELECT COALESCE(ai_usage_count,0) AS cnt, membership_tier, premium_until FROM public.users WHERE firebase_uid=%s", (uid,))
            etier = _effective_tier_from_row((urow or {}).get("membership_tier"), (urow or {}).get("premium_until")) if urow else "member"
            limit = None if etier == "premium" else 3
            cached["ai_usage_count"] = int((urow or {}).get("cnt") or 0)
            cached["ai_usage_limit"] = limit
        except Exception:
            pass
        return cached

    # ── AI 분석 쿼터 체크 + 월 자동 리셋 ──
    ai_limit = None
    ai_count = 0
    try:
        # 이번 달 1일 기준으로 리셋 체크
        now_utc = datetime.now(timezone.utc)
        first_of_month = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # ai_usage_reset이 이번 달 1일 이전이면 카운트 리셋
        db_execute(
            """UPDATE public.users
            SET ai_usage_count = 0, ai_usage_reset = %s
            WHERE firebase_uid = %s
              AND (ai_usage_reset IS NULL OR ai_usage_reset < %s)""",
            (first_of_month, uid, first_of_month),
        )

        urow = db_fetchone(
            "SELECT COALESCE(ai_usage_count,0) AS cnt, membership_tier, premium_until FROM public.users WHERE firebase_uid=%s",
            (uid,),
        )
        if urow:
            etier = _effective_tier_from_row(urow.get("membership_tier"), urow.get("premium_until"))
            ai_count = int(urow.get("cnt") or 0)
            if etier != "premium":
                ai_limit = 3
                if ai_count >= ai_limit:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "AI 케어 분석 무료 횟수(3회)를 모두 사용했어요. Pro 구독으로 무제한 이용하세요!",
                            "code": "AI_QUOTA_EXCEEDED",
                            "ai_usage_count": ai_count,
                            "ai_usage_limit": ai_limit,
                        },
                    )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[AI Care] Quota check failed (ignored): %s", _sanitize_for_log(str(e)))

    # ── Gemini 설정 확인 ──
    if not settings.GEMINI_ENABLED or not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI analysis is not available (Gemini not configured)")

    # ── condition_tags 로드 ──
    tag_codes_info = ""
    try:
        from condition_tags import CONDITION_TAGS as _CT
        tag_lines = []
        for code, cfg in _CT.items():
            if code != cfg.code:
                continue  # alias 건너뜀
            tag_lines.append(f"  - {cfg.code}: {cfg.label} (group={cfg.group})")
        tag_codes_info = "\n".join(tag_lines[:80])  # 상위 80개
    except Exception:
        tag_codes_info = "(태그 목록 로드 실패)"

    # ── 프롬프트 구성 ──
    profile = req.profile or {}
    weights = req.recent_weights or []
    history = req.medical_history or []
    scheds = req.schedules or []

    prompt_lines = [
        "너는 반려동물 건강 분석 전문 AI야.",
        "아래 정보를 분석해서 **순수 JSON만** 응답해. 마크다운(```), 설명, 인사말 절대 금지.",
        "",
        f"## 프로필",
        f"- 이름: {profile.get('name', '?')}",
        f"- 종: {profile.get('species', '?')}",
        f"- 나이: {profile.get('age_text', '?')}",
        f"- 현재 체중: {profile.get('weight_current', '?')}kg",
        f"- 알레르기: {', '.join(profile.get('allergies', []) or ['없음'])}",
        "",
    ]

    if weights:
        prompt_lines.append("## 최근 체중 기록")
        for w in weights[:12]:
            prompt_lines.append(f"- {w.get('date', '?')}: {w.get('weight', '?')}kg")
        prompt_lines.append("")

    if history:
        prompt_lines.append("## 진료 이력")
        for h in history[:20]:
            tags_str = ", ".join(h.get("tags", []) or [])
            prompt_lines.append(
                f"- {h.get('visit_date', '?')} {h.get('clinic_name', '?')} "
                f"(항목 {h.get('item_count', 0)}개) tags=[{tags_str}]"
            )
        prompt_lines.append("")

    if scheds:
        prompt_lines.append("## 예방/진료 스케줄")
        for s in scheds[:15]:
            upcoming = "예정" if s.get("is_upcoming") else "지남"
            prompt_lines.append(f"- {s.get('date', '?')} {s.get('title', '?')} ({upcoming})")
        prompt_lines.append("")

    prompt_lines.extend([
        "## 사용 가능한 태그 코드 (tag 필드에 아래 코드만 사용할 것)",
        tag_codes_info,
        "",
        "## 핵심 원칙",
        "당신은 경험 많은 수의사입니다. 보호자에게 건강 편지를 쓰듯 따뜻하지만 전문적으로 조언합니다.",
        "절대로 보호자가 '이미 아는 사실'을 반복하지 마세요. 대신 다음을 제공하세요:",
        "- 진료 기록 간의 연관성 (예: '수술 후 면역력 저하 → 감염 주의 시기')",
        "- 구체적인 시점/기한 (예: '수술 2주 후 경과 확인 필요', '3개월마다 혈액검사 추천')",
        "- 보호자가 놓칠 수 있는 패턴 (예: '최근 3개월간 수술이 잦았어요 — 면역 관리가 중요해요')",
        "- 약물 간 주의사항이나 관리 팁 (예: '스테로이드 장기 복용 시 간 수치 모니터링 필요')",
        "- '수술 받았어요', '검사했어요' 같은 당연한 사실은 생략하세요.",
        "",
        "## 응답 필드 (순수 JSON만 출력. 마크다운 금지. 이모지 금지. ~해요 체.)",
        "1. greeting: '{이름} 보호자님께' (고정 형식)",
        "2. health_keywords: 건강 프로필 키워드 2~3개 배열. 각 3~7자 한국어.",
        "   - 예: ['수술회복기', '식이알러지'], 진료 이력 없으면: ['건강관리시작']",
        "3. key_message: 가장 중요한 메시지 1~2문장. 보호자가 모르는 인사이트.",
        "   - 예: '수술 후 2주가 감염 위험이 가장 높은 시기예요. 상처 부위를 매일 확인해 주세요.'",
        "4. insights: 분야별 인사이트 2~3개 객체 배열. 각 객체:",
        "   - title: 짧은 제목 (3~6자). 예: '수술 회복', '식이 관리'",
        "   - body: 보호자가 모르는 정보 1~2문장. 왜 중요한지 포함.",
        "   - priority: 'high'(긴급/중요) 또는 'normal'",
        "5. care_actions: 보호자 실천 항목 2~3개 객체 배열. 각 객체:",
        "   - action: 구체적 행동 (15~25자). 예: '수술 경과 확인 예약하기'",
        "   - reason: 왜 해야 하는지 (10~20자). 예: '감염 위험이 높은 시기예요'",
        "6. closing: 마무리 1~2문장. 잘 하고 있다는 격려 + 응원.",
        "7. summary: (하위호환) greeting + key_message + closing을 합친 자연스러운 문단.",
        "8. tags: 진료 이력 기반 태그 (최대 5개). 진료 이력 있으면 빈 배열 금지.",
        "9. period_stats: 1m, 3m, 1y 기간별 태그 횟수.",
        "10. group_summary: group별 1줄 한국어 설명.",
        "11. care_guide: 빈 객체 {}. 서버가 자동 채움.",
        "12. 전체 응답이 반드시 완전한 JSON. 절대 중간에 끊기면 안 됨.",
        "",
        "## 응답 JSON 형식 (이 형식을 정확히 따를 것)",
        '{"greeting":"보리 보호자님께","health_keywords":["정기검진형","피부관리필요"],"key_message":"최근 피부염이 반복되고 있어요. 알러지 원인 검사를 받으면 근본적인 관리가 가능해요.","insights":[{"title":"피부 관리","body":"3개월간 피부염 치료 2회는 재발 패턴이에요. 환경성 알러지 검사로 원인을 특정하면 치료 효과가 높아져요.","priority":"high"},{"title":"예방접종","body":"기본 접종은 완료됐지만, 4세부터는 항체가 검사 후 선택 접종을 추천드려요.","priority":"normal"}],"care_actions":[{"action":"알러지 원인 검사 예약하기","reason":"재발 방지의 첫 단계예요"},{"action":"다음 달 광견병 접종 확인","reason":"접종 시기가 다가오고 있어요"}],"closing":"정기 검진을 꾸준히 받고 있어서 건강 관리가 잘 되고 있어요. 보리의 건강을 응원합니다.","summary":"보리 보호자님, ...","tags":[{"tag":"exam_blood_general","label":"혈액검사","count":2,"recent_dates":["2025-11-28"]}],"period_stats":{"1m":{"exam_blood_general":0},"3m":{"exam_blood_general":2},"1y":{"exam_blood_general":2}},"group_summary":{"exam":"혈액검사와 초음파를 정기적으로 받고 있어요"},"care_guide":{}}',
    ])

    prompt = "\n".join(prompt_lines)

    # ── Gemini 호출 ──
    import urllib.request as _ureq

    api_key = settings.GEMINI_API_KEY
    model = settings.GEMINI_MODEL_NAME or "gemini-2.5-flash"
    timeout = max(settings.GEMINI_TIMEOUT_SECONDS or 60, 60)

    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    gemini_payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }
    gemini_body = json.dumps(gemini_payload).encode("utf-8")
    gemini_req = _ureq.Request(
        gemini_url,
        data=gemini_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with _ureq.urlopen(gemini_req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        j = json.loads(raw)
        candidate = (j.get("candidates") or [{}])[0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        txt = (
            candidate
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        ).strip()

        logger.info("[AI Care] Gemini raw response length=%d finishReason=%s", len(txt), finish_reason)

        # finishReason이 MAX_TOKENS이면 응답이 잘린 것
        if finish_reason == "MAX_TOKENS":
            logger.warning("[AI Care] Gemini response was truncated (MAX_TOKENS)")

    except Exception as e:
        logger.exception("[AI Care] Gemini call failed: %r", e)
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {e}")

    # ── JSON 파싱 ──
    import re as _re

    logger.info("[AI Care] Gemini raw text (first 800): %s", txt[:800])

    # 1) 마크다운 코드블록 제거 (멀티라인)
    cleaned = _re.sub(r"```(?:json)?\s*\n?", "", txt, flags=_re.IGNORECASE)
    cleaned = _re.sub(r"\n?\s*```", "", cleaned)
    cleaned = cleaned.strip()

    # 2) JSON 객체 추출 (첫 번째 { ~ 마지막 })
    i = cleaned.find("{")
    k = cleaned.rfind("}")
    if i >= 0 and k > i:
        cleaned = cleaned[i : k + 1]
    else:
        logger.error("[AI Care] No JSON object found in: %s", cleaned[:500])
        cleaned = ""

    # 3) 파싱
    result = None
    if cleaned:
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as je:
            logger.error("[AI Care] JSON parse failed: %s | raw: %s", je, cleaned[:500])

    # 4) 파싱 실패 시 summary만이라도 추출
    if not result or not isinstance(result, dict):
        # summary 필드만이라도 regex로 추출 시도
        m = _re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]', txt, flags=_re.DOTALL)
        fallback_summary = m.group(1).replace('\\"', '"').replace("\\n", "\n") if m else ""
        if not fallback_summary:
            fallback_summary = "AI 분석 결과를 파싱하지 못했어요. 다시 시도해 주세요."

        result = {
            "summary": fallback_summary,
            "tags": [],
            "period_stats": {},
            "care_guide": {},
        }

    # ── 응답 정규화 ──
    summary_val = result.get("summary", "")
    # summary에 JSON이 통째로 들어간 경우 방어
    if summary_val.strip().startswith("{") or summary_val.strip().startswith("```"):
        m2 = _re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]', txt, flags=_re.DOTALL)
        if m2:
            summary_val = m2.group(1).replace('\\"', '"').replace("\\n", "\n")

    # ── care_guide: condition_tags.py에서 자동 보강 ──
    raw_tags = result.get("tags", [])
    care_guide = {}
    try:
        from condition_tags import CONDITION_TAGS as _CT
        for t in raw_tags:
            if isinstance(t, dict):
                tag_code = t.get("tag", "")
            elif isinstance(t, str):
                tag_code = t
            else:
                continue
            if tag_code and tag_code in _CT:
                care_guide[tag_code] = _CT[tag_code].guide
    except Exception as e:
        logger.warning("[AI Care] care_guide auto-fill failed: %s", e)
        care_guide = result.get("care_guide", {})

    # ── summary_lines / group_summary 추출 ──
    summary_lines = result.get("summary_lines", [])
    if not isinstance(summary_lines, list):
        summary_lines = []
    # summary_lines가 비었으면 summary에서 자동 생성 (하위호환)
    if not summary_lines and summary_val:
        # 문장 분리 시도
        sentences = [s.strip() for s in _re.split(r'[.!?]\s+|(?<=[요다죠세])\s+', summary_val) if s.strip()]
        if len(sentences) >= 3:
            summary_lines = sentences[:3]
        elif summary_val:
            summary_lines = [summary_val]

    group_summary = result.get("group_summary", {})
    if not isinstance(group_summary, dict):
        group_summary = {}

    health_keywords = result.get("health_keywords", [])
    if not isinstance(health_keywords, list):
        health_keywords = []
    if not health_keywords:
        health_keywords = ["건강관리시작"]

    # ── 건강 편지 필드 추출 ──
    greeting = result.get("greeting", "")
    if not isinstance(greeting, str) or not greeting:
        greeting = f"{pet_name} 보호자님께" if pet_name else "보호자님께"

    key_message = result.get("key_message", "")
    if not isinstance(key_message, str):
        key_message = ""
    if not key_message and summary_lines:
        key_message = summary_lines[0] if summary_lines else summary_val

    insights = result.get("insights", [])
    if not isinstance(insights, list):
        insights = []
    # insights 각 항목 검증
    validated_insights = []
    for ins in insights:
        if isinstance(ins, dict) and "title" in ins and "body" in ins:
            validated_insights.append({
                "title": str(ins.get("title", "")),
                "body": str(ins.get("body", "")),
                "priority": str(ins.get("priority", "normal")),
            })
    insights = validated_insights

    care_actions = result.get("care_actions", [])
    if not isinstance(care_actions, list):
        care_actions = []
    # care_actions 각 항목 검증 (객체 또는 문자열 둘 다 수용)
    validated_actions = []
    for act in care_actions:
        if isinstance(act, dict) and "action" in act:
            validated_actions.append({
                "action": str(act.get("action", "")),
                "reason": str(act.get("reason", "")),
            })
        elif isinstance(act, str) and act:
            validated_actions.append({"action": act, "reason": ""})
    care_actions = validated_actions
    # care_actions가 비었으면 summary_lines에서 생성
    if not care_actions and summary_lines and len(summary_lines) >= 3:
        care_actions = [{"action": summary_lines[2], "reason": ""}]

    closing = result.get("closing", "")
    if not isinstance(closing, str):
        closing = ""

    response = {
        "summary": summary_val,
        "summary_lines": summary_lines,
        "health_keywords": health_keywords,
        "greeting": greeting,
        "key_message": key_message,
        "insights": insights,
        "care_actions": care_actions,
        "closing": closing,
        "tags": raw_tags,
        "period_stats": result.get("period_stats", {}),
        "group_summary": group_summary,
        "care_guide": care_guide,
    }

    # ── 캐시 저장 ──
    if len(_ai_cache) >= _AI_CACHE_MAX:
        # 오래된 항목 절반 제거
        keys = list(_ai_cache.keys())
        for k in keys[:_AI_CACHE_MAX // 2]:
            _ai_cache.pop(k, None)
    _ai_cache[cache_key] = response
    logger.info("[AI Care] Cache STORED for %s", cache_key)

    # ✅ v2.3.1: 실제 Gemini 호출 시에만 카운트 증가 (캐시 MISS)
    new_count = ai_count
    try:
        db_execute(
            "UPDATE public.users SET ai_usage_count = COALESCE(ai_usage_count, 0) + 1 WHERE firebase_uid = %s",
            (uid,),
        )
        new_count = ai_count + 1
        logger.info("[AI Care] Usage count incremented for %s: %d -> %d", uid, ai_count, new_count)
    except Exception as e:
        logger.warning("[AI Care] Usage count update failed (ignored): %s", _sanitize_for_log(str(e)))

    response["ai_usage_count"] = new_count
    response["ai_usage_limit"] = ai_limit

    return response


@app.get("/api/admin/overview")
def admin_overview(admin: Dict[str, Any] = Depends(get_admin_user)):
    users = db_fetchone("SELECT COUNT(*) AS cnt FROM public.users")
    pets = db_fetchone("SELECT COUNT(*) AS cnt FROM public.pets")
    records = db_fetchone("SELECT COUNT(*) AS cnt FROM public.health_records WHERE deleted_at IS NULL")
    docs = db_fetchone("SELECT COUNT(*) AS cnt FROM public.pet_documents WHERE deleted_at IS NULL")
    claims = db_fetchone("SELECT COUNT(*) AS cnt FROM public.insurance_claims WHERE deleted_at IS NULL")
    schedules = db_fetchone("SELECT COUNT(*) AS cnt FROM public.prevent_schedules WHERE deleted_at IS NULL")
    dau = db_fetchone("SELECT COUNT(*) AS cnt FROM public.user_daily_active WHERE day = CURRENT_DATE")
    storage = db_fetchone("SELECT COALESCE(SUM(total_storage_bytes),0) AS total FROM public.users")
    return {
        "users": int((users or {}).get("cnt") or 0),
        "pets": int((pets or {}).get("cnt") or 0),
        "records": int((records or {}).get("cnt") or 0),
        "documents": int((docs or {}).get("cnt") or 0),
        "claims": int((claims or {}).get("cnt") or 0),
        "schedules": int((schedules or {}).get("cnt") or 0),
        "dauToday": int((dau or {}).get("cnt") or 0),
        "totalStorageBytes": int((storage or {}).get("total") or 0),
    }


# =========================================================
# Subscription tier update (v2.3.0+)
# =========================================================
class UpdateTierRequest(BaseModel):
    membership_tier: str


@app.post("/api/me/update-tier")
def api_update_tier(
    req: UpdateTierRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    tier = (req.membership_tier or "").strip().lower()
    if tier not in ("member", "premium"):
        raise HTTPException(status_code=400, detail=f"invalid tier: {tier}")

    try:
        db_execute(
            "UPDATE public.users SET membership_tier=%s, updated_at=now() WHERE firebase_uid=%s",
            (tier, uid),
        )
    except Exception as e:
        _raise_mapped_db_error(e)

    return {"ok": True, "tier": tier}

