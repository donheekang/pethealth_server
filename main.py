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
from datetime import datetime, date, timedelta
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

    RECEIPT_MAX_WIDTH: int = 1024
    RECEIPT_WEBP_QUALITY: int = 85

    MAX_RECEIPT_IMAGE_BYTES: int = 10 * 1024 * 1024
    MAX_PDF_BYTES: int = 20 * 1024 * 1024
    MAX_BACKUP_BYTES: int = 5 * 1024 * 1024
    IMAGE_MAX_PIXELS: int = 20_000_000

    SIGNED_URL_DEFAULT_TTL_SECONDS: int = 600
    SIGNED_URL_MAX_TTL_SECONDS: int = 3600

    MIGRATION_TOKEN_TTL_SECONDS: int = 10 * 60

    OCR_HOSPITAL_CANDIDATE_LIMIT: int = 3

    GEMINI_ENABLED: bool = False
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT_SECONDS: int = 10

    TAG_RECORD_THRESHOLD: int = 125
    TAG_ITEM_THRESHOLD: int = 140

    DB_ENABLED: bool = True
    DATABASE_URL: str = ""
    DB_POOL_MIN: int = 1
    DB_POOL_MAX: int = 5
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
    visit_date: date = Field(alias="visitDate")
    hospital_name: Optional[str] = Field(default=None, alias="hospitalName")
    hospital_mgmt_no: Optional[str] = Field(default=None, alias="hospitalMgmtNo")
    total_amount: Optional[int] = Field(default=None, alias="totalAmount")
    pet_weight_at_visit: Optional[float] = Field(default=None, alias="petWeightAtVisit")
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
app = FastAPI(title="PetHealth+ Server", version="2.3.0")

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


@app.on_event("startup")
def _startup():
    init_db_pool()
    if settings.AUTH_REQUIRED and (not settings.STUB_MODE):
        try:
            init_firebase_admin(require_init=True)
        except Exception as e:
            logger.warning("[Startup] Firebase init failed: %s", _sanitize_for_log(repr(e)))


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
    return {"status": "ok", "message": "PetHealth+ Server Running (v2.3.0)"}


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
        "version": "2.3.0",
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
        "claim_count": int(row.get("claim_count") or 0),
        "schedule_count": int(row.get("schedule_count") or 0),
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
    visit_date = req.visit_date
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
                if p < 0:
                    raise HTTPException(status_code=400, detail="item price must be >= 0")
                has_any_price = True
                items_sum += p
    tags = _clean_tags(req.tags)
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
                            if price < 0:
                                raise HTTPException(status_code=400, detail="item price must be >= 0")
                        category_tag = it.category_tag.strip() if isinstance(it.category_tag, str) and it.category_tag.strip() else None
                        cur.execute("INSERT INTO public.health_items (record_id, item_name, price, category_tag) VALUES (%s, %s, %s, %s)", (record_uuid, item_name, price, category_tag))
                cur.execute("SELECT id, record_id, item_name, price, category_tag, created_at, updated_at FROM public.health_items WHERE record_id=%s ORDER BY created_at ASC", (record_uuid,))
                items_rows = cur.fetchall() or []
                payload = dict(row)
                payload["items"] = [dict(x) for x in items_rows]
                return jsonable_encoder(payload)
    except HTTPException:
        raise
    except Exception as e:
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
                "SELECT r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount, r.pet_weight_at_visit, r.tags, r.receipt_image_path, r.file_size_bytes, r.created_at, r.updated_at FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND p.id=%s AND r.deleted_at IS NULL ORDER BY r.visit_date DESC, r.created_at DESC",
                (uid, pet_uuid),
            )
        else:
            rows = db_fetchall(
                "SELECT r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount, r.pet_weight_at_visit, r.tags, r.receipt_image_path, r.file_size_bytes, r.created_at, r.updated_at FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND r.deleted_at IS NULL ORDER BY r.visit_date DESC, r.created_at DESC",
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
            "SELECT r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount, r.pet_weight_at_visit, r.tags, r.receipt_image_path, r.file_size_bytes, r.created_at, r.updated_at FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND r.id=%s AND r.deleted_at IS NULL",
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
                 "넥스가드", "브라벡토", "레볼루션", "하트가드", "프론트라인"],
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
    pet = db_fetchone("SELECT id, weight_kg FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    raw = _read_upload_limited(file, int(settings.MAX_RECEIPT_IMAGE_BYTES))
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")

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
    webp_bytes = result.get("webp_bytes") or raw          # 마스킹본
    original_webp = result.get("original_webp_bytes")      # 원본
    content_type = result.get("content_type") or "image/webp"

    ocr_hospital_name_raw = meta.get("hospital_name") or ""
    ocr_visit_date_raw = meta.get("visit_date") or ""

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
                except Exception:
                    pass
        if not cat_tag:
            cat_tag = _fallback_classify_item(mapped)

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

    # --- parse visit date ---
    vd: Optional[date] = None
    if visitDate:
        try:
            vd = date.fromisoformat(visitDate)
        except Exception:
            pass
    if vd is None and ocr_visit_date_raw:
        try:
            vd = date.fromisoformat(ocr_visit_date_raw)
        except Exception:
            pass
    if vd is None:
        vd = date.today()

    # --- hospital matching ---
    hosp_name = hospitalName.strip() if isinstance(hospitalName, str) and hospitalName.strip() else None
    hosp_mgmt = hospitalMgmtNo.strip() if isinstance(hospitalMgmtNo, str) and hospitalMgmtNo.strip() else None

    if not hosp_name and ocr_hospital_name_raw:
        hosp_name = ocr_hospital_name_raw.strip()

    # --- record upsert ---
    record_uuid = _uuid_or_new(existingRecordId, "existingRecordId")

    total_amount = sum(fi.get("price") or 0 for fi in final_items if fi.get("price") is not None and fi["price"] >= 0)

    pet_weight = None
    try:
        wkg = pet.get("weight_kg")
        pet_weight = float(wkg) if wkg is not None else None
    except Exception:
        pass

    file_path = _receipt_path(uid, str(pet_uuid), str(record_uuid))
    file_size = int(len(webp_bytes))

    logger.info("[receipt] original_webp present=%s, size=%s, webp_bytes size=%s",
                original_webp is not None,
                len(original_webp) if original_webp else 0,
                len(webp_bytes))

    try:
        # 마스킹본 (앱 표시용)
        upload_bytes_to_storage(file_path, webp_bytes, content_type)
        # 원본 (보험 청구 PDF용)
        if original_webp:
            orig_path = _receipt_original_path(uid, str(pet_uuid), str(record_uuid))
            logger.info("[receipt] uploading original to %s (%d bytes)", orig_path, len(original_webp))
            try:
                upload_bytes_to_storage(orig_path, original_webp, content_type)
                # verify upload
                b = get_bucket()
                blob = b.blob(orig_path)
                exists = blob.exists()
                logger.info("[receipt] original upload verify: exists=%s, path=%s", exists, orig_path)
            except Exception as orig_err:
                logger.error("[receipt] original upload FAILED: %s", orig_err)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))

    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO public.health_records
                        (id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags, receipt_image_path, file_size_bytes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        hospital_mgmt_no = EXCLUDED.hospital_mgmt_no,
                        hospital_name = EXCLUDED.hospital_name,
                        visit_date = EXCLUDED.visit_date,
                        total_amount = EXCLUDED.total_amount,
                        pet_weight_at_visit = EXCLUDED.pet_weight_at_visit,
                        tags = EXCLUDED.tags,
                        receipt_image_path = EXCLUDED.receipt_image_path,
                        file_size_bytes = EXCLUDED.file_size_bytes
                    WHERE public.health_records.deleted_at IS NULL
                        AND EXISTS (SELECT 1 FROM public.pets p WHERE p.id = public.health_records.pet_id AND p.user_uid = %s)
                    RETURNING id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags,
                        receipt_image_path, file_size_bytes, created_at, updated_at
                    """,
                    (record_uuid, pet_uuid, hosp_mgmt, hosp_name, vd, total_amount, pet_weight, record_tags, file_path, file_size, uid),
                )
                rec_row = cur.fetchone()
                if not rec_row:
                    raise HTTPException(status_code=500, detail=_internal_detail("receipt record upsert failed", kind="DB error"))

                cur.execute("DELETE FROM public.health_items WHERE record_id=%s", (record_uuid,))
                for fi in final_items:
                    iname = (fi.get("itemName") or "").strip()
                    if not iname:
                        continue
                    iprice = fi.get("price")
                    if iprice is not None:
                        iprice = int(iprice)
                    cur.execute("INSERT INTO public.health_items (record_id, item_name, price, category_tag) VALUES (%s, %s, %s, %s)", (record_uuid, iname, iprice, fi.get("categoryTag")))

                cur.execute("SELECT id, record_id, item_name, price, category_tag, created_at, updated_at FROM public.health_items WHERE record_id=%s ORDER BY created_at ASC", (record_uuid,))
                items_rows = cur.fetchall() or []

                # --- hospital candidates ---
                candidate_limit = int(settings.OCR_HOSPITAL_CANDIDATE_LIMIT or 3)
                if hosp_name and not hosp_mgmt and candidate_limit > 0:
                    like = f"%{hosp_name}%"
                    cur.execute(
                        "SELECT hospital_mgmt_no, name, road_address, jibun_address, lat, lng, is_custom_entry, similarity(search_vector, %s) AS score FROM public.hospitals WHERE (is_custom_entry = false OR created_by_uid = %s) AND search_vector ILIKE %s ORDER BY score DESC LIMIT %s",
                        (hosp_name, uid, like, candidate_limit),
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

                # Build hospital candidates list for response
                resp_candidates = []
                if hosp_name and not hosp_mgmt:
                    cur.execute(
                        "SELECT c.rank, c.score, h.hospital_mgmt_no, h.name, h.road_address, h.jibun_address, h.lat, h.lng, h.is_custom_entry FROM public.health_record_hospital_candidates c JOIN public.hospitals h ON h.hospital_mgmt_no = c.hospital_mgmt_no WHERE c.record_id = %s ORDER BY c.rank ASC",
                        (record_uuid,),
                    )
                    cand_rows = cur.fetchall() or []
                    for cr in cand_rows:
                        resp_candidates.append({
                            "rank": int(cr["rank"]),
                            "score": float(cr["score"]) if cr.get("score") is not None else None,
                            "hospitalMgmtNo": cr["hospital_mgmt_no"],
                            "name": cr["name"],
                            "roadAddress": cr.get("road_address"),
                            "jibunAddress": cr.get("jibun_address"),
                            "lat": cr.get("lat"),
                            "lng": cr.get("lng"),
                            "isCustomEntry": bool(cr.get("is_custom_entry")),
                        })

                # ── Response: iOS ReceiptDraftResponseDTO 포맷 ──
                # Build standardName lookup from final_items
                std_name_map: Dict[str, str] = {}
                for fi in final_items:
                    fn = (fi.get("itemName") or "").strip().lower()
                    sn = fi.get("standardName") or ""
                    if fn and sn:
                        std_name_map[fn] = sn

                resp_items = []
                for x in items_rows:
                    iname = (x.get("item_name") or "").strip()
                    resp_items.append({
                        "itemName": iname,
                        "price": x.get("price"),
                        "categoryTag": x.get("category_tag"),
                        "standardName": std_name_map.get(iname.lower()),
                    })

                draft_response = {
                    "mode": "direct",
                    "draftId": str(rec_row["id"]),
                    "petId": str(rec_row["pet_id"]),
                    "draftReceiptPath": rec_row.get("receipt_image_path") or file_path,
                    "fileSizeBytes": int(rec_row.get("file_size_bytes") or file_size),
                    "visitDate": rec_row["visit_date"].isoformat() if hasattr(rec_row.get("visit_date"), "isoformat") else str(rec_row.get("visit_date") or vd),
                    "hospitalName": rec_row.get("hospital_name"),
                    "hospitalMgmtNo": rec_row.get("hospital_mgmt_no"),
                    "totalAmount": int(rec_row.get("total_amount") or 0),
                    "petWeightAtVisit": float(rec_row["pet_weight_at_visit"]) if rec_row.get("pet_weight_at_visit") is not None else None,
                    "tags": rec_row.get("tags") or [],
                    "items": resp_items,
                    "hospitalCandidates": resp_candidates if resp_candidates else None,
                    "hospitalCandidateCount": len(resp_candidates) if resp_candidates else 0,
                    "hospitalConfirmed": bool(rec_row.get("hospital_mgmt_no")),
                    "tagEvidence": None,
                    "ocrText": ocr_text[:2000],
                    "ocrMeta": meta,
                }

                return draft_response

    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


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
    rec = db_fetchone("SELECT 1 FROM public.health_records r JOIN public.pets p ON p.id = r.pet_id WHERE p.user_uid=%s AND r.deleted_at IS NULL AND r.receipt_image_path=%s LIMIT 1", (uid, path))
    if rec:
        return {"kind": "receipt"}
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

