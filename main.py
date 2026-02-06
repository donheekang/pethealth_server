# main.py (PetHealth+ Server) — v2.2.1-ops (DB-aligned)
# Firebase Storage + Signed URL + Migration
# + DB schema aligned with:
#   - Soft delete SoT (deleted_at => path NULL + bytes=0) via DB triggers
#   - No undelete (enforced by DB)
#   - Quota enforcement via SELECT ... FOR UPDATE on users row (DB triggers call fn_guard_quota_for_user)
#   - Accounting via triggers (users.total_storage_bytes, counts)
#   - Storage delete jobs enqueued by DB triggers on soft-delete transition
#
# Architecture:
#   main.py     : API + DB I/O + pipeline wiring
#   ocr_policy  : OCR + redaction + receipt parsing -> items/meta (NO tag decision)
#   tag_policy  : items/text -> standard tag codes (SoT is your ReceiptTag codes)
#
# ✅ Tier policy (DB is source of truth):
#   membership_tier: 'guest' | 'member' | 'premium'
#   effective_tier : 'premium' only when premium_until > now(), otherwise membership_tier.
#   quota          : public.get_tier_quota(effective_tier)

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
    # --- OCR policy module may use these ---
    GOOGLE_APPLICATION_CREDENTIALS: str = ""  # JSON string or file path
    OCR_TIMEOUT_SECONDS: int = 12
    OCR_MAX_CONCURRENCY: int = 4
    OCR_SEMA_ACQUIRE_TIMEOUT_SECONDS: float = 1.0

    # --- Firebase Auth / Storage ---
    AUTH_REQUIRED: bool = True
    STUB_MODE: bool = False

    FIREBASE_ADMIN_SA_JSON: str = ""  # service account JSON string
    FIREBASE_ADMIN_SA_B64: str = ""   # base64-encoded JSON string (optional)
    FIREBASE_STORAGE_BUCKET: str = ""  # e.g. "<project-id>.appspot.com"

    # --- Receipt image pipeline (used by ocr_policy) ---
    RECEIPT_MAX_WIDTH: int = 1024
    RECEIPT_WEBP_QUALITY: int = 85

    # Upload hardening
    MAX_RECEIPT_IMAGE_BYTES: int = 10 * 1024 * 1024
    MAX_PDF_BYTES: int = 20 * 1024 * 1024
    MAX_BACKUP_BYTES: int = 5 * 1024 * 1024
    IMAGE_MAX_PIXELS: int = 20_000_000

    # Signed URL
    SIGNED_URL_DEFAULT_TTL_SECONDS: int = 600
    SIGNED_URL_MAX_TTL_SECONDS: int = 3600

    # Migration token TTL
    MIGRATION_TOKEN_TTL_SECONDS: int = 10 * 60

    # OCR hospital candidates
    OCR_HOSPITAL_CANDIDATE_LIMIT: int = 3

    # --- Postgres ---
    DB_ENABLED: bool = True
    DATABASE_URL: str = ""
    DB_POOL_MIN: int = 1
    DB_POOL_MAX: int = 5
    DB_AUTO_UPSERT_USER: bool = True

    # db_touch_user write throttle (seconds)
    USER_TOUCH_THROTTLE_SECONDS: int = 300

    # Error detail exposure (prod에서는 false 권장)
    EXPOSE_ERROR_DETAILS: bool = False

    # CORS
    CORS_ALLOW_ORIGINS: str = "*"  # comma-separated
    CORS_ALLOW_CREDENTIALS: bool = False

    # --- Admin ---
    ADMIN_UIDS: str = ""

    # =========================================================
    # ✅ Gemini (AI enrichment for receipt understanding)
    # =========================================================
    GEMINI_ENABLED: bool = False
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT_SECONDS: int = 8

    # (옵션) GCP 위치/프로젝트를 쓰는 코드가 있다면 대비용
    GCP_PROJECT_ID: str = ""
    GCP_LOCATION: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()




# =========================================================
# Optional policy modules (separated files)
# =========================================================
try:
    import ocr_policy  # type: ignore
except Exception as e:
    ocr_policy = None
    logger.exception("[Import] ocr_policy import failed: %r", e)

try:
    import tag_policy  # type: ignore
except Exception as e:
    tag_policy = None
    logger.exception("[Import] tag_policy import failed: %r", e)


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
    """
    effective_tier: premium only when premium_until > now(), else membership_tier
    membership_tier is DB SoT: guest/member/premium
    """
    tier = (membership_tier or "guest").strip().lower()
    if tier not in ("guest", "member", "premium"):
        tier = "guest"

    try:
        if premium_until is not None:
            # psycopg2 usually returns datetime for timestamptz
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

    # ---- High-signal schema mismatch / operator errors ----
    if "function public.get_effective_tier" in msg and "does not exist" in msg:
        raise HTTPException(status_code=503, detail="DB schema mismatch: get_effective_tier() is missing")
    if "users_membership_tier_check" in msg or "membership_tier" in msg and "check constraint" in msg:
        raise HTTPException(status_code=500, detail="Server tier mapping bug: membership_tier check failed")

    # ---- User-safe errors ----
    if "Quota exceeded" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "Undelete is not allowed" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "Ownership mismatch" in msg or "Hospital access denied" in msg:
        raise HTTPException(status_code=403, detail=msg)

    if "Candidates not allowed" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "Hard DELETE is blocked" in msg:
        raise HTTPException(status_code=409, detail="Hard DELETE is blocked. Use soft delete.")

    if "violates foreign key constraint" in msg:
        raise HTTPException(status_code=400, detail="Invalid reference (foreign key)")

    if "duplicate key value violates unique constraint" in msg:
        raise HTTPException(status_code=409, detail="Duplicate value (unique constraint)")

    # Generic
    raise HTTPException(status_code=500, detail=_internal_detail(msg, kind="DB error"))


# =========================================================
# Auth / membership helpers (DB-aligned)
# =========================================================
def _infer_membership_tier_from_token(decoded: Dict[str, Any]) -> Optional[str]:
    """
    DB tiers:
      guest   : Firebase Anonymous Auth
      member  : any non-anonymous provider (google.com, apple.com, password, etc.)
      premium : NEVER auto-set here (payments/admin only)
    """
    fb = decoded.get("firebase") or {}
    if isinstance(fb, dict):
        provider = (fb.get("sign_in_provider") or "").lower()
        if provider == "anonymous":
            return "guest"
        if provider:
            return "member"
    return None


def db_touch_user(firebase_uid: str, desired_tier: Optional[str] = None) -> Dict[str, Any]:
    """
    users schema:
      firebase_uid (pk)
      membership_tier (guest/member/premium)
      premium_until
      pet_count/record_count/doc_count/total_storage_bytes
      last_seen_at/updated_at/created_at

    정책:
      - 기본 guest
      - 자동 업그레이드: desired_tier='member' 이고 현재 tier가 guest면 member로 올림
      - premium은 절대 자동으로 덮어쓰지 않음
      - write throttle: last_seen_at 갱신은 N초 간격으로만
    """
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

            # conflict + WHERE false면 RETURNING이 비어있을 수 있음 -> SELECT로 보정
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

            # optional DAU
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
# Firebase Admin init (Auth/Storage decoupled)
# =========================================================
_firebase_initialized = False
auth_scheme = HTTPBearer(auto_error=False)

# Stub storage (in-memory)
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
    """
    Initialize Firebase Admin SDK.

    ✅ ops-hardening (auth/storage decoupled):
    - _firebase_initialized는 "실제로 firebase_admin 앱이 존재할 때만" True.
    - Auth(verify_id_token)에는 Service Account만 필요. (bucket 없어도 초기화 가능)
    - Storage bucket은 get_bucket()에서 별도로 강제한다.
    """
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
# Path helpers (fixed structure)
# users/{uid}/pets/{petId}/receipts/{recordId}.webp
# users/{uid}/pets/{petId}/{lab|cert}/{docId}.pdf
# users/{uid}/backups/{backupId}.json
# =========================================================
def _user_prefix(uid: str, pet_id: str) -> str:
    return f"users/{uid}/pets/{pet_id}"


def _backup_prefix(uid: str) -> str:
    return f"users/{uid}/backups"


def _receipt_path(uid: str, pet_id: str, record_id: str) -> str:
    return f"{_user_prefix(uid, pet_id)}/receipts/{record_id}.webp"


def _doc_pdf_path(uid: str, pet_id: str, doc_type: str, doc_id: str) -> str:
    return f"{_user_prefix(uid, pet_id)}/{doc_type}/{doc_id}.pdf"


# =========================================================
# Upload read limits
# =========================================================
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


class DocumentUploadResponse(BaseModel):
    id: str
    petId: str
    docType: str
    displayName: str
    filePath: str
    fileSizeBytes: int
    createdAt: str
    updatedAt: str


class OCRItemMapUpsertRequest(BaseModel):
    ocrItemName: str
    canonicalName: str
    isActive: Optional[bool] = True


class OCRItemMapDeactivateRequest(BaseModel):
    ocrItemName: str


# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(title="PetHealth+ Server", version="2.2.1-ops")

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
    return {"status": "ok", "message": "PetHealth+ Server Running (v2.2.1-ops)"}


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
        "version": "2.2.1-ops",
        "storage": "stub" if settings.STUB_MODE else "firebase",
        "db_enabled": bool(settings.DB_ENABLED),
        "db_configured": bool(settings.DATABASE_URL),
        "db_schema": db_checks,
        "firebase": fb_check,
        "cors": {"origins": _origins, "allowCredentials": _allow_credentials},
        "modules": {
            "ocr_policy": bool(ocr_policy is not None),
            "tag_policy": bool(tag_policy is not None),
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
    # keep user row warm
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
# Hospitals: search + custom create/delete
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
def hospitals_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=50),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = (user.get("uid") or "").strip()
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)

    query = (q or "").strip()
    like = f"%{query}%"

    rows = db_fetchall(
        """
        SELECT
            hospital_mgmt_no,
            name,
            road_address,
            jibun_address,
            lng,
            lat,
            is_custom_entry
        FROM public.hospitals
        WHERE
            (is_custom_entry = false OR created_by_uid = %s)
            AND search_vector ILIKE %s
        ORDER BY similarity(search_vector, %s) DESC
        LIMIT %s
        """,
        (uid, like, query, limit),
    )

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "hospitalMgmtNo": r["hospital_mgmt_no"],
                "name": r["name"],
                "roadAddress": r.get("road_address"),
                "jibunAddress": r.get("jibun_address"),
                "lng": r.get("lng"),
                "lat": r.get("lat"),
                "isCustomEntry": bool(r.get("is_custom_entry")),
            }
        )
    return out


@app.get("/api/hospitals/nearby")
def hospitals_nearby(
    lat: float = Query(...),
    lng: float = Query(...),
    radiusM: int = Query(3000, ge=200, le=20000),
    limit: int = Query(50, ge=1, le=200),
    user: Dict[str, Any] = Depends(get_current_user),
):
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
        """
        SELECT
            hospital_mgmt_no, name,
            road_address, jibun_address,
            lat, lng,
            is_custom_entry
        FROM public.hospitals
        WHERE
            (is_custom_entry = false OR created_by_uid = %s)
            AND lat IS NOT NULL AND lng IS NOT NULL
            AND lat BETWEEN %s AND %s
            AND lng BETWEEN %s AND %s
        LIMIT %s
        """,
        (uid, min_lat, max_lat, min_lng, max_lng, limit * 8),
    )

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = _haversine_m(lat, lng, r["lat"], r["lng"])
        if d <= radiusM:
            out.append(
                {
                    "hospitalMgmtNo": r["hospital_mgmt_no"],
                    "name": r["name"],
                    "roadAddress": r.get("road_address"),
                    "jibunAddress": r.get("jibun_address"),
                    "lat": r["lat"],
                    "lng": r["lng"],
                    "distanceM": int(d),
                    "isCustomEntry": bool(r.get("is_custom_entry")),
                }
            )

    out.sort(key=lambda x: x["distanceM"])
    return out[:limit]


class HospitalCustomCreateRequest(BaseModel):
    name: str
    roadAddress: Optional[str] = None
    jibunAddress: Optional[str] = None
    lng: Optional[float] = None
    lat: Optional[float] = None


@app.post("/api/hospitals/custom/create")
def hospitals_custom_create(
    req: HospitalCustomCreateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
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
            """
            INSERT INTO public.hospitals
                (hospital_mgmt_no, name, road_address, jibun_address, lng, lat, is_custom_entry, created_by_uid)
            VALUES
                (%s, %s, %s, %s, %s, %s, true, %s)
            RETURNING hospital_mgmt_no, name, road_address, jibun_address, lng, lat, is_custom_entry
            """,
            (mgmt_no, name, road_val, jibun_val, req.lng, req.lat, uid),
        )
        if not row:
            raise HTTPException(status_code=500, detail=_internal_detail("failed to create hospital", kind="DB error"))

        return {
            "hospitalMgmtNo": row["hospital_mgmt_no"],
            "name": row["name"],
            "roadAddress": row.get("road_address"),
            "jibunAddress": row.get("jibun_address"),
            "lng": row.get("lng"),
            "lat": row.get("lat"),
            "isCustomEntry": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.delete("/api/hospitals/custom/delete")
def hospitals_custom_delete(
    hospitalMgmtNo: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = (user.get("uid") or "").strip()
    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)

    mg = (hospitalMgmtNo or "").strip()
    if not mg:
        raise HTTPException(status_code=400, detail="hospitalMgmtNo is required")

    row = db_fetchone(
        """
        SELECT hospital_mgmt_no
        FROM public.hospitals
        WHERE hospital_mgmt_no=%s AND is_custom_entry=true AND created_by_uid=%s
        """,
        (mg, uid),
    )
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
#   - IMPORTANT: active records are deleted_at IS NULL
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
                cur.execute(
                    "SELECT id, weight_kg FROM public.pets WHERE id=%s AND user_uid=%s",
                    (pet_uuid, uid),
                )
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

                # ✅ important: block updates to soft-deleted records
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
                    WHERE
                        public.health_records.deleted_at IS NULL
                        AND EXISTS (
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

    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.get("/api/db/records/list")
def api_records_list(
    petId: Optional[str] = Query(None),
    includeItems: bool = Query(False),
    user: Dict[str, Any] = Depends(get_current_user),
):
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
                """
                SELECT
                    r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount,
                    r.pet_weight_at_visit, r.tags,
                    r.receipt_image_path, r.file_size_bytes,
                    r.created_at, r.updated_at
                FROM public.health_records r
                JOIN public.pets p ON p.id = r.pet_id
                WHERE
                    p.user_uid=%s AND p.id=%s
                    AND r.deleted_at IS NULL
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
                WHERE
                    p.user_uid=%s
                    AND r.deleted_at IS NULL
                ORDER BY r.visit_date DESC, r.created_at DESC
                """,
                (uid,),
            )

        if not includeItems:
            return jsonable_encoder(rows)

        record_ids = [r["id"] for r in rows if r.get("id")]
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

    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.get("/api/db/records/get")
def api_record_get(
    recordId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = (user.get("uid") or "").strip()
    rid = (recordId or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="recordId is required")

    try:
        record_uuid = _uuid_or_400(rid, "recordId")

        row = db_fetchone(
            """
            SELECT
                r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount,
                r.pet_weight_at_visit, r.tags,
                r.receipt_image_path, r.file_size_bytes,
                r.created_at, r.updated_at
            FROM public.health_records r
            JOIN public.pets p ON p.id = r.pet_id
            WHERE
                p.user_uid=%s AND r.id=%s
                AND r.deleted_at IS NULL
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

    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Record delete (SOFT DELETE)
# =========================================================
@app.delete("/api/db/records/delete")
def api_record_delete(
    recordId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    ✅ Soft delete only.
    - sets deleted_at=now()
    - trigger enqueues storage delete job (once) using OLD path
    - trigger enforces SoT: receipt_image_path NULL, file_size_bytes=0
    """
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    record_uuid = _uuid_or_400(recordId, "recordId")

    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT r.id, r.receipt_image_path
                    FROM public.health_records r
                    JOIN public.pets p ON p.id = r.pet_id
                    WHERE p.user_uid=%s AND r.id=%s AND r.deleted_at IS NULL
                    """,
                    (uid, record_uuid),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="record not found (or already deleted)")

                old_path = row.get("receipt_image_path")

                cur.execute(
                    """
                    UPDATE public.health_records r
                    SET deleted_at = now()
                    WHERE r.id=%s
                      AND r.deleted_at IS NULL
                      AND EXISTS (
                        SELECT 1 FROM public.pets p
                        WHERE p.id = r.pet_id AND p.user_uid = %s
                      )
                    """,
                    (record_uuid, uid),
                )
                if (cur.rowcount or 0) <= 0:
                    raise HTTPException(status_code=404, detail="record not found (or already deleted)")

        return {
            "ok": True,
            "recordId": str(record_uuid),
            "softDeleted": True,
            "oldReceiptPath": old_path,
            "storageDeletion": "queued_by_trigger",
        }

    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Record -> hospital candidates + confirm
# =========================================================
@app.get("/api/db/records/hospital-candidates")
def api_record_hospital_candidates(
    recordId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = (user.get("uid") or "").strip()
    record_uuid = _uuid_or_400(recordId, "recordId")

    rows = db_fetchall(
        """
        SELECT
            c.rank,
            c.score,
            h.hospital_mgmt_no,
            h.name,
            h.road_address,
            h.jibun_address,
            h.lat,
            h.lng,
            h.is_custom_entry
        FROM public.health_record_hospital_candidates c
        JOIN public.hospitals h
          ON h.hospital_mgmt_no = c.hospital_mgmt_no
        JOIN public.health_records r
          ON r.id = c.record_id
        JOIN public.pets p
          ON p.id = r.pet_id
        WHERE
            p.user_uid = %s
            AND r.id = %s
            AND r.deleted_at IS NULL
        ORDER BY c.rank ASC
        """,
        (uid, record_uuid),
    )

    out = []
    for r in rows:
        out.append(
            {
                "rank": int(r["rank"]),
                "score": float(r["score"]) if r.get("score") is not None else None,
                "hospitalMgmtNo": r["hospital_mgmt_no"],
                "name": r["name"],
                "roadAddress": r.get("road_address"),
                "jibunAddress": r.get("jibun_address"),
                "lat": r.get("lat"),
                "lng": r.get("lng"),
                "isCustomEntry": bool(r.get("is_custom_entry")),
            }
        )
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

    rec = db_fetchone(
        """
        SELECT r.id
        FROM public.health_records r
        JOIN public.pets p ON p.id = r.pet_id
        WHERE p.user_uid = %s AND r.id = %s AND r.deleted_at IS NULL
        """,
        (uid, record_uuid),
    )
    if not rec:
        raise HTTPException(status_code=404, detail="record not found")

    hosp = db_fetchone(
        """
        SELECT hospital_mgmt_no, is_custom_entry, created_by_uid
        FROM public.hospitals
        WHERE hospital_mgmt_no=%s
        """,
        (mgmt,),
    )
    if not hosp:
        raise HTTPException(status_code=400, detail="Invalid hospitalMgmtNo")
    if hosp.get("is_custom_entry") and (hosp.get("created_by_uid") or "") != uid:
        raise HTTPException(status_code=403, detail="custom hospital belongs to another user")

    try:
        row = db_fetchone(
            """
            UPDATE public.health_records r
            SET hospital_mgmt_no = %s
            WHERE r.id = %s
              AND r.deleted_at IS NULL
              AND EXISTS (
                SELECT 1
                FROM public.pets p
                WHERE p.id = r.pet_id AND p.user_uid = %s
              )
            RETURNING
                r.id, r.pet_id, r.hospital_mgmt_no, r.hospital_name, r.visit_date, r.total_amount,
                r.pet_weight_at_visit, r.tags,
                r.receipt_image_path, r.file_size_bytes,
                r.created_at, r.updated_at
            """,
            (mgmt, record_uuid, uid),
        )
        if not row:
            raise HTTPException(status_code=500, detail=_internal_detail("failed to confirm hospital", kind="DB error"))

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
        row["hospitalCandidatesCleared"] = True
        return jsonable_encoder(row)
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.post("/api/receipts/process")
def api_receipts_process(
    petId: str = Form(...),
    recordId: Optional[str] = Form(None),
    hospitalMgmtNo: Optional[str] = Form(None),
    replaceItems: bool = Form(True),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
    # ---- local helpers (copy/paste safe) ----
    _AMOUNT_RE = re.compile(r"([0-9][0-9,]*)")
    _MONEY_TOKEN_RE = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,}")  # 30,000 or 30000+
    _DATE_RE = re.compile(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b")

    _NOISE_TOKENS = [
        "고객", "고객번호", "고객 번호",
        "발행", "발행일", "발행 일",
        "사업자", "사업자등록", "대표", "전화", "주소",
        "serial", "sign", "승인", "카드", "현금",
        "부가세", "vat", "면세", "과세", "공급가",
        "소계", "합계", "총액", "총 금액", "총금액", "청구", "결제",
    ]

    def _coerce_amount_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):  # bool is subclass of int
            return None
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            try:
                return int(v)
            except Exception:
                return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            m = _AMOUNT_RE.search(s.replace(" ", ""))
            if not m:
                return None
            num = m.group(1).replace(",", "")
            try:
                return int(num)
            except Exception:
                return None
        return None

    def _norm_key(s: str) -> str:
        return re.sub(r"[^0-9a-zA-Z가-힣]", "", (s or "").lower())

    def _is_noise_line(name: str) -> bool:
        n = (name or "").strip()
        if not n:
            return True
        low = n.lower()

        # 날짜만 있는 라인 등 제거 (OCR 전체 텍스트 fallback에서 중요)
        if _DATE_RE.search(n):
            # "날짜: 2025-11-28" 같은 건 토큰으로 걸러지지만
            # "2025-11-28" 단독 라인 같은 것 방지
            if len(_norm_key(n)) <= 10:
                return True

        # 너무 짧은 단어는 제외 (오탐 방지)
        if len(_norm_key(n)) < 2:
            return True

        for t in _NOISE_TOKENS:
            if t in n or t in low:
                return True

        return False

    def _extract_item_name(it: Dict[str, Any]) -> str:
        for k in ("itemName", "item_name", "name", "desc", "description", "label", "text"):
            v = it.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _clean_item_name_from_line(line: str) -> str:
        s = (line or "").strip()
        if not s:
            return ""
        # 앞쪽 불릿/별 제거
        s = re.sub(r"^[\*\-\•\·\+]+", "", s).strip()

        # "Rabbies 30,000 1 30,000" 같은 테이블 로우는
        # 첫 '금액처럼 보이는' 토큰 앞까지만 이름으로 사용
        m = _MONEY_TOKEN_RE.search(s)
        if m:
            left = s[:m.start()].strip()
            if len(_norm_key(left)) >= 2:
                s = left

        return s.strip()

    def _guess_price_from_text(line: str) -> Optional[int]:
        s = (line or "").strip()
        if not s:
            return None

        # 1) "30,000원" 우선
        m_won = re.findall(r"([0-9][0-9,]*)\s*원", s)
        cand = []
        for x in m_won:
            n = _coerce_amount_int(x)
            if isinstance(n, int) and n >= 100:
                cand.append(n)
        if cand:
            return max(cand)

        # 2) 그 외 숫자들 중 '금액처럼 보이는' 최대값
        m_all = _AMOUNT_RE.findall(s)
        nums: List[int] = []
        for x in m_all:
            n = _coerce_amount_int(x)
            if isinstance(n, int) and n >= 100:
                nums.append(n)
        if not nums:
            return None
        return max(nums)

    def _parse_date_flex(s: str) -> Optional[date]:
        ss = (s or "").strip()
        if not ss:
            return None
        for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%y-%m-%d", "%y.%m.%d", "%y/%m/%d"):
            try:
                return datetime.strptime(ss, fmt).date()
            except Exception:
                pass
        return None

    def _pick_first(d: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            v = d.get(k)
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            return v
        return None

    # ---- original logic ----
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="file/image is required")

    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)

    pet_uuid = _uuid_or_400(petId, "petId")
    record_uuid = _uuid_or_new(recordId, "recordId")

    pet = db_fetchone("SELECT id, weight_kg FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    pet_weight_kg: Optional[float] = None
    try:
        wkg = pet.get("weight_kg") if isinstance(pet, dict) else None
        pet_weight_kg = float(wkg) if wkg is not None else None
    except Exception:
        pet_weight_kg = None

    raw = _read_upload_limited(upload, int(settings.MAX_RECEIPT_IMAGE_BYTES))
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    _require_module(ocr_policy, "ocr_policy")
    try:
        webp_bytes, parsed, hints = ocr_policy.process_receipt(
    raw,
    google_credentials=settings.GOOGLE_APPLICATION_CREDENTIALS,
    ocr_timeout_seconds=settings.OCR_TIMEOUT_SECONDS,
    ocr_max_concurrency=settings.OCR_MAX_CONCURRENCY,
    ocr_sema_acquire_timeout_seconds=settings.OCR_SEMA_ACQUIRE_TIMEOUT_SECONDS,
    receipt_max_width=settings.RECEIPT_MAX_WIDTH,
    receipt_webp_quality=settings.RECEIPT_WEBP_QUALITY,
    image_max_pixels=settings.IMAGE_MAX_PIXELS,

    # ✅ Gemini 환경변수 기반 강제 주입
    gemini_enabled=settings.GEMINI_ENABLED,
    gemini_api_key=settings.GEMINI_API_KEY,
    gemini_model_name=settings.GEMINI_MODEL_NAME,
    gemini_timeout_seconds=settings.GEMINI_TIMEOUT_SECONDS,
)




    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Receipt processing failed"))

    receipt_path = _receipt_path(uid, str(pet_uuid), str(record_uuid))
    try:
        upload_bytes_to_storage(receipt_path, webp_bytes, "image/webp")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))

    file_size_bytes = int(len(webp_bytes))

    # ✅ parsed 키/중첩 구조가 달라도 최대한 살려서 읽기
    visit_date_raw = None
    hospital_name_raw = None
    total_amount_raw = None
    items_raw = None
    ocr_text = None

    roots: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        roots.append(parsed)
        for k in ("parsed", "receipt", "data", "result"):
            v = parsed.get(k)
            if isinstance(v, dict):
                roots.append(v)

    for r in roots:
        if visit_date_raw is None:
            visit_date_raw = _pick_first(r, ["visitDate", "visit_date", "date", "visitedAt"])
        if hospital_name_raw is None:
            hospital_name_raw = _pick_first(r, ["hospitalName", "hospital_name", "hospital", "clinicName"])
        if total_amount_raw is None:
            total_amount_raw = _pick_first(r, ["totalAmount", "total_amount", "amountTotal", "grandTotal", "total"])
        if items_raw is None:
            items_raw = _pick_first(r, ["items", "lineItems", "line_items", "rows", "rowItems", "details"])
        if ocr_text is None:
            ocr_text = _pick_first(r, ["text", "fullText", "rawText", "ocrText", "redactedText", "receiptText"])

    extracted_items: List[Any] = items_raw if isinstance(items_raw, list) else []

    # ✅ items가 없으면 lines류라도 잡아보기
    if not extracted_items:
        for r in roots:
            lines_raw = _pick_first(r, ["lines", "textLines", "ocrLines", "rawLines", "lineTexts"])
            if isinstance(lines_raw, list) and lines_raw:
                extracted_items = lines_raw
                break

    # ✅ 그것도 없으면 OCR 전체 텍스트를 라인으로 쪼개서 후보로 사용
    if (not extracted_items) and isinstance(ocr_text, str) and ocr_text.strip():
        extracted_items = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]

    vd_from_ocr: Optional[date] = None
    if isinstance(visit_date_raw, str):
        vd_from_ocr = _parse_date_flex(visit_date_raw)

    hn = hospital_name_raw.strip() if isinstance(hospital_name_raw, str) and hospital_name_raw.strip() else None

    ta_from_ocr: Optional[int] = _coerce_amount_int(total_amount_raw)
    if ta_from_ocr is not None and ta_from_ocr <= 0:
        ta_from_ocr = None

    mgmt_input = hospitalMgmtNo.strip() if isinstance(hospitalMgmtNo, str) and hospitalMgmtNo.strip() else None

    # ✅ normalize items: dict + str 둘 다 지원
    safe_items: List[Dict[str, Any]] = []
    for it in extracted_items[:250]:
        nm = ""
        raw_price = None
        ct = None

        if isinstance(it, dict):
            nm = _extract_item_name(it)
            raw_price = it.get("price")
            if raw_price is None:
                raw_price = it.get("amount") or it.get("value") or it.get("total")
            ct = it.get("categoryTag")
            if ct is None:
                ct = it.get("category_tag")

        elif isinstance(it, str):
            nm = it.strip()

        else:
            continue

        if not nm:
            continue

        # str 라인은 "Rabbies 30,000 1 30,000" 같은 경우가 많으니 이름만 정리
        nm_clean = _clean_item_name_from_line(nm)
        if not nm_clean:
            continue

        if _is_noise_line(nm_clean):
            continue

        pr = _coerce_amount_int(raw_price)
        if pr is None:
            # 라인에서 직접 가격 추측
            pr = _guess_price_from_text(nm)

        if pr is not None and pr < 0:
            pr = None

        # 너무 작은 금액 제거 (9원/58원 등 OCR 쓰레기)
        if pr is not None and pr < 100:
            continue

        safe_items.append({"itemName": nm_clean[:200], "price": pr, "categoryTag": ct})

    # 3) resolve record tags
    _require_module(tag_policy, "tag_policy")
    try:
        # ✅ ocr_text도 넘겨두면(tag_policy가 원하면 사용) items가 빈 케이스에 도움됨
        tag_result = tag_policy.resolve_record_tags(  # type: ignore
            items=safe_items,
            hospital_name=hn,
            ocr_text=(ocr_text[:4000] if isinstance(ocr_text, str) else None),
        )
        resolved_tags = _clean_tags(tag_result.get("tags") if isinstance(tag_result, dict) else [])
        tag_evidence = tag_result.get("evidence") if isinstance(tag_result, dict) else None
    except Exception as e:
        resolved_tags = []
        tag_evidence = None
        logger.warning("[TagPolicy] resolve failed (ignored): %s", _sanitize_for_log(repr(e)))

    # ✅ tag_policy의 itemCategoryTags를 safe_items.categoryTag에 반영 (DB 저장용)
    item_tag_map: Dict[str, str] = {}
    try:
        if isinstance(tag_result, dict):
            rows = tag_result.get("itemCategoryTags")
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    nm2 = (r.get("itemName") or "").strip()
                    ct2 = (r.get("categoryTag") or "").strip()
                    if nm2 and ct2:
                        item_tag_map[_norm_key(nm2)] = ct2
    except Exception:
        item_tag_map = {}

    for it in safe_items:
        if not it.get("categoryTag"):
            ct2 = item_tag_map.get(_norm_key(it.get("itemName") or ""))
            if ct2:
                it["categoryTag"] = ct2

    # ✅ 가격이 하나도 없으면: (1) 항목 1개면 그 항목에 total을 넣고, (2) 항목이 없으면 fallback 추가
    has_any_price = any(isinstance(x.get("price"), int) and int(x.get("price")) > 0 for x in safe_items)
    if (not has_any_price) and (ta_from_ocr is not None) and (ta_from_ocr >= 100):
        if len(safe_items) == 1 and safe_items[0].get("price") is None:
            safe_items[0]["price"] = int(ta_from_ocr)
        elif len(safe_items) == 0:
            safe_items.append(
                {
                    "itemName": "진료비",
                    "price": int(ta_from_ocr),
                    "categoryTag": (resolved_tags[0] if resolved_tags else "checkup_general"),
                }
            )

    # hints
    addr_hint = hints.get("addressHint") if isinstance(hints, dict) else None
    addr_hint = addr_hint if isinstance(addr_hint, str) else None

    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # mgmt_no scope check
                if mgmt_input is not None:
                    cur.execute(
                        """
                        SELECT hospital_mgmt_no, is_custom_entry, created_by_uid
                        FROM public.hospitals
                        WHERE hospital_mgmt_no=%s
                        """,
                        (mgmt_input,),
                    )
                    hosp = cur.fetchone()
                    if not hosp:
                        raise HTTPException(status_code=400, detail="Invalid hospitalMgmtNo")
                    if hosp.get("is_custom_entry") and (hosp.get("created_by_uid") or "") != uid:
                        raise HTTPException(status_code=403, detail="custom hospital belongs to another user")

                # block attaching to deleted record (no undelete)
                cur.execute(
                    """
                    SELECT r.visit_date, r.total_amount, r.pet_weight_at_visit, r.tags, r.deleted_at
                    FROM public.health_records r
                    JOIN public.pets p ON p.id = r.pet_id
                    WHERE p.user_uid = %s AND r.id = %s
                    """,
                    (uid, record_uuid),
                )
                existing = cur.fetchone() or {}
                if existing.get("deleted_at") is not None:
                    raise HTTPException(status_code=409, detail="record is deleted (cannot attach receipt)")

                existing_total = existing.get("total_amount")
                existing_visit = existing.get("visit_date")
                existing_weight = existing.get("pet_weight_at_visit")
                existing_tags = existing.get("tags") if isinstance(existing.get("tags"), list) else []

                items_sum = 0
                for it in safe_items:
                    pr = it.get("price")
                    if isinstance(pr, int) and pr > 0:
                        items_sum += pr

                ta_candidate = int(ta_from_ocr) if ta_from_ocr is not None and int(ta_from_ocr) > 0 else (items_sum if items_sum > 0 else 0)

                vd_final = existing_visit or vd_from_ocr or date.today()
                ta_final = int(existing_total) if existing_total is not None and int(existing_total) > 0 else int(ta_candidate)
                w_final = existing_weight if existing_weight is not None else pet_weight_kg

                tags_final = existing_tags if len(existing_tags) > 0 else resolved_tags

                cur.execute(
                    """
                    INSERT INTO public.health_records
                      (id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags,
                       receipt_image_path, file_size_bytes)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s, %s,
                       %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                      pet_id = EXCLUDED.pet_id,
                      hospital_mgmt_no = COALESCE(EXCLUDED.hospital_mgmt_no, public.health_records.hospital_mgmt_no),
                      hospital_name = COALESCE(public.health_records.hospital_name, EXCLUDED.hospital_name),
                      pet_weight_at_visit = COALESCE(public.health_records.pet_weight_at_visit, EXCLUDED.pet_weight_at_visit),
                      visit_date = COALESCE(public.health_records.visit_date, EXCLUDED.visit_date),
                      total_amount = CASE
                        WHEN COALESCE(public.health_records.total_amount, 0) > 0 THEN public.health_records.total_amount
                        ELSE EXCLUDED.total_amount
                      END,
                      tags = CASE
                        WHEN public.health_records.tags IS NOT NULL AND cardinality(public.health_records.tags) > 0
                          THEN public.health_records.tags
                        ELSE EXCLUDED.tags
                      END,
                      receipt_image_path = EXCLUDED.receipt_image_path,
                      file_size_bytes = EXCLUDED.file_size_bytes
                    WHERE
                      public.health_records.deleted_at IS NULL
                      AND EXISTS (
                        SELECT 1 FROM public.pets p
                        WHERE p.id = EXCLUDED.pet_id AND p.user_uid = %s
                      )
                    RETURNING
                      id, pet_id, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags,
                      receipt_image_path, file_size_bytes,
                      created_at, updated_at
                    """,
                    (record_uuid, pet_uuid, mgmt_input, hn, vd_final, ta_final, w_final, tags_final, receipt_path, file_size_bytes, uid),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=500, detail=_internal_detail("Failed to upsert health_record", kind="DB error"))

                if replaceItems:
                    cur.execute("DELETE FROM public.health_items WHERE record_id=%s", (record_uuid,))
                    for it in safe_items:
                        cur.execute(
                            """
                            INSERT INTO public.health_items (record_id, item_name, price, category_tag)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (record_uuid, it["itemName"], it.get("price"), it.get("categoryTag")),
                        )

                # hospital candidates (best-effort; never block receipt save)
                candidates_out: List[Dict[str, Any]] = []
                record_confirmed = row.get("hospital_mgmt_no") is not None
                if not record_confirmed:
                    limit_n = int(settings.OCR_HOSPITAL_CANDIDATE_LIMIT or 3)
                    limit_n = max(1, min(limit_n, 10))

                    name_for_match = hn or ""
                    addr_for_match = addr_hint or ""
                    if (name_for_match.strip() != "") or (addr_for_match.strip() != ""):
                        cur.execute("SAVEPOINT cand_sp")
                        try:
                            cur.execute(
                                "DELETE FROM public.health_record_hospital_candidates WHERE record_id=%s",
                                (record_uuid,),
                            )
                            cur.execute(
                                """
                                SELECT
                                    c.hospital_mgmt_no,
                                    c.name,
                                    c.road_address,
                                    c.jibun_address,
                                    c.lat,
                                    c.lng,
                                    c.score,
                                    h.is_custom_entry
                                FROM public.find_hospital_candidates_weighted(%s, %s, %s, %s) c
                                JOIN public.hospitals h
                                  ON h.hospital_mgmt_no = c.hospital_mgmt_no
                                """,
                                (uid, name_for_match, addr_for_match, limit_n),
                            )
                            cand_rows = cur.fetchall() or []

                            for idx, c in enumerate(cand_rows, start=1):
                                cur.execute(
                                    """
                                    INSERT INTO public.health_record_hospital_candidates
                                        (record_id, hospital_mgmt_no, rank, score)
                                    VALUES
                                        (%s, %s, %s, %s)
                                    ON CONFLICT (record_id, hospital_mgmt_no) DO UPDATE SET
                                        rank = EXCLUDED.rank,
                                        score = EXCLUDED.score
                                    """,
                                    (record_uuid, c["hospital_mgmt_no"], idx, c.get("score")),
                                )

                                candidates_out.append(
                                    {
                                        "rank": idx,
                                        "score": float(c["score"]) if c.get("score") is not None else None,
                                        "hospitalMgmtNo": c["hospital_mgmt_no"],
                                        "name": c["name"],
                                        "roadAddress": c.get("road_address"),
                                        "jibunAddress": c.get("jibun_address"),
                                        "lat": c.get("lat"),
                                        "lng": c.get("lng"),
                                        "isCustomEntry": bool(c.get("is_custom_entry")),
                                    }
                                )

                            cur.execute("RELEASE SAVEPOINT cand_sp")
                        except Exception as e:
                            try:
                                cur.execute("ROLLBACK TO SAVEPOINT cand_sp")
                                cur.execute("RELEASE SAVEPOINT cand_sp")
                            except Exception:
                                pass
                            candidates_out = []
                            logger.warning("[Candidates] generation failed (ignored): %s", _sanitize_for_log(_pg_message(e)))

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
                payload["hospitalCandidates"] = candidates_out
                payload["hospitalCandidateCount"] = len(candidates_out)
                payload["hospitalConfirmed"] = bool(payload.get("hospital_mgmt_no"))
                payload["tagEvidence"] = tag_evidence
                return jsonable_encoder(payload)

    except HTTPException as he:
        try:
            delete_storage_object_if_exists(receipt_path)
        except Exception:
            pass
        raise he

    except Exception as e:
        try:
            delete_storage_object_if_exists(receipt_path)
        except Exception:
            pass
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Documents (PDF)
# =========================================================
def _is_pdf_bytes(data: bytes) -> bool:
    return bool(data) and data[:5] == b"%PDF-"


@app.post("/api/docs/upload-pdf", response_model=DocumentUploadResponse)
def upload_pdf_document(
    petId: str = Form(...),
    docType: str = Form(...),  # lab or cert
    displayName: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    desired = _infer_membership_tier_from_token(user)
    db_touch_user(uid, desired_tier=desired)

    pet_uuid = _uuid_or_400(petId, "petId")

    dt = (docType or "").strip().lower()
    if dt not in ("lab", "cert"):
        raise HTTPException(status_code=400, detail="docType must be 'lab' or 'cert'")

    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    data = _read_upload_limited(file, int(settings.MAX_PDF_BYTES))
    if not data:
        raise HTTPException(status_code=400, detail="empty pdf")
    if not _is_pdf_bytes(data):
        raise HTTPException(status_code=400, detail="file is not a valid PDF")

    doc_uuid = uuid.uuid4()
    name = (displayName or "").strip()
    if not name:
        name = (file.filename or "").strip() or ("lab.pdf" if dt == "lab" else "cert.pdf")

    file_path = _doc_pdf_path(uid, str(pet_uuid), dt, str(doc_uuid))
    file_size_bytes = int(len(data))

    try:
        upload_bytes_to_storage(file_path, data, "application/pdf")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))

    try:
        row = db_fetchone(
            """
            INSERT INTO public.pet_documents
                (id, pet_id, doc_type, display_name, file_path, file_size_bytes)
            VALUES
                (%s, %s, %s, %s, %s, %s)
            RETURNING
                id, pet_id, doc_type, display_name, file_path, file_size_bytes,
                created_at, updated_at
            """,
            (doc_uuid, pet_uuid, dt, name, file_path, file_size_bytes),
        )
        if not row:
            raise HTTPException(status_code=500, detail=_internal_detail("failed to insert pet_document", kind="DB error"))

        return {
            "id": str(row["id"]),
            "petId": str(row["pet_id"]),
            "docType": row["doc_type"],
            "displayName": row["display_name"],
            "filePath": row["file_path"],
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


@app.get("/api/docs/list")
def list_pdf_documents(
    petId: str = Query(...),
    docType: Optional[str] = Query(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = (user.get("uid") or "").strip()

    try:
        pet_uuid = _uuid_or_400(petId, "petId")

        dt = (docType or "").strip().lower() if docType else None
        if dt is not None:
            alias = {"vaccine": "cert", "vacc": "cert", "vaccination": "cert"}
            dt = alias.get(dt, dt)

        if dt is not None and dt not in ("lab", "cert"):
            raise HTTPException(status_code=400, detail="docType must be 'lab' or 'cert'")

        if dt:
            rows = db_fetchall(
                """
                SELECT d.*
                FROM public.pet_documents d
                JOIN public.pets p ON p.id = d.pet_id
                WHERE
                    p.user_uid=%s AND p.id=%s AND d.doc_type=%s
                    AND d.deleted_at IS NULL
                ORDER BY d.created_at DESC
                """,
                (uid, pet_uuid, dt),
            )
        else:
            rows = db_fetchall(
                """
                SELECT d.*
                FROM public.pet_documents d
                JOIN public.pets p ON p.id = d.pet_id
                WHERE
                    p.user_uid=%s AND p.id=%s
                    AND d.deleted_at IS NULL
                ORDER BY d.created_at DESC
                """,
                (uid, pet_uuid),
            )

        out = []
        for r in rows:
            out.append(
                {
                    "id": str(r["id"]),
                    "petId": str(r["pet_id"]),
                    "docType": r["doc_type"],
                    "displayName": r["display_name"],
                    "filePath": r["file_path"],
                    "fileSizeBytes": int(r["file_size_bytes"]),
                    "createdAt": r["created_at"].isoformat() if r.get("created_at") else None,
                    "updatedAt": r["updated_at"].isoformat() if r.get("updated_at") else None,
                }
            )
        return out

    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


@app.delete("/api/docs/delete")
def delete_pdf_document(
    docId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    ✅ Soft delete only.
    - sets deleted_at=now()
    - trigger enqueues storage delete job using OLD path
    - trigger sets bytes=0 and path NULL per schema policy
    """
    uid = (user.get("uid") or "").strip()
    doc_uuid = _uuid_or_400(docId, "docId")

    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT d.id, d.file_path
                    FROM public.pet_documents d
                    JOIN public.pets p ON p.id = d.pet_id
                    WHERE p.user_uid=%s AND d.id=%s AND d.deleted_at IS NULL
                    """,
                    (uid, doc_uuid),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="document not found (or already deleted)")

                old_path = row.get("file_path")

                cur.execute(
                    """
                    UPDATE public.pet_documents d
                    SET deleted_at = now()
                    WHERE d.id=%s
                      AND d.deleted_at IS NULL
                      AND EXISTS (
                        SELECT 1 FROM public.pets p
                        WHERE p.id = d.pet_id AND p.user_uid = %s
                      )
                    """,
                    (doc_uuid, uid),
                )
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
def upload_lab_pdf_compat(
    petId: str = Form(...),
    displayName: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    return upload_pdf_document(petId=petId, docType="lab", displayName=displayName, file=file, user=user)


@app.post("/api/cert/upload-pdf", response_model=DocumentUploadResponse)
def upload_cert_pdf_compat(
    petId: str = Form(...),
    displayName: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    return upload_pdf_document(petId=petId, docType="cert", displayName=displayName, file=file, user=user)


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
    # backups: allow without DB rows (not counted in quota by DB triggers)
    if path.startswith(f"users/{uid}/backups/") and path.endswith(".json"):
        return {"kind": "backup"}

    rec = db_fetchone(
        """
        SELECT 1
        FROM public.health_records r
        JOIN public.pets p ON p.id = r.pet_id
        WHERE
            p.user_uid=%s
            AND r.deleted_at IS NULL
            AND r.receipt_image_path=%s
        LIMIT 1
        """,
        (uid, path),
    )
    if rec:
        return {"kind": "receipt"}

    doc = db_fetchone(
        """
        SELECT 1
        FROM public.pet_documents d
        JOIN public.pets p ON p.id = d.pet_id
        WHERE
            p.user_uid=%s
            AND d.deleted_at IS NULL
            AND d.file_path=%s
        LIMIT 1
        """,
        (uid, path),
    )
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
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=ttl_seconds),
            method="GET",
            response_disposition=response_disposition,
        )
        return url, expires_at
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Signed URL error"))


@app.get("/api/storage/signed-url", response_model=SignedUrlResponse)
def storage_signed_url(
    path: str = Query(...),
    ttl: int = Query(None, ge=60),
    filename: Optional[str] = Query(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
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
# Backup endpoints (Snapshot storage)
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
        out.append(
            {
                "backupId": bid,
                "objectPath": blob.name,
                "lastModified": blob.updated.isoformat() if getattr(blob, "updated", None) else None,
                "size": int(getattr(blob, "size", 0) or 0),
            }
        )
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
# Migration tokens + migration execution (snapshot fallback)
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
        db_execute(
            """
            INSERT INTO public.migration_tokens (old_uid, code_hash, expires_at, status)
            VALUES (%s, %s, %s, 'issued')
            """,
            (old_uid, code_hash, expires_at),
        )
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

    token_row = db_fetchone(
        """
        SELECT *
        FROM public.migration_tokens
        WHERE code_hash=%s
        """,
        (code_hash,),
    )
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
        db_execute(
            """
            UPDATE public.migration_tokens
            SET status='completed', used_at=now(), new_uid=%s
            WHERE code_hash=%s
            """,
            (new_uid, code_hash),
        )
        return {"ok": True, "oldUid": old_uid, "newUid": new_uid, "copied": 0, "deleted": 0, "dbUpdated": False, "warnings": ["oldUid == newUid (no-op)"]}

    db_execute(
        """
        UPDATE public.migration_tokens
        SET status='processing', new_uid=%s
        WHERE code_hash=%s
        """,
        (new_uid, code_hash),
    )

    try:
        copied = _copy_prefix(old_uid, new_uid)
    except Exception as e:
        db_execute("UPDATE public.migration_tokens SET status='failed' WHERE code_hash=%s", (code_hash,))
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage copy failed"))

    # DB-side migration: should handle rollups + path rewrite + job merge safely
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

    db_execute(
        """
        UPDATE public.migration_tokens
        SET status='completed', used_at=now()
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
        "steps": [dict(s) for s in steps],
        "warnings": warnings,
    }


# =========================================================
# OCR ItemName Map APIs
# =========================================================
@app.get("/api/ocr/item-maps/list")
def list_item_maps(
    includeInactive: bool = Query(False),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = (user.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="missing uid")

    rows = db_fetchall(
        """
        SELECT
            id,
            ocr_item_name,
            canonical_name,
            is_active,
            is_custom_entry,
            created_by_uid,
            created_at,
            updated_at
        FROM public.ocr_item_name_maps
        WHERE
            (is_custom_entry = false OR created_by_uid = %s)
            AND (%s OR is_active = true)
        ORDER BY
            is_custom_entry DESC,
            created_at DESC
        """,
        (uid, includeInactive),
    )

    out = []
    for r in rows:
        out.append(
            {
                "id": str(r["id"]),
                "ocrItemName": r["ocr_item_name"],
                "canonicalName": r["canonical_name"],
                "isActive": bool(r["is_active"]),
                "isCustomEntry": bool(r["is_custom_entry"]),
                "createdAt": r["created_at"].isoformat() if r.get("created_at") else None,
                "updatedAt": r["updated_at"].isoformat() if r.get("updated_at") else None,
            }
        )
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
                cur.execute(
                    """
                    UPDATE public.ocr_item_name_maps
                    SET
                        canonical_name = %s,
                        is_active = %s
                    WHERE
                        is_custom_entry = true
                        AND created_by_uid = %s
                        AND lower(ocr_item_name) = lower(%s)
                        AND is_active = true
                    RETURNING
                        id, ocr_item_name, canonical_name, is_active, is_custom_entry, created_at, updated_at
                    """,
                    (canonical, is_active, uid, ocr_name),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "id": str(row["id"]),
                        "ocrItemName": row["ocr_item_name"],
                        "canonicalName": row["canonical_name"],
                        "isActive": bool(row["is_active"]),
                        "isCustomEntry": True,
                        "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
                        "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
                    }

                cur.execute(
                    """
                    SELECT id
                    FROM public.ocr_item_name_maps
                    WHERE
                        is_custom_entry = true
                        AND created_by_uid = %s
                        AND lower(ocr_item_name) = lower(%s)
                        AND is_active = false
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (uid, ocr_name),
                )
                inactive = cur.fetchone()
                if inactive:
                    cur.execute(
                        """
                        UPDATE public.ocr_item_name_maps
                        SET
                            canonical_name = %s,
                            is_active = %s
                        WHERE id = %s
                        RETURNING
                            id, ocr_item_name, canonical_name, is_active, is_custom_entry, created_at, updated_at
                        """,
                        (canonical, is_active, inactive["id"]),
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            "id": str(row["id"]),
                            "ocrItemName": row["ocr_item_name"],
                            "canonicalName": row["canonical_name"],
                            "isActive": bool(row["is_active"]),
                            "isCustomEntry": True,
                            "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
                            "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
                        }

                cur.execute(
                    """
                    INSERT INTO public.ocr_item_name_maps
                        (ocr_item_name, canonical_name, is_active, is_custom_entry, created_by_uid)
                    VALUES
                        (%s, %s, %s, true, %s)
                    RETURNING
                        id, ocr_item_name, canonical_name, is_active, is_custom_entry, created_at, updated_at
                    """,
                    (ocr_name, canonical, is_active, uid),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=500, detail=_internal_detail("failed to upsert item map", kind="DB error"))

                return {
                    "id": str(row["id"]),
                    "ocrItemName": row["ocr_item_name"],
                    "canonicalName": row["canonical_name"],
                    "isActive": bool(row["is_active"]),
                    "isCustomEntry": True,
                    "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
                    "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
                }

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
        row = db_fetchone(
            """
            UPDATE public.ocr_item_name_maps
            SET is_active = false
            WHERE
                is_custom_entry = true
                AND created_by_uid = %s
                AND lower(ocr_item_name) = lower(%s)
                AND is_active = true
            RETURNING id, ocr_item_name, canonical_name, is_active, created_at, updated_at
            """,
            (uid, ocr_name),
        )
        if not row:
            raise HTTPException(status_code=404, detail="active mapping not found")
        return {
            "ok": True,
            "id": str(row["id"]),
            "ocrItemName": row["ocr_item_name"],
            "canonicalName": row["canonical_name"],
            "isActive": bool(row["is_active"]),
            "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }
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
                cur.execute(
                    """
                    UPDATE public.ocr_item_name_maps
                    SET canonical_name=%s, is_active=%s
                    WHERE
                        is_custom_entry=false
                        AND is_active=true
                        AND lower(ocr_item_name)=lower(%s)
                    RETURNING id, ocr_item_name, canonical_name, is_active, created_at, updated_at
                    """,
                    (canonical, is_active, ocr_name),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "id": str(row["id"]),
                        "ocrItemName": row["ocr_item_name"],
                        "canonicalName": row["canonical_name"],
                        "isActive": bool(row["is_active"]),
                        "isCustomEntry": False,
                        "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
                    }

                cur.execute(
                    """
                    SELECT id
                    FROM public.ocr_item_name_maps
                    WHERE
                        is_custom_entry=false
                        AND is_active=false
                        AND lower(ocr_item_name)=lower(%s)
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (ocr_name,),
                )
                inactive = cur.fetchone()
                if inactive:
                    cur.execute(
                        """
                        UPDATE public.ocr_item_name_maps
                        SET canonical_name=%s, is_active=%s
                        WHERE id=%s
                        RETURNING id, ocr_item_name, canonical_name, is_active, created_at, updated_at
                        """,
                        (canonical, is_active, inactive["id"]),
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            "id": str(row["id"]),
                            "ocrItemName": row["ocr_item_name"],
                            "canonicalName": row["canonical_name"],
                            "isActive": bool(row["is_active"]),
                            "isCustomEntry": False,
                            "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
                        }

                cur.execute(
                    """
                    INSERT INTO public.ocr_item_name_maps
                        (ocr_item_name, canonical_name, is_active, is_custom_entry, created_by_uid)
                    VALUES
                        (%s, %s, %s, false, NULL)
                    RETURNING id, ocr_item_name, canonical_name, is_active, created_at, updated_at
                    """,
                    (ocr_name, canonical, is_active),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=500, detail=_internal_detail("failed to upsert global item map", kind="DB error"))
                return {
                    "id": str(row["id"]),
                    "ocrItemName": row["ocr_item_name"],
                    "canonicalName": row["canonical_name"],
                    "isActive": bool(row["is_active"]),
                    "isCustomEntry": False,
                    "updatedAt": row["updated_at"].isoformat() if row.get("updated_at") else None,
                }
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
    # total (including deleted)
    users_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.users") or {"c": 0}
    pets_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.pets") or {"c": 0}
    records_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.health_records") or {"c": 0}
    docs_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.pet_documents") or {"c": 0}

    # active (deleted_at IS NULL)
    active_records = db_fetchone("SELECT COUNT(*)::int AS c FROM public.health_records WHERE deleted_at IS NULL") or {"c": 0}
    active_docs = db_fetchone("SELECT COUNT(*)::int AS c FROM public.pet_documents WHERE deleted_at IS NULL") or {"c": 0}

    total_amount = db_fetchone("SELECT COALESCE(SUM(total_amount),0)::bigint AS s FROM public.health_records WHERE deleted_at IS NULL") or {"s": 0}

    return {
        "users": int(users_cnt["c"]),
        "pets": int(pets_cnt["c"]),
        "records_total": int(records_cnt["c"]),
        "records_active": int(active_records["c"]),
        "docs_total": int(docs_cnt["c"]),
        "docs_active": int(active_docs["c"]),
        "totalAmountSum_active": int(total_amount["s"]),
        "updatedAt": datetime.utcnow().isoformat() + "Z",
    }
