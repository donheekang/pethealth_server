# main.py (PetHealth+ Server)
# Firebase Storage + Receipt Redaction + Signed URL + Migration
# + Private Hospitals (road/jibun) + OCR Hospital Candidates + OCR ItemName Map
# v2.1.22-ops (refactor + hardening)

from __future__ import annotations

import os
import io
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

# Pillow (image processing)
from PIL import Image, ImageDraw, ImageFile, UnidentifiedImageError, ImageOps

# Google Vision
from google.cloud import vision

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
    # --- Google Vision ---
    GOOGLE_APPLICATION_CREDENTIALS: str = ""  # JSON string or file path

    # Vision runtime hardening
    OCR_TIMEOUT_SECONDS: int = 12
    OCR_MAX_CONCURRENCY: int = 4
    OCR_SEMA_ACQUIRE_TIMEOUT_SECONDS: float = 1.0  # OCR 폭주 시 빠른 실패

    # --- Firebase Auth / Storage ---
    AUTH_REQUIRED: bool = True
    STUB_MODE: bool = False

    FIREBASE_ADMIN_SA_JSON: str = ""  # service account JSON string
    FIREBASE_ADMIN_SA_B64: str = ""   # base64-encoded JSON string (optional)
    FIREBASE_STORAGE_BUCKET: str = ""  # e.g. "<project-id>.appspot.com"

    # --- Receipt image pipeline ---
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
    OCR_HOSPITAL_CANDIDATE_LIMIT: int = 3  # UI에 3개까지만

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

    # CORS (웹 어드민 붙일 때 사용)
    CORS_ALLOW_ORIGINS: str = "*"  # comma-separated
    CORS_ALLOW_CREDENTIALS: bool = False  # "*" 와 credentials는 같이 못 씀

    # --- Admin ---
    ADMIN_UIDS: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Pillow hardening
Image.MAX_IMAGE_PIXELS = int(settings.IMAGE_MAX_PIXELS or 20_000_000)
ImageFile.LOAD_TRUNCATED_IMAGES = False

# OCR concurrency limiter (process-wide)
_OCR_SEMA = threading.BoundedSemaphore(max(1, int(settings.OCR_MAX_CONCURRENCY or 4)))

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


def _new_error_id() -> str:
    return uuid.uuid4().hex[:12]


def _sanitize_for_log(s: str, max_len: int = 800) -> str:
    # 줄바꿈/탭 등 로그 깨짐 방지 + 너무 긴 메시지 truncate
    t = (s or "")
    t = t.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if len(t) > max_len:
        t = t[:max_len] + "...(truncated)"
    return t


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


def _internal_detail(msg: str, *, kind: str = "Internal Server Error") -> str:
    eid = _new_error_id()
    safe = _sanitize_for_log(msg)
    logger.error("[ERR:%s] %s: %s", eid, kind, safe)
    if settings.EXPOSE_ERROR_DETAILS:
        return f"{kind}: {msg} (errorId={eid})"
    return f"{kind} (errorId={eid})"


@contextmanager
def db_conn() -> Iterator[psycopg2.extensions.connection]:
    """
    ops-hardening:
      - Pool exhausted / getconn() 실패는 500이 아니라 503로 (errorId 포함)
      - getconn 실패 시 putconn/rollback 시도하지 않음 (conn=None)
      - putconn 실패는 swallow (로그만)
    """
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
                # putconn 실패는 서비스 응답에 영향 주면 더 큰 장애로 이어질 수 있어서 로그만
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

    # User-safe errors
    if "Quota exceeded" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "Ownership mismatch" in msg or "Hospital access denied" in msg:
        raise HTTPException(status_code=403, detail=msg)

    if "Candidates not allowed" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "Direct update of pets.user_uid is not allowed" in msg:
        raise HTTPException(status_code=409, detail=msg)

    if "hospitals_lat_lng_range_check" in msg:
        raise HTTPException(status_code=400, detail="Invalid latitude/longitude range")

    if "violates foreign key constraint" in msg:
        raise HTTPException(status_code=400, detail="Invalid reference (foreign key)")

    if "duplicate key value violates unique constraint" in msg:
        raise HTTPException(status_code=409, detail="Duplicate value (unique constraint)")

    # Generic (hide details in prod by default)
    raise HTTPException(status_code=500, detail=_internal_detail(msg, kind="DB error"))


# =========================================================
# Auth / membership helpers
# =========================================================
def _infer_membership_tier_from_token(decoded: Dict[str, Any]) -> Optional[str]:
    """
    guest: Firebase Anonymous Auth
    member: any non-anonymous provider (google.com, apple.com, password, etc.)
    premium: handled elsewhere (don't auto-set here)
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
      - write throttle: last_seen_at 갱신은 N초 간격으로만 (그 외는 DO NOTHING)
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
                logger.info("[DB] user_daily_active insert failed (ignored): %s", _sanitize_for_log(_pg_message(e)))

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
# Firebase Admin init (Auth/Storage decoupled)
# =========================================================
_firebase_initialized = False
auth_scheme = HTTPBearer(auto_error=False)

# Stub storage (in-memory) - thread-safe
_stub_objects: Dict[str, Dict[str, Any]] = {}  # path -> {data, content_type, updated, size}
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
        # stub
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

    # mimic google.cloud.storage.bucket.Bucket.copy_blob(blob, destination_bucket, new_name)
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
        try:
            info = json.loads(settings.FIREBASE_ADMIN_SA_JSON)
        except Exception as e:
            raise RuntimeError(f"FIREBASE_ADMIN_SA_JSON JSON parse failed: {e}")
        return _normalize_private_key_newlines(info)

    if settings.FIREBASE_ADMIN_SA_B64:
        try:
            raw = base64.b64decode(settings.FIREBASE_ADMIN_SA_B64).decode("utf-8")
            info = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"FIREBASE_ADMIN_SA_B64 decode/parse failed: {e}")
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

    # 이미 앱이 있으면 OK
    if firebase_admin._apps:
        _firebase_initialized = True
        return

    # 이전 경로에서 플래그만 True가 된 상태 방어
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

        # bucket이 없어도 Auth용으로 initialize_app 가능
        if bucket:
            firebase_admin.initialize_app(cred, {"storageBucket": bucket})
            print("[Firebase] Admin initialized (with storage bucket).")

        else:
            firebase_admin.initialize_app(cred)
            print("[Firebase] Admin initialized (no storage bucket).")

        _firebase_initialized = True

    except Exception as e:
        if require_init:
            raise RuntimeError(f"Firebase Admin initialize failed: {e}")
        print("[Firebase] init failed (ignored):", _sanitize_for_log(repr(e)))
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
    # no-auth or stub mode: still allow server to run (storage can be used separately)
    if settings.STUB_MODE or (not settings.AUTH_REQUIRED):
        return {"uid": "dev", "email": "dev@example.com", "firebase": {"sign_in_provider": "password"}}

    # auth requires firebase init
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
        # prod에서는 내부 예외를 숨기고 errorId로 통일
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
    # stub storage
    if settings.STUB_MODE:
        return _stub_bucket_singleton

    # ✅ Auth와 Storage를 분리: 앱 초기화(=Auth)는 bucket 없이도 가능
    try:
        init_firebase_admin(require_init=True)
    except Exception as e:
        raise HTTPException(status_code=503, detail=_internal_detail(str(e), kind="Firebase init error"))

    bucket = (settings.FIREBASE_STORAGE_BUCKET or "").strip()
    if not bucket:
        # Storage가 필요한 API에서만 bucket을 강제
        raise HTTPException(
            status_code=503,
            detail=_internal_detail("FIREBASE_STORAGE_BUCKET is not set", kind="Firebase init error"),
        )

    try:
        # 항상 bucket 이름을 명시해서 app options에 의존하지 않게 한다.
        return fb_storage.bucket(bucket)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))


def upload_bytes_to_storage(path: str, data: bytes, content_type: str) -> str:
    """
    Upload bytes to Firebase Storage at 'path'.
    Returns the same relative path (no URL).
    """
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
    Never store raw OCR full text in DB/logs.
    """
    client = get_vision_client()
    img = vision.Image(content=image_bytes)

    timeout_s = float(settings.OCR_TIMEOUT_SECONDS or 12)
    sema_timeout = float(settings.OCR_SEMA_ACQUIRE_TIMEOUT_SECONDS or 1.0)
    sema_timeout = max(0.1, min(sema_timeout, 5.0))

    # concurrency limit + timeout (무한대기 방지)
    acquired = _OCR_SEMA.acquire(timeout=sema_timeout)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail=_internal_detail("too many concurrent OCR requests", kind="OCR busy"),
        )

    try:
        try:
            resp = client.text_detection(image=img, timeout=timeout_s)
        except TypeError:
            # 일부 버전 호환
            resp = client.text_detection(image=img)
    finally:
        _OCR_SEMA.release()

    if resp.error.message:
        # Vision 자체 오류는 502/500 느낌이지만, 일단 500으로 분류 (errorId로 추적)
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
# Receipt parsing (best-effort)
# - 주소 힌트는 저장/응답 금지, 후보검색 보정용으로만 사용
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
        if any(x in s for x in ["TEL", "전화", "FAX", "팩스", "도로명", "주소", "지번"]):
            score -= 2
        digit_count = sum(c.isdigit() for c in s)
        if digit_count >= 8:
            score -= 1
        if len(s) < 2 or len(s) > 40:
            score -= 1
        if score > best_score:
            best_score = score
            best_line = s
    return best_line.strip()


def _guess_address_hint_from_lines(all_lines: List[str]) -> Optional[str]:
    best = None
    best_score = -999

    for idx, line in enumerate((all_lines or [])[:60]):
        t = (line or "").strip()
        if not t:
            continue
        if not _looks_like_address_line(t):
            continue

        score = 0
        if "주소" in t:
            score += 3
        if "도로명" in t:
            score += 2
        if "지번" in t:
            score += 2
        if idx <= 20:
            score += 1
        if 10 <= len(t) <= 80:
            score += 1
        if sum(c.isdigit() for c in t) >= 18:
            score -= 2

        if score > best_score:
            best_score = score
            best = t

    if not best:
        return None

    best = re.sub(r"^(주소|도로명주소|지번주소)\s*[:：]?\s*", "", best).strip()
    if not best:
        return None
    return best[:200]


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

        item_name = ln
        item_name = re.sub(re.escape(nums[-1]), "", item_name, count=1).strip()
        item_name = re.sub(r"(₩|원|KRW)", "", item_name, flags=re.IGNORECASE).strip()

        if len(item_name) < 1:
            continue

        out.append({"itemName": item_name[:200], "price": price, "categoryTag": None})
        if len(out) >= max_items:
            break
    return out


def process_receipt_image_and_parse(raw_image_bytes: bytes) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      - webp_bytes (redacted)
      - parsed (safe to return to client)
      - hints (do NOT store; used for candidate matching only)
    """
    try:
        img = Image.open(io.BytesIO(raw_image_bytes))
        img = ImageOps.exif_transpose(img)  # ✅ orientation fix
        img = img.convert("RGB")  # strip alpha/exif
    except Image.DecompressionBombError:
        raise RuntimeError("image too large (decompression bomb)")
    except UnidentifiedImageError:
        raise RuntimeError("invalid image")
    except Exception as e:
        raise RuntimeError(f"image open failed: {e}")

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

    hints = {
        "addressHint": _guess_address_hint_from_lines(line_texts),  # 저장/응답 금지
    }
    return webp, parsed, hints


# =========================================================
# OCR item_name -> canonical_name mapping (read helpers)
# =========================================================
def _fetch_item_name_map_for_lower_names(uid: str, lower_names: List[str]) -> Dict[str, str]:
    if not lower_names:
        return {}

    keys: List[str] = []
    seen = set()
    for n in lower_names:
        k = (n or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        keys.append(k)

    if not keys:
        return {}

    rows = db_fetchall(
        """
        SELECT DISTINCT ON (lower(ocr_item_name))
            lower(ocr_item_name) AS k,
            canonical_name
        FROM public.ocr_item_name_maps
        WHERE
            is_active = true
            AND lower(ocr_item_name) = ANY(%s::text[])
            AND (is_custom_entry = false OR created_by_uid = %s)
        ORDER BY
            lower(ocr_item_name),
            is_custom_entry DESC,
            created_at DESC
        """,
        (keys, uid),
    )

    out: Dict[str, str] = {}
    for r in rows:
        k = (r.get("k") or "").strip().lower()
        cn = (r.get("canonical_name") or "").strip()
        if k and cn:
            out[k] = cn
    return out


def _enrich_items_with_canonical(uid: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return items

    lower_names: List[str] = []
    for it in items:
        nm = (it.get("item_name") or it.get("itemName") or "")
        if isinstance(nm, str):
            lower_names.append(nm.strip().lower())

    mapping = _fetch_item_name_map_for_lower_names(uid, lower_names)

    out = []
    for it in items:
        d = dict(it)
        nm = (d.get("item_name") or d.get("itemName") or "")
        key = nm.strip().lower() if isinstance(nm, str) else ""
        canonical = mapping.get(key)
        d["canonicalName"] = canonical if canonical else None
        d["isMapped"] = bool(canonical)
        out.append(d)
    return out


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


class HospitalSearchResult(BaseModel):
    hospitalMgmtNo: str
    name: str
    roadAddress: Optional[str] = None
    jibunAddress: Optional[str] = None
    lng: Optional[float] = None
    lat: Optional[float] = None
    isCustomEntry: bool


class HospitalNearbyResult(BaseModel):
    hospitalMgmtNo: str
    name: str
    roadAddress: Optional[str] = None
    jibunAddress: Optional[str] = None
    lat: float
    lng: float
    distanceM: int
    isCustomEntry: bool


class HospitalCustomCreateRequest(BaseModel):
    name: str
    roadAddress: Optional[str] = None
    jibunAddress: Optional[str] = None
    lng: Optional[float] = None
    lat: Optional[float] = None


class HospitalCustomCreateResponse(BaseModel):
    hospitalMgmtNo: str
    name: str
    roadAddress: Optional[str] = None
    jibunAddress: Optional[str] = None
    lng: Optional[float] = None
    lat: Optional[float] = None
    isCustomEntry: bool


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
app = FastAPI(title="PetHealth+ Server", version="2.1.22-ops")

# CORS hardening
_origins = [o.strip() for o in (settings.CORS_ALLOW_ORIGINS or "*").split(",") if o.strip()]
_allow_credentials = bool(settings.CORS_ALLOW_CREDENTIALS)
if "*" in _origins and _allow_credentials:
    # wildcard origins can't be used with credentials
    _allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global psycopg2 error handler
@app.exception_handler(psycopg2.Error)
async def _pg_error_handler(request: Request, exc: psycopg2.Error):
    try:
        _raise_mapped_db_error(exc)
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    return JSONResponse(status_code=500, content={"detail": _internal_detail(_pg_message(exc), kind="DB error")})


# Global fallback exception handler (DB 외 영역도 errorId 통일 + 로그 정리)
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
    return {"status": "ok", "message": "PetHealth+ Server Running (v2.1.22-ops)"}


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
        "version": "2.1.22-ops",
        "storage": "stub" if settings.STUB_MODE else "firebase",
        "storage_bucket": settings.FIREBASE_STORAGE_BUCKET,
        "receipt_webp_quality": int(settings.RECEIPT_WEBP_QUALITY),
        "receipt_max_width": int(settings.RECEIPT_MAX_WIDTH),
        "max_receipt_bytes": int(settings.MAX_RECEIPT_IMAGE_BYTES),
        "max_pdf_bytes": int(settings.MAX_PDF_BYTES),
        "image_max_pixels": int(settings.IMAGE_MAX_PIXELS),
        "signed_url_default_ttl": int(settings.SIGNED_URL_DEFAULT_TTL_SECONDS),
        "signed_url_max_ttl": int(settings.SIGNED_URL_MAX_TTL_SECONDS),
        "stub_mode": bool(settings.STUB_MODE),
        "auth_required": bool(settings.AUTH_REQUIRED),
        "db_enabled": bool(settings.DB_ENABLED),
        "db_configured": bool(settings.DATABASE_URL),
        "db_schema": db_checks,
        "firebase": fb_check,
        "ocr_hospital_candidate_limit": int(settings.OCR_HOSPITAL_CANDIDATE_LIMIT),
        "ocr_timeout_seconds": int(settings.OCR_TIMEOUT_SECONDS),
        "ocr_max_concurrency": int(settings.OCR_MAX_CONCURRENCY),
        "ocr_sema_acquire_timeout_seconds": float(settings.OCR_SEMA_ACQUIRE_TIMEOUT_SECONDS),
        "expose_error_details": bool(settings.EXPOSE_ERROR_DETAILS),
        "cors": {"origins": _origins, "allowCredentials": _allow_credentials},
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


# =========================================================
# DB APIs: users
# =========================================================
@app.post("/api/db/user/upsert")
def api_user_upsert(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or ""
    desired = _infer_membership_tier_from_token(user)
    row = db_touch_user(uid, desired_tier=desired)
    return jsonable_encoder(row)


# =========================================================
# DB APIs: pets
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
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


@app.get("/api/hospitals/search", response_model=List[HospitalSearchResult])
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


@app.get("/api/hospitals/nearby", response_model=List[HospitalNearbyResult])
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


@app.post("/api/hospitals/custom/create", response_model=HospitalCustomCreateResponse)
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

    total_amount = int(req.total_amount or 0)
    if total_amount < 0:
        raise HTTPException(status_code=400, detail="totalAmount must be >= 0")

    pet_weight_at_visit = float(req.pet_weight_at_visit) if req.pet_weight_at_visit is not None else None
    tags = _clean_tags(req.tags)

    try:
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
                        receipt_image_path, file_size_bytes,
                        created_at, updated_at
                    """,
                    (record_uuid, pet_uuid, hospital_mgmt_no, hospital_name, visit_date, total_amount, pet_weight_at_visit, tags, uid),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=500, detail=_internal_detail("Failed to upsert record", kind="DB error"))

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
                items_enriched = _enrich_items_with_canonical(uid, [dict(x) for x in items_rows])

                payload = dict(row)
                payload["items"] = items_enriched
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

        record_ids = [r["id"] for r in rows if r.get("id")]  # UUID 그대로
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

        items_enriched = _enrich_items_with_canonical(uid, items)

        by_record: Dict[str, List[Dict[str, Any]]] = {}
        for it in items_enriched:
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
        row["items"] = _enrich_items_with_canonical(uid, items)
        return jsonable_encoder(row)

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
        WHERE p.user_uid = %s AND r.id = %s
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
        row["items"] = _enrich_items_with_canonical(uid, items)
        row["hospitalCandidatesCleared"] = True
        return jsonable_encoder(row)
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise


# =========================================================
# Receipts (SYNC handler to avoid event-loop blocking)
# =========================================================
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

    pet = db_fetchone("SELECT id FROM public.pets WHERE id=%s AND user_uid=%s", (pet_uuid, uid))
    if not pet:
        raise HTTPException(status_code=404, detail="pet not found")

    raw = _read_upload_limited(upload, int(settings.MAX_RECEIPT_IMAGE_BYTES))
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    # 1) OCR -> redact -> resize -> webp
    try:
        webp_bytes, parsed, hints = process_receipt_image_and_parse(raw)
    except HTTPException:
        # OCR busy(503) 같은 의도된 예외는 그대로 전달
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Receipt processing failed"))

    # 2) upload to storage
    receipt_path = _receipt_path(uid, str(pet_uuid), str(record_uuid))
    try:
        upload_bytes_to_storage(receipt_path, webp_bytes, "image/webp")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))

    file_size_bytes = int(len(webp_bytes))

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

    mgmt_input = hospitalMgmtNo.strip() if isinstance(hospitalMgmtNo, str) and hospitalMgmtNo.strip() else None

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

    addr_hint = hints.get("addressHint") if isinstance(hints, dict) else None
    addr_hint = addr_hint if isinstance(addr_hint, str) else None

    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # optional: mgmt_no가 들어왔다면 scope 미리 체크(친절한 에러)
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

                # record upsert: 확정된 hospital_mgmt_no는 NULL로 덮어쓰지 않게 COALESCE
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
                      hospital_mgmt_no = COALESCE(EXCLUDED.hospital_mgmt_no, public.health_records.hospital_mgmt_no),
                      hospital_name = COALESCE(public.health_records.hospital_name, EXCLUDED.hospital_name),
                      visit_date = EXCLUDED.visit_date,
                      total_amount = EXCLUDED.total_amount,
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
                    (record_uuid, pet_uuid, mgmt_input, hn, vd, ta, [], receipt_path, file_size_bytes, uid),
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
                            (record_uuid, it["itemName"], it["price"], None),
                        )

                # hospital candidates (only if record not confirmed)
                candidates_out: List[Dict[str, Any]] = []
                record_confirmed = row.get("hospital_mgmt_no") is not None
                if not record_confirmed:
                    limit_n = int(settings.OCR_HOSPITAL_CANDIDATE_LIMIT or 3)
                    limit_n = max(1, min(limit_n, 10))

                    name_for_match = hn or ""
                    addr_for_match = addr_hint or ""
                    if (name_for_match.strip() != "") or (addr_for_match.strip() != ""):
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
                items_enriched = _enrich_items_with_canonical(uid, [dict(x) for x in items_rows])

                payload = dict(row)
                payload["items"] = items_enriched
                payload["hospitalCandidates"] = candidates_out
                payload["hospitalCandidateCount"] = len(candidates_out)
                payload["hospitalConfirmed"] = bool(payload.get("hospital_mgmt_no"))
                return jsonable_encoder(payload)

    except HTTPException as he:
        # DB 실패 시 업로드 파일 정리 시도
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
# Documents (PDF) - SYNC handler to avoid event-loop blocking
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
                WHERE p.user_uid=%s AND p.id=%s AND d.doc_type=%s
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
                WHERE p.user_uid=%s AND p.id=%s
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
    uid = (user.get("uid") or "").strip()
    doc_uuid = _uuid_or_400(docId, "docId")

    row = db_fetchone(
        """
        SELECT d.id, d.file_path
        FROM public.pet_documents d
        JOIN public.pets p ON p.id = d.pet_id
        WHERE p.user_uid=%s AND d.id=%s
        """,
        (uid, doc_uuid),
    )
    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    file_path = row["file_path"]

    try:
        delete_storage_object_if_exists(file_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=_internal_detail(str(e), kind="Storage error"))

    try:
        db_execute("DELETE FROM public.pet_documents WHERE id=%s", (doc_uuid,))
    except HTTPException:
        raise
    except Exception as e:
        _raise_mapped_db_error(e)
        raise

    return {"ok": True, "deleted": {"docId": str(doc_uuid), "filePath": file_path}}


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
    # backups: allow without DB rows
    if path.startswith(f"users/{uid}/backups/") and path.endswith(".json"):
        return {"kind": "backup"}

    rec = db_fetchone(
        """
        SELECT 1
        FROM public.health_records r
        JOIN public.pets p ON p.id = r.pet_id
        WHERE p.user_uid=%s AND r.receipt_image_path=%s
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
        WHERE p.user_uid=%s AND d.file_path=%s
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
# Backup endpoints (SYNC)
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
# Migration tokens + migration execution
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
        db_execute(
            "UPDATE public.migration_tokens SET status='failed' WHERE code_hash=%s",
            (code_hash,),
        )
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
        db_execute(
            "UPDATE public.migration_tokens SET status='failed' WHERE code_hash=%s",
            (code_hash,),
        )
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
    users_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.users") or {"c": 0}
    pets_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.pets") or {"c": 0}
    records_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.health_records") or {"c": 0}
    items_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.health_items") or {"c": 0}
    docs_cnt = db_fetchone("SELECT COUNT(*)::int AS c FROM public.pet_documents") or {"c": 0}
    total_amount = db_fetchone("SELECT COALESCE(SUM(total_amount),0)::bigint AS s FROM public.health_records") or {"s": 0}

    return {
        "users": int(users_cnt["c"]),
        "pets": int(pets_cnt["c"]),
        "records": int(records_cnt["c"]),
        "items": int(items_cnt["c"]),
        "docs": int(docs_cnt["c"]),
        "totalAmountSum": int(total_amount["s"]),
        "updatedAt": datetime.utcnow().isoformat() + "Z",
    }


