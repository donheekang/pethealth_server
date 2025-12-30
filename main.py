import os
import io
import json
import uuid
import tempfile
import re
import base64
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from google.cloud import vision
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Firebase Admin (Auth only)
import firebase_admin
from firebase_admin import credentials as fb_credentials
from firebase_admin import auth as fb_auth


# ------------------------------------------------
# 1. 설정 / 외부 모듈
# ------------------------------------------------
try:
    from condition_tags import CONDITION_TAGS
except ImportError:
    CONDITION_TAGS = {}
    print("Warning: condition_tags.py not found. AI tagging will be limited.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google.generativeai not installed. Gemini features disabled.")


class Settings(BaseSettings):
    # --- AWS / S3 ---
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-northeast-2"
    S3_BUCKET_NAME: str

    # --- Google Vision ---
    GOOGLE_APPLICATION_CREDENTIALS: str = ""  # JSON string or file path

    # --- Gemini ---
    GEMINI_ENABLED: str = "false"  # "true"/"false"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    # --- Firebase Auth (verify ID token) ---
    AUTH_REQUIRED: str = "true"  # "true"/"false"
    FIREBASE_ADMIN_SA_JSON: str = ""  # service account JSON string
    FIREBASE_ADMIN_SA_B64: str = ""   # base64-encoded JSON string (optional)

    # --- 디버그용 스텁 모드 ---
    STUB_MODE: str = "false"  # "true"면 인증/외부 호출 일부를 우회 가능

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


# ------------------------------------------------
# 2. Firebase Admin 초기화 & 인증 의존성
# ------------------------------------------------
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
            "Firebase Auth가 활성화(AUTH_REQUIRED=true)인데, "
            "FIREBASE_ADMIN_SA_JSON 또는 FIREBASE_ADMIN_SA_B64가 비어있습니다."
        )

    try:
        cred = fb_credentials.Certificate(info)
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        print("[Auth] Firebase Admin initialized.")
    except Exception as e:
        raise RuntimeError(f"Firebase Admin initialize 실패: {e}")


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
) -> Dict[str, Any]:
    """
    Authorization: Bearer <Firebase ID Token>
    을 검증하고, decoded token dict를 반환.
    """
    if settings.STUB_MODE.lower() == "true" or settings.AUTH_REQUIRED.lower() != "true":
        return {"uid": "dev", "email": "dev@example.com"}

    init_firebase_admin()

    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    token = credentials.credentials
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Firebase token: {e}")


def _safe_segment(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s or "unknown")


def _user_prefix(user_uid: str, pet_id: str) -> str:
    """
    ✅ S3 키를 사용자(uid) 단위로 네임스페이스 분리.
    """
    safe_uid = _safe_segment(user_uid)
    safe_pet = _safe_segment(pet_id)
    return f"users/{safe_uid}/pets/{safe_pet}"


def _backup_prefix(user_uid: str) -> str:
    safe_uid = _safe_segment(user_uid)
    return f"users/{safe_uid}/backups"


# ------------------------------------------------
# 3. S3 클라이언트
# ------------------------------------------------
s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


def upload_to_s3(file_obj, key: str, content_type: str) -> str:
    try:
        s3_client.upload_fileobj(
            file_obj,
            settings.S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )
        return url
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 인증 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")


def delete_from_s3(key: str) -> None:
    try:
        s3_client.head_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        s3_client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=key)

    except ClientError as e:
        err = e.response.get("Error") or {}
        code = err.get("Code", "")

        if code in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

        raise HTTPException(status_code=500, detail=f"S3 삭제 실패: {e}")

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 인증 실패")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 삭제 실패: {e}")


def get_json_from_s3(key: str) -> Dict[str, Any]:
    try:
        obj = s3_client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        raw = obj["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except ClientError as e:
        err = e.response.get("Error") or {}
        code = err.get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(status_code=404, detail="백업 파일을 찾을 수 없습니다.")
        raise HTTPException(status_code=500, detail=f"S3 get_object 실패: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"백업 JSON 로드 실패: {e}")


# ------------------------------------------------
# 4. Google Vision OCR
# ------------------------------------------------
_vision_client: Optional[vision.ImageAnnotatorClient] = None


def get_vision_client() -> vision.ImageAnnotatorClient:
    global _vision_client
    if _vision_client is not None:
        return _vision_client

    cred_value = settings.GOOGLE_APPLICATION_CREDENTIALS
    if not cred_value:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS 환경변수가 비어있습니다.")

    try:
        info = json.loads(cred_value)
        if isinstance(info, dict) and isinstance(info.get("private_key"), str):
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        _vision_client = vision.ImageAnnotatorClient.from_service_account_info(info)
        return _vision_client
    except json.JSONDecodeError:
        if not os.path.exists(cred_value):
            raise Exception(
                "GOOGLE_APPLICATION_CREDENTIALS가 JSON도 아니고, "
                f"파일 경로({cred_value})도 아닙니다."
            )
        _vision_client = vision.ImageAnnotatorClient.from_service_account_file(cred_value)
        return _vision_client
    except Exception as e:
        raise Exception(f"OCR 클라이언트 생성 실패: {e}")


def run_vision_ocr(image_path: str) -> str:
    client = get_vision_client()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"OCR 에러: {response.error.message}")

    texts = response.text_annotations
    if not texts:
        return ""

    return texts[0].description


# ------------------------------------------------
# 5. 영수증 파서 (✅ 최소 추출: 병원명/날짜만)
# ------------------------------------------------
def guess_hospital_name(lines: List[str]) -> str:
    keywords = [
        "동물병원", "동물 병원", "동물의료", "동물메디컬", "동물 메디컬",
        "동물클리닉", "동물 클리닉",
        "애견병원", "애완동물병원", "펫병원", "펫 병원",
        "종합동물병원", "동물의원", "동물병의원",
    ]

    best_line = None
    best_score = -1

    for idx, line in enumerate(lines):
        score = 0
        text = line.replace(" ", "")

        if any(k in text for k in keywords):
            score += 5

        if idx <= 4:
            score += 2

        if any(x in line for x in ["TEL", "전화", "FAX", "팩스", "도로명"]):
            score -= 2

        digit_count = sum(c.isdigit() for c in line)
        if digit_count >= 8:
            score -= 1

        if len(line) < 2 or len(line) > 25:
            score -= 1

        if score > best_score:
            best_score = score
            best_line = line

    if best_line is None and lines:
        return lines[0]
    return best_line or ""


def parse_receipt_minimal(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    hospital_name = guess_hospital_name(lines)

    visit_at = None
    dt_pattern = re.compile(
        r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2}).*?(\d{1,2}):(\d{2})"
    )
    for line in lines:
        m = dt_pattern.search(line)
        if m:
            y, mo, d, h, mi = map(int, m.groups())
            visit_at = datetime(y, mo, d, h, mi).strftime("%Y-%m-%d %H:%M")
            break

    if not visit_at:
        dt_pattern_short = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})")
        for line in lines:
            m = dt_pattern_short.search(line)
            if m:
                y, mo, d = map(int, m.groups())
                visit_at = datetime(y, mo, d).strftime("%Y-%m-%d")
                break

    return {
        "hospitalName": hospital_name,
        "visitAt": visit_at,
    }


# ------------------------------------------------
# 6. DTO (참고용)
# ------------------------------------------------
class CamelBase(BaseModel):
    class Config:
        populate_by_name = True
        extra = "ignore"


# ------------------------------------------------
# 7. BACKUP DTO
# ------------------------------------------------
class BackupUploadRequest(BaseModel):
    snapshot: Any
    clientTime: Optional[str] = None
    appVersion: Optional[str] = None
    device: Optional[str] = None
    note: Optional[str] = None


class BackupMeta(BaseModel):
    uid: str
    backupId: str
    createdAt: str
    clientTime: Optional[str] = None
    appVersion: Optional[str] = None
    device: Optional[str] = None
    note: Optional[str] = None


class BackupUploadResponse(BaseModel):
    ok: bool
    uid: str
    backupId: str
    objectKey: str
    createdAt: str


class BackupDocument(BaseModel):
    meta: BackupMeta
    snapshot: Any


# ------------------------------------------------
# 8. FASTAPI APP
# ------------------------------------------------
app = FastAPI(title="PetHealth+ Server", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영에서는 도메인 제한 권장
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


@app.get("/")
def root():
    return {"status": "ok", "message": "PetHealth+ Server Running"}


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_model": settings.GEMINI_MODEL_NAME,
        "gemini_enabled": settings.GEMINI_ENABLED,
        "stub_mode": settings.STUB_MODE,
        "auth_required": settings.AUTH_REQUIRED,
    }


@app.get("/api/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return {"uid": user.get("uid"), "email": user.get("email")}


# ------------------------------------------------
# 9. BACKUP ENDPOINTS (업로드/복구/리스트)
# ------------------------------------------------
@app.post("/api/backup/upload", response_model=BackupUploadResponse)
async def backup_upload(
    req: BackupUploadRequest = Body(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    backup_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat()

    meta = BackupMeta(
        uid=uid,
        backupId=backup_id,
        createdAt=created_at,
        clientTime=req.clientTime,
        appVersion=req.appVersion,
        device=req.device,
        note=req.note,
    ).model_dump()

    doc = {"meta": meta, "snapshot": req.snapshot}

    key = f"{_backup_prefix(uid)}/{backup_id}.json"
    raw = json.dumps(doc, ensure_ascii=False).encode("utf-8")
    bio = io.BytesIO(raw)
    bio.seek(0)

    upload_to_s3(bio, key, "application/json")

    return {
        "ok": True,
        "uid": uid,
        "backupId": backup_id,
        "objectKey": key,
        "createdAt": created_at,
    }


@app.get("/api/backup/latest", response_model=BackupDocument)
def backup_latest(
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    prefix = f"{_backup_prefix(uid)}/"

    resp = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)
    contents = resp.get("Contents") or []
    if not contents:
        raise HTTPException(status_code=404, detail="백업이 아직 없어요.")

    latest = max(contents, key=lambda o: o["LastModified"])
    key = latest["Key"]

    doc = get_json_from_s3(key)

    # 최소 방어
    if not isinstance(doc, dict) or "meta" not in doc or "snapshot" not in doc:
        raise HTTPException(status_code=500, detail="백업 파일 형식이 올바르지 않습니다.")

    return doc


@app.get("/api/backup/get", response_model=BackupDocument)
def backup_get(
    backupId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    bid = (backupId or "").strip()
    if not bid:
        raise HTTPException(status_code=400, detail="backupId is required")

    key = f"{_backup_prefix(uid)}/{bid}.json"
    doc = get_json_from_s3(key)

    if not isinstance(doc, dict) or "meta" not in doc or "snapshot" not in doc:
        raise HTTPException(status_code=500, detail="백업 파일 형식이 올바르지 않습니다.")
    return doc


@app.get("/api/backup/list")
def backup_list(
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    prefix = f"{_backup_prefix(uid)}/"
    resp = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)

    out = []
    for obj in resp.get("Contents") or []:
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        filename = key.split("/")[-1]
        backup_id = filename.replace(".json", "")
        out.append(
            {
                "backupId": backup_id,
                "objectKey": key,
                "lastModified": obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%S"),
                "size": obj.get("Size", 0),
            }
        )

    out.sort(key=lambda x: x["lastModified"], reverse=True)
    return out


# ------------------------------------------------
# 10. ENDPOINTS – 영수증 / PDF (기존 로직 + user_prefix 오타 수정)
# ------------------------------------------------
@app.post("/api/receipt/upload")
async def upload_receipt(
    petId: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
    upload: Optional[UploadFile] = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="no file or image field")

    uid = user.get("uid") or "unknown"

    rec_id = str(uuid.uuid4())
    _, ext = os.path.splitext(upload.filename or "")
    if not ext:
        ext = ".jpg"

    key = f"{_user_prefix(uid, petId)}/receipts/{rec_id}{ext}"

    data = await upload.read()
    file_like = io.BytesIO(data)
    file_like.seek(0)

    file_url = upload_to_s3(
        file_like,
        key,
        content_type=upload.content_type or "image/jpeg",
    )

    ocr_text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as tmp:
            tmp.write(data)
            tmp.flush()
            ocr_text = run_vision_ocr(tmp.name)
    except Exception as e:
        print("OCR error:", e)
        ocr_text = ""

    minimal = parse_receipt_minimal(ocr_text) if ocr_text else {"hospitalName": "", "visitAt": None}

    parsed_for_dto = {
        "clinicName": (minimal.get("hospitalName") or "").strip(),
        "visitDate": minimal.get("visitAt"),
        "diseaseName": None,
        "symptomsSummary": None,
        "items": [],
        "totalAmount": None,
    }

    clinic_name = (parsed_for_dto.get("clinicName") or "").strip()
    clinic_name = re.sub(r"^원\s*명[:：]?\s*", "", clinic_name)
    parsed_for_dto["clinicName"] = clinic_name

    return {
        "uid": uid,
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,
        "objectKey": key,
    }


@app.post("/api/lab/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form("검사결과"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    original_base = os.path.splitext(file.filename or "")[0].strip() or "검사결과"
    safe_base = re.sub(r"[^a-zA-Z0-9_\-가-힣]", "_", original_base)

    key = f"{user_prefix(uid, petId)}/lab/{safe_base}{uuid.uuid4().hex}.pdf"

    url = upload_to_s3(file.file, key, "application/pdf")
    created_at_iso = datetime.utcnow().isoformat()

    filename = key.split("/")[-1]
    base_name, _ = os.path.splitext(filename)

    return {
        "id": base_name,
        "uid": uid,
        "petId": petId,
        "title": title or original_base,
        "memo": memo,
        "s3Url": url,
        "createdAt": created_at_iso,
        "objectKey": key,
    }


@app.post("/api/cert/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form("증명서"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    original_base = os.path.splitext(file.filename or "")[0].strip() or "증명서"
    safe_base = re.sub(r"[^a-zA-Z0-9_\-가-힣]", "_", original_base)

    key = f"{user_prefix(uid, petId)}/cert/{safe_base}{uuid.uuid4().hex}.pdf"

    url = upload_to_s3(file.file, key, "application/pdf")
    created_at_iso = datetime.utcnow().isoformat()

    filename = key.split("/")[-1]
    base_name, _ = os.path.splitext(filename)

    return {
        "id": base_name,
        "uid": uid,
        "petId": petId,
        "title": title or original_base,
        "memo": memo,
        "s3Url": url,
        "createdAt": created_at_iso,
        "objectKey": key,
    }


@app.get("/api/labs/list")
def get_lab_list(
    petId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    prefix = f"{_user_prefix(uid, petId)}/lab/"
    response = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)

    items: List[Dict[str, Any]] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            base_name, _ = os.path.splitext(filename)

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )

            items.append(
                {
                    "id": base_name,
                    "uid": uid,
                    "petId": petId,
                    "title": f"검사결과 ({date_str})",
                    "s3Url": url,
                    "createdAt": created_at_iso,
                    "objectKey": key,
                }
            )

    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


@app.get("/api/cert/list")
def get_cert_list(
    petId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    prefix = f"{_user_prefix(uid, petId)}/cert/"
    response = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=prefix)

    items: List[Dict[str, Any]] = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue

            filename = key.split("/")[-1]
            base_name, _ = os.path.splitext(filename)

            created_dt = obj["LastModified"]
            created_at_iso = created_dt.strftime("%Y-%m-%dT%H:%M:%S")
            date_str = created_dt.strftime("%Y-%m-%d")

            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
                ExpiresIn=7 * 24 * 3600,
            )

            items.append(
                {
                    "id": base_name,
                    "uid": uid,
                    "petId": petId,
                    "title": f"증명서 ({date_str})",
                    "s3Url": url,
                    "createdAt": created_at_iso,
                    "objectKey": key,
                }
            )

    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


@app.delete("/api/lab/delete")
def delete_lab_pdf(
    petId: str = Query(...),
    id: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    object_id = (id or "").strip()
    if not object_id:
        raise HTTPException(status_code=400, detail="id is required")

    filename = object_id if object_id.endswith(".pdf") else f"{object_id}.pdf"
    key = f"{_user_prefix(uid, petId)}/lab/{filename}"

    delete_from_s3(key)
    return {"ok": True, "deletedKey": key}


@app.delete("/api/cert/delete")
def delete_cert_pdf(
    petId: str = Query(...),
    id: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    object_id = (id or "").strip()
    if not object_id:
        raise HTTPException(status_code=400, detail="id is required")

    filename = object_id if object_id.endswith(".pdf") else f"{object_id}.pdf"
    key = f"{_user_prefix(uid, petId)}/cert/{filename}"

    delete_from_s3(key)
    return {"ok": True, "deletedKey": key}


# ------------------------------------------------
# 11. AI 케어 (기존 유지)
# ------------------------------------------------
def _parse_visit_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        try:
            part = s.split()[0]
            return datetime.strptime(part, "%Y-%m-%d").date()
        except Exception:
            return None


def _build_tag_stats(
    medical_history: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    today = date.today()

    agg: Dict[str, Dict[str, Any]] = {}
    period_stats: Dict[str, Dict[str, int]] = {"1m": {}, "3m": {}, "1y": {}}

    for mh in medical_history:
        visit_str = mh.get("visitDate") or mh.get("visit_date") or ""
        visit_dt = _parse_visit_date(visit_str)
        visit_date_str = visit_dt.isoformat() if visit_dt else None

        record_tags: List[str] = mh.get("tags") or []
        if not record_tags:
            continue

        used_codes: set[str] = set()

        for code in record_tags:
            cfg = CONDITION_TAGS.get(code)
            if not cfg:
                continue

            canonical_code = cfg.code
            if canonical_code in used_codes:
                continue
            used_codes.add(canonical_code)

            stat = agg.setdefault(
                canonical_code,
                {
                    "tag": canonical_code,
                    "label": cfg.label,
                    "group": getattr(cfg, "group", ""),
                    "count": 0,
                    "recentDates": [],
                },
            )
            stat["count"] += 1
            if visit_date_str:
                stat["recentDates"].append(visit_date_str)

            if visit_dt:
                days = (today - visit_dt).days
                if days <= 365:
                    period_stats["1y"][canonical_code] = period_stats["1y"].get(canonical_code, 0) + 1
                if days <= 90:
                    period_stats["3m"][canonical_code] = period_stats["3m"].get(canonical_code, 0) + 1
                if days <= 30:
                    period_stats["1m"][canonical_code] = period_stats["1m"].get(canonical_code, 0) + 1

    for stat in agg.values():
        stat["recentDates"] = sorted(stat["recentDates"], reverse=True)

    tags = sorted(agg.values(), key=lambda x: x["count"], reverse=True)
    return tags, period_stats


def _build_gemini_prompt(
    pet_name: str,
    tags: List[Dict[str, Any]],
    period_stats: Dict[str, Dict[str, int]],
    body: Dict[str, Any],
) -> str:
    profile = body.get("profile") or {}
    species = profile.get("species", "dog")
    age_text = profile.get("ageText") or profile.get("age_text") or ""
    weight = profile.get("weightCurrent") or profile.get("weight_current")

    mh_list = body.get("medicalHistory") or []
    mh_summary_lines = []
    for mh in mh_list[:8]:
        clinic = mh.get("clinicName") or mh.get("clinic_name") or ""
        visit = mh.get("visitDate") or mh.get("visit_date") or ""
        tag_list = mh.get("tags") or []
        mh_summary_lines.append(f"- {visit} / {clinic} / tags={tag_list}")

    tag_lines = []
    for t in tags:
        recent_dates = ", ".join(t.get("recentDates", [])[:3])
        group = t.get("group") or ""
        tag_lines.append(
            f"- {t['label']} ({t['tag']}, group={group}) : {t['count']}회 (최근: {recent_dates or '정보 없음'})"
        )

    prompt = f"""
당신은 반려동물 건강 기록을 '정리/요약'해 주는 어시스턴트입니다. 아래 입력은 보호자가 앱에서 직접 기록한 태그(확정 정보)입니다.

[중요 규칙]
1) 아래 '태그'에 없는 질환/컨디션을 새로 추정하거나 만들어내지 마세요.
2) group=exam/medication/procedure 태그는 '질환 진단'이 아니라 검사/처방/시술 기록입니다. 이를 근거로 질환을 단정하지 마세요.
3) 의료적 진단/판단을 하지 말고, 보호자 관점의 정리/안심/현실적인 관리 팁만 제공하세요.
4) 출력은 마크다운 없이 '문장만' 출력합니다. 불릿/번호/따옴표/코드블록 금지.

[반려동물 기본 정보]
이름: {pet_name}
종: {species}
나이: {age_text or '정보 없음'}
현재 체중: {weight if weight is not None else '정보 없음'} kg

[확정 태그 통계]
{os.linesep.join(tag_lines) if tag_lines else '태그 없음'}

[최근 진료 이력 요약(최대 8개)]
{os.linesep.join(mh_summary_lines) if mh_summary_lines else '진료 내역 없음'}

요약은 3~5문장으로 짧게 작성해 주세요. 너무 무섭게 말하지 말고, "잘 관리되고 있다"는 안정감을 주는 톤으로 작성해 주세요.
""".strip()

    return prompt


def _generate_gemini_summary(
    pet_name: str,
    tags: List[Dict[str, Any]],
    period_stats: Dict[str, Dict[str, int]],
    body: Dict[str, Any],
) -> Optional[str]:
    if settings.GEMINI_ENABLED.lower() != "true":
        return None
    if not settings.GEMINI_API_KEY:
        return None
    if genai is None:
        return None

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        prompt = _build_gemini_prompt(pet_name, tags, period_stats, body)
        resp = model.generate_content(prompt)

        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts)

        summary = (text or "").strip()
        if not summary:
            return None

        summary = summary.strip("`").strip()
        return summary

    except Exception as e:
        print("[AI] Gemini summary error:", e)
        return None


@app.post("/api/ai/analyze")
async def analyze_pet_health(
    body: Dict[str, Any],
    user: Dict[str, Any] = Depends(get_current_user),
):
    profile = body.get("profile") or {}
    pet_name = profile.get("name") or "반려동물"

    medical_history = body.get("medicalHistory") or body.get("medical_history") or []
    has_history = len(medical_history) > 0

    tags, period_stats = _build_tag_stats(medical_history)

    care_guide: Dict[str, List[str]] = {}
    for t in tags:
        code = t["tag"]
        cfg = CONDITION_TAGS.get(code)
        if cfg and getattr(cfg, "guide", None):
            care_guide[code] = list(cfg.guide)

    ai_summary = _generate_gemini_summary(pet_name, tags, period_stats, body)

    if ai_summary:
        summary = ai_summary
    else:
        if not has_history:
            summary = (
                f"{pet_name}의 병원 기록이 아직 많지 않아요. "
                "필요할 때 한 줄씩만 남겨도, 다음부터는 더 잘 정리해 드릴게요."
            )
        elif not tags:
            summary = (
                f"{pet_name}의 병원 기록은 있지만, "
                "아직 확정 태그가 저장된 기록이 없어서 컨디션별 통계를 만들기 어려워요. "
                "기록할 때 태그가 함께 저장되면 더 정확하게 정리해 드릴게요."
            )
        else:
            top = tags[0]
            summary = (
                f"최근 기록에서 '{top['label']}' 관련 태그가 {top['count']}회 확인됐어요. "
                "기록을 바탕으로 관리 포인트를 차분히 정리해 드렸어요."
            )

    return {
        "uid": user.get("uid"),
        "summary": summary,
        "tags": tags,
        "periodStats": period_stats,
        "careGuide": care_guide,
    }
