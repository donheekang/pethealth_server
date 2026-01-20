import os
import io
import json
import uuid
import tempfile
import re
import base64
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple, Set

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from google.cloud import vision

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Firebase Admin (Auth only)
import firebase_admin
from firebase_admin import credentials as fb_credentials
from firebase_admin import auth as fb_auth

# ------------------------------------------------
# 1. 설정 / 외부 모듈
# ------------------------------------------------
try:
    # condition_tags.py (확장판) 기준
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

    # --- ✅ 태그 매칭 강화 옵션 ---
    # tags가 없거나(혹은 전부 unknown)일 때, record 텍스트에서 키워드 기반 태그 추론(비진단 그룹만 기본)
    TAG_INFERENCE_ENABLED: str = "true"
    # 기본은 진단(orthopedics/dermatology/cardiology 등) 제외하고 안전한 그룹만
    TAG_INFERENCE_ALLOWED_GROUPS: str = "exam,medication,procedure,preventive,wellness"
    # 점수 임계값(높을수록 오탐 감소)
    TAG_INFERENCE_MIN_SCORE: int = 170
    # 한 레코드에서 최대 몇 개까지 자동 부여
    TAG_INFERENCE_MAX_TAGS: int = 6

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()


# ------------------------------------------------
# 1.1 ✅ 태그 매칭 강화 유틸 (문장 길어도 안 빠지게)
# ------------------------------------------------
def _is_single_latin_character(s: str) -> bool:
    s = (s or "").strip()
    if len(s) != 1:
        return False
    c = ord(s)
    return (65 <= c <= 90) or (97 <= c <= 122)


def _normalize_text(s: str) -> str:
    """
    소문자 + 알파뉴메릭만 남김 (한글도 isalnum=True로 포함됨)
    """
    s = (s or "").lower()
    return "".join(ch for ch in s if ch.isalnum())


def _tokenize_text(s: str) -> List[str]:
    """
    알파뉴메릭 덩어리로 토큰화 (공백/특수문자 기준)
    """
    out: List[str] = []
    cur: List[str] = []
    for ch in (s or ""):
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _is_short_ascii_token(norm: str) -> bool:
    """
    us/ua/pi/l2 같은 짧은 ASCII 토큰은 포함검색 오탐이 많아서 "토큰 일치"만 허용
    """
    if not norm:
        return False
    if len(norm) > 2:
        return False
    for ch in norm:
        o = ord(ch)
        ok = (48 <= o <= 57) or (97 <= o <= 122)  # 0-9, a-z
        if not ok:
            return False
    return True


_WEAK_KEYWORDS_NORM: Set[str] = set(map(_normalize_text, [
    # 너무 광범위해서 오탐 위험 큰 단어(약하게 취급)
    "검사", "진료", "기본진료", "상담", "진찰",
    "약", "약값", "처방", "처방약",
    "예방", "예방접종", "백신", "접종",
    "처치", "시술", "처치료", "시술료",
    "기타", "other", "etc",
]))


def _match_score(query_raw: str, keywords: List[str], strong_fields: Optional[List[str]] = None) -> int:
    """
    iOS에서 했던 방식 그대로: normalize + tokenize + (contains 양방향) + 짧은 약어는 token match
    """
    q_raw = (query_raw or "").strip()
    if not q_raw:
        return 0

    if _is_single_latin_character(q_raw):
        return 0

    q_norm = _normalize_text(q_raw)
    if not q_norm:
        return 0

    tokens = [_normalize_text(t) for t in _tokenize_text(q_raw)]
    tokens = [t for t in tokens if t]
    token_set = set(tokens)

    best = 0
    hit_count = 0
    strong_hit = False

    def bump(score: int, is_strong: bool) -> None:
        nonlocal best, hit_count, strong_hit
        if score > 0:
            hit_count += 1
        if is_strong and score > 0:
            strong_hit = True
        best = max(best, score)

    # 1) strong_fields (code/label 같은 “정체성 필드”)가 문장 안에 포함되면 크게
    if strong_fields:
        for sf in strong_fields:
            sf_norm = _normalize_text(sf)
            if sf_norm and sf_norm in q_norm:
                bump(200 + min(30, len(sf_norm)), is_strong=True)

    # 2) keywords 매칭
    for k in (keywords or []):
        k_norm = _normalize_text(k)
        if not k_norm:
            continue

        is_weak = (k_norm in _WEAK_KEYWORDS_NORM)

        # 짧은 토큰은 토큰 일치만
        if _is_short_ascii_token(k_norm):
            if k_norm in token_set:
                bump(70 if is_weak else 135, is_strong=not is_weak)
            continue

        if len(k_norm) <= 1:
            continue

        if k_norm == q_norm:
            bump(110 if is_weak else 180, is_strong=not is_weak)
        elif k_norm in q_norm:
            base = 55 if is_weak else 120
            bump(base + min(60, len(k_norm) * 2), is_strong=not is_weak)
        elif q_norm in k_norm:
            base = 35 if is_weak else 90
            bump(base + min(40, len(q_norm) * 2), is_strong=not is_weak)

    # 3) 여러 히트 보너스
    if hit_count >= 2:
        best += min(35, hit_count * (8 if strong_hit else 5))

    return best


def _normalize_tag_code(code: str) -> str:
    """
    tag code 입력이 제멋대로 들어와도 snake_case로 최대한 정리
    - examXray / exam-xray / EXAM_XRAY → exam_xray
    """
    s = (code or "").strip()
    if not s:
        return ""

    # 공백/하이픈 → _
    s = re.sub(r"[\s\-]+", "_", s)

    # camelCase → snake_case
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)

    s = s.lower()
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _resolve_tag_config(code: str) -> Optional[Any]:
    """
    들어온 tag code를 CONDITION_TAGS에서 최대한 찾아서 config로 반환
    """
    if not code:
        return None

    raw = code.strip()
    if raw in CONDITION_TAGS:
        return CONDITION_TAGS[raw]

    norm = _normalize_tag_code(raw)
    if norm in CONDITION_TAGS:
        return CONDITION_TAGS[norm]

    # 마지막 fallback: 대소문자/공백 제거 키도 시도
    norm2 = _normalize_text(raw)
    if norm2 and norm2 in CONDITION_TAGS:
        return CONDITION_TAGS[norm2]

    return None


def _build_canonical_tag_map() -> Dict[str, Any]:
    """
    CONDITION_TAGS에는 alias key들도 들어있으니, canonical(code) 기준으로 1개씩만 남김
    """
    out: Dict[str, Any] = {}
    for cfg in (CONDITION_TAGS or {}).values():
        try:
            c = getattr(cfg, "code", None)
            if not c:
                continue
            if c not in out:
                out[c] = cfg
        except Exception:
            continue
    return out


_CANONICAL_TAGS: Dict[str, Any] = _build_canonical_tag_map()


def _allowed_inference_groups() -> Set[str]:
    raw = (settings.TAG_INFERENCE_ALLOWED_GROUPS or "").strip()
    if not raw:
        return set()
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return set(parts)


_ALLOWED_GROUPS = _allowed_inference_groups()


def _collect_record_text(mh: Dict[str, Any]) -> str:
    """
    레코드 텍스트를 최대한 합쳐서(문장형 입력에서도) 태그 매칭에 사용
    """
    fields: List[str] = []

    # 흔히 들어오는 필드들
    for key in [
        "clinicName", "clinic_name",
        "diseaseName", "disease_name",
        "symptomsSummary", "symptoms_summary",
        "memo", "note", "notes", "description",
        "diagnosis", "diagnosisText", "diagnosis_text",
        "rawText", "raw_text",
    ]:
        v = mh.get(key)
        if isinstance(v, str) and v.strip():
            fields.append(v.strip())

    # items(name)
    items = mh.get("items")
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                name = it.get("name") or it.get("title")
                if isinstance(name, str) and name.strip():
                    fields.append(name.strip())
            elif isinstance(it, str) and it.strip():
                fields.append(it.strip())

    # 혹시 tags도 텍스트로 섞어두는 케이스 방어
    tags = mh.get("tags")
    if isinstance(tags, list):
        fields.extend([str(t) for t in tags if t])

    return " ".join(fields)


def _infer_tags_from_text(
    mh: Dict[str, Any],
    species: str = "both",
    limit: Optional[int] = None,
) -> List[str]:
    """
    tags가 비어있을 때, record 텍스트에서 키워드 매칭으로 태그 보강.
    기본은 "진단 추정" 위험 줄이기 위해 allowed group만 대상으로 함.
    """
    if (settings.TAG_INFERENCE_ENABLED or "").lower() != "true":
        return []

    text = _collect_record_text(mh)
    if not text.strip():
        return []

    limit = limit if limit is not None else settings.TAG_INFERENCE_MAX_TAGS
    min_score = int(settings.TAG_INFERENCE_MIN_SCORE)

    scored: List[Tuple[str, int]] = []

    for code, cfg in _CANONICAL_TAGS.items():
        try:
            group = (getattr(cfg, "group", "") or "").lower()
            if _ALLOWED_GROUPS and group not in _ALLOWED_GROUPS:
                continue

            cfg_species = (getattr(cfg, "species", "both") or "both").lower()
            # species 필터(고양이/강아지)
            if species in ("dog", "cat"):
                if cfg_species not in ("both", species):
                    continue

            kw_list = []
            # strong fields: code/label
            strong_fields = [getattr(cfg, "code", ""), getattr(cfg, "label", "")]
            # keywords: code/label + cfg.keywords
            kw_list.extend(strong_fields)
            kw_list.extend(getattr(cfg, "keywords", []) or [])

            s = _match_score(text, kw_list, strong_fields=strong_fields)
            if s >= min_score:
                scored.append((code, s))
        except Exception:
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for (c, _) in scored[: max(1, int(limit))]]


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


# ------------------------------------------------
# 3. S3 클라이언트 & 경로 헬퍼
# ------------------------------------------------
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


s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)


def _presign_get_url(key: str, expires_seconds: int = 7 * 24 * 3600) -> str:
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
        ExpiresIn=expires_seconds,
    )


def _list_all_objects(prefix: str) -> List[Dict[str, Any]]:
    """
    ✅ list_objects_v2는 1000개 제한이 있으니 pagination 처리.
    """
    out: List[Dict[str, Any]] = []
    token = None
    while True:
        kwargs = {"Bucket": settings.S3_BUCKET_NAME, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3_client.list_objects_v2(**kwargs)
        out.extend(resp.get("Contents") or [])

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return out


def upload_to_s3(file_obj, key: str, content_type: str) -> str:
    """
    파일을 S3에 업로드하고 7일 presigned URL을 반환.
    ⚠️ S3 object Metadata는 ASCII만 허용이라, title/memo 같은 한글은 절대 Metadata로 넣지 않음.
    """
    try:
        try:
            file_obj.seek(0)
        except Exception:
            pass

        s3_client.upload_fileobj(
            file_obj,
            settings.S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        return _presign_get_url(key)
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS S3 인증 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {e}")


def upload_bytes_to_s3(data: bytes, key: str, content_type: str) -> str:
    bio = io.BytesIO(data)
    bio.seek(0)
    return upload_to_s3(bio, key, content_type)


def upload_json_to_s3(obj: Any, key: str) -> str:
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return upload_bytes_to_s3(raw, key, "application/json")


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


def delete_from_s3_if_exists(key: str) -> None:
    try:
        s3_client.head_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
    except ClientError as e:
        err = e.response.get("Error") or {}
        code = err.get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return
        raise HTTPException(status_code=500, detail=f"S3 head_object 실패: {e}")
    s3_client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=key)


def get_json_from_s3(key: str) -> Dict[str, Any]:
    try:
        obj = s3_client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        raw = obj["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except ClientError as e:
        err = e.response.get("Error") or {}
        code = err.get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        raise HTTPException(status_code=500, detail=f"S3 get_object 실패: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 JSON 로드 실패: {e}")


def get_json_from_s3_optional(key: str) -> Optional[Dict[str, Any]]:
    try:
        return get_json_from_s3(key)
    except HTTPException as e:
        if e.status_code == 404:
            return None
        raise


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
    dt_pattern = re.compile(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2}).*?(\d{1,2}):(\d{2})")
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

    return {"hospitalName": hospital_name, "visitAt": visit_at}


# ------------------------------------------------
# 6. BACKUP DTO
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
# 7. FASTAPI APP
# ------------------------------------------------
app = FastAPI(title="PetHealth+ Server", version="1.7.0")

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
        "tag_inference_enabled": settings.TAG_INFERENCE_ENABLED,
        "tag_inference_allowed_groups": settings.TAG_INFERENCE_ALLOWED_GROUPS,
        "tag_inference_min_score": settings.TAG_INFERENCE_MIN_SCORE,
    }


@app.get("/api/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return {"uid": user.get("uid"), "email": user.get("email")}


# ------------------------------------------------
# 8. BACKUP ENDPOINTS (업로드/복구/리스트)
# ------------------------------------------------
@app.post("/backup/upload")
@app.post("/api/backup/upload", response_model=BackupUploadResponse)
async def backup_upload(
    req: BackupUploadRequest = Body(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    backup_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat()

    try:
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
        raw = json.dumps(doc, ensure_ascii=False).encode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"backup snapshot JSON 직렬화 실패: {e}")

    key = f"{_backup_prefix(uid)}/{backup_id}.json"
    bio = io.BytesIO(raw)
    bio.seek(0)

    upload_to_s3(bio, key, "application/json")

    return {"ok": True, "uid": uid, "backupId": backup_id, "objectKey": key, "createdAt": created_at}


@app.get("/backup/latest")
@app.get("/api/backup/latest", response_model=BackupDocument)
def backup_latest(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or "unknown"
    prefix = f"{_backup_prefix(uid)}/"

    contents = _list_all_objects(prefix)
    if not contents:
        raise HTTPException(status_code=404, detail="백업이 아직 없어요.")

    latest = max(contents, key=lambda o: o["LastModified"])
    key = latest["Key"]

    doc = get_json_from_s3(key)
    if not isinstance(doc, dict) or "meta" not in doc or "snapshot" not in doc:
        raise HTTPException(status_code=500, detail="백업 파일 형식이 올바르지 않습니다.")
    return doc


@app.get("/backup/get", response_model=BackupDocument)
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


@app.get("/backup/list")
@app.get("/api/backup/list")
def backup_list(user: Dict[str, Any] = Depends(get_current_user)):
    uid = user.get("uid") or "unknown"
    prefix = f"{_backup_prefix(uid)}/"

    objects = _list_all_objects(prefix)

    out = []
    for obj in objects:
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
# 9. 영수증 업로드/저장 (OCR + 이미지 + meta.json 저장)
# ------------------------------------------------
def _receipt_keys(uid: str, petId: str, rec_id: str, ext: str) -> Tuple[str, str]:
    base = _user_prefix(uid, petId)
    img_key = f"{base}/receipts/{rec_id}{ext}"
    meta_key = f"{base}/receipts/{rec_id}.json"
    return img_key, meta_key


def _build_receipt_item(
    uid: str,
    petId: str,
    meta_key: str,
    meta: Dict[str, Any],
    include_notes: bool = False,
) -> Dict[str, Any]:
    rec_id = meta.get("id") or meta_key.split("/")[-1].replace(".json", "")
    img_key = meta.get("imageObjectKey") or meta.get("objectKey") or ""
    parsed = meta.get("parsed") or {}
    notes = meta.get("notes") if include_notes else None

    created_at = meta.get("createdAt") or meta.get("created_at") or None

    return {
        "id": rec_id,
        "uid": uid,
        "petId": petId,
        "createdAt": created_at,
        "s3Url": _presign_get_url(img_key) if img_key else None,
        "objectKey": img_key,
        "metaObjectKey": meta_key,
        "parsed": parsed,
        "notes": notes,
    }


@app.post("/receipts/upload")
@app.post("/api/receipt/upload")
@app.post("/api/receipts/upload")
@app.post("/api/receipt/analyze")
@app.post("/api/receipts/analyze")
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

    rec_id = uuid.uuid4().hex
    _, ext = os.path.splitext(upload.filename or "")
    if not ext:
        ext = ".jpg"

    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    img_key, meta_key = _receipt_keys(uid, petId, rec_id, ext)

    # 1) 이미지 업로드
    file_url = upload_bytes_to_s3(
        data,
        img_key,
        content_type=upload.content_type or "image/jpeg",
    )

    # 2) OCR
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

    # ✅ 3) meta.json 저장
    created_at = datetime.utcnow().isoformat()
    receipt_meta = {
        "schemaVersion": 1,
        "id": rec_id,
        "uid": uid,
        "petId": petId,
        "createdAt": created_at,
        "imageObjectKey": img_key,
        "contentType": upload.content_type or "image/jpeg",
        "parsed": parsed_for_dto,
        "notes": ocr_text,
    }
    upload_json_to_s3(receipt_meta, meta_key)

    return {
        "id": rec_id,
        "createdAt": created_at,
        "uid": uid,
        "petId": petId,
        "s3Url": file_url,
        "parsed": parsed_for_dto,
        "notes": ocr_text,
        "objectKey": img_key,
        "metaObjectKey": meta_key,
    }


@app.get("/receipts/list")
@app.get("/receipt/list")
@app.get("/api/receipts/list")
@app.get("/api/receipt/list")
def list_receipts(
    petId: str = Query(...),
    includeNotes: bool = Query(False),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    base = _user_prefix(uid, petId)
    prefix = f"{base}/receipts/"

    objects = _list_all_objects(prefix)

    meta_keys = [o["Key"] for o in objects if o["Key"].endswith(".json")]
    meta_keys.sort(reverse=True)

    items: List[Dict[str, Any]] = []
    for mk in meta_keys:
        meta = get_json_from_s3_optional(mk)
        if not meta:
            continue
        items.append(_build_receipt_item(uid, petId, mk, meta, include_notes=includeNotes))

    def _sort_key(it: Dict[str, Any]) -> str:
        return (it.get("createdAt") or "") + (it.get("id") or "")

    items.sort(key=_sort_key, reverse=True)
    return items


@app.get("/receipts/get")
@app.get("/receipt/get")
@app.get("/api/receipts/get")
@app.get("/api/receipt/get")
def get_receipt(
    petId: str = Query(...),
    id: str = Query(...),
    includeNotes: bool = Query(True),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    rec_id = (id or "").strip()
    if not rec_id:
        raise HTTPException(status_code=400, detail="id is required")

    base = _user_prefix(uid, petId)
    meta_key = f"{base}/receipts/{rec_id}.json"

    meta = get_json_from_s3_optional(meta_key)
    if not meta:
        raise HTTPException(status_code=404, detail="영수증 메타를 찾을 수 없습니다.")

    return _build_receipt_item(uid, petId, meta_key, meta, include_notes=includeNotes)


@app.delete("/receipts/delete")
@app.delete("/receipt/delete")
@app.delete("/api/receipts/delete")
@app.delete("/api/receipt/delete")
def delete_receipt(
    petId: str = Query(...),
    id: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    rec_id = (id or "").strip()
    if not rec_id:
        raise HTTPException(status_code=400, detail="id is required")

    base = _user_prefix(uid, petId)

    meta_key = f"{base}/receipts/{rec_id}.json"
    meta = get_json_from_s3_optional(meta_key)

    deleted_keys = []

    if meta:
        img_key = meta.get("imageObjectKey") or meta.get("objectKey")
        if img_key:
            try:
                delete_from_s3(img_key)
                deleted_keys.append(img_key)
            except HTTPException as e:
                if e.status_code != 404:
                    raise
        delete_from_s3_if_exists(meta_key)
        deleted_keys.append(meta_key)
        return {"ok": True, "deleted": deleted_keys}

    prefix = f"{base}/receipts/{rec_id}"
    objects = _list_all_objects(prefix)
    if not objects:
        raise HTTPException(status_code=404, detail="삭제할 영수증을 찾을 수 없습니다.")

    for obj in objects:
        k = obj["Key"]
        s3_client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=k)
        deleted_keys.append(k)

    return {"ok": True, "deleted": deleted_keys}


# ------------------------------------------------
# 10. PDF 업로드/리스트/삭제 (검사결과 / 접종증명서)
# ------------------------------------------------
def _pdf_prefix_candidates(uid: str, petId: str, kind: str) -> List[str]:
    base = _user_prefix(uid, petId)

    if kind == "lab":
        return [f"{base}/lab/", f"{base}/labs/"]
    if kind == "cert":
        return [f"{base}/cert/", f"{base}/certs/", f"{base}/certificates/"]

    return [f"{base}/{kind}/"]


def _make_pdf_keys(uid: str, petId: str, kind: str, record_id: str) -> Tuple[str, str]:
    base = _user_prefix(uid, petId)
    pdf_key = f"{base}/{kind}/{record_id}.pdf"
    meta_key = f"{base}/{kind}/{record_id}.json"
    return pdf_key, meta_key


def _build_pdf_item(
    kind: str,
    uid: str,
    petId: str,
    pdf_key: str,
    last_modified: datetime,
    meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    filename = pdf_key.split("/")[-1]
    record_id, _ = os.path.splitext(filename)

    created_at_iso = last_modified.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = last_modified.strftime("%Y-%m-%d")

    default_title = "검사결과" if kind == "lab" else "접종증명서"
    title = (meta or {}).get("title") or f"{default_title} ({date_str})"
    memo = (meta or {}).get("memo")

    return {
        "id": record_id,
        "uid": uid,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": _presign_get_url(pdf_key),
        "createdAt": created_at_iso,
        "objectKey": pdf_key,
    }


@app.post("/lab/upload-pdf")
@app.post("/labs/upload-pdf")
@app.post("/api/lab/upload-pdf")
@app.post("/api/labs/upload-pdf")
async def upload_lab_pdf(
    petId: str = Form(...),
    title: str = Form("검사결과"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 PDF 파일입니다.")

    record_id = uuid.uuid4().hex
    pdf_key, meta_key = _make_pdf_keys(uid, petId, "lab", record_id)

    pdf_url = upload_bytes_to_s3(data, pdf_key, "application/pdf")

    meta = {
        "id": record_id,
        "uid": uid,
        "petId": petId,
        "kind": "lab",
        "title": title,
        "memo": memo,
        "originalFilename": file.filename,
        "createdAt": datetime.utcnow().isoformat(),
        "objectKey": pdf_key,
    }
    upload_json_to_s3(meta, meta_key)

    return {
        "id": record_id,
        "uid": uid,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": pdf_url,
        "createdAt": meta["createdAt"],
        "objectKey": pdf_key,
    }


@app.post("/cert/upload-pdf")
@app.post("/certs/upload-pdf")
@app.post("/certificates/upload-pdf")
@app.post("/api/cert/upload-pdf")
@app.post("/api/certs/upload-pdf")
@app.post("/api/certificates/upload-pdf")
async def upload_cert_pdf(
    petId: str = Form(...),
    title: str = Form("접종증명서"),
    memo: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 PDF 파일입니다.")

    record_id = uuid.uuid4().hex
    pdf_key, meta_key = _make_pdf_keys(uid, petId, "cert", record_id)

    pdf_url = upload_bytes_to_s3(data, pdf_key, "application/pdf")

    meta = {
        "id": record_id,
        "uid": uid,
        "petId": petId,
        "kind": "cert",
        "title": title,
        "memo": memo,
        "originalFilename": file.filename,
        "createdAt": datetime.utcnow().isoformat(),
        "objectKey": pdf_key,
    }
    upload_json_to_s3(meta, meta_key)

    return {
        "id": record_id,
        "uid": uid,
        "petId": petId,
        "title": title,
        "memo": memo,
        "s3Url": pdf_url,
        "createdAt": meta["createdAt"],
        "objectKey": pdf_key,
    }


@app.get("/lab/list")
@app.get("/labs/list")
@app.get("/api/lab/list")
@app.get("/api/labs/list")
def get_lab_list(
    petId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for prefix in _pdf_prefix_candidates(uid, petId, "lab"):
        for obj in _list_all_objects(prefix):
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue
            if key in seen:
                continue
            seen.add(key)

            filename = key.split("/")[-1]
            record_id = os.path.splitext(filename)[0]
            meta_key = f"{prefix}{record_id}.json"
            meta = get_json_from_s3_optional(meta_key)

            items.append(_build_pdf_item("lab", uid, petId, key, obj["LastModified"], meta))

    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


@app.get("/cert/list")
@app.get("/certs/list")
@app.get("/certificates/list")
@app.get("/api/cert/list")
@app.get("/api/certs/list")
@app.get("/api/certificates/list")
def get_cert_list(
    petId: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"

    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for prefix in _pdf_prefix_candidates(uid, petId, "cert"):
        for obj in _list_all_objects(prefix):
            key = obj["Key"]
            if not key.endswith(".pdf"):
                continue
            if key in seen:
                continue
            seen.add(key)

            filename = key.split("/")[-1]
            record_id = os.path.splitext(filename)[0]
            meta_key = f"{prefix}{record_id}.json"
            meta = get_json_from_s3_optional(meta_key)

            items.append(_build_pdf_item("cert", uid, petId, key, obj["LastModified"], meta))

    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items


@app.delete("/lab/delete")
@app.delete("/labs/delete")
@app.delete("/api/lab/delete")
@app.delete("/api/labs/delete")
def delete_lab_pdf(
    petId: str = Query(...),
    id: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    record_id = (id or "").strip()
    if not record_id:
        raise HTTPException(status_code=400, detail="id is required")

    deleted = False
    for prefix in _pdf_prefix_candidates(uid, petId, "lab"):
        pdf_key = f"{prefix}{record_id}.pdf"
        meta_key = f"{prefix}{record_id}.json"

        try:
            delete_from_s3(pdf_key)
            deleted = True
        except HTTPException as e:
            if e.status_code != 404:
                raise
        delete_from_s3_if_exists(meta_key)

        if deleted:
            return {"ok": True, "deletedKey": pdf_key}

    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")


@app.delete("/cert/delete")
@app.delete("/certs/delete")
@app.delete("/certificates/delete")
@app.delete("/api/cert/delete")
@app.delete("/api/certs/delete")
@app.delete("/api/certificates/delete")
def delete_cert_pdf(
    petId: str = Query(...),
    id: str = Query(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    uid = user.get("uid") or "unknown"
    record_id = (id or "").strip()
    if not record_id:
        raise HTTPException(status_code=400, detail="id is required")

    deleted = False
    for prefix in _pdf_prefix_candidates(uid, petId, "cert"):
        pdf_key = f"{prefix}{record_id}.pdf"
        meta_key = f"{prefix}{record_id}.json"

        try:
            delete_from_s3(pdf_key)
            deleted = True
        except HTTPException as e:
            if e.status_code != 404:
                raise
        delete_from_s3_if_exists(meta_key)

        if deleted:
            return {"ok": True, "deletedKey": pdf_key}

    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")


# ------------------------------------------------
# 11. AI 케어 (태그 매칭/정규화 강화)
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


def _resolve_record_tags(raw_tags: Any) -> List[str]:
    """
    record.tags 배열을 받아서:
    - code 정규화
    - alias/camel/hyphen 등 최대한 CONDITION_TAGS로 resolve
    - 최종 canonical code list 반환
    """
    if not raw_tags:
        return []
    if not isinstance(raw_tags, list):
        return []

    out: List[str] = []
    for t in raw_tags:
        if not t:
            continue
        t_str = str(t).strip()
        if not t_str:
            continue

        cfg = _resolve_tag_config(t_str)
        if cfg:
            out.append(getattr(cfg, "code", t_str))
            continue

        # 그래도 못 찾으면, snake로 정규화된 코드 자체를 넣어두되(후속 분석/로그용)
        out.append(_normalize_tag_code(t_str))

    # 빈값 제거 + 중복 제거(순서 유지)
    seen: Set[str] = set()
    cleaned: List[str] = []
    for c in out:
        c = (c or "").strip()
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        cleaned.append(c)

    return cleaned


def _build_tag_stats(
    medical_history: List[Dict[str, Any]],
    species: str = "both",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    today = date.today()

    agg: Dict[str, Dict[str, Any]] = {}
    period_stats: Dict[str, Dict[str, int]] = {"1m": {}, "3m": {}, "1y": {}}

    for mh in medical_history:
        visit_str = mh.get("visitDate") or mh.get("visit_date") or ""
        visit_dt = _parse_visit_date(visit_str)
        visit_date_str = visit_dt.isoformat() if visit_dt else None

        raw_tags = mh.get("tags") or []
        resolved = _resolve_record_tags(raw_tags)

        # ✅ tags가 없거나, 전부 unknown(=CONDITION_TAGS resolve 안 된 것들)인 경우 텍스트에서 보강
        # 여기서 "unknown" 판단 기준: resolve된 코드가 CONDITION_TAGS에 실제로 존재하는 canonical로 변환되는지
        has_known = False
        for code in resolved:
            cfg = _resolve_tag_config(code)
            if cfg:
                has_known = True
                break

        if not has_known:
            inferred = _infer_tags_from_text(mh, species=species, limit=settings.TAG_INFERENCE_MAX_TAGS)
            # inferred는 canonical code들
            if inferred:
                resolved = inferred

        if not resolved:
            continue

        used_codes = set()

        for code in resolved:
            cfg = _resolve_tag_config(code)
            if not cfg:
                # unknown이면 건너뜀 (통계/가이드에 쓰기 애매)
                continue

            canonical_code = getattr(cfg, "code", code)
            if canonical_code in used_codes:
                continue
            used_codes.add(canonical_code)

            stat = agg.setdefault(
                canonical_code,
                {
                    "tag": canonical_code,
                    "label": getattr(cfg, "label", canonical_code),
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
당신은 반려동물 건강 기록을 '정리/요약'해 주는 어시스턴트입니다.

[중요 규칙]
1) 아래 '태그 통계'에 없는 질환/컨디션을 새로 추정하거나 만들어내지 마세요.
2) group=exam/medication/procedure/preventive/wellness 태그는 '질환 진단'이 아니라 검사/처방/시술/예방/기록 분류입니다. 이를 근거로 질환을 단정하지 마세요.
3) 태그는 보호자가 저장한 tags가 기본이며, tags가 비어있는 일부 기록은 메모/항목 텍스트에서 키워드 매칭으로 분류된(비진단 위주) 태그가 포함될 수 있습니다.
4) 의료적 진단/판단을 하지 말고, 보호자 관점의 정리/안심/현실적인 관리 팁만 제공하세요.
5) 출력은 마크다운 없이 '문장만' 출력합니다. 불릿/번호/따옴표/코드블록 금지.

[반려동물 기본 정보]
이름: {pet_name}
종: {species}
나이: {age_text or '정보 없음'}
현재 체중: {weight if weight is not None else '정보 없음'} kg

[태그 통계]
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

    species_raw = (profile.get("species") or "dog")
    species_norm = str(species_raw).strip().lower()
    if species_norm not in ("dog", "cat"):
        species_norm = "both"

    medical_history = body.get("medicalHistory") or body.get("medical_history") or []
    has_history = len(medical_history) > 0

    tags, period_stats = _build_tag_stats(medical_history, species=species_norm)

    care_guide: Dict[str, List[str]] = {}
    for t in tags:
        code = t["tag"]
        cfg = _resolve_tag_config(code)
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
                "아직 태그가 확인되는 기록이 충분하지 않아서 컨디션별 통계를 만들기 어려워요. "
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


