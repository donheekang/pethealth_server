# ocr_policy.py
# PetHealth+ OCR Policy (A안: minimal redaction + heuristic parse + optional AI fallback)
# - Google Vision OCR로 word bbox를 얻고
# - "영수증 전체 footer wipe" 같은 과격한 마스킹을 기본적으로 하지 않음
# - PII 최소 마스킹(고객명/보호자/카드번호 등) + 진단/항목/금액 영역은 최대한 보존
# - OCR 원문 전체 텍스트는 반환/저장하지 않는 것을 권장 (필요하면 safe_lines만 내부적으로 사용)

from __future__ import annotations

import io
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFile, ImageFilter, ImageOps, UnidentifiedImageError

# Optional: Google Vision
try:
    from google.cloud import vision  # type: ignore
except Exception:  # pragma: no cover
    vision = None  # type: ignore


# =========================================================
# Config
# =========================================================

@dataclass(frozen=True)
class OcrPolicyConfig:
    # image handling
    receipt_max_width: int = 1024
    receipt_webp_quality: int = 85
    image_max_pixels: int = 20_000_000  # Pillow decompression bomb guard

    # OCR runtime
    ocr_timeout_seconds: float = 12.0
    ocr_max_concurrency: int = 4
    ocr_sema_acquire_timeout_seconds: float = 1.0  # 폭주 시 빠른 실패

    # redaction policy
    # - "minimal": 고객/보호자/이름/카드/이메일 같은 강한 PII만
    # - "standard": + (사업자번호/전화번호)도 마스킹
    # - "aggressive": + 주소/승인번호/긴 숫자열 등도 마스킹
    redaction_level: str = "minimal"  # minimal | standard | aggressive

    # style: "blur" or "black"
    redaction_style: str = "blur"

    # blur strength (only for blur style)
    blur_radius: int = 10

    # parsing
    max_items: int = 60

    # Optional AI fallback (Gemini 등) — 설치/키 없으면 자동으로 무시됨
    enable_ai_fallback: bool = False
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"


class OcrPolicyError(RuntimeError):
    pass


# =========================================================
# Regex / Keywords
# =========================================================

# Strong PII patterns
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_RE_CARDLIKE = re.compile(r"(?:\d[\-\s]?){13,19}")  # 13~19 digits with separators
_RE_APPROVAL = re.compile(r"(승인|승인번호|approval|auth)[^\d]{0,12}\d{3,}", re.IGNORECASE)

# KR business number (사업자등록번호)
_RE_BIZNO = re.compile(r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{5}\b")

# KR phone-ish (휴대폰/전화)
_RE_PHONE = re.compile(
    r"(01[016789][\-\s]?\d{3,4}[\-\s]?\d{4})|(0\d{1,2}[\-\s]?\d{3,4}[\-\s]?\d{4})"
)

# address-ish (주의: 기본 minimal에서는 이걸로 마스킹하지 않음)
_ADDR_KEYWORDS = ["주소", "도로명", "지번", "우편", "시", "군", "구", "읍", "면", "동", "로", "길", "번길"]

_OWNER_LINE_KEYWORDS = [
    "고객", "고객명", "성명", "이름", "보호자", "연락처", "휴대", "핸드폰",
    "patient", "owner", "name",
]

# money parsing
_MONEY_HINT = re.compile(r"(₩|원|krw)", re.IGNORECASE)
_MONEY_NUM = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")

_TOTAL_KEYS = [
    "합계", "총액", "총 금액", "총금액", "결제", "청구", "청구금액", "결제금액", "결제요청", "TOTAL", "AMOUNT"
]
_EXCLUDE_ITEM_KEYS = [
    "소계", "소 계", "합계", "총액", "총 금액", "총금액",
    "부가세", "과세", "면세", "거스름", "결제", "청구", "승인", "vat",
]


# =========================================================
# Concurrency (process-wide)
# =========================================================

_SEMA_LOCK = threading.Lock()
_SEMA: Optional[threading.BoundedSemaphore] = None
_SEMA_MAX: int = 0


def _get_ocr_sema(max_concurrency: int) -> threading.BoundedSemaphore:
    global _SEMA, _SEMA_MAX
    mc = max(1, int(max_concurrency or 1))
    with _SEMA_LOCK:
        if _SEMA is None or _SEMA_MAX != mc:
            _SEMA = threading.BoundedSemaphore(mc)
            _SEMA_MAX = mc
        return _SEMA


# =========================================================
# Google Vision client
# =========================================================

_VISION_LOCK = threading.Lock()
_VISION_CLIENT = None


def _normalize_private_key_newlines(info: dict) -> dict:
    if not isinstance(info, dict):
        return info
    pk = info.get("private_key")
    if isinstance(pk, str) and "\\n" in pk:
        info["private_key"] = pk.replace("\\n", "\n")
    return info


def get_vision_client(google_application_credentials: str):
    """
    google_application_credentials:
      - JSON string or
      - file path to service account json
    """
    global _VISION_CLIENT
    if vision is None:
        raise OcrPolicyError("google-cloud-vision is not installed. (pip install google-cloud-vision)")

    cred_value = (google_application_credentials or "").strip()
    if not cred_value:
        raise OcrPolicyError("GOOGLE_APPLICATION_CREDENTIALS is empty")

    with _VISION_LOCK:
        if _VISION_CLIENT is not None:
            return _VISION_CLIENT

        # 1) JSON string
        try:
            info = json.loads(cred_value)
            if isinstance(info, dict):
                info = _normalize_private_key_newlines(info)
                _VISION_CLIENT = vision.ImageAnnotatorClient.from_service_account_info(info)
                return _VISION_CLIENT
        except json.JSONDecodeError:
            pass

        # 2) file path
        if not os.path.exists(cred_value):
            raise OcrPolicyError("GOOGLE_APPLICATION_CREDENTIALS is neither JSON string nor an existing file path")

        _VISION_CLIENT = vision.ImageAnnotatorClient.from_service_account_file(cred_value)
        return _VISION_CLIENT


# =========================================================
# OCR (words + bboxes)
# =========================================================

def run_vision_ocr_words(
    image_bytes: bytes,
    *,
    google_application_credentials: str,
    timeout_s: float,
    max_concurrency: int,
    sema_acquire_timeout_s: float,
) -> List[Dict[str, Any]]:
    """
    Returns list of {text, bbox=(x1,y1,x2,y2)}.
    DO NOT persist raw OCR text.
    """
    client = get_vision_client(google_application_credentials)
    img = vision.Image(content=image_bytes)

    sema = _get_ocr_sema(max_concurrency)
    acquired = sema.acquire(timeout=max(0.1, float(sema_acquire_timeout_s or 0.1)))
    if not acquired:
        raise OcrPolicyError("OCR busy: too many concurrent OCR requests")

    try:
        # text_detection is usually enough for receipts (word boxes)
        try:
            resp = client.text_detection(image=img, timeout=float(timeout_s))
        except TypeError:
            resp = client.text_detection(image=img)
    finally:
        sema.release()

    if getattr(resp, "error", None) and getattr(resp.error, "message", ""):
        raise OcrPolicyError(f"OCR error: {resp.error.message}")

    anns = resp.text_annotations or []
    out: List[Dict[str, Any]] = []

    # anns[0] is full text; skip it
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
        out.append({"text": desc, "bbox": (int(x1), int(y1), int(x2), int(y2))})

    return out


# =========================================================
# Group words into lines
# =========================================================

def _group_words_into_lines(words: List[Dict[str, Any]], img_h: int) -> List[Dict[str, Any]]:
    """
    Rough line clustering by y-center proximity.
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
    threshold = max(8, int(img_h * 0.012))  # ~1.2% height

    lines: List[Dict[str, Any]] = []
    for it in items:
        placed = False
        for ln in lines:
            if abs(it["cy"] - ln["cy"]) <= threshold:
                ln["words"].append(it)
                # update avg center
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


# =========================================================
# Redaction policy (A안: minimal)
# =========================================================

def _looks_like_address_line(line_text: str) -> bool:
    t = (line_text or "").strip()
    if not t:
        return False
    has_kw = any(k in t for k in _ADDR_KEYWORDS)
    if has_kw and re.search(r"\d", t):
        return True
    return False


def _contains_owner_keyword(t: str) -> bool:
    low = (t or "").lower()
    return any(k.lower() in low for k in _OWNER_LINE_KEYWORDS)


def _should_redact_line(text: str, level: str) -> Tuple[bool, List[str]]:
    """
    Returns (should_redact, reasons)
    """
    t = (text or "").strip()
    if not t:
        return False, []

    lvl = (level or "minimal").strip().lower()
    reasons: List[str] = []

    # Always strong patterns
    if _RE_EMAIL.search(t):
        reasons.append("email")
    if _RE_CARDLIKE.search(t):
        reasons.append("cardlike")
    if _RE_APPROVAL.search(t):
        reasons.append("approval")

    # Owner line keyword => strong PII
    if _contains_owner_keyword(t):
        reasons.append("owner_kw")

    # standard/aggressive: add more
    if lvl in ("standard", "aggressive"):
        if _RE_BIZNO.search(t):
            reasons.append("bizno")
        if _RE_PHONE.search(t):
            reasons.append("phone")

    if lvl == "aggressive":
        if _looks_like_address_line(t):
            reasons.append("address")
        # long digit runs (non-money) — aggressive only
        compact = re.sub(r"[,\s\-]", "", t)
        if re.search(r"\d{8,}", compact) and not _MONEY_HINT.search(t):
            reasons.append("long_digits")

    return (len(reasons) > 0), reasons


def _pad_box(box: Tuple[int, int, int, int], img_w: int, img_h: int, pad_x: int, pad_y: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    )


def _merge_boxes(boxes: List[Tuple[int, int, int, int]], *, gap: int = 6) -> List[Tuple[int, int, int, int]]:
    """
    Merge overlapping or very-close boxes to reduce patchiness.
    Does NOT create a giant footer wipe; only merges near overlaps.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[Tuple[int, int, int, int]] = []

    def overlaps(a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        # allow small gap
        return not (ax2 < bx1 - gap or bx2 < ax1 - gap or ay2 < by1 - gap or by2 < ay1 - gap)

    cur = boxes[0]
    for b in boxes[1:]:
        if overlaps(cur, b):
            cur = (min(cur[0], b[0]), min(cur[1], b[1]), max(cur[2], b[2]), max(cur[3], b[3]))
        else:
            merged.append(cur)
            cur = b
    merged.append(cur)
    return merged


def compute_redaction_boxes(lines: List[Dict[str, Any]], img_w: int, img_h: int, *, level: str) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    ✅ A안 기본: line 단위 마스킹만, footer 전체 wipe 없음.
    """
    pad_x = max(10, int(img_w * 0.01))
    pad_y = max(6, int(img_h * 0.008))

    boxes: List[Tuple[int, int, int, int]] = []
    reasons_count: Dict[str, int] = {}

    for ln in lines:
        text = ln.get("text") or ""
        bbox = ln.get("bbox")
        if not bbox:
            continue

        should, reasons = _should_redact_line(text, level)
        if not should:
            continue

        for r in reasons:
            reasons_count[r] = reasons_count.get(r, 0) + 1

        boxes.append(_pad_box(tuple(map(int, bbox)), img_w, img_h, pad_x, pad_y))

    boxes = _merge_boxes(boxes, gap=max(6, int(img_w * 0.006)))

    debug = {
        "redactionLevel": level,
        "lineCount": len(lines),
        "redactedBoxCount": len(boxes),
        "reasonsCount": reasons_count,
    }
    return boxes, debug


def apply_redaction(img: Image.Image, boxes: List[Tuple[int, int, int, int]], *, style: str, blur_radius: int) -> Image.Image:
    if not boxes:
        return img

    out = img.copy()
    st = (style or "blur").strip().lower()

    if st == "black":
        draw = ImageDraw.Draw(out)
        for (x1, y1, x2, y2) in boxes:
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        return out

    # default: blur
    radius = int(blur_radius or 10)
    radius = max(4, min(radius, 30))
    for (x1, y1, x2, y2) in boxes:
        region = out.crop((x1, y1, x2, y2))
        region = region.filter(ImageFilter.GaussianBlur(radius=radius))
        out.paste(region, (x1, y1))
    return out


# =========================================================
# Image helpers
# =========================================================

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


def _safe_open_image(raw: bytes, *, image_max_pixels: int) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = int(image_max_pixels or 20_000_000)
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # orientation fix
        img = img.convert("RGB")  # strip alpha/exif
        return img
    except Image.DecompressionBombError:
        raise OcrPolicyError("image too large (decompression bomb)")
    except UnidentifiedImageError:
        raise OcrPolicyError("invalid image")
    except Exception as e:
        raise OcrPolicyError(f"image open failed: {e}")


# =========================================================
# Parsing (heuristics)
# =========================================================

def _parse_visit_date_from_text(text: str) -> Optional[date]:
    t = (text or "").strip()
    if not t:
        return None
    # 2025-11-28 / 2025.11.28 / 2025년 11월 28일
    m = re.search(r"(20\d{2})[.\-\/년 ]+(\d{1,2})[.\-\/월 ]+(\d{1,2})", t)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return date(y, mo, d)
    except Exception:
        return None


def extract_visit_date(lines: List[str]) -> Optional[date]:
    for ln in (lines or [])[:80]:
        vd = _parse_visit_date_from_text(ln)
        if vd:
            return vd
    return None


def extract_hospital_name(lines: List[str]) -> Optional[str]:
    """
    우선순위:
      1) "병원명:" 라인 파싱
      2) 상단 20줄에서 "동물병원/동물의원" 포함한 가장 그럴듯한 라인
    """
    if not lines:
        return None

    # 1) explicit key
    for ln in lines[:40]:
        compact = ln.replace(" ", "")
        if "병원명" in compact or "병원" in compact and ("병원명" in compact or "병원명:" in compact):
            # split by ":" or "："
            parts = re.split(r"[:：]", ln, maxsplit=1)
            if len(parts) == 2:
                name = parts[1].strip()
                if 2 <= len(name) <= 60:
                    return name

        # e.g. "병 원 명 : 해랑동물병원"
        if re.search(r"병\s*원\s*명", ln) and (":" in ln or "：" in ln):
            parts = re.split(r"[:：]", ln, maxsplit=1)
            if len(parts) == 2:
                name = parts[1].strip()
                if 2 <= len(name) <= 60:
                    return name

    # 2) heuristic
    keywords = [
        "동물병원", "동물 병원", "동물의원", "동물 의원", "동물메디컬", "동물 메디컬",
        "animal hospital", "vet", "veterinary",
    ]
    best = None
    best_score = -10_000

    for idx, ln in enumerate(lines[:25]):
        s = (ln or "").strip()
        if not s:
            continue
        score = 0
        compact = s.replace(" ", "").lower()

        if any(k.replace(" ", "").lower() in compact for k in keywords):
            score += 20

        # header area bonus
        if idx <= 4:
            score += 6

        # too many digits penalty
        digit_count = sum(c.isdigit() for c in s)
        if digit_count >= 8:
            score -= 6

        # exclude obvious non-name
        if any(x in s for x in ["청구서", "영수증", "소계", "합계", "결제", "사업자", "전화", "TEL", "주소"]):
            score -= 6

        if 2 <= len(s) <= 40:
            score += 2
        else:
            score -= 2

        if score > best_score:
            best_score = score
            best = s

    if best and best_score >= 8:
        return best
    return None


def _extract_total_amount(lines: List[str]) -> Optional[int]:
    """
    total 후보 라인에서 가장 큰 금액을 total로 선택.
    """
    best: Optional[int] = None
    for ln in lines[:120]:
        t = (ln or "").strip()
        if not t:
            continue

        up = t.upper()
        if not any(k.upper() in up for k in _TOTAL_KEYS):
            continue

        nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d{3,}", t)
        for raw in nums:
            try:
                n = int(raw.replace(",", ""))
            except Exception:
                continue
            if best is None or n > best:
                best = n

    return best


def _line_is_item_candidate(line_text: str) -> bool:
    t = (line_text or "").strip()
    if not t:
        return False
    # must contain a number, ideally money hint or "reasonable" last number
    if not _MONEY_NUM.search(t):
        return False

    # exclude totals/taxes
    low = t.lower()
    for k in _EXCLUDE_ITEM_KEYS:
        if k.lower() in low:
            return False

    # avoid masking lines (business/phone etc) as items
    if _contains_owner_keyword(t) or _RE_BIZNO.search(t) or _RE_PHONE.search(t) or _RE_EMAIL.search(t):
        return False

    return True


def _extract_items(lines: List[str], *, max_items: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ln in (lines or [])[:200]:
        if len(out) >= max_items:
            break
        if not _line_is_item_candidate(ln):
            continue

        nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d{3,}", ln)
        if not nums:
            continue

        # price: take last number
        price_raw = nums[-1].replace(",", "")
        try:
            price = int(price_raw)
        except Exception:
            price = None

        # name: remove last number + currency hints
        name = ln
        name = re.sub(re.escape(nums[-1]), "", name, count=1).strip()
        name = re.sub(r"(₩|원|KRW)", "", name, flags=re.IGNORECASE).strip()
        # remove qty-like trailing numbers (e.g. "30,000 1 30,000" -> remove "30,000 1")
        name = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", " ", name).strip()
        name = re.sub(r"\s{2,}", " ", name).strip()
        name = name.lstrip("*").strip()

        if not name or len(name) < 1:
            continue

        out.append({"itemName": name[:200], "price": price, "categoryTag": None})
    return out


def _guess_address_hint(lines_all: List[str]) -> Optional[str]:
    """
    hospital 후보검색 보정용 힌트. (저장/응답 금지 권장)
    """
    best = None
    best_score = -999
    for idx, line in enumerate((lines_all or [])[:80]):
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

        if score > best_score:
            best_score = score
            best = t

    if not best:
        return None

    best = re.sub(r"^(주소|도로명주소|지번주소)\s*[:：]?\s*", "", best).strip()
    return best[:200] if best else None


def build_safe_lines_for_parsing(lines: List[str], *, redaction_level: str) -> List[str]:
    """
    AI fallback/후속 처리용 "safe lines" (PII 가능성이 높은 줄 제거)
    - 이 결과를 DB에 저장하지 않는 걸 권장
    """
    out: List[str] = []
    for ln in (lines or [])[:200]:
        should, _ = _should_redact_line(ln, redaction_level)
        if should:
            continue
        out.append(ln)
    return out


# =========================================================
# Optional AI fallback (Gemini)
# =========================================================

def _try_parse_with_gemini(
    safe_lines: List[str],
    *,
    api_key: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    If google-generativeai is installed and api_key is provided,
    ask Gemini to return JSON:
      {hospitalName, visitDate(YYYY-MM-DD), totalAmount(int), items:[{itemName, price}]}
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return None

    if not api_key:
        return None

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model or "gemini-1.5-flash")

    # Keep prompt short and strict JSON-only
    text = "\n".join(safe_lines[:120])

    prompt = f"""
You are parsing a Korean veterinary receipt OCR lines.
Return ONLY valid JSON (no markdown).
Schema:
{{
  "hospitalName": string|null,
  "visitDate": string|null,   // YYYY-MM-DD
  "totalAmount": number|null, // integer KRW
  "items": [{{"itemName": string, "price": number|null}}]
}}
Rules:
- If uncertain, use null.
- Do not include any personal info like customer name/phone/address.
- Extract only medical/service item names and their prices if possible.
OCR lines:
{text}
""".strip()

    try:
        resp = m.generate_content(prompt)
        raw = (getattr(resp, "text", None) or "").strip()
        if not raw:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


# =========================================================
# Main entry
# =========================================================

def process_receipt_image_and_parse(
    raw_image_bytes: bytes,
    *,
    google_application_credentials: str,
    config: Optional[OcrPolicyConfig] = None,
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      - redacted webp bytes
      - parsed dict (safe to return to client)
      - hints dict (do NOT store; used internally e.g. hospital candidate search)
    """
    cfg = config or OcrPolicyConfig()

    # open image
    img = _safe_open_image(raw_image_bytes, image_max_pixels=cfg.image_max_pixels)
    w, h = img.size

    # OCR words
    words = run_vision_ocr_words(
        raw_image_bytes,
        google_application_credentials=google_application_credentials,
        timeout_s=cfg.ocr_timeout_seconds,
        max_concurrency=cfg.ocr_max_concurrency,
        sema_acquire_timeout_s=cfg.ocr_sema_acquire_timeout_seconds,
    )

    # lines
    lines_struct = _group_words_into_lines(words, img_h=h)
    all_lines_text = [ln.get("text", "") for ln in lines_struct if ln.get("text")]

    # redaction boxes (A안 기본: footer wipe 없음)
    boxes, redaction_debug = compute_redaction_boxes(
        lines_struct, img_w=w, img_h=h, level=cfg.redaction_level
    )

    redacted_img = apply_redaction(
        img, boxes, style=cfg.redaction_style, blur_radius=cfg.blur_radius
    )

    # resize + encode
    redacted_small = _resize_to_width(redacted_img, int(cfg.receipt_max_width))
    webp_bytes = _encode_webp(redacted_small, int(cfg.receipt_webp_quality))

    # safe lines for parsing (drop PII-ish lines)
    safe_lines = build_safe_lines_for_parsing(all_lines_text, redaction_level=cfg.redaction_level)

    # heuristics parse
    hospital_name = extract_hospital_name(safe_lines)
    visit_dt = extract_visit_date(safe_lines)
    total_amount = _extract_total_amount(safe_lines)
    items = _extract_items(safe_lines, max_items=int(cfg.max_items))

    parsed: Dict[str, Any] = {
        "hospitalName": hospital_name or None,
        "visitDate": visit_dt.isoformat() if visit_dt else None,
        "totalAmount": int(total_amount) if isinstance(total_amount, int) else None,
        "items": items,
    }

    # Optional AI fallback (only if enabled and heuristics are weak)
    if cfg.enable_ai_fallback:
        need_ai = False
        if not parsed["hospitalName"]:
            need_ai = True
        if not parsed["visitDate"]:
            need_ai = True
        if parsed["totalAmount"] is None:
            need_ai = True
        if not parsed["items"]:
            need_ai = True

        if need_ai:
            api_key = cfg.gemini_api_key or os.getenv("GEMINI_API_KEY", "").strip()
            ai = _try_parse_with_gemini(safe_lines, api_key=api_key, model=cfg.gemini_model)
            if isinstance(ai, dict):
                # merge carefully (AI가 이상하면 기존 heuristic 유지)
                hn = ai.get("hospitalName")
                vd = ai.get("visitDate")
                ta = ai.get("totalAmount")
                its = ai.get("items")

                if not parsed["hospitalName"] and isinstance(hn, str) and hn.strip():
                    parsed["hospitalName"] = hn.strip()[:80]

                if not parsed["visitDate"] and isinstance(vd, str) and re.match(r"^20\d{2}-\d{2}-\d{2}$", vd.strip()):
                    parsed["visitDate"] = vd.strip()

                if parsed["totalAmount"] is None:
                    try:
                        if ta is not None:
                            parsed["totalAmount"] = int(ta)
                    except Exception:
                        pass

                if not parsed["items"] and isinstance(its, list):
                    cleaned_items: List[Dict[str, Any]] = []
                    for it in its[:cfg.max_items]:
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
                        cleaned_items.append({"itemName": nm[:200], "price": pr, "categoryTag": None})
                    parsed["items"] = cleaned_items

    # hints (do not store)
    hints: Dict[str, Any] = {
        "addressHint": _guess_address_hint(all_lines_text),  # 저장/응답은 금지 권장(내부 후보검색용)
        "redactionDebug": redaction_debug,                   # 내용 없이 통계만
        # safeLines는 "내부 처리용"일 때만 쓰는 걸 추천 (서버 로그/DB 저장 금지)
        # "safeLines": safe_lines,
    }

    return webp_bytes, parsed, hints


