# ocr_policy.py (PetHealth+)
# OCR + minimal redaction + receipt parsing
# Returns: (webp_bytes, parsed_dict, hints_dict)

from __future__ import annotations

import os
import io
import re
import json
import math
import base64
import threading
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Regex / constants
# -----------------------------
_DATE_RE_1 = re.compile(r"\b(20\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})\b")
_DATE_RE_2 = re.compile(r"\b(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\b")

# NOTE:
# - 콤마 금액(30,000) OR 3+ digits(3000+)는 금액 후보로 잡되,
# - '날짜: 2025-11-28' 같은 라인은 _is_noise_line에서 먼저 제거하도록 강화함.
_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")

_HOSP_RE = re.compile(r"(병원\s*명|원\s*명)\s*[:：]?\s*(.+)$")

# PII patterns for image redaction (best-effort)
_PHONE_RE = re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b")
_BIZ_RE = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")
_CARD_RE = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")

# Rabies keyword (OCR typo tolerant)
_RABIES_RE = re.compile(
    r"(rabies|rabbies|rabie|rabis|rabiess|rables|rabeis|ra\b|r\s*[/\-\._ ]\s*a\b|광견병|광견)",
    re.IGNORECASE,
)

# -----------------------------
# Noise tokens (for filtering non-item lines)
# -----------------------------
_NOISE_TOKENS = [
    # customer / admin
    "고객", "고객번호", "고객 번호", "고객명", "고객 이름",
    "사업자", "사업자등록", "사업자 등록", "대표",
    "전화", "연락처", "주소",
    "serial", "sign", "승인", "카드", "현금",
    "부가세", "vat", "면세", "과세",
    "공급가", "공급가액", "과세공급가액", "부가세액",

    # totals / payment lines
    "소계", "합계", "총", "총액", "총 금액", "총금액",
    "청구", "청구금액", "결제", "결제요청", "결제요청:", "결제예정",
    "거래", "거래일", "거래 일",

    # date-ish labels (✅ 핵심: '날짜 2025...'가 item으로 잡히는 문제 방지)
    "날짜", "일자", "방문일", "방문 일", "진료일", "진료 일", "발행", "발행일", "발행 일",
    "입원", "퇴원",

    # headers / table labels
    "항목", "단가", "수량", "금액",
    "동물명", "환자", "환자명", "품종",
]

def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", (s or "").lower())

_NOISE_TOKENS_NORM = [_norm(x) for x in _NOISE_TOKENS if _norm(x)]


def _looks_like_date_line(t: str) -> bool:
    """
    날짜 라인(예: '날짜: 2025-11-28')은 금액처럼 보이는 2025가 잡혀서
    item으로 잘못 들어가는 원인이 됨 -> 강하게 제거.
    """
    if not t:
        return False
    if _DATE_RE_1.search(t) or _DATE_RE_2.search(t):
        return True

    # date-only-ish: "2025-11-28" 단독 라인도 제거
    k = _norm(t)
    if re.fullmatch(r"20\d{2}(0?\d|1[0-2])(0?\d|[12]\d|3[01])", k):
        return True
    if re.fullmatch(r"20\d{2}(0?\d|1[0-2])(0?\d|[12]\d|3[01])\d{0,6}", k) and len(k) <= 12:
        # ex) "20251128" 같은 형태
        return True
    return False


def _is_noise_line(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True

    low = t.lower()
    k = _norm(t)

    if len(k) < 2:
        return True

    # ✅ 날짜 라인 강제 제거
    if _looks_like_date_line(t):
        return True

    # no letters and only numbers/separators -> noise
    has_letter = any(("a" <= ch.lower() <= "z") or ("가" <= ch <= "힣") for ch in t)
    has_digit = any(ch.isdigit() for ch in t)
    if has_digit and (not has_letter) and len(k) <= 12:
        return True

    for xn in _NOISE_TOKENS_NORM:
        if xn and xn in k:
            return True

    # also handle some raw english keywords w/ punctuation
    for w in ("serial", "sign", "vat"):
        if w in low:
            return True

    return False


def _parse_date_from_text(text: str) -> Optional[date]:
    if not text:
        return None
    m = _DATE_RE_1.search(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except Exception:
            pass
    m = _DATE_RE_2.search(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except Exception:
            pass
    return None


def _coerce_int_amount(s: Any) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, bool):
        return None
    if isinstance(s, int):
        return int(s)
    if isinstance(s, float):
        try:
            return int(s)
        except Exception:
            return None
    if isinstance(s, str):
        m = _AMOUNT_RE.search(s.replace(" ", ""))
        if not m:
            return None
        try:
            return int(m.group(0).replace(",", ""))
        except Exception:
            return None
    return None


# -----------------------------
# Concurrency guard
# -----------------------------
_SEMA_LOCK = threading.Lock()
_SEMA_BY_N: Dict[int, threading.BoundedSemaphore] = {}


def _get_sema(n: int) -> threading.BoundedSemaphore:
    nn = max(1, min(int(n or 1), 32))
    with _SEMA_LOCK:
        if nn not in _SEMA_BY_N:
            _SEMA_BY_N[nn] = threading.BoundedSemaphore(nn)
        return _SEMA_BY_N[nn]


# -----------------------------
# Image processing (PIL)
# -----------------------------
def _load_pil():
    try:
        from PIL import Image, ImageOps, ImageDraw
        return Image, ImageOps, ImageDraw
    except Exception as e:
        raise RuntimeError("Pillow is required for receipt image processing. Install: pip install pillow") from e


def _ensure_max_pixels(img, max_pixels: int):
    w, h = img.size
    if max_pixels and (w * h) > int(max_pixels):
        scale = math.sqrt(float(max_pixels) / float(w * h))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        img = img.resize((nw, nh))
    return img


def _resize_to_width(img, max_width: int):
    if not max_width:
        return img
    w, h = img.size
    if w <= int(max_width):
        return img
    scale = float(max_width) / float(w)
    nw = int(max_width)
    nh = max(1, int(h * scale))
    return img.resize((nw, nh))


def _to_webp_bytes(img, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=int(max(30, min(int(quality or 85), 95))), method=6)
    return buf.getvalue()


# -----------------------------
# Google Vision OCR (optional)
# -----------------------------
def _build_vision_client(google_credentials: str):
    # import locally (avoid import-time crash)
    from google.cloud import vision
    from google.oauth2 import service_account

    gc = (google_credentials or "").strip()
    if not gc:
        # try env fallback (typo-safe)
        env1 = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        env2 = (os.getenv("GOOGLE_APPLICATION_CREDENTIALIALS") or "").strip()  # common typo in consoles
        gc = env1 or env2

    if not gc:
        # default credentials (ADC)
        return vision.ImageAnnotatorClient()

    # JSON string?
    if gc.startswith("{") and gc.endswith("}"):
        info = json.loads(gc)
        pk = info.get("private_key")
        if isinstance(pk, str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds)

    # file path?
    if os.path.exists(gc):
        return vision.ImageAnnotatorClient.from_service_account_file(gc)

    # last resort: treat as env var path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc
    return vision.ImageAnnotatorClient()


def _vision_ocr(
    image_bytes: bytes,
    google_credentials: str,
    timeout_seconds: int,
) -> Tuple[str, Any]:
    from google.cloud import vision  # type: ignore

    client = _build_vision_client(google_credentials)
    img = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=img, timeout=float(timeout_seconds or 12))

    full_text = ""
    try:
        if resp and resp.full_text_annotation and resp.full_text_annotation.text:
            full_text = resp.full_text_annotation.text
    except Exception:
        full_text = ""

    return full_text, resp


# -----------------------------
# Redaction (best-effort): mask phone/biz/card tokens
# -----------------------------
def _redact_image_with_vision_tokens(img, vision_response) -> Any:
    Image, ImageOps, ImageDraw = _load_pil()
    if vision_response is None:
        return img

    try:
        anns = getattr(vision_response, "text_annotations", None)
        if not anns or len(anns) <= 1:
            return img

        draw = ImageDraw.Draw(img)
        for a in anns[1:]:
            desc = str(getattr(a, "description", "") or "")
            if not desc:
                continue

            if not (_PHONE_RE.search(desc) or _BIZ_RE.search(desc) or _CARD_RE.search(desc)):
                continue

            poly = getattr(a, "bounding_poly", None)
            if not poly or not getattr(poly, "vertices", None):
                continue

            xs: List[int] = []
            ys: List[int] = []
            for v in poly.vertices:
                x = getattr(v, "x", None)
                y = getattr(v, "y", None)
                if x is None or y is None:
                    continue
                xs.append(int(x))
                ys.append(int(y))
            if not xs or not ys:
                continue

            x0, x1 = max(0, min(xs)), max(xs)
            y0, y1 = max(0, min(ys)), max(ys)
            draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

        return img
    except Exception:
        return img


# -----------------------------
# Text parsing
# -----------------------------
def _clean_item_name(name: str) -> str:
    s = (name or "").strip()
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lstrip("*•·-—").strip()
    return s[:200]


def _canonicalize_item_name(name: str) -> str:
    """
    OCR 흔한 오타/약어를 canonical로 맞춤.
    - Rabbies / Rabies / RA / R/A -> Rabies
    """
    s = (name or "").strip()
    if not s:
        return s

    low = s.lower()

    # Rabies normalize
    if _RABIES_RE.search(s):
        return "Rabies"

    # 흔한 것들 (필요하면 계속 추가)
    # ex) "x ray" -> "X-ray"
    if re.search(r"\bx\s*[- ]?\s*ray\b", low):
        return "X-ray"

    return s


def _extract_items_from_text(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    out: List[Dict[str, Any]] = []
    seen = set()

    for ln in lines:
        if not ln:
            continue
        if _is_noise_line(ln):
            continue

        # item line should contain at least one "real" amount
        nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(ln)]
        nums = [n for n in nums if n >= 100]  # drop tiny garbage
        if not nums:
            continue

        price = nums[-1]

        # remove amounts from line to get name
        name_part = _AMOUNT_RE.sub(" ", ln)

        # remove small standalone qty tokens
        name_part = re.sub(r"\b\d{1,2}\b", " ", name_part)

        name_part = _clean_item_name(name_part)
        if not name_part:
            continue

        # 마지막 방어: '날짜' 같은 게 남아도 여기서 제거
        if _is_noise_line(name_part):
            continue

        name_part = _canonicalize_item_name(name_part)

        key = (_norm(name_part), int(price))
        if key in seen:
            continue
        seen.add(key)

        out.append({"itemName": name_part, "price": int(price), "categoryTag": None})

    return out[:120]


def _extract_total_amount(text: str) -> Optional[int]:
    if not text:
        return None
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    candidates: List[int] = []

    for ln in lines:
        if not ln:
            continue
        low = ln.lower()
        if any(k in low for k in ["합계", "총", "총액", "총금액", "청구", "결제", "소계", "grand", "total"]):
            nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(ln)]
            nums = [n for n in nums if n >= 100]
            candidates.extend(nums)

    # fallback: take max amount anywhere (but still >=100)
    if not candidates:
        nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(text)]
        nums = [n for n in nums if n >= 100]
        candidates = nums

    if not candidates:
        return None
    return int(max(candidates))


def _extract_hospital_name(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:40]:
        m = _HOSP_RE.search(ln)
        if m:
            v = m.group(2).strip()
            v = re.split(r"\s{2,}|/|\||,", v)[0].strip()
            if v:
                return v[:80]
    for ln in lines[:60]:
        if "동물병원" in ln:
            return ln[:80]
    return None


def _extract_address_hint(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if "주소" in ln:
            return ln[:120]
    for ln in lines:
        if any(tok in ln for tok in ["시 ", "시", "구 ", "구", "동 ", "로 ", "길 ", "번지", "도 "]):
            if len(ln) >= 10 and any(ch.isdigit() for ch in ln):
                return ln[:120]
    return None


def _parse_receipt_from_text(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parsed: Dict[str, Any] = {
        "hospitalName": _extract_hospital_name(text),
        "visitDate": None,
        "totalAmount": _extract_total_amount(text),
        "items": [],
        # ✅ main.py가 ocr text를 찾을 수 있게 제공
        "ocrText": (text or "")[:8000],
    }

    vd = _parse_date_from_text(text)
    if vd:
        parsed["visitDate"] = vd.isoformat()

    items = _extract_items_from_text(text)

    # ✅ Rabies 키워드는 있는데 item이 제대로 안 잡힐 때 보강
    low = (text or "").lower()
    has_rabies = bool(_RABIES_RE.search(text or ""))
    if has_rabies:
        # items에 rabies가 하나도 없으면 keyword item 1개라도 넣어줌 (price는 가능한 경우만)
        has_any = any(_RABIES_RE.search(str(it.get("itemName") or "")) for it in (items or []))
        if not has_any:
            ta = parsed.get("totalAmount")
            price = int(ta) if isinstance(ta, int) and ta >= 100 else None
            items = (items or []) + [{"itemName": "Rabies", "price": price, "categoryTag": None}]

    parsed["items"] = items[:120]

    hints: Dict[str, Any] = {
        "addressHint": _extract_address_hint(text),
        "ocrTextPreview": (text or "")[:400],
    }
    return parsed, hints


# -----------------------------
# Gemini (Generative Language API) - best-effort assist
# -----------------------------
def _env_bool(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _call_gemini_generate_content(
    *,
    api_key: str,
    model: str,
    parts: List[Dict[str, Any]],
    timeout_seconds: int = 10,
) -> str:
    import urllib.request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=float(timeout_seconds or 10)) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    j = json.loads(data)
    txt = (
        (j.get("candidates") or [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    return (txt or "").strip()


def _extract_json_from_model_text(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    t = s.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    i = t.find("{")
    j = t.rfind("}")
    if i < 0 or j < 0 or j <= i:
        return None
    blob = t[i : j + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


def _gemini_parse_receipt(
    *,
    image_bytes: bytes,
    ocr_text: str,
    api_key: str,
    model: str,
    timeout_seconds: int = 10,
) -> Optional[Dict[str, Any]]:
    api_key = (api_key or "").strip()
    model = (model or "gemini-2.5-flash").strip()
    if not api_key or not model:
        return None

    prompt = (
        "You are a receipt parser for Korean veterinary receipts.\n"
        "Return ONLY valid JSON with keys:\n"
        '  hospitalName (string|null), visitDate (YYYY-MM-DD|null), totalAmount (integer|null),\n'
        '  items (array of {itemName:string, price:integer|null}).\n'
        "Rules:\n"
        "- items must be REAL treatment/vaccine/medicine line-items.\n"
        "- Do NOT include date lines (e.g. '날짜: 2025-11-28') as items.\n"
        "- Do NOT include totals/taxes/payment lines as items.\n"
        "- If you see 'Rabies' but OCR typo like 'Rabbies' or abbreviation 'RA'/'R/A', normalize itemName to 'Rabies'.\n"
        "- If uncertain, best guess.\n"
    )

    b64 = base64.b64encode(image_bytes).decode("ascii")
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/webp", "data": b64}},
    ]
    if (ocr_text or "").strip():
        parts.append({"text": "OCR text (may be noisy):\n" + (ocr_text or "")[:6000]})

    try:
        out = _call_gemini_generate_content(api_key=api_key, model=model, parts=parts, timeout_seconds=timeout_seconds)
        j = _extract_json_from_model_text(out)
        if isinstance(j, dict):
            return j
    except Exception:
        # fallback: text-only
        try:
            parts2 = [{"text": prompt + "\n\nHere is OCR text:\n" + (ocr_text or "")[:6000]}]
            out = _call_gemini_generate_content(api_key=api_key, model=model, parts=parts2, timeout_seconds=timeout_seconds)
            j = _extract_json_from_model_text(out)
            if isinstance(j, dict):
                return j
        except Exception:
            return None

    return None


def _is_items_suspicious(items: List[Dict[str, Any]]) -> bool:
    if not items:
        return True
    bad = 0
    for it in items[:20]:
        nm = str((it or {}).get("itemName") or "")
        if _is_noise_line(nm):
            bad += 1
    return bad >= max(1, min(3, len(items)))


def _normalize_gemini_parsed(j: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"hospitalName": None, "visitDate": None, "totalAmount": None, "items": []}

    hn = j.get("hospitalName")
    if isinstance(hn, str) and hn.strip():
        out["hospitalName"] = hn.strip()[:80]

    vd = j.get("visitDate")
    if isinstance(vd, str) and vd.strip():
        d = _parse_date_from_text(vd.strip())
        out["visitDate"] = d.isoformat() if d else vd.strip()[:20]

    ta = _coerce_int_amount(j.get("totalAmount"))
    out["totalAmount"] = ta if ta and ta > 0 else None

    items = j.get("items")
    if isinstance(items, list):
        cleaned: List[Dict[str, Any]] = []
        for it in items[:120]:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("itemName") or "").strip()
            if not nm:
                continue
            nm = _canonicalize_item_name(_clean_item_name(nm))
            if not nm or _is_noise_line(nm):
                continue
            pr = _coerce_int_amount(it.get("price"))
            if pr is not None and pr < 0:
                pr = None
            cleaned.append({"itemName": nm, "price": pr, "categoryTag": None})
        out["items"] = cleaned

    return out


# -----------------------------
# Public API
# -----------------------------
def process_receipt(
    raw_bytes: bytes,
    *,
    google_credentials: str = "",
    ocr_timeout_seconds: int = 12,
    ocr_max_concurrency: int = 4,
    ocr_sema_acquire_timeout_seconds: float = 1.0,
    receipt_max_width: int = 1024,
    receipt_webp_quality: int = 85,
    image_max_pixels: int = 20_000_000,
    # Gemini (optional; if not passed, env vars will be used)
    gemini_enabled: Optional[bool] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model_name: Optional[str] = None,
    gemini_timeout_seconds: Optional[int] = None,
    **kwargs,
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      webp_bytes: redacted (best-effort) WEBP bytes for storage
      parsed: {hospitalName, visitDate(YYYY-MM-DD), totalAmount(int), items:[{itemName, price, categoryTag}], ocrText}
      hints: {addressHint, ocrTextPreview, ocrEngine, geminiUsed, ...}
    """
    if not raw_bytes:
        raise ValueError("empty raw bytes")

    Image, ImageOps, ImageDraw = _load_pil()

    # 0) load image
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg

    # 1) resize/harden
    img = _ensure_max_pixels(img, int(image_max_pixels or 0))
    img = _resize_to_width(img, int(receipt_max_width or 0))

    # bytes for OCR (use PNG to preserve text sharpness)
    ocr_buf = io.BytesIO()
    img.save(ocr_buf, format="PNG")
    ocr_image_bytes = ocr_buf.getvalue()

    # 2) OCR with concurrency guard (Vision optional)
    ocr_text = ""
    vision_resp = None
    ocr_engine = "none"
    ocr_error = None

    sema = _get_sema(int(ocr_max_concurrency or 4))
    acquired = sema.acquire(timeout=float(ocr_sema_acquire_timeout_seconds or 1.0))
    if not acquired:
        raise RuntimeError("OCR is busy (semaphore acquire timeout)")

    try:
        try:
            ocr_text, vision_resp = _vision_ocr(
                ocr_image_bytes,
                google_credentials=google_credentials,
                timeout_seconds=int(ocr_timeout_seconds or 12),
            )
            ocr_engine = "google_vision"
        except Exception as e:
            # Vision이 없거나 credential 문제면 여기로 떨어짐.
            ocr_error = str(e)[:200]
            ocr_text = ""
            vision_resp = None
            ocr_engine = "none"
    finally:
        try:
            sema.release()
        except Exception:
            pass

    # 3) redact image (best-effort)
    redacted = _redact_image_with_vision_tokens(img.copy(), vision_resp)
    webp_bytes = _to_webp_bytes(redacted, quality=int(receipt_webp_quality or 85))

    # 4) parse from OCR text
    parsed, hints = _parse_receipt_from_text(ocr_text or "")
    hints["ocrEngine"] = ocr_engine
    hints["geminiUsed"] = False
    if ocr_error:
        hints["ocrError"] = ocr_error

    # 5) Gemini assist (more aggressive for robustness)
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))

    need_ai = False
    items_now = parsed.get("items") if isinstance(parsed.get("items"), list) else []
    if not ocr_text.strip():
        need_ai = True
    if _is_items_suspicious(items_now):
        need_ai = True
    if parsed.get("hospitalName") is None:
        need_ai = True
    if parsed.get("totalAmount") is None:
        need_ai = True

    if g_enabled and g_key.strip() and need_ai:
        try:
            gj = _gemini_parse_receipt(
                image_bytes=webp_bytes,
                ocr_text=ocr_text or "",
                api_key=g_key,
                model=g_model,
                timeout_seconds=g_timeout,
            )
            if isinstance(gj, dict):
                gparsed = _normalize_gemini_parsed(gj)

                # merge strategy:
                # - items: gemini가 준 게 있고, 기존이 비었거나 suspicious면 교체
                # - 필드: 기존이 비었으면 채움
                g_items = gparsed.get("items") if isinstance(gparsed.get("items"), list) else []
                if g_items and (_is_items_suspicious(items_now)):
                    parsed["items"] = g_items
                    hints["geminiUsed"] = True

                if not parsed.get("hospitalName") and gparsed.get("hospitalName"):
                    parsed["hospitalName"] = gparsed["hospitalName"]
                    hints["geminiUsed"] = True

                if not parsed.get("visitDate") and gparsed.get("visitDate"):
                    parsed["visitDate"] = gparsed["visitDate"]
                    hints["geminiUsed"] = True

                if (not parsed.get("totalAmount")) and gparsed.get("totalAmount"):
                    parsed["totalAmount"] = gparsed["totalAmount"]
                    hints["geminiUsed"] = True

        except Exception as e:
            hints["geminiError"] = str(e)[:200]

    # final sanitize
    if not isinstance(parsed.get("items"), list):
        parsed["items"] = []
    parsed["items"] = parsed["items"][:120]

    if parsed.get("totalAmount") is not None:
        try:
            parsed["totalAmount"] = int(parsed["totalAmount"])
        except Exception:
            parsed["totalAmount"] = None

    # keep ocrText for server-side heuristics
    parsed["ocrText"] = (ocr_text or "")[:8000]

    return webp_bytes, parsed, hints


