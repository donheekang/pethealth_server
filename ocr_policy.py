# ocr_policy.py
# PetHealth+ - Receipt OCR + robust parsing + optional Gemini fallback
#
# Required public API:
#   process_receipt(raw: bytes, **kwargs) -> (webp_bytes: bytes, parsed: dict, hints: dict)
#
# Goals:
# - Never crash the server due to OCR/AI problems.
# - Always return a valid webp/jpg bytes + parsed dict (best-effort).
# - Google Vision OCR optional, Gemini optional.

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageOps  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pillow (PIL) is required for ocr_policy.py. Add 'Pillow' to requirements.txt.") from e


# ---------------------------------------------------------
# Concurrency control (optional)
# ---------------------------------------------------------
_OCR_SEMA: Optional[threading.BoundedSemaphore] = None
_OCR_SEMA_N: int = 0
_OCR_SEMA_LOCK = threading.Lock()

_GEMINI_SEMA: Optional[threading.BoundedSemaphore] = None
_GEMINI_SEMA_N: int = 0
_GEMINI_SEMA_LOCK = threading.Lock()


def _get_sema(max_concurrency: int) -> threading.BoundedSemaphore:
    global _OCR_SEMA, _OCR_SEMA_N
    n = max(1, int(max_concurrency or 1))
    with _OCR_SEMA_LOCK:
        if _OCR_SEMA is None or _OCR_SEMA_N != n:
            _OCR_SEMA = threading.BoundedSemaphore(value=n)
            _OCR_SEMA_N = n
        return _OCR_SEMA


def _get_gemini_sema(max_concurrency: int) -> threading.BoundedSemaphore:
    global _GEMINI_SEMA, _GEMINI_SEMA_N
    n = max(1, int(max_concurrency or 1))
    with _GEMINI_SEMA_LOCK:
        if _GEMINI_SEMA is None or _GEMINI_SEMA_N != n:
            _GEMINI_SEMA = threading.BoundedSemaphore(value=n)
            _GEMINI_SEMA_N = n
        return _GEMINI_SEMA


# ---------------------------------------------------------
# Image helpers
# ---------------------------------------------------------
def _load_image(raw: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _downscale_to_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    max_pixels = int(max_pixels or 0)
    if max_pixels <= 0:
        return img

    w, h = img.size
    px = w * h
    if px <= max_pixels:
        return img

    scale = (max_pixels / float(px)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _resize_max_width(img: Image.Image, max_width: int) -> Image.Image:
    max_width = int(max_width or 0)
    if max_width <= 0:
        return img

    w, h = img.size
    if w <= max_width:
        return img

    scale = max_width / float(w)
    new_h = max(1, int(h * scale))
    return img.resize((max_width, new_h), Image.LANCZOS)


def _encode_webp(img: Image.Image, quality: int) -> bytes:
    quality = int(quality or 85)
    quality = max(30, min(quality, 95))

    out = io.BytesIO()
    try:
        img.save(out, format="WEBP", quality=quality, method=6)
        return out.getvalue()
    except Exception:
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85, optimize=True)
        return out.getvalue()


def _encode_jpeg(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue()


# ---------------------------------------------------------
# Optional Google Vision OCR
# ---------------------------------------------------------
def _load_google_credentials(google_credentials: str):
    """
    Supports:
    - JSON string
    - File path to a JSON file
    """
    s = (google_credentials or "").strip()
    if not s:
        return None

    try:
        from google.oauth2 import service_account  # type: ignore
    except Exception:
        return None

    try:
        if s.startswith("{"):
            info = json.loads(s)
            return service_account.Credentials.from_service_account_info(info)
        if os.path.exists(s):
            return service_account.Credentials.from_service_account_file(s)
    except Exception:
        return None

    return None


def _google_ocr_text(
    image_bytes: bytes,
    google_credentials: str,
    timeout_seconds: int,
) -> str:
    try:
        from google.cloud import vision  # type: ignore
    except Exception:
        return ""

    creds = _load_google_credentials(google_credentials)
    try:
        client = vision.ImageAnnotatorClient(credentials=creds) if creds else vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)

        resp = client.document_text_detection(image=image, timeout=timeout_seconds)
        if getattr(resp, "error", None) and getattr(resp.error, "message", None):
            return ""

        txt = ""
        try:
            txt = resp.full_text_annotation.text or ""
        except Exception:
            txt = ""

        if not txt:
            try:
                ann = resp.text_annotations or []
                if ann:
                    txt = ann[0].description or ""
            except Exception:
                txt = ""

        return txt.strip()
    except Exception:
        return ""


# ---------------------------------------------------------
# Gemini (optional) - REST call (no extra deps)
# ---------------------------------------------------------
def _env_truthy(v: str) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _gemini_cfg() -> Dict[str, Any]:
    return {
        "enabled": _env_truthy(os.environ.get("GEMINI_ENABLED", "")),
        "api_key": (os.environ.get("GEMINI_API_KEY", "") or "").strip(),
        "model": (os.environ.get("GEMINI_MODEL_NAME", "") or "gemini-1.5-flash").strip(),
    }


def _http_post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try direct parse
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            pass

    # Find first {...} block
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def _gemini_generate(prompt: str, parts: List[Dict[str, Any]], timeout: int) -> str:
    cfg = _gemini_cfg()
    if not (cfg["enabled"] and cfg["api_key"]):
        return ""

    model = cfg["model"]
    key = cfg["api_key"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}] + parts}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    try:
        data = _http_post_json(url, payload, timeout=timeout)
        cands = data.get("candidates") or []
        if not cands:
            return ""
        content = (cands[0].get("content") or {})
        p = content.get("parts") or []
        if not p:
            return ""
        txt = p[0].get("text") or ""
        return str(txt).strip()
    except Exception:
        return ""


def _gemini_parse_receipt_from_text(ocr_text: str, timeout: int = 12) -> Dict[str, Any]:
    """
    Returns dict possibly containing: visitDate, hospitalName, totalAmount, items
    """
    if not (ocr_text or "").strip():
        return {}

    prompt = (
        "You are a receipt parser for veterinary clinic receipts in Korea.\n"
        "Given OCR text, extract:\n"
        "- visitDate: YYYY-MM-DD or null\n"
        "- hospitalName: string or null\n"
        "- totalAmount: integer KRW or null\n"
        "- items: array of {itemName: string, price: integer}\n\n"
        "Rules:\n"
        "- Ignore customer number, phone, address, business registration number, issue time, VAT/tax lines.\n"
        "- Items should be actual billed medical/grooming/services.\n"
        "- If there is a table with columns (단가/수량/금액), use 금액 for price.\n"
        "- Output ONLY valid JSON.\n\n"
        f"OCR_TEXT:\n{ocr_text[:6000]}"
    )

    txt = _gemini_generate(prompt, parts=[], timeout=timeout)
    out = _extract_json_from_text(txt) or {}
    return out if isinstance(out, dict) else {}


def _gemini_parse_receipt_from_image(jpeg_bytes: bytes, timeout: int = 18) -> Dict[str, Any]:
    """
    Stronger fallback: send image to Gemini (multimodal).
    """
    if not jpeg_bytes:
        return {}

    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    prompt = (
        "You are a receipt parser for veterinary clinic receipts in Korea.\n"
        "From the receipt image, extract:\n"
        "- visitDate: YYYY-MM-DD or null\n"
        "- hospitalName: string or null\n"
        "- totalAmount: integer KRW or null\n"
        "- items: array of {itemName: string, price: integer}\n\n"
        "Rules:\n"
        "- Ignore customer number, phone, address, business registration number, issue time, VAT/tax lines.\n"
        "- Items should be actual billed medical/grooming/services.\n"
        "- If there is a table with columns (단가/수량/금액), use 금액 for price.\n"
        "- Output ONLY valid JSON.\n"
    )
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": b64}}]
    txt = _gemini_generate(prompt, parts=parts, timeout=timeout)
    out = _extract_json_from_text(txt) or {}
    return out if isinstance(out, dict) else {}


# ---------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------
_DATE_PATTERNS = [
    re.compile(r"(20\d{2})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})", re.UNICODE),
    re.compile(r"\b(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b"),
]

_TOTAL_KEYWORDS = [
    "합계", "총액", "총 금액", "총금액",
    "결제", "결제금액", "결제요청", "결제예정",
    "청구", "청구금액",
    "소계",
    "총진료비", "총 진료비",
    "total", "amount", "sum",
]

# 이 키워드가 포함된 라인은 "아이템 라인"이 아니라 메타일 가능성이 높음
_META_BANNED = [
    "고객", "고객번호", "고객 번호", "고객이름", "고객 이름",
    "발행일", "발행", "날짜", "시간",
    "사업자", "등록번호", "대표", "원장",
    "전화", "tel", "fax",
    "주소", "소재지",
    "serial", "sign",
    "단가", "수량", "금액",
    "과세", "비과세", "부가세", "vat",
    "승인", "카드", "현금",
]


def _parse_amount_token(s: str) -> Optional[int]:
    """
    OCR에서 O/0 혼동이 자주 나서 O->0도 같이 처리.
    """
    if not s:
        return None
    s2 = s.replace("O", "0").replace("o", "0")
    digits = re.sub(r"[^\d]", "", s2)
    if not digits:
        return None
    try:
        v = int(digits)
        return v if v >= 0 else None
    except Exception:
        return None


def _pick_visit_date(lines: List[str]) -> Optional[str]:
    for line in lines[:120]:
        l = line.strip()
        if not l:
            continue

        m = _DATE_PATTERNS[0].search(l)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"

        m2 = _DATE_PATTERNS[1].search(l)
        if m2:
            yy, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            y = 2000 + yy
            if 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"

    return None


def _clean_hospital_name(s: str) -> str:
    s = (s or "").strip()
    # common prefixes on receipts / OCR
    s = re.sub(r"^(?:병원명|상호|상호명)\s*[:：]\s*", "", s)
    s = re.sub(r"^\s*원\s*명\s*[:：]\s*", "", s)
    s = re.sub(r"^\s*병\s*원\s*명\s*[:：]\s*", "", s)
    return s.strip()


def _pick_hospital_name(lines: List[str]) -> Optional[str]:
    for line in lines[:40]:
        l = line.strip()
        if not l:
            continue
        if "병원" in l or "동물" in l:
            if any(k in l for k in ["사업자", "등록번호", "전화", "주소"]):
                continue
            return _clean_hospital_name(l)[:60]

    for line in lines[:40]:
        l = line.strip()
        if not l:
            continue
        if len(re.sub(r"[\d\W_]+", "", l)) < 2:
            continue
        if any(k in l for k in ["사업자", "등록번호", "전화", "주소"]):
            continue
        return _clean_hospital_name(l)[:60]

    return None


def _pick_total_amount(lines: List[str]) -> Optional[int]:
    candidates: List[int] = []

    for line in lines:
        l = line.strip()
        if not l:
            continue
        lower = l.lower()

        if any(k in l for k in _TOTAL_KEYWORDS) or any(k in lower for k in _TOTAL_KEYWORDS):
            for m in re.finditer(r"[0-9Oo][0-9Oo,\.]*", l):
                amt = _parse_amount_token(m.group(0))
                if amt is not None and amt > 0:
                    candidates.append(amt)

    if candidates:
        # receipts can contain "0" and tax lines; pick max
        best = max(candidates)
        return best if best >= 0 else None

    # fallback: max "money-like" number
    all_nums: List[int] = []
    for line in lines:
        for m in re.finditer(r"[0-9Oo][0-9Oo,\.]*", line):
            amt = _parse_amount_token(m.group(0))
            if amt is None:
                continue
            if amt >= 1000:
                all_nums.append(amt)

    return max(all_nums) if all_nums else None


def _is_meta_line(l: str) -> bool:
    ll = (l or "").strip()
    if not ll:
        return True
    low = ll.lower()
    return any(k in ll for k in _META_BANNED) or any(k in low for k in _META_BANNED)


def _extract_items(lines: List[str], limit: int = 50, min_price: int = 1000) -> List[Dict[str, Any]]:
    """
    Robust item extraction:
    - supports table rows: name unit qty amount
    - supports simple rows: name amount
    - filters meta lines (customer number, issue time, tel, address, etc.)
    """
    out: List[Dict[str, Any]] = []
    min_price = int(min_price or 0)

    for line in lines:
        l = " ".join((line or "").strip().split())
        if not l:
            continue

        # Strong meta filter
        if _is_meta_line(l):
            continue

        # --------
        # 1) Table row: name unit qty amount  (ex: "*Rabies 30,000 1 30,000")
        # --------
        m = re.match(
            r"^[\*\-•]?\s*(.{2,60}?)\s+([0-9Oo][0-9Oo,\.]*)\s+(\d{1,3})\s+([0-9Oo][0-9Oo,\.]*)\s*$",
            l,
        )
        if m:
            name = (m.group(1) or "").strip()
            amount = _parse_amount_token(m.group(4) or "")
            if amount is not None and amount > 0 and (min_price <= 0 or amount >= min_price):
                out.append({"itemName": name[:200], "price": amount, "categoryTag": None})
                if len(out) >= limit:
                    break
            continue

        # --------
        # 2) Simple row: name amount  (ex: "진료비 30000원")
        # --------
        m2 = re.match(r"^(.{2,60}?)\s+([0-9Oo][0-9Oo,\.]*)\s*원?\s*$", l)
        if not m2:
            m2 = re.match(r"^(.{2,60}?)[\s:₩]+([0-9Oo][0-9Oo,\.]*)\s*원?\s*$", l)

        if m2:
            name = (m2.group(1) or "").strip()
            price = _parse_amount_token(m2.group(2) or "")

            if price is None or price <= 0:
                continue
            if min_price > 0 and price < min_price:
                # prevent "고객번호 9원", "발행일 ... 58" 같은 오탐 방지
                continue

            # extra filter: lines that look like "발행일: ... 58"
            if any(k in name for k in ["발행", "고객", "사업자", "등록번호", "전화", "주소", "serial", "sign"]):
                continue

            out.append({"itemName": name[:200], "price": price, "categoryTag": None})
            if len(out) >= limit:
                break

    return out


def _address_hint(lines: List[str]) -> Optional[str]:
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if "주소" in l:
            return l[:140]
        if any(tok in l for tok in ["로", "길", "동", "구", "시", "군", "읍", "면"]) and re.search(r"\d", l):
            if len(l) >= 8 and any(k in l for k in ["시", "구", "군"]):
                return l[:140]
    return None


def _looks_suspicious(parsed: Dict[str, Any]) -> bool:
    items = parsed.get("items") or []
    total = parsed.get("totalAmount")
    try:
        total_i = int(total) if total is not None else None
    except Exception:
        total_i = None

    # empty items is suspicious
    if not items:
        return True

    # total missing or too small -> suspicious
    if total_i is None or total_i < 1000:
        # but items could still be valid; check items max
        prices = []
        for it in items:
            p = it.get("price")
            if isinstance(p, int):
                prices.append(p)
        if not prices:
            return True
        if max(prices) < 1000:
            return True

    # if items are only tiny numbers (<= 1000), likely meta lines
    prices = []
    for it in items:
        p = it.get("price")
        if isinstance(p, int):
            prices.append(p)
    if prices and max(prices) < 1000:
        return True

    return False


def _merge_parsed(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge 'extra' into 'base' conservatively.
    Use extra when base is missing or suspicious.
    """
    out = dict(base or {})

    for k in ("visitDate", "hospitalName", "totalAmount"):
        if (out.get(k) is None or str(out.get(k) or "").strip() == "") and extra.get(k) is not None:
            out[k] = extra.get(k)

    base_items = out.get("items") or []
    extra_items = extra.get("items") or []
    if (not base_items) and extra_items:
        out["items"] = extra_items

    # if totalAmount looks wrong, accept extra
    try:
        bt = out.get("totalAmount")
        bt_i = int(bt) if bt is not None else None
    except Exception:
        bt_i = None
    try:
        et = extra.get("totalAmount")
        et_i = int(et) if et is not None else None
    except Exception:
        et_i = None

    if (bt_i is None or bt_i < 1000) and (et_i is not None and et_i >= 1000):
        out["totalAmount"] = et_i

    return out


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
def process_receipt(
    raw: bytes,
    *,
    google_credentials: str = "",
    ocr_timeout_seconds: int = 12,
    ocr_max_concurrency: int = 4,
    ocr_sema_acquire_timeout_seconds: float = 1.0,
    receipt_max_width: int = 1024,
    receipt_webp_quality: int = 85,
    image_max_pixels: int = 20_000_000,
    item_min_price: int = 1000,
    gemini_timeout_text: int = 12,
    gemini_timeout_image: int = 18,
    gemini_max_concurrency: int = 2,
    **kwargs,
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      webp_bytes: bytes (best-effort WEBP, may fallback to JPEG if WEBP unsupported)
      parsed: { visitDate, hospitalName, totalAmount, items }
      hints: { addressHint?, ocrTextPreview?, geminiUsed? }
    """
    if raw is None:
        raw = b""
    if len(raw) == 0:
        raise ValueError("empty file")

    img = _load_image(raw)
    img = _downscale_to_max_pixels(img, int(image_max_pixels))
    img = _resize_max_width(img, int(receipt_max_width))

    webp_bytes = _encode_webp(img, int(receipt_webp_quality))
    jpeg_bytes = _encode_jpeg(img)

    # OCR (best-effort)
    text = ""
    sema = _get_sema(int(ocr_max_concurrency))
    acquired = sema.acquire(timeout=float(ocr_sema_acquire_timeout_seconds or 0))
    try:
        if acquired:
            text = _google_ocr_text(jpeg_bytes, google_credentials, int(ocr_timeout_seconds))
    finally:
        if acquired:
            try:
                sema.release()
            except Exception:
                pass

    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]

    parsed: Dict[str, Any] = {
        "visitDate": _pick_visit_date(lines),
        "hospitalName": _pick_hospital_name(lines),
        "totalAmount": _pick_total_amount(lines),
        "items": _extract_items(lines, limit=50, min_price=int(item_min_price)),
    }

    hints: Dict[str, Any] = {}
    ah = _address_hint(lines)
    if ah:
        hints["addressHint"] = ah
    if text:
        hints["ocrTextPreview"] = text[:500]

    # Gemini fallback (text -> image)
    cfg = _gemini_cfg()
    if cfg["enabled"] and cfg["api_key"] and _looks_suspicious(parsed):
        sema_g = _get_gemini_sema(int(gemini_max_concurrency))
        got = sema_g.acquire(timeout=1.0)
        try:
            if got:
                g1 = _gemini_parse_receipt_from_text(text or "", timeout=int(gemini_timeout_text))
                if isinstance(g1, dict) and g1:
                    parsed = _merge_parsed(parsed, g1)

                # if still suspicious, try image
                if _looks_suspicious(parsed):
                    g2 = _gemini_parse_receipt_from_image(jpeg_bytes, timeout=int(gemini_timeout_image))
                    if isinstance(g2, dict) and g2:
                        parsed = _merge_parsed(parsed, g2)

                hints["geminiUsed"] = True
                hints["geminiModel"] = cfg["model"]
        finally:
            if got:
                try:
                    sema_g.release()
                except Exception:
                    pass

    # Final normalization
    if parsed.get("hospitalName"):
        parsed["hospitalName"] = _clean_hospital_name(str(parsed["hospitalName"]))

    # Ensure types
    try:
        if parsed.get("totalAmount") is not None:
            parsed["totalAmount"] = int(parsed["totalAmount"])
    except Exception:
        parsed["totalAmount"] = None

    # Ensure item shape
    safe_items = []
    for it in (parsed.get("items") or []):
        if not isinstance(it, dict):
            continue
        nm = (it.get("itemName") or "").strip()
        pr = it.get("price")
        try:
            pr_i = int(pr) if pr is not None else None
        except Exception:
            pr_i = None
        if not nm or pr_i is None or pr_i <= 0:
            continue
        safe_items.append({"itemName": nm[:200], "price": pr_i, "categoryTag": it.get("categoryTag")})
    parsed["items"] = safe_items[:50]

    return webp_bytes, parsed, hints


