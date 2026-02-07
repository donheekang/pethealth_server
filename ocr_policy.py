# ocr_policy.py (PetHealth+)
# OCR + minimal redaction + receipt parsing
# Returns: (webp_bytes, parsed_dict, hints_dict)

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
_DATE_RE_1 = re.compile(r"\b(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b")
_DATE_RE_2 = re.compile(r"\b(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\b")

# 3+ digits or comma-style number
_AMOUNT_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d{3,})")
_HOSP_RE = re.compile(r"(?:병원\s*명|원\s*명)\s*[:：]?\s*(.+)$")

# PII patterns (best-effort)
_PHONE_RE = re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b")
_BIZ_RE = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")
_CARD_RE = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")

_NOISE_TOKENS = [
    "고객", "고객번호", "고객 번호", "발행", "발행일", "발행 일", "사업자", "사업자등록", "대표",
    "전화", "주소", "serial", "sign", "승인", "카드", "현금", "부가세", "vat", "면세", "과세",
    "공급가", "소계", "합계", "총액", "총 금액", "총금액", "청구", "결제", "결제요청", "결제예정",
]

_RABIES_TOKENS = ("rabies", "rabbies", "광견병", "광견")


def _load_pil():
    try:
        from PIL import Image, ImageOps, ImageDraw
        return Image, ImageOps, ImageDraw
    except Exception as e:
        raise RuntimeError("Pillow is required. Install: pip install Pillow") from e


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", (s or "").lower())


def _is_noise_line(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    low = t.lower()
    k = _norm(t)
    if len(k) < 2:
        return True
    for x in _NOISE_TOKENS:
        if x in t or x in low:
            return True
    return False


def _parse_date_from_text(text: str) -> Optional[date]:
    if not text:
        return None
    m = _DATE_RE_1.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    m = _DATE_RE_2.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    return None


def _coerce_int_amount(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
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
# Image helpers
# -----------------------------
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
    q = int(max(30, min(int(quality or 85), 95)))
    img.save(buf, format="WEBP", quality=q, method=6)
    return buf.getvalue()


# -----------------------------
# Google Vision OCR
# -----------------------------
def _maybe_load_sa_info(google_credentials: str) -> Optional[dict]:
    gc = (google_credentials or "").strip()
    if not gc:
        return None

    # JSON string
    if gc.startswith("{") and gc.endswith("}"):
        info = json.loads(gc)
        pk = info.get("private_key")
        if isinstance(pk, str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n", "\n")
        return info

    # base64 JSON (optional)
    if len(gc) > 200 and all(c.isalnum() or c in "+/=\n\r" for c in gc):
        try:
            raw = base64.b64decode(gc).decode("utf-8", errors="replace")
            if raw.strip().startswith("{"):
                info = json.loads(raw)
                pk = info.get("private_key")
                if isinstance(pk, str) and "\\n" in pk:
                    info["private_key"] = pk.replace("\\n", "\n")
                return info
        except Exception:
            pass

    return None


def _build_vision_client(google_credentials: str):
    try:
        from google.cloud import vision
        from google.oauth2 import service_account
    except Exception as e:
        raise RuntimeError("google-cloud-vision is required. Install: pip install google-cloud-vision") from e

    gc = (google_credentials or "").strip()
    if not gc:
        return vision.ImageAnnotatorClient()

    # service account JSON string/base64
    info = _maybe_load_sa_info(gc)
    if isinstance(info, dict):
        creds = service_account.Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds)

    # file path
    if os.path.exists(gc):
        return vision.ImageAnnotatorClient.from_service_account_file(gc)

    # env path fallback
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc
    return vision.ImageAnnotatorClient()


def _vision_ocr(image_bytes: bytes, google_credentials: str, timeout_seconds: int) -> Tuple[str, Any]:
    from google.cloud import vision
    client = _build_vision_client(google_credentials)
    img = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=img, timeout=float(timeout_seconds or 12))
    text = ""
    try:
        if resp and resp.full_text_annotation and resp.full_text_annotation.text:
            text = resp.full_text_annotation.text
    except Exception:
        text = ""
    return text or "", resp


def _redact_image_with_tokens(img, vision_response) -> Any:
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
            verts = getattr(poly, "vertices", None) if poly else None
            if not verts:
                continue

            xs, ys = [], []
            for v in verts:
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
# Parsing helpers
# -----------------------------
def _clean_item_name(name: str) -> str:
    s = (name or "").strip()
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lstrip("*•·-—").strip()
    return s[:200]


def _extract_hospital_name(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:40]:
        m = _HOSP_RE.search(ln)
        if m:
            v = m.group(1).strip()
            v = re.split(r"\s{2,}|/|\||,", v)[0].strip()
            if v:
                return v[:80]
    for ln in lines[:40]:
        if "동물병원" in ln:
            return ln[:80]
    return None


def _extract_total_amount(text: str) -> Optional[int]:
    if not text:
        return None
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    keys = ("청구", "결제", "합계", "총", "총액", "총금액", "소계", "total", "grand")
    cands: List[int] = []
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in keys) or any(k in ln for k in keys):
            nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(ln)]
            nums = [n for n in nums if n >= 100]
            cands.extend(nums)
    if not cands:
        nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(text)]
        nums = [n for n in nums if n >= 100]
        cands = nums
    if not cands:
        return None
    return int(max(cands))


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

        nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(ln)]
        nums = [n for n in nums if n >= 100]
        if not nums:
            continue

        price = nums[-1]

        name_part = _AMOUNT_RE.sub(" ", ln)
        name_part = re.sub(r"\b\d{1,2}\b", " ", name_part)
        name_part = _clean_item_name(name_part)
        if not name_part:
            continue

        key = (_norm(name_part), int(price))
        if key in seen:
            continue
        seen.add(key)

        out.append({"itemName": name_part, "price": int(price), "categoryTag": None})

    return out[:120]


def _extract_address_hint(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if "주소" in ln:
            return ln[:120]
    for ln in lines:
        if len(ln) >= 10 and any(ch.isdigit() for ch in ln) and any(tok in ln for tok in ["시", "구", "동", "로", "길", "번지"]):
            return ln[:120]
    return None


def _ensure_rabies_item(items: List[Dict[str, Any]], text: str, total_amount: Optional[int]) -> List[Dict[str, Any]]:
    low = (text or "").lower()
    has_rabies = any(_norm(str(it.get("itemName") or "")) in ("rabies", "rabbies", "광견병", "광견") or any(t in str(it.get("itemName") or "").lower() for t in ("rabies", "rabbies")) or ("광견병" in str(it.get("itemName") or "")) for it in (items or []))
    if has_rabies:
        return items

    if any(tok in low for tok in ("rabies", "rabbies")) or ("광견병" in (text or "")) or ("광견" in (text or "")):
        # item line이 누락된 케이스 보강
        items = list(items or [])
        items.append({"itemName": "Rabies", "price": int(total_amount) if isinstance(total_amount, int) and total_amount >= 100 else None, "categoryTag": None})
    return items


def _parse_receipt_from_text(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lines_all = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines_all if ln]

    hospital = _extract_hospital_name(text)
    vd = _parse_date_from_text(text)
    total = _extract_total_amount(text)
    items = _extract_items_from_text(text)
    items = _ensure_rabies_item(items, text, total)

    parsed: Dict[str, Any] = {
        "hospitalName": hospital,
        "visitDate": vd.isoformat() if vd else None,
        "totalAmount": int(total) if isinstance(total, int) and total > 0 else None,
        "items": items[:120],

        # downstream 디버그/보강용
        "text": text or "",
        "ocrText": text or "",
        "lines": lines[:400],
    }
    hints: Dict[str, Any] = {
        "addressHint": _extract_address_hint(text),
        "ocrTextPreview": (text or "")[:500],
        "hasLines": bool(lines),
    }
    return parsed, hints


# -----------------------------
# Gemini assist (optional)
# -----------------------------
def _env_bool(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _call_gemini_generate_content(*, api_key: str, model: str, parts: List[Dict[str, Any]], timeout_seconds: int) -> str:
    import urllib.request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=float(timeout_seconds or 10)) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _extract_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    t = s.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i : j + 1])
    except Exception:
        return None


def _items_suspicious(items: List[Dict[str, Any]]) -> bool:
    if not items:
        return True
    bad = 0
    for it in items[:20]:
        nm = str((it or {}).get("itemName") or "")
        if _is_noise_line(nm):
            bad += 1
    return bad >= max(1, min(3, len(items)))


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

    gemini_enabled: Optional[bool] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model_name: Optional[str] = None,
    gemini_timeout_seconds: Optional[int] = None,
    **kwargs,
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    if not raw_bytes:
        raise ValueError("empty raw bytes")

    Image, ImageOps, ImageDraw = _load_pil()

    # load + normalize
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img = _ensure_max_pixels(img, int(image_max_pixels or 0))
    img = _resize_to_width(img, int(receipt_max_width or 0))

    # OCR bytes
    ocr_buf = io.BytesIO()
    img.save(ocr_buf, format="PNG")
    ocr_image_bytes = ocr_buf.getvalue()

    sema = _get_sema(int(ocr_max_concurrency or 4))
    acquired = sema.acquire(timeout=float(ocr_sema_acquire_timeout_seconds or 1.0))
    if not acquired:
        raise RuntimeError("OCR is busy (semaphore acquire timeout)")

    try:
        ocr_text, vision_resp = _vision_ocr(
            ocr_image_bytes,
            google_credentials=google_credentials,
            timeout_seconds=int(ocr_timeout_seconds or 12),
        )
    finally:
        try:
            sema.release()
        except Exception:
            pass

    # redact + webp
    redacted = _redact_image_with_tokens(img.copy(), vision_resp)
    webp_bytes = _to_webp_bytes(redacted, quality=int(receipt_webp_quality or 85))

    parsed, hints = _parse_receipt_from_text(ocr_text or "")
    hints["ocrEngine"] = "google_vision"
    hints["geminiUsed"] = False

    # gemini fallback only when items look bad
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))

    if g_enabled and g_key.strip():
        try:
            items_now = parsed.get("items") if isinstance(parsed.get("items"), list) else []
            if _items_suspicious(items_now):
                prompt = (
                    "You parse Korean veterinary receipts.\n"
                    "Return ONLY JSON with keys:\n"
                    'hospitalName (string|null), visitDate (YYYY-MM-DD|null), totalAmount (integer|null),\n'
                    'items (array of {itemName:string, price:integer|null}).\n'
                    "Do NOT include totals/taxes/payment lines as items.\n"
                    "If you see Rabies/Rabbies/광견병 then include an item 'Rabies'.\n"
                )
                b64 = base64.b64encode(webp_bytes).decode("ascii")
                parts = [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/webp", "data": b64}},
                    {"text": "OCR text:\n" + (ocr_text or "")[:4000]},
                ]
                raw = _call_gemini_generate_content(api_key=g_key, model=g_model, parts=parts, timeout_seconds=g_timeout)
                j = json.loads(raw)
                txt = (
                    (j.get("candidates") or [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                obj = _extract_json(txt or "")
                if isinstance(obj, dict):
                    # merge only if it provides better items
                    g_items = obj.get("items")
                    if isinstance(g_items, list) and g_items:
                        cleaned = []
                        for it in g_items[:120]:
                            if not isinstance(it, dict):
                                continue
                            nm = str(it.get("itemName") or "").strip()
                            if not nm:
                                continue
                            pr = _coerce_int_amount(it.get("price"))
                            cleaned.append({"itemName": _clean_item_name(nm), "price": pr, "categoryTag": None})
                        if cleaned:
                            parsed["items"] = cleaned
                            hints["geminiUsed"] = True

                    if not parsed.get("hospitalName") and isinstance(obj.get("hospitalName"), str):
                        parsed["hospitalName"] = obj.get("hospitalName").strip()[:80]
                    if not parsed.get("visitDate") and isinstance(obj.get("visitDate"), str):
                        parsed["visitDate"] = obj.get("visitDate").strip()[:20]
                    if not parsed.get("totalAmount"):
                        ta = _coerce_int_amount(obj.get("totalAmount"))
                        parsed["totalAmount"] = ta if ta and ta > 0 else parsed.get("totalAmount")

        except Exception as e:
            hints["geminiError"] = str(e)[:200]

    # ensure downstream keys
    if not isinstance(parsed.get("ocrText"), str):
        parsed["ocrText"] = ocr_text or ""
    if not isinstance(parsed.get("text"), str):
        parsed["text"] = ocr_text or ""
    if not isinstance(parsed.get("lines"), list):
        parsed["lines"] = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()][:400]

    return webp_bytes, parsed, hints


