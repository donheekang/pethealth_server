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
import time
import threading
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Regex / constants
# -----------------------------
_DATE_RE_1 = re.compile(r"\b(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b")
_DATE_RE_2 = re.compile(r"\b(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\b")
_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")  # 3+ digits or comma-style
_HOSP_RE = re.compile(r"(병원\s*명|원\s*명)\s*[:：]?\s*(.+)$")

# PII patterns for image redaction (best-effort)
_PHONE_RE = re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b")
_BIZ_RE = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")
_CARD_RE = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")

_NOISE_TOKENS = [
    "고객", "고객번호", "고객 번호", "발행", "발행일", "발행 일", "사업자", "사업자등록", "대표",
    "전화", "주소", "serial", "sign", "승인", "카드", "현금", "부가세", "vat", "면세", "과세",
    "공급가", "소계", "합계", "총액", "총 금액", "총금액", "청구", "결제", "결제요청", "결제예정",
]

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
# Google Vision OCR
# -----------------------------
def _build_vision_client(google_credentials: str):
    # import locally (avoid import-time crash)
    from google.cloud import vision
    from google.oauth2 import service_account

    gc = (google_credentials or "").strip()
    if not gc:
        # default credentials
        return vision.ImageAnnotatorClient()

    # JSON string?
    if gc.startswith("{") and gc.endswith("}"):
        info = json.loads(gc)
        # fix private_key newlines
        pk = info.get("private_key")
        if isinstance(pk, str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds)

    # file path
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
    from google.cloud import vision
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
# Redaction (best-effort): mask tokens with hyphen digits patterns (phone/biz/card)
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
            dlow = desc.lower()

            if not (_PHONE_RE.search(desc) or _BIZ_RE.search(desc) or _CARD_RE.search(desc)):
                continue

            poly = getattr(a, "bounding_poly", None)
            if not poly or not getattr(poly, "vertices", None):
                continue

            xs = []
            ys = []
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
            # white rectangle
            draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

        return img
    except Exception:
        return img

# -----------------------------
# Text parsing (robust line-item extraction)
# -----------------------------
def _clean_item_name(name: str) -> str:
    s = (name or "").strip()
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # remove leading bullets/stars
    s = s.lstrip("*•·-—").strip()
    return s[:200]

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

        # item line should contain at least one "real" amount (>= 1000 or comma style)
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
    for ln in lines[:30]:
        m = _HOSP_RE.search(ln)
        if m:
            v = m.group(2).strip()
            # stop at weird separators
            v = re.split(r"\s{2,}|/|\||,", v)[0].strip()
            if v:
                return v[:80]
    # fallback: first line containing "동물병원"
    for ln in lines[:40]:
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
    # heuristic: korean address-ish line
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
    }

    vd = _parse_date_from_text(text)
    if vd:
        parsed["visitDate"] = vd.isoformat()

    items = _extract_items_from_text(text)
    # ✅ 강제 보강: text에 rabbies/rabies/광견병이 있는데 items가 없으면 한 줄 생성
    low = (text or "").lower()
    if (not items) and (("rabbies" in low) or ("rabies" in low) or ("광견병" in (text or "")) or ("광견" in (text or ""))):
        ta = parsed.get("totalAmount")
        if isinstance(ta, int) and ta >= 100:
            items = [{"itemName": "Rabies", "price": int(ta), "categoryTag": None}]

    parsed["items"] = items

    hints: Dict[str, Any] = {
        "addressHint": _extract_address_hint(text),
        "ocrTextPreview": (text or "")[:400],
    }
    return parsed, hints

# -----------------------------
# Gemini (Generative Language API) - best-effort fallback
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
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
        },
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
    # remove code fences
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    # find first { ... } block
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
        "- Do NOT include totals, taxes, card approval, '결제요청/결제예정/합계/소계/청구금액' as items.\n"
        "- If you see 'Rabies' but OCR typo like 'Rabbies', normalize to 'Rabies'.\n"
        "- If uncertain, best guess.\n"
    )

    b64 = base64.b64encode(image_bytes).decode("ascii")
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/webp", "data": b64}},
        {"text": "OCR text (may be noisy):\n" + (ocr_text or "")[:4000]},
    ]

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
    # if all items look like payment/total lines -> suspicious
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
        # allow flexible date, normalize if possible
        d = _parse_date_from_text(vd.strip())
        out["visitDate"] = d.isoformat() if d else vd.strip()[:20]

    ta = j.get("totalAmount")
    ta_i = _coerce_int_amount(ta)
    out["totalAmount"] = ta_i if ta_i and ta_i > 0 else None

    items = j.get("items")
    if isinstance(items, list):
        cleaned: List[Dict[str, Any]] = []
        for it in items[:120]:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("itemName") or "").strip()
            if not nm or _is_noise_line(nm):
                continue
            pr = _coerce_int_amount(it.get("price"))
            if pr is not None and pr < 0:
                pr = None
            cleaned.append({"itemName": _clean_item_name(nm), "price": pr, "categoryTag": None})
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
      parsed: {hospitalName, visitDate(YYYY-MM-DD), totalAmount(int), items:[{itemName, price, categoryTag}]}
      hints: {addressHint, ...}
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
        # flatten
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

    # 2) OCR with concurrency guard
    sema = _get_sema(int(ocr_max_concurrency or 4))
    acquired = sema.acquire(timeout=float(ocr_sema_acquire_timeout_seconds or 1.0))
    if not acquired:
        # keep it simple (caller wraps exceptions)
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

    # 3) redact image (best-effort) using vision tokens (phone/biz/card)
    redacted = _redact_image_with_vision_tokens(img.copy(), vision_resp)
    webp_bytes = _to_webp_bytes(redacted, quality=int(receipt_webp_quality or 85))

    # 4) parse from OCR text
    parsed, hints = _parse_receipt_from_text(ocr_text or "")
    hints["ocrEngine"] = "google_vision"
    hints["geminiUsed"] = False

    # 5) Gemini assist (fallback when items suspicious/empty)
    # enable logic: explicit param > env
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))

    if g_enabled and g_key.strip():
        try:
            if _is_items_suspicious(parsed.get("items") if isinstance(parsed.get("items"), list) else []):
                gj = _gemini_parse_receipt(
                    image_bytes=webp_bytes,  # use resized+redacted webp
                    ocr_text=ocr_text or "",
                    api_key=g_key,
                    model=g_model,
                    timeout_seconds=g_timeout,
                )
                if isinstance(gj, dict):
                    gparsed = _normalize_gemini_parsed(gj)

                    # merge: gemini items win if they look better
                    if gparsed.get("items"):
                        parsed["items"] = gparsed["items"]
                        hints["geminiUsed"] = True

                    # fill missing fields
                    if not parsed.get("hospitalName") and gparsed.get("hospitalName"):
                        parsed["hospitalName"] = gparsed["hospitalName"]
                    if not parsed.get("visitDate") and gparsed.get("visitDate"):
                        parsed["visitDate"] = gparsed["visitDate"]
                    if (not parsed.get("totalAmount")) and gparsed.get("totalAmount"):
                        parsed["totalAmount"] = gparsed["totalAmount"]

        except Exception as e:
            # never fail whole pipeline
            hints["geminiError"] = str(e)[:200]

    # final sanitize
    if not isinstance(parsed.get("items"), list):
        parsed["items"] = []
    parsed["items"] = parsed["items"][:120]

    # ensure types
    if parsed.get("totalAmount") is not None:
        try:
            parsed["totalAmount"] = int(parsed["totalAmount"])
        except Exception:
            parsed["totalAmount"] = None

    return webp_bytes, parsed, hints


