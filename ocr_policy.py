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

# ✅ Rabies high-signal tokens (OCR often outputs partials)
_RABIES_STRONG = ("rabies", "rabbies", "광견병", "광견")
_RABIES_PARTIAL_NORMS = {"ra", "rab", "rabi", "rabb"}

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
    """
    기본 정책:
      - 금액(>=100) 포함 라인만 item 후보로 추출
      - 결제/합계/주소 등 노이즈 라인은 제외
    """
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

        # item line should contain at least one "real" amount (>= 100 or comma style)
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

def _contains_rabies_strong(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if "rabies" in low or "rabbies" in low:
        return True
    if "광견병" in text or "광견" in text:
        return True
    return False

def _find_rabies_partial_line(lines: List[str]) -> Optional[str]:
    for ln in (lines or [])[:400]:
        if not ln:
            continue
        if _is_noise_line(ln):
            continue
        n = _norm(ln)
        low = ln.lower()
        # strong tokens
        if "rabies" in low or "rabbies" in low or ("광견병" in ln) or ("광견" in ln):
            return ln
        # partial tokens (ra/rab/rabi/rabb) line-level
        if n in _RABIES_PARTIAL_NORMS:
            return ln
        # sometimes OCR gives "r a" -> norm becomes "ra"
        if n.startswith("ra") and n in _RABIES_PARTIAL_NORMS:
            return ln
    return None

def _parse_receipt_from_text(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # ✅ lines를 항상 만들어서 main.py가 ocr_text/lines fallback을 할 수 있게 함
    lines_all = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines_all if ln]

    parsed: Dict[str, Any] = {
        "hospitalName": _extract_hospital_name(text),
        "visitDate": None,
        "totalAmount": _extract_total_amount(text),
        "items": [],

        # ✅ main.py가 읽을 수 있는 OCR 원문 키 제공
        "text": text or "",
        "ocrText": text or "",
        "lines": lines[:400],
    }

    vd = _parse_date_from_text(text)
    if vd:
        parsed["visitDate"] = vd.isoformat()

    items = _extract_items_from_text(text)

    # ✅ Rabies 보강:
    # - items가 비었거나
    # - items에 rabies가 없는데 lines에서 ra/rab/...가 발견되는 경우
    #   => Rabies 아이템을 하나 추가 (price는 None로 둬서 총합 계산을 망치지 않음)
    has_rabies_in_items = False
    for it in items:
        nm = str((it or {}).get("itemName") or "")
        if _contains_rabies_strong(nm) or (_norm(nm) in _RABIES_PARTIAL_NORMS):
            has_rabies_in_items = True
            break

    rab_line = _find_rabies_partial_line(lines)
    if rab_line and (not has_rabies_in_items):
        # 가능하면 itemName은 표준으로
        items.append({"itemName": "Rabies", "price": None, "categoryTag": None, "rawLine": rab_line[:120]})

    # ✅ 기존: text에 rabies/광견병 신호가 있는데 items가 없으면 한 줄 생성(총액을 가격으로)
    low = (text or "").lower()
    if (not items) and (("rabbies" in low) or ("rabies" in low) or ("광견병" in (text or "")) or ("광견" in (text or ""))):
        ta = parsed.get("totalAmount")
        if isinstance(ta, int) and ta >= 100:
            items = [{"itemName": "Rabies", "price": int(ta), "categoryTag": None}]

    parsed["items"] = items

    hints: Dict[str, Any] = {
        "addressHint": _extract_address_hint(text),
        "ocrTextPreview": (text or "")[:400],
        "hasLines": bool(lines),
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
        "- If you see partial like 'ra'/'rab' in a vaccine context, interpret as 'Rabies' if reasonable.\n"
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
            if not nm:
                continue
            # keep even short tokens like "ra" (rabies partial) -> tag_policy will decide
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
      parsed: {hospitalName, visitDate(YYYY-MM-DD), totalAmount(int), items:[{itemName, price, categoryTag}],
               text, ocrText, lines}
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
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))

    if g_enabled and g_key.strip():
        try:
            items_now = parsed.get("items") if isinstance(parsed.get("items"), list) else []
            if _is_items_suspicious(items_now):
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

                    # keep raw text/lines
                    parsed["text"] = parsed.get("text") or (ocr_text or "")
                    parsed["ocrText"] = parsed.get("ocrText") or (ocr_text or "")
                    if not isinstance(parsed.get("lines"), list):
                        parsed["lines"] = [ln for ln in (ocr_text or "").splitlines() if ln.strip()][:400]

        except Exception as e:
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

    # ensure text/lines exist for downstream fallback
    if not isinstance(parsed.get("text"), str):
        parsed["text"] = ocr_text or ""
    if not isinstance(parsed.get("ocrText"), str):
        parsed["ocrText"] = ocr_text or ""
    if not isinstance(parsed.get("lines"), list):
        parsed["lines"] = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()][:400]

    return webp_bytes, parsed, hints

# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
# - catalog-first
# - returns record-level tags + per-item categoryTag suggestions

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Tag catalog (expandable)
# code == ReceiptTag.rawValue
# -----------------------------
TAG_CATALOG: List[Dict[str, Any]] = [
    # --- Exams ---
    {"code": "exam_xray", "group": "exam",
     "aliases": ["x-ray","xray","xr","radiograph","radiology","엑스레이","방사선","x선","x선촬영"]},
    {"code": "exam_ultrasound", "group": "exam",
     "aliases": ["ultrasound","sono","sonography","u/s","us","초음파","복부초음파","심장초음파"]},
    {"code": "exam_blood", "group": "exam",
     "aliases": ["cbc","blood test","chemistry","biochem","profile","혈액","혈액검사","생화학","전해질","검사"]},
    {"code": "exam_lab_panel", "group": "exam",
     "aliases": ["lab panel","screening","health check","종합검사","종합검진","패널검사"]},
    {"code": "exam_urine", "group": "exam",
     "aliases": ["urinalysis","u/a","ua","urine test","요검사","소변검사"]},
    {"code": "exam_fecal", "group": "exam",
     "aliases": ["fecal","stool test","대변검사","분변검사","배변검사"]},

    # ✅ missing iOS tags (필수 보강)
    {"code": "exam_allergy", "group": "exam",
     "aliases": ["allergy test","ige","atopy","알러지검사","알레르기검사","알러지","알레르기"]},
    {"code": "exam_heart", "group": "exam",
     "aliases": ["ecg","ekg","echo","cardiac","heart","심전도","심초음파","심장초음파","심장검사"]},
    {"code": "exam_eye", "group": "exam",
     "aliases": ["schirmer","fluorescein","iop","ophthalmic exam","안압","형광염색","안과검사","안과"]},
    {"code": "exam_skin", "group": "exam",
     "aliases": ["skin scraping","cytology","fungal test","malassezia","피부스크래핑","피부검사","진균","곰팡이","말라세지아"]},

    # --- Vaccines / Preventives ---
    {"code": "vaccine_comprehensive", "group": "vaccine",
     "aliases": ["dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine","종합백신","혼합백신","5종","6종"]},

    # ✅ Rabies / "Rabbies" OCR typo 대응
    {"code": "vaccine_rabies", "group": "vaccine",
     "aliases": ["rabies","rabbies","rabie","광견병","광견"]},

    {"code": "vaccine_kennel", "group": "vaccine",
     "aliases": ["kennel cough","bordetella","켄넬코프","기관지염백신","보르데텔라"]},

    {"code": "prevent_heartworm", "group": "preventive_med",
     "aliases": ["heartworm","hw","dirofilaria","heartgard","심장사상충","하트가드","넥스가드스펙트라","simparica trio","revolution"]},

    # --- Medicines ---
    {"code": "medicine_antibiotic", "group": "medicine",
     "aliases": ["antibiotic","abx","amoxicillin","clavamox","augmentin","cephalexin","convenia","doxycycline","metronidazole","baytril","항생제"]},
    {"code": "medicine_anti_inflammatory", "group": "medicine",
     "aliases": ["nsaid","anti-inflammatory","meloxicam","metacam","carprofen","rimadyl","onsior","galliprant","소염","소염제"]},
    {"code": "medicine_painkiller", "group": "medicine",
     "aliases": ["analgesic","tramadol","gabapentin","buprenorphine","진통","진통제"]},
    {"code": "medicine_steroid", "group": "medicine",
     "aliases": ["steroid","prednisone","prednisolone","dexamethasone","스테로이드"]},

    # --- Care / Checkup ---
    {"code": "checkup_general", "group": "checkup",
     "aliases": ["checkup","consult","opd","진료","상담","초진","재진","진찰","진료비","진료비용"]},

    # --- Fallback (마지막에만) ---
    {"code": "etc_other", "group": "etc",
     "aliases": ["기타","etc","other"]},
]

# ---- Normalization/token rules
def _normalize(s: str) -> str:
    s = (s or "").lower()
    return "".join(ch for ch in s if ch.isalnum() or ("가" <= ch <= "힣"))

def _tokenize(s: str) -> List[str]:
    raw = re.findall(r"[0-9a-zA-Z가-힣]+", s or "")
    return [t for t in raw if t]

def _is_short_ascii_token(norm: str) -> bool:
    if len(norm) > 2:
        return False
    return all(("0" <= c <= "9") or ("a" <= c <= "z") for c in norm)

def _is_single_latin_char(s: str) -> bool:
    if len(s) != 1:
        return False
    c = s.lower()
    return "a" <= c <= "z"


def _match_score(tag: Dict[str, Any], query: str) -> Tuple[int, Dict[str, Any]]:
    q_raw = (query or "").strip()
    if not q_raw:
        return 0, {}

    # iOS와 동일하게 영문 1글자 입력은 차단
    if _is_single_latin_char(q_raw):
        return 0, {}

    q_norm = _normalize(q_raw)
    if not q_norm:
        return 0, {}

    tokens = [_normalize(t) for t in _tokenize(q_raw)]
    token_set = set([t for t in tokens if t])

    best = 0
    hit = 0
    strong = False
    why: List[str] = []

    code_norm = _normalize(tag["code"])
    if code_norm == q_norm:
        return 230, {"why": ["code==query"]}

    for alias in tag.get("aliases", []):
        a = str(alias or "").strip()
        if not a:
            continue
        a_norm = _normalize(a)
        if not a_norm:
            continue

        # 짧은 약어(us/ua 등)는 contains 오탐이 크니 token match + exact match만
        if _is_short_ascii_token(a_norm):
            if a_norm == q_norm or a_norm in token_set:
                best = max(best, 160)
                hit += 1
                strong = True
                why.append(f"shortEqOrToken:{a}")
            continue

        if a_norm == q_norm:
            best = max(best, 190)
            hit += 1
            strong = True
            why.append(f"eq:{a}")
        elif q_norm.find(a_norm) >= 0:
            s = 120 + min(70, len(a_norm) * 3)
            best = max(best, s)
            hit += 1
            strong = True
            why.append(f"inQuery:{a}")
        elif a_norm.find(q_norm) >= 0:
            s = 95 + min(45, len(q_norm) * 3)
            best = max(best, s)
            hit += 1
            why.append(f"queryInAlias:{a}")

    if hit >= 2:
        best += min(35, hit * (8 if strong else 5))
        why.append(f"bonus:{hit}")

    return best, {"why": why[:8]}


def _pick_from_catalog(query: str, *, thresh: int, max_tags: int) -> Tuple[List[str], Dict[str, Any]]:
    scored: List[Tuple[str, int, Dict[str, Any]]] = []
    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, query)
        if s > 0:
            scored.append((tag["code"], s, ev))

    scored.sort(key=lambda x: x[1], reverse=True)

    picked: List[str] = []
    evidence: Dict[str, Any] = {"policy": "catalog", "query": query[:400], "candidates": []}

    for code, score, ev in scored[:15]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= int(thresh) and code not in picked:
            picked.append(code)
        if len(picked) >= int(max_tags):
            break

    return picked, evidence


def _tag_items(items: List[Dict[str, Any]], *, thresh: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate((items or [])[:120]):
        if not isinstance(it, dict):
            continue
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if not nm:
            continue

        best_code = None
        best_score = 0
        best_ev: Dict[str, Any] = {}

        for tag in TAG_CATALOG:
            s, ev = _match_score(tag, nm)
            if s > best_score:
                best_score = s
                best_code = tag["code"]
                best_ev = ev or {}

        if best_code and best_score >= int(thresh):
            out.append({
                "idx": idx,
                "itemName": nm[:200],
                "categoryTag": best_code,
                "score": int(best_score),
                "why": best_ev.get("why", []) if isinstance(best_ev, dict) else [],
            })

    return out


def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    # build a single query text from items + hospital
    parts: List[str] = []
    if hospital_name:
        parts.append(str(hospital_name))
    for it in (items or [])[:120]:
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if nm:
            parts.append(nm)

    query = " | ".join(parts)
    if not query.strip():
        return {"tags": [], "itemCategoryTags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}

    # ✅ threshold: record-level은 너무 빡빡하면 “아무것도 안 붙는” 문제가 커서 125 권장
    record_tags, evidence = _pick_from_catalog(query, thresh=125, max_tags=6)

    # ✅ item-level은 record보다 살짝 보수적으로 140 권장
    item_tags = _tag_items(items, thresh=140)

    # etc_other는 마지막 fallback (정말 아무것도 못 잡았을 때만)
    if not record_tags and (items or hospital_name):
        record_tags = ["etc_other"]

    return {"tags": record_tags, "itemCategoryTags": item_tags, "evidence": evidence}


