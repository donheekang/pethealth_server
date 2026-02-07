# ocr_policy.py (PetHealth+)
# OCR + minimal redaction + receipt parsing
# Returns: (webp_bytes, parsed_dict, hints_dict)
#
# 목표:
# - import-time에서 외부 SDK 때문에 죽지 않게 (Render 배포 시 중요)
# - Google Vision OCR 실패해도 500으로 서버를 죽이지 않고, 가능하면 Gemini(옵션)로 대체
# - parsed에 항상 text/ocrText/lines 키를 채워서 main.py fallback이 동작하도록 보장

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import threading
from datetime import date
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Regex / constants
# -----------------------------
_DATE_RE_1 = re.compile(r"\b(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b")
_DATE_RE_2 = re.compile(r"\b(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\b")

_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")  # 30,000 or 30000+
_HOSP_RE = re.compile(r"(병원\s*명|원\s*명)\s*[:：]?\s*(.+)$")

# PII patterns (best-effort, token-level redaction when bbox exists)
_PHONE_RE = re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b")
_BIZ_RE = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")
_CARD_RE = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")

_NOISE_TOKENS = [
    "고객", "고객번호", "고객 번호", "발행", "발행일", "발행 일", "사업자", "사업자등록", "대표",
    "전화", "주소", "serial", "sign", "승인", "카드", "현금", "부가세", "vat", "면세", "과세",
    "공급가", "소계", "합계", "총액", "총 금액", "총금액", "청구", "결제", "결제요청", "결제예정",
]

# Rabies signals (OCR typo/partials)
_RABIES_STRONG = ("rabies", "rabbies", "광견병", "광견")
_RABIES_PARTIAL_NORMS = {"ra", "rab", "rabi", "rabb"}


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", (s or "").lower())


def _is_noise_line(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    k = _norm(t)
    if len(k) < 2:
        return True
    low = t.lower()
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
# Concurrency guard (OCR)
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
# Pillow helpers (import lazily)
# -----------------------------
def _load_pil():
    try:
        from PIL import Image, ImageOps, ImageDraw  # type: ignore
        return Image, ImageOps, ImageDraw
    except Exception as e:
        raise RuntimeError("Pillow is required. Install: pip install Pillow") from e


def _ensure_max_pixels(img, max_pixels: int):
    if not max_pixels:
        return img
    w, h = img.size
    if (w * h) <= int(max_pixels):
        return img
    scale = math.sqrt(float(max_pixels) / float(w * h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh))


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
# Google Vision OCR (import lazily)
# -----------------------------
def _try_import_vision():
    try:
        from google.cloud import vision  # type: ignore
        return vision
    except Exception:
        return None


def _maybe_decode_b64_json(s: str) -> Optional[dict]:
    # "eyJ0eXBlIjoi..." 같은 base64 JSON 대응(옵션)
    ss = (s or "").strip()
    if not ss or len(ss) < 40:
        return None
    try:
        raw = base64.b64decode(ss).decode("utf-8", errors="strict")
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            return json.loads(raw)
    except Exception:
        return None
    return None


def _build_vision_client(google_credentials: str):
    # NOTE: 이 함수는 런타임에만 호출되어 import-time crash를 피함
    vision = _try_import_vision()
    if vision is None:
        raise RuntimeError("google-cloud-vision is not installed")

    from google.oauth2 import service_account  # type: ignore

    gc = (google_credentials or "").strip()

    # 1) JSON string
    if gc.startswith("{") and gc.endswith("}"):
        info = json.loads(gc)
        pk = info.get("private_key")
        if isinstance(pk, str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds)

    # 2) base64 JSON
    info2 = _maybe_decode_b64_json(gc)
    if isinstance(info2, dict):
        pk = info2.get("private_key")
        if isinstance(pk, str) and "\\n" in pk:
            info2["private_key"] = pk.replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(info2)
        return vision.ImageAnnotatorClient(credentials=creds)

    # 3) file path
    if gc and os.path.exists(gc):
        return vision.ImageAnnotatorClient.from_service_account_file(gc)

    # 4) default credentials (Render에선 보통 실패함)
    return vision.ImageAnnotatorClient()


def _vision_ocr(
    image_bytes: bytes,
    google_credentials: str,
    timeout_seconds: int,
) -> Tuple[str, Any]:
    vision = _try_import_vision()
    if vision is None:
        raise RuntimeError("google-cloud-vision is not installed")

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
# Redaction (bbox-based)
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
        name_part = re.sub(r"\b\d{1,2}\b", " ", name_part)  # qty 제거
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
            v = re.split(r"\s{2,}|/|\||,", v)[0].strip()
            if v:
                return v[:80]
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
    for ln in lines:
        if any(tok in ln for tok in ["시", "구", "동", "로", "길", "번지", "도"]):
            if len(ln) >= 10 and any(ch.isdigit() for ch in ln):
                return ln[:120]
    return None


def _contains_rabies_strong(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return ("rabies" in low) or ("rabbies" in low) or ("광견병" in text) or ("광견" in text)


def _find_rabies_partial_line(lines: List[str]) -> Optional[str]:
    for ln in (lines or [])[:400]:
        if not ln:
            continue
        if _is_noise_line(ln):
            continue
        n = _norm(ln)
        low = ln.lower()
        if ("rabies" in low) or ("rabbies" in low) or ("광견병" in ln) or ("광견" in ln):
            return ln
        # "r a" -> "ra"
        if n in _RABIES_PARTIAL_NORMS:
            return ln
    return None


def _parse_receipt_from_text(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    lines_all = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines_all if ln]

    parsed: Dict[str, Any] = {
        "hospitalName": _extract_hospital_name(text),
        "visitDate": None,
        "totalAmount": _extract_total_amount(text),
        "items": [],
        # downstream fallback keys
        "text": text or "",
        "ocrText": text or "",
        "lines": lines[:400],
    }

    vd = _parse_date_from_text(text)
    if vd:
        parsed["visitDate"] = vd.isoformat()

    items = _extract_items_from_text(text)

    # Rabies 보강: items에 없는데 text에 partial/strong이 보이면 한 줄 추가
    has_rabies = any(_contains_rabies_strong(str((it or {}).get("itemName") or "")) for it in items)
    rab_line = _find_rabies_partial_line(lines)
    if rab_line and not has_rabies:
        items.append({"itemName": "Rabies", "price": None, "categoryTag": None, "rawLine": rab_line[:120]})

    # 그래도 items가 없고 강한 signal이면 totalAmount를 Rabies로
    if (not items) and _contains_rabies_strong(text or ""):
        ta = parsed.get("totalAmount")
        if isinstance(ta, int) and ta >= 100:
            items = [{"itemName": "Rabies", "price": int(ta), "categoryTag": None}]

    parsed["items"] = items[:120]

    hints: Dict[str, Any] = {
        "addressHint": _extract_address_hint(text),
        "ocrTextPreview": (text or "")[:400],
        "hasLines": bool(lines),
    }
    return parsed, hints


# -----------------------------
# Gemini (optional) - direct HTTP
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
    out["totalAmount"] = int(ta) if isinstance(ta, int) and ta > 0 else None

    items = j.get("items")
    if isinstance(items, list):
        cleaned: List[Dict[str, Any]] = []
        for it in items[:120]:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("itemName") or "").strip()
            if not nm:
                continue
            pr = _coerce_int_amount(it.get("price"))
            if pr is not None and pr < 0:
                pr = None
            # normalize Rabies typo
            low = nm.lower()
            if "rabbies" in low or "rabies" in low:
                nm = "Rabies"
            cleaned.append({"itemName": _clean_item_name(nm), "price": pr, "categoryTag": None})
        out["items"] = cleaned
    return out


def _is_items_suspicious(items: List[Dict[str, Any]]) -> bool:
    if not items:
        return True
    bad = 0
    for it in items[:20]:
        nm = str((it or {}).get("itemName") or "")
        if _is_noise_line(nm):
            bad += 1
    return bad >= max(1, min(3, len(items)))


def _gemini_parse_receipt(
    *,
    image_webp_bytes: bytes,
    ocr_text: str,
    api_key: str,
    model: str,
    timeout_seconds: int = 10,
) -> Optional[Dict[str, Any]]:
    api_key = (api_key or "").strip()
    model = (model or "gemini-2.5-flash").strip()
    if not api_key:
        return None

    prompt = (
        "You are a receipt parser for Korean veterinary receipts.\n"
        "Return ONLY valid JSON with keys:\n"
        '  hospitalName (string|null), visitDate (YYYY-MM-DD|null), totalAmount (integer|null),\n'
        '  items (array of {itemName:string, price:integer|null}).\n'
        "Rules:\n"
        "- items must be REAL treatment/vaccine/medicine line-items.\n"
        "- Do NOT include totals, taxes, approvals as items.\n"
        "- If you see 'Rabbies' typo, normalize to 'Rabies'.\n"
        "- If you see partial 'ra'/'rab' on receipt, interpret as Rabies if reasonable.\n"
    )

    b64 = base64.b64encode(image_webp_bytes).decode("ascii")
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/webp", "data": b64}},
    ]
    if isinstance(ocr_text, str) and ocr_text.strip():
        parts.append({"text": "OCR text (may be noisy):\n" + ocr_text[:4000]})

    try:
        out = _call_gemini_generate_content(api_key=api_key, model=model, parts=parts, timeout_seconds=timeout_seconds)
        j = _extract_json_from_model_text(out)
        return j if isinstance(j, dict) else None
    except Exception:
        return None


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
    # Gemini optional (env fallback)
    gemini_enabled: Optional[bool] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model_name: Optional[str] = None,
    gemini_timeout_seconds: Optional[int] = None,
    **kwargs,
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      webp_bytes: redacted (best-effort) WEBP bytes for storage
      parsed: {hospitalName, visitDate, totalAmount, items, text, ocrText, lines}
      hints:  {ocrEngine, ocrOk, ocrError, geminiUsed, geminiError, addressHint, ...}
    """
    if not raw_bytes:
        raise ValueError("empty raw bytes")

    Image, ImageOps, ImageDraw = _load_pil()

    # 0) Load image safely
    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg

    # 1) Resize / harden
    img = _ensure_max_pixels(img, int(image_max_pixels or 0))
    img = _resize_to_width(img, int(receipt_max_width or 0))

    # OCR input as PNG (sharper text)
    ocr_buf = io.BytesIO()
    img.save(ocr_buf, format="PNG")
    ocr_image_bytes = ocr_buf.getvalue()

    hints: Dict[str, Any] = {
        "ocrEngine": None,
        "ocrOk": False,
        "ocrError": None,
        "geminiUsed": False,
        "geminiError": None,
    }

    # 2) OCR (best-effort)
    ocr_text = ""
    vision_resp = None

    sema = _get_sema(int(ocr_max_concurrency or 4))
    acquired = sema.acquire(timeout=float(ocr_sema_acquire_timeout_seconds or 1.0))
    if acquired:
        try:
            ocr_text, vision_resp = _vision_ocr(
                ocr_image_bytes,
                google_credentials=google_credentials,
                timeout_seconds=int(ocr_timeout_seconds or 12),
            )
            hints["ocrEngine"] = "google_vision"
            hints["ocrOk"] = True
        except Exception as e:
            # OCR 실패해도 서버를 죽이지 않음
            hints["ocrEngine"] = "google_vision"
            hints["ocrOk"] = False
            hints["ocrError"] = (str(e) or repr(e))[:200]
            ocr_text = ""
            vision_resp = None
        finally:
            try:
                sema.release()
            except Exception:
                pass
    else:
        # 동시성 때문에 OCR 못 잡아도 서버를 죽이지 않음
        hints["ocrEngine"] = "google_vision"
        hints["ocrOk"] = False
        hints["ocrError"] = "semaphore_acquire_timeout"
        ocr_text = ""
        vision_resp = None

    # 3) Redaction (best-effort; only if bbox exists)
    redacted = _redact_image_with_vision_tokens(img.copy(), vision_resp) if vision_resp is not None else img
    webp_bytes = _to_webp_bytes(redacted, quality=int(receipt_webp_quality or 85))

    # 4) Parse from OCR text (may be empty)
    parsed, parse_hints = _parse_receipt_from_text(ocr_text or "")
    hints.update(parse_hints or {})

    # 5) Gemini assist
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))

    if g_enabled and g_key.strip():
        try:
            items_now = parsed.get("items") if isinstance(parsed.get("items"), list) else []
            need_help = (not hints.get("ocrOk")) or _is_items_suspicious(items_now)
            if need_help:
                gj = _gemini_parse_receipt(
                    image_webp_bytes=webp_bytes,
                    ocr_text=ocr_text or "",
                    api_key=g_key,
                    model=g_model,
                    timeout_seconds=g_timeout,
                )
                if isinstance(gj, dict):
                    gparsed = _normalize_gemini_parsed(gj)

                    # merge
                    if gparsed.get("items"):
                        parsed["items"] = gparsed["items"]
                        hints["geminiUsed"] = True

                    if not parsed.get("hospitalName") and gparsed.get("hospitalName"):
                        parsed["hospitalName"] = gparsed["hospitalName"]
                    if not parsed.get("visitDate") and gparsed.get("visitDate"):
                        parsed["visitDate"] = gparsed["visitDate"]
                    if (not parsed.get("totalAmount")) and gparsed.get("totalAmount"):
                        parsed["totalAmount"] = gparsed["totalAmount"]

        except Exception as e:
            hints["geminiError"] = (str(e) or repr(e))[:200]

    # 6) Final sanitize (downstream fallback keys guaranteed)
    if not isinstance(parsed.get("items"), list):
        parsed["items"] = []
    parsed["items"] = parsed["items"][:120]

    if parsed.get("totalAmount") is not None:
        try:
            parsed["totalAmount"] = int(parsed["totalAmount"])
        except Exception:
            parsed["totalAmount"] = None

    if not isinstance(parsed.get("text"), str):
        parsed["text"] = ocr_text or ""
    if not isinstance(parsed.get("ocrText"), str):
        parsed["ocrText"] = ocr_text or ""
    if not isinstance(parsed.get("lines"), list):
        parsed["lines"] = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()][:400]

    return webp_bytes, parsed, hints


