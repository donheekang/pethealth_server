# ocr_policy.py (PetHealth+)
# OCR + minimal redaction + receipt parsing
# Returns: (webp_bytes, parsed_dict, hints_dict)
#
# ✅ Key upgrades (2026-02):
# - "Rabies/Rabbies/광견병" 뿐 아니라 OCR이 잘라낸 "ra" 같은 짧은 조각도
#   "백신/접종/주사/vaccine" 컨텍스트가 있으면 Rabies로 보정
# - item 추출에서 "검사" 같은 단어를 noise로 과하게 제거하던 구조 개선
# - Gemini(옵션) fallback이 "items 비었거나 의심스러울 때" + "rabies 의심인데 못잡을 때" 동작
# - parsed에 ocrText(프리뷰)도 넣어 tag_policy에서 활용 가능

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

# 3+ digits or comma-style
_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")

_HOSP_RE = re.compile(r"(병원\s*명|원\s*명)\s*[:：]?\s*(.+)$")

# PII patterns for image redaction (best-effort)
_PHONE_RE = re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b")
_BIZ_RE = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")
_CARD_RE = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")

# 합계/결제/승인/카드 등 "아이템이 아닌" 라인 필터용
_TOTALISH_TOKENS = [
    "소계", "합계", "총", "총액", "총금액", "총 금액", "청구", "결제", "결제요청", "결제예정",
    "승인", "카드", "현금", "부가세", "vat", "면세", "과세", "공급가", "거스름", "change",
    "발행", "발행일", "발행 일", "영수", "영수증", "매출", "거래", "승인번호", "승인 번호",
]

# header/metadata-ish 라인 제거용(너무 과하면 items를 잃음 → "검사" 같은 일반어는 넣지 않음)
_META_NOISE_TOKENS = [
    "고객", "고객번호", "고객 번호",
    "사업자", "사업자등록", "대표",
    "전화", "주소", "serial", "sign",
]

# Rabies / vaccine context
_RABIES_STRONG = [
    "rabies", "rabbies", "광견병", "광견",
]
# OCR이 잘라먹는 흔한 형태들 (너무 공격적이면 오탐 ↑ → 아래 컨텍스트 조건과 함께 사용)
_RABIES_WEAK = [
    "rab", "rabi", "rabb", "rabis",
    "ra",  # ✅ 핵심: ra 단독은 컨텍스트가 있을 때만 Rabies로 인정
]

_VACCINE_CONTEXT = [
    "vaccine", "vacc", "shot", "inj", "injection", "im", "sc",
    "접종", "백신", "예방", "주사",
]


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", (s or "").lower())


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


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


def _looks_totalish(line: str) -> bool:
    low = (line or "").strip().lower()
    if not low:
        return False
    for t in _TOTALISH_TOKENS:
        if t.lower() in low:
            return True
    return False


def _looks_meta_noise(line: str) -> bool:
    low = (line or "").strip().lower()
    if not low:
        return True
    k = _norm(low)
    if len(k) < 2:
        return True
    for t in _META_NOISE_TOKENS:
        if t.lower() in low:
            return True
    return False


def _has_vaccine_context(text: str) -> bool:
    low = (text or "").lower()
    for t in _VACCINE_CONTEXT:
        if t in low:
            return True
    return False


def _contains_rabies_strong(text: str) -> bool:
    low = (text or "").lower()
    for t in _RABIES_STRONG:
        if t in low:
            return True
    return False


def _contains_rabies_weak(text: str) -> bool:
    # weak는 'ra' 포함 → 반드시 컨텍스트로 보정
    low = (text or "").lower()
    for t in _RABIES_WEAK:
        if t in low:
            return True
    return False


def _canonicalize_rabies_name(name: str, *, vaccine_context: bool) -> str:
    """
    itemName 후보를 Rabies로 보정
    - "rabies/rabbies/광견병/광견" => Rabies
    - "ra/rab/rabi/rabb..." => 백신 컨텍스트가 있을 때만 Rabies
    """
    raw = (name or "").strip()
    if not raw:
        return raw

    low = raw.lower()
    if any(x in low for x in ["rabies", "rabbies"]) or ("광견병" in raw) or ("광견" in raw):
        return "Rabies"

    n = _norm(raw)
    if n in ("rabies", "rabbies", "광견병", "광견"):
        return "Rabies"

    # ✅ "ra" 같은 초단축은 컨텍스트가 있을 때만
    if vaccine_context and n in ("ra", "rab", "rabi", "rabb", "rabis"):
        return "Rabies"

    return raw


def _find_rabies_line_price(text: str) -> Optional[int]:
    """
    rabies가 있는 "같은 줄"의 금액을 뽑아보는 용도.
    - "Rabbies 30,000" / "ra 30000" 등
    """
    if not text:
        return None
    vaccine_ctx = _has_vaccine_context(text)
    lines = [_normalize_spaces(ln) for ln in (text or "").splitlines()]
    best: Optional[int] = None

    for ln in lines:
        if not ln:
            continue
        low = ln.lower()

        strong = _contains_rabies_strong(ln)
        weak = _contains_rabies_weak(ln) and vaccine_ctx  # ✅ ra 같은건 컨텍스트 필요
        if not (strong or weak):
            continue

        nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(ln)]
        nums = [n for n in nums if n >= 100]
        if not nums:
            continue
        cand = max(nums)
        if best is None or cand > best:
            best = cand

    return best


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
    from google.cloud import vision
    from google.oauth2 import service_account

    gc = (google_credentials or "").strip()

    # ✅ main.py env 키 오타가 흔해서, 여기서도 안전망
    if not gc:
        gc = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if not gc:
        gc = (os.getenv("GOOGLE_APPLICATION_CREDENTIALIALS", "") or "").strip()  # 흔한 오타

    if not gc:
        return vision.ImageAnnotatorClient()

    # JSON string?
    if gc.startswith("{") and gc.endswith("}"):
        info = json.loads(gc)
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

    # document_text_detection이 영수증에 더 잘 맞음
    resp = client.document_text_detection(image=img, timeout=float(timeout_seconds or 12))
    full_text = ""
    try:
        if resp and resp.full_text_annotation and resp.full_text_annotation.text:
            full_text = resp.full_text_annotation.text
    except Exception:
        full_text = ""

    # 혹시 full_text가 비었으면 text_detection도 한 번 시도
    if not full_text:
        try:
            resp2 = client.text_detection(image=img, timeout=float(timeout_seconds or 12))
            if resp2 and resp2.text_annotations:
                full_text = str(resp2.text_annotations[0].description or "")
                resp = resp2
        except Exception:
            pass

    return full_text, resp


# -----------------------------
# Redaction (best-effort)
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

            xs, ys = [], []
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
    s = (name or "").strip().replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lstrip("*•·-—").strip()
    return s[:200]


def _extract_items_from_text(text: str) -> List[Dict[str, Any]]:
    """
    ✅ 개선 포인트:
    - 기존처럼 "검사" 같은 단어를 무조건 noise 처리하지 않음
    - 먼저 금액 존재 여부로 아이템 후보를 잡고,
      그 다음 total/결제 계열을 제외하는 방식
    """
    if not text:
        return []
    lines = [_normalize_spaces(ln) for ln in (text or "").splitlines()]
    out: List[Dict[str, Any]] = []
    seen = set()

    vaccine_ctx = _has_vaccine_context(text)

    for ln in lines:
        if not ln:
            continue

        # 1) 금액 존재해야 아이템 후보
        nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(ln)]
        nums = [n for n in nums if n >= 100]
        if not nums:
            continue

        # 2) total/결제 라인은 아이템에서 제외
        if _looks_totalish(ln):
            continue

        # 3) 메타(주소/전화/사업자) 라인 제외 (하지만 너무 공격적이면 안됨)
        if _looks_meta_noise(ln) and len(nums) <= 1:
            # 주소 라인도 숫자 들어갈 수 있어서, 금액이 1개이고 메타면 제외
            continue

        price = nums[-1]

        # 금액들 지우고 이름 추출
        name_part = _AMOUNT_RE.sub(" ", ln)
        name_part = re.sub(r"\b\d{1,2}\b", " ", name_part)  # qty/short junk
        name_part = _clean_item_name(name_part)
        if not name_part:
            continue

        # Rabies 보정
        name_part = _canonicalize_rabies_name(name_part, vaccine_context=vaccine_ctx)

        key = (_norm(name_part), int(price))
        if key in seen:
            continue
        seen.add(key)

        out.append({"itemName": name_part, "price": int(price), "categoryTag": None})

    return out[:120]


def _extract_total_amount(text: str) -> Optional[int]:
    if not text:
        return None
    lines = [_normalize_spaces(ln) for ln in (text or "").splitlines()]
    candidates: List[int] = []

    for ln in lines:
        if not ln:
            continue
        if _looks_totalish(ln):
            nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(ln)]
            nums = [n for n in nums if n >= 100]
            candidates.extend(nums)

    # fallback: take max amount anywhere
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
    lines = [_normalize_spaces(ln) for ln in (text or "").splitlines() if _normalize_spaces(ln)]
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
    lines = [_normalize_spaces(ln) for ln in (text or "").splitlines() if _normalize_spaces(ln)]
    for ln in lines:
        if "주소" in ln:
            return ln[:120]
    for ln in lines:
        if any(tok in ln for tok in ["시", "구", "동", "로", "길", "번지", "도"]):
            if len(ln) >= 10 and any(ch.isdigit() for ch in ln):
                return ln[:120]
    return None


def _parse_receipt_from_text(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    vaccine_ctx = _has_vaccine_context(text)
    rabies_strong = _contains_rabies_strong(text)
    rabies_weak = _contains_rabies_weak(text) and vaccine_ctx

    parsed: Dict[str, Any] = {
        "hospitalName": _extract_hospital_name(text),
        "visitDate": None,
        "totalAmount": _extract_total_amount(text),
        "items": [],
        # ✅ tag_policy/main.py에서 쓰기 좋게 OCR 텍스트 프리뷰도 넣음
        "ocrText": (text or "")[:6000],
    }

    vd = _parse_date_from_text(text)
    if vd:
        parsed["visitDate"] = vd.isoformat()

    items = _extract_items_from_text(text)

    # ✅ items에 Rabies 보정 2차 적용 (Gemini merge 대비)
    for it in items:
        it["itemName"] = _canonicalize_rabies_name(str(it.get("itemName") or ""), vaccine_context=vaccine_ctx)

    # ✅ 강제 보강:
    # - 텍스트에 rabies 징후가 있는데 items가 없거나,
    # - items에 Rabies가 하나도 없으면(특히 OCR이 ra만 남긴 케이스)
    low = (text or "").lower()
    has_rabies_item = any(_norm(str(it.get("itemName") or "")) in ("rabies", "rabbies", "광견병", "광견") or str(it.get("itemName") or "") == "Rabies" for it in items)

    if (rabies_strong or rabies_weak) and (not items or not has_rabies_item):
        rab_price = _find_rabies_line_price(text)
        if rab_price is None:
            ta = parsed.get("totalAmount")
            rab_price = int(ta) if isinstance(ta, int) and ta >= 100 else None

        # 너무 공격적으로 중복 추가하지 않게, 이미 Rabies 있으면 스킵
        if not has_rabies_item:
            items.append({"itemName": "Rabies", "price": rab_price, "categoryTag": None})

    parsed["items"] = items[:120]

    hints: Dict[str, Any] = {
        "addressHint": _extract_address_hint(text),
        "ocrTextPreview": (text or "")[:400],
        "rabiesDetected": bool(rabies_strong or rabies_weak),
        "rabiesEvidence": {
            "strong": bool(rabies_strong),
            "weakWithVaccineContext": bool(rabies_weak),
            "vaccineContext": bool(vaccine_ctx),
        },
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
            "maxOutputTokens": 768,
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
        "- Do NOT include totals, taxes, approval, '결제/합계/소계/청구금액' as items.\n"
        "- If you see Rabies but OCR typo like 'Rabbies' or truncated like 'ra' in vaccine context, normalize to 'Rabies'.\n"
        "- Prefer extracting the actual Rabies line item if present.\n"
    )

    b64 = base64.b64encode(image_bytes).decode("ascii")
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/webp", "data": b64}},
        {"text": "OCR text (may be noisy):\n" + (ocr_text or "")[:6000]},
    ]

    try:
        out = _call_gemini_generate_content(api_key=api_key, model=model, parts=parts, timeout_seconds=timeout_seconds)
        j = _extract_json_from_model_text(out)
        if isinstance(j, dict):
            return j
    except Exception:
        # fallback: text-only
        try:
            parts2 = [{"text": prompt + "\n\nHere is OCR text:\n" + (ocr_text or "")[:8000]}]
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

    # total/결제 계열만 잔뜩이면 의심
    bad = 0
    for it in items[:20]:
        nm = str((it or {}).get("itemName") or "")
        if _looks_totalish(nm) or _looks_meta_noise(nm):
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
            if _looks_totalish(nm) or _looks_meta_noise(nm):
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

    # bytes for OCR (PNG keeps text sharp)
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

    # 3) redact image (best-effort)
    redacted = _redact_image_with_vision_tokens(img.copy(), vision_resp)
    webp_bytes = _to_webp_bytes(redacted, quality=int(receipt_webp_quality or 85))

    # 4) parse from OCR text
    parsed, hints = _parse_receipt_from_text(ocr_text or "")
    hints["ocrEngine"] = "google_vision"
    hints["geminiUsed"] = False

    # 5) Gemini assist (fallback)
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))

    # ✅ gemini를 "확실히" 돌리고 싶은 케이스:
    # - items suspicious/empty
    # - rabies 의심(ra 포함)인데 items에 Rabies가 없을 때
    vaccine_ctx = _has_vaccine_context(ocr_text or "")
    rabies_strong = _contains_rabies_strong(ocr_text or "")
    rabies_weak = _contains_rabies_weak(ocr_text or "") and vaccine_ctx
    has_rabies_item = any(str(it.get("itemName") or "") == "Rabies" for it in (parsed.get("items") or [] if isinstance(parsed.get("items"), list) else []))

    should_try_gemini = _is_items_suspicious(parsed.get("items") if isinstance(parsed.get("items"), list) else [])
    if (rabies_strong or rabies_weak) and (not has_rabies_item):
        should_try_gemini = True

    if g_enabled and g_key.strip() and should_try_gemini:
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

                # merge items
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
            hints["geminiError"] = str(e)[:200]

    # ✅ 최종 Rabies 보정(혹시 Gemini가 ra로 준 경우 등)
    vaccine_ctx2 = _has_vaccine_context(ocr_text or "")
    final_items = parsed.get("items") if isinstance(parsed.get("items"), list) else []
    for it in final_items:
        it["itemName"] = _canonicalize_rabies_name(str(it.get("itemName") or ""), vaccine_context=vaccine_ctx2)

    parsed["items"] = final_items[:120]

    # ensure types
    if parsed.get("totalAmount") is not None:
        try:
            parsed["totalAmount"] = int(parsed["totalAmount"])
        except Exception:
            parsed["totalAmount"] = None

    return webp_bytes, parsed, hints


