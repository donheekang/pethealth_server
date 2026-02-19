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


# =========================================================
# Custom exception classes (used by main.py)
# =========================================================
class OCRTimeoutError(Exception):
    pass

class OCRConcurrencyError(Exception):
    pass

class OCRImageError(Exception):
    pass


# -----------------------------
# Regex / constants
# -----------------------------
_DATE_RE_1 = re.compile(r"\b(20\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})\b")
_DATE_RE_2 = re.compile(r"\b(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\b")
_AMOUNT_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")
_HOSP_RE = re.compile(r"(병원\s*명|원\s*명)\s*[:：]?\s*(.+)$")
_PHONE_RE = re.compile(r"\b\d{2,3}-\d{3,4}-\d{4}\b")
_BIZ_RE = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")
_CARD_RE = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")
_KOREAN_NAME_RE = re.compile(r"^[가-힣]{2,4}$")
_NAME_LABEL_KEYWORDS = {
    "고객명", "고객이름", "고객 이름", "보호자", "보호자명",
    "이름", "성명", "수신인", "수납자", "환자명",
    "대표자", "대표", "원장",
}
_RABIES_RE = re.compile(
    r"(rabies|rabbies|rabie|rabis|rabiess|rables|rabeis|ra\b|r\s*[/\-\._ ]\s*a\b|광견병|광견)",
    re.IGNORECASE,
)

# -----------------------------
# Noise tokens (for filtering non-item lines)
# -----------------------------
_NOISE_TOKENS = [
    "고객", "고객번호", "고객 번호", "고객명", "고객 이름",
    "사업자", "사업자등록", "사업자 등록", "대표",
    "전화", "연락처", "주소",
    "serial", "sign", "승인", "카드", "현금",
    "부가세", "vat", "면세", "과세",
    "공급가", "공급가액", "과세공급가액", "부가세액",
    "소계", "합계", "총", "총액", "총 금액", "총금액",
    "청구", "청구금액", "결제", "결제요청", "결제요청:", "결제예정",
    "거래", "거래일", "거래 일",
    "날짜", "일자", "방문일", "방문 일", "진료일", "진료 일", "발행", "발행일", "발행 일",
    "퇴원",
    "항목", "단가", "수량",
    "동물명", "환자", "환자명", "품종",
    "경기도", "서울", "인천", "부산", "대구", "대전", "광주", "울산", "세종",
    "충북", "충남", "전북", "전남", "경북", "경남", "강원", "제주",
]

_ADDRESS_RE = re.compile(
    r"(경기도|서울|인천|부산|대구|대전|광주|울산|세종|충청|전라|경상|강원|제주)"
    r"|(\S+시\s+\S+구)"
    r"|(\S+[시군구]\s+\S+[동읍면로길])"
    r"|(\d+번지)"
    r"|(\(\S+동[,\s])"
)


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", (s or "").lower())


_NOISE_TOKENS_NORM = [_norm(x) for x in _NOISE_TOKENS if _norm(x)]


def _looks_like_date_line(t: str) -> bool:
    if not t:
        return False
    if _DATE_RE_1.search(t) or _DATE_RE_2.search(t):
        return True
    k = _norm(t)
    if re.fullmatch(r"20\d{2}(0?\d|1[0-2])(0?\d|[12]\d|3[01])", k):
        return True
    if re.fullmatch(r"20\d{2}(0?\d|1[0-2])(0?\d|[12]\d|3[01])\d{0,6}", k) and len(k) <= 12:
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
    if _looks_like_date_line(t):
        return True
    if _ADDRESS_RE.search(t):
        return True
    has_letter = any(("a" <= ch.lower() <= "z") or ("가" <= ch <= "힣") for ch in t)
    has_digit = any(ch.isdigit() for ch in t)
    if has_digit and (not has_letter) and len(k) <= 12:
        return True
    for xn in _NOISE_TOKENS_NORM:
        if xn and xn in k:
            return True
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
        cleaned = s.replace(" ", "")
        # 음수 처리: "-88,000" or "-88000"
        is_negative = cleaned.startswith("-")
        if is_negative:
            cleaned = cleaned[1:]
        m = _AMOUNT_RE.search(cleaned)
        if not m:
            return None
        try:
            val = int(m.group(0).replace(",", ""))
            return -val if is_negative else val
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
# ✅ v2.6.0: 영수증 이미지 전처리 (OCR 인식률 향상)
# -----------------------------
def _preprocess_for_ocr(img):
    """
    감열지 영수증 대비 강화 + 선명도 + 노이즈 제거.
    Pillow의 ImageEnhance/ImageFilter만 사용 (추가 의존성 없음).
    """
    try:
        from PIL import ImageEnhance, ImageFilter

        # 1) 그레이스케일 변환 → 색상 노이즈 제거
        gray = img.convert("L")

        # 2) 대비 강화 (감열지는 대비가 낮음)
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.8)  # 1.8배 대비 강화

        # 3) 밝기 살짝 올림 (어두운 영수증 보정)
        enhancer = ImageEnhance.Brightness(gray)
        gray = enhancer.enhance(1.1)

        # 4) 선명도 강화 (흐릿한 글씨 대응)
        enhancer = ImageEnhance.Sharpness(gray)
        gray = enhancer.enhance(2.0)

        # 5) 가벼운 노이즈 제거
        gray = gray.filter(ImageFilter.MedianFilter(size=3))

        # 다시 RGB로 (OCR API 호환)
        return gray.convert("RGB")
    except Exception:
        return img


# -----------------------------
# Google Vision OCR (optional)
# -----------------------------
def _build_vision_client(google_credentials: str):
    from google.cloud import vision
    from google.oauth2 import service_account
    gc = (google_credentials or "").strip()
    if not gc:
        env1 = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        env2 = (os.getenv("GOOGLE_APPLICATION_CREDENTIALIALS") or "").strip()
        gc = env1 or env2
    if not gc:
        return vision.ImageAnnotatorClient()
    if gc.startswith("{") and gc.endswith("}"):
        info = json.loads(gc)
        pk = info.get("private_key")
        if isinstance(pk, str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(info)
        return vision.ImageAnnotatorClient(credentials=creds)
    if os.path.exists(gc):
        return vision.ImageAnnotatorClient.from_service_account_file(gc)
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
            _fill_rect(draw, a)

        tokens = list(anns[1:])
        label_indices: set = set()

        for i, a in enumerate(tokens):
            desc = str(getattr(a, "description", "") or "").strip()
            desc_clean = desc.replace(" ", "")
            if desc_clean in _NAME_LABEL_KEYWORDS or desc_clean.rstrip(":：") in _NAME_LABEL_KEYWORDS:
                label_indices.add(i)
                continue
            if i > 0:
                prev = str(getattr(tokens[i - 1], "description", "") or "").strip()
                combo = (prev + desc).replace(" ", "")
                if combo in _NAME_LABEL_KEYWORDS or combo.rstrip(":：") in _NAME_LABEL_KEYWORDS:
                    label_indices.add(i)

        for li in label_indices:
            for offset in range(1, 4):
                ni = li + offset
                if ni >= len(tokens):
                    break
                nd = str(getattr(tokens[ni], "description", "") or "").strip()
                if nd in (":", "：", "-", ")", "(", "/", "·"):
                    continue
                if _KOREAN_NAME_RE.match(nd):
                    _fill_rect(draw, tokens[ni])
                    break
                break

        return img
    except Exception:
        return img


def _fill_rect(draw, annotation) -> None:
    poly = getattr(annotation, "bounding_poly", None)
    if not poly or not getattr(poly, "vertices", None):
        return
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
        return
    x0, x1 = max(0, min(xs)), max(xs)
    y0, y1 = max(0, min(ys)), max(ys)
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))


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
    s = (name or "").strip()
    if not s:
        return s
    low = s.lower()
    if _RABIES_RE.search(s):
        return "Rabies"
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
        "ocrText": (text or "")[:8000],
    }
    vd = _parse_date_from_text(text)
    if vd:
        parsed["visitDate"] = vd.isoformat()
    items = _extract_items_from_text(text)
    low = (text or "").lower()
    has_rabies = bool(_RABIES_RE.search(text or ""))
    if has_rabies:
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
# Gemini (Generative Language API)
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
    media_resolution: Optional[str] = None,
) -> str:
    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    # ✅ Gemini 3: thinking에 토큰을 많이 쓰므로 maxOutputTokens를 크게 설정
    # thinkingBudget으로 사고 토큰을 제한하고, 실제 응답에 충분한 여유 확보
    gen_config: Dict[str, Any] = {
        "temperature": 0.0,
        "maxOutputTokens": 16384,
        "thinkingConfig": {"thinkingBudget": 2048},
    }

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": gen_config,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=float(timeout_seconds or 10)) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    j = json.loads(data)
    # ✅ Gemini 3: thinking part를 건너뛰고 실제 응답 텍스트 추출
    parts_out = (
        (j.get("candidates") or [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    txt_parts = []
    for p in parts_out:
        # thought=true인 파트는 건너뛰기 (Gemini 3 thinking 모드)
        if p.get("thought") or p.get("thinking"):
            continue
        t = p.get("text", "")
        if t:
            txt_parts.append(t)
    # thinking 파트밖에 없으면 마지막 파트라도 사용
    if not txt_parts and parts_out:
        last = parts_out[-1].get("text", "")
        if last:
            txt_parts.append(last)
    return "\n".join(txt_parts).strip()


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
    model = (model or "gemini-3-flash-preview").strip()
    if not api_key or not model:
        return None
    prompt = (
        "You are a receipt parser for Korean veterinary receipts.\n"
        "Return ONLY valid JSON with keys:\n"
        '  hospitalName (string|null), visitDate (YYYY-MM-DD|null), totalAmount (integer|null),\n'
        '  items (array of {itemName:string, price:integer|null}).\n'
        "Rules:\n"
        "- items must be REAL treatment/vaccine/medicine line-items only.\n"
        "- Do NOT include date lines (e.g. '날짜: 2025-11-28') as items.\n"
        "- Do NOT include totals/taxes/payment lines as items.\n"
        "- Do NOT include addresses (시/구/동/로/길) as items.\n"
        "- Do NOT include phone numbers, business registration numbers as items.\n"
        "- Do NOT include hospital name/address as items.\n"
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


# =========================================================
# ✅ Gemini-first receipt parsing prompt (세분화 태그 동기화)
# =========================================================
_GEMINI_RECEIPT_PROMPT = """\
You are a precision OCR data extractor for Korean veterinary hospital receipts.
Your ONLY job: read the receipt image and output a JSON object with EXACT data from the receipt.
Do NOT interpret, rename, translate, standardize, or infer anything. Just copy what you see.

CRITICAL: Your item prices MUST add up to the totalAmount on the receipt (within 100원).
If they don't match, you MUST re-read the receipt and find missing items or fix wrong prices.
This is the #1 quality requirement. DO NOT skip this check.

Return ONLY valid JSON (no markdown, no code fences, no explanation):

{
  "hospitalName": "exact hospital name from receipt" or null,
  "visitDate": "YYYY-MM-DD" or null,
  "totalAmount": integer or null,
  "discountAmount": integer or null,
  "items": [
    {
      "itemName": "exact item text from receipt",
      "price": integer_or_null,
      "originalPrice": integer_or_null,
      "discount": integer_or_null
    }
  ]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1: EXTRACT EVERY LINE ITEM — ZERO EXCEPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Read the receipt TOP to BOTTOM. Every row that has a name + price = one item.
- Include ALL categories found on receipts:
  · 진료비/진찰료/재진/초진 (consultation fees)
  · 검사 (tests): 혈액, CBC, 화학, 전해질, CRP, X-ray, 초음파, 심전도, 세포검사 등
  · 처치 (procedures): 수액, 주사, 소독, 드레싱, 천자, 카테터 등
  · 수술 (surgery): 마취, 중성화, 발치, 정형외과 등
  · 약 (medication): 내복약, 외용제, 연고, 점안액, 귀약, 주사약, 항생제 등
  · 미용 (grooming): 목욕, 미용, 발바닥 밀기 등
  · 용품 (supplies): 넥칼라, 사료, 캔, 간식, 보조제 등
  · 입원 (hospitalization): 입원비, 식이 관리, 모니터링 등
  · 할인 (discounts): 절사할인, 쿠폰할인, 회원할인 등
  · 기타: 제증명, 진단서, 기타 비용
- Items with *, **, †, brackets [], or any symbol → STILL extract them.
- If the receipt has MULTIPLE SECTIONS (e.g. "진료 내역", "용품 내역") → extract from ALL sections.
- TABLE FORMAT: Many receipts have table columns (항목명 | 수량 | 단가 | 금액).
  → Extract EVERY ROW in the table. Do NOT skip rows just because they look similar.
  → 예: "혈액(혈청)-기본(7항목)" and "혈액(혈청)-Chemistry 1항" are DIFFERENT items — extract BOTH.
- SIMILAR NAMES: Items may have similar names but different details (e.g. 혈액 검사 types).
  → Each row is a SEPARATE item. Never merge similar-looking rows.
  → CRITICAL EXAMPLE: A receipt may have ALL of these as separate items:
    · *혈액(혈구)-CBC → 30,000
    · *혈액(혈액가스분석) → 50,000
    · *혈액(혈청)-기본(7항목) → 60,000
    · *혈액(혈청)-Chemistry 1항 → 30,000
    · *혈액-젖산측정(Lactate) → 15,000
  → These are 5 DIFFERENT items! Extract ALL of them, not just 1-2.
  → Similarly, multiple "고가약물-XXX" lines are each separate items.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2: itemName = EXACT COPY FROM RECEIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Copy the item name CHARACTER BY CHARACTER from the receipt.
- 예시:
  · "진찰료-초진" → "진찰료-초진" (NOT "진료비", NOT "기본진료")
  · "*병리검사(혈액-CRP)" → "*병리검사(혈액-CRP)"
  · "검사-귀-set(검이경,현미경도말)" → "검사-귀-set(검이경,현미경도말)" (NOT "귀검사")
  · "처치-복수천자(복수 제거 목적)" → "처치-복수천자(복수 제거 목적)" (NOT "처치료")
  · "내복약 1일 2회 (5~10kg)" → "내복약 1일 2회 (5~10kg)"
  · "로얄- cat) 마더앤 베이비캣 소프트 무스" → "로얄- cat) 마더앤 베이비캣 소프트 무스"
- Keep ALL: parentheses (), brackets [], asterisks *, hyphens -, slashes /, dots.
- Do NOT shorten, summarize, translate to English, or replace with a category name.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3: PRICE = EXACT NUMBER FROM RECEIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Copy the EXACT integer from the receipt. Do NOT round, guess, or approximate.
- 예시: receipt shows "77,000" → price: 77000 (NOT 71000, NOT 70000)
       receipt shows "9,900" → price: 9900 (NOT 5500, NOT 10000)
       receipt shows "2,600" → price: 2600 (NOT 2000)
- QUANTITY HANDLING:
  · If receipt shows "수량: 4 × 단가: 22,000 = 88,000" → price: 88000 (use the total)
  · If receipt shows "내복약 1일 × 7 = 31,500" → price: 31500 (use the total)
  · If only unit price visible with quantity → multiply: price = 수량 × 단가
  · IMPORTANT: Some receipts show columns like "항목 | 수량 | 단가 | 금액"
    → Always use the FINAL 금액 column, not 단가.
    → If 수량 > 1 and only 단가 is shown → price = 수량 × 단가.
- price = null ONLY if truly no price is visible for that item.
- CRITICAL: Read every digit carefully. Similar-looking digits:
  · 3 vs 8, 5 vs 6, 0 vs 8, 1 vs 7 — zoom in and verify.
  · If the total doesn't match your items sum, re-read the prices digit by digit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4: DISCOUNT ITEMS → NEGATIVE PRICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Discount lines are separate items with NEGATIVE price.
- ANY item containing: 할인, 절사, 감면, 환급, 쿠폰, discount → price MUST be negative.
- 예시:
  · "동물종합검진B코스 할인" → price: -88000
  · "쿠폰 할인 뼈없는소고기캔" → price: -12600
  · "절사할인" → price: -200
  · "할인/할증" line showing -100,600 → price: -100600
- originalPrice and discount fields: for regular items that received a discount.
  · originalPrice = amount before discount. null if no discount on this item.
  · discount = discount amount as positive integer. null if no discount.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 5: totalAmount = FINAL PAYMENT AMOUNT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- totalAmount = the amount the customer actually pays.
- Look for: 결제요청, 청구금액, 합계, 총액, 결제금액, 수납액.
- If multiple totals exist (소계 vs 청구금액), use the FINAL one (결제요청/청구금액).
- discountAmount = total discount amount (할인/할증 합계). null if none.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6: MANDATORY SELF-VERIFICATION (반드시 수행)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST perform ALL checks below. If any check fails, go back and re-read the receipt.
DO NOT return your JSON until all checks pass.

STEP 1 — Count visible line items:
  Scan the receipt image from top to bottom.
  Count EVERY row that has a name + price. Write this count down.
  Your items array must have the SAME count (± 1 for discount lines).
  If your array is shorter → you MISSED items. Go back and find them.
  COMMON MISTAKE: Similar-looking items (혈액 검사 types, 고가약물-XXX) are SEPARATE rows.

STEP 2 — Calculate your sum vs totalAmount (THIS IS THE MOST IMPORTANT CHECK):
  Add up ALL your item prices: item1 + item2 + item3 + ...
  Compare this sum to the totalAmount on the receipt.
  If the difference is more than 100원 → you have ERRORS.
  Possible causes:
    - You MISSED an item (most common!) → re-scan the receipt
    - You read a PRICE wrong → re-check each price
    - You read the TOTAL wrong → re-check the 청구금액/합계
  DO NOT RETURN until |your_sum - totalAmount| < 100.

STEP 3 — Name accuracy:
  For each item, verify the itemName matches the receipt text CHARACTER BY CHARACTER.
  If you wrote a generic name → fix it to match the receipt exactly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 7: DO NOT INCLUDE THESE AS ITEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 합계/소계/총액 summary lines
- 부가세/VAT lines
- 주소, 전화번호, 사업자등록번호
- 날짜, 시간
- 병원명, 수의사명, 환자명/보호자명
- 카드 결제 정보, 승인번호
"""


def _gemini_parse_receipt_full(
    *,
    image_bytes: bytes,
    api_key: str,
    model: str,
    timeout_seconds: int = 15,
    ocr_text: str = "",
) -> Optional[Dict[str, Any]]:
    api_key = (api_key or "").strip()
    model = (model or "gemini-3-flash-preview").strip()
    if not api_key or not model:
        return None

    b64 = base64.b64encode(image_bytes).decode("ascii")

    mime = "image/png"
    if image_bytes[:4] == b"RIFF":
        mime = "image/webp"
    elif image_bytes[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"

    parts = [
        {"text": _GEMINI_RECEIPT_PROMPT},
        {"inline_data": {"mime_type": mime, "data": b64}},
    ]

    # ✅ Google Vision OCR 텍스트: 가격(숫자) 검증용으로만 사용
    if (ocr_text or "").strip():
        parts.append({"text": (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SUPPLEMENTARY: OCR text (for PRICE/NUMBER verification ONLY)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ WARNING: This OCR text often has GARBLED item names (e.g. '고기인물' instead of '고가약물').\n"
            "DO NOT copy item names from this text. ALWAYS read item names from the IMAGE.\n"
            "\n"
            "USE THIS TEXT ONLY FOR:\n"
            "1. Counting how many line items exist (to make sure you don't miss any)\n"
            "2. Verifying NUMBERS/PRICES (the digits are usually accurate)\n"
            "\n"
            "DO NOT USE THIS TEXT FOR:\n"
            "- Item names (read names from the IMAGE only)\n"
            "- Replacing or modifying names you read from the image\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + (ocr_text or "").strip()[:6000]
        )})

    try:
        out = _call_gemini_generate_content(
            api_key=api_key,
            model=model,
            parts=parts,
            timeout_seconds=timeout_seconds,
            # media_resolution 제거 — API 호환성 문제로 Gemini 호출 실패 원인
        )
        import logging
        _glog = logging.getLogger("ocr_policy")
        _glog.info(f"[Gemini-call] raw_out_len={len(out) if out else 0}, preview={repr((out or '')[:300])}")
        j = _extract_json_from_model_text(out)
        _glog.info(f"[Gemini-call] parsed_json={'ok' if isinstance(j, dict) else 'FAIL'}, type={type(j).__name__}")
        if isinstance(j, dict):
            return j
    except Exception as e:
        import logging
        logging.getLogger("ocr_policy").error(f"[Gemini-call] EXCEPTION: {type(e).__name__}: {e}")
    return None


# =========================================================
# ✅ Claude API를 이용한 영수증 OCR (Gemini 대체)
# =========================================================
def _claude_parse_receipt(
    *,
    image_bytes: bytes,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    timeout_seconds: int = 60,
    ocr_text: str = "",
) -> Optional[Dict[str, Any]]:
    """Claude Vision API로 영수증 이미지를 분석하여 구조화된 JSON 반환."""
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    import logging
    _clog = logging.getLogger("ocr_policy.claude")

    b64 = base64.b64encode(image_bytes).decode("ascii")

    mime = "image/png"
    if image_bytes[:4] == b"RIFF":
        mime = "image/webp"
    elif image_bytes[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"

    # Claude용 프롬프트 (Gemini와 동일한 구조)
    user_content: List[Dict[str, Any]] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        },
        {
            "type": "text",
            "text": _GEMINI_RECEIPT_PROMPT,  # 같은 프롬프트 재사용
        },
    ]

    # ✅ OCR 힌트 전달 (항목 수 + 총액만 — 깨진 이름은 보내지 않음)
    if (ocr_text or "").strip():
        user_content.append({
            "type": "text",
            "text": ocr_text.strip(),
        })

    import urllib.request

    payload = {
        "model": model,
        "max_tokens": 8192,
        "system": (
            "You are a precision OCR data extractor for Korean veterinary hospital receipts. "
            "You read receipt images with extreme accuracy. "
            "You MUST extract EVERY line item visible on the receipt — never skip items. "
            "Your item prices MUST add up to the totalAmount on the receipt. "
            "If your sum doesn't match, re-read the receipt until it does. "
            "Return ONLY valid JSON, no explanation."
        ),
        "messages": [
            {"role": "user", "content": user_content}
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        j = json.loads(data)

        # Claude 응답에서 텍스트 추출
        content_blocks = j.get("content", [])
        txt_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                txt_parts.append(block.get("text", ""))

        raw_text = "\n".join(txt_parts).strip()
        _clog.info(f"[Claude-call] raw_len={len(raw_text)}, preview={repr(raw_text[:300])}")

        # JSON 추출
        result = _extract_json_from_model_text(raw_text)
        _clog.info(f"[Claude-call] parsed={'ok' if isinstance(result, dict) else 'FAIL'}")

        if isinstance(result, dict):
            return result

    except Exception as e:
        _clog.error(f"[Claude-call] EXCEPTION: {type(e).__name__}: {e}")

    return None


# =========================================================
# ✅ Valid standard tag codes (ReceiptTags.swift 완전 동기화)
# =========================================================
_VALID_TAG_CODES: set = {
    # 검사 — 영상
    "exam_xray", "exam_ct", "exam_mri", "exam_endoscope", "exam_biopsy",
    # 검사 — 초음파 (세분화)
    "exam_echo", "exam_us_abdomen", "exam_us_general",
    # 검사 — 혈액 (세분화)
    "exam_blood_cbc", "exam_blood_chem", "exam_blood_general",
    "exam_blood_type", "exam_coagulation", "exam_electrolyte", "exam_crp",
    # 검사 — 심장 (세분화)
    "exam_ecg", "exam_heart_general",
    # 검사 — 활력징후
    "exam_vitals",
    # 검사 — 호르몬
    "exam_hormone",
    # 검사 — 기타
    "exam_lab_panel", "exam_urine", "exam_fecal", "exam_fecal_pcr",
    "exam_sdma", "exam_probnp", "exam_fructosamine", "exam_glucose_curve",
    "exam_blood_gas", "exam_allergy", "exam_eye", "exam_skin", "exam_general",
    "exam_ear", "exam_microscope",
    # 예방접종
    "vaccine_rabies", "vaccine_comprehensive", "vaccine_corona",
    "vaccine_kennel", "vaccine_fip", "vaccine_parainfluenza", "vaccine_lepto",
    # 예방약/구충
    "prevent_heartworm", "prevent_external", "prevent_deworming",
    # 처방약 (medicine_ prefix — iOS ReceiptTag 기준)
    "medicine_antibiotic", "medicine_anti_inflammatory", "medicine_allergy",
    "medicine_gi", "medicine_ear", "medicine_skin", "medicine_eye",
    "medicine_painkiller", "medicine_steroid", "medicine_oral",
    # 처치/진료/입원
    "care_injection", "care_fluid", "care_transfusion", "care_oxygen",
    "care_emergency", "care_catheter", "care_procedure_fee", "care_dressing",
    "care_anal_gland", "care_ear_flush",
    "hospitalization",
    # 수술 (세분화)
    "surgery_general", "surgery_spay_neuter", "surgery_tumor",
    "surgery_foreign_body", "surgery_cesarean", "surgery_hernia", "surgery_eye",
    # 치과
    "dental_scaling", "dental_extraction", "dental_treatment",
    # 관절
    "ortho_patella", "ortho_arthritis",
    # 재활
    "rehab_therapy",
    # 기타
    "microchip", "euthanasia", "funeral",
    "care_e_collar", "care_prescription_diet",
    "supply_food", "supply_supplement", "supply_goods",
    "checkup_general", "grooming_basic", "etc_fee", "etc_discount", "etc_other",
}

# ✅ 레거시 태그 → 새 태그 자동 변환 (Gemini가 옛 코드 리턴 시)
_TAG_MIGRATION: Dict[str, str] = {
    # 옛 통합 태그 → 새 일반 태그
    "exam_blood": "exam_blood_general",
    "exam_ultrasound": "exam_us_general",
    "exam_heart": "exam_heart_general",
    # drug_ → medicine_ (Gemini 프롬프트에 drug_ 안 쓰지만 혹시)
    "drug_antibiotic": "medicine_antibiotic",
    "drug_pain_antiinflammatory": "medicine_anti_inflammatory",
    "drug_steroid": "medicine_steroid",
    "drug_gi": "medicine_gi",
    "drug_allergy": "medicine_allergy",
    "drug_eye": "medicine_eye",
    "drug_ear": "medicine_ear",
    "drug_skin": "medicine_skin",
    "drug_general": "medicine_oral",
    # 기타 레거시
    "grooming": "grooming_basic",
    "surgery": "surgery_general",
    "dental": "dental_scaling",
    "checkup": "checkup_general",
    "medicine": "medicine_oral",
    "emergency": "care_emergency",
}


def _migrate_tag(code: str) -> Optional[str]:
    """태그 코드를 정규화: valid면 그대로, 레거시면 변환, 아니면 None."""
    if not code:
        return None
    c = code.strip().lower()
    if c in _VALID_TAG_CODES:
        return c
    migrated = _TAG_MIGRATION.get(c)
    if migrated and migrated in _VALID_TAG_CODES:
        return migrated
    return None


# =========================================================
# ✅ AI ↔ Vision OCR 교차검증 (로깅 전용 — 교정 안 함)
# =========================================================
def _cross_validate_prices(
    ai_items: List[Dict[str, Any]],
    ocr_text: str,
) -> List[Dict[str, Any]]:
    """
    AI가 추출한 항목의 가격이 Vision OCR 텍스트에 존재하는지 로깅.
    OCR 이름이 깨질 수 있으므로 가격 교정은 하지 않음 (AI를 신뢰).
    로그로 불일치 항목만 기록.
    """
    if not ocr_text or not ai_items:
        return ai_items

    import logging
    _xlog = logging.getLogger("ocr_policy.xval")

    # OCR 텍스트에 등장하는 모든 숫자 집합
    all_ocr_nums: set = set()
    for raw_ln in ocr_text.splitlines():
        for m in re.findall(r"[\d,]+\d", raw_ln):
            try:
                n = int(m.replace(",", ""))
                if 100 <= n <= 50_000_000:
                    all_ocr_nums.add(n)
            except ValueError:
                pass

    mismatch_count = 0
    for item in ai_items:
        pr = item.get("price")
        if pr is None:
            continue
        if abs(pr) not in all_ocr_nums:
            nm = (item.get("itemName") or "")[:40]
            _xlog.warning(f"[XVAL] price not in OCR: '{nm}' = {pr}")
            mismatch_count += 1

    if mismatch_count:
        _xlog.warning(f"[XVAL] {mismatch_count}/{len(ai_items)} prices not found in OCR text")

    return ai_items  # 교정 없이 그대로 반환


# =========================================================
# ✅ 누락 항목 복구: Vision OCR → Gemini 보충
# =========================================================
def _recover_missing_items(
    gemini_items: List[Dict[str, Any]],
    ocr_text: str,
    total_amount: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Vision OCR 텍스트에서 추출한 항목 중 Gemini 결과에 없는 것을 보충.
    특히 비슷한 이름의 항목(혈액 검사 등)이 누락될 때 유효.
    """
    if not ocr_text or not gemini_items:
        return gemini_items

    import logging
    _rlog = logging.getLogger("ocr_policy.recover")

    # Vision OCR 텍스트에서 항목 추출
    ocr_items = _extract_items_from_text(ocr_text)
    if not ocr_items:
        return gemini_items

    # Gemini 합계 vs totalAmount 비교 → 누락 여부 판단
    gemini_sum = sum(it.get("price") or 0 for it in gemini_items)
    if total_amount and abs(gemini_sum - total_amount) < 500:
        # 합계 거의 맞음 → 누락 없음
        _rlog.info(f"[RECOVER] sum matches: gemini={gemini_sum}, total={total_amount}, skip")
        return gemini_items

    _rlog.warning(
        f"[RECOVER] sum mismatch: gemini={gemini_sum}, total={total_amount}, "
        f"diff={abs(gemini_sum - (total_amount or 0))}, "
        f"gemini_count={len(gemini_items)}, ocr_count={len(ocr_items)}"
    )

    def _norm_name(s: str) -> str:
        return re.sub(r"[^가-힣a-zA-Z0-9]", "", s.lower())

    # Gemini 항목의 (정규화된이름, 가격) 집합
    gemini_keys: set = set()
    gemini_names_norm: set = set()
    for it in gemini_items:
        nm = _norm_name(it.get("itemName") or "")
        pr = it.get("price") or 0
        gemini_keys.add((nm, pr))
        gemini_names_norm.add(nm)

    # OCR 항목 중 Gemini에 없는 것 찾기
    recovered: List[Dict[str, Any]] = []
    for ocr_it in ocr_items:
        ocr_nm = _norm_name(ocr_it.get("itemName") or "")
        ocr_pr = ocr_it.get("price") or 0

        if not ocr_nm or ocr_pr == 0:
            continue

        # 정확히 같은 (이름, 가격) 이미 있으면 스킵
        if (ocr_nm, ocr_pr) in gemini_keys:
            continue

        # 이름이 같고 가격만 다른 경우 → 가격 차이가 크면 별도 항목일 수 있음
        # 이름이 아예 없는 경우 → 누락된 항목
        name_exists = False
        for gn in gemini_names_norm:
            # 서로 포함 관계인지 체크 (부분 매칭)
            if ocr_nm and gn and (ocr_nm in gn or gn in ocr_nm):
                # 이름은 비슷하지만 가격이 상당히 다르면 별도 항목
                matching_prices = [
                    it.get("price") or 0 for it in gemini_items
                    if _norm_name(it.get("itemName") or "") == gn
                    or ocr_nm in _norm_name(it.get("itemName") or "")
                    or _norm_name(it.get("itemName") or "") in ocr_nm
                ]
                if any(abs(mp - ocr_pr) < max(500, abs(ocr_pr) * 0.1) for mp in matching_prices):
                    name_exists = True
                    break
                # 가격이 많이 다르면 → 별도 항목으로 간주 (추가)

        if not name_exists:
            _rlog.warning(
                f"[RECOVER] adding missing item: '{ocr_it.get('itemName')}' = {ocr_pr}"
            )
            recovered.append({
                "itemName": ocr_it["itemName"],
                "price": ocr_pr,
                "categoryTag": None,
                "_recovered_from_ocr": True,
            })

    if recovered:
        result = list(gemini_items) + recovered
        new_sum = sum(it.get("price") or 0 for it in result)
        _rlog.warning(
            f"[RECOVER] added {len(recovered)} items, "
            f"new_sum={new_sum} (was {gemini_sum}, total={total_amount})"
        )
        # 새 합계가 totalAmount에서 너무 벗어나면 잘못된 복구 → 원래대로
        if total_amount and abs(new_sum - total_amount) > abs(gemini_sum - total_amount):
            _rlog.warning("[RECOVER] new sum is WORSE, reverting")
            return gemini_items
        return result

    return gemini_items


def _normalize_gemini_full_result(
    j: Dict[str, Any],
    skip_noise_filter: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize AI result into (parsed, tags).
    skip_noise_filter=True: AI(Claude)가 반환한 항목은 노이즈 필터 건너뜀.
    """
    parsed: Dict[str, Any] = {
        "hospitalName": None,
        "visitDate": None,
        "totalAmount": None,
        "items": [],
        "ocrText": "",
    }

    hn = j.get("hospitalName")
    if isinstance(hn, str) and hn.strip():
        parsed["hospitalName"] = hn.strip()[:80]

    vd = j.get("visitDate")
    if isinstance(vd, str) and vd.strip():
        d = _parse_date_from_text(vd.strip())
        parsed["visitDate"] = d.isoformat() if d else vd.strip()[:20]

    ta = _coerce_int_amount(j.get("totalAmount"))
    parsed["totalAmount"] = ta if ta and ta > 0 else None

    items = j.get("items")
    tags_set: set = set()
    cleaned: List[Dict[str, Any]] = []

    if isinstance(items, list):
        for it in items[:120]:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("itemName") or "").strip()
            if not nm:
                continue
            nm = _canonicalize_item_name(_clean_item_name(nm))
            if not nm:
                continue
            # ✅ AI(Claude) 결과는 노이즈 필터 건너뜀 — AI가 이미 진짜 항목만 반환
            if not skip_noise_filter and _is_noise_line(nm):
                continue
            pr = _coerce_int_amount(it.get("price"))
            orig_pr = _coerce_int_amount(it.get("originalPrice"))
            disc = _coerce_int_amount(it.get("discount"))
            # 음수 금액은 할인 항목으로 허용

            ct = (it.get("categoryTag") or "").strip() or None
            sn = (it.get("standardName") or "").strip() or None

            # ✅ 태그 마이그레이션 적용
            if ct:
                ct = _migrate_tag(ct)
            if ct:
                tags_set.add(ct)

            item_entry = {
                "itemName": nm,
                "price": pr,
                "categoryTag": ct,
                "standardName": sn,
            }
            if orig_pr is not None:
                item_entry["originalPrice"] = orig_pr
            if disc is not None and disc > 0:
                item_entry["discount"] = disc
            cleaned.append(item_entry)

    parsed["items"] = cleaned

    raw_tags = j.get("tags")
    if isinstance(raw_tags, list):
        for t in raw_tags:
            migrated = _migrate_tag(str(t).strip())
            if migrated:
                tags_set.add(migrated)

    return parsed, list(tags_set)[:10]


# =========================================================
# Core function: process_receipt
# =========================================================
def process_receipt(
    raw_bytes: bytes,
    *,
    google_credentials: str = "",
    ocr_timeout_seconds: int = 12,
    ocr_max_concurrency: int = 4,
    ocr_sema_acquire_timeout_seconds: float = 1.0,
    receipt_max_width: int = 2048,       # ✅ 1024→2048: OCR 해상도 대폭 향상
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

    img = Image.open(io.BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg

    img = _ensure_max_pixels(img, int(image_max_pixels or 0))

    # ✅ 원본 해상도 보존 (저장/표시용)
    img_display = _resize_to_width(img.copy(), int(receipt_max_width or 0))

    # ✅ OCR용 이미지: 더 높은 해상도 유지 + 전처리 적용 (Vision OCR 전용)
    ocr_max_w = max(int(receipt_max_width or 0), 2048)  # OCR엔 최소 2048px 보장
    img_ocr = _resize_to_width(img.copy(), ocr_max_w)
    img_ocr = _preprocess_for_ocr(img_ocr)

    ocr_buf = io.BytesIO()
    img_ocr.save(ocr_buf, format="PNG")
    ocr_image_bytes = ocr_buf.getvalue()

    # ✅ Claude용 이미지: 원본 컬러 그대로 (전처리 없음 — Claude Vision은 원본이 더 정확)
    claude_buf = io.BytesIO()
    img_claude = _resize_to_width(img.copy(), ocr_max_w)
    img_claude.save(claude_buf, format="PNG")
    claude_image_bytes = claude_buf.getvalue()

    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")) or "gemini-3-flash-preview"
    g_timeout = max(60, int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60") or "60")))

    # ✅ Claude API 설정
    c_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    c_model = os.getenv("CLAUDE_OCR_MODEL", "claude-sonnet-4-20250514").strip()
    c_timeout = max(60, int(os.getenv("CLAUDE_OCR_TIMEOUT", "90") or "90"))
    c_enabled = bool(c_key)  # API 키가 있으면 자동 활성화

    hints: Dict[str, Any] = {
        "ocrEngine": "none",
        "geminiUsed": False,
        "pipeline": "claude_first" if c_enabled else "gemini_first",
    }

    ai_parsed = None
    ai_tags: List[str] = []

    import logging
    _log = logging.getLogger("ocr_policy")
    _log.info(f"[AI] claude_enabled={c_enabled}, gemini_enabled={g_enabled}, claude_model={c_model}, gemini_model={g_model}")

    # ✅ Step 1: Google Vision OCR 먼저 실행 (텍스트 추출)
    ocr_text = ""
    vision_resp = None

    sema = _get_sema(int(ocr_max_concurrency or 4))
    acquired = sema.acquire(timeout=float(ocr_sema_acquire_timeout_seconds or 1.0))
    if not acquired:
        _log.warning("[Vision] semaphore busy, will try Gemini without OCR text")
        hints["visionSkipped"] = "semaphore_busy"
    else:
        try:
            try:
                ocr_text, vision_resp = _vision_ocr(
                    ocr_image_bytes,
                    google_credentials=google_credentials,
                    timeout_seconds=int(ocr_timeout_seconds or 12),
                )
                if hints["ocrEngine"] == "none":
                    hints["ocrEngine"] = "google_vision"
                _log.info(f"[Vision] ocr_text_len={len(ocr_text)}")
            except Exception as e:
                hints["visionError"] = str(e)[:200]
                ocr_text = ""
                vision_resp = None
        finally:
            try:
                sema.release()
            except Exception:
                pass

    # ✅ Step 1.5: Vision OCR에서 항목 수 + 총액 힌트 추출
    ocr_item_count = 0
    ocr_total_amount = None
    if ocr_text:
        try:
            ocr_extracted = _extract_items_from_text(ocr_text)
            ocr_item_count = len(ocr_extracted)
            ocr_total_amount = _extract_total_amount(ocr_text)
            _log.info(f"[OCR-hint] item_count={ocr_item_count}, total={ocr_total_amount}")
        except Exception:
            pass

    # ✅ Step 2: AI 영수증 분석 (Claude 우선 → Gemini 폴백)
    ai_result_json = None

    # --- 2a: Claude API 시도 (원본 컬러 이미지 + OCR 힌트) ---
    if c_enabled:
        try:
            # OCR에서 추출한 숫자 힌트 (이름은 깨지니 숫자만)
            ocr_hint = ""
            if ocr_item_count > 0 or ocr_total_amount:
                hint_parts = []
                if ocr_item_count > 0:
                    hint_parts.append(f"- Line item count from OCR: approximately {ocr_item_count} items")
                if ocr_total_amount:
                    hint_parts.append(f"- Total amount from OCR: {ocr_total_amount:,}원")
                ocr_hint = (
                    "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "VERIFICATION TARGETS (from machine OCR — numbers are accurate):\n"
                    + "\n".join(hint_parts) + "\n"
                    "Your extracted items MUST match these targets.\n"
                    "If your item count or price sum differs significantly, re-read the receipt.\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )

            _log.info(f"[Claude] calling model={c_model}, timeout={c_timeout}, ocr_hint_len={len(ocr_hint)}")
            cj = _claude_parse_receipt(
                image_bytes=claude_image_bytes,  # ✅ 원본 컬러 이미지 (전처리 없음)
                api_key=c_key,
                model=c_model,
                timeout_seconds=c_timeout,
                ocr_text=ocr_hint,  # ✅ 숫자 힌트만 전달 (깨진 이름 없음)
            )
            if isinstance(cj, dict):
                ai_result_json = cj
                hints["ocrEngine"] = f"claude:{c_model}"
                hints["claudeUsed"] = True
                _log.info(f"[Claude] SUCCESS: items={len(cj.get('items', []))}")
            else:
                _log.warning("[Claude] returned non-dict, falling back to Gemini")
        except Exception as e:
            _log.error(f"[Claude] ERROR: {type(e).__name__}: {e}")
            hints["claudeError"] = str(e)[:200]

    # --- 2b: Gemini 폴백 (Claude 실패 시) ---
    if ai_result_json is None and g_enabled and g_key.strip():
        try:
            _log.info(f"[Gemini] fallback: model={g_model}, timeout={g_timeout}")
            gj = _gemini_parse_receipt_full(
                image_bytes=ocr_image_bytes,
                api_key=g_key,
                model=g_model,
                timeout_seconds=g_timeout,
                ocr_text=ocr_text,
            )
            if isinstance(gj, dict):
                ai_result_json = gj
                hints["ocrEngine"] = f"gemini:{g_model}"
                hints["geminiUsed"] = True
                _log.info(f"[Gemini] fallback SUCCESS: items={len(gj.get('items', []))}")
        except Exception as e:
            _log.error(f"[Gemini] ERROR: {e}")
            hints["geminiError"] = str(e)[:200]

    if ai_result_json is None:
        _log.warning("[AI] both Claude and Gemini failed or disabled")

    # --- 정규화 ---
    if isinstance(ai_result_json, dict):
        # ✅ Claude 결과는 노이즈 필터 건너뜀 (Claude가 이미 진짜 항목만 반환)
        is_claude = bool(hints.get("claudeUsed"))
        ai_parsed, ai_tags = _normalize_gemini_full_result(
            ai_result_json, skip_noise_filter=is_claude,
        )
        _log.info(f"[AI] normalized: items={len(ai_parsed.get('items', []))}, tags={ai_tags}, skip_noise={is_claude}")

        # Claude 사용 시: 이미지 직접 읽기 → 후처리 불필요 (Claude를 신뢰)
        # Gemini 폴백 시: OCR 교차검증 + 누락 복구 적용
        if not hints.get("claudeUsed") and ocr_text and ai_parsed.get("items"):
            # Gemini 폴백인 경우만 OCR 기반 로깅
            _cross_validate_prices(ai_parsed["items"], ocr_text)

            total_amt = ai_parsed.get("totalAmount")
            before_recover = len(ai_parsed["items"])
            ai_parsed["items"] = _recover_missing_items(
                ai_parsed["items"], ocr_text, total_amt
            )
            recovered_count = len(ai_parsed["items"]) - before_recover
            if recovered_count > 0:
                _log.warning(f"[RECOVER] {recovered_count} items recovered from Vision OCR")
                hints["recovered_items"] = recovered_count

    # ✅ 저장/표시용은 display 이미지 사용 (용량 절약)
    original_webp = _to_webp_bytes(img_display, quality=int(receipt_webp_quality or 85))
    redacted = _redact_image_with_vision_tokens(img_display.copy(), vision_resp)
    redacted_webp = _to_webp_bytes(redacted, quality=int(receipt_webp_quality or 85))
    webp_bytes = redacted_webp

    if ai_parsed and ai_parsed.get("items"):
        parsed = ai_parsed
        parsed["ocrText"] = (ocr_text or "")[:8000]
        hints["pipeline"] = hints.get("pipeline", "ai_primary")
    else:
        parsed, regex_hints = _parse_receipt_from_text(ocr_text or "")
        hints.update(regex_hints)
        hints["pipeline"] = "vision_regex_fallback"

        if ai_parsed:
            if not parsed.get("hospitalName") and ai_parsed.get("hospitalName"):
                parsed["hospitalName"] = ai_parsed["hospitalName"]
            if not parsed.get("visitDate") and ai_parsed.get("visitDate"):
                parsed["visitDate"] = ai_parsed["visitDate"]
            if not parsed.get("totalAmount") and ai_parsed.get("totalAmount"):
                parsed["totalAmount"] = ai_parsed["totalAmount"]

    if not isinstance(parsed.get("items"), list):
        parsed["items"] = []
    parsed["items"] = parsed["items"][:120]
    if parsed.get("totalAmount") is not None:
        try:
            parsed["totalAmount"] = int(parsed["totalAmount"])
        except Exception:
            parsed["totalAmount"] = None
    parsed["ocrText"] = (ocr_text or "")[:8000]

    hints["tags"] = ai_tags if ai_tags else []

    return webp_bytes, original_webp, parsed, hints


# =========================================================
# Bridge function for main.py compatibility
# =========================================================
def process_receipt_image(
    raw_bytes: bytes,
    *,
    timeout: int = 12,
    max_concurrency: int = 4,
    sema_timeout: float = 1.0,
    max_pixels: int = 20_000_000,
    receipt_max_width: int = 2048,       # ✅ 1024→2048
    receipt_webp_quality: int = 85,
    gemini_enabled: bool = True,
    gemini_api_key: str = "",
    gemini_model_name: str = "gemini-3-flash-preview",
    gemini_timeout: int = 20,            # ✅ 10→20: Gemini 3 thinking 시간 확보
    **kwargs,
) -> dict:
    if not raw_bytes:
        raise OCRImageError("empty raw bytes")

    try:
        webp_bytes, original_webp, parsed, hints = process_receipt(
            raw_bytes,
            google_credentials="",
            ocr_timeout_seconds=timeout,
            ocr_max_concurrency=max_concurrency,
            ocr_sema_acquire_timeout_seconds=sema_timeout,
            receipt_max_width=receipt_max_width,
            receipt_webp_quality=receipt_webp_quality,
            image_max_pixels=max_pixels,
            gemini_enabled=gemini_enabled,
            gemini_api_key=gemini_api_key,
            gemini_model_name=gemini_model_name,
            gemini_timeout_seconds=gemini_timeout,
        )
    except OCRConcurrencyError:
        raise
    except OCRTimeoutError:
        raise
    except OCRImageError:
        raise
    except RuntimeError as e:
        msg = str(e).lower()
        if "semaphore" in msg or "busy" in msg:
            raise OCRConcurrencyError(str(e)) from e
        if "timeout" in msg:
            raise OCRTimeoutError(str(e)) from e
        raise
    except ValueError as e:
        raise OCRImageError(str(e)) from e

    items_for_main = []
    for it in (parsed.get("items") or []):
        entry = {
            "name": it.get("itemName") or "",
            "price": it.get("price"),
            "categoryTag": it.get("categoryTag"),
            "standardName": it.get("standardName"),
        }
        items_for_main.append(entry)

    meta = {
        "hospital_name": parsed.get("hospitalName"),
        "visit_date": parsed.get("visitDate"),
        "total_amount": parsed.get("totalAmount"),
        "address_hint": (hints or {}).get("addressHint"),
        "ocr_engine": (hints or {}).get("ocrEngine"),
        "gemini_used": (hints or {}).get("geminiUsed", False),
        "pipeline": (hints or {}).get("pipeline", "unknown"),
        "tags": (hints or {}).get("tags", []),
    }

    return {
        "ocr_text": parsed.get("ocrText") or "",
        "items": items_for_main,
        "meta": meta,
        "webp_bytes": webp_bytes,
        "original_webp_bytes": original_webp,
        "content_type": "image/webp",
    }

