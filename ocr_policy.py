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
RULE 6: SELF-VERIFICATION (필수 검증)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before returning your JSON, perform these checks:

CHECK 1 — Item count:
  Count the number of line items visible on the receipt.
  Your items array must have the SAME count (± discount lines).
  If you have fewer items than visible lines → you MISSED items. Re-read the receipt.

CHECK 2 — Price sum:
  Calculate: sum of all your item prices (including negative discount prices).
  Compare to totalAmount.
  If |sum - totalAmount| > 1000 → you have WRONG prices or MISSING items.
  Go back and fix before returning.

CHECK 3 — Name accuracy:
  For each item, verify the itemName matches the receipt text exactly.
  If you wrote a generic name like "진료비" but the receipt says "진찰료-초진" → fix it.

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

    # ✅ Google Vision OCR 텍스트가 있으면 Gemini에 참고자료로 전달
    if (ocr_text or "").strip():
        parts.append({"text": (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "MANDATORY REFERENCE: Google Vision OCR text\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CRITICAL: You MUST cross-check EVERY price and item name against this OCR text.\n"
            "If your extracted price differs from what appears in this text, USE THE OCR TEXT NUMBER.\n"
            "The OCR text below is machine-read and highly accurate for NUMBERS.\n"
            "Steps:\n"
            "1. For each item you extract, find the matching line in this OCR text.\n"
            "2. Compare your price with the number in the OCR text.\n"
            "3. If they differ, trust the OCR text number.\n"
            "4. Check if you missed any items that appear in the OCR text but not in your extraction.\n"
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
# ✅ Gemini ↔ Vision OCR 교차검증 (가격 교정)
# =========================================================
def _cross_validate_prices(
    gemini_items: List[Dict[str, Any]],
    ocr_text: str,
) -> List[Dict[str, Any]]:
    """
    Gemini가 추출한 항목의 가격을 Vision OCR 텍스트와 교차검증.
    OCR 텍스트에서 항목명 근처 라인의 숫자를 찾아 Gemini 가격과 비교.
    불일치 시 OCR 숫자로 교정.
    """
    if not ocr_text or not gemini_items:
        return gemini_items

    import logging
    _xlog = logging.getLogger("ocr_policy.xval")

    # OCR 텍스트를 라인별로 파싱: (라인텍스트, [숫자들])
    ocr_lines: List[Tuple[str, List[int]]] = []
    for raw_ln in ocr_text.splitlines():
        ln = re.sub(r"\s+", " ", raw_ln).strip()
        if not ln:
            continue
        nums = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+\d", ln)]
        nums = [n for n in nums if 100 <= abs(n) <= 50_000_000]
        ocr_lines.append((ln, nums))

    # OCR 텍스트에 등장하는 모든 가격 집합
    all_ocr_prices: set = set()
    for _, nums in ocr_lines:
        for n in nums:
            all_ocr_prices.add(n)
            all_ocr_prices.add(-n)  # 할인 항목 대비

    def _normalize_for_match(s: str) -> str:
        s = re.sub(r"[^가-힣a-zA-Z0-9]", "", s.lower())
        return s

    corrected = []
    for item in gemini_items:
        nm = (item.get("itemName") or "").strip()
        pr = item.get("price")
        if not nm or pr is None:
            corrected.append(item)
            continue

        # 가격이 OCR 텍스트에 존재하면 → OK
        if pr in all_ocr_prices:
            corrected.append(item)
            continue

        # 가격 불일치 → OCR 라인에서 항목명 매칭 후 올바른 가격 찾기
        nm_norm = _normalize_for_match(nm)
        # 항목명의 핵심 키워드 추출 (2글자 이상)
        keywords = [w for w in re.findall(r"[가-힣a-zA-Z]{2,}", nm) if len(w) >= 2]

        best_line_idx = -1
        best_match_score = 0

        for i, (ln_text, ln_nums) in enumerate(ocr_lines):
            if not ln_nums:
                continue
            ln_norm = _normalize_for_match(ln_text)
            # 전체 이름 포함 체크
            if nm_norm and nm_norm in ln_norm:
                score = len(nm_norm) * 3
            else:
                # 키워드 매칭 점수
                score = sum(3 for kw in keywords if kw.lower() in ln_text.lower())
            if score > best_match_score:
                best_match_score = score
                best_line_idx = i

        if best_line_idx >= 0 and best_match_score >= 4:
            _, candidate_nums = ocr_lines[best_line_idx]
            # 후보 숫자 중 Gemini 가격과 가장 가까운 것 선택
            if candidate_nums:
                # 금액 컬럼: 보통 마지막 숫자가 최종 금액
                # 단가×수량인 경우도 있으니, Gemini 가격과 가장 가까운 것 우선
                closest = min(candidate_nums, key=lambda n: abs(abs(n) - abs(pr)))
                # 할인 항목이면 부호 유지
                if pr < 0:
                    closest = -abs(closest)
                if closest != pr:
                    _xlog.warning(
                        f"[XVAL] price corrected: '{nm}' "
                        f"gemini={pr} → ocr={closest} "
                        f"(line: {ocr_lines[best_line_idx][0][:60]})"
                    )
                    item = dict(item)
                    item["price"] = closest
                    item["_price_corrected"] = True

        corrected.append(item)

    return corrected


def _normalize_gemini_full_result(j: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize Gemini-first result into (parsed, tags)."""
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
            if not nm or _is_noise_line(nm):
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

    # ✅ OCR용 이미지: 더 높은 해상도 유지 + 전처리 적용
    ocr_max_w = max(int(receipt_max_width or 0), 2048)  # OCR엔 최소 2048px 보장
    img_ocr = _resize_to_width(img.copy(), ocr_max_w)
    img_ocr = _preprocess_for_ocr(img_ocr)

    ocr_buf = io.BytesIO()
    img_ocr.save(ocr_buf, format="PNG")
    ocr_image_bytes = ocr_buf.getvalue()

    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")) or "gemini-3-flash-preview"
    g_timeout = max(60, int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60") or "60")))

    hints: Dict[str, Any] = {
        "ocrEngine": "none",
        "geminiUsed": False,
        "pipeline": "gemini_first",
    }

    gemini_parsed = None
    gemini_tags: List[str] = []

    import logging
    _log = logging.getLogger("ocr_policy")
    _log.info(f"[Gemini] enabled={g_enabled}, key_present={bool(g_key.strip())}, model={g_model}")

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

    # ✅ Step 2: Gemini 실행 (이미지 + Vision OCR 텍스트 동시 전달)
    if g_enabled and g_key.strip():
        try:
            gj = _gemini_parse_receipt_full(
                image_bytes=ocr_image_bytes,
                api_key=g_key,
                model=g_model,
                timeout_seconds=g_timeout,
                ocr_text=ocr_text,
            )
            _log.info(f"[Gemini] raw result type={type(gj).__name__}, keys={list(gj.keys()) if isinstance(gj, dict) else 'N/A'}")
            if isinstance(gj, dict):
                gemini_parsed, gemini_tags = _normalize_gemini_full_result(gj)
                hints["geminiUsed"] = True
                hints["ocrEngine"] = f"gemini:{g_model}"
                _log.info(f"[Gemini] items={len(gemini_parsed.get('items', []))}, tags={gemini_tags}")

                # ✅ Step 3: Gemini ↔ Vision OCR 가격 교차검증
                if ocr_text and gemini_parsed.get("items"):
                    before_count = len(gemini_parsed["items"])
                    gemini_parsed["items"] = _cross_validate_prices(
                        gemini_parsed["items"], ocr_text
                    )
                    corrected_count = sum(
                        1 for it in gemini_parsed["items"] if it.get("_price_corrected")
                    )
                    if corrected_count > 0:
                        _log.warning(f"[XVAL] {corrected_count}/{before_count} prices corrected by Vision OCR")
                        hints["xval_corrected"] = corrected_count
        except Exception as e:
            _log.error(f"[Gemini] ERROR: {e}")
            hints["geminiError"] = str(e)[:200]
    else:
        _log.warning(f"[Gemini] SKIPPED: enabled={g_enabled}, key_present={bool(g_key.strip())}")

    # ✅ 저장/표시용은 display 이미지 사용 (용량 절약)
    original_webp = _to_webp_bytes(img_display, quality=int(receipt_webp_quality or 85))
    redacted = _redact_image_with_vision_tokens(img_display.copy(), vision_resp)
    redacted_webp = _to_webp_bytes(redacted, quality=int(receipt_webp_quality or 85))
    webp_bytes = redacted_webp

    if gemini_parsed and gemini_parsed.get("items"):
        parsed = gemini_parsed
        parsed["ocrText"] = (ocr_text or "")[:8000]
        hints["pipeline"] = "gemini_primary"
    else:
        parsed, regex_hints = _parse_receipt_from_text(ocr_text or "")
        hints.update(regex_hints)
        hints["pipeline"] = "vision_regex_fallback"

        if gemini_parsed:
            if not parsed.get("hospitalName") and gemini_parsed.get("hospitalName"):
                parsed["hospitalName"] = gemini_parsed["hospitalName"]
            if not parsed.get("visitDate") and gemini_parsed.get("visitDate"):
                parsed["visitDate"] = gemini_parsed["visitDate"]
            if not parsed.get("totalAmount") and gemini_parsed.get("totalAmount"):
                parsed["totalAmount"] = gemini_parsed["totalAmount"]

    if not isinstance(parsed.get("items"), list):
        parsed["items"] = []
    parsed["items"] = parsed["items"][:120]
    if parsed.get("totalAmount") is not None:
        try:
            parsed["totalAmount"] = int(parsed["totalAmount"])
        except Exception:
            parsed["totalAmount"] = None
    parsed["ocrText"] = (ocr_text or "")[:8000]

    hints["tags"] = gemini_tags if gemini_tags else []

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

