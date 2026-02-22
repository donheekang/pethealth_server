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
# ✅ v2.7.0: 영수증 이미지 전처리 (그림자 제거 + OCR 인식률 향상)
# -----------------------------
def _preprocess_for_ocr(img):
    """
    v2.8.0: 강화된 그림자 제거 + 적응형 이진화 + OCR 최적화.
    1단계: 그림자 제거 (배경 조명 추정 → 나누기 기법) — 2-pass
    2단계: 적응형 이진화 (adaptive threshold) — 글씨/배경 분리
    3단계: 대비/선명도 강화
    """
    import logging
    _plog = logging.getLogger("ocr_policy")
    _plog.warning(f"[PREPROCESS] v2.8.0 shadow removal STARTED, img_size={img.size}")
    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps
        # 1) 그레이스케일 변환
        gray = img.convert("L")
        # 2) ✅ 그림자 제거: 2-pass divide 기법
        #    Pass 1: 큰 블러로 넓은 그림자 제거
        #    Pass 2: 작은 블러로 잔여 그림자/불균일 조명 제거
        try:
            import numpy as np
            _plog.warning(f"[PREPROCESS] numpy available, doing 2-pass divide shadow removal")
            g_arr = np.array(gray, dtype=np.float32)

            # Pass 1: 큰 블러 (넓은 그림자 제거)
            blur_r1 = max(gray.width, gray.height) // 10  # 더 넓은 블러
            blur_r1 = max(blur_r1, 40)
            bg1 = gray.filter(ImageFilter.GaussianBlur(radius=blur_r1))
            b1_arr = np.maximum(np.array(bg1, dtype=np.float32), 1.0)
            pass1 = np.clip(g_arr / b1_arr * 255.0, 0, 255).astype(np.uint8)
            _plog.warning(f"[PREPROCESS] pass1 done, blur_r={blur_r1}")

            # Pass 2: 작은 블러 (잔여 불균일 조명 제거)
            from PIL import Image as _PILImage
            pass1_img = _PILImage.fromarray(pass1, mode="L")
            blur_r2 = max(gray.width, gray.height) // 30
            blur_r2 = max(blur_r2, 15)
            bg2 = pass1_img.filter(ImageFilter.GaussianBlur(radius=blur_r2))
            p1_arr = np.array(pass1_img, dtype=np.float32)
            b2_arr = np.maximum(np.array(bg2, dtype=np.float32), 1.0)
            pass2 = np.clip(p1_arr / b2_arr * 255.0, 0, 255).astype(np.uint8)
            _plog.warning(f"[PREPROCESS] pass2 done, blur_r={blur_r2}")

            gray = _PILImage.fromarray(pass2, mode="L")
            gray = ImageOps.autocontrast(gray, cutoff=2)

            # 3) 적응형 이진화 (adaptive threshold) — 글씨/배경 완전 분리
            #    로컬 블록별로 임계값 계산 → 그림자 영역도 글씨 살림
            try:
                block_size = max(gray.width, gray.height) // 20
                block_size = max(block_size, 15)
                if block_size % 2 == 0:
                    block_size += 1  # 홀수로
                local_blur = gray.filter(ImageFilter.GaussianBlur(radius=block_size))
                g_final = np.array(gray, dtype=np.float32)
                l_arr = np.array(local_blur, dtype=np.float32)
                # 로컬 평균보다 C만큼 어두우면 글씨(검정), 아니면 배경(흰색)
                C = 10  # 임계값 오프셋
                binary = np.where(g_final < (l_arr - C), 0, 255).astype(np.uint8)
                gray = _PILImage.fromarray(binary, mode="L")
                _plog.warning(f"[PREPROCESS] adaptive threshold done, block={block_size}, C={C}")
            except Exception as e_at:
                _plog.warning(f"[PREPROCESS] adaptive threshold skipped: {e_at}")
                # 이진화 실패 시 autocontrast만 적용된 상태로 계속

            _plog.warning(f"[PREPROCESS] shadow removal SUCCESS (numpy 2-pass)")
        except ImportError:
            _plog.warning(f"[PREPROCESS] numpy not available, trying ImageMath fallback")
            try:
                blur_radius = max(gray.width, gray.height) // 10
                blur_radius = max(blur_radius, 40)
                bg = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                from PIL import ImageMath
                gray = ImageMath.eval(
                    "convert(min((a * 255) / (b + 1), 255), 'L')",
                    a=gray, b=bg
                )
                gray = ImageOps.autocontrast(gray, cutoff=2)
                _plog.warning(f"[PREPROCESS] shadow removal SUCCESS (ImageMath)")
            except Exception as e2:
                _plog.warning(f"[PREPROCESS] ImageMath fallback FAILED: {e2}")
        except Exception as e1:
            _plog.warning(f"[PREPROCESS] shadow removal FAILED: {e1}")
        # 4) 대비 강화
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)  # 이진화 후이므로 1.8→1.5로 줄임
        # 5) 선명도 강화
        enhancer = ImageEnhance.Sharpness(gray)
        gray = enhancer.enhance(2.0)
        # 6) 가벼운 노이즈 제거
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
        # ✅ 카테고리 소계/헤더 라인 필터 (3가지 패턴):
        # 패턴1: "진료 (1,964,600)" — 괄호 안에 큰 숫자
        _paren_num_m = re.search(r'\(\s*([\d,]+)\s*\)', ln)
        if _paren_num_m:
            try:
                _pn = int(_paren_num_m.group(1).replace(",", ""))
                _outside = re.sub(r'\([^)]*\)', '', ln)
                _outside_nums = [int(x.replace(",", "")) for x in _AMOUNT_RE.findall(_outside)]
                _outside_nums = [n for n in _outside_nums if n >= 100]
                if _pn >= 100_000 and not _outside_nums:
                    continue
            except ValueError:
                pass
        # 패턴2: "진료 ( . )" "입원 ( . )" — OCR이 숫자를 잘못 읽어 점/공백만 남은 경우
        # 이런 이름은 카테고리 헤더임 (실제 항목명에는 괄호 안에 점만 있지 않음)
        _garbled_paren = re.search(r'\(\s*[.\s,]+\s*\)', ln)
        if _garbled_paren:
            continue
        # 패턴3: 이름이 한글 1~3글자 + 괄호 → 카테고리 헤더 (예: "진료 (...)", "용품 (...)")
        _short_header = re.match(r'^[가-힣]{1,3}\s*\(', ln.strip())
        if _short_header:
            _paren_content = re.search(r'\([^)]*\)', ln)
            if _paren_content:
                _inner = _paren_content.group(0)
                # 괄호 안에 한글이 없으면 (= 숫자/기호만) → 카테고리 헤더
                if not re.search(r'[가-힣]', _inner):
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
        '  hospitalName (string|null), visitDate (YYYY-MM-DD|null), visitTime (HH:MM 24h format|null), totalAmount (integer|null),\n'
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
    out: Dict[str, Any] = {"hospitalName": None, "visitDate": None, "visitTime": None, "totalAmount": None, "items": []}
    hn = j.get("hospitalName")
    if isinstance(hn, str) and hn.strip():
        out["hospitalName"] = hn.strip()[:80]
    vd = j.get("visitDate")
    if isinstance(vd, str) and vd.strip():
        d = _parse_date_from_text(vd.strip())
        out["visitDate"] = d.isoformat() if d else vd.strip()[:20]
    # ✅ visitTime 추출
    vt = j.get("visitTime")
    if isinstance(vt, str) and vt.strip():
        vt_clean = vt.strip()[:5]  # "HH:MM"
        if re.match(r"^\d{1,2}:\d{2}$", vt_clean):
            out["visitTime"] = vt_clean
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
                continue  # ✅ 할인 라인(음수 가격) 제외
            disc = _coerce_int_amount(it.get("discount"))
            # ✅ 항목별 할인 적용
            if disc is not None and disc > 0 and pr is not None:
                net_price = pr - disc
                if net_price <= 0:
                    continue  # 100% 할인 → 무료 항목 제외
                pr = net_price
            ct_raw = (it.get("categoryTag") or "").strip() or None
            ct_val = _migrate_tag(ct_raw) if ct_raw else None
            cleaned.append({"itemName": nm, "price": pr, "categoryTag": ct_val})
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
  "visitTime": "HH:MM" (24h) or null,
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
- ⚠️ COMMON MISTAKE: Do NOT replace specific product/vaccine names with generic category names!
  · "DHPPi" → "DHPPi" (NOT "진찰료", NOT "종합백신")
  · "Nexgard Spectra" → "Nexgard Spectra" (NOT "Rabies", NOT "구충제")
  · "Corona" → "Corona" (NOT "코로나백신")
  · "노비박 Puppy DP" → "노비박 Puppy DP" (NOT "종합백신")
  · "브라벡토" → "브라벡토" (NOT "외부기생충")
  · "하트가드" → "하트가드" (NOT "심장사상충")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3: PRICE = EXACT NUMBER FROM RECEIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Copy the EXACT integer from the receipt. Do NOT round, guess, or approximate.
- 예시: receipt shows "77,000" → price: 77000 (NOT 71000, NOT 70000)
       receipt shows "9,900" → price: 9900 (NOT 5500, NOT 10000)
       receipt shows "2,600" → price: 2600 (NOT 2000)
- QUANTITY HANDLING:
  · ⚠️ DO NOT multiply quantity × unit price yourself. Just record the price number as printed.
  · If receipt shows columns like "항목 | 수량 | 단가 | 금액" with a FINAL 금액 column
    → Use the 금액 column value.
  · If receipt shows "수량: 4 × 단가: 22,000 = 88,000" → price: 88000 (use the total shown)
  · If receipt shows "내복약 1일 × 7 = 31,500" → price: 31500 (use the total shown)
  · If receipt shows ONLY unit price with quantity but NO total → record the unit price AS IS.
    → Example: "주사1회당 | 2 | 16,940" → price: 16940 (NOT 33880, do NOT multiply)
  · NEVER do arithmetic yourself. Only use numbers that are PRINTED on the receipt.
- price = null ONLY if truly no price is visible for that item.
- ⭐ #1 MOST COMMON ERROR: Confusing digit 6 and 8!
  · On thermal receipts, "6" and "8" look VERY similar. This is the biggest source of errors.
  · "6" has an OPEN top-right (like a hook). "8" is CLOSED on both top and bottom (two loops).
  · ALWAYS double-check every 6 and 8 in prices: Is it 66,000 or 68,000? Is it 16,000 or 18,000?
  · When uncertain between 6 and 8, CHECK THE OCR TEXT — it usually has the correct digit.
  · Example errors to AVOID: 66,000→68,000, 16,940→18,940, 36,000→38,000
- Other similar-looking digits to verify:
  · 3 vs 8, 5 vs 6, 0 vs 8, 1 vs 7 — zoom in and verify.
  · If the total doesn't match your items sum, the cause is usually a 6↔8 confusion.
- ⚠️ ABSOLUTELY NEVER invent or adjust a price to make your sum equal the totalAmount.
  Every price must come directly from the receipt image. Accuracy per item > matching total.
- ⚠️ OCR TEXT CROSS-CHECK (when SUPPLEMENTARY OCR text is provided):
  · The OCR text's NUMBERS/DIGITS are highly accurate even when item names are garbled.
  · For EVERY price you read from the image, cross-check it against the OCR text numbers.
  · If you read "68,000" from image but OCR text shows "66,000" for the same line → USE 66,000.
  · Trust OCR text digits MORE than your image reading for prices.
  · Example: Image looks like 68,000 but OCR text says 66,000 → the correct price is 66,000.
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
STEP 2 — Calculate your sum vs totalAmount:
  Add up ALL your item prices: item1 + item2 + item3 + ...
  Compare this sum to the totalAmount on the receipt.
  If the difference is more than 100원 → you may have missed an item or misread a price.
  Try to re-read the receipt to find the cause.
  ⚠️ CRITICAL WARNING: NEVER change an item's price just to make the sum match the total!
  Each price MUST be exactly what is printed on the receipt.
  If your sum doesn't match after careful re-reading, return the EXACT prices anyway.
  An accurate price with a mismatched total is BETTER than a fake price that matches.
  WRONG: Reading 96,000 on receipt but writing 56,000 to make sum work.
  RIGHT: Always write 96,000 if that's what the receipt shows, even if total doesn't match.
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 8: ⚠️ CATEGORY HEADERS vs ITEMS — DO NOT CONFUSE (매우 중요!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Korean vet receipts often have CATEGORY HEADERS with subtotals:
  진료 (350,000)             ← THIS IS A CATEGORY HEADER, NOT AN ITEM!
    진찰료-재진    15,000    ← This is the actual item
    *검사-혈액     50,000    ← This is the actual item
    처치-수액세트  85,000    ← This is the actual item
  용품 (25,000)              ← CATEGORY HEADER
    처방사료 캔    25,000    ← Actual item
  입원 (120,000)             ← CATEGORY HEADER
    내복약-1일     45,000    ← Actual item
    입원비         75,000    ← Actual item

- The parenthesized number after a category name (e.g. "진료 (350,000)") is a SUBTOTAL.
- NEVER use category subtotals as item prices! They are the SUM of items below them.
- ONLY extract the INDENTED line items underneath each category header.
- If you see a huge price for a simple consultation item → WRONG! That's likely a subtotal.
  The actual consultation fee (진찰료/진료비) is usually 10,000~30,000원.
- Category header keywords to watch: 진료, 검사, 처치, 수술, 약, 용품, 입원, 미용, 기타

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 9: QUANTITY COLUMN — DO NOT CONFUSE WITH PRICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Receipts with table format often have: 항목명 | 수량 | 단가 | 금액
- 수량(quantity) is usually a small number: 1, 2, 3, 5, 7, 10, etc.
- NEVER read quantity as price. Example:
  · "내복약-1일 | 7 | 5,500 | 38,500" → price is 38,500 (rightmost 금액 column)
  · NOT 7, NOT 5,500
- Always use the RIGHTMOST large number as the price (the 금액/total column).
- If a row shows: "항목 | 수량 | 단가 | 금액" columns → price = 금액 column value.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 10: RECEIPT BOTTOM — READ CAREFULLY (영수증 하단 주의)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Thermal receipt paper often FADES at the bottom. Text becomes lighter and harder to read.
- Items near the bottom (입원비, 용품, 내복약, etc.) are FREQUENTLY misread or skipped.
- PAY EXTRA ATTENTION to the last 3-5 items on the receipt.
- If a bottom item's price looks unusually low (e.g. 500 instead of 55,000), re-read carefully.
- Cross-check bottom item prices against the OCR text — OCR is more reliable for faded text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 11: PRICE RANGE SANITY CHECK (가격 범위 검증)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Korean veterinary prices have typical ranges:
- 진료비/진찰료: 5,000 ~ 30,000원
- 혈액검사: 10,000 ~ 100,000원
- X-ray/초음파: 30,000 ~ 150,000원
- CT/MRI: 200,000 ~ 1,000,000원
- 수술: 50,000 ~ 2,000,000원
- 내복약: 3,000 ~ 50,000원
- 입원비: 20,000 ~ 200,000원/일
- 용품/사료: 3,000 ~ 50,000원

⚠️ RED FLAGS — if you see these prices, RE-READ the receipt:
- Any item under 1,000원 (e.g. 500원 for 입원비 → almost certainly wrong)
- Any single non-CT/MRI/surgery item over 500,000원 → likely a category subtotal, not an item
- 진료비/진찰료 over 100,000원 → likely reading category subtotal instead of item price
- If a price seems wildly out of range, check if you accidentally read a category subtotal.
"""
def _gemini_parse_receipt_full(
    *,
    image_bytes: bytes,
    api_key: str,
    model: str,
    timeout_seconds: int = 15,
    ocr_text: str = "",
    preprocessed_image_bytes: bytes = b"",
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
    ]
    # ✅ 전처리 흑백 이미지만 전달: 글자가 선명하고, AI가 "해석"하지 않고 그대로 읽음
    if preprocessed_image_bytes:
        pp_b64 = base64.b64encode(preprocessed_image_bytes).decode("ascii")
        pp_mime = "image/png"
        if preprocessed_image_bytes[:4] == b"RIFF":
            pp_mime = "image/webp"
        parts.append({"inline_data": {"mime_type": pp_mime, "data": pp_b64}})
    else:
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    # ✅ Google Vision OCR 텍스트: 가격(숫자) 검증용으로만 사용
    if (ocr_text or "").strip():
        # OCR 텍스트에서 금액 목록 추출하여 Gemini에게 힌트로 제공
        ocr_prices_hint = ""
        try:
            ocr_nums_for_hint: list = []
            for ln in (ocr_text or "").splitlines():
                for m in re.findall(r"[\d,]+\d", ln):
                    try:
                        n = int(m.replace(",", ""))
                        if 1000 <= n <= 50_000_000:
                            ocr_nums_for_hint.append(n)
                    except ValueError:
                        pass
            if ocr_nums_for_hint:
                # 중복 제거하고 정렬
                unique_prices = sorted(set(ocr_nums_for_hint))
                price_list = ", ".join(f"{p:,}" for p in unique_prices)
                ocr_prices_hint = (
                    f"\n⭐ PRICE REFERENCE LIST (from OCR — these numbers are RELIABLE):\n"
                    f"[{price_list}]\n"
                    f"Your item prices should come from THIS list. "
                    f"If you read a price that is NOT in this list, you probably misread a digit.\n"
                )
        except Exception:
            pass
        parts.append({"text": (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SUPPLEMENTARY: OCR text (for PRICE/NUMBER verification ONLY)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ WARNING: This OCR text often has GARBLED item names (e.g. '고기인물' instead of '고가약물').\n"
            "DO NOT copy item names from this text. ALWAYS read item names from the IMAGE.\n"
            "\n"
            "USE THIS TEXT ONLY FOR:\n"
            "1. Counting how many line items exist (to make sure you don't miss any)\n"
            "2. ⭐ Verifying NUMBERS/PRICES — OCR digits are MORE RELIABLE than image reading!\n"
            "   If your image reading differs from OCR text number → TRUST THE OCR TEXT NUMBER.\n"
            "   Example: You read 68,000 from image but OCR text shows 66,000 → USE 66,000.\n"
            "\n"
            "DO NOT USE THIS TEXT FOR:\n"
            "- Item names (read names from the IMAGE only)\n"
            "- Replacing or modifying names you read from the image\n"
            + ocr_prices_hint
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + (ocr_text or "").strip()[:6000]
        )})
    try:
        out = _call_gemini_generate_content(
            api_key=api_key,
            model=model,
            parts=parts,
            timeout_seconds=timeout_seconds,
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
# ✅ 키워드 기반 자동 태그 추론 (Gemini가 태그를 누락한 경우 폴백)
# =========================================================
_TAG_KEYWORD_RULES: List[tuple] = [
    # (키워드 리스트, 태그코드) — 먼저 매치되는 규칙 우선
    # ── 검사: 영상 ──
    (["x-ray", "xray", "x ray", "방사선", "엑스레이"], "exam_xray"),
    (["ct ", "ct-", "ct촬영", "컴퓨터단층"], "exam_ct"),
    (["mri", "자기공명"], "exam_mri"),
    (["내시경", "endoscop"], "exam_endoscope"),
    (["조직검사", "생검", "biopsy", "cytology", "fna"], "exam_biopsy"),
    # ── 검사: 초음파 ──
    (["심장초음파", "심초음파", "echo", "심장 초음파"], "exam_echo"),
    (["초음파(복부", "초음파-복부", "복부초음파", "복부 초음파", "us abdomen"], "exam_us_abdomen"),
    (["초음파"], "exam_us_general"),
    # ── 검사: 혈액 ──
    (["cbc", "혈구", "혈액(혈구)", "hemogram", "procyte"], "exam_blood_cbc"),
    (["chem", "chemistry", "혈청", "생화학"], "exam_blood_chem"),
    (["crp", "c-reactive", "c반응"], "exam_crp"),
    (["saa"], "exam_crp"),
    (["전해질", "electrolyte", "카탈리스트", "catalyst"], "exam_electrolyte"),
    (["혈액가스", "가스분석", "blood gas", "정맥 가스"], "exam_blood_gas"),
    (["혈액형", "blood type"], "exam_blood_type"),
    (["응고", "coagul", "pt/aptt"], "exam_coagulation"),
    (["sdma"], "exam_sdma"),
    (["probnp", "bnp"], "exam_probnp"),
    (["프럭토사민", "fructosamine"], "exam_fructosamine"),
    (["혈당곡선", "glucose curve"], "exam_glucose_curve"),
    (["혈액"], "exam_blood_general"),
    # ── 검사: 심장 ──
    (["심전도", "ecg", "ekg"], "exam_ecg"),
    # ── 검사: 소변/분변 ──
    (["소변", "뇨검사", "urinalysis", "urine"], "exam_urine"),
    (["분변", "대변", "fecal", "변검사"], "exam_fecal"),
    (["pcr"], "exam_fecal_pcr"),
    # ── 검사: 알레르기/안과/피부/귀 ──
    (["알레르기검사", "allergy test"], "exam_allergy"),
    (["안압", "안과검사", "eye exam", "슬릿램프", "안검사"], "exam_eye"),
    (["피부검사", "skin test", "피부과"], "exam_skin"),
    (["이경", "귀검사", "ear exam", "otoscop"], "exam_ear"),
    (["현미경", "microscop"], "exam_microscope"),
    # ── 검사: 호르몬 ──
    (["호르몬", "갑상선", "t4", "cortisol", "acth"], "exam_hormone"),
    # ── 예방접종 ──
    (["광견병", "rabies"], "vaccine_rabies"),
    (["종합백신", "dhppl", "dhppi", "종합접종", "5종", "7종", "혼합예방", "노비박", "유리칸", "버박", "뱅가드", "인터벳", "듀라문", "puppy dp"], "vaccine_comprehensive"),
    (["코로나", "corona", "코로나장염"], "vaccine_corona"),
    (["켄넬", "kennel", "기관지염"], "vaccine_kennel"),
    (["fip", "복막염"], "vaccine_fip"),
    # ── 예방약/구충 ──
    (["심장사상충", "heartworm", "하트가드", "heartgard"], "prevent_heartworm"),
    (["외부기생충", "프론트라인", "frontline", "브라벡토", "bravecto", "외부구충", "크레델리오", "simparica"], "prevent_external"),
    (["넥스가드스펙트라", "nexgard spectra", "넥스가드 스펙트라"], "prevent_heartworm"),
    (["넥스가드", "nexgard"], "prevent_external"),
    (["구충", "내부기생충", "deworming", "드론탈", "펜벤다졸"], "prevent_deworming"),
    # ── 처방약 ──
    (["항생제", "antibiotic", "아목시실린", "세팔렉신", "바이트릴", "엔로플록사신", "클라목실"], "medicine_antibiotic"),
    (["소염", "항염", "anti-inflammatory", "멜록시캄", "메타캄", "카프로펜", "리마딜", "온시올", "갈라프란트"], "medicine_anti_inflammatory"),
    (["진통", "painkiller", "트라마돌", "가바펜틴"], "medicine_painkiller"),
    (["스테로이드", "steroid", "프레드니솔론", "덱사메타손"], "medicine_steroid"),
    (["위장약", "위장관", "세레니아", "메토클로프라미드", "란소프라졸", "오메프라졸", "수크랄페이트", "장약"], "medicine_gi"),
    (["안약", "eye drop", "점안"], "medicine_eye"),
    (["귀약", "이어클리너", "ear drop", "이약"], "medicine_ear"),
    (["피부약", "아포퀠", "사이토포인트", "아토피카", "피부"], "medicine_skin"),
    (["알레르기약", "항히스타민", "세티리진"], "medicine_allergy"),
    (["팔라디아", "팔리디아", "palladia", "항암", "빈크리스틴", "독소루비신", "사이클로포스파마이드", "종양약"], "medicine_oral"),
    (["내복약", "경구약", "조제", "oral med"], "medicine_oral"),
    (["칼슘", "calcium"], "medicine_oral"),
    (["비타민", "vitamin", "vit k", "vit b"], "medicine_oral"),
    # ── 마취 (처치보다 먼저 — "마취-모니터링"이 "산소"보다 우선) ──
    (["마취", "anesthes", "호흡마취", "모니터링"], "surgery_general"),
    # ── 처치/진료 ──
    (["주사", "injection", "inj", "피하주사", "근육주사", "정맥주사", "앰플"], "care_injection"),
    (["수액", "fluid", "링거", "수액세트", "카테타", "catheter"], "care_fluid"),
    (["수혈", "transfusion"], "care_transfusion"),
    (["산소", "oxygen", "산소방"], "care_oxygen"),
    (["응급", "emergency"], "care_emergency"),
    (["드레싱", "dressing", "붕대", "bandage"], "care_dressing"),
    (["항문낭", "anal gland"], "care_anal_gland"),
    (["이세정", "ear flush", "귀세정"], "care_ear_flush"),
    (["진료비", "재진", "초진", "진찰", "상담"], "care_procedure_fee"),
    (["복수천자", "천자", "흉수"], "care_procedure_fee"),
    # ── 입원 ──
    (["입원", "hospitalization", "격리"], "hospitalization"),
    # ── 수술 ──
    (["중성화", "spay", "neuter", "거세", "난소", "자궁"], "surgery_spay_neuter"),
    (["종양제거", "종양절제", "tumor"], "surgery_tumor"),
    (["이물", "foreign body"], "surgery_foreign_body"),
    (["제왕절개", "cesarean"], "surgery_cesarean"),
    (["헤르니아", "hernia", "탈장"], "surgery_hernia"),
    (["안과수술", "eye surgery", "안구"], "surgery_eye"),
    (["수술"], "surgery_general"),
    # ── 치과 ──
    (["스케일링", "scaling", "치석"], "dental_scaling"),
    (["발치", "extraction", "치아발거"], "dental_extraction"),
    (["치과", "dental", "치아"], "dental_treatment"),
    # ── 관절 ──
    (["슬개골", "patella"], "ortho_patella"),
    (["관절염", "arthritis", "관절"], "ortho_arthritis"),
    # ── 재활 ──
    (["재활", "물리치료", "rehab", "수중", "레이저치료"], "rehab_therapy"),
    # ── 건강검진 ──
    (["건강검진", "종합검진", "checkup", "기본검진"], "checkup_general"),
    # ── 기타 ──
    (["마이크로칩", "microchip", "칩삽입"], "microchip"),
    (["안락사", "euthanasia"], "euthanasia"),
    (["장례", "화장", "funeral"], "funeral"),
    (["넥카라", "e-collar", "엘리자베스"], "care_e_collar"),
    (["처방식", "처방사료"], "care_prescription_diet"),
    (["사료", "캔", "파우치", "간식", "로얄캔", "로얄-", "힐스", "처방전용", "리커버리"], "supply_food"),
    (["영양제", "supplement", "유산균", "오메가"], "supply_supplement"),
    (["용품", "패드", "장난감"], "supply_goods"),
    (["그루밍", "미용", "목욕"], "grooming_basic"),
    (["할인", "감면", "discount"], "etc_discount"),
]

def _infer_tag_by_keyword(item_name: str) -> Optional[str]:
    """항목 이름에서 키워드 매칭으로 태그 추론. 매치 없으면 None."""
    if not item_name:
        return None
    low = item_name.lower()
    for keywords, tag_code in _TAG_KEYWORD_RULES:
        for kw in keywords:
            if kw.lower() in low:
                return tag_code
    return None

# =========================================================
# ✅ OCR 항목 이름 매칭 보정 (총액 불필요 — 직접 비교)
# =========================================================
def _fix_prices_by_ocr_name_match(
    ai_items: List[Dict[str, Any]],
    ocr_text: str,
) -> None:
    """
    OCR 텍스트에서 항목(이름+가격)을 추출하고,
    Gemini 항목과 이름을 매칭하여 가격이 다르면 OCR 가격으로 교체.
    총액에 의존하지 않고 개별 항목 단위로 보정.
    """
    if not ocr_text or not ai_items:
        return
    import logging
    _mlog = logging.getLogger("ocr_policy.name_match")
    ocr_items = _extract_items_from_text(ocr_text)
    if not ocr_items:
        return
    def _norm_name(s: str) -> str:
        return re.sub(r"[^가-힣a-zA-Z0-9]", "", s.lower())
    def _name_similarity(a: str, b: str) -> float:
        na, nb = _norm_name(a), _norm_name(b)
        if not na or not nb:
            return 0.0
        short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
        if short in long:
            return 0.9
        common = sum(1 for c in short if c in long)
        return common / max(len(short), 1)
    # Gemini 항목의 가격을 set으로
    ai_prices = set()
    for it in ai_items:
        if it.get("price"):
            ai_prices.add(abs(it["price"]))
    corrected = 0
    _used_ocr_indices: set = set()  # 🔒 이미 매칭된 OCR 항목 재사용 방지
    for ai_item in ai_items:
        ai_pr = ai_item.get("price")
        if ai_pr is None or ai_pr == 0:
            continue
        ai_name = ai_item.get("itemName") or ""
        abs_ai_pr = abs(ai_pr)
        # 이 Gemini 항목과 가장 잘 매칭되는 OCR 항목 찾기
        # ✅ 동일 유사도일 때 이름이 더 긴(구체적인) OCR 항목 우선
        best_ocr = None
        best_sim = 0.0
        best_name_len = 0
        best_ocr_idx = -1
        for idx, ocr_item in enumerate(ocr_items):
            if idx in _used_ocr_indices:
                continue  # 🔒 이미 다른 AI 항목에 매칭됨
            ocr_name = ocr_item.get("itemName") or ""
            ocr_pr = ocr_item.get("price")
            if ocr_pr is None:
                continue
            # 🔒 OCR 항목 이름이 너무 짧으면 신뢰 불가 ("can", "LAD" 등)
            _ocr_nm_clean = _norm_name(ocr_name)
            _has_kor = any("\uac00" <= ch <= "\ud7a3" for ch in ocr_name)
            if not _has_kor and len(_ocr_nm_clean) < 5:
                continue
            if len(_ocr_nm_clean) < 2:
                continue
            sim = _name_similarity(ai_name, ocr_name)
            ocr_nlen = len(_ocr_nm_clean)
            # 유사도가 더 높거나, 같으면 이름이 더 긴(구체적인) 쪽 우선
            if sim > best_sim or (sim == best_sim and sim > 0 and ocr_nlen > best_name_len):
                best_sim = sim
                best_ocr = ocr_item
                best_name_len = ocr_nlen
                best_ocr_idx = idx
        if best_ocr and best_sim >= 0.5:
            ocr_pr = best_ocr["price"]
            ocr_name = best_ocr.get("itemName", "")
            if ocr_pr == abs_ai_pr:
                # 가격 동일 → 매칭은 하되 교정 불필요, OCR 항목 사용 표시
                _used_ocr_indices.add(best_ocr_idx)
                continue
            # 가격이 다를 때
            # ✅ 안전장치: 가격 차이가 너무 크면 교체하지 않음
            # 정상적 OCR 보정은 2배 이내 (예: 66000↔68000=1.03배, 80000↔88000=1.1배)
            # 3배 이상 차이 = OCR이 엉뚱한 숫자를 읽은 것
            ratio = max(ocr_pr, abs_ai_pr) / max(min(ocr_pr, abs_ai_pr), 1)
            if ratio >= 3:
                _mlog.warning(
                    f"[NAME_MATCH] SKIP '{ai_name}' {abs_ai_pr} → {ocr_pr} "
                    f"(ratio {ratio:.1f}x too large)"
                )
                continue
            # OCR 가격이 다른 Gemini 항목에는 없는 고유 가격이면 → 강하게 보정
            ocr_is_unique = ocr_pr not in ai_prices
            # 또는 이름 유사도가 매우 높으면 → 보정
            if ocr_is_unique or best_sim >= 0.8:
                sign = -1 if ai_pr < 0 else 1
                old_pr = ai_pr
                ai_item["price"] = sign * ocr_pr
                corrected += 1
                _used_ocr_indices.add(best_ocr_idx)
                _mlog.warning(
                    f"[NAME_MATCH] '{ai_name}' {old_pr} → {sign * ocr_pr} "
                    f"(OCR item '{ocr_name}', sim={best_sim:.2f}, "
                    f"unique={ocr_is_unique})"
                )
    if corrected:
        _mlog.warning(f"[NAME_MATCH] {corrected} items corrected by OCR name matching")
# =========================================================
# ✅ AI ↔ Vision OCR 교차검증
# =========================================================
def _cross_validate_prices(
    ai_items: List[Dict[str, Any]],
    ocr_text: str,
) -> List[Dict[str, Any]]:
    """
    AI가 추출한 항목의 가격이 Vision OCR 텍스트에 존재하는지 검증.
    OCR 텍스트에 없는 가격이면, 가장 가까운 OCR 숫자로 보정 시도.
    (OCR 숫자가 더 정확하므로 — 예: Gemini가 66000을 68000으로 읽는 경우)
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
    if not all_ocr_nums:
        return ai_items
    # 별도로 OCR에서 총액 추출
    ocr_total = _extract_total_amount(ocr_text)
    # 🔒 이미 AI 항목이 사용 중인 가격 집합 (다른 항목의 가격을 빼앗지 않도록)
    ai_used_prices: set = set()
    for item in ai_items:
        _p = item.get("price")
        if _p is not None:
            ai_used_prices.add(abs(_p))
    mismatch_count = 0
    corrected_count = 0
    for item in ai_items:
        pr = item.get("price")
        if pr is None:
            continue
        abs_pr = abs(pr)
        if abs_pr in all_ocr_nums:
            continue  # OK — OCR에도 같은 숫자 있음
        # 🔍 OCR에 없는 가격 → 가장 가까운 OCR 숫자 찾기
        nm = (item.get("itemName") or "")[:40]
        _xlog.warning(f"[XVAL] price not in OCR: '{nm}' = {pr}")
        mismatch_count += 1
        # 유사 가격 후보 찾기 (차이가 10% 이내이고, 자릿수 같고, 100원 단위인 것)
        best_candidate = None
        best_diff = float("inf")
        for ocr_n in all_ocr_nums:
            diff = abs(ocr_n - abs_pr)
            # 차이가 5% 이내이고 자릿수가 같아야 함
            if diff > 0 and diff < abs_pr * 0.05 and len(str(ocr_n)) == len(str(abs_pr)):
                # 🔒 OCR 쓰레기 숫자 필터: 한국 영수증 가격은 100원 단위
                # 52215, 60978 같은 건 OCR 오독 → 후보 제외
                if ocr_n % 100 != 0:
                    _xlog.warning(
                        f"[XVAL] SKIP candidate {ocr_n} for '{nm}' — "
                        f"not a round price (mod100={ocr_n % 100})"
                    )
                    continue
                # 🔒 이미 다른 AI 항목이 이 가격을 사용 중이면 후보 제외
                if ocr_n in ai_used_prices:
                    _xlog.warning(
                        f"[XVAL] SKIP candidate {ocr_n} for '{nm}' — "
                        f"already used by another AI item"
                    )
                    continue
                if diff < best_diff:
                    best_diff = diff
                    best_candidate = ocr_n
        if best_candidate is not None:
            sign = -1 if pr < 0 else 1
            old_price = pr
            new_price = sign * best_candidate
            # 보정 적용
            item["price"] = new_price
            corrected_count += 1
            # 사용 가격 집합 업데이트
            ai_used_prices.discard(abs_pr)
            ai_used_prices.add(best_candidate)
            _xlog.warning(
                f"[XVAL] CORRECTED: '{nm}' {old_price} → {new_price} "
                f"(OCR number {best_candidate} found, diff was {best_diff})"
            )
    if mismatch_count:
        _xlog.warning(
            f"[XVAL] {mismatch_count}/{len(ai_items)} prices not in OCR, "
            f"{corrected_count} corrected"
        )
    return ai_items
# =========================================================
# ✅ 총액 기반 OCR 숫자 혼동 역보정
# =========================================================
# 감열지에서 흔한 숫자 혼동 패턴
_DIGIT_SWAPS = [
    ("6", "8"), ("8", "6"),   # 가장 흔함
    ("5", "6"), ("6", "5"),
    ("3", "8"), ("8", "3"),
    ("0", "8"), ("8", "0"),
    ("1", "7"), ("7", "1"),
    ("5", "8"), ("8", "5"),
    ("2", "0"), ("0", "2"),
    ("3", "5"), ("5", "3"),
    ("9", "0"), ("0", "9"),
    ("9", "4"), ("4", "9"),
    ("7", "2"), ("2", "7"),
]
def _fix_prices_by_total(
    ai_items: List[Dict[str, Any]],
    total_amount: Optional[int],
    ocr_text: str = "",
) -> List[Dict[str, Any]]:
    """
    항목 합계가 totalAmount와 다를 때:
    1단계: 흔한 OCR 숫자 혼동(6↔8 등) 시도 (최대 2항목)
    2단계: 1단계 실패 시 → OCR 텍스트에 있는 숫자로 대체해서 총액 맞추기
    **최소 변경량 우선**: 여러 보정 조합이 가능하면 가격 변동이 가장 작은 것을 선택.
    """
    if not total_amount or not ai_items:
        return ai_items
    import logging
    _flog = logging.getLogger("ocr_policy.fix_total")

    # ✅ 할인 항목(음수 가격)은 합계 계산에서 제외
    #    totalAmount는 이미 할인 적용된 최종 결제액이므로,
    #    양수 항목만 합산해서 비교해야 정확한 보정 가능
    _DISCOUNT_KW = ("할인", "할증", "감면", "절사", "쿠폰", "discount")
    def _is_discount_item(it: Dict[str, Any]) -> bool:
        nm = (it.get("itemName") or "").lower()
        pr = it.get("price") or 0
        return pr < 0 or any(kw in nm for kw in _DISCOUNT_KW)

    current_sum = sum(
        it.get("price") or 0 for it in ai_items if not _is_discount_item(it)
    )
    diff = current_sum - total_amount
    if abs(diff) < 100:
        return ai_items  # 이미 맞음
    _flog.info(f"[FIX_TOTAL] sum={current_sum}, total={total_amount}, diff={diff} (discount items excluded)")
    # 각 항목별 가능한 보정 후보 생성
    candidates: List[tuple] = []
    for idx, item in enumerate(ai_items):
        if _is_discount_item(item):
            continue  # 할인 항목은 보정 대상에서 제외
        pr = item.get("price")
        if pr is None or pr == 0:
            continue
        pr_str = str(abs(pr))
        sign = -1 if pr < 0 else 1
        for old_d, new_d in _DIGIT_SWAPS:
            if old_d not in pr_str:
                continue
            for pos in range(len(pr_str)):
                if pr_str[pos] == old_d:
                    new_pr_str = pr_str[:pos] + new_d + pr_str[pos+1:]
                    new_pr = sign * int(new_pr_str)
                    change = new_pr - pr
                    if change != 0:
                        candidates.append((idx, new_pr, change))
    if not candidates:
        return ai_items
    # 1개 항목 보정: 최소 변경량 우선
    single_fixes = []
    for idx, new_pr, change in candidates:
        if abs(diff + change) < 100:
            single_fixes.append((idx, new_pr, change, abs(change)))
    if single_fixes:
        single_fixes.sort(key=lambda x: x[3])
        idx, new_pr, change, _ = single_fixes[0]
        old_pr = ai_items[idx]["price"]
        nm = (ai_items[idx].get("itemName") or "")[:40]
        ai_items[idx]["price"] = new_pr
        _flog.warning(
            f"[FIX_TOTAL] SINGLE CORRECTION: '{nm}' {old_pr} → {new_pr} "
            f"(diff {diff} → {diff + change})"
        )
        return ai_items
    # 2개 항목 조합 보정: 최소 총 변경량 우선
    double_fixes = []
    for i, (idx1, new_pr1, change1) in enumerate(candidates):
        for idx2, new_pr2, change2 in candidates[i+1:]:
            if idx1 == idx2:
                continue
            if abs(diff + change1 + change2) < 100:
                total_change = abs(change1) + abs(change2)
                double_fixes.append((idx1, new_pr1, change1, idx2, new_pr2, change2, total_change))
    if double_fixes:
        double_fixes.sort(key=lambda x: x[6])
        idx1, new_pr1, change1, idx2, new_pr2, change2, _ = double_fixes[0]
        nm1 = (ai_items[idx1].get("itemName") or "")[:40]
        nm2 = (ai_items[idx2].get("itemName") or "")[:40]
        old_pr1 = ai_items[idx1]["price"]
        old_pr2 = ai_items[idx2]["price"]
        ai_items[idx1]["price"] = new_pr1
        ai_items[idx2]["price"] = new_pr2
        _flog.warning(
            f"[FIX_TOTAL] DOUBLE CORRECTION: "
            f"'{nm1}' {old_pr1}→{new_pr1}, "
            f"'{nm2}' {old_pr2}→{new_pr2} "
            f"(diff {diff} → {diff + change1 + change2})"
        )
        return ai_items
    # ═══════════════════════════════════════════════
    # 2단계: OCR 항목 매칭 기반 보정
    # ═══════════════════════════════════════════════
    if not ocr_text:
        _flog.info(f"[FIX_TOTAL] no digit-swap correction found, no OCR text for phase 2")
        return ai_items
    ocr_items = _extract_items_from_text(ocr_text)
    if not ocr_items:
        _flog.info(f"[FIX_TOTAL] no OCR items for phase 2")
        return ai_items
    def _name_sim(a: str, b: str) -> float:
        na = re.sub(r"[^가-힣a-zA-Z0-9]", "", a.lower())
        nb = re.sub(r"[^가-힣a-zA-Z0-9]", "", b.lower())
        if not na or not nb:
            return 0.0
        short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
        if short in long:
            return 0.9
        common = sum(1 for c in short if c in long)
        return common / max(len(short), 1)
    ocr_replacements = []
    for idx, ai_item in enumerate(ai_items):
        ai_pr = ai_item.get("price")
        if ai_pr is None or ai_pr == 0:
            continue
        ai_name = ai_item.get("itemName") or ""
        for ocr_item in ocr_items:
            ocr_pr = ocr_item.get("price")
            ocr_name = ocr_item.get("itemName") or ""
            if ocr_pr is None or ocr_pr == abs(ai_pr):
                continue
            sim = _name_sim(ai_name, ocr_name)
            if sim < 0.4:
                continue
            sign = -1 if ai_pr < 0 else 1
            new_pr = sign * ocr_pr
            change = new_pr - ai_pr
            new_diff = diff + change
            if abs(new_diff) < 100:
                ocr_replacements.append((idx, new_pr, change, abs(change), sim, ocr_name))
    if ocr_replacements:
        ocr_replacements.sort(key=lambda x: (-x[4], x[3]))
        idx, new_pr, change, _, sim, ocr_nm = ocr_replacements[0]
        old_pr = ai_items[idx]["price"]
        nm = (ai_items[idx].get("itemName") or "")[:40]
        ai_items[idx]["price"] = new_pr
        _flog.warning(
            f"[FIX_TOTAL] OCR MATCH REPLACEMENT: '{nm}' {old_pr} → {new_pr} "
            f"(matched OCR item '{ocr_nm}', sim={sim:.2f}, diff {diff} → {diff + change})"
        )
        return ai_items
    # 3단계 폴백: 이름 매칭 없이 숫자만으로 대체 (최소 변경량)
    all_ocr_nums: set = set()
    for raw_ln in ocr_text.splitlines():
        for m in re.findall(r"[\d,]+\d", raw_ln):
            try:
                n = int(m.replace(",", ""))
                if 100 <= n <= 50_000_000:
                    all_ocr_nums.add(n)
            except ValueError:
                pass
    fallback_replacements = []
    for idx, item in enumerate(ai_items):
        pr = item.get("price")
        if pr is None or pr == 0:
            continue
        abs_pr = abs(pr)
        sign = -1 if pr < 0 else 1
        for ocr_n in all_ocr_nums:
            if ocr_n == abs_pr:
                continue
            new_pr = sign * ocr_n
            change = new_pr - pr
            new_diff = diff + change
            if abs(new_diff) < 100:
                fallback_replacements.append((idx, new_pr, change, abs(change)))
    if fallback_replacements:
        fallback_replacements.sort(key=lambda x: x[3])
        idx, new_pr, change, _ = fallback_replacements[0]
        old_pr = ai_items[idx]["price"]
        nm = (ai_items[idx].get("itemName") or "")[:40]
        ai_items[idx]["price"] = new_pr
        _flog.warning(
            f"[FIX_TOTAL] OCR FALLBACK REPLACEMENT: '{nm}' {old_pr} → {new_pr} "
            f"(diff {diff} → {diff + change})"
        )
        return ai_items
    _flog.info(f"[FIX_TOTAL] no correction found for diff={diff}")
    return ai_items


# =========================================================
# ✅ 카테고리 소계 오염 감지 및 교정
# =========================================================
def _fix_subtotal_contamination(
    ai_items: List[Dict[str, Any]],
    total_amount: Optional[int],
    ocr_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Gemini가 카테고리 소계를 개별 항목 가격으로 잘못 읽는 경우를 감지/교정.

    3단계 방어 (괄호 없어도 작동):
      1단계: 접미사 감지 — OCR의 큰 숫자(소계)의 뒷부분이 항목 가격과 일치
             예: OCR에 1964600 있고 항목 가격이 964600 → 소계 오염
      2단계: 비정상 고가 감지 — 항목 가격이 totalAmount의 40% 초과 시
             진료비/진찰료/검진 등 일반 항목은 이렇게 비쌀 수 없음
      3단계: 총액 기반 감지 — 합계가 totalAmount의 130% 초과 시 교정
    """
    if not ai_items:
        return ai_items

    import logging
    _slog = logging.getLogger("ocr_policy.subtotal")

    # ===== OCR 텍스트에서 모든 숫자 + 라인별 이름→가격 매핑 추출 =====
    all_ocr_nums: set = set()
    large_ocr_nums: set = set()          # 100,000 이상 큰 숫자 (잠재적 소계)
    item_line_prices: dict = {}          # {정규화된_이름: [가격들]}

    if ocr_text:
        for ln in ocr_text.splitlines():
            stripped = ln.strip()
            if not stripped:
                continue
            # 이 라인의 모든 숫자 추출
            line_nums = []
            for m in re.findall(r'[\d,]+\d', stripped):
                try:
                    n = int(m.replace(",", ""))
                    if n >= 100:
                        all_ocr_nums.add(n)
                        line_nums.append(n)
                    if n >= 100_000:
                        large_ocr_nums.add(n)
                except ValueError:
                    pass
            # 이름 → 가격 매핑 (숫자가 2개 이상 있는 라인 = 테이블 항목)
            if len(line_nums) >= 2:
                name_part = re.sub(r'[\d,]+', ' ', stripped)
                name_part = re.sub(r'[^\w가-힣a-zA-Z]', ' ', name_part).strip()
                name_part = re.sub(r'\s+', '', name_part)
                if name_part and len(name_part) >= 2:
                    if name_part not in item_line_prices:
                        item_line_prices[name_part] = []
                    item_line_prices[name_part].extend(line_nums)

    _slog.info(
        f"[SUBTOTAL] large_ocr_nums={large_ocr_nums}, "
        f"item_lines={len(item_line_prices)}, total_amount={total_amount}"
    )

    # ===== 이름으로 OCR에서 올바른 가격 찾는 헬퍼 =====
    def _find_correct_price_by_name(item_name: str, bad_price: int) -> Optional[int]:
        nm_norm = re.sub(r'[^\w가-힣a-zA-Z]', '', item_name)
        if len(nm_norm) < 2:
            return None
        best = None
        for ocr_name, ocr_prices in item_line_prices.items():
            if len(ocr_name) < 2:
                continue
            # 이름 매칭: 부분 포함 (양방향)
            if nm_norm in ocr_name or ocr_name in nm_norm:
                for op in ocr_prices:
                    if 100 <= op < bad_price:
                        if best is None or op > best:  # 가장 큰 합리적 가격
                            best = op
        return best

    # ===== 1단계: 접미사 감지 (괄호 불필요, totalAmount 불필요) =====
    # 예: 항목 가격 964600, OCR에 1964600 존재 → 964600은 1964600의 접미사
    fixed_any = False
    for idx, item in enumerate(ai_items):
        price = item.get("price") or 0
        if price < 100_000:  # 10만원 미만은 소계일 가능성 낮음
            continue
        price_str = str(price)
        nm = item.get("itemName") or ""
        is_suffix_of_larger = False
        for big_n in large_ocr_nums:
            if big_n <= price:
                continue
            big_str = str(big_n)
            # 항목 가격이 큰 OCR 숫자의 접미사인지 확인
            if big_str.endswith(price_str) and len(big_str) > len(price_str):
                is_suffix_of_larger = True
                _slog.warning(
                    f"[SUBTOTAL-SUFFIX] '{nm}' price {price} is suffix of OCR number {big_n}"
                )
                break
        if not is_suffix_of_larger:
            continue
        # 올바른 가격 찾기
        correct = _find_correct_price_by_name(nm, price)
        if correct:
            old_pr = price
            ai_items[idx]["price"] = correct
            fixed_any = True
            _slog.warning(f"[SUBTOTAL-SUFFIX] FIXED: '{nm}' {old_pr} → {correct}")

    if fixed_any:
        new_sum = sum(it.get("price") or 0 for it in ai_items)
        _slog.warning(f"[SUBTOTAL-SUFFIX] after fix, new_sum={new_sum}")
        return ai_items

    # ===== 2단계: 비정상 고가 감지 (totalAmount 필요) =====
    if total_amount and total_amount > 0:
        for idx, item in enumerate(ai_items):
            price = item.get("price") or 0
            if price < 100_000:
                continue
            nm = item.get("itemName") or ""
            nm_lower = nm.lower()
            # 진료/진찰/검진/내복약 등 일반 항목이 totalAmount의 40% 초과 → 비정상
            is_common_item = any(k in nm_lower for k in [
                "진료", "진찰", "검진", "내복약", "외용", "입원",
                "기본", "재진", "초진",
            ])
            if is_common_item and price > total_amount * 0.4:
                _slog.warning(
                    f"[SUBTOTAL-HIGHPRICE] '{nm}' = {price} is >{total_amount * 0.4:.0f} "
                    f"(40% of total {total_amount}) — suspicious for common item"
                )
                correct = _find_correct_price_by_name(nm, price)
                if correct:
                    old_pr = price
                    ai_items[idx]["price"] = correct
                    fixed_any = True
                    _slog.warning(f"[SUBTOTAL-HIGHPRICE] FIXED: '{nm}' {old_pr} → {correct}")

        if fixed_any:
            new_sum = sum(it.get("price") or 0 for it in ai_items)
            _slog.warning(f"[SUBTOTAL-HIGHPRICE] after fix, new_sum={new_sum}")
            return ai_items

    # ===== 3단계: 총액 기반 감지 (가장 넓은 안전망) =====
    if not total_amount:
        return ai_items

    current_sum = sum(it.get("price") or 0 for it in ai_items)

    if current_sum <= total_amount * 1.3:
        return ai_items

    excess = current_sum - total_amount
    _slog.warning(
        f"[SUBTOTAL] sum={current_sum} >> total={total_amount}, "
        f"excess={excess}, checking for subtotal contamination"
    )

    # 소계 후보에서 교체 시도
    best_fix = None
    best_new_diff = abs(current_sum - total_amount)

    # 대체용 OCR 숫자 (큰 숫자 제외 — 소계일 수 있으므로)
    replacement_nums = {n for n in all_ocr_nums if n < 500_000}

    for idx, item in enumerate(ai_items):
        price = item.get("price") or 0
        if price <= 0:
            continue
        if price < excess * 0.5:
            continue

        nm = (item.get("itemName") or "")[:40]
        _slog.info(f"[SUBTOTAL] suspect item: '{nm}' = {price} (excess={excess})")

        for ocr_n in replacement_nums:
            if ocr_n >= price:
                continue
            new_sum = current_sum - price + ocr_n
            new_diff = abs(new_sum - total_amount)
            if new_diff < best_new_diff and new_diff < total_amount * 0.05:
                best_new_diff = new_diff
                best_fix = (idx, ocr_n, price)

    if best_fix:
        idx, new_price, old_price = best_fix
        nm = (ai_items[idx].get("itemName") or "")[:40]
        ai_items[idx]["price"] = new_price
        new_sum = sum(it.get("price") or 0 for it in ai_items)
        _slog.warning(
            f"[SUBTOTAL] FIXED: '{nm}' {old_price} → {new_price} "
            f"(new_sum={new_sum}, total={total_amount})"
        )
    else:
        _slog.info("[SUBTOTAL] no subtotal contamination fix found")

    return ai_items


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
    ocr_items = _extract_items_from_text(ocr_text)
    if not ocr_items:
        return gemini_items
    gemini_sum = sum(it.get("price") or 0 for it in gemini_items)
    if total_amount and abs(gemini_sum - total_amount) < 500:
        _rlog.info(f"[RECOVER] sum matches: gemini={gemini_sum}, total={total_amount}, skip")
        return gemini_items
    _rlog.warning(
        f"[RECOVER] sum mismatch: gemini={gemini_sum}, total={total_amount}, "
        f"diff={abs(gemini_sum - (total_amount or 0))}, "
        f"gemini_count={len(gemini_items)}, ocr_count={len(ocr_items)}"
    )
    def _norm_name(s: str) -> str:
        return re.sub(r"[^가-힣a-zA-Z0-9]", "", s.lower())
    gemini_keys: set = set()
    gemini_names_norm: set = set()
    for it in gemini_items:
        nm = _norm_name(it.get("itemName") or "")
        pr = it.get("price") or 0
        gemini_keys.add((nm, pr))
        gemini_names_norm.add(nm)
    recovered: List[Dict[str, Any]] = []
    for ocr_it in ocr_items:
        ocr_nm = _norm_name(ocr_it.get("itemName") or "")
        ocr_pr = ocr_it.get("price") or 0
        if not ocr_nm or ocr_pr == 0:
            continue
        # 🔒 OCR 쓰레기 필터: 이름이 너무 짧거나 noise
        raw_name = (ocr_it.get("itemName") or "").strip()
        if len(ocr_nm) < 2 or _is_noise_line(raw_name):
            _rlog.warning(f"[RECOVER] SKIP noise/short item: '{raw_name}' = {ocr_pr}")
            continue
        # 🔒 한글 없는 짧은 영문은 OCR 쓰레기 (BMOS, LAD 등)
        _has_korean = any("\uac00" <= ch <= "\ud7a3" for ch in raw_name)
        if not _has_korean and len(ocr_nm) <= 6:
            _rlog.warning(f"[RECOVER] SKIP no-korean short name: '{raw_name}' = {ocr_pr}")
            continue
        # 🔒 100원 단위가 아닌 가격은 OCR 쓰레기일 가능성 높음
        if ocr_pr > 0 and ocr_pr % 100 != 0:
            _rlog.warning(f"[RECOVER] SKIP non-round price: '{raw_name}' = {ocr_pr}")
            continue
        if (ocr_nm, ocr_pr) in gemini_keys:
            continue
        name_exists = False
        for gn in gemini_names_norm:
            if ocr_nm and gn and (ocr_nm in gn or gn in ocr_nm):
                matching_prices = [
                    it.get("price") or 0 for it in gemini_items
                    if _norm_name(it.get("itemName") or "") == gn
                    or ocr_nm in _norm_name(it.get("itemName") or "")
                    or _norm_name(it.get("itemName") or "") in ocr_nm
                ]
                if any(abs(mp - ocr_pr) < max(500, abs(ocr_pr) * 0.1) for mp in matching_prices):
                    name_exists = True
                    break
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
        "visitTime": None,
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
    # ✅ visitTime 추출
    vt = j.get("visitTime")
    if isinstance(vt, str) and vt.strip():
        vt_clean = vt.strip()[:5]
        if re.match(r"^\d{1,2}:\d{2}$", vt_clean):
            parsed["visitTime"] = vt_clean
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
            if not skip_noise_filter and _is_noise_line(nm):
                continue
            pr = _coerce_int_amount(it.get("price"))
            orig_pr = _coerce_int_amount(it.get("originalPrice"))
            disc = _coerce_int_amount(it.get("discount"))
            ct = (it.get("categoryTag") or "").strip() or None
            sn = (it.get("standardName") or "").strip() or None

            # ✅ 할인/할증 항목 필터링: 음수 가격이거나 할인 키워드 포함 시 제외
            #    할인은 totalAmount에 이미 반영되어 있으므로 별도 아이템으로 불필요
            _DISC_KW = ("할인", "할증", "감면", "절사", "쿠폰", "discount")
            nm_lower = nm.lower()
            if pr is not None and pr < 0:
                continue  # 음수 가격 항목 제외
            if any(kw in nm_lower for kw in _DISC_KW) and (pr is None or pr <= 0):
                continue  # 할인 키워드 + 가격 0 이하 항목 제외

            # ✅ 항목별 할인 적용: discount가 있으면 실결제 가격으로 조정
            if disc is not None and disc > 0 and pr is not None:
                net_price = pr - disc
                if net_price <= 0:
                    continue  # 100% 할인 항목 → 무료이므로 제외
                pr = net_price  # 할인 적용된 실결제 가격으로 교체

            if ct:
                ct = _migrate_tag(ct)
            # 🔒 Gemini 태그가 없거나 유효하지 않으면 키워드 기반 폴백
            if not ct:
                ct = _infer_tag_by_keyword(nm)
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
    receipt_max_width: int = 2048,
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
    img_display = _resize_to_width(img.copy(), int(receipt_max_width or 0))
    # ✅ Vision OCR용 이미지: 전처리 적용 (흑백 + 대비 + 선명도)
    ocr_max_w = max(int(receipt_max_width or 0), 2048)
    img_ocr = _resize_to_width(img.copy(), ocr_max_w)
    img_ocr = _preprocess_for_ocr(img_ocr)
    ocr_buf = io.BytesIO()
    img_ocr.save(ocr_buf, format="PNG")
    ocr_image_bytes = ocr_buf.getvalue()
    # ✅ Gemini용 이미지: 컬러 유지 + 그림자/어두운 영수증 밝기 보정
    gemini_max_w = max(int(receipt_max_width or 0), 3072)
    img_gemini = _resize_to_width(img.copy(), gemini_max_w)
    try:
        from PIL import ImageEnhance, ImageStat
        # 평균 밝기 측정 (0~255)
        stat = ImageStat.Stat(img_gemini.convert("L"))
        mean_brightness = stat.mean[0]
        _log = logging.getLogger("ocr_policy")
        _log.info(f"[GEMINI-IMG] mean_brightness={mean_brightness:.1f}")
        if mean_brightness < 140:
            # 어두운 이미지: 밝기 + 대비 보정 (컬러 유지)
            boost = min(1.6, 140 / max(mean_brightness, 30))
            img_gemini = ImageEnhance.Brightness(img_gemini).enhance(boost)
            img_gemini = ImageEnhance.Contrast(img_gemini).enhance(1.3)
            img_gemini = ImageEnhance.Sharpness(img_gemini).enhance(1.5)
            _log.info(f"[GEMINI-IMG] dark image corrected: brightness_boost={boost:.2f}")
    except Exception:
        pass
    gemini_buf = io.BytesIO()
    img_gemini.save(gemini_buf, format="PNG")
    gemini_image_bytes = gemini_buf.getvalue()
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")) or "gemini-3-flash-preview"
    g_timeout = max(60, int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "60") or "60")))
    hints: Dict[str, Any] = {
        "ocrEngine": "none",
        "geminiUsed": False,
        "pipeline": "gemini_first",
    }
    ai_parsed = None
    ai_tags: List[str] = []
    import logging
    _log = logging.getLogger("ocr_policy")
    _log.info(f"[AI] gemini_enabled={g_enabled}, gemini_model={g_model}")
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
    # ✅ Step 2: Gemini AI 영수증 분석
    ai_result_json = None
    if g_enabled and g_key.strip():
        try:
            _log.info(f"[Gemini] calling model={g_model}, timeout={g_timeout}")
            gj = _gemini_parse_receipt_full(
                image_bytes=gemini_image_bytes,
                api_key=g_key,
                model=g_model,
                timeout_seconds=g_timeout,
                ocr_text=ocr_text,
                preprocessed_image_bytes=ocr_image_bytes,
            )
            if isinstance(gj, dict):
                ai_result_json = gj
                hints["ocrEngine"] = f"gemini:{g_model}"
                hints["geminiUsed"] = True
                _log.info(f"[Gemini] SUCCESS: items={len(gj.get('items', []))}")
            else:
                _log.warning("[Gemini] returned non-dict")
        except Exception as e:
            _log.error(f"[Gemini] ERROR: {e}")
            hints["geminiError"] = str(e)[:200]
    if ai_result_json is None:
        _log.warning("[AI] Gemini failed or disabled")
    # --- 정규화 ---
    if isinstance(ai_result_json, dict):
        ai_parsed, ai_tags = _normalize_gemini_full_result(
            ai_result_json, skip_noise_filter=True,
        )
        _log.info(f"[AI] normalized: items={len(ai_parsed.get('items', []))}, tags={ai_tags}")
        # OCR 교차검증 + 총액 기반 보정 + 누락 복구 적용
        if ocr_text and ai_parsed.get("items"):
            # 🔍 보정 전 상태 로깅
            _pre_prices = {(it.get("itemName") or "")[:30]: it.get("price") for it in ai_parsed["items"]}
            _log.warning(f"[DEBUG-PRE] version=v10-tag-boost | Gemini prices: {_pre_prices}")
            _ocr_extracted = _extract_items_from_text(ocr_text)
            _ocr_prices = {(it.get("itemName") or "")[:30]: it.get("price") for it in _ocr_extracted}
            _log.warning(f"[DEBUG-OCR] extracted items: {_ocr_prices}")
            # totalAmount 결정 (소계 감지에 필요)
            gemini_total = ai_parsed.get("totalAmount")
            total_amt = gemini_total
            gemini_sum = sum(it.get("price") or 0 for it in ai_parsed["items"])
            if ocr_total_amount and ocr_total_amount > 0:
                _log.warning(
                    f"[TOTAL] OCR total={ocr_total_amount}, Gemini total={gemini_total}, "
                    f"Gemini sum={gemini_sum}"
                )
                # ✅ OCR total 신뢰성 검증:
                # Gemini total과 sum이 일치하면 → Gemini가 거의 확실히 맞음
                _use_ocr_total = True
                if gemini_total and gemini_sum and gemini_total > 0:
                    _gem_exact = abs(gemini_sum - gemini_total) < max(500, gemini_total * 0.01)
                    _gem_close = abs(gemini_sum - gemini_total) < gemini_total * 0.1
                    _ocr_diff = abs(ocr_total_amount - gemini_total)
                    if _gem_exact and _ocr_diff > 0:
                        # Gemini sum ≈ Gemini total (거의 정확히 일치)
                        # → OCR total이 조금이라도 다르면 Gemini 신뢰
                        _log.warning(
                            f"[TOTAL] Gemini sum==total ({gemini_sum}=={gemini_total}), "
                            f"OCR total={ocr_total_amount} differs by {_ocr_diff}, "
                            f"trusting Gemini"
                        )
                        _use_ocr_total = False
                    elif _gem_close and _ocr_diff > gemini_total * 0.05:
                        # Gemini sum ≈ total (10% 이내) + OCR total이 5% 이상 벗어남
                        _log.warning(
                            f"[TOTAL] OCR total={ocr_total_amount} is far from "
                            f"Gemini total={gemini_total} (sum={gemini_sum}), "
                            f"keeping Gemini total"
                        )
                        _use_ocr_total = False
                    # ✅ 할인 시나리오 감지:
                    # Gemini sum ≈ OCR total AND Gemini total < Gemini sum
                    # → 할인 전 합계 = item sum, 결제요청 = Gemini total
                    elif (gemini_total < gemini_sum
                          and abs(ocr_total_amount - gemini_sum) < max(500, gemini_sum * 0.02)):
                        _disc_diff = gemini_sum - gemini_total
                        _log.warning(
                            f"[TOTAL] DISCOUNT detected: "
                            f"Gemini sum({gemini_sum}) ≈ OCR total({ocr_total_amount}), "
                            f"Gemini total({gemini_total}) = 결제요청, "
                            f"discount ≈ {_disc_diff}, trusting Gemini total"
                        )
                        _use_ocr_total = False
                if _use_ocr_total:
                    total_amt = ocr_total_amount
                    ai_parsed["totalAmount"] = ocr_total_amount
            # 🔍 OCR 텍스트 원본 로깅 (디버깅용 — 500자)
            _log.warning(f"[DEBUG-OCRTEXT] {repr((ocr_text or '')[:500])}")
            # 🔍 OCR 텍스트의 모든 숫자 로깅
            _all_nums_debug = sorted(set(
                int(m.replace(",", ""))
                for ln in (ocr_text or "").splitlines()
                for m in re.findall(r"[\d,]+\d", ln)
                if int(m.replace(",", "")) >= 1000
            ))
            _log.warning(f"[DEBUG-OCRNUMS] all numbers >= 1000: {_all_nums_debug}")
            # Step B-1: ✅ 카테고리 소계 오염 감지/교정 (★ 가장 먼저 실행!)
            _pre_b1 = {(it.get("itemName") or "")[:20]: it.get("price") for it in ai_parsed["items"]}
            ai_parsed["items"] = _fix_subtotal_contamination(
                ai_parsed["items"], total_amt, ocr_text=ocr_text
            )
            _post_b1 = {(it.get("itemName") or "")[:20]: it.get("price") for it in ai_parsed["items"]}
            if _pre_b1 != _post_b1:
                _log.warning(f"[DEBUG-B1] CHANGED: {_pre_b1} → {_post_b1}")
            else:
                _log.warning(f"[DEBUG-B1] no change: {_post_b1}")
            # Step 0: ✅ OCR 항목 이름 매칭 보정
            _fix_prices_by_ocr_name_match(ai_parsed["items"], ocr_text)
            _post_s0 = {(it.get("itemName") or "")[:20]: it.get("price") for it in ai_parsed["items"]}
            if _post_b1 != _post_s0:
                _log.warning(f"[DEBUG-S0] CHANGED: {_post_b1} → {_post_s0}")
            # Step A: OCR 텍스트 숫자 교차검증
            _cross_validate_prices(ai_parsed["items"], ocr_text)
            _post_sa = {(it.get("itemName") or "")[:20]: it.get("price") for it in ai_parsed["items"]}
            if _post_s0 != _post_sa:
                _log.warning(f"[DEBUG-SA] CHANGED: {_post_s0} → {_post_sa}")

            # Step B-2: ✅ 총액 기반 숫자 혼동 역보정 (6↔8 등)
            ai_parsed["items"] = _fix_prices_by_total(
                ai_parsed["items"], total_amt, ocr_text=ocr_text
            )
            # Step C: 누락 항목 복구
            before_recover = len(ai_parsed["items"])
            ai_parsed["items"] = _recover_missing_items(
                ai_parsed["items"], ocr_text, total_amt
            )
            recovered_count = len(ai_parsed["items"]) - before_recover
            if recovered_count > 0:
                _log.warning(f"[RECOVER] {recovered_count} items recovered from Vision OCR")
                hints["recovered_items"] = recovered_count
            # Step D: ✅ OCR 텍스트 기반 항목명 교정 (Gemini가 이름을 카테고리명으로 바꿔버리는 문제 대응)
            ocr_extracted_items = _extract_items_from_text(ocr_text or "")
            if ocr_extracted_items and ai_parsed.get("items"):
                _log.info(f"[NAME-FIX] OCR items: {[(it['itemName'], it['price']) for it in ocr_extracted_items]}")
                _log.info(f"[NAME-FIX] AI  items: {[(it['itemName'], it['price']) for it in ai_parsed['items']]}")
                # 가격 기반으로 OCR 항목 ↔ AI 항목 매칭 후 이름 교정
                ocr_by_price: Dict[int, List[str]] = {}
                for oi in ocr_extracted_items:
                    p = oi.get("price") or 0
                    if p >= 100:
                        ocr_by_price.setdefault(p, []).append(oi.get("itemName") or "")
                used_ocr_names: set = set()
                for ai_it in ai_parsed["items"]:
                    ai_price = ai_it.get("price") or 0
                    ai_name = (ai_it.get("itemName") or "").strip()
                    # 같은 가격의 OCR 항목이 있으면 이름 교정
                    if ai_price in ocr_by_price:
                        for ocr_name in ocr_by_price[ai_price]:
                            if ocr_name in used_ocr_names:
                                continue
                            ocr_low = ocr_name.lower()
                            ai_low = ai_name.lower()
                            # 이미 같으면 스킵
                            if ocr_low == ai_low:
                                used_ocr_names.add(ocr_name)
                                break
                            # AI가 카테고리명으로 바꿨으면 OCR 원본명으로 교정
                            _log.warning(f"[NAME-FIX] '{ai_name}' → '{ocr_name}' (price={ai_price})")
                            ai_it["itemName"] = ocr_name
                            used_ocr_names.add(ocr_name)
                            break
            # 🔒 최종 세금/합계 항목 필터 (Gemini가 비과세/부가세를 항목으로 넣는 경우 제거)
            _TAX_KEYWORDS = {"비과세", "부가세", "과세", "공급가액", "과세공급가액", "부가세액",
                             "소계", "합계", "총액", "총금액", "청구금액", "결제요청", "결제금액"}
            _before_tax_filter = len(ai_parsed["items"])
            _filtered_items = []
            for _it in ai_parsed["items"]:
                _iname = (_it.get("itemName") or "").strip()
                _iname_norm = re.sub(r"[^가-힣a-zA-Z]", "", _iname)
                if _iname_norm in _TAX_KEYWORDS:
                    _log.warning(f"[TAX-FILTER] removing: '{_iname}' = {_it.get('price')}")
                    continue
                _filtered_items.append(_it)
            ai_parsed["items"] = _filtered_items
            if len(_filtered_items) < _before_tax_filter:
                _log.warning(f"[TAX-FILTER] removed {_before_tax_filter - len(_filtered_items)} tax/summary items")
            # 🔍 보정 후 상태 로깅
            _post_prices = {(it.get("itemName") or "")[:30]: it.get("price") for it in ai_parsed["items"]}
            _final_sum = sum(it.get("price") or 0 for it in ai_parsed["items"])
            _log.warning(f"[DEBUG-POST] final prices: {_post_prices}")
            _log.warning(f"[DEBUG-POST] final_sum={_final_sum}, totalAmount={ai_parsed.get('totalAmount')}")
    # ✅ 저장/표시용은 display 이미지 사용
    original_webp = _to_webp_bytes(img_display, quality=int(receipt_webp_quality or 85))
    redacted = _redact_image_with_vision_tokens(img_display.copy(), vision_resp)
    redacted_webp = _to_webp_bytes(redacted, quality=int(receipt_webp_quality or 85))
    webp_bytes = redacted_webp
    if ai_parsed and ai_parsed.get("items"):
        parsed = ai_parsed
        parsed["ocrText"] = (ocr_text or "")[:8000]
        hints["pipeline"] = hints.get("pipeline", "ai_primary")
        hints["_ocr_version"] = "v10-tag-boost"
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
    receipt_max_width: int = 2048,
    receipt_webp_quality: int = 85,
    gemini_enabled: bool = True,
    gemini_api_key: str = "",
    gemini_model_name: str = "gemini-3-flash-preview",
    gemini_timeout: int = 20,
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
        "visit_time": parsed.get("visitTime"),
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
