# ocr_policy.py (PetHealth+)
# - OCR + minimal redaction + receipt parsing (NO tag decision)
# - Google Vision OCR (primary)
# - Gemini (optional) to repair/structure parse

from __future__ import annotations

import io
import os
import re
import json
import math
import threading
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from PIL import Image, ImageOps, ImageDraw


# -----------------------------
# Globals: OCR concurrency guard
# -----------------------------
_OCR_SEMA_LOCK = threading.Lock()
_OCR_SEMA: Optional[threading.Semaphore] = None
_OCR_SEMA_N: int = 0


def _get_ocr_sema(max_concurrency: int) -> threading.Semaphore:
    global _OCR_SEMA, _OCR_SEMA_N
    n = int(max_concurrency or 1)
    n = max(1, min(n, 16))
    with _OCR_SEMA_LOCK:
        if _OCR_SEMA is None or _OCR_SEMA_N != n:
            _OCR_SEMA = threading.Semaphore(n)
            _OCR_SEMA_N = n
        return _OCR_SEMA


# -----------------------------
# Regex helpers
# -----------------------------
_DATE_RE = re.compile(r"(19\d{2}|20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})")
_AMOUNT_TOKEN_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{3,}\b")  # excludes "1"
_PHONE_RE = re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b")
_BIZNO_RE = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")
_CARDISH_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3,4}\b|\b\d{4}\s\d{4}\s\d{4}\s\d{3,4}\b")

_NOISE_LINE_TOKENS = [
    "serial", "sign",
    "고객", "고객번호", "고객 번호", "고객이름", "고객 이름",
    "발행", "발행일", "발행 일",
    "사업자", "사업자등록", "사업자 등록",
    "대표", "원장",
    "전화", "tel", "fax",
    "주소",
    "승인", "카드", "현금",
    "부가세", "vat", "면세", "과세", "공급가",
    "소계", "합계", "총액", "총 금액", "총금액",
    "청구", "결제",
]


def _is_pdf_bytes(data: bytes) -> bool:
    return bool(data) and data[:5] == b"%PDF-"


def _coerce_amount_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
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
        m = _AMOUNT_TOKEN_RE.search(s.replace(" ", ""))
        if not m:
            return None
        num = m.group(0).replace(",", "")
        try:
            return int(num)
        except Exception:
            return None
    return None


def _parse_date_flex(s: str) -> Optional[date]:
    if not isinstance(s, str):
        return None
    ss = s.strip()
    if not ss:
        return None
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%y-%m-%d", "%y.%m.%d", "%y/%m/%d"):
        try:
            return datetime.strptime(ss, fmt).date()
        except Exception:
            pass
    m = _DATE_RE.search(ss)
    if m:
        try:
            y = int(m.group(1))
            mo = int(m.group(2))
            d = int(m.group(3))
            return date(y, mo, d)
        except Exception:
            return None
    return None


def _safe_image_from_bytes(raw: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(raw))
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        return im
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _resize_harden(im: Image.Image, *, max_width: int, max_pixels: int) -> Image.Image:
    w, h = im.size
    if w <= 0 or h <= 0:
        raise HTTPException(status_code=400, detail="Invalid image size")

    # pixel cap
    px = w * h
    if px > int(max_pixels):
        scale = math.sqrt(float(max_pixels) / float(px))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        im = im.resize((nw, nh), Image.LANCZOS)
        w, h = im.size

    # width cap
    mw = int(max_width or 1024)
    mw = max(320, min(mw, 2400))
    if w > mw:
        scale = mw / float(w)
        nh = max(1, int(h * scale))
        im = im.resize((mw, nh), Image.LANCZOS)

    return im


def _img_to_jpeg_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()


def _img_to_webp_bytes(im: Image.Image, *, quality: int) -> bytes:
    q = int(quality or 85)
    q = max(40, min(q, 95))
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=q, method=6)
    return buf.getvalue()


def _load_google_creds_any(google_credentials: str) -> Optional[dict]:
    """
    google_credentials can be:
      - JSON string
      - file path
      - empty => try env var
    Also supports the user's common typo env var:
      GOOGLE_APPLICATION_CREDENTIALIALS
    """
    s = (google_credentials or "").strip()

    # env fallback (correct + common typo)
    if not s:
        s = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if not s:
        s = (os.getenv("GOOGLE_APPLICATION_CREDENTIALIALS") or "").strip()

    if not s:
        return None

    # file path
    if os.path.exists(s):
        try:
            with open(s, "r", encoding="utf-8") as f:
                info = json.load(f)
            return _normalize_private_key_newlines(info)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read service account file: {e}")

    # JSON string
    if s.startswith("{"):
        try:
            info = json.loads(s)
            return _normalize_private_key_newlines(info)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Invalid service account JSON: {e}")

    # unknown format
    return None


def _normalize_private_key_newlines(info: dict) -> dict:
    if not isinstance(info, dict):
        return info
    pk = info.get("private_key")
    if isinstance(pk, str) and "\\n" in pk:
        info["private_key"] = pk.replace("\\n", "\n")
    return info


def _vision_ocr_text(
    *,
    jpeg_bytes: bytes,
    google_credentials: str,
    timeout_seconds: int,
) -> Tuple[str, List[Any]]:
    """
    Returns (full_text, text_annotations)
    Lazy-imports google libs to avoid import-time crashes killing the module.
    """
    try:
        from google.cloud import vision  # type: ignore
        from google.oauth2 import service_account  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"google-cloud-vision not available: {e}")

    info = _load_google_creds_any(google_credentials)

    try:
        if info:
            creds = service_account.Credentials.from_service_account_info(info)
            client = vision.ImageAnnotatorClient(credentials=creds)
        else:
            client = vision.ImageAnnotatorClient()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vision client init failed: {e}")

    image = vision.Image(content=jpeg_bytes)
    ctx = vision.ImageContext(language_hints=["ko", "en"])

    try:
        resp = client.document_text_detection(image=image, image_context=ctx, timeout=int(timeout_seconds or 12))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vision OCR failed: {e}")

    if getattr(resp, "error", None) and getattr(resp.error, "message", None):
        raise HTTPException(status_code=502, detail=f"Vision OCR error: {resp.error.message}")

    full_text = ""
    try:
        full_text = (resp.full_text_annotation.text or "").strip()
    except Exception:
        full_text = ""

    anns = []
    try:
        anns = list(resp.text_annotations or [])
    except Exception:
        anns = []

    return full_text, anns


def _redact_image_minimal(im: Image.Image, text_annotations: List[Any]) -> Image.Image:
    """
    Very small redaction: phone / biz number / card-like tokens
    Uses bounding boxes from Vision text_annotations when available.
    """
    if not text_annotations:
        return im

    draw = ImageDraw.Draw(im)
    pad = 2

    def _box_from_vertices(verts: Any) -> Optional[Tuple[int, int, int, int]]:
        try:
            xs = [int(v.x or 0) for v in verts]
            ys = [int(v.y or 0) for v in verts]
            x0, x1 = max(0, min(xs) - pad), min(im.size[0], max(xs) + pad)
            y0, y1 = max(0, min(ys) - pad), min(im.size[1], max(ys) + pad)
            if x1 <= x0 or y1 <= y0:
                return None
            return (x0, y0, x1, y1)
        except Exception:
            return None

    # skip [0] which is usually the full text block
    for ann in text_annotations[1:]:
        desc = ""
        try:
            desc = str(getattr(ann, "description", "") or "").strip()
        except Exception:
            desc = ""
        if not desc:
            continue

        if not (_PHONE_RE.search(desc) or _BIZNO_RE.search(desc) or _CARDISH_RE.search(desc)):
            continue

        try:
            bp = ann.bounding_poly
            verts = getattr(bp, "vertices", None)
            if not verts:
                continue
            box = _box_from_vertices(verts)
            if box:
                draw.rectangle(box, fill=(0, 0, 0))
        except Exception:
            continue

    return im


def _extract_address_hint(lines: List[str]) -> Optional[str]:
    # very light heuristic
    regions = ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종",
               "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
    for ln in lines:
        if len(ln) < 6:
            continue
        if any(r in ln for r in regions) and ("로" in ln or "길" in ln or "동" in ln):
            return ln[:140]
    return None


def _guess_hospital_name(lines: List[str]) -> Optional[str]:
    # Prefer explicit "원 명:" etc
    for ln in lines:
        s = ln.replace(" ", "")
        if "원명:" in s or "병원명:" in s:
            # try split by :
            parts = re.split(r"[:：]", ln, maxsplit=1)
            if len(parts) == 2:
                cand = parts[1].strip()
                if cand:
                    return cand[:80]
    # Else first line containing 동물병원/병원
    for ln in lines:
        if "동물병원" in ln:
            return ln.strip()[:80]
    for ln in lines:
        if ln.strip().endswith("병원"):
            return ln.strip()[:80]
    return None


def _guess_visit_date(lines: List[str]) -> Optional[str]:
    # Prefer lines with "날짜"
    for ln in lines:
        if "날짜" in ln or "일자" in ln:
            m = _DATE_RE.search(ln)
            if m:
                try:
                    y = int(m.group(1)); mo = int(m.group(2)); d = int(m.group(3))
                    return date(y, mo, d).isoformat()
                except Exception:
                    pass
            # fallback parsing
            dt = _parse_date_flex(ln)
            if dt:
                return dt.isoformat()

    # Else first date-like token anywhere
    for ln in lines:
        m = _DATE_RE.search(ln)
        if m:
            try:
                y = int(m.group(1)); mo = int(m.group(2)); d = int(m.group(3))
                return date(y, mo, d).isoformat()
            except Exception:
                pass
    return None


def _guess_total_amount(lines: List[str]) -> Optional[int]:
    key_tokens = ["청구", "결제", "합계", "총액", "총금액", "총 금액", "소계"]
    candidates: List[int] = []

    def amounts_in_line(ln: str) -> List[int]:
        out = []
        for tok in _AMOUNT_TOKEN_RE.findall(ln):
            v = _coerce_amount_int(tok)
            if isinstance(v, int) and v > 0:
                out.append(v)
        return out

    for ln in lines:
        low = ln.lower()
        if any(t in ln or t in low for t in key_tokens):
            candidates += amounts_in_line(ln)

    if candidates:
        return max(candidates)

    # fallback: max amount in entire text
    all_amt: List[int] = []
    for ln in lines:
        all_amt += amounts_in_line(ln)
    return max(all_amt) if all_amt else None


def _is_noise_line(ln: str) -> bool:
    s = (ln or "").strip()
    if not s:
        return True
    low = s.lower()
    # very short => likely garbage
    if len(re.sub(r"[^0-9a-zA-Z가-힣]", "", low)) < 2:
        return True
    for t in _NOISE_LINE_TOKENS:
        if t in s or t in low:
            return True
    return False


def _parse_items(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Extract receipt line items.
    Looks for the table section after header like "항목 / 금액".
    Falls back to scanning all lines if no section found.
    """
    start = None
    for i, ln in enumerate(lines):
        if ("항목" in ln and "금액" in ln) or ("단가" in ln and "수량" in ln and "금액" in ln):
            start = i + 1
            break

    end = None
    if start is not None:
        for j in range(start, min(len(lines), start + 80)):
            if any(k in lines[j] for k in ("소계", "합계", "총액", "총금액", "청구", "결제")):
                end = j
                break

    scan = lines[start:end] if (start is not None) else lines
    items: List[Dict[str, Any]] = []

    for ln in scan:
        if _is_noise_line(ln):
            continue

        # amounts (>= 3 digits or comma-format) - excludes qty like "1"
        amt_tokens = _AMOUNT_TOKEN_RE.findall(ln)
        if not amt_tokens:
            continue

        # pick last amount as price
        price = _coerce_amount_int(amt_tokens[-1])
        if not isinstance(price, int) or price <= 0:
            continue

        # name = before first amount token
        first_tok = amt_tokens[0]
        pos = ln.find(first_tok)
        name = ln[:pos].strip() if pos > 0 else ln.strip()

        # cleanup name
        name = re.sub(r"^[\*\-\·\•\+\#\(\)\[\]\{\}\s]+", "", name).strip()
        name = re.sub(r"\s{2,}", " ", name).strip()
        if not name:
            continue

        # skip if name is basically total-ish token
        if any(k in name for k in ("소계", "합계", "총액", "총금액", "청구", "결제")):
            continue

        items.append({"itemName": name[:200], "price": int(price), "categoryTag": None})

    # de-dup by (name, price) while keeping order
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = (it["itemName"].lower(), int(it["price"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)

    return out[:80]


def _gemini_enabled() -> bool:
    v = (os.getenv("GEMINI_ENABLED") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _gemini_parse_receipt(ocr_text: str, heuristic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    model_name = (os.getenv("GEMINI_MODEL_NAME") or "gemini-2.5-flash").strip()

    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception:
        return None

    prompt = f"""
너는 영수증 OCR 텍스트를 구조화 JSON으로 변환하는 파서야.
반드시 JSON만 출력해. (설명/마크다운/코드펜스 금지)

스키마:
{{
  "hospitalName": string|null,
  "visitDate": "YYYY-MM-DD"|null,
  "totalAmount": integer|null,
  "items": [{{"itemName": string, "price": integer|null}}]
}}

규칙:
- "Rabbies" 같은 오타는 "Rabies"로 교정해도 됨.
- 가격은 원 단위 정수 (쉼표 제거)
- items는 실제 항목만 넣고 "합계/총액/부가세/결제" 같은 라인은 제외
- 확실하지 않으면 null

휴리스틱(참고용):
{json.dumps(heuristic, ensure_ascii=False)}

OCR 원문:
\"\"\"{ocr_text[:12000]}\"\"\"
""".strip()

    try:
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            return None
    except Exception:
        return None

    # extract JSON object safely
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    blob = m.group(0)

    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def process_receipt(
    raw: bytes,
    *,
    google_credentials: str,
    ocr_timeout_seconds: int = 12,
    ocr_max_concurrency: int = 4,
    ocr_sema_acquire_timeout_seconds: float = 1.0,
    receipt_max_width: int = 1024,
    receipt_webp_quality: int = 85,
    image_max_pixels: int = 20_000_000,
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      (webp_bytes, parsed, hints)
    parsed:
      {
        "hospitalName": str|None,
        "visitDate": "YYYY-MM-DD"|None,
        "totalAmount": int|None,
        "items": [{"itemName": str, "price": int|None, "categoryTag": None}, ...]
      }
    hints:
      { "addressHint": str|None, "ocrProvider": str, "geminiUsed": bool }
    """
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    if _is_pdf_bytes(raw):
        raise HTTPException(status_code=400, detail="PDF is not supported for receipts (upload an image)")

    im = _safe_image_from_bytes(raw)
    im = _resize_harden(im, max_width=int(receipt_max_width), max_pixels=int(image_max_pixels))

    # OCR (guarded)
    sema = _get_ocr_sema(int(ocr_max_concurrency))
    acquired = sema.acquire(timeout=float(ocr_sema_acquire_timeout_seconds or 1.0))
    if not acquired:
        raise HTTPException(status_code=429, detail="OCR is busy. Try again in a moment.")

    ocr_text = ""
    text_annotations: List[Any] = []
    try:
        jpeg_bytes = _img_to_jpeg_bytes(im)
        ocr_text, text_annotations = _vision_ocr_text(
            jpeg_bytes=jpeg_bytes,
            google_credentials=google_credentials,
            timeout_seconds=int(ocr_timeout_seconds),
        )
    finally:
        try:
            sema.release()
        except Exception:
            pass

    lines = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()]

    # Minimal redaction on image before storing
    try:
        im_redacted = im.copy()
        im_redacted = _redact_image_minimal(im_redacted, text_annotations)
    except Exception:
        im_redacted = im

    webp_bytes = _img_to_webp_bytes(im_redacted, quality=int(receipt_webp_quality))

    # Heuristic parse
    hospital_name = _guess_hospital_name(lines)
    visit_date_iso = _guess_visit_date(lines)
    total_amount = _guess_total_amount(lines)
    items = _parse_items(lines)

    heuristic = {
        "hospitalName": hospital_name,
        "visitDate": visit_date_iso,
        "totalAmount": total_amount,
        "items": items,
    }

    hints: Dict[str, Any] = {
        "addressHint": _extract_address_hint(lines),
        "ocrProvider": "google_vision",
        "geminiUsed": False,
    }

    # Gemini repair/structure (optional)
    if _gemini_enabled() and ocr_text.strip():
        g = _gemini_parse_receipt(ocr_text, heuristic)
        if isinstance(g, dict):
            hints["geminiUsed"] = True

            hn = g.get("hospitalName")
            if isinstance(hn, str) and hn.strip():
                hospital_name = hn.strip()[:80]

            vd = g.get("visitDate")
            if isinstance(vd, str):
                dt = _parse_date_flex(vd)
                if dt:
                    visit_date_iso = dt.isoformat()

            ta = _coerce_amount_int(g.get("totalAmount"))
            if isinstance(ta, int) and ta > 0:
                total_amount = int(ta)

            gi = g.get("items")
            if isinstance(gi, list):
                repaired_items: List[Dict[str, Any]] = []
                for it in gi[:80]:
                    if not isinstance(it, dict):
                        continue
                    nm = (it.get("itemName") or "").strip()
                    if not nm:
                        continue
                    pr = _coerce_amount_int(it.get("price"))
                    if pr is not None and pr <= 0:
                        pr = None
                    repaired_items.append({"itemName": nm[:200], "price": pr, "categoryTag": None})
                if repaired_items:
                    items = repaired_items

    parsed: Dict[str, Any] = {
        "hospitalName": hospital_name,
        "visitDate": visit_date_iso,
        "totalAmount": total_amount,
        "items": items,
    }

    return webp_bytes, parsed, hints


