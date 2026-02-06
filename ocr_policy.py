# ocr_policy.py
# PetHealth+ - Receipt OCR + minimal parsing
#
# This module is imported by main.py.
# Required public API:
#   process_receipt(raw: bytes, **kwargs) -> (webp_bytes: bytes, parsed: dict, hints: dict)
#
# Design goals:
# - Never crash the server due to OCR problems. If OCR fails, still return a valid webp and empty parsed data.
# - Keep dependencies minimal (Pillow). Google Vision OCR is optional.

from __future__ import annotations

import io
import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageOps  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Pillow (PIL) is required for ocr_policy.py. Add 'Pillow' to requirements.txt."
    ) from e


# ---------------------------------------------------------
# Concurrency control (optional)
# ---------------------------------------------------------
_OCR_SEMA: Optional[threading.BoundedSemaphore] = None
_OCR_SEMA_N: int = 0
_OCR_SEMA_LOCK = threading.Lock()


def _get_sema(max_concurrency: int) -> threading.BoundedSemaphore:
    global _OCR_SEMA, _OCR_SEMA_N
    n = max(1, int(max_concurrency or 1))
    with _OCR_SEMA_LOCK:
        if _OCR_SEMA is None or _OCR_SEMA_N != n:
            _OCR_SEMA = threading.BoundedSemaphore(value=n)
            _OCR_SEMA_N = n
        return _OCR_SEMA


# ---------------------------------------------------------
# Image helpers
# ---------------------------------------------------------
def _load_image(raw: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(raw))
    # Apply EXIF orientation (common for iOS photos)
    img = ImageOps.exif_transpose(img)
    # Normalize mode
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
        # Fallback (if WebP is not compiled in Pillow build)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85, optimize=True)
        return out.getvalue()


def _encode_jpeg_for_ocr(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85, optimize=True)
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

        # document_text_detection usually works better for receipts
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
# Parsing helpers
# ---------------------------------------------------------
_DATE_PATTERNS = [
    # 2024-02-05 / 2024.02.05 / 2024/2/5
    re.compile(r"(20\d{2})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})", re.UNICODE),
    # 24-02-05 -> assume 20xx
    re.compile(r"\b(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b"),
]

_TOTAL_KEYWORDS = [
    "합계",
    "총액",
    "총 금액",
    "총금액",
    "결제",
    "결제금액",
    "청구",
    "청구금액",
    "총진료비",
    "총 진료비",
    "total",
    "amount",
    "sum",
]


def _parse_amount(s: str) -> Optional[int]:
    digits = re.sub(r"[^\d]", "", s or "")
    if not digits:
        return None
    try:
        v = int(digits)
        return v if v >= 0 else None
    except Exception:
        return None


def _pick_visit_date(lines: List[str]) -> Optional[str]:
    for line in lines[:80]:
        l = line.strip()
        if not l:
            continue

        # YYYY...
        m = _DATE_PATTERNS[0].search(l)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"

        # YY...
        m2 = _DATE_PATTERNS[1].search(l)
        if m2:
            yy, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            y = 2000 + yy
            if 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"

    return None


def _pick_hospital_name(lines: List[str]) -> Optional[str]:
    # Prefer lines that contain "병원" or "동물"
    for line in lines[:20]:
        l = line.strip()
        if not l:
            continue
        if "병원" in l or "동물" in l:
            # Avoid pure address/phone lines
            if re.search(r"\d{2,}", l) and len(l) < 6:
                continue
            return l[:60]

    # fallback: first meaningful non-numeric header line
    for line in lines[:20]:
        l = line.strip()
        if not l:
            continue
        # ignore lines that are mostly digits/punctuation
        if len(re.sub(r"[\d\W_]+", "", l)) < 2:
            continue
        return l[:60]

    return None


def _pick_total_amount(lines: List[str]) -> Optional[int]:
    candidates: List[int] = []

    for line in lines:
        l = line.strip()
        if not l:
            continue

        lower = l.lower()
        if any(k in l for k in _TOTAL_KEYWORDS) or any(k in lower for k in _TOTAL_KEYWORDS):
            # Collect all numbers in this line
            for m in re.finditer(r"\d[\d,]*", l):
                amt = _parse_amount(m.group(0))
                if amt is not None and amt > 0:
                    candidates.append(amt)

    if candidates:
        return max(candidates)

    # fallback: pick the maximum number that "looks like money" across all lines
    all_nums: List[int] = []
    for line in lines:
        for m in re.finditer(r"\d[\d,]*", line):
            amt = _parse_amount(m.group(0))
            if amt is None:
                continue
            # heuristics: avoid tiny numbers like 2024, 2, 5
            if amt >= 1000:
                all_nums.append(amt)

    if all_nums:
        return max(all_nums)

    return None


def _extract_items(lines: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    """
    Robust item extraction for receipts.

    Supports patterns:
      A) "ItemName 10,000"
      B) "ItemName 10,000원"
      C) "ItemName 10,000 1 10,000"   (unit price / qty / total)
      D) "ItemName : 10,000" or "ItemName ₩10,000"

    Filters:
      - total/vat/payment lines
      - phone/business number/time/date etc.
      - tiny numbers (e.g., 9, 58) that often come from "customer no" or timestamps
    """
    out: List[Dict[str, Any]] = []

    # Strong “not an item” keywords
    banned_words = [
        # totals/payments
        "합계", "총액", "총금액", "총 금액", "총진료비", "총 진료비", "결제", "결제금액", "청구", "청구금액",
        "부가", "vat", "면세", "과세", "카드", "현금", "승인", "영수증", "거스름", "거래", "승인번호",
        # header/meta
        "사업자", "대표", "전화", "tel", "fax", "주소", "약도", "고객", "고객번호", "번호", "발행", "일시",
        "시간", "날짜", "no.", "serial", "code",
    ]

    # Skip lines that are too “meta”
    phone_like = re.compile(r"(01[016789]|0\d{1,2})[-\s]?\d{3,4}[-\s]?\d{4}")
    bizno_like = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{5}\b")
    time_like = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
    date_like = re.compile(r"(20\d{2})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})")

    def has_banned(name: str) -> bool:
        l = name.lower()
        for b in banned_words:
            if b.lower() in l:
                return True
        return False

    # Heuristic: drop tiny “prices” that are almost always noise
    def price_is_noise(p: int) -> bool:
        return p < 500  # <- tweak: 500원 미만은 거의 잡음으로 간주

    # Normalize spaces
    for line in lines:
        l = " ".join((line or "").strip().split())
        if not l:
            continue

        low = l.lower()

        # Hard filters
        if phone_like.search(l) or bizno_like.search(l) or time_like.search(l):
            continue
        # date line often not an item
        if date_like.search(l) and len(re.sub(r"[^\d]", "", l)) >= 6 and len(l) < 30:
            continue

        # Ignore lines that are mostly digits/punct
        if len(re.sub(r"[\d\W_]+", "", l)) < 2:
            continue

        # -------------------------------------------------
        # Pattern C: "Name 30,000 1 30,000" (table-like)
        # -------------------------------------------------
        # Capture: name + (unit) + qty + total  OR  name + qty + total
        m = re.match(r"^(.{2,80}?)\s+(\d[\d,]*)\s+(\d{1,3})\s+(\d[\d,]*)\s*원?$", l)
        if not m:
            m = re.match(r"^(.{2,80}?)\s+(\d{1,3})\s+(\d[\d,]*)\s*원?$", l)

        if m:
            if len(m.groups()) == 4:
                name = (m.group(1) or "").strip()
                total = _parse_amount(m.group(4) or "")
            else:
                name = (m.group(1) or "").strip()
                total = _parse_amount(m.group(3) or "")

            if not name or total is None or total <= 0:
                continue
            if has_banned(name):
                continue
            if price_is_noise(total):
                continue

            # Avoid ultra-long “name”
            name = name[:200]
            out.append({"itemName": name, "price": total, "categoryTag": None})
            if len(out) >= limit:
                break
            continue

        # -------------------------------------------------
        # Pattern A/B/D: "Name 10,000" or "Name: 10,000"
        # -------------------------------------------------
        m2 = re.match(r"^(.{2,80}?)[\s:₩]+(\d[\d,]*)\s*원?\s*$", l)
        if m2:
            name = (m2.group(1) or "").strip()
            price = _parse_amount(m2.group(2) or "")

            if not name or price is None or price <= 0:
                continue
            if has_banned(name):
                continue
            if price_is_noise(price):
                continue

            # Another guard: if name is basically “고객번호” 같은 형태면 제외
            if re.search(r"(고객\s*번호|발행|승인|거래|No\.?)", name, re.IGNORECASE):
                continue

            out.append({"itemName": name[:200], "price": price, "categoryTag": None})
            if len(out) >= limit:
                break
            continue

    return out


def _address_hint(lines: List[str]) -> Optional[str]:
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if "주소" in l:
            return l[:120]
        # rough Korean address heuristic
        if any(tok in l for tok in ["로", "길", "동", "구", "시", "군", "읍", "면"]) and re.search(r"\d", l):
            if len(l) >= 8:
                return l[:120]
    return None


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
    **kwargs,
) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      webp_bytes: bytes (best-effort WEBP, may fallback to JPEG if WEBP unsupported)
      parsed: { visitDate, hospitalName, totalAmount, items }
      hints: { addressHint? }
    """
    if raw is None:
        raw = b""
    if len(raw) == 0:
        raise ValueError("empty file")

    img = _load_image(raw)
    img = _downscale_to_max_pixels(img, int(image_max_pixels))
    img = _resize_max_width(img, int(receipt_max_width))

    webp_bytes = _encode_webp(img, int(receipt_webp_quality))

    # OCR (best-effort)
    text = ""
    sema = _get_sema(int(ocr_max_concurrency))
    acquired = sema.acquire(timeout=float(ocr_sema_acquire_timeout_seconds or 0))
    try:
        if acquired:
            ocr_bytes = _encode_jpeg_for_ocr(img)
            text = _google_ocr_text(ocr_bytes, google_credentials, int(ocr_timeout_seconds))
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
        "items": _extract_items(lines, limit=50),
    }

    hints: Dict[str, Any] = {}
    ah = _address_hint(lines)
    if ah:
        hints["addressHint"] = ah

    return webp_bytes, parsed, hints
