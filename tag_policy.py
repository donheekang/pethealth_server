# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
# Public API: resolve_record_tags(...)

from __future__ import annotations

import re
import os
import json
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Tag catalog
# -----------------------------
TAG_CATALOG: List[Dict[str, Any]] = [
    # --- Exams ---
    {"code": "exam_xray", "group": "exam", "aliases": ["x-ray","xray","xr","radiograph","radiology","엑스레이","방사선","x선","x선촬영"]},
    {"code": "exam_ultrasound", "group": "exam", "aliases": ["ultrasound","sono","sonography","us","초음파","복부초음파","심장초음파","심초음파"]},
    {"code": "exam_blood", "group": "exam", "aliases": ["cbc","blood test","chemistry","biochem","profile","혈액","혈액검사","생화학","전해질","검사"]},
    {"code": "exam_lab_panel", "group": "exam", "aliases": ["lab panel","screening","health check","종합검사","종합검진","패널검사"]},
    {"code": "exam_urine", "group": "exam", "aliases": ["urinalysis","ua","urine test","요검사","소변검사"]},
    {"code": "exam_fecal", "group": "exam", "aliases": ["fecal","stool test","대변검사","분변검사","배변검사"]},
    {"code": "exam_fecal_pcr", "group": "exam", "aliases": ["fecal pcr","stool pcr","gi pcr","panel pcr","대변pcr","대변 pcr","분변 pcr","배설물 pcr"]},

    {"code": "exam_allergy", "group": "exam", "aliases": ["allergy test","ige","atopy","알러지검사","알레르기검사","알러지","알레르기"]},
    {"code": "exam_heart", "group": "exam", "aliases": ["ecg","ekg","echo","cardiac","heart","심전도","심초음파","심장초음파","심장검사"]},
    {"code": "exam_eye", "group": "exam", "aliases": ["schirmer","fluorescein","iop","ophthalmic exam","안압","형광염색","안과검사","안과","눈검사"]},
    {"code": "exam_skin", "group": "exam", "aliases": ["skin scraping","cytology","fungal test","malassezia","피부스크래핑","피부검사","진균","곰팡이","말라세지아"]},

    # --- Vaccines / Preventives ---
    {"code": "vaccine_comprehensive", "group": "vaccine", "aliases": ["dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine","종합백신","혼합백신","5종","6종"]},
    {"code": "vaccine_rabies", "group": "vaccine", "aliases": ["rabies","rabbies","광견병","광견","ra","rab","rabi","rabb"]},  # ✅ partial도 alias로 포함
    {"code": "vaccine_kennel", "group": "vaccine", "aliases": ["kennel cough","bordetella","켄넬코프","기관지염백신","보르데텔라"]},
    {"code": "vaccine_corona", "group": "vaccine", "aliases": ["corona","coronavirus","corona enteritis","코로나","코로나장염"]},
    {"code": "vaccine_lepto", "group": "vaccine", "aliases": ["lepto","leptospirosis","leptospira","lepto2","lepto4","l2","l4","렙토","렙토2","렙토4"]},
    {"code": "vaccine_parainfluenza", "group": "vaccine", "aliases": ["parainfluenza","cpiv","cpi","pi","파라인플루엔자","파라인","파라"]},
    {"code": "vaccine_fip", "group": "vaccine", "aliases": ["fip","primucell","feline infectious peritonitis","전염성복막염","복막염"]},

    {"code": "prevent_heartworm", "group": "preventive_med", "aliases": ["heartworm","hw","dirofilaria","heartgard","심장사상충","하트가드","넥스가드스펙트라","simparica trio","revolution"]},
    {"code": "prevent_external", "group": "preventive_med", "aliases": ["flea","tick","bravecto","nexgard","frontline","revolution","벼룩","진드기","외부기생충"]},
    {"code": "prevent_deworming", "group": "preventive_med", "aliases": ["deworm","deworming","drontal","milbemax","fenbendazole","panacur","구충","구충제","내부기생충"]},

    # --- Medicines ---
    {"code": "medicine_antibiotic", "group": "medicine", "aliases": ["antibiotic","abx","amoxicillin","clavamox","augmentin","cephalexin","convenia","doxycycline","metronidazole","baytril","항생제"]},
    {"code": "medicine_anti_inflammatory", "group": "medicine", "aliases": ["nsaid","anti-inflammatory","meloxicam","metacam","carprofen","rimadyl","onsior","galliprant","소염","소염제"]},
    {"code": "medicine_painkiller", "group": "medicine", "aliases": ["analgesic","tramadol","gabapentin","buprenorphine","진통","진통제"]},
    {"code": "medicine_steroid", "group": "medicine", "aliases": ["steroid","prednisone","prednisolone","dexamethasone","스테로이드"]},
    {"code": "medicine_gi", "group": "medicine", "aliases": ["famotidine","pepcid","omeprazole","sucralfate","cerenia","ondansetron","reglan","위장약","구토","설사","장염"]},
    {"code": "medicine_eye", "group": "medicine", "aliases": ["eye drop","ophthalmic","tobramycin","ofloxacin","cyclosporine","안약","점안","결막염","각막"]},
    {"code": "medicine_ear", "group": "medicine", "aliases": ["ear drop","otic","otitis","otomax","surolan","posatex","easotic","귀약","이염","외이염"]},
    {"code": "medicine_skin", "group": "medicine", "aliases": ["dermatitis","chlorhexidine","ketoconazole","miconazole","피부약","피부염"]},
    {"code": "medicine_allergy", "group": "medicine", "aliases": ["apoquel","cytopoint","cetirizine","zyrtec","benadryl","알러지","알레르기","가려움"]},

    # --- Care / procedures ---
    {"code": "care_injection", "group": "checkup", "aliases": ["inj","injection","shot","sc","im","iv","주사","주사제","피하주사","근육주사","정맥주사","주사료"]},
    {"code": "care_procedure_fee", "group": "checkup", "aliases": ["procedure fee","treatment fee","handling fee","처치료","시술료","처치비","시술비","처치","시술"]},
    {"code": "care_dressing", "group": "checkup", "aliases": ["dressing","bandage","gauze","wrap","disinfection","드레싱","붕대","거즈","소독","세척","상처처치"]},

    {"code": "checkup_general", "group": "checkup", "aliases": ["checkup","consult","opd","진료","상담","초진","재진","진찰","진료비"]},
    {"code": "grooming_basic", "group": "grooming", "aliases": ["grooming","bath","trim","미용","목욕","클리핑"]},

    {"code": "etc_other", "group": "etc", "aliases": ["기타","etc","other"]},
]

# -----------------------------
# Normalization/token rules
# -----------------------------
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

        # ✅ 짧은 약어(us/ua/pi/l2 등)는 contains 오탐이 크니 token/eq만 허용
        if _is_short_ascii_token(a_norm):
            if a_norm == q_norm or a_norm in token_set:
                best = max(best, 160)
                hit += 1
                strong = True
                why.append(f"shortEqOrToken:{a}")
            continue

        if a_norm == q_norm:
            best = max(best, 180)
            hit += 1
            strong = True
            why.append(f"eq:{a}")
        elif q_norm.find(a_norm) >= 0:
            s = 120 + min(60, len(a_norm) * 2)
            best = max(best, s)
            hit += 1
            strong = True
            why.append(f"inQuery:{a}")
        elif a_norm.find(q_norm) >= 0:
            s = 90 + min(40, len(q_norm) * 2)
            best = max(best, s)
            hit += 1
            why.append(f"queryInAlias:{a}")

    if hit >= 2:
        best += min(35, hit * (8 if strong else 5))
        why.append(f"bonus:{hit}")

    return best, {"why": why[:8]}


def _build_record_query(items: List[Dict[str, Any]], hospital_name: Optional[str], ocr_text: Optional[str] = None) -> str:
    parts: List[str] = []
    if hospital_name:
        parts.append(str(hospital_name))
    for it in (items or [])[:120]:
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if nm:
            parts.append(nm)
    # OCR 원문 일부도 보강
    if isinstance(ocr_text, str) and ocr_text.strip():
        parts.append((ocr_text or "")[:1200])
    return " | ".join(parts)


# -----------------------------
# ✅ Rabies force tagging
# - 요구사항: "ra"만 있어도 rabies로 태깅
# -----------------------------
_RABIES_CODE = "vaccine_rabies"
_RABIES_PARTIAL = {"ra", "rab", "rabi", "rabb"}
_VACCINE_CONTEXT = {"백신", "접종", "예방", "예방접종", "vaccine", "vacc", "immun", "shot", "inj", "주사"}


def _env_bool(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _has_any_substr(text: str, keys: set) -> bool:
    t = text or ""
    low = t.lower()
    for k in keys:
        if k in t or k in low:
            return True
    return False


def _token_norms(text: str) -> List[str]:
    return [_normalize(t) for t in _tokenize(text or "") if _normalize(t)]


def _contains_strong_rabies(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if ("rabies" in low) or ("rabbies" in low):
        return True
    if ("광견병" in text) or ("광견" in text):
        return True
    return False


def _find_partial_rabies_in_text(ocr_text: Optional[str]) -> Tuple[bool, List[str], List[str]]:
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        return False, [], []

    toks = set(_token_norms(ocr_text))
    hit_tokens = sorted(list(toks.intersection(_RABIES_PARTIAL)))

    matched_lines: List[str] = []
    for ln in (ocr_text or "").splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        n = _normalize(s)
        if n in _RABIES_PARTIAL:
            matched_lines.append(s[:120])

    found = bool(hit_tokens or matched_lines)
    return found, hit_tokens, matched_lines[:8]


def _detect_rabies_force(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str],
    ocr_text: Optional[str],
) -> Dict[str, Any]:
    combined = " | ".join([
        (hospital_name or ""),
        (ocr_text or ""),
        " ".join([(it.get("itemName") or "") for it in (items or [])[:120]]),
    ])

    if _contains_strong_rabies(combined):
        return {"force": True, "confidence": 1.0, "reason": "strong_text"}

    # partial token check (items + ocr_text)
    item_tokens = set()
    for it in (items or [])[:200]:
        nm = (it.get("itemName") or "").strip()
        if not nm:
            continue
        nn = _normalize(nm)  # "r a" -> "ra"
        if nn in _RABIES_PARTIAL:
            item_tokens.add(nn)
        for t in _token_norms(nm):
            if t in _RABIES_PARTIAL:
                item_tokens.add(t)

    found_text, text_tokens, _lines = _find_partial_rabies_in_text(ocr_text)
    union = sorted(list(set(list(item_tokens) + list(text_tokens))))

    if not union:
        return {"force": False, "confidence": 0.0, "reason": "no_signal"}

    # ✅ ra만 있어도 무조건 True
    has_ctx = _has_any_substr(combined, _VACCINE_CONTEXT)
    only_ra = (union == ["ra"])

    if only_ra and not has_ctx:
        return {"force": True, "confidence": 0.6, "reason": "partial_ra_no_ctx_force_true"}
    return {"force": True, "confidence": 0.85 if has_ctx else 0.75, "reason": "partial_tokens"}


# -----------------------------
# Main API
# -----------------------------
def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    record_thresh: int = 125,
    item_thresh: int = 140,
    max_tags: int = 6,
    return_item_tags: bool = True,
    ocr_text: Optional[str] = None,
    # (선택) gemini 인자들 - 지금은 사용 안 해도 됨 (main에서 넘겨도 무시 가능)
    gemini_enabled: Optional[bool] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model_name: Optional[str] = None,
    gemini_timeout_seconds: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "tags": [...],
        "itemCategoryTags": [{"idx":0,"itemName":"...","categoryTag":"...","score":...,"why":[...]}],
        "evidence": {...}
      }
    """
    query = _build_record_query(items or [], hospital_name, ocr_text=ocr_text)
    if not query.strip():
        return {"tags": [], "itemCategoryTags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}

    rab = _detect_rabies_force(items=items or [], hospital_name=hospital_name, ocr_text=ocr_text)

    scored: List[Tuple[str, int, Dict[str, Any]]] = []
    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, query)
        if s > 0:
            scored.append((tag["code"], s, ev))

    scored.sort(key=lambda x: x[1], reverse=True)

    picked: List[str] = []
    evidence: Dict[str, Any] = {
        "policy": "catalog",
        "query": query[:600],
        "recordThresh": int(record_thresh),
        "itemThresh": int(item_thresh),
        "candidates": [],
        "rabiesForce": rab,
    }

    for code, score, ev in scored[:20]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= int(record_thresh) and code not in picked:
            if code == "etc_other":
                continue
            picked.append(code)
        if len(picked) >= int(max_tags):
            break

    if not picked:
        for code, score, _ in scored[:30]:
            if code == "etc_other" and score >= 90:
                picked.append("etc_other")
                break

    # ✅ Rabies 강제 적용 (record tags)
    if rab.get("force") is True and _RABIES_CODE not in picked:
        picked.insert(0, _RABIES_CODE)
        picked = picked[: int(max_tags)]

    item_tags: List[Dict[str, Any]] = []
    if return_item_tags:
        for idx, it in enumerate((items or [])[:200]):
            nm = (it.get("itemName") or "").strip()
            if not nm:
                continue
            best_code: Optional[str] = None
            best_score: int = 0
            best_ev: Dict[str, Any] = {}
            for tag in TAG_CATALOG:
                s, ev = _match_score(tag, nm)
                if s > best_score:
                    best_score = s
                    best_code = tag["code"]
                    best_ev = ev or {}
            if best_code and best_score >= int(item_thresh):
                if best_code == "etc_other":
                    continue
                item_tags.append({"idx": idx, "itemName": nm, "categoryTag": best_code, "score": best_score, **(best_ev or {})})

        # Rabies 강제 item override (ra 포함)
        if rab.get("force") is True:
            for idx, it in enumerate((items or [])[:200]):
                nm = (it.get("itemName") or "").strip()
                nn = _normalize(nm)
                if _contains_strong_rabies(nm) or (nn in _RABIES_PARTIAL):
                    item_tags.append({"idx": idx, "itemName": nm, "categoryTag": _RABIES_CODE, "score": 999, "why": [f"forced:{rab.get('reason')}"]})

    return {"tags": picked, "itemCategoryTags": item_tags, "evidence": evidence}


