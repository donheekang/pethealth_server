# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
#
# Returns:
# {
#   "tags": [...],
#   "itemCategoryTags": [{"idx":0,"itemName":"...","categoryTag":"...","score":...,"why":[...]}],
#   "evidence": {...}
# }

from __future__ import annotations

import os
import re
import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Tag catalog
# -----------------------------
TAG_CATALOG: List[Dict[str, Any]] = [
    # --- Exams ---
    {"code": "exam_xray", "group": "exam", "aliases": ["x-ray", "xray", "xr", "radiograph", "radiology", "엑스레이", "방사선", "x선", "x선촬영"]},
    {"code": "exam_ultrasound", "group": "exam", "aliases": ["ultrasound", "sono", "sonography", "us", "초음파", "복부초음파", "심장초음파", "심초음파"]},
    {"code": "exam_blood", "group": "exam", "aliases": ["cbc", "blood test", "chemistry", "biochem", "profile", "혈액", "혈액검사", "생화학", "전해질", "검사"]},
    {"code": "exam_lab_panel", "group": "exam", "aliases": ["lab panel", "screening", "health check", "종합검사", "종합검진", "패널검사"]},
    {"code": "exam_urine", "group": "exam", "aliases": ["urinalysis", "ua", "urine test", "요검사", "소변검사"]},
    {"code": "exam_fecal", "group": "exam", "aliases": ["fecal", "stool test", "대변검사", "분변검사", "배변검사"]},
    {"code": "exam_fecal_pcr", "group": "exam", "aliases": ["fecal pcr", "stool pcr", "gi pcr", "panel pcr", "대변pcr", "대변 pcr", "분변 pcr", "배설물 pcr"]},
    {"code": "exam_sdma", "group": "exam", "aliases": ["sdma", "symmetrical dimethylarginine", "idexx sdma", "renal sdma", "신장마커", "신장검사"]},
    {"code": "exam_probnp", "group": "exam", "aliases": ["probnp", "pro bnp", "pro-bnp", "ntprobnp", "nt-probnp", "bnp", "cardiopet", "심장마커", "프로비엔피"]},
    {"code": "exam_fructosamine", "group": "exam", "aliases": ["fructosamine", "fru", "glycated albumin", "ga", "프럭토사민", "당화알부민"]},
    {"code": "exam_glucose_curve", "group": "exam", "aliases": ["glucose curve", "blood glucose curve", "bg curve", "혈당곡선", "혈당커브", "혈당 커브", "연속혈당"]},
    {"code": "exam_blood_gas", "group": "exam", "aliases": ["blood gas", "bga", "bgas", "i-stat", "istat", "혈액가스", "가스분석"]},
    {"code": "exam_allergy", "group": "exam", "aliases": ["allergy test", "ige", "atopy", "알러지검사", "알레르기검사", "알러지", "알레르기"]},
    {"code": "exam_heart", "group": "exam", "aliases": ["ecg", "ekg", "echo", "cardiac", "heart", "심전도", "심초음파", "심장초음파", "심장검사"]},
    {"code": "exam_eye", "group": "exam", "aliases": ["schirmer", "fluorescein", "iop", "ophthalmic exam", "안압", "형광염색", "안과검사", "안과", "눈검사"]},
    {"code": "exam_skin", "group": "exam", "aliases": ["skin scraping", "cytology", "fungal test", "malassezia", "피부스크래핑", "피부검사", "진균", "곰팡이", "말라세지아"]},

    # --- Vaccines / Preventives ---
    {"code": "vaccine_comprehensive", "group": "vaccine", "aliases": ["dhpp", "dhppi", "dhlpp", "5-in-1", "6-in-1", "fvrcp", "combo vaccine", "종합백신", "혼합백신", "5종", "6종"]},
    {"code": "vaccine_rabies", "group": "vaccine", "aliases": ["rabies", "rabbies", "광견병", "광견"]},
    {"code": "vaccine_kennel", "group": "vaccine", "aliases": ["kennel cough", "bordetella", "켄넬코프", "기관지염백신", "보르데텔라"]},
    {"code": "vaccine_corona", "group": "vaccine", "aliases": ["corona", "coronavirus", "corona enteritis", "코로나", "코로나장염"]},
    {"code": "vaccine_lepto", "group": "vaccine", "aliases": ["lepto", "leptospirosis", "leptospira", "lepto2", "lepto4", "l2", "l4", "렙토", "렙토2", "렙토4"]},
    {"code": "vaccine_parainfluenza", "group": "vaccine", "aliases": ["parainfluenza", "cpiv", "cpi", "pi", "파라인플루엔자", "파라인", "파라"]},
    {"code": "vaccine_fip", "group": "vaccine", "aliases": ["fip", "primucell", "feline infectious peritonitis", "전염성복막염", "복막염"]},

    {"code": "prevent_heartworm", "group": "preventive_med", "aliases": ["heartworm", "hw", "dirofilaria", "heartgard", "심장사상충", "하트가드", "넥스가드스펙트라", "simparica trio", "revolution"]},
    {"code": "prevent_external", "group": "preventive_med", "aliases": ["flea", "tick", "bravecto", "nexgard", "frontline", "revolution", "벼룩", "진드기", "외부기생충"]},
    {"code": "prevent_deworming", "group": "preventive_med", "aliases": ["deworm", "deworming", "drontal", "milbemax", "fenbendazole", "panacur", "구충", "구충제", "내부기생충"]},

    # --- Medicines ---
    {"code": "medicine_antibiotic", "group": "medicine", "aliases": ["antibiotic", "abx", "amoxicillin", "clavamox", "augmentin", "cephalexin", "convenia", "doxycycline", "metronidazole", "baytril", "항생제"]},
    {"code": "medicine_anti_inflammatory", "group": "medicine", "aliases": ["nsaid", "anti-inflammatory", "meloxicam", "metacam", "carprofen", "rimadyl", "onsior", "galliprant", "소염", "소염제"]},
    {"code": "medicine_painkiller", "group": "medicine", "aliases": ["analgesic", "tramadol", "gabapentin", "buprenorphine", "진통", "진통제"]},
    {"code": "medicine_steroid", "group": "medicine", "aliases": ["steroid", "prednisone", "prednisolone", "dexamethasone", "스테로이드"]},
    {"code": "medicine_gi", "group": "medicine", "aliases": ["famotidine", "pepcid", "omeprazole", "sucralfate", "cerenia", "ondansetron", "reglan", "위장약", "구토", "설사", "장염"]},
    {"code": "medicine_eye", "group": "medicine", "aliases": ["eye drop", "ophthalmic", "tobramycin", "ofloxacin", "cyclosporine", "안약", "점안", "결막염", "각막"]},
    {"code": "medicine_ear", "group": "medicine", "aliases": ["ear drop", "otic", "otitis", "otomax", "surolan", "posatex", "easotic", "귀약", "이염", "외이염"]},
    {"code": "medicine_skin", "group": "medicine", "aliases": ["dermatitis", "chlorhexidine", "ketoconazole", "miconazole", "피부약", "피부염"]},
    {"code": "medicine_allergy", "group": "medicine", "aliases": ["apoquel", "cytopoint", "cetirizine", "zyrtec", "benadryl", "알러지", "알레르기", "가려움"]},

    # --- Care / Procedures / Goods ---
    {"code": "care_injection", "group": "checkup", "aliases": ["inj", "injection", "shot", "sc", "im", "iv", "주사", "주사제", "피하주사", "근육주사", "정맥주사", "주사료"]},
    {"code": "care_procedure_fee", "group": "checkup", "aliases": ["procedure fee", "treatment fee", "handling fee", "처치료", "시술료", "처치비", "시술비", "처치", "시술"]},
    {"code": "care_dressing", "group": "checkup", "aliases": ["dressing", "bandage", "gauze", "wrap", "disinfection", "드레싱", "붕대", "거즈", "소독", "세척", "상처처치"]},
    {"code": "care_e_collar", "group": "etc", "aliases": ["e-collar", "ecollar", "cone", "elizabethan collar", "넥카라", "엘리자베스카라", "보호카라"]},
    {"code": "care_prescription_diet", "group": "etc", "aliases": ["prescription diet", "rx diet", "therapeutic diet", "처방식", "처방사료", "병원사료", "hill's", "hills", "royal canin", "k/d", "c/d", "i/d", "z/d"]},

    # --- Surgery / Dental / Ortho / General ---
    {"code": "surgery_general", "group": "surgery", "aliases": ["surgery", "operation", "spay", "neuter", "castration", "수술", "중성화", "봉합", "마취"]},
    {"code": "dental_scaling", "group": "dental", "aliases": ["scaling", "dental cleaning", "tartar", "스케일링", "치석"]},
    {"code": "dental_extraction", "group": "dental", "aliases": ["extraction", "dental extraction", "발치"]},
    {"code": "ortho_patella", "group": "orthopedic", "aliases": ["mpl", "lpl", "patella", "patellar luxation", "슬개골탈구", "슬탈", "파행"]},
    {"code": "ortho_arthritis", "group": "orthopedic", "aliases": ["arthritis", "oa", "osteoarthritis", "관절염", "퇴행성관절"]},

    {"code": "checkup_general", "group": "checkup", "aliases": ["checkup", "consult", "opd", "진료", "상담", "초진", "재진", "진찰", "진료비"]},
    {"code": "grooming_basic", "group": "grooming", "aliases": ["grooming", "bath", "trim", "미용", "목욕", "클리핑"]},

    {"code": "etc_other", "group": "etc", "aliases": ["기타", "etc", "other"]},
]


# -----------------------------
# Normalization / tokenization
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

        # 짧은 약어(us/ua/pi/l2)는 contains 오탐이 커서 eq/token만
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

    # OCR 원문도 조금만 섞기 (과하면 오탐↑)
    if isinstance(ocr_text, str) and ocr_text.strip():
        parts.append((ocr_text or "")[:1200])

    return " | ".join(parts)


# -----------------------------
# ✅ Rabies(광견병) 보강: "ra"만 있어도 vaccine_rabies 강제
# -----------------------------
_RABIES_CODE = "vaccine_rabies"
_RABIES_PARTIAL = {"ra", "rab", "rabi", "rabb"}
_VACCINE_CONTEXT = {"백신", "접종", "예방", "예방접종", "vaccine", "vacc", "immun", "shot", "inj", "주사"}


def _env_bool(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _contains_strong_rabies(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return ("rabies" in low) or ("rabbies" in low) or ("광견병" in text) or ("광견" in text)


def _token_norms(text: str) -> List[str]:
    toks = []
    for t in _tokenize(text or ""):
        n = _normalize(t)
        if n:
            toks.append(n)
    return toks


def _has_any_substr(text: str, keys: set) -> bool:
    t = text or ""
    low = t.lower()
    for k in keys:
        if k in t or k in low:
            return True
    return False


def _find_partial_rabies_item_idxs(items: List[Dict[str, Any]]) -> Tuple[List[int], List[str]]:
    idxs: List[int] = []
    toks: List[str] = []
    for i, it in enumerate((items or [])[:200]):
        nm = (it.get("itemName") or "").strip()
        if not nm:
            continue

        nm_norm = _normalize(nm)  # "r a" -> "ra"
        hit = []
        if nm_norm in _RABIES_PARTIAL:
            hit.append(nm_norm)

        tn = set(_token_norms(nm))
        hit += sorted(list(tn.intersection(_RABIES_PARTIAL)))
        hit = sorted(list(set(hit)))
        if hit:
            idxs.append(i)
            toks.extend(hit)

    return idxs, sorted(list(set(toks)))


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


def _call_gemini_bool(prompt: str, api_key: str, model: str, timeout_seconds: int) -> Optional[bool]:
    """
    (선택) 'ra' 같은 애매한 경우만 확인용.
    ✅ key names corrected: inlineData/mimeType
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 128},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=float(timeout_seconds or 8)) as resp:
            data = resp.read().decode("utf-8", errors="replace")

        j = json.loads(data)
        txt = (
            (j.get("candidates") or [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        ).strip()

        # try JSON
        txt2 = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt2 = re.sub(r"\s*```$", "", txt2)
        i = txt2.find("{")
        k = txt2.rfind("}")
        if i >= 0 and k > i:
            obj = json.loads(txt2[i : k + 1])
            v = obj.get("rabies")
            if isinstance(v, bool):
                return v

        low = txt.lower()
        if "true" in low or "yes" in low:
            return True
        if "false" in low or "no" in low:
            return False
        return None
    except Exception:
        return None


def _detect_rabies_force(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str],
    ocr_text: Optional[str],
    gemini_enabled: Optional[bool] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model_name: Optional[str] = None,
    gemini_timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    combined = " | ".join([
        (hospital_name or ""),
        (ocr_text or ""),
        " ".join([(it.get("itemName") or "") for it in (items or [])[:120]]),
    ])

    # strong
    if _contains_strong_rabies(combined):
        return {
            "force": True,
            "confidence": 1.0,
            "reason": "strong_text",
            "itemIdxs": [],
            "partialTokens": [],
            "textLines": [],
            "hasVaccineContext": _has_any_substr(combined, _VACCINE_CONTEXT),
        }

    idxs, partials_item = _find_partial_rabies_item_idxs(items)
    found_text, partials_text, lines_text = _find_partial_rabies_in_text(ocr_text)

    union_tokens = sorted(list(set((partials_item or []) + (partials_text or []))))
    has_ctx = _has_any_substr(combined, _VACCINE_CONTEXT)

    if not union_tokens:
        return {
            "force": False,
            "confidence": 0.0,
            "reason": "no_signal",
            "itemIdxs": [],
            "partialTokens": [],
            "textLines": [],
            "hasVaccineContext": has_ctx,
        }

    # ✅ 요구사항: ra만 있어도 rabies
    only_ra = (union_tokens == ["ra"])
    if only_ra and not has_ctx:
        # optional gemini check (but STILL force true even if no)
        ge = bool(gemini_enabled) if isinstance(gemini_enabled, bool) else _env_bool("GEMINI_ENABLED")
        gk = (gemini_api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        gm = (gemini_model_name or os.getenv("GEMINI_MODEL_NAME") or "gemini-2.5-flash").strip()
        gt = int(gemini_timeout_seconds or int(os.getenv("GEMINI_TIMEOUT_SECONDS") or "8"))

        if ge and gk:
            item_lines = "\n".join([f"- {it.get('itemName')} / price={it.get('price')}" for it in (items or [])[:20]])
            prompt = (
                "Return ONLY JSON.\n"
                "Decide whether this veterinary receipt indicates a rabies vaccination.\n"
                'JSON: {"rabies": true/false}\n'
                "Notes:\n"
                "- OCR sometimes outputs only 'ra' for rabies.\n\n"
                f"hospital: {hospital_name or ''}\n"
                f"ocr_text: {(ocr_text or '')[:1200]}\n"
                f"items:\n{item_lines}\n"
            )
            b = _call_gemini_bool(prompt, api_key=gk, model=gm, timeout_seconds=gt)
            if b is True:
                return {"force": True, "confidence": 0.9, "reason": "partial_ra_gemini_yes",
                        "itemIdxs": idxs, "partialTokens": union_tokens, "textLines": lines_text, "hasVaccineContext": has_ctx}
            if b is False:
                return {"force": True, "confidence": 0.55, "reason": "partial_ra_gemini_no_fallback_true",
                        "itemIdxs": idxs, "partialTokens": union_tokens, "textLines": lines_text, "hasVaccineContext": has_ctx}

        return {"force": True, "confidence": 0.6, "reason": "partial_ra_no_ctx_fallback_true",
                "itemIdxs": idxs, "partialTokens": union_tokens, "textLines": lines_text, "hasVaccineContext": has_ctx}

    conf = 0.9 if has_ctx else 0.8
    if only_ra and has_ctx:
        conf = 0.85

    return {"force": True, "confidence": conf, "reason": "partial_tokens" if not only_ra else "partial_ra_with_ctx",
            "itemIdxs": idxs, "partialTokens": union_tokens, "textLines": lines_text, "hasVaccineContext": has_ctx}


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

    rab = _detect_rabies_force(
        items=items or [],
        hospital_name=hospital_name,
        ocr_text=ocr_text,
        gemini_enabled=gemini_enabled,
        gemini_api_key=gemini_api_key,
        gemini_model_name=gemini_model_name,
        gemini_timeout_seconds=gemini_timeout_seconds,
    )

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

    for code, score, ev in scored[:30]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= int(record_thresh) and code not in picked:
            if code == "etc_other":
                continue
            picked.append(code)
        if len(picked) >= int(max_tags):
            break

    if not picked:
        # weak fallback
        for code, score, _ in scored[:30]:
            if code == "etc_other" and score >= 90:
                picked.append("etc_other")
                break

    # ✅ Rabies 강제 적용 (record tags)
    if rab.get("force") is True:
        if _RABIES_CODE not in picked:
            picked.insert(0, _RABIES_CODE)
        picked = picked[: int(max_tags)]

    item_tags: List[Dict[str, Any]] = []
    if return_item_tags:
        # normal item tagging (idx 기반)
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

        # ✅ Rabies item override (partial token이 잡힌 idx들)
        if rab.get("force") is True:
            idxs = rab.get("itemIdxs") or []
            for idx in idxs:
                if not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= len(items or []):
                    continue
                nm = ((items[idx] or {}).get("itemName") or "").strip()
                if not nm:
                    continue
                item_tags.append({"idx": idx, "itemName": nm, "categoryTag": _RABIES_CODE, "score": 999, "why": [f"forced:{rab.get('reason')}"]})

    return {"tags": picked, "itemCategoryTags": item_tags, "evidence": evidence}


