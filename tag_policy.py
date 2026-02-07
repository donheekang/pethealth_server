# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
#
# ✅ Key upgrades (2026-02):
# - rabies: "rabies/rabbies/광견병" 뿐 아니라 OCR이 잘라낸 "ra", "rabb", "rabi"도 더 잘 잡게 보강
# - resolve_record_tags()에 ocr_text + Gemini(옵션) AI assist 추가 (kwargs로 들어와도 동작)
# - catalog 기반이 1차, 실패/저신뢰 때만 Gemini 보조로 tags/itemCategoryTags 채움

from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Tag catalog
# -----------------------------
TAG_CATALOG: List[Dict[str, Any]] = [
    # --- Exams ---
    {"code": "exam_xray", "group": "exam", "aliases": ["x-ray", "xray", "xr", "radiograph", "radiology", "엑스레이", "방사선", "x선", "x선촬영"]},
    {"code": "exam_ultrasound", "group": "exam", "aliases": ["ultrasound", "sono", "sonography", "us", "초음파", "복부초음파", "심장초음파", "심초음파"]},
    {"code": "exam_blood", "group": "exam", "aliases": ["cbc", "blood test", "chemistry", "biochem", "profile", "혈액", "혈액검사", "생화학", "전해질"]},
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

    # ✅ 핵심 보강:
    # - OCR이 Rabbies로 찍히는 케이스
    # - OCR이 잘라먹어 "ra", "rab", "rabi", "rabb" 정도만 남는 케이스
    {"code": "vaccine_rabies", "group": "vaccine", "aliases": ["rabies", "rabbies", "rabie", "rabi", "rabb", "rabis", "rab", "ra", "광견병", "광견"]},

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


# ---- Normalization/token rules
_alnum = re.compile(r"[0-9a-zA-Z가-힣]+")


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

    # ✅ rabies 초단축(ra/rab/rabi/rabb)은 사용자가 검색하거나 OCR이 찢어진 케이스에서 중요
    if tag.get("code") == "vaccine_rabies" and q_norm in ("ra", "rab", "rabi", "rabb", "rabis"):
        # 너무 낮으면 threshold 못넘음 → 확실히 잡히게
        return 175, {"why": [f"rabiesShortcut:{q_norm}"]}

    for alias in tag.get("aliases", []):
        a = str(alias or "").strip()
        if not a:
            continue
        a_norm = _normalize(a)
        if not a_norm:
            continue

        # 짧은 약어(us/ua/pi/l2/ra)는 contains 오탐이 크니,
        # token match + 정규화 완전일치만 허용
        if _is_short_ascii_token(a_norm):
            if a_norm == q_norm or a_norm in token_set:
                # ✅ ra는 위에서 이미 175 리턴하지만,
                # 혹시 다른 경로로 들어오면 최소한 여기서도 잡히게
                score = 165 if (tag.get("code") == "vaccine_rabies" and a_norm == "ra") else 160
                best = max(best, score)
                hit += 1
                strong = True
                why.append(f"shortEqOrToken:{a}")
            continue

        if a_norm == q_norm:
            best = max(best, 180)
            hit += 1
            strong = True
            why.append(f"eq:{a}")
        elif q_norm.startswith(a_norm) and len(a_norm) >= 4:
            # query가 code/alias를 길게 포함하는 케이스 (안전한 강매칭)
            best = max(best, 170)
            hit += 1
            strong = True
            why.append(f"queryStartsWithAlias:{a}")
        elif a_norm.startswith(q_norm) and len(q_norm) >= 3:
            # ✅ 사용자가 앞부분만 입력한 케이스: 'rabb' 같은 것 보강
            best = max(best, 150)
            hit += 1
            why.append(f"aliasStartsWithQuery:{a}")
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


def _build_record_query(items: List[Dict[str, Any]], hospital_name: Optional[str], ocr_text: Optional[str]) -> str:
    parts: List[str] = []
    if hospital_name:
        parts.append(str(hospital_name))
    for it in (items or [])[:120]:
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if nm:
            parts.append(nm)

    # ✅ ocr_text는 그대로 붙이면 노이즈도 많음 → 일부만, 그리고 길이 제한
    if isinstance(ocr_text, str) and ocr_text.strip():
        parts.append(ocr_text.strip()[:800])

    return " | ".join(parts)


# -----------------------------
# Gemini (optional AI assist)
# -----------------------------
def _env_bool(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _call_gemini_generate_content(
    *,
    api_key: str,
    model: str,
    prompt: str,
    timeout_seconds: int = 10,
) -> str:
    import urllib.request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
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


def _gemini_resolve_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str],
    ocr_text: Optional[str],
    api_key: str,
    model: str,
    timeout_seconds: int,
    max_tags: int,
) -> Optional[Dict[str, Any]]:
    api_key = (api_key or "").strip()
    model = (model or "gemini-2.5-flash").strip()
    if not api_key:
        return None

    allowed = [t["code"] for t in TAG_CATALOG if isinstance(t.get("code"), str)]

    # 아이템 이름만 주고, 모델이 "허용된 code" 중에서만 고르게 강제
    item_lines = []
    for idx, it in enumerate((items or [])[:80]):
        nm = str(it.get("itemName") or "").strip()
        if nm:
            item_lines.append(f"- [{idx}] {nm}")

    prompt = (
        "You are a veterinary receipt tag classifier.\n"
        "Return ONLY valid JSON.\n"
        "Allowed tag codes (choose only from this list):\n"
        f"{allowed}\n\n"
        "Input:\n"
        f"- hospitalName: {hospital_name or ''}\n"
        f"- items:\n{chr(10).join(item_lines) if item_lines else '(none)'}\n\n"
        f"- ocrTextSnippet:\n{(ocr_text or '')[:800]}\n\n"
        "Output JSON schema:\n"
        "{\n"
        f'  "tags": [string, ...]  (max {max_tags}),\n'
        '  "itemCategoryTags": [{"idx":int,"categoryTag":string}],\n'
        '  "reason": string\n'
        "}\n"
        "Rules:\n"
        "- tags must be in allowed tag codes.\n"
        "- itemCategoryTags.categoryTag must be allowed.\n"
        "- If rabies is mentioned or looks truncated like 'ra' in vaccine context, include vaccine_rabies.\n"
        "- Prefer specific tags over etc_other.\n"
    )

    try:
        out = _call_gemini_generate_content(api_key=api_key, model=model, prompt=prompt, timeout_seconds=timeout_seconds)
        j = _extract_json_from_model_text(out)
        if isinstance(j, dict):
            return j
        return None
    except Exception:
        return None


def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    ocr_text: Optional[str] = None,
    record_thresh: int = 125,
    item_thresh: int = 140,
    max_tags: int = 6,
    return_item_tags: bool = True,
    # ✅ AI assist (optional)
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
    query = _build_record_query(items or [], hospital_name, ocr_text)
    if not query.strip():
        return {"tags": [], "itemCategoryTags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}

    scored: List[Tuple[str, int, Dict[str, Any]]] = []
    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, query)
        if s > 0:
            scored.append((tag["code"], s, ev))

    scored.sort(key=lambda x: x[1], reverse=True)

    picked: List[str] = []
    evidence: Dict[str, Any] = {
        "policy": "catalog",
        "query": query[:800],
        "recordThresh": int(record_thresh),
        "itemThresh": int(item_thresh),
        "candidates": [],
        "geminiUsed": False,
    }

    for code, score, ev in scored[:20]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= int(record_thresh) and code not in picked:
            if code == "etc_other":
                continue
            picked.append(code)
        if len(picked) >= int(max_tags):
            break

    # 아무것도 못 잡으면 etc_other를 마지막에만
    if not picked:
        for code, score, _ in scored[:30]:
            if code == "etc_other" and score >= 90:
                picked.append("etc_other")
                break

    # item-level tags
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
                item_tags.append(
                    {
                        "idx": idx,
                        "itemName": nm,
                        "categoryTag": best_code,
                        "score": best_score,
                        **(best_ev or {}),
                    }
                )

    # -----------------------------
    # ✅ Gemini assist (optional)
    # - catalog가 비었거나 etc_other 뿐일 때
    # - 혹은 rabies 같은 핵심 태그가 놓칠 수 있는 저신뢰 케이스
    # -----------------------------
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))

    catalog_low_conf = (len(picked) == 0) or (picked == ["etc_other"])
    # rabies 관련 힌트가 있는데 태그가 없으면 보조 시도
    txt = (ocr_text or "")
    rabies_hint = ("rabies" in txt.lower()) or ("rabbies" in txt.lower()) or ("광견병" in txt) or (" ra " in f" {txt.lower()} ")

    if g_enabled and g_key.strip() and (catalog_low_conf or rabies_hint):
        gj = _gemini_resolve_tags(
            items=items or [],
            hospital_name=hospital_name,
            ocr_text=ocr_text,
            api_key=g_key,
            model=g_model,
            timeout_seconds=g_timeout,
            max_tags=int(max_tags),
        )
        if isinstance(gj, dict):
            evidence["geminiUsed"] = True
            evidence["geminiReason"] = str(gj.get("reason") or "")[:200]

            # merge tags
            gtags = gj.get("tags")
            if isinstance(gtags, list):
                for t in gtags:
                    s = str(t).strip()
                    if not s:
                        continue
                    if s == "etc_other":
                        continue
                    if s not in picked:
                        picked.append(s)
                    if len(picked) >= int(max_tags):
                        break

            # merge item tags (idx 기반)
            gitem = gj.get("itemCategoryTags")
            if isinstance(gitem, list):
                # item_tags에 idx 중복 있으면 catalog 우선, 없으면 gemini 추가
                existing_idx = {int(x.get("idx")) for x in item_tags if isinstance(x.get("idx"), int)}
                for r in gitem:
                    if not isinstance(r, dict):
                        continue
                    idx = r.get("idx")
                    ct = str(r.get("categoryTag") or "").strip()
                    if not isinstance(idx, int) or not ct or ct == "etc_other":
                        continue
                    if idx in existing_idx:
                        continue
                    nm = ""
                    try:
                        nm = str((items[idx] or {}).get("itemName") or "").strip()
                    except Exception:
                        nm = ""
                    if not nm:
                        continue
                    item_tags.append({"idx": idx, "itemName": nm, "categoryTag": ct, "score": 0, "why": ["gemini"]})

    return {"tags": picked, "itemCategoryTags": item_tags, "evidence": evidence}


