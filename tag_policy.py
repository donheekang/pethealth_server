# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Tag catalog
# code == ReceiptTag.rawValue
# aliases == ReceiptTag.hospitalAliases (+ keywords)
# -----------------------------
TAG_CATALOG: List[Dict[str, Any]] = [
    # --- Exams ---
    {"code": "exam_xray", "group": "exam", "aliases": ["x-ray", "xray", "xr", "radiograph", "radiology", "엑스레이", "방사선", "x선", "x선촬영"]},
    {"code": "exam_ultrasound", "group": "exam", "aliases": ["ultrasound", "sono", "sonography", "us", "초음파", "복부초음파", "심장초음파"]},
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

    # ✅ iOS에 있는데 catalog에 빠지면 절대 못 잡음 (필수)
    {"code": "exam_allergy", "group": "exam", "aliases": ["allergy test", "ige", "atopy", "알러지검사", "알레르기검사", "알러지", "알레르기"]},
    {"code": "exam_heart", "group": "exam", "aliases": ["ecg", "ekg", "echo", "cardiac", "heart", "심전도", "심초음파", "심장초음파", "심장검사"]},
    {"code": "exam_eye", "group": "exam", "aliases": ["schirmer", "fluorescein", "iop", "ophthalmic exam", "안압", "형광염색", "안과검사", "안과"]},
    {"code": "exam_skin", "group": "exam", "aliases": ["skin scraping", "cytology", "fungal test", "malassezia", "피부스크래핑", "피부검사", "진균", "곰팡이", "말라세지아"]},

    # --- Vaccines / Preventives ---
    {"code": "vaccine_comprehensive", "group": "vaccine", "aliases": ["dhpp", "dhppi", "dhlpp", "5-in-1", "6-in-1", "fvrcp", "combo vaccine", "종합백신", "혼합백신", "5종", "6종"]},
    {"code": "vaccine_rabies", "group": "vaccine", "aliases": ["rabies","rabbies", "광견병", "광견"]},
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
    {"code": "ortho_patella", "group": "orthopedic", "aliases": ["pl", "mpl", "lpl", "patella", "patellar luxation", "슬개골탈구", "슬탈", "파행"]},
    {"code": "ortho_arthritis", "group": "orthopedic", "aliases": ["arthritis", "oa", "osteoarthritis", "관절염", "퇴행성관절"]},
    {"code": "checkup_general", "group": "checkup", "aliases": ["checkup", "consult", "opd", "진료", "상담", "초진", "재진", "진찰"]},
    {"code": "grooming_basic", "group": "grooming", "aliases": ["grooming", "bath", "trim", "미용", "목욕", "클리핑"]},

    # --- Fallback ---
    {"code": "etc_other", "group": "etc", "aliases": ["기타", "etc", "other"]},
]

_ALLOWED_CODES = {t["code"] for t in TAG_CATALOG}

# ---- Normalization/token rules (similar to iOS) ----
_ALNUM_KO_RE = re.compile(r"[0-9a-zA-Z가-힣]+")  # tokenization


def _normalize(s: str) -> str:
    s = (s or "").lower()
    # keep alnum + korean
    return "".join(ch for ch in s if ch.isalnum() or ("가" <= ch <= "힣"))


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[0-9a-zA-Z가-힣]+", s or "")


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

        # ✅ "U/S" 같은 케이스 대응:
        # - q_norm 자체가 "us"가 되는 경우가 많으니, short token이라도 q_norm==a_norm은 인정해야 함
        if _is_short_ascii_token(a_norm):
            if (a_norm == q_norm) or (a_norm in token_set):
                best = max(best, 165)  # 짧은 약어는 토큰일치면 강하게
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


def _pick_record_tags(
    query: str,
    *,
    record_thresh: int,
    max_tags: int,
) -> Tuple[List[str], Dict[str, Any]]:
    scored: List[Tuple[str, int, Dict[str, Any]]] = []
    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, query)
        if s > 0:
            scored.append((tag["code"], s, ev))

    scored.sort(key=lambda x: x[1], reverse=True)

    picked: List[str] = []
    evidence: Dict[str, Any] = {
        "policy": "catalog",
        "query": query[:1200],
        "record_thresh": record_thresh,
        "candidates": [],
    }

    for code, score, ev in scored[:20]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= record_thresh and code not in picked:
            picked.append(code)
        if len(picked) >= max_tags:
            break

    return picked, evidence


def _pick_item_tags(
    items: List[Dict[str, Any]],
    *,
    item_thresh: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(items or []):
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
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
                best_ev = ev

        if best_code and best_score >= item_thresh:
            out.append(
                {
                    "idx": idx,
                    "itemName": nm[:120],
                    "categoryTag": best_code,
                    "score": best_score,
                    **(best_ev or {}),
                }
            )
    return out


def _truthy_env(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _gemini_fallback(
    *,
    hospital_name: Optional[str],
    items: List[Dict[str, Any]],
    receipt_text: Optional[str],
    timeout_s: int = 12,
) -> Optional[Dict[str, Any]]:
    """
    Gemini 보조:
    - catalog로 recordTags가 비거나 너무 약할 때만 사용 권장
    - 실패/에러 시 None (서버 절대 안 죽게)
    """
    if not _truthy_env("GEMINI_ENABLED"):
        return None
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    model = (os.getenv("GEMINI_MODEL_NAME") or "gemini-1.5-flash").strip()
    if not model:
        model = "gemini-1.5-flash"

    # 너무 길게 보내지 말기 (비용/속도)
    item_names = [str((it.get("itemName") or "")).strip() for it in (items or [])[:60]]
    item_names = [x for x in item_names if x][:60]

    rt = (receipt_text or "").strip()
    if len(rt) > 2000:
        rt = rt[:2000]

    allowed = sorted(list(_ALLOWED_CODES))

    prompt = f"""
너는 동물병원 영수증을 표준 태그 코드로 매핑하는 분류기야.
아래 allowed_codes 중에서만 골라서 JSON만 출력해.

allowed_codes = {allowed}

입력:
- hospital_name: {hospital_name or ""}
- item_names: {item_names}
- receipt_text: {rt}

출력 JSON 형식(반드시 이 키들만):
{{
  "recordTags": ["..."],                 // 최대 6개
  "itemCategoryTags": [{{"idx":0, "categoryTag":"..."}}]  // idx는 item_names 기준
}}

규칙:
- 확실하지 않으면 빈 배열로 둬.
- recordTags는 가장 대표적인 태그 위주.
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)
        text = (
            (((data.get("candidates") or [])[0] or {}).get("content") or {}).get("parts") or [{}]
        )[0].get("text")
        if not isinstance(text, str) or not text.strip():
            return None

        t = text.strip()
        # 모델이 앞뒤에 설명을 붙이면 JSON 부분만 잘라보기
        if not t.startswith("{"):
            a = t.find("{")
            b = t.rfind("}")
            if a >= 0 and b > a:
                t = t[a : b + 1]

        j = json.loads(t)

        rec = j.get("recordTags")
        item_tags = j.get("itemCategoryTags")
        if not isinstance(rec, list):
            rec = []
        if not isinstance(item_tags, list):
            item_tags = []

        rec2 = [str(x).strip() for x in rec if str(x).strip() in _ALLOWED_CODES][:6]

        it2: List[Dict[str, Any]] = []
        for r in item_tags[:80]:
            if not isinstance(r, dict):
                continue
            idx = r.get("idx")
            ct = str(r.get("categoryTag") or "").strip()
            if isinstance(idx, int) and ct in _ALLOWED_CODES:
                it2.append({"idx": idx, "categoryTag": ct})

        return {"recordTags": rec2, "itemCategoryTags": it2}
    except Exception:
        return None


def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    receipt_text: Optional[str] = None,
    record_thresh: int = 125,
    item_thresh: int = 140,
    max_tags: int = 6,
    **kwargs,
) -> Dict[str, Any]:
    """
    returns:
    {
      "tags": [...],
      "itemCategoryTags": [{"idx":0,"itemName":"...","categoryTag":"...","score":...}, ...],
      "evidence": {...}
    }
    """
    parts: List[str] = []
    if hospital_name:
        parts.append(str(hospital_name))
    for it in (items or [])[:80]:
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if nm:
            parts.append(nm)

    # ✅ items가 부실할 때를 대비해서 OCR 전체 텍스트도 같이 넣을 수 있게
    if receipt_text and receipt_text.strip():
        parts.append(receipt_text.strip()[:2000])

    query = " | ".join(parts)
    query = query.strip()
    if not query:
        return {"tags": [], "itemCategoryTags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}

    # 1) catalog 기반
    record_tags, evidence = _pick_record_tags(query, record_thresh=record_thresh, max_tags=max_tags)
    item_tags = _pick_item_tags(items or [], item_thresh=item_thresh)

    # 2) Gemini 보조 (정말 필요할 때만)
    if (not record_tags) and _truthy_env("GEMINI_ENABLED"):
        g = _gemini_fallback(hospital_name=hospital_name, items=items or [], receipt_text=receipt_text)
        if g:
            record_tags = g.get("recordTags") or record_tags
            # Gemini item tags는 idx만 주므로, 기존 item_tags가 없을 때만 채워
            if not item_tags:
                # itemName은 증거용으로 붙여줌
                tmp = []
                for r in (g.get("itemCategoryTags") or []):
                    idx = r.get("idx")
                    ct = r.get("categoryTag")
                    if isinstance(idx, int) and isinstance(ct, str) and ct in _ALLOWED_CODES:
                        nm = ""
                        if 0 <= idx < len(items or []):
                            nm = str((items[idx].get("itemName") or "")).strip()
                        tmp.append({"idx": idx, "itemName": nm[:120], "categoryTag": ct, "score": 999, "why": ["gemini"]})
                item_tags = tmp

            evidence["geminiUsed"] = True
        else:
            evidence["geminiUsed"] = False

    return {"tags": record_tags, "itemCategoryTags": item_tags, "evidence": evidence}


