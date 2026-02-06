# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Tag catalog (FULL iOS ReceiptTag coverage)
# code == ReceiptTag.rawValue
# aliases == ReceiptTag.hospitalAliases (+ keywords)
# -----------------------------
TAG_CATALOG: List[Dict[str, Any]] = [
    # --- Exams ---
    {"code": "exam_xray", "group": "exam",
     "aliases": ["x-ray","xray","xr","radiograph","radiology","엑스레이","방사선","x선","x선촬영"]},
    {"code": "exam_ultrasound", "group": "exam",
     "aliases": ["ultrasound","sono","sonography","us","초음파","복부초음파","심장초음파"]},
    {"code": "exam_blood", "group": "exam",
     "aliases": ["cbc","blood test","chemistry","biochem","profile","혈액","혈액검사","생화학","전해질","검사"]},
    {"code": "exam_lab_panel", "group": "exam",
     "aliases": ["lab panel","screening","health check","종합검사","종합검진","패널검사"]},
    {"code": "exam_urine", "group": "exam",
     "aliases": ["urinalysis","ua","urine test","요검사","소변검사"]},
    {"code": "exam_fecal", "group": "exam",
     "aliases": ["fecal","stool test","대변검사","분변검사","배변검사"]},
    {"code": "exam_allergy", "group": "exam",
     "aliases": ["allergy test","ige","atopy","알러지검사","알레르기검사","알러지","알레르기"]},
    {"code": "exam_heart", "group": "exam",
     "aliases": ["echo","ecco","ecg","ekg","cardiac","heart check","심초음파","심장초음파","심전도","심장검사"]},
    {"code": "exam_eye", "group": "exam",
     "aliases": ["schirmer","fluorescein","iop","ophthalmic exam","안압","형광염색","눈검사","안과검사","안과"]},
    {"code": "exam_skin", "group": "exam",
     "aliases": ["skin scraping","fungal test","cytology","yeast","말라세지아","곰팡이","진균","피부스크래핑","피부검사"]},

    # --- Exams (extra) ---
    {"code": "exam_sdma", "group": "exam",
     "aliases": ["sdma","symmetrical dimethylarginine","idexx sdma","renal sdma","SDMA","신장마커","신장검사","신장수치"]},
    {"code": "exam_probnp", "group": "exam",
     "aliases": ["probnp","pro bnp","pro-bnp","ntprobnp","nt-probnp","bnp","cardiopet","proBNP","NT-proBNP","심장마커","프로비엔피"]},
    {"code": "exam_fructosamine", "group": "exam",
     "aliases": ["fructosamine","fru","glycated albumin","ga","프럭토사민","프룩토사민","당화알부민","당화"]},
    {"code": "exam_glucose_curve", "group": "exam",
     "aliases": ["glucose curve","blood glucose curve","bg curve","혈당곡선","혈당커브","혈당 커브","연속혈당","혈당측정"]},
    {"code": "exam_blood_gas", "group": "exam",
     "aliases": ["blood gas","bga","bgas","i-stat","istat","i stat","혈액가스","가스분석"]},
    {"code": "exam_fecal_pcr", "group": "exam",
     "aliases": ["fecal pcr","stool pcr","gi pcr","panel pcr","fecal-PCR","stool-PCR","대변pcr","대변 pcr","분변 pcr","배설물 pcr"]},

    # --- Vaccines ---
    {"code": "vaccine_comprehensive", "group": "vaccine",
     "aliases": ["dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine","종합백신","혼합백신","5종","6종"]},
    {"code": "vaccine_rabies", "group": "vaccine",
     "aliases": ["rabies","광견병","광견","rabie"]},
    {"code": "vaccine_kennel", "group": "vaccine",
     "aliases": ["kennel cough","bordetella","켄넬코프","기관지염백신","보르데텔라"]},
    {"code": "vaccine_corona", "group": "vaccine",
     "aliases": ["corona","coronavirus","corona enteritis","코로나","코로나장염"]},
    {"code": "vaccine_lepto", "group": "vaccine",
     "aliases": ["lepto","leptospirosis","leptospira","lepto2","lepto4","l2","l4","렙토","렙토2","렙토4","렙토 2종","렙토 4종"]},
    {"code": "vaccine_parainfluenza", "group": "vaccine",
     "aliases": ["parainfluenza","cpiv","cpi","pi","파라인플루엔자","파라인","파라"]},
    {"code": "vaccine_fip", "group": "vaccine",
     "aliases": ["fip","primucell","feline infectious peritonitis","전염성복막염","복막염","FIP"]},

    # --- Preventives ---
    {"code": "prevent_heartworm", "group": "preventive_med",
     "aliases": ["heartworm","hw","dirofilaria","heartgard","심장사상충","하트가드","넥스가드스펙트라","simparica trio","revolution"]},
    {"code": "prevent_external", "group": "preventive_med",
     "aliases": ["flea","tick","bravecto","nexgard","frontline","revolution","벼룩","진드기","외부기생충"]},
    {"code": "prevent_deworming", "group": "preventive_med",
     "aliases": ["deworm","deworming","drontal","milbemax","fenbendazole","panacur","구충","구충제","내부기생충"]},

    # --- Medicines ---
    {"code": "medicine_antibiotic", "group": "medicine",
     "aliases": ["antibiotic","abx","amoxicillin","clavamox","augmentin","cephalexin","convenia","doxycycline","metronidazole","baytril","항생제"]},
    {"code": "medicine_anti_inflammatory", "group": "medicine",
     "aliases": ["nsaid","anti-inflammatory","meloxicam","metacam","carprofen","rimadyl","onsior","galliprant","소염","소염제"]},
    {"code": "medicine_allergy", "group": "medicine",
     "aliases": ["apoquel","cytopoint","cetirizine","zyrtec","benadryl","알러지","알레르기","가려움","allergy","atopy"]},
    {"code": "medicine_gi", "group": "medicine",
     "aliases": ["famotidine","pepcid","omeprazole","sucralfate","cerenia","ondansetron","reglan","위장약","구토","설사","장염"]},
    {"code": "medicine_ear", "group": "medicine",
     "aliases": ["ear drop","otic","otitis","otomax","surolan","posatex","easotic","귀약","이염","외이염"]},
    {"code": "medicine_skin", "group": "medicine",
     "aliases": ["dermatitis","chlorhexidine","ketoconazole","miconazole","피부약","피부염"]},
    {"code": "medicine_eye", "group": "medicine",
     "aliases": ["eye drop","ophthalmic","tobramycin","ofloxacin","cyclosporine","안약","점안","결막염","각막"]},
    {"code": "medicine_painkiller", "group": "medicine",
     "aliases": ["analgesic","tramadol","gabapentin","buprenorphine","진통","진통제"]},
    {"code": "medicine_steroid", "group": "medicine",
     "aliases": ["steroid","prednisone","prednisolone","dexamethasone","스테로이드","pred","dex"]},

    # --- Care / Procedures / Goods ---
    {"code": "care_injection", "group": "checkup",
     "aliases": ["inj","injection","shot","sc","im","iv","주사","주사제","피하주사","근육주사","정맥주사","주사료"]},
    {"code": "care_procedure_fee", "group": "checkup",
     "aliases": ["procedure fee","treatment fee","handling fee","처치료","시술료","처치비","시술비","처치","시술"]},
    {"code": "care_dressing", "group": "checkup",
     "aliases": ["dressing","bandage","gauze","wrap","disinfection","드레싱","붕대","거즈","소독","세척","상처처치"]},
    {"code": "care_e_collar", "group": "etc",
     "aliases": ["e-collar","ecollar","cone","elizabethan collar","넥카라","엘리자베스카라","보호카라"]},
    {"code": "care_prescription_diet", "group": "etc",
     "aliases": ["prescription diet","rx diet","therapeutic diet","처방식","처방사료","병원사료","hill's","hills","royal canin","k/d","c/d","i/d","z/d"]},

    # --- Surgery / Dental / Ortho / General ---
    {"code": "surgery_general", "group": "surgery",
     "aliases": ["surgery","operation","spay","neuter","castration","수술","중성화","봉합","마취"]},
    {"code": "dental_scaling", "group": "dental",
     "aliases": ["scaling","dental cleaning","tartar","스케일링","치석"]},
    {"code": "dental_extraction", "group": "dental",
     "aliases": ["extraction","dental extraction","발치"]},
    {"code": "ortho_patella", "group": "orthopedic",
     "aliases": ["pl","mpl","lpl","patella","patellar luxation","슬개골탈구","슬탈","파행"]},
    {"code": "ortho_arthritis", "group": "orthopedic",
     "aliases": ["arthritis","oa","osteoarthritis","관절염","퇴행성관절","djd"]},

    {"code": "checkup_general", "group": "checkup",
     "aliases": ["checkup","consult","opd","진료","상담","초진","재진","진찰"]},
    {"code": "grooming_basic", "group": "grooming",
     "aliases": ["grooming","bath","trim","미용","목욕","클리핑"]},

    # fallback
    {"code": "etc_other", "group": "etc", "aliases": ["기타","etc","other"]},
]

# ---- Normalization/token rules (similar to iOS)
_ALNUM_RE = re.compile(r"[0-9a-zA-Z가-힣]+")

def _normalize(s: str) -> str:
    s = (s or "").lower()
    # keep only alnum + hangul
    return "".join(ch for ch in s if ch.isalnum() or ("가" <= ch <= "힣"))

def _tokenize(s: str) -> List[str]:
    return [t for t in _ALNUM_RE.findall(s or "") if t]

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

    if _is_single_latin_char(q_raw):
        return 0, {}

    q_norm = _normalize(q_raw)
    if not q_norm:
        return 0, {}

    tokens = [_normalize(t) for t in _tokenize(q_raw)]
    token_set = set([t for t in tokens if t])

    # strongest: code direct
    code_norm = _normalize(str(tag.get("code") or ""))
    if code_norm and code_norm == q_norm:
        return 230, {"why": ["code==query"]}

    best = 0
    hit = 0
    strong = False
    why: List[str] = []

    for alias in tag.get("aliases", []) or []:
        a = str(alias or "").strip()
        if not a:
            continue
        a_norm = _normalize(a)
        if not a_norm:
            continue

        # short tokens: allow either token match OR normalized full equality (fix U/S -> us)
        if _is_short_ascii_token(a_norm):
            if (a_norm == q_norm) or (a_norm in token_set):
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
            # contains (문장형에서도 살아남게)
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

def _best_tag_for_text(text: str) -> Tuple[Optional[str], int, Dict[str, Any]]:
    best_code: Optional[str] = None
    best_score = 0
    best_ev: Dict[str, Any] = {}

    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, text)
        if s > best_score:
            best_score = s
            best_code = str(tag.get("code") or "")
            best_ev = ev or {}
    return best_code, best_score, best_ev

def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    raw_text: Optional[str] = None,
    record_thresh: int = 125,
    item_thresh: int = 140,
    max_tags: int = 6,
    **kwargs,
) -> Dict[str, Any]:
    # record query: hospital + all itemNames + (optional) raw_text snippet
    parts: List[str] = []
    if hospital_name:
        parts.append(str(hospital_name))

    # raw_text는 너무 길면 비용만 증가 → 앞부분만
    if isinstance(raw_text, str) and raw_text.strip():
        parts.append(raw_text.strip()[:1200])

    for it in (items or [])[:120]:
        if not isinstance(it, dict):
            continue
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if nm:
            parts.append(nm)

    query = " | ".join(parts)
    if not query.strip():
        return {"tags": [], "itemCategoryTags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}

    # 1) record-level tags
    scored: List[Tuple[str, int, Dict[str, Any]]] = []
    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, query)
        if s > 0:
            scored.append((str(tag["code"]), s, ev))

    scored.sort(key=lambda x: x[1], reverse=True)

    picked: List[str] = []
    evidence: Dict[str, Any] = {
        "policy": "catalog",
        "record_thresh": int(record_thresh),
        "item_thresh": int(item_thresh),
        "query": query[:400],
        "candidates": [],
    }

    for code, score, ev in scored[:16]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= record_thresh and code not in picked:
            picked.append(code)
        if len(picked) >= max_tags:
            break

    # 2) item-level categoryTag (idx 기반)
    item_tags: List[Dict[str, Any]] = []
    for idx, it in enumerate((items or [])[:120]):
        if not isinstance(it, dict):
            continue
        name = (it.get("itemName") or it.get("item_name") or "").strip()
        if not name:
            continue

        best_code, best_score, best_ev = _best_tag_for_text(name)
        if best_code and best_score >= item_thresh:
            item_tags.append({
                "idx": idx,
                "itemName": name[:120],
                "categoryTag": best_code,
                "score": best_score,
                **(best_ev or {}),
            })

    return {"tags": picked, "itemCategoryTags": item_tags, "evidence": evidence}


