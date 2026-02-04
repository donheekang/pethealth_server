# tag_policy.py
# PetHealth+ Tag Policy
# - Convert OCR/AI-extracted text (itemName/diagnosis/service) -> canonical ReceiptTag codes
# - Designed to match your iOS ReceiptTag.swift standard codes
#
# Usage examples:
#   from tag_policy import tag_items_and_record_tags
#   tagged_items, record_tags, debug = tag_items_and_record_tags(parsed["items"])
#
# Notes:
# - Item-level: assigns ONE categoryTag (string code) by default.
# - Record-level: union of per-line inferred tags (can include multiple like rabies + dhpp + patella).
# - Fallback: if nothing matches confidently, return "etc_other" (optional).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
import re


# =========================================================
# Data model
# =========================================================

@dataclass(frozen=True)
class TagPreset:
    code: str
    label: str
    group_title: str
    strong_keywords: List[str] = field(default_factory=list)
    weak_keywords: List[str] = field(default_factory=list)


# =========================================================
# Helpers (normalize / tokenize)
# =========================================================

_ALLOWED_SHORT_ABBR: Set[str] = {
    # exam / procedure abbreviations that appear on receipts
    "us", "ua", "xr", "ecg", "ekg", "hw", "abx",
    "im", "iv", "sc",
    "pl", "mpl", "lpl",
    "l2", "l4",
    "pi",
    # common lab markers
    "bnp", "sdma", "fru",
}

def normalize(s: str) -> str:
    # keep unicode letters+digits (Korean 포함), remove punctuation/spaces
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())

def tokenize(s: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for ch in (s or ""):
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out

def is_single_latin_char(s: str) -> bool:
    if not s or len(s) != 1:
        return False
    c = s[0]
    return ("a" <= c.lower() <= "z")

def is_korean_single_char(s: str) -> bool:
    if not s or len(s) != 1:
        return False
    u = ord(s)
    return 0xAC00 <= u <= 0xD7A3

def is_short_ascii_token(norm: str) -> bool:
    if not norm or len(norm) > 2:
        return False
    for ch in norm:
        if not (("a" <= ch <= "z") or ("0" <= ch <= "9")):
            return False
    return True


# =========================================================
# Canonical tag presets (based on your ReceiptTag.swift)
# =========================================================

TAG_PRESETS: List[TagPreset] = [

    # -------------------------
    # 검사
    # -------------------------
    TagPreset(
        code="exam_xray",
        label="엑스레이",
        group_title="검사",
        strong_keywords=[
            "x-ray", "xray", "xr", "radiograph", "radiology",
            "엑스레이", "방사선", "x선", "x선촬영",
        ],
        weak_keywords=["검사", "촬영"],
    ),
    TagPreset(
        code="exam_ultrasound",
        label="초음파",
        group_title="검사",
        strong_keywords=[
            "ultrasound", "sono", "sonography", "usg",
            "초음파", "복부초음파", "심장초음파",
            "u/s", "u s", "us",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_blood",
        label="혈액검사",
        group_title="검사",
        strong_keywords=[
            "cbc", "blood test", "chemistry", "biochem", "profile", "electrolyte",
            "혈액검사", "혈검", "피검", "생화학", "전해질",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_lab_panel",
        label="종합검사",
        group_title="검사",
        strong_keywords=[
            "lab panel", "screening panel", "profile panel",
            "종합검사", "종합검진", "패널검사",
        ],
        weak_keywords=["검사", "검진"],
    ),
    TagPreset(
        code="exam_urine",
        label="소변검사",
        group_title="검사",
        strong_keywords=[
            "urinalysis", "urine test", "ua",
            "소변검사", "요검사",
            "u/a", "ua",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_fecal",
        label="대변검사",
        group_title="검사",
        strong_keywords=[
            "fecal", "stool test",
            "대변검사", "분변검사", "배변검사",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_fecal_pcr",
        label="대변 PCR",
        group_title="검사",
        strong_keywords=[
            "fecal pcr", "stool pcr", "gi pcr", "panel pcr",
            "대변pcr", "대변 pcr", "분변pcr", "분변 pcr",
            "장염 pcr", "설사 pcr", "gi panel",
        ],
        weak_keywords=["검사", "pcr"],
    ),
    TagPreset(
        code="exam_allergy",
        label="알러지 검사",
        group_title="검사",
        strong_keywords=[
            "allergy test", "ige", "atopy",
            "알러지검사", "알레르기검사",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_heart",
        label="심장 검사",
        group_title="검사",
        strong_keywords=[
            "echo", "ecg", "ekg", "cardiac",
            "심장검사", "심전도", "심초음파", "심장초음파",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_eye",
        label="안과 검사",
        group_title="검사",
        strong_keywords=[
            "schirmer", "fluorescein", "iop", "ophthalmic exam",
            "안과검사", "안압", "형광염색", "쉬르머",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_skin",
        label="피부 검사",
        group_title="검사",
        strong_keywords=[
            "skin scraping", "cytology", "fungal test",
            "말라세지아", "진균", "곰팡이",
            "피부검사", "피부스크래핑",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_sdma",
        label="SDMA",
        group_title="검사",
        strong_keywords=[
            "sdma", "renal sdma", "kidney", "renal",
            "신장마커", "신장검사", "신장수치",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_probnp",
        label="proBNP",
        group_title="검사",
        strong_keywords=[
            "probnp", "pro bnp", "nt-probnp", "ntprobnp", "bnp",
            "cardiopet", "cardio pet",
            "심장마커", "심장수치", "프로비엔피",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_fructosamine",
        label="당화알부민",
        group_title="검사",
        strong_keywords=[
            "fructosamine", "fru", "glycated albumin", "ga",
            "당화알부민", "당뇨", "혈당",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_glucose_curve",
        label="혈당곡선",
        group_title="검사",
        strong_keywords=[
            "glucose curve", "bg curve", "glucose monitoring",
            "혈당곡선", "혈당커브", "혈당 커브", "연속혈당",
        ],
        weak_keywords=["검사"],
    ),
    TagPreset(
        code="exam_blood_gas",
        label="혈액가스",
        group_title="검사",
        strong_keywords=[
            "blood gas", "bga", "i-stat", "istat",
            "혈액가스", "가스분석", "혈가",
        ],
        weak_keywords=["검사"],
    ),

    # -------------------------
    # 예방접종
    # -------------------------
    TagPreset(
        code="vaccine_comprehensive",
        label="종합백신",
        group_title="예방접종",
        strong_keywords=[
            "dhpp", "dhppi", "dhppl", "dhlpp", "fvrcp",
            "combo vaccine",
            "5-in-1", "6-in-1", "5in1", "6in1",
            "종합백신", "혼합백신", "5종", "6종",
        ],
        weak_keywords=["백신", "접종"],
    ),
    TagPreset(
        code="vaccine_rabies",
        label="광견병 백신",
        group_title="예방접종",
        strong_keywords=["rabies", "광견병", "광견"],
        weak_keywords=["백신", "접종"],
    ),
    TagPreset(
        code="vaccine_kennel",
        label="켄넬코프 백신",
        group_title="예방접종",
        strong_keywords=["kennel cough", "bordetella", "켄넬코프", "보르데텔라", "기관지염백신"],
        weak_keywords=["백신", "접종"],
    ),
    TagPreset(
        code="vaccine_corona",
        label="코로나 백신",
        group_title="예방접종",
        strong_keywords=["corona", "coronavirus", "코로나", "코로나장염"],
        weak_keywords=["백신", "접종"],
    ),
    TagPreset(
        code="vaccine_fip",
        label="FIP",
        group_title="예방접종",
        strong_keywords=["fip", "primucell", "feline infectious peritonitis", "전염성복막염", "복막염", "FIP 백신"],
        weak_keywords=["백신", "접종"],
    ),
    TagPreset(
        code="vaccine_parainfluenza",
        label="파라인플루엔자",
        group_title="예방접종",
        strong_keywords=["parainfluenza", "cpiv", "cpi", "pi", "파라인플루엔자", "파라인", "파라"],
        weak_keywords=["백신", "접종"],
    ),
    TagPreset(
        code="vaccine_lepto",
        label="렙토",
        group_title="예방접종",
        strong_keywords=["lepto", "leptospirosis", "leptospira", "l2", "l4", "lepto2", "lepto4", "렙토", "렙토2", "렙토4"],
        weak_keywords=["백신", "접종"],
    ),

    # -------------------------
    # 예방약/구충
    # -------------------------
    TagPreset(
        code="prevent_heartworm",
        label="심장사상충 예방",
        group_title="예방약/구충",
        strong_keywords=[
            "heartworm", "hw", "dirofilaria",
            "heartgard", "ivermectin", "milbemycin",
            "nexgard spectra", "simparica trio", "revolution",
            "심장사상충", "하트가드", "넥스가드스펙트라", "리볼루션",
        ],
        weak_keywords=["예방", "구충"],
    ),
    TagPreset(
        code="prevent_external",
        label="외부기생충 예방",
        group_title="예방약/구충",
        strong_keywords=[
            "flea", "tick",
            "bravecto", "nexgard", "frontline", "revolution",
            "벼룩", "진드기", "외부기생충",
        ],
        weak_keywords=["예방", "구충"],
    ),
    TagPreset(
        code="prevent_deworming",
        label="구충",
        group_title="예방약/구충",
        strong_keywords=[
            "deworm", "deworming", "dewormer", "internal parasite",
            "drontal", "milbemax", "panacur", "fenbendazole", "pyrantel", "praziquantel",
            "구충", "구충제", "내부기생충", "회충", "선충",
        ],
        weak_keywords=["예방", "구충"],
    ),

    # -------------------------
    # 처방약
    # -------------------------
    TagPreset(
        code="medicine_antibiotic",
        label="항생제",
        group_title="처방약",
        strong_keywords=[
            "antibiotic", "abx",
            "amoxicillin", "amoxi", "amox", "augmentin", "clavamox",
            "cephalexin", "cefovecin", "convenia",
            "doxycycline", "doxy", "metronidazole", "metro",
            "enrofloxacin", "baytril", "clindamycin",
            "항생제",
        ],
        weak_keywords=["약", "처방"],
    ),
    TagPreset(
        code="medicine_anti_inflammatory",
        label="소염제",
        group_title="처방약",
        strong_keywords=[
            "nsaid", "anti-inflammatory", "anti inflammatory",
            "meloxicam", "metacam",
            "carprofen", "rimadyl",
            "onsior", "galliprant",
            "소염", "소염제",
        ],
        weak_keywords=["약", "처방"],
    ),
    TagPreset(
        code="medicine_painkiller",
        label="진통제",
        group_title="처방약",
        strong_keywords=[
            "pain", "analgesic",
            "tramadol", "gabapentin", "buprenorphine", "codeine",
            "진통", "진통제",
        ],
        weak_keywords=["약", "처방"],
    ),
    TagPreset(
        code="medicine_steroid",
        label="스테로이드",
        group_title="처방약",
        strong_keywords=[
            "steroid",
            "pred", "prednisone", "prednisolone", "dexamethasone", "dex",
            "스테로이드",
        ],
        weak_keywords=["약", "처방"],
    ),
    TagPreset(
        code="medicine_gi",
        label="위장약",
        group_title="처방약",
        strong_keywords=[
            "gi", "gastro",
            "famotidine", "pepcid",
            "omeprazole", "sucralfate",
            "cerenia", "maropitant", "ondansetron", "reglan", "metoclopramide",
            "위장", "위장약", "구토", "설사", "장염",
        ],
        weak_keywords=["약", "처방"],
    ),
    TagPreset(
        code="medicine_eye",
        label="안약",
        group_title="처방약",
        strong_keywords=[
            "eye drop", "eyedrop", "ophthalmic", "gtt", "drops",
            "tobramycin", "ofloxacin", "ciprofloxacin", "cyclosporine",
            "안약", "점안", "결막염", "각막",
        ],
        weak_keywords=["약", "처방"],
    ),
    TagPreset(
        code="medicine_ear",
        label="귀약",
        group_title="처방약",
        strong_keywords=[
            "ear drop", "otic", "otitis",
            "otomax", "surolan", "posatex", "easotic",
            "귀약", "이염", "외이염",
        ],
        weak_keywords=["약", "처방"],
    ),
    TagPreset(
        code="medicine_skin",
        label="피부약",
        group_title="처방약",
        strong_keywords=[
            "derm", "dermatitis", "topical",
            "chlorhexidine", "ketoconazole", "miconazole",
            "피부", "피부약", "피부염",
        ],
        weak_keywords=["약", "처방"],
    ),
    TagPreset(
        code="medicine_allergy",
        label="알러지약",
        group_title="처방약",
        strong_keywords=[
            "allergy", "atopy",
            "apoquel", "cytopoint",
            "cetirizine", "zyrtec", "loratadine", "claritin",
            "알러지", "알레르기", "가려움",
        ],
        weak_keywords=["약", "처방"],
    ),

    # -------------------------
    # 처치/진료(소모품 포함)
    # -------------------------
    TagPreset(
        code="care_injection",
        label="주사/주사제",
        group_title="기본/진료",
        strong_keywords=[
            "inj", "injection", "shot",
            "sc", "s/c", "im", "i/m", "iv", "i/v",
            "주사", "주사제", "피하주사", "근육주사", "정맥주사",
        ],
        weak_keywords=["처치", "시술"],
    ),
    TagPreset(
        code="care_procedure_fee",
        label="처치료",
        group_title="기본/진료",
        strong_keywords=[
            "procedure fee", "treatment fee", "handling fee",
            "procedure", "treatment",
            "처치료", "처치비", "시술료", "시술비", "처치", "시술",
        ],
        weak_keywords=["진료", "비용"],
    ),
    TagPreset(
        code="care_dressing",
        label="드레싱",
        group_title="기본/진료",
        strong_keywords=[
            "dressing", "bandage", "bandaging", "wrap", "gauze",
            "cleaning", "disinfection",
            "드레싱", "붕대", "거즈", "소독", "세척", "상처처치",
        ],
        weak_keywords=["처치"],
    ),
    TagPreset(
        code="care_e_collar",
        label="넥카라",
        group_title="기타",
        strong_keywords=[
            "e-collar", "ecollar", "cone",
            "elizabethan collar", "elizabeth collar",
            "넥카라", "엘리자베스카라", "보호카라",
        ],
        weak_keywords=["소모품"],
    ),
    TagPreset(
        code="care_prescription_diet",
        label="처방식",
        group_title="기타",
        strong_keywords=[
            "prescription diet", "rx diet", "therapeutic diet",
            "처방식", "처방사료", "병원사료",
            "hill's", "hills", "royal canin", "로얄캐닌",
            "k/d", "c/d", "s/d", "i/d", "z/d", "h/d",
            "renal diet", "urinary diet", "gastrointestinal diet", "hypoallergenic diet",
        ],
        weak_keywords=["사료"],
    ),

    # -------------------------
    # 수술/치과/정형
    # -------------------------
    TagPreset(
        code="surgery_general",
        label="수술/처치",
        group_title="수술/처치",
        strong_keywords=[
            "surgery", "operation", "spay", "neuter", "castration",
            "anesthesia", "봉합", "마취",
            "수술", "중성화",
        ],
        weak_keywords=["처치"],
    ),
    TagPreset(
        code="dental_scaling",
        label="스케일링",
        group_title="치과",
        strong_keywords=["scaling", "dental cleaning", "tartar", "스케일링", "치석"],
        weak_keywords=["치과"],
    ),
    TagPreset(
        code="dental_extraction",
        label="발치",
        group_title="치과",
        strong_keywords=["extraction", "dental extraction", "발치"],
        weak_keywords=["치과"],
    ),
    TagPreset(
        code="ortho_patella",
        label="슬개골",
        group_title="관절/정형",
        strong_keywords=[
            "pl", "mpl", "lpl",
            "patella", "patellar", "patellar luxation",
            "stifle", "knee",
            "슬개골탈구", "슬탈", "무릎탈구", "파행",
        ],
        weak_keywords=["정형", "관절"],
    ),
    TagPreset(
        code="ortho_arthritis",
        label="관절염",
        group_title="관절/정형",
        strong_keywords=[
            "arthritis", "oa", "osteoarthritis",
            "djd", "degenerative joint disease",
            "lameness", "joint pain",
            "관절통", "퇴행성", "퇴행성관절", "관절염",
        ],
        weak_keywords=["정형", "관절"],
    ),

    # -------------------------
    # 기본/진료 / 미용 / 기타
    # -------------------------
    TagPreset(
        code="checkup_general",
        label="기본진료/검진",
        group_title="기본/진료",
        strong_keywords=[
            "checkup", "consult", "opd",
            "초진", "재진", "진료", "진찰", "상담", "검진",
        ],
        weak_keywords=["기본", "진료"],
    ),
    TagPreset(
        code="grooming_basic",
        label="미용",
        group_title="미용",
        strong_keywords=["grooming", "bath", "trim", "clipping", "미용", "목욕", "클리핑"],
        weak_keywords=[],
    ),

    # fallback
    TagPreset(
        code="etc_other",
        label="기타",
        group_title="기타",
        strong_keywords=["기타", "etc", "other"],
        weak_keywords=[],
    ),
]


# =========================================================
# Legacy/alias code support (optional)
# If older client uses MedicalTagPreset codes like "drug_antibiotic"
# =========================================================

TAG_CODE_ALIASES: Dict[str, str] = {
    "drug_antibiotic": "medicine_antibiotic",
    "drug_pain_antiinflammatory": "medicine_anti_inflammatory",
    "drug_allergy": "medicine_allergy",
    "drug_gi": "medicine_gi",
    "drug_ear": "medicine_ear",
    "drug_skin": "medicine_skin",
    "drug_eye": "medicine_eye",
    "drug_steroid": "medicine_steroid",
    "exam_general": "checkup_general",
}

def canonicalize_tag_code(code: str) -> str:
    c = (code or "").strip()
    if not c:
        return c
    return TAG_CODE_ALIASES.get(c, c)


# =========================================================
# Scoring engine (Swift logic port + small OCR-friendly tweaks)
# =========================================================

def score_preset_for_text(p: TagPreset, text: str) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (score, debug)
    """
    raw = (text or "").strip()
    if not raw:
        return 0, {"reason": "empty"}

    # avoid 1-letter latin blow-ups
    if is_single_latin_char(raw):
        return 0, {"reason": "single_latin"}

    q_norm = normalize(raw)
    if not q_norm:
        return 0, {"reason": "norm_empty"}

    token_norms = [normalize(t) for t in tokenize(raw)]
    token_norms = [t for t in token_norms if t]
    token_set = set(token_norms)

    # strongest: code/label exact
    if normalize(p.code) == q_norm:
        return 230, {"hit": "code_exact"}
    if normalize(p.label) == q_norm:
        return 220, {"hit": "label_exact"}

    best = 0
    hit_count = 0
    strong_hit = False
    hits: List[str] = []

    def consider(score: int, *, is_strong: bool, why: str):
        nonlocal best, hit_count, strong_hit
        if score <= 0:
            return
        hit_count += 1
        best = max(best, score)
        strong_hit = strong_hit or is_strong
        hits.append(why)

    # label/code contained in sentence
    label_norm = normalize(p.label)
    if label_norm and (label_norm in q_norm):
        consider(200 + min(30, len(label_norm)), is_strong=True, why="label_in_text")

    code_norm = normalize(p.code)
    if code_norm and (code_norm in q_norm):
        consider(195 + min(25, len(code_norm)), is_strong=True, why="code_in_text")

    # keyword matching
    def apply_keywords(keys: List[str], *, base_exact: int, base_contains: int, base_reverse: int, is_weak: bool):
        nonlocal best
        for k in keys:
            k_raw = (k or "").strip()
            if not k_raw:
                continue
            k_norm = normalize(k_raw)
            if not k_norm:
                continue

            # short tokens: avoid false positives
            if is_short_ascii_token(k_norm):
                # token match first
                if k_norm in token_set:
                    consider((70 if is_weak else 135), is_strong=not is_weak, why=f"short_token:{k_norm}")
                    continue
                # OCR-friendly: allow collapsed match for known abbreviations (e.g. "U/S" -> "us")
                if k_norm in _ALLOWED_SHORT_ABBR and (k_norm in q_norm):
                    consider((55 if is_weak else 120), is_strong=not is_weak, why=f"abbr_in_norm:{k_norm}")
                    continue
                continue

            if k_norm == q_norm:
                consider((110 if is_weak else base_exact), is_strong=not is_weak, why=f"kw_exact:{k_norm}")
            elif k_norm in q_norm:
                consider((55 if is_weak else base_contains) + min(60, len(k_norm) * 2), is_strong=not is_weak, why=f"kw_in_text:{k_norm}")
            elif q_norm in k_norm:
                consider((35 if is_weak else base_reverse) + min(40, len(q_norm) * 2), is_strong=not is_weak, why=f"text_in_kw:{k_norm}")

    apply_keywords(p.strong_keywords, base_exact=180, base_contains=120, base_reverse=90, is_weak=False)
    apply_keywords(p.weak_keywords, base_exact=110, base_contains=55, base_reverse=35, is_weak=True)

    # multi-hit bonus
    if hit_count >= 2:
        best += min(35, hit_count * (8 if strong_hit else 5))

    # Korean single-char assist (e.g. "슬" -> 슬개골)
    if len(raw) == 1 and is_korean_single_char(raw):
        if raw in (p.label or ""):
            best = max(best, 130)
            hits.append("korean_single_char_in_label")

    debug = {
        "preset": p.code,
        "best": best,
        "hitCount": hit_count,
        "strongHit": strong_hit,
        "hits": hits[:10],
    }
    return best, debug


def rank_tags(text: str) -> List[Tuple[TagPreset, int, Dict[str, Any]]]:
    scored: List[Tuple[TagPreset, int, Dict[str, Any]]] = []
    for p in TAG_PRESETS:
        s, dbg = score_preset_for_text(p, text)
        if s > 0:
            scored.append((p, s, dbg))
    scored.sort(key=lambda x: (-x[1], x[0].label))
    return scored


# =========================================================
# Public APIs
# =========================================================

def infer_tag_codes_from_text(
    text: str,
    *,
    max_tags: int = 2,
    min_score: int = 140,
    within_top: int = 28,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Return 1~2 tag codes if strong enough.
    - max_tags=2 is for record-level union (e.g. one line contains 2 vaccines).
    - For item-level categoryTag, you usually use max_tags=1.
    """
    ranked = rank_tags(text)
    if not ranked:
        return [], {"reason": "no_match"}

    top_p, top_s, top_dbg = ranked[0]
    picked: List[Tuple[TagPreset, int]] = []
    debug = {"top": (top_p.code, top_s), "candidates": []}

    for (p, s, dbg) in ranked[:8]:
        debug["candidates"].append({"code": p.code, "label": p.label, "score": s, "hits": dbg.get("hits", [])})

    if top_s < min_score:
        return [], {"reason": "below_threshold", **debug}

    picked.append((top_p, top_s))

    # optionally pick 2nd if it's close enough & strong enough
    for (p, s, _) in ranked[1:]:
        if len(picked) >= max_tags:
            break
        if s < min_score:
            continue
        if s >= top_s - within_top:
            picked.append((p, s))
        else:
            break

    codes = [canonicalize_tag_code(p.code) for (p, _) in picked]
    return codes, {"reason": "ok", **debug}


def infer_item_category_tag(
    item_name: str,
    *,
    min_score: int = 145,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Assign ONE categoryTag code for a single item line.
    """
    codes, dbg = infer_tag_codes_from_text(
        item_name,
        max_tags=1,
        min_score=min_score,
        within_top=0,
    )
    if not codes:
        return None, dbg
    return codes[0], dbg


def tag_items_and_record_tags(
    items: List[Dict[str, Any]],
    *,
    fill_fallback_etc: bool = False,
    item_min_score: int = 145,
    record_min_score: int = 140,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """
    Input items: [{"itemName": "...", "price": 1234, "categoryTag": None}, ...]
    Output:
      - tagged_items: categoryTag set when inferred
      - record_tags: union of strong matches across lines (can include multiple)
      - debug: per-item scoring info + record union info
    """
    record_tag_set: Set[str] = set()
    tagged: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {"items": [], "recordTags": []}

    for it in (items or []):
        name = (it.get("itemName") or it.get("item_name") or "").strip()
        if not name:
            tagged.append(dict(it))
            debug["items"].append({"itemName": name, "skipped": True})
            continue

        # item-level 1 tag
        primary, item_dbg = infer_item_category_tag(name, min_score=item_min_score)

        out_it = dict(it)
        if primary:
            out_it["categoryTag"] = primary
        elif fill_fallback_etc:
            out_it["categoryTag"] = "etc_other"

        tagged.append(out_it)

        # record-level: allow up to 2 tags per line for union
        codes2, rec_dbg = infer_tag_codes_from_text(
            name,
            max_tags=2,
            min_score=record_min_score,
            within_top=28,
        )
        for c in codes2:
            record_tag_set.add(c)

        debug["items"].append({
            "itemName": name,
            "itemCategoryTag": primary,
            "itemDebug": item_dbg,
            "recordDebug": rec_dbg,
        })

    # If record tags ended empty, optionally add etc_other or checkup_general (policy choice)
    record_tags = sorted(record_tag_set)

    debug["recordTags"] = record_tags
    return tagged, record_tags, debug


def validate_tag_codes(codes: List[str]) -> List[str]:
    """
    Keep only known canonical codes (after alias normalization).
    """
    known = {p.code for p in TAG_PRESETS}
    out: List[str] = []
    seen: Set[str] = set()
    for c in (codes or []):
        cc = canonicalize_tag_code(c)
        if not cc or cc not in known:
            continue
        if cc in seen:
            continue
        seen.add(cc)
        out.append(cc)
    return out


def tag_label_map() -> Dict[str, str]:
    """
    code -> label (UI friendly)
    """
    return {p.code: p.label for p in TAG_PRESETS}


