# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
# - catalog-first
# - returns record-level tags + per-item categoryTag suggestions

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Tag catalog (expandable)
# code == ReceiptTag.rawValue
# -----------------------------
TAG_CATALOG: List[Dict[str, Any]] = [
    # --- Exams ---
    {"code": "exam_xray", "group": "exam",
     "aliases": ["x-ray","xray","xr","radiograph","radiology","엑스레이","방사선","x선","x선촬영"]},
    {"code": "exam_ultrasound", "group": "exam",
     "aliases": ["ultrasound","sono","sonography","u/s","us","초음파","복부초음파","심장초음파"]},
    {"code": "exam_blood", "group": "exam",
     "aliases": ["cbc","blood test","chemistry","biochem","profile","혈액","혈액검사","생화학","전해질","검사"]},
    {"code": "exam_lab_panel", "group": "exam",
     "aliases": ["lab panel","screening","health check","종합검사","종합검진","패널검사"]},
    {"code": "exam_urine", "group": "exam",
     "aliases": ["urinalysis","u/a","ua","urine test","요검사","소변검사"]},
    {"code": "exam_fecal", "group": "exam",
     "aliases": ["fecal","stool test","대변검사","분변검사","배변검사"]},

    # ✅ missing iOS tags (필수 보강)
    {"code": "exam_allergy", "group": "exam",
     "aliases": ["allergy test","ige","atopy","알러지검사","알레르기검사","알러지","알레르기"]},
    {"code": "exam_heart", "group": "exam",
     "aliases": ["ecg","ekg","echo","cardiac","heart","심전도","심초음파","심장초음파","심장검사"]},
    {"code": "exam_eye", "group": "exam",
     "aliases": ["schirmer","fluorescein","iop","ophthalmic exam","안압","형광염색","안과검사","안과"]},
    {"code": "exam_skin", "group": "exam",
     "aliases": ["skin scraping","cytology","fungal test","malassezia","피부스크래핑","피부검사","진균","곰팡이","말라세지아"]},

    # --- Vaccines / Preventives ---
    {"code": "vaccine_comprehensive", "group": "vaccine",
     "aliases": ["dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine","종합백신","혼합백신","5종","6종"]},

    # ✅ Rabies / "Rabbies" OCR typo 대응
    {"code": "vaccine_rabies", "group": "vaccine",
     "aliases": ["rabies","rabbies","rabie","광견병","광견"]},

    {"code": "vaccine_kennel", "group": "vaccine",
     "aliases": ["kennel cough","bordetella","켄넬코프","기관지염백신","보르데텔라"]},

    {"code": "prevent_heartworm", "group": "preventive_med",
     "aliases": ["heartworm","hw","dirofilaria","heartgard","심장사상충","하트가드","넥스가드스펙트라","simparica trio","revolution"]},

    # --- Medicines ---
    {"code": "medicine_antibiotic", "group": "medicine",
     "aliases": ["antibiotic","abx","amoxicillin","clavamox","augmentin","cephalexin","convenia","doxycycline","metronidazole","baytril","항생제"]},
    {"code": "medicine_anti_inflammatory", "group": "medicine",
     "aliases": ["nsaid","anti-inflammatory","meloxicam","metacam","carprofen","rimadyl","onsior","galliprant","소염","소염제"]},
    {"code": "medicine_painkiller", "group": "medicine",
     "aliases": ["analgesic","tramadol","gabapentin","buprenorphine","진통","진통제"]},
    {"code": "medicine_steroid", "group": "medicine",
     "aliases": ["steroid","prednisone","prednisolone","dexamethasone","스테로이드"]},

    # --- Care / Checkup ---
    {"code": "checkup_general", "group": "checkup",
     "aliases": ["checkup","consult","opd","진료","상담","초진","재진","진찰","진료비","진료비용"]},

    # --- Fallback (마지막에만) ---
    {"code": "etc_other", "group": "etc",
     "aliases": ["기타","etc","other"]},
]

# ---- Normalization/token rules
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

        # 짧은 약어(us/ua 등)는 contains 오탐이 크니 token match + exact match만
        if _is_short_ascii_token(a_norm):
            if a_norm == q_norm or a_norm in token_set:
                best = max(best, 160)
                hit += 1
                strong = True
                why.append(f"shortEqOrToken:{a}")
            continue

        if a_norm == q_norm:
            best = max(best, 190)
            hit += 1
            strong = True
            why.append(f"eq:{a}")
        elif q_norm.find(a_norm) >= 0:
            s = 120 + min(70, len(a_norm) * 3)
            best = max(best, s)
            hit += 1
            strong = True
            why.append(f"inQuery:{a}")
        elif a_norm.find(q_norm) >= 0:
            s = 95 + min(45, len(q_norm) * 3)
            best = max(best, s)
            hit += 1
            why.append(f"queryInAlias:{a}")

    if hit >= 2:
        best += min(35, hit * (8 if strong else 5))
        why.append(f"bonus:{hit}")

    return best, {"why": why[:8]}


def _pick_from_catalog(query: str, *, thresh: int, max_tags: int) -> Tuple[List[str], Dict[str, Any]]:
    scored: List[Tuple[str, int, Dict[str, Any]]] = []
    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, query)
        if s > 0:
            scored.append((tag["code"], s, ev))

    scored.sort(key=lambda x: x[1], reverse=True)

    picked: List[str] = []
    evidence: Dict[str, Any] = {"policy": "catalog", "query": query[:400], "candidates": []}

    for code, score, ev in scored[:15]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= int(thresh) and code not in picked:
            picked.append(code)
        if len(picked) >= int(max_tags):
            break

    return picked, evidence


def _tag_items(items: List[Dict[str, Any]], *, thresh: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate((items or [])[:120]):
        if not isinstance(it, dict):
            continue
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if not nm:
            continue

        best_code = None
        best_score = 0
        best_ev: Dict[str, Any] = {}

        for tag in TAG_CATALOG:
            s, ev = _match_score(tag, nm)
            if s > best_score:
                best_score = s
                best_code = tag["code"]
                best_ev = ev or {}

        if best_code and best_score >= int(thresh):
            out.append({
                "idx": idx,
                "itemName": nm[:200],
                "categoryTag": best_code,
                "score": int(best_score),
                "why": best_ev.get("why", []) if isinstance(best_ev, dict) else [],
            })

    return out


def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    # build a single query text from items + hospital
    parts: List[str] = []
    if hospital_name:
        parts.append(str(hospital_name))
    for it in (items or [])[:120]:
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if nm:
            parts.append(nm)

    query = " | ".join(parts)
    if not query.strip():
        return {"tags": [], "itemCategoryTags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}

    # ✅ threshold: record-level은 너무 빡빡하면 “아무것도 안 붙는” 문제가 커서 125 권장
    record_tags, evidence = _pick_from_catalog(query, thresh=125, max_tags=6)

    # ✅ item-level은 record보다 살짝 보수적으로 140 권장
    item_tags = _tag_items(items, thresh=140)

    # etc_other는 마지막 fallback (정말 아무것도 못 잡았을 때만)
    if not record_tags and (items or hospital_name):
        record_tags = ["etc_other"]

    return {"tags": record_tags, "itemCategoryTags": item_tags, "evidence": evidence}


