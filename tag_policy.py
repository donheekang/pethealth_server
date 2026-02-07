# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
#
# Hardening goals:
# - Strong catalog-based tagging.
# - Explicit Rabies robustness: rabies/rabbies/rabi/rab/ra/광견 -> vaccine_rabies
# - Optional Gemini fallback ONLY when catalog fails to pick meaningful tags.

from __future__ import annotations

import os
import re
import json
import base64
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Tag catalog (expand as needed)
# code == ReceiptTag.rawValue
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
    {"code": "exam_sdma", "group": "exam", "aliases": ["sdma","symmetrical dimethylarginine","idexx sdma","renal sdma","신장마커","신장검사"]},
    {"code": "exam_probnp", "group": "exam", "aliases": ["probnp","pro bnp","pro-bnp","ntprobnp","nt-probnp","bnp","cardiopet","심장마커","프로비엔피"]},
    {"code": "exam_fructosamine", "group": "exam", "aliases": ["fructosamine","fru","glycated albumin","ga","프럭토사민","당화알부민"]},
    {"code": "exam_glucose_curve", "group": "exam", "aliases": ["glucose curve","blood glucose curve","bg curve","혈당곡선","혈당커브","혈당 커브","연속혈당"]},
    {"code": "exam_blood_gas", "group": "exam", "aliases": ["blood gas","bga","bgas","i-stat","istat","혈액가스","가스분석"]},

    # --- Extra Exams ---
    {"code": "exam_allergy", "group": "exam", "aliases": ["allergy test","ige","atopy","알러지검사","알레르기검사","알러지","알레르기"]},
    {"code": "exam_heart", "group": "exam", "aliases": ["ecg","ekg","echo","cardiac","heart","심전도","심초음파","심장초음파","심장검사"]},
    {"code": "exam_eye", "group": "exam", "aliases": ["schirmer","fluorescein","iop","ophthalmic exam","안압","형광염색","안과검사","안과","눈검사"]},
    {"code": "exam_skin", "group": "exam", "aliases": ["skin scraping","cytology","fungal test","malassezia","피부스크래핑","피부검사","진균","곰팡이","말라세지아"]},

    # --- Vaccines / Preventives ---
    {"code": "vaccine_comprehensive", "group": "vaccine", "aliases": ["dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine","종합백신","혼합백신","5종","6종"]},

    # ✅ 핵심: OCR이 Rabbies / Rabi / Rab / Ra 로 깨지는 케이스 대응
    {"code": "vaccine_rabies", "group": "vaccine", "aliases": ["rabies","rabbies","rabi","rab","ra","광견병","광견"]},

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

    # --- Care / Procedures / Goods ---
    {"code": "care_injection", "group": "checkup", "aliases": ["inj","injection","shot","sc","im","iv","주사","주사제","피하주사","근육주사","정맥주사","주사료"]},
    {"code": "care_procedure_fee", "group": "checkup", "aliases": ["procedure fee","treatment fee","handling fee","처치료","시술료","처치비","시술비","처치","시술"]},
    {"code": "care_dressing", "group": "checkup", "aliases": ["dressing","bandage","gauze","wrap","disinfection","드레싱","붕대","거즈","소독","세척","상처처치"]},
    {"code": "care_e_collar", "group": "etc", "aliases": ["e-collar","ecollar","cone","elizabethan collar","넥카라","엘리자베스카라","보호카라"]},
    {"code": "care_prescription_diet", "group": "etc", "aliases": ["prescription diet","rx diet","therapeutic diet","처방식","처방사료","병원사료","hill's","hills","royal canin","k/d","c/d","i/d","z/d"]},

    # --- Surgery / Dental / Ortho / General ---
    {"code": "surgery_general", "group": "surgery", "aliases": ["surgery","operation","spay","neuter","castration","수술","중성화","봉합","마취"]},
    {"code": "dental_scaling", "group": "dental", "aliases": ["scaling","dental cleaning","tartar","스케일링","치석"]},
    {"code": "dental_extraction", "group": "dental", "aliases": ["extraction","dental extraction","발치"]},
    {"code": "ortho_patella", "group": "orthopedic", "aliases": ["mpl","lpl","patella","patellar luxation","슬개골탈구","슬탈","파행"]},
    {"code": "ortho_arthritis", "group": "orthopedic", "aliases": ["arthritis","oa","osteoarthritis","관절염","퇴행성관절"]},

    {"code": "checkup_general", "group": "checkup", "aliases": ["checkup","consult","opd","진료","상담","초진","재진","진찰","진료비"]},
    {"code": "grooming_basic", "group": "grooming", "aliases": ["grooming","bath","trim","미용","목욕","클리핑"]},

    # last fallback
    {"code": "etc_other", "group": "etc", "aliases": ["기타","etc","other"]},
]


_TAG_CODE_SET = {t["code"] for t in TAG_CATALOG}


# -----------------------------
# Normalization/token rules
# -----------------------------
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


def _mentions_rabies(text: str) -> bool:
    if not text:
        return False
    if "광견병" in text or "광견" in text:
        return True

    toks = [t.lower() for t in _tokenize(text)]
    for t in toks:
        if t in ("rabies", "rabbies"):
            return True
        if t == "ra":
            return True
        if t.startswith("rab") and len(t) >= 3:
            return True
        if t.startswith("rabi") and len(t) >= 4:
            return True

    n = _normalize(text)
    if "rabies" in n or "rabbies" in n:
        return True
    if "rabi" in n and "arab" not in n:
        return True
    return False


# -----------------------------
# Matching
# -----------------------------
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

    # --- Special hardening: vaccine_rabies ---
    if tag.get("code") == "vaccine_rabies":
        if _mentions_rabies(q_raw):
            return 220, {"why": ["rabies_hard_match"]}

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

        # 짧은 약어(us/ua/pi/l2)는 contains 오탐이 크니 token/완전일치만
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
            # query가 alias의 부분(=OCR이 잘림)인 경우 점수 보강
            # 기존보다 조금 더 주되, 너무 짧은 query는 과대평가 방지
            if len(q_norm) >= 3:
                s = 115 + min(55, len(q_norm) * 3)  # e.g. rabi(4) -> 127, rab(3)->124
            else:
                s = 90 + min(40, len(q_norm) * 2)
            best = max(best, s)
            hit += 1
            why.append(f"queryInAlias:{a}")

    if hit >= 2:
        best += min(35, hit * (8 if strong else 5))
        why.append(f"bonus:{hit}")

    return best, {"why": why[:8]}


def _build_record_query(
    items: List[Dict[str, Any]],
    hospital_name: Optional[str],
) -> str:
    parts: List[str] = []
    if hospital_name:
        parts.append(str(hospital_name))
    for it in (items or [])[:120]:
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if nm:
            parts.append(nm)
    return " | ".join(parts)


# -----------------------------
# Optional Gemini fallback for tags (ONLY when catalog fails)
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
) -> str:
    import urllib.request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
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
    if not api_key or not model:
        return None

    # keep prompt small but strict
    allowed = sorted(list(_TAG_CODE_SET))
    prompt = (
        "You are a classifier for Korean veterinary receipts.\n"
        "Return ONLY valid JSON with keys:\n"
        "  tags: array of tag codes (max {max_tags})\n"
        "  itemCategoryTags: array of {{idx:int, itemName:string, categoryTag:string}}\n"
        "Rules:\n"
        "- Only use codes from this allowed list:\n"
        f"{allowed}\n"
        "- Prefer specific medical tags; avoid 'etc_other' unless nothing matches.\n"
        "- Treat Rabbies/Rabi/Rab/Ra/광견/광견병 as Rabies vaccine -> vaccine_rabies.\n"
    ).format(max_tags=int(max_tags))

    # Provide structured context
    items_text = []
    for idx, it in enumerate((items or [])[:120]):
        nm = str((it or {}).get("itemName") or "").strip()
        pr = (it or {}).get("price")
        if nm:
            items_text.append(f"{idx}. {nm} ({pr})")

    parts = [{"text": prompt}]
    parts.append({"text": "Hospital: " + (hospital_name or "")})
    parts.append({"text": "Items:\n" + "\n".join(items_text)})
    if ocr_text and str(ocr_text).strip():
        # minimal slice to avoid overfitting on noisy full text
        parts.append({"text": "OCR text snippet:\n" + str(ocr_text)[:1200]})

    try:
        out = _call_gemini_generate_content(api_key=api_key, model=model, parts=parts, timeout_seconds=timeout_seconds)
        j = _extract_json_from_model_text(out)
        if isinstance(j, dict):
            return j
    except Exception:
        return None
    return None


def _clean_tags_list(tags: Any) -> List[str]:
    if not tags or not isinstance(tags, list):
        return []
    out: List[str] = []
    seen = set()
    for t in tags:
        s = str(t).strip()
        if not s or s in seen:
            continue
        if s not in _TAG_CODE_SET:
            continue
        seen.add(s)
        out.append(s)
    return out


# -----------------------------
# Public API
# -----------------------------
def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    ocr_text: Optional[str] = None,   # ✅ main.py에서 넘겨도 됨
    record_thresh: int = 125,
    item_thresh: int = 140,
    max_tags: int = 6,
    return_item_tags: bool = True,
    # Optional Gemini (main.py에서 넘겨도 되고 env로도 됨)
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
    query = _build_record_query(items or [], hospital_name)
    if not query.strip() and not (ocr_text or "").strip():
        return {"tags": [], "itemCategoryTags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}

    # --- 1) Catalog scoring on query (items/hospital)
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
    }

    for code, score, ev in scored[:20]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= int(record_thresh) and code not in picked:
            if code == "etc_other":
                continue
            picked.append(code)
        if len(picked) >= int(max_tags):
            break

    # --- 2) Rabies hardening using OCR text too
    if "vaccine_rabies" not in picked:
        if _mentions_rabies(query) or _mentions_rabies(ocr_text or ""):
            picked.insert(0, "vaccine_rabies")
            picked = picked[: int(max_tags)]

    # --- 3) If nothing picked, allow etc_other as last resort
    if not picked:
        for code, score, _ in scored[:30]:
            if code == "etc_other" and score >= 90:
                picked.append("etc_other")
                break

    # --- 4) Per-item tagging (catalog)
    item_tags: List[Dict[str, Any]] = []
    if return_item_tags:
        for idx, it in enumerate((items or [])[:200]):
            nm = (it.get("itemName") or "").strip()
            if not nm:
                continue

            # item-level rabies hardening
            if _mentions_rabies(nm):
                item_tags.append(
                    {"idx": idx, "itemName": nm, "categoryTag": "vaccine_rabies", "score": 220, "why": ["rabies_item_hard_match"]}
                )
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
                    {"idx": idx, "itemName": nm, "categoryTag": best_code, "score": best_score, **(best_ev or {})}
                )

    # --- 5) Optional Gemini fallback when catalog basically failed
    # Condition: no meaningful tags OR only etc_other
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))

    needs_ai = (not picked) or (picked == ["etc_other"])
    if g_enabled and g_key.strip() and needs_ai:
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
            ai_tags = _clean_tags_list(gj.get("tags"))
            if ai_tags:
                # Merge (ai first), but keep rabies hardening too
                merged = []
                for t in ai_tags:
                    if t not in merged:
                        merged.append(t)
                for t in picked:
                    if t not in merged:
                        merged.append(t)
                picked = merged[: int(max_tags)]
                evidence["geminiUsed"] = True
                evidence["geminiTags"] = ai_tags

            # itemCategoryTags (ai)
            ai_item_rows = gj.get("itemCategoryTags")
            if isinstance(ai_item_rows, list):
                # only keep valid codes
                cleaned_ai_items = []
                for r in ai_item_rows[:200]:
                    if not isinstance(r, dict):
                        continue
                    idx = r.get("idx")
                    nm = str(r.get("itemName") or "").strip()
                    ct = str(r.get("categoryTag") or "").strip()
                    if not nm or ct not in _TAG_CODE_SET:
                        continue
                    if ct == "etc_other":
                        continue
                    if isinstance(idx, int) and idx >= 0:
                        cleaned_ai_items.append({"idx": idx, "itemName": nm, "categoryTag": ct, "score": 0, "why": ["gemini"]})
                if cleaned_ai_items:
                    evidence["geminiItemTags"] = cleaned_ai_items

    # final: rabies hardening again
    if "vaccine_rabies" not in picked and (_mentions_rabies(query) or _mentions_rabies(ocr_text or "")):
        picked = (["vaccine_rabies"] + picked)[: int(max_tags)]

    return {"tags": picked, "itemCategoryTags": item_tags, "evidence": evidence}


