# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)

from __future__ import annotations

import os
import re
import json
import base64
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Tag catalog
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

    {"code": "exam_allergy", "group": "exam", "aliases": ["allergy test","ige","atopy","알러지검사","알레르기검사","알러지","알레르기"]},
    {"code": "exam_heart", "group": "exam", "aliases": ["ecg","ekg","echo","cardiac","heart","심전도","심초음파","심장초음파","심장검사"]},
    {"code": "exam_eye", "group": "exam", "aliases": ["schirmer","fluorescein","iop","ophthalmic exam","안압","형광염색","안과검사","안과","눈검사"]},
    {"code": "exam_skin", "group": "exam", "aliases": ["skin scraping","cytology","fungal test","malassezia","피부스크래핑","피부검사","진균","곰팡이","말라세지아"]},

    # --- Vaccines / Preventives ---
    {"code": "vaccine_comprehensive", "group": "vaccine", "aliases": ["dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine","종합백신","혼합백신","5종","6종"]},

    # ✅ 핵심: OCR/약어 대응 (rabb / ra 도)
    {"code": "vaccine_rabies", "group": "vaccine",
     "aliases": ["rabies","rabbies","rabb","rab","ra","광견병","광견","광견백신","광견병백신","광견병접종","광견접종"]},

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

    {"code": "etc_other", "group": "etc", "aliases": ["기타","etc","other"]},
]

_ALLOWED_CODES = {t["code"] for t in TAG_CATALOG}

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

def _edit_distance_max(a: str, b: str, max_dist: int = 1) -> int:
    # small, cutoff DP
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if la == 0:
        return lb
    if lb == 0:
        return la

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        min_row = cur[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
            min_row = min(min_row, cur[j])
        prev = cur
        if min_row > max_dist:
            return max_dist + 1
    return prev[lb]

def _match_score(tag: Dict[str, Any], query: str) -> Tuple[int, Dict[str, Any]]:
    q_raw = (query or "").strip()
    if not q_raw:
        return 0, {}

    # iOS와 동일하게 영문 1글자 입력은 차단 (단, 자동 태깅에서는 query가 길어서 영향 거의 없음)
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

        # ✅ 짧은 약어(us/ua/pi/l2/ra)는 contains 오탐이 크니 token/equal만 허용
        if _is_short_ascii_token(a_norm):
            if a_norm == q_norm or a_norm in token_set:
                best = max(best, 170)
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
        else:
            # ✅ OCR typo 보강: token과 alias가 1글자 정도 다를 때
            if 4 <= len(a_norm) <= 14 and token_set:
                for tk in token_set:
                    if 3 <= len(tk) <= 16 and abs(len(tk) - len(a_norm)) <= 2:
                        d = _edit_distance_max(tk, a_norm, max_dist=1)
                        if d <= 1:
                            best = max(best, 125)
                            hit += 1
                            why.append(f"fuzzy1:{tk}~{a}")
                            break

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

    # ✅ items가 빈약할 때 OCR text를 같이 섞어서 태깅률 올림(너무 길면 자름)
    if ocr_text and isinstance(ocr_text, str):
        txt = ocr_text.strip()
        if txt:
            parts.append(txt[:1200])

    return " | ".join(parts)

def _detect_rabies(items: List[Dict[str, Any]], ocr_text: Optional[str]) -> bool:
    # itemName 기준 우선
    for it in (items or [])[:200]:
        nm = str((it or {}).get("itemName") or "").lower()
        if not nm:
            continue
        if "광견" in nm or "광견병" in nm:
            return True
        toks = {_normalize(t) for t in _tokenize(nm)}
        if ("rabies" in toks) or ("rabbies" in toks) or ("rabb" in toks) or ("rab" in toks):
            return True
        # ra는 단독이면 오탐 가능 → itemName 토큰에 'ra' 있으면 rabies로 인정 (현장 약어 케이스)
        if "ra" in toks:
            return True

    # ocr_text에도 있으면 True
    if ocr_text:
        s = ocr_text.lower()
        if "광견" in s:
            return True
        toks = {_normalize(t) for t in _tokenize(s)}
        if ("rabies" in toks) or ("rabbies" in toks) or ("rabb" in toks) or ("rab" in toks) or ("ra" in toks):
            return True

    return False


# -----------------------------
# Gemini tag assist (optional)
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
    blob = t[i: j + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None

def _gemini_tag_assist(
    *,
    api_key: str,
    model: str,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str],
    ocr_text: Optional[str],
    timeout_seconds: int,
    max_tags: int,
) -> Optional[Dict[str, Any]]:
    api_key = (api_key or "").strip()
    model = (model or "gemini-2.5-flash").strip()
    if not api_key or not model:
        return None

    allowed = sorted(list(_ALLOWED_CODES))
    # 너무 길면 모델이 헛소리하니 요약해서 전달
    item_names = [str((it or {}).get("itemName") or "")[:120] for it in (items or [])[:40] if str((it or {}).get("itemName") or "").strip()]
    ctx = {
        "hospitalName": hospital_name,
        "items": item_names,
        "ocrTextPreview": (ocr_text or "")[:1200] if isinstance(ocr_text, str) else "",
        "allowedCodes": allowed,
        "maxTags": int(max_tags),
    }

    prompt = (
        "You are a veterinary receipt tagger.\n"
        "Return ONLY valid JSON: {\"tags\": [..], \"itemCategoryTags\": [{\"itemName\":\"...\",\"categoryTag\":\"...\"}] }\n"
        "- tags must be chosen ONLY from allowedCodes.\n"
        "- choose up to maxTags.\n"
        "- If you see Rabies (including OCR typo rabbies / abbreviation RA / Korean 광견병), include vaccine_rabies.\n"
    )

    parts = [{"text": prompt + "\n\nINPUT:\n" + json.dumps(ctx, ensure_ascii=False)}]
    try:
        out = _call_gemini_generate_content(api_key=api_key, model=model, parts=parts, timeout_seconds=timeout_seconds)
        j = _extract_json_from_model_text(out)
        if isinstance(j, dict):
            return j
    except Exception:
        return None
    return None


# -----------------------------
# Public API
# -----------------------------
def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    ocr_text: Optional[str] = None,
    record_thresh: int = 125,
    item_thresh: int = 140,
    max_tags: int = 6,
    return_item_tags: bool = True,
    # optional gemini
    gemini_enabled: Optional[bool] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model_name: Optional[str] = None,
    gemini_timeout_seconds: Optional[int] = None,
    gemini_force: Optional[bool] = None,
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

    # 아무것도 못 잡으면 etc_other 마지막 fallback
    if not picked:
        for code, score, _ in scored[:30]:
            if code == "etc_other" and score >= 90:
                picked.append("etc_other")
                break

    # ✅ 하드 룰: rabies 신호 있으면 무조건 vaccine_rabies 포함
    if _detect_rabies(items or [], ocr_text):
        if "vaccine_rabies" not in picked:
            picked.insert(0, "vaccine_rabies")
            picked = picked[: int(max_tags)]

    # -----------------------------
    # item-level tags
    # -----------------------------
    item_tags: List[Dict[str, Any]] = []
    if return_item_tags:
        for idx, it in enumerate((items or [])[:200]):
            nm = (it.get("itemName") or "").strip()
            if not nm:
                continue

            # rabies item hard tag
            if _detect_rabies([{"itemName": nm}], None):
                item_tags.append(
                    {"idx": idx, "itemName": nm, "categoryTag": "vaccine_rabies", "score": 999, "why": ["rabies_hard_rule"]}
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

    # -----------------------------
    # Gemini assist (optional)
    # -----------------------------
    g_enabled = bool(gemini_enabled) if gemini_enabled is not None else _env_bool("GEMINI_ENABLED")
    g_key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")) or ""
    g_model = (gemini_model_name if gemini_model_name is not None else os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")) or "gemini-2.5-flash"
    g_timeout = int(gemini_timeout_seconds if gemini_timeout_seconds is not None else int(os.getenv("GEMINI_TIMEOUT_SECONDS", "10") or "10"))
    g_force = bool(gemini_force) if gemini_force is not None else _env_bool("GEMINI_TAG_FORCE")

    need_gemini = False
    if g_force:
        need_gemini = True
    else:
        # catalog가 너무 빈약한 경우만
        if (not picked) or (picked == ["etc_other"]):
            need_gemini = True

    if g_enabled and g_key.strip() and need_gemini:
        gj = _gemini_tag_assist(
            api_key=g_key,
            model=g_model,
            items=items or [],
            hospital_name=hospital_name,
            ocr_text=ocr_text,
            timeout_seconds=g_timeout,
            max_tags=max_tags,
        )
        if isinstance(gj, dict):
            tags_in = gj.get("tags")
            if isinstance(tags_in, list):
                clean = []
                for t in tags_in:
                    s = str(t).strip()
                    if s in _ALLOWED_CODES and s not in clean:
                        clean.append(s)
                if clean:
                    picked = clean[: int(max_tags)]
                    evidence["geminiUsed"] = True

            # itemCategoryTags from gemini (optional)
            rows = gj.get("itemCategoryTags")
            if isinstance(rows, list):
                for r in rows[:200]:
                    if not isinstance(r, dict):
                        continue
                    nm = str(r.get("itemName") or "").strip()
                    ct = str(r.get("categoryTag") or "").strip()
                    if nm and ct in _ALLOWED_CODES:
                        # idx 없이 들어오면 itemName 기반으로만 전달
                        item_tags.append({"idx": -1, "itemName": nm, "categoryTag": ct, "score": 0, "why": ["gemini"]})

    # rabies hard rule 다시 한번 보장
    if _detect_rabies(items or [], ocr_text):
        if "vaccine_rabies" not in picked:
            picked.insert(0, "vaccine_rabies")
            picked = picked[: int(max_tags)]

    return {"tags": picked, "itemCategoryTags": item_tags, "evidence": evidence}


