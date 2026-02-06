# tag_policy.py (PetHealth+)
# items/text -> ReceiptTag codes (aligned with iOS ReceiptTag.rawValue)
#
# Required public API:
#   resolve_record_tags(items: list, hospital_name: Optional[str] = None, **kwargs) -> dict
#
# Strategy:
#  1) Catalog-based matching (fast, deterministic)
#  2) Optional Gemini mapping (when enabled + API key exists, and catalog is weak/empty)
#
# Env:
#  - GEMINI_ENABLED=true|false
#  - GEMINI_API_KEY=...
#  - GEMINI_MODEL_NAME=gemini-2.5-flash   (default)
#
# Notes:
#  - Gemini call is best-effort. Any failure falls back to catalog.
#  - This module must NEVER crash the server.

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Tag catalog (subset; expand as needed)
# code == ReceiptTag.rawValue
# aliases == ReceiptTag.hospitalAliases (+ keywords)
# -----------------------------
TAG_CATALOG: List[Dict[str, Any]] = [
    # --- Exams ---
    {"code": "exam_xray", "group": "exam", "aliases": ["x-ray","xray","xr","radiograph","radiology","엑스레이","방사선","x선","x선촬영"]},
    {"code": "exam_ultrasound", "group": "exam", "aliases": ["ultrasound","sono","sonography","us","초음파","복부초음파","심장초음파"]},
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

    # --- Vaccines / Preventives ---
    {"code": "vaccine_comprehensive", "group": "vaccine", "aliases": ["dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine","종합백신","혼합백신","5종","6종"]},
    {"code": "vaccine_rabies", "group": "vaccine", "aliases": ["rabies","광견병","광견"]},
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

    {"code": "checkup_general", "group": "checkup", "aliases": ["checkup","consult","opd","진료","상담","초진","재진","진찰"]},
    {"code": "grooming_basic", "group": "grooming", "aliases": ["grooming","bath","trim","미용","목욕","클리핑"]},
]


# ---- Normalization/token rules (similar to iOS)
def _normalize(s: str) -> str:
    s = (s or "").lower()
    # keep alnum + korean only
    return "".join(ch for ch in s if ch.isalnum() or ("가" <= ch <= "힣"))

def _tokenize(s: str) -> List[str]:
    # tokens by non-alnum boundaries
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


# ---- Noise filters (to avoid "고객번호 9원" 같은 OCR 쓰레기 라인)
_NOISE_TOKENS = [
    "고객", "고객번호", "고객 번호",
    "발행", "발행일", "발행 일",
    "사업자", "사업자등록", "대표", "전화", "주소",
    "serial", "sign", "승인", "카드", "현금",
    "부가세", "vat", "면세", "과세", "공급가",
    "소계", "합계", "총액", "총 금액", "총금액", "청구", "결제",
]

def _is_noise_name(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return True
    low = n.lower()

    # 너무 짧은 단어는 태그 매핑에서 제외
    if len(_normalize(n)) < 2:
        return True

    # 노이즈 토큰 포함
    for t in _NOISE_TOKENS:
        if t in n or t in low:
            return True

    return False

def _is_plausible_amount(price: Optional[int]) -> bool:
    # 아주 작은 금액(예: 9원, 58원)은 OCR 오탐일 가능성이 높음
    # (원하면 50/100/500 등으로 조절 가능)
    if price is None:
        return True
    try:
        return int(price) >= 100
    except Exception:
        return True


# -----------------------------
# Catalog matching
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

    best = 0
    hit = 0
    strong = False
    why: List[str] = []

    code_norm = _normalize(tag.get("code", ""))
    if code_norm and code_norm == q_norm:
        return 230, {"why": ["code==query"]}

    # alias match
    for alias in tag.get("aliases", []):
        a = str(alias or "").strip()
        if not a:
            continue
        a_norm = _normalize(a)
        if not a_norm:
            continue

        # 짧은 약어(us/ua/pi/l2)는 contains 오탐이 크니 토큰일치만
        if _is_short_ascii_token(a_norm):
            if a_norm in token_set:
                best = max(best, 135)
                hit += 1
                strong = True
                why.append(f"token:{a}")
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

    # multi-hit bonus
    if hit >= 2:
        best += min(35, hit * (8 if strong else 5))
        why.append(f"bonus:{hit}")

    return best, {"why": why[:8]}

def _catalog_candidates(query: str, limit: int = 12) -> List[Tuple[str, int, Dict[str, Any]]]:
    scored: List[Tuple[str, int, Dict[str, Any]]] = []
    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, query)
        if s > 0:
            scored.append((tag["code"], s, ev))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]

def _pick_from_catalog(query: str, thresh: int = 135, max_tags: int = 6) -> Tuple[List[str], Dict[str, Any]]:
    cands = _catalog_candidates(query, limit=20)

    picked: List[str] = []
    evidence: Dict[str, Any] = {
        "policy": "catalog",
        "query": query[:500],
        "candidates": [],
    }

    for code, score, ev in cands[:12]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= thresh and code not in picked:
            picked.append(code)
        if len(picked) >= max_tags:
            break

    return picked, evidence

def _best_item_tag_from_catalog(item_name: str) -> Tuple[Optional[str], int, Dict[str, Any]]:
    # item 단독으로도 매칭
    cands = _catalog_candidates(item_name, limit=5)
    if not cands:
        return None, 0, {"policy": "catalog_item", "why": ["no_candidates"]}
    code, score, ev = cands[0]
    return code, score, {"policy": "catalog_item", "top": {"code": code, "score": score, **(ev or {})}}


# -----------------------------
# Gemini (optional)
# -----------------------------
def _gemini_enabled() -> bool:
    return (os.environ.get("GEMINI_ENABLED", "true").strip().lower() == "true")

def _gemini_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY", "") or "").strip()

def _gemini_model_name() -> str:
    return (os.environ.get("GEMINI_MODEL_NAME", "") or "").strip() or "gemini-2.5-flash"

def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None

    # strip ```json fences
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("```", "").strip()

    # try direct
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # find a JSON object substring
    first = s.find("{")
    last = s.rfind("}")
    if first >= 0 and last > first:
        sub = s[first:last+1]
        try:
            obj = json.loads(sub)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None

def _gemini_generate_text(prompt: str) -> str:
    """
    Best-effort Gemini call.
    Supports either google-generativeai OR google-genai if installed.
    Returns "" on any failure.
    """
    api_key = _gemini_api_key()
    if not api_key:
        return ""

    model_name = _gemini_model_name()

    # 1) google-generativeai (google.generativeai)
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 700,
            },
        )
        txt = getattr(resp, "text", "") or ""
        return txt.strip()
    except Exception:
        pass

    # 2) google-genai (from google import genai)
    try:
        from google import genai as genai2  # type: ignore

        client = genai2.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        txt = getattr(resp, "text", "") or ""
        return txt.strip()
    except Exception:
        return ""

def _build_gemini_prompt(
    *,
    hospital_name: Optional[str],
    items: List[Dict[str, Any]],
    allowed_codes: List[str],
    catalog_top: List[Tuple[str, int, Dict[str, Any]]],
) -> str:
    # item names only (cleaned)
    names: List[str] = []
    for it in items[:80]:
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if not nm:
            continue
        if _is_noise_name(nm):
            continue
        names.append(nm[:120])

    payload = {
        "hospitalName": (hospital_name or None),
        "items": names,
        "allowedReceiptTagCodes": allowed_codes,
        "catalogTopCandidates": [{"code": c, "score": s} for (c, s, _ev) in catalog_top[:8]],
        "outputSchema": {
            "tags": "string[] (<=6, unique, allowed codes only)",
            "itemCategoryTags": [
                {"itemName": "string", "categoryTag": "string|null", "confidence": "number 0..1"}
            ],
            "notes": "string(optional)"
        }
    }

    # IMPORTANT: force strict JSON output
    return (
        "You are a mapping engine for veterinary receipts.\n"
        "Task: Map receipt line item names to standardized ReceiptTag codes.\n"
        "Constraints:\n"
        "- Use ONLY codes from allowedReceiptTagCodes.\n"
        "- Return STRICT JSON only. No markdown, no extra text.\n"
        "- tags must be <= 6, unique.\n"
        "- itemCategoryTags: for each itemName, pick categoryTag or null if unsure.\n"
        "- If uncertain, prefer null / fewer tags.\n"
        "- Ignore non-medical/noise lines (customer number, issue date, addresses, phones, tax lines).\n\n"
        "INPUT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )

def _gemini_map(
    *,
    hospital_name: Optional[str],
    items: List[Dict[str, Any]],
    catalog_top: List[Tuple[str, int, Dict[str, Any]]],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: (tags, itemCategoryTags, evidence)
    """
    allowed = [t["code"] for t in TAG_CATALOG if t.get("code")]
    prompt = _build_gemini_prompt(
        hospital_name=hospital_name,
        items=items,
        allowed_codes=allowed,
        catalog_top=catalog_top,
    )

    raw = _gemini_generate_text(prompt)
    if not raw:
        return [], [], {"policy": "gemini", "error": "empty_response"}

    obj = _extract_json_obj(raw)
    if not obj:
        return [], [], {"policy": "gemini", "error": "json_parse_failed", "raw": raw[:500]}

    tags = obj.get("tags") or []
    item_tags = obj.get("itemCategoryTags") or []

    # validate tags
    allowed_set = set(allowed)
    cleaned_tags: List[str] = []
    if isinstance(tags, list):
        for x in tags:
            if not isinstance(x, str):
                continue
            code = x.strip()
            if not code or code not in allowed_set:
                continue
            if code not in cleaned_tags:
                cleaned_tags.append(code)
            if len(cleaned_tags) >= 6:
                break

    # validate itemCategoryTags
    cleaned_item_tags: List[Dict[str, Any]] = []
    if isinstance(item_tags, list):
        for row in item_tags[:120]:
            if not isinstance(row, dict):
                continue
            nm = (row.get("itemName") or "").strip()
            ct = row.get("categoryTag")
            conf = row.get("confidence", None)

            if not nm:
                continue
            if _is_noise_name(nm):
                continue

            if isinstance(ct, str):
                ct = ct.strip()
                if ct not in allowed_set:
                    ct = None
            else:
                ct = None

            try:
                conf_f = float(conf) if conf is not None else None
                if conf_f is not None:
                    conf_f = max(0.0, min(1.0, conf_f))
            except Exception:
                conf_f = None

            cleaned_item_tags.append({
                "itemName": nm[:200],
                "categoryTag": ct,
                "confidence": conf_f,
            })

    ev = {
        "policy": "gemini",
        "model": _gemini_model_name(),
        "rawPreview": raw[:300],
        "notes": obj.get("notes"),
    }
    return cleaned_tags, cleaned_item_tags, ev


# -----------------------------
# Public API
# -----------------------------
def resolve_record_tags(
    *,
    items: List[Dict[str, Any]],
    hospital_name: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Return shape example:
      {
        "tags": ["vaccine_rabies"],
        "itemCategoryTags": [{"itemName": "...", "categoryTag": "vaccine_rabies", "confidence": 0.92}],
        "evidence": {...}
      }

    - Conservative by design. If unsure: returns fewer tags.
    - Never throws.
    """
    try:
        # 1) clean & build query
        parts: List[str] = []
        if hospital_name:
            parts.append(str(hospital_name))

        cleaned_items: List[Dict[str, Any]] = []
        for it in (items or [])[:120]:
            nm = (it.get("itemName") or it.get("item_name") or "").strip()
            price = it.get("price")
            if not nm:
                continue
            if _is_noise_name(nm):
                continue
            if not _is_plausible_amount(price if isinstance(price, int) else None):
                continue

            cleaned_items.append(it)
            parts.append(nm)

        query = " | ".join(parts).strip()
        if not query:
            return {"tags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}

        # 2) catalog baseline
        catalog_tags, catalog_ev = _pick_from_catalog(query, thresh=135, max_tags=6)
        catalog_top = _catalog_candidates(query, limit=12)

        # 3) per-item category tags (catalog quick pass)
        #    - 이건 “아이템에 categoryTag를 채워서 UI에서 바로 표준 태그 표시”용
        item_category_tags: List[Dict[str, Any]] = []
        for it in cleaned_items[:80]:
            nm = (it.get("itemName") or it.get("item_name") or "").strip()
            if not nm:
                continue
            best_code, best_score, ev = _best_item_tag_from_catalog(nm)
            # 아이템 단독은 조금 더 보수적으로
            if best_code and best_score >= 160:
                item_category_tags.append({
                    "itemName": nm[:200],
                    "categoryTag": best_code,
                    "confidence": None,
                    "evidence": ev,
                })

        # record tags에 아이템 태그를 합치기
        merged = list(dict.fromkeys(catalog_tags + [r["categoryTag"] for r in item_category_tags if r.get("categoryTag")]))
        merged = merged[:6]

        # 4) decide gemini
        #    - catalog가 너무 약하거나(0개) 애매하면 AI로 보정
        use_ai = _gemini_enabled() and bool(_gemini_api_key())
        top_score = catalog_top[0][1] if catalog_top else 0
        need_ai = use_ai and (len(merged) == 0 or top_score < 190)

        if need_ai:
            ai_tags, ai_item_tags, ai_ev = _gemini_map(
                hospital_name=hospital_name,
                items=cleaned_items,
                catalog_top=catalog_top,
            )

            # AI tags 우선 + catalog의 아주 강한 것(220+)은 보강
            strong_catalog = [c for (c, s, _ev) in catalog_top if s >= 220]
            final_tags = list(dict.fromkeys(ai_tags + strong_catalog))
            final_tags = final_tags[:6]

            # AI itemCategoryTags를 채택하되, categoryTag 없는 것은 유지(혹은 제거)
            # 필요하면 catalog item 태그와 합쳐도 됨
            final_item_tags: List[Dict[str, Any]] = []
            if ai_item_tags:
                final_item_tags = ai_item_tags[:120]
            else:
                final_item_tags = [{"itemName": r["itemName"], "categoryTag": r.get("categoryTag"), "confidence": r.get("confidence")} for r in item_category_tags]

            return {
                "tags": final_tags,
                "itemCategoryTags": final_item_tags,
                "evidence": {
                    "policy": "gemini+catalog",
                    "catalog": catalog_ev,
                    "gemini": ai_ev,
                },
            }

        # 5) catalog only
        return {
            "tags": merged,
            "itemCategoryTags": [{"itemName": r["itemName"], "categoryTag": r.get("categoryTag"), "confidence": r.get("confidence")} for r in item_category_tags],
            "evidence": catalog_ev,
        }

    except Exception as e:
        # never crash
        return {"tags": [], "evidence": {"policy": "safe_fallback", "error": str(e)[:200]}}


