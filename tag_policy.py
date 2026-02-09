# tag_policy.py (PetHealth+)
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

TAG_CATALOG: List[Dict[str, Any]] = [
    # === 검사 — 영상 ===
    {"code": "exam_xray", "group": "exam", "aliases": ["x-ray","xray","xr","radiograph","radiology","엑스레이","방사선","x선","x선촬영","치아 방사선","dental xray","dental x-ray"]},
    {"code": "exam_ct", "group": "exam", "aliases": ["ct","ct촬영","ct검사","ct조영","ct scan","computed tomography","컴퓨터단층촬영","씨티","조영ct","contrast ct"]},
    {"code": "exam_mri", "group": "exam", "aliases": ["mri","mri촬영","mri검사","mri scan","magnetic resonance","자기공명","엠알아이"]},
    {"code": "exam_endoscope", "group": "exam", "aliases": ["내시경","endoscopy","endoscope","gastroscopy","위내시경","장내시경","기관지내시경","비강내시경","방광경","arthroscopy","관절경","내시경검사"]},
    {"code": "exam_biopsy", "group": "exam", "aliases": ["생검","조직검사","biopsy","tissue biopsy","fna","fine needle","세침흡인","병리검사","조직병리","histopathology","pathology"]},
    # === 초음파 세분화 ===
    {"code": "exam_echo", "group": "exam", "aliases": ["심장초음파","심초음파","심장 초음파","cardiac ultrasound","cardiac us","echocardiogram","echocardiography","echo","ecco","심장 에코","에코"]},
    {"code": "exam_us_abdomen", "group": "exam", "aliases": ["복부초음파","복부 초음파","abdominal ultrasound","abdominal us","abd us","abd sono","복부에코"]},
    {"code": "exam_us_general", "group": "exam", "aliases": ["ultrasound","sono","sonography","us","초음파","초음파검사"]},
    # === 혈액검사 세분화 ===
    {"code": "exam_blood_cbc", "group": "exam", "aliases": ["cbc","complete blood count","혈구검사","혈구","CBC검사","blood count"]},
    {"code": "exam_blood_chem", "group": "exam", "aliases": ["chemistry","biochem","biochemistry","생화학","생화학검사","간수치","신장수치","간기능","신장기능","chemistry panel","chem"]},
    {"code": "exam_blood_general", "group": "exam", "aliases": ["blood test","profile","혈액","혈액검사","피검사","피검","blood work","bloodwork"]},
    {"code": "exam_blood_type", "group": "exam", "aliases": ["혈액형","혈액형검사","blood type","blood typing","crossmatch","교차시험"]},
    {"code": "exam_coagulation", "group": "exam", "aliases": ["응고검사","응고","coagulation","pt","aptt","pt/aptt","프로트롬빈","피브리노겐","fibrinogen"]},
    {"code": "exam_electrolyte", "group": "exam", "aliases": ["전해질","전해질검사","electrolyte","나트륨","칼륨","칼슘","calcium","phosphorus"]},
    {"code": "exam_crp", "group": "exam", "aliases": ["crp","c-reactive protein","염증수치","염증검사","염증마커","crp검사","씨알피"]},
    # === 심장검사 세분화 ===
    {"code": "exam_ecg", "group": "exam", "aliases": ["ecg","ekg","심전도","electrocardiogram","electrocardiography","심전도검사","12 lead"]},
    {"code": "exam_heart_general", "group": "exam", "aliases": ["cardiac","heart","심장검사","심장 검사","심장","heart check","cardiac exam"]},
    # === 호르몬 ===
    {"code": "exam_hormone", "group": "exam", "aliases": ["호르몬","호르몬검사","hormone","hormone test","t4","t3","ft4","tsh","갑상선","갑상선검사","thyroid","cortisol","코르티솔","acth","부신","부신검사","adrenal"]},
    # === 기타 검사 ===
    {"code": "exam_lab_panel", "group": "exam", "aliases": ["lab panel","screening","health check","종합검사","종합검진","패널검사"]},
    {"code": "exam_urine", "group": "exam", "aliases": ["urinalysis","ua","urine test","요검사","소변검사"]},
    {"code": "exam_fecal", "group": "exam", "aliases": ["fecal","stool test","대변검사","분변검사","배변검사"]},
    {"code": "exam_fecal_pcr", "group": "exam", "aliases": ["fecal pcr","stool pcr","gi pcr","panel pcr","대변pcr","대변 pcr","분변 pcr"]},
    {"code": "exam_sdma", "group": "exam", "aliases": ["sdma","symmetrical dimethylarginine","idexx sdma","신장마커","신장검사"]},
    {"code": "exam_probnp", "group": "exam", "aliases": ["probnp","pro bnp","pro-bnp","ntprobnp","nt-probnp","bnp","cardiopet","심장마커","프로비엔피"]},
    {"code": "exam_fructosamine", "group": "exam", "aliases": ["fructosamine","fru","glycated albumin","ga","프럭토사민","당화알부민"]},
    {"code": "exam_glucose_curve", "group": "exam", "aliases": ["glucose curve","blood glucose curve","bg curve","혈당곡선","혈당커브","연속혈당"]},
    {"code": "exam_blood_gas", "group": "exam", "aliases": ["blood gas","bga","bgas","i-stat","istat","혈액가스","가스분석"]},
    {"code": "exam_allergy", "group": "exam", "aliases": ["allergy test","ige","atopy","알러지검사","알레르기검사","알러지","알레르기"]},
    {"code": "exam_eye", "group": "exam", "aliases": ["schirmer","fluorescein","iop","ophthalmic exam","안압","형광염색","안과검사","안과","눈검사"]},
    {"code": "exam_skin", "group": "exam", "aliases": ["skin scraping","cytology","fungal test","malassezia","피부스크래핑","피부검사","진균","곰팡이","말라세지아"]},
    # === 예방접종 ===
    {"code": "vaccine_comprehensive", "group": "vaccine", "aliases": ["dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine","종합백신","혼합백신","5종백신","6종백신","5종접종","6종접종"]},
    {"code": "vaccine_rabies", "group": "vaccine", "aliases": ["rabies","rabbies","rabie","rabis","rab","rabi","ra","rabies vac","rabies vaccine","광견병","광견","광견병백신","광견병접종","예방접종"]},
    {"code": "vaccine_kennel", "group": "vaccine", "aliases": ["kennel cough","bordetella","켄넬코프","기관지염백신","보르데텔라"]},
    {"code": "vaccine_corona", "group": "vaccine", "aliases": ["corona","coronavirus","corona enteritis","코로나","코로나장염"]},
    {"code": "vaccine_lepto", "group": "vaccine", "aliases": ["lepto","leptospirosis","leptospira","lepto2","lepto4","l2","l4","렙토","렙토2","렙토4"]},
    {"code": "vaccine_parainfluenza", "group": "vaccine", "aliases": ["parainfluenza","cpiv","cpi","pi","파라인플루엔자","파라인","파라"]},
    {"code": "vaccine_fip", "group": "vaccine", "aliases": ["fip","primucell","feline infectious peritonitis","전염성복막염","복막염"]},
    # === 예방약 ===
    {"code": "prevent_heartworm", "group": "preventive_med", "aliases": ["heartworm","hw","dirofilaria","heartgard","심장사상충","하트가드","넥스가드스펙트라","simparica trio","revolution"]},
    {"code": "prevent_external", "group": "preventive_med", "aliases": ["flea","tick","bravecto","nexgard","frontline","revolution","벼룩","진드기","외부기생충"]},
    {"code": "prevent_deworming", "group": "preventive_med", "aliases": ["deworm","deworming","drontal","milbemax","fenbendazole","panacur","구충","구충제","내부기생충"]},
    # === 처방약 ===
    {"code": "medicine_antibiotic", "group": "medicine", "aliases": ["antibiotic","abx","amoxicillin","clavamox","augmentin","cephalexin","convenia","doxycycline","metronidazole","baytril","항생제"]},
    {"code": "medicine_anti_inflammatory", "group": "medicine", "aliases": ["nsaid","anti-inflammatory","meloxicam","metacam","carprofen","rimadyl","onsior","galliprant","소염","소염제"]},
    {"code": "medicine_painkiller", "group": "medicine", "aliases": ["analgesic","tramadol","gabapentin","buprenorphine","진통","진통제"]},
    {"code": "medicine_steroid", "group": "medicine", "aliases": ["steroid","prednisone","prednisolone","dexamethasone","스테로이드"]},
    {"code": "medicine_gi", "group": "medicine", "aliases": ["famotidine","pepcid","omeprazole","sucralfate","cerenia","ondansetron","reglan","위장약","구토","설사","장염"]},
    {"code": "medicine_eye", "group": "medicine", "aliases": ["eye drop","ophthalmic","tobramycin","ofloxacin","cyclosporine","안약","점안","결막염","각막"]},
    {"code": "medicine_ear", "group": "medicine", "aliases": ["ear drop","otic","otitis","otomax","surolan","posatex","easotic","귀약","이염","외이염"]},
    {"code": "medicine_skin", "group": "medicine", "aliases": ["dermatitis","chlorhexidine","ketoconazole","miconazole","피부약","피부염"]},
    {"code": "medicine_allergy", "group": "medicine", "aliases": ["apoquel","cytopoint","cetirizine","zyrtec","benadryl","알러지","알레르기","가려움"]},
    {"code": "medicine_oral", "group": "medicine", "aliases": ["내복약","경구약","먹는약","oral","oral med","oral medication","po","per os","처방약","약값"]},
    # === 처치/수액/응급/수혈 ===
    {"code": "care_injection", "group": "checkup", "aliases": ["inj","injection","shot","sc","im","iv","주사","주사제","피하주사","근육주사","정맥주사","주사료"]},
    {"code": "care_fluid", "group": "checkup", "aliases": ["수액","링거","iv fluid","fluid therapy","수액처치","수액치료","피하수액","정맥수액","lactated ringer","생리식염수","normal saline","수액세트","링거액","hartmann"]},
    {"code": "care_transfusion", "group": "checkup", "aliases": ["수혈","transfusion","blood transfusion","전혈","packed rbc","혈장","plasma","수혈비","혈액제제","fresh frozen plasma","ffp"]},
    {"code": "care_oxygen", "group": "checkup", "aliases": ["산소","산소치료","산소방","산소텐트","oxygen","oxygen therapy","o2","산소공급","산소케이지"]},
    {"code": "care_emergency", "group": "checkup", "aliases": ["응급","응급처치","응급진료","emergency","ER","응급비","응급진료비","야간진료","야간응급","심폐소생","CPR","cpr"]},
    {"code": "care_catheter", "group": "checkup", "aliases": ["카테터","도뇨관","유치도뇨관","catheter","urinary catheter","정맥카테터","iv catheter","도뇨","방광세척"]},
    {"code": "care_procedure_fee", "group": "checkup", "aliases": ["procedure fee","treatment fee","handling fee","처치료","시술료","처치비","시술비","처치","시술"]},
    {"code": "care_dressing", "group": "checkup", "aliases": ["dressing","bandage","gauze","wrap","disinfection","드레싱","붕대","거즈","소독","세척","상처처치"]},
    {"code": "care_anal_gland", "group": "checkup", "aliases": ["항문낭","항문낭짜기","항문낭세척","anal gland","anal sac","항문선","항문낭 압출"]},
    {"code": "care_ear_flush", "group": "checkup", "aliases": ["귀세척","이도세척","ear flush","ear cleaning","ear irrigation","귀청소"]},
    # === 입원 ===
    {"code": "hospitalization", "group": "checkup", "aliases": ["입원","입원비","입원료","hospitalization","hospital stay","icu","중환자","중환자실","집중치료","입원관리","입원케어","케이지","cage"]},
    # === 수술 세분화 ===
    {"code": "surgery_general", "group": "surgery", "aliases": ["surgery","operation","수술","봉합","마취","마취료","마취-호흡","흡입마취","전신마취","국소마취"]},
    {"code": "surgery_spay_neuter", "group": "surgery", "aliases": ["중성화","spay","neuter","castration","ovariohysterectomy","ohe","수컷 중성화","암컷 중성화","자궁적출","난소적출","고환적출","중성화수술"]},
    {"code": "surgery_tumor", "group": "surgery", "aliases": ["종양","종양제거","종양수술","tumor","tumor removal","mass removal","mass","lump","혹제거","혹","종괴","종괴제거"]},
    {"code": "surgery_foreign_body", "group": "surgery", "aliases": ["이물","이물제거","이물수술","foreign body","foreign body removal","이물질","위절개","장절개","gastrotomy","enterotomy"]},
    {"code": "surgery_cesarean", "group": "surgery", "aliases": ["제왕절개","cesarean","c-section","caesarean","제왕","제왕절개수술"]},
    {"code": "surgery_hernia", "group": "surgery", "aliases": ["탈장","탈장수술","hernia","hernia repair","회음부탈장","서혜부탈장","perineal hernia","inguinal hernia","배꼽탈장","umbilical hernia"]},
    {"code": "surgery_eye", "group": "surgery", "aliases": ["안과수술","eye surgery","체리아이","cherry eye","백내장","백내장수술","cataract","안구적출","enucleation","안구","눈수술","각막수술"]},
    # === 치과 ===
    {"code": "dental_scaling", "group": "dental", "aliases": ["scaling","dental cleaning","tartar","스케일링","치석"]},
    {"code": "dental_extraction", "group": "dental", "aliases": ["extraction","dental extraction","발치"]},
    {"code": "dental_treatment", "group": "dental", "aliases": ["잇몸","잇몸치료","치주","치주치료","periodontal","gingival","불소","불소도포","fluoride","치아치료","root canal","신경치료"]},
    # === 관절/정형 ===
    {"code": "ortho_patella", "group": "orthopedic", "aliases": ["mpl","lpl","patella","patellar luxation","슬개골탈구","슬탈","파행"]},
    {"code": "ortho_arthritis", "group": "orthopedic", "aliases": ["arthritis","oa","osteoarthritis","관절염","퇴행성관절"]},
    # === 재활 ===
    {"code": "rehab_therapy", "group": "checkup", "aliases": ["재활","재활치료","물리치료","rehabilitation","physical therapy","physio","수중치료","수중런닝머신","underwater treadmill","hydrotherapy","레이저","레이저치료","laser therapy","cold laser","침","침치료","acupuncture"]},
    # === 마이크로칩 ===
    {"code": "microchip", "group": "etc", "aliases": ["마이크로칩","microchip","chip","칩","내장형칩","동물등록","동물 등록","pet registration","칩삽입"]},
    # === 안락사/장례 ===
    {"code": "euthanasia", "group": "etc", "aliases": ["안락사","euthanasia","peaceful passing","임종","임종처치"]},
    {"code": "funeral", "group": "etc", "aliases": ["장례","화장","cremation","funeral","pet funeral","장례비","화장비","반려동물장례","개별화장","합동화장","유골","납골"]},
    # === 기타 ===
    {"code": "care_e_collar", "group": "etc", "aliases": ["e-collar","ecollar","cone","elizabethan collar","넥카라","엘리자베스카라","보호카라"]},
    {"code": "care_prescription_diet", "group": "etc", "aliases": ["prescription diet","rx diet","therapeutic diet","처방식","처방사료","병원사료","hill's","hills","royal canin","k/d","c/d","i/d","z/d"]},
    {"code": "checkup_general", "group": "checkup", "aliases": ["checkup","consult","opd","진료","상담","초진","재진","진찰","진료비"]},
    {"code": "grooming_basic", "group": "grooming", "aliases": ["grooming","bath","trim","미용","목욕","클리핑"]},
    {"code": "etc_other", "group": "etc", "aliases": ["기타","etc","other"]},
]

_alnum = re.compile(r"[0-9a-zA-Z가-힣]+")
def _normalize(s: str) -> str:
    s = (s or "").lower()
    return "".join(ch for ch in s if ch.isalnum() or ("가" <= ch <= "힣"))
def _tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[0-9a-zA-Z가-힣]+", s or "") if t]
def _is_short_ascii_token(norm: str) -> bool:
    if len(norm) > 3: return False
    return all(("0" <= c <= "9") or ("a" <= c <= "z") for c in norm)
def _is_single_latin_char(s: str) -> bool:
    if len(s) != 1: return False
    return "a" <= s.lower() <= "z"

_RABIES_RA_RE = re.compile(r"(?<![0-9a-z])ra(?![0-9a-z])", re.IGNORECASE)
_RABIES_R_A_RE = re.compile(r"\br\s*[/\-\._ ]\s*a\b", re.IGNORECASE)
_TAG_NOISE = ["사업자","대표","전화","주소","고객","승인","카드","현금","합계","총액","총금액","청구","결제","소계","vat","부가세","면세","과세","serial","sign","발행","발행일","날짜","일자"]
_TAG_NOISE_N = [_normalize(x) for x in _TAG_NOISE if _normalize(x)]

def _is_noise_textline(line: str) -> bool:
    t = (line or "").strip()
    if not t: return True
    n = _normalize(t)
    if len(n) < 2: return True
    if re.search(r"\b20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\b", t): return True
    for x in _TAG_NOISE_N:
        if x in n: return True
    return False

def _match_score(tag: Dict[str, Any], query: str) -> Tuple[int, Dict[str, Any]]:
    q_raw = (query or "").strip()
    if not q_raw: return 0, {}
    if _is_single_latin_char(q_raw): return 0, {}
    q_norm = _normalize(q_raw)
    if not q_norm: return 0, {}
    tokens = [_normalize(t) for t in _tokenize(q_raw)]
    token_set = set(t for t in tokens if t)
    best = 0; hit = 0; strong = False; why: List[str] = []
    code_norm = _normalize(tag["code"])
    if code_norm == q_norm: return 230, {"why": ["code==query"]}
    if tag.get("code") == "vaccine_rabies":
        if _RABIES_RA_RE.search(q_raw) or _RABIES_R_A_RE.search(q_raw):
            best = max(best, 170); hit += 1; strong = True; why.append("regex:ra_or_r/a")
    for alias in tag.get("aliases", []):
        a = str(alias or "").strip()
        if not a: continue
        a_norm = _normalize(a)
        if not a_norm: continue
        if _is_short_ascii_token(a_norm):
            if a_norm == q_norm or a_norm in token_set:
                best = max(best, 160); hit += 1; strong = True; why.append(f"shortEqOrToken:{a}")
            continue
        if a_norm == q_norm:
            best = max(best, 180); hit += 1; strong = True; why.append(f"eq:{a}")
        elif q_norm.find(a_norm) >= 0:
            s = 120 + min(60, len(a_norm) * 2)
            best = max(best, s); hit += 1; strong = True; why.append(f"inQuery:{a}")
        elif a_norm.find(q_norm) >= 0:
            s = 90 + min(40, len(q_norm) * 2)
            best = max(best, s); hit += 1; why.append(f"queryInAlias:{a}")
    if hit >= 2: best += min(35, hit * (8 if strong else 5)); why.append(f"bonus:{hit}")
    return best, {"why": why[:10]}

def _build_record_query(items, hospital_name, ocr_text=None):
    parts = []
    if hospital_name: parts.append(str(hospital_name))
    for it in (items or [])[:200]:
        nm = (it.get("itemName") or it.get("item_name") or "").strip()
        if nm: parts.append(nm)
    if ocr_text:
        ocr_lines = []
        for ln in (ocr_text or "").splitlines()[:160]:
            ln = ln.strip()
            if not ln or _is_noise_textline(ln): continue
            if len(_normalize(re.sub(r"\d+", "", ln))) < 2: continue
            ocr_lines.append(ln)
            if len(ocr_lines) >= 40: break
        if ocr_lines: parts.append(" | ".join(ocr_lines)[:2000])
    return " | ".join(parts)[:4000]

def resolve_record_tags(*, items, hospital_name=None, ocr_text=None, record_thresh=125, item_thresh=120, max_tags=8, return_item_tags=True, **kw):
    query = _build_record_query(items or [], hospital_name, ocr_text=ocr_text)
    if not query.strip(): return {"tags": [], "itemCategoryTags": [], "evidence": {"policy": "catalog", "reason": "empty_query"}}
    scored = []
    for tag in TAG_CATALOG:
        s, ev = _match_score(tag, query)
        if s > 0: scored.append((tag["code"], s, ev))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = []; evidence = {"policy": "catalog", "query": query[:600], "recordThresh": int(record_thresh), "itemThresh": int(item_thresh), "candidates": []}
    for code, score, ev in scored[:30]:
        evidence["candidates"].append({"code": code, "score": score, **(ev or {})})
        if score >= int(record_thresh) and code not in picked:
            if code == "etc_other": continue
            picked.append(code)
        if len(picked) >= int(max_tags): break
    if not picked:
        for code, score, _ in scored[:50]:
            if code == "etc_other" and score >= 90: picked.append("etc_other"); break
    item_tags = []
    if return_item_tags:
        for idx, it in enumerate((items or [])[:250]):
            nm = (it.get("itemName") or "").strip()
            if not nm: continue
            bc = None; bs = 0; be = {}
            for tag in TAG_CATALOG:
                s, ev = _match_score(tag, nm)
                if s > bs: bs = s; bc = tag["code"]; be = ev or {}
            if bc and bs >= int(item_thresh):
                if bc == "etc_other": continue
                item_tags.append({"idx": idx, "itemName": nm, "categoryTag": bc, "score": bs, **(be or {})})
    return {"tags": picked, "itemCategoryTags": item_tags, "evidence": evidence}

