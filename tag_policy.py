# tag_policy.py (PetHealth+)
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

TAG_CATALOG: List[Dict[str, Any]] = [
    # === 검사 — 영상 ===
    {"code": "exam_xray", "group": "exam", "aliases": [
        "x-ray","xray","xr","x ray","radiograph","radiology","radiographic",
        "엑스레이","방사선","x선","x선촬영","치아 방사선","dental xray","dental x-ray",
        "방사선촬영","방사선검사","흉부방사선","복부방사선","흉부촬영","복부촬영",
        "사지촬영","골반촬영","척추촬영","두부촬영","전신촬영","vd","dv","lateral",
    ]},
    {"code": "exam_ct", "group": "exam", "aliases": [
        "ct","ct촬영","ct검사","ct조영","ct scan","computed tomography",
        "컴퓨터단층촬영","씨티","조영ct","contrast ct","ct스캔",
    ]},
    {"code": "exam_mri", "group": "exam", "aliases": [
        "mri","mri촬영","mri검사","mri scan","magnetic resonance","자기공명","엠알아이",
    ]},
    {"code": "exam_endoscope", "group": "exam", "aliases": [
        "내시경","endoscopy","endoscope","gastroscopy","위내시경","장내시경",
        "기관지내시경","비강내시경","방광경","arthroscopy","관절경","내시경검사",
        "내시경생검","이물내시경","식도내시경","직장내시경","비내시경",
    ]},
    {"code": "exam_biopsy", "group": "exam", "aliases": [
        "생검","조직검사","biopsy","tissue biopsy","fna","fine needle",
        "세침흡인","병리검사","조직병리","histopathology","pathology",
        "세포검사","세포학","세포진","세포학적검사",
    ]},
    # === 초음파 세분화 ===
    {"code": "exam_echo", "group": "exam", "aliases": [
        "심장초음파","심초음파","심장 초음파","cardiac ultrasound","cardiac us",
        "echocardiogram","echocardiography","echo","ecco","심장 에코","에코",
        "심장us","심장 us",
    ]},
    {"code": "exam_us_abdomen", "group": "exam", "aliases": [
        "복부초음파","복부 초음파","abdominal ultrasound","abdominal us",
        "abd us","abd sono","복부에코","복부us","abd초음파",
    ]},
    {"code": "exam_us_general", "group": "exam", "aliases": [
        "ultrasound","sono","sonography","us","초음파","초음파검사",
        "근골격초음파","경부초음파","갑상선초음파","방광초음파","신장초음파",
    ]},
    # === 혈액검사 세분화 ===
    {"code": "exam_blood_cbc", "group": "exam", "aliases": [
        "cbc","complete blood count","혈구검사","혈구","CBC검사","blood count",
        "혈구계산","혈구분석","전혈구","전혈구검사","혈구측정",
    ]},
    {"code": "exam_blood_chem", "group": "exam", "aliases": [
        "chemistry","biochem","biochemistry","생화학","생화학검사",
        "간수치","신장수치","간기능","신장기능","chemistry panel","chem",
        "간기능검사","신기능검사","신장기능검사","간패널","chem10","chem17","chem12",
        "간효소","alt","ast","bun","creatinine","크레아티닌","알부민","albumin",
        "총단백","빌리루빈","bilirubin","알피","alp","ggt","gpt","got",
    ]},
    {"code": "exam_blood_general", "group": "exam", "aliases": [
        "blood test","profile","혈액","혈액검사","피검사","피검","blood work",
        "bloodwork","혈액프로필","혈액 프로필","혈액 패널","혈액종합","종합혈액",
        "혈액정밀","idexx","vcheck","catalyst","프로켐","prochem","혈액검사비",
    ]},
    {"code": "exam_blood_type", "group": "exam", "aliases": [
        "혈액형","혈액형검사","blood type","blood typing","crossmatch","교차시험",
        "교차적합","혈액형판정",
    ]},
    {"code": "exam_coagulation", "group": "exam", "aliases": [
        "응고검사","응고","coagulation","pt","aptt","pt/aptt","프로트롬빈",
        "피브리노겐","fibrinogen","응고시간","출혈시간",
        "d-dimer","d dimer","ddimer","디다이머","혈전","thrombin","트롬빈",
        "혈전검사","응고인자","혈액응고",
    ]},
    {"code": "exam_electrolyte", "group": "exam", "aliases": [
        "전해질","전해질검사","electrolyte","나트륨","칼륨","칼슘",
        "calcium","phosphorus","인","마그네슘","magnesium","na","cl","전해질패널",
    ]},
    {"code": "exam_crp", "group": "exam", "aliases": [
        "crp","c-reactive protein","염증수치","염증검사","염증마커","crp검사","씨알피",
        "saa","혈청아밀로이드",
    ]},
    # === 심장검사 세분화 ===
    {"code": "exam_ecg", "group": "exam", "aliases": [
        "ecg","ekg","심전도","electrocardiogram","electrocardiography","심전도검사","12 lead",
    ]},
    {"code": "exam_heart_general", "group": "exam", "aliases": [
        "cardiac","heart","심장검사","심장 검사","심장","heart check","cardiac exam","심장평가",
    ]},
    # === 호르몬 ===
    {"code": "exam_hormone", "group": "exam", "aliases": [
        "호르몬","호르몬검사","hormone","hormone test","t4","t3","ft4","tsh",
        "갑상선","갑상선검사","thyroid","cortisol","코르티솔","acth","부신","부신검사","adrenal",
        "갑상선기능","갑상선호르몬","에스트로겐","프로게스테론","testosterone","테스토스테론",
        "인슐린","insulin","성장호르몬","growth hormone",
    ]},
    # === 기타 검사 ===
    {"code": "exam_lab_panel", "group": "exam", "aliases": [
        "lab panel","screening","health check","종합검사","종합검진","패널검사",
        "건강검진","건강검사","정기검진","종합건강","기본검진","annual check",
        "예방검진","스크리닝","건강진단",
        "검사료","검사비","검사비용","수가적","종합검사료",
    ]},
    {"code": "exam_urine", "group": "exam", "aliases": [
        "urinalysis","ua","urine test","요검사","소변검사","뇨검사","요분석",
        "요비중","upc","소변배양","urine culture","요침사","요시험지",
    ]},
    {"code": "exam_fecal", "group": "exam", "aliases": [
        "fecal","stool test","대변검사","분변검사","배변검사","변검사",
        "분변부유","분변직접","분변도말","기생충검사",
    ]},
    {"code": "exam_fecal_pcr", "group": "exam", "aliases": [
        "fecal pcr","stool pcr","gi pcr","panel pcr","대변pcr","대변 pcr","분변 pcr",
        "gi패널","장패널","소화기패널",
    ]},
    {"code": "exam_sdma", "group": "exam", "aliases": [
        "sdma","symmetrical dimethylarginine","idexx sdma","신장마커","신장검사",
        "신기능마커","조기신장",
    ]},
    {"code": "exam_probnp", "group": "exam", "aliases": [
        "probnp","pro bnp","pro-bnp","ntprobnp","nt-probnp","bnp","cardiopet",
        "심장마커","프로비엔피","심장바이오마커",
    ]},
    {"code": "exam_fructosamine", "group": "exam", "aliases": [
        "fructosamine","fru","glycated albumin","ga","프럭토사민","당화알부민",
    ]},
    {"code": "exam_glucose_curve", "group": "exam", "aliases": [
        "glucose curve","blood glucose curve","bg curve","혈당곡선","혈당커브","연속혈당",
        "혈당모니터링","glucose monitoring","혈당측정",
    ]},
    {"code": "exam_blood_gas", "group": "exam", "aliases": [
        "blood gas","bga","bgas","i-stat","istat","혈액가스","가스분석",
        "동맥혈가스","정맥혈가스","abg","vbg",
    ]},
    {"code": "exam_allergy", "group": "exam", "aliases": [
        "allergy test","ige","atopy","알러지검사","알레르기검사","알러지","알레르기",
        "아토피검사","아토피","알레르겐","allergen","식이알러지","환경알러지",
    ]},
    {"code": "exam_eye", "group": "exam", "aliases": [
        "schirmer","fluorescein","iop","ophthalmic exam","안압","형광염색",
        "안과검사","안과","눈검사","안압측정","눈물량검사","쉬르머","안저검사",
        "세극등","slit lamp","슬릿램프","안구초음파","망막검사",
    ]},
    {"code": "exam_skin", "group": "exam", "aliases": [
        "skin scraping","cytology","fungal test","malassezia","피부스크래핑",
        "피부검사","진균","곰팡이","말라세지아","피부사상균","피부도말",
        "피부세포","tape cytology","테이프도말","우드램프","wood lamp","dtm",
        "피부배양","진균배양","피부조직검사",
    ]},
    {"code": "exam_ear", "group": "exam", "aliases": [
        "귀검사","이경","검이경","이경검사","otoscope","otoscopy","이도검사",
        "귀내시경","이도내시경","ear exam","ear examination","ear check",
        "검사-귀","귀-set","귀set","이경+현미경","귀검진","이도검진",
        "귀진찰","외이도검사","이개검사",
        "검사귀set","검사귀","귀현미경","귀현미경도말",
    ]},
    {"code": "exam_microscope", "group": "exam", "aliases": [
        "현미경","현미경검사","현미경도말","도말검사","도말","microscopy",
        "microscope","현미경관찰","도말표본","smear","smear test","도말시험",
        "현미경분석","현미경판독",
    ]},
    # === 감염병 검사 (snap test 등) ===
    {"code": "exam_snap_test", "group": "exam", "aliases": [
        "snap","snap test","snap4dx","4dx","snap fiv","snap felv","fiv","felv",
        "fiv/felv","심장사상충검사","hw검사","hw test","heartworm test","항원검사",
        "항체검사","pcr","pcr검사","파보","디스템퍼","파보바이러스",
        "parvovirus","distemper","panleukopenia","범백","코로나검사","giardia","지아디아",
        "snap combo","combo test","키트검사","신속검사","rapid test",
        "감염병검사","감염검사","전염병검사","바이러스검사","ehrlichia","에를리히아",
        "anaplasma","아나플라즈마","lyme","라임","라임병","리케치아","rickettsia",
        "cpl","spec cpl","cpli","감염병","감염","전염병",
    ]},
    # === 기본 신체검사 / 활력징후 ===
    {"code": "exam_vitals", "group": "exam", "aliases": [
        "혈압","혈압측정","혈압검사","blood pressure","bp측정","bp",
        "체온","체온측정","체중","체중측정","몸무게","체중계",
        "심박","심박수","맥박","활력징후","vitals","vital sign",
        "신체검사","이학적검사","physical exam","physical examination","pe",
        "기본검사","기본신체","바이탈","바이탈사인","bcs","체형평가",
    ]},
    # === 예방접종 ===
    {"code": "vaccine_comprehensive", "group": "vaccine", "aliases": [
        "dhpp","dhppi","dhlpp","5-in-1","6-in-1","fvrcp","combo vaccine",
        "종합백신","혼합백신","5종백신","6종백신","5종접종","6종접종",
        "5종","6종","7종","8종","종합접종","종합예방접종","예방접종",
        "혼합예방접종","기본접종","기본예방접종","puppy shot","kitty shot",
        "5종혼합","6종혼합","종합예방","종합주사","dhppl","dapp","접종",
    ]},
    {"code": "vaccine_rabies", "group": "vaccine", "aliases": [
        "rabies","rabbies","rabie","rabis","rab","rabi","ra","rabies vac","rabies vaccine",
        "광견병","광견","광견병백신","광견병접종","광견병주사",
    ]},
    {"code": "vaccine_kennel", "group": "vaccine", "aliases": [
        "kennel cough","bordetella","켄넬코프","기관지염백신","보르데텔라",
        "kennel","kc","기관지보르데텔라",
    ]},
    {"code": "vaccine_corona", "group": "vaccine", "aliases": [
        "corona","coronavirus","corona enteritis","코로나","코로나장염",
        "코로나바이러스","코로나백신","코로나접종",
    ]},
    {"code": "vaccine_lepto", "group": "vaccine", "aliases": [
        "lepto","leptospirosis","leptospira","lepto2","lepto4","l2","l4",
        "렙토","렙토2","렙토4","렙토스피라",
    ]},
    {"code": "vaccine_parainfluenza", "group": "vaccine", "aliases": [
        "parainfluenza","cpiv","cpi","pi","파라인플루엔자","파라인","파라",
    ]},
    {"code": "vaccine_fip", "group": "vaccine", "aliases": [
        "fip","primucell","feline infectious peritonitis","전염성복막염","복막염",
        "복막염백신",
    ]},
    # === 예방약 ===
    {"code": "prevent_heartworm", "group": "preventive_med", "aliases": [
        "heartworm","hw","dirofilaria","heartgard","심장사상충","하트가드",
        "넥스가드스펙트라","simparica trio","revolution","심장사상충예방",
        "하트가드플러스","하트가드정","인터셉터","interceptor","밀베마이신",
        "프로하트","proheart","모시덱틴","moxidectin","셀라멕틴","selamectin",
        "사상충","사상충예방","사상충약",
    ]},
    {"code": "prevent_external", "group": "preventive_med", "aliases": [
        "flea","tick","bravecto","nexgard","frontline","revolution",
        "벼룩","진드기","외부기생충","넥스가드","브라벡토","프론트라인",
        "어드밴티지","advantage","advantix","세레스토","seresto","외부구충",
        "레볼루션","크레델리오","credelio","simparica","심파리카","벼룩약","진드기약",
        "스팟온","spot-on","spot on","피프로닐","fipronil","퍼메트린","permethrin",
        "이미드캐브","이미독스","imidox",
    ]},
    {"code": "prevent_deworming", "group": "preventive_med", "aliases": [
        "deworm","deworming","drontal","milbemax","fenbendazole","panacur",
        "구충","구충제","내부기생충","드론탈","밀베맥스","펜벤다졸",
        "내부구충","회충","촌충","편충","구충약","기생충약","프라지콴텔","praziquantel",
        "말라론","malarone","아토바쿠온","atovaquone","바베시아","babesia",
        "이미도카브","imidocarb","이미드캐브","항원충","항기생충",
        "메트로니다졸","트리코모나스","콕시듐","coccidia","기아르디아","giardia",
    ]},
    # === 처방약 ===
    {"code": "medicine_antibiotic", "group": "medicine", "aliases": [
        "antibiotic","abx","amoxicillin","clavamox","augmentin","cephalexin",
        "convenia","doxycycline","metronidazole","baytril","항생제",
        "아목시실린","세팔렉신","독시사이클린","메트로니다졸","엔로플록사신",
        "enrofloxacin","marbofloxacin","마보플록사신","항생","클라바목스",
        "세파졸린","cefazolin","아지스로마이신","azithromycin","린코마이신","lincomycin",
        "클린다마이신","clindamycin",
        "지스로맥스","zithromax","세프트리악손","ceftriaxone","세파클러","cefaclor",
        "바이트릴","오비악스","obiaxe","리팜피신","rifampicin",
        "세프포독심","cefpodoxime","세팔로스포린","cephalosporin",
        "프라목스","pradofloxacin","프라도플록사신","오르비플록사신","orbax",
    ]},
    {"code": "medicine_anti_inflammatory", "group": "medicine", "aliases": [
        "nsaid","anti-inflammatory","meloxicam","metacam","carprofen","rimadyl",
        "onsior","galliprant","소염","소염제","멜록시캄","카프로펜","온시올","갈리프란트",
        "소염진통","소염진통제","firocoxib","피록시캄","로베나콕시브","robenacoxib",
    ]},
    {"code": "medicine_painkiller", "group": "medicine", "aliases": [
        "analgesic","tramadol","gabapentin","buprenorphine","진통","진통제",
        "트라마돌","가바펜틴","부프레노르핀","펜타닐","fentanyl","모르핀","morphine",
        "통증관리","통증치료","pain","painkiller",
    ]},
    {"code": "medicine_steroid", "group": "medicine", "aliases": [
        "steroid","prednisone","prednisolone","dexamethasone","스테로이드",
        "프레드니손","프레드니솔론","덱사메타손","부데소니드","budesonide",
        "트리암시놀론","triamcinolone","코르티코","corticosteroid",
    ]},
    {"code": "medicine_gi", "group": "medicine", "aliases": [
        "famotidine","pepcid","omeprazole","sucralfate","cerenia","ondansetron",
        "reglan","위장약","구토","설사","장염","위장관","소화제","지사제",
        "파모티딘","오메프라졸","세레니아","수크랄페이트","메토클로프라미드",
        "정장제","프로바이오틱스","probiotics","유산균","소화효소",
        "란소프라졸","lansoprazole","판토프라졸","pantoprazole","라니티딘","ranitidine",
        "마로필란트","maropitant","온단세트론","돔페리돈","domperidone",
        "설파살라진","sulfasalazine","미소프로스톨","misoprostol",
    ]},
    {"code": "medicine_eye", "group": "medicine", "aliases": [
        "eye drop","ophthalmic","tobramycin","ofloxacin","cyclosporine",
        "안약","점안","결막염","각막","점안액","안연고","눈약","안과약",
        "인공눈물","타크로리무스","tacrolimus","tobra","겐타마이신","gentamicin",
    ]},
    {"code": "medicine_ear", "group": "medicine", "aliases": [
        "ear drop","otic","otitis","otomax","surolan","posatex","easotic",
        "귀약","이염","외이염","점이액","귀연고","이도약","중이염","이개",
        "오토맥스","수로란","포사텍스","이소틱",
    ]},
    {"code": "medicine_skin", "group": "medicine", "aliases": [
        "dermatitis","chlorhexidine","ketoconazole","miconazole","피부약","피부염",
        "클로르헥시딘","케토코나졸","미코나졸","피부연고","연고","외용제","외용약",
        "샴푸","medicated shampoo","약용샴푸","항진균제","antifungal",
    ]},
    {"code": "medicine_allergy", "group": "medicine", "aliases": [
        "apoquel","cytopoint","cetirizine","zyrtec","benadryl","알러지","알레르기","가려움",
        "아포퀠","사이토포인트","세티리진","항히스타민","antihistamine",
        "아토피치료","아토피약","오클라시티닙","oclacitinib",
    ]},
    {"code": "medicine_oral", "group": "medicine", "aliases": [
        "내복약","경구약","먹는약","oral","oral med","oral medication","po","per os",
        "처방약","약값","처방료","조제료","약제비","약제","투약","투약료","복약",
        "약비","처방전","처방","조제","약국","약","medicine","medication","med",
        "복용","경구투여",
        "고가약물","고가약","약물","캡슐조제","캡슐조제료","내복약조제","내복약조제료",
        "조제비","약제료","처방조제","조제약",
        "zonisamide","leflunomide","gabapentin","phenobarbital","prednisolone",
        "cyclosporine","oclacitinib","apoquel","atopica","metronidazole",
        "amoxicillin","cephalexin","enrofloxacin","doxycycline","clindamycin",
        "mycophenolate","마이코페놀레이트","tacrolimus","타크로리무스",
        "chlorambucil","클로람부실","vincristine","빈크리스틴",
        "piroxicam","피록시캄","토세라닙","toceranib","palladia",
        "실데나필","sildenafil","피모벤단","pimobendan","베트메딘","vetmedin",
        "암로디핀","amlodipine","베나제프릴","benazepril","포르테코르","fortekor",
        "에날라프릴","enalapril","텔미사르탄","telmisartan","세미트라","semintra",
        "레보티록신","levothyroxine","솔록신","soloxine","갑상선약",
    ]},
    # === 처치/수액/응급/수혈 ===
    {"code": "care_injection", "group": "checkup", "aliases": [
        "inj","injection","shot","sc","im","iv","주사","주사제","피하주사",
        "근육주사","정맥주사","주사료","주사비","주사처치",
        "피하","근육","정맥","주사투여","주사비용","예방주사",
    ]},
    {"code": "care_fluid", "group": "checkup", "aliases": [
        "수액","링거","iv fluid","fluid therapy","수액처치","수액치료",
        "피하수액","정맥수액","lactated ringer","생리식염수","normal saline",
        "수액세트","링거액","hartmann","수액비","수액료",
        "수액제","링거세트","수액팩","보충수액","유지수액","lr","ns",
    ]},
    {"code": "care_transfusion", "group": "checkup", "aliases": [
        "수혈","transfusion","blood transfusion","전혈","packed rbc","혈장",
        "plasma","수혈비","혈액제제","fresh frozen plasma","ffp",
        "적혈구농축액","prbc","혈소판","platelet","수혈료",
        "농축적혈구","수혈모니터링","수혈반응","크로스매칭","crossmatching",
        "혈액은행","blood bank","수혈전검사","수혈키트","수혈세트",
    ]},
    {"code": "care_oxygen", "group": "checkup", "aliases": [
        "산소","산소치료","산소방","산소텐트","oxygen","oxygen therapy","o2",
        "산소공급","산소케이지","산소실","네뷸라이저","nebulizer","네블라이저",
    ]},
    {"code": "care_emergency", "group": "checkup", "aliases": [
        "응급","응급처치","응급진료","emergency","ER","응급비","응급진료비",
        "야간진료","야간응급","심폐소생","CPR","cpr","야간","야간비",
        "야간진료비","공휴일진료","휴일진료","특수시간",
    ]},
    {"code": "care_catheter", "group": "checkup", "aliases": [
        "카테터","도뇨관","유치도뇨관","catheter","urinary catheter",
        "정맥카테터","iv catheter","도뇨","방광세척","요도카테터",
        "방광천자","cystocentesis","복강천자","흉강천자","배액","드레인","drain",
    ]},
    {"code": "care_procedure_fee", "group": "checkup", "aliases": [
        "procedure fee","treatment fee","handling fee","처치료","시술료",
        "처치비","시술비","처치","시술","관리료","관리비","관리",
    ]},
    {"code": "care_dressing", "group": "checkup", "aliases": [
        "dressing","bandage","gauze","wrap","disinfection","드레싱","붕대",
        "거즈","소독","세척","상처처치","상처관리","상처소독","창상처치",
        "반창고","테이핑","taping","스플린트","splint","부목",
    ]},
    {"code": "care_anal_gland", "group": "checkup", "aliases": [
        "항문낭","항문낭짜기","항문낭세척","anal gland","anal sac","항문선","항문낭 압출",
        "항문","항문낭압출","항문낭배출",
    ]},
    {"code": "care_ear_flush", "group": "checkup", "aliases": [
        "귀세척","이도세척","ear flush","ear cleaning","ear irrigation","귀청소",
        "귀관리","이도관리","외이도세정","외이도세척","이도세정","귀세정",
        "ear wash","외이세정","외이세척",
    ]},
    # === 입원 ===
    {"code": "hospitalization", "group": "checkup", "aliases": [
        "입원","입원비","입원료","hospitalization","hospital stay","icu",
        "중환자","중환자실","집중치료","입원관리","입원케어","케이지","cage",
        "입원실","입원1일","입원관찰","데이케어","day care","반일입원","1일입원",
        "입원치료","관찰입원","모니터링입원","입원모니터링",
    ]},
    # === 수술 세분화 ===
    {"code": "surgery_general", "group": "surgery", "aliases": [
        "surgery","operation","수술","봉합","마취","마취료","마취-호흡",
        "흡입마취","전신마취","국소마취","수술비","수술료","수술재료","수술재료비","봉합사",
        "수술준비","수술세트","수술전처치","수술후관리","마취관리","마취감시",
        "마취모니터링","마취유지","삽관","기관삽관","수술후처치","수술처치",
    ]},
    {"code": "surgery_spay_neuter", "group": "surgery", "aliases": [
        "중성화","spay","neuter","castration","ovariohysterectomy","ohe",
        "수컷 중성화","암컷 중성화","자궁적출","난소적출","고환적출","중성화수술",
        "불임수술","피임수술","거세","난소자궁적출","ovh","잠복고환",
        "cryptorchid","복강내고환",
    ]},
    {"code": "surgery_tumor", "group": "surgery", "aliases": [
        "종양","종양제거","종양수술","tumor","tumor removal","mass removal",
        "mass","lump","혹제거","혹","종괴","종괴제거","절제","절제술",
        "피부종양","유선종양","림프종","lymphoma","비만세포종","mct",
    ]},
    {"code": "surgery_foreign_body", "group": "surgery", "aliases": [
        "이물","이물제거","이물수술","foreign body","foreign body removal",
        "이물질","위절개","장절개","gastrotomy","enterotomy","이물적출",
    ]},
    {"code": "surgery_cesarean", "group": "surgery", "aliases": [
        "제왕절개","cesarean","c-section","caesarean","제왕","제왕절개수술",
    ]},
    {"code": "surgery_hernia", "group": "surgery", "aliases": [
        "탈장","탈장수술","hernia","hernia repair","회음부탈장","서혜부탈장",
        "perineal hernia","inguinal hernia","배꼽탈장","umbilical hernia",
        "횡격막탈장","diaphragmatic hernia",
    ]},
    {"code": "surgery_eye", "group": "surgery", "aliases": [
        "안과수술","eye surgery","체리아이","cherry eye","백내장","백내장수술",
        "cataract","안구적출","enucleation","안구","눈수술","각막수술",
        "안검수술","entropion","ectropion","내안각","안검내반","안검외반",
    ]},
    {"code": "surgery_orthopedic", "group": "surgery", "aliases": [
        "정형수술","정형외과","orthopedic surgery","fracture repair","골절",
        "골절수술","골절정복","핀삽입","핀고정","플레이트","plate","뼈수술",
        "관절수술","십자인대","ccl","tplo","tta","관절경수술",
    ]},
    # === 치과 ===
    {"code": "dental_scaling", "group": "dental", "aliases": [
        "scaling","dental cleaning","tartar","스케일링","치석","치석제거","스켈링",
        "dental prophylaxis","치과스케일링","초음파스케일링","폴리싱","polishing",
    ]},
    {"code": "dental_extraction", "group": "dental", "aliases": [
        "extraction","dental extraction","발치","발치술","치아발치","tooth extraction",
        "발거","치아발거","치아제거","잔근","잔근제거","영구치발치","유치발치",
    ]},
    {"code": "dental_treatment", "group": "dental", "aliases": [
        "잇몸","잇몸치료","치주","치주치료","periodontal","gingival","불소","불소도포",
        "fluoride","치아치료","root canal","신경치료","치과","치과치료","구강","구강검진",
        "치과진료","구강관리","치과처치","구강처치","치은","치은절제","gingivectomy",
    ]},
    # === 관절/정형 ===
    {"code": "ortho_patella", "group": "orthopedic", "aliases": [
        "mpl","lpl","patella","patellar luxation","슬개골탈구","슬탈","파행",
        "슬개골","무릎","슬개골수술","슬관절",
    ]},
    {"code": "ortho_arthritis", "group": "orthopedic", "aliases": [
        "arthritis","oa","osteoarthritis","관절염","퇴행성관절",
        "관절","관절관리","관절영양","관절보조제","글루코사민","glucosamine",
        "콘드로이틴","chondroitin","관절주사","관절치료",
    ]},
    # === 재활 ===
    {"code": "rehab_therapy", "group": "checkup", "aliases": [
        "재활","재활치료","물리치료","rehabilitation","physical therapy","physio",
        "수중치료","수중런닝머신","underwater treadmill","hydrotherapy",
        "레이저","레이저치료","laser therapy","cold laser",
        "침","침치료","acupuncture","전기침","electroacupuncture",
        "초음파치료","ultrasound therapy","체외충격파","eswt","도수치료",
        "운동치료","재활운동","밸런스볼","수영치료",
    ]},
    # === 마이크로칩 ===
    {"code": "microchip", "group": "etc", "aliases": [
        "마이크로칩","microchip","chip","내장형칩","동물등록","동물 등록",
        "pet registration","칩삽입","반려동물등록","인식칩",
    ]},
    # === 안락사/장례 ===
    {"code": "euthanasia", "group": "etc", "aliases": [
        "안락사","euthanasia","peaceful passing","임종","임종처치","안락사처치",
    ]},
    {"code": "funeral", "group": "etc", "aliases": [
        "장례","화장","cremation","funeral","pet funeral","장례비","화장비",
        "반려동물장례","개별화장","합동화장","유골","납골","장례식","추모",
    ]},
    # === 기타 ===
    {"code": "care_e_collar", "group": "etc", "aliases": [
        "e-collar","ecollar","cone","elizabethan collar","넥카라","넥칼라",
        "엘리자베스카라","보호카라","보호대","깔대기","목카라","목칼라",
        "넥카라 소","넥카라 중","넥카라 대","넥칼라 소","넥칼라 중","넥칼라 대",
    ]},
    {"code": "care_prescription_diet", "group": "etc", "aliases": [
        "prescription diet","rx diet","therapeutic diet","처방식","처방사료",
        "병원사료","hill's","hills","royal canin","k/d","c/d","i/d","z/d",
        "처방전용사료","의료용사료","치료식","치료용사료","로얄캐닌","힐스",
        "처방식이","처방용사료","처방캔","처방파우치","처방간식",
        "로얄","로얄독","로얄캣","하이포알러지","하이포알러제닉","hypoallergenic",
        "스몰독","미디엄독","라지독","가수분해","hydrolyzed",
        "소화기케어","유리너리","레날","hepatic","스킨케어","뉴트리드",
        "sensitivity control","sensitivity","세인시티비티",
        "로얄독하이포","로얄독하이포알러지","독하이포알러지","하이포알러지스몰",
        "인도어","라이프스테이지","스킨앤코트","skin&coat",
        "로얄 독 하이포","독 하이포 알러지",
    ]},
    {"code": "supply_food", "group": "etc", "aliases": [
        "사료","건사료","습식사료","캔","캔사료","간식","간식류","트릿","treat",
        "food","pet food","dog food","cat food","kibble","wet food","dry food",
        "사료비","사료구매","사료대","파우치","토퍼","습식","건식","자연식",
        "생식","화식","수제사료","수제간식",
        "일반사료","일반간식",
    ]},
    {"code": "supply_supplement", "group": "etc", "aliases": [
        "영양제","보조제","supplement","nutritional supplement","비타민","vitamin",
        "오메가3","omega","유산균","프로바이오틱","프리바이오틱","관절영양제",
        "관절보조","피부영양제","면역보조","영양보조","건강보조식품","건강보조",
        "종합영양제","멀티비타민","눈영양제","간영양제","신장영양제",
        "뉴트리플러스","nutri","뉴트리","헤파틱","hepatic support","데나마린","denamarin",
        "사밀","samil","실리마린","silymarin","밀크시슬","간보호제",
        "레날어드밴스","renal advance","아조딜","azodyl","이파키틴","ipakitine",
        "코세퀸","cosequin","다스퀸","dasuquin","항산화제","antioxidant",
    ]},
    {"code": "supply_goods", "group": "etc", "aliases": [
        "용품","pet supply","pet supplies","펫용품","반려동물용품",
        "장난감","toy","리드줄","목줄","하네스","harness","leash","collar",
        "식기","밥그릇","물그릇","배변패드","패드","배변","pet pad",
        "케이지","캐리어","carrier","이동장","방석","침대","pet bed",
    ]},
    {"code": "checkup_general", "group": "checkup", "aliases": [
        "checkup","consult","opd","진료","상담","초진","재진","진찰",
        "진료비","초진료","재진료","진찰료","상담료","진료비용","진찰비",
        "consultation","진료료","외래","진료상담","내원","방문",
        "일반진료","기본진료","진료비","외래진료","통원",
    ]},
    {"code": "grooming_basic", "group": "grooming", "aliases": [
        "grooming","bath","trim","미용","목욕","클리핑","발톱","발톱깎기",
        "nail trim","nail clip","귀털","귀털제거","위생미용","부분미용",
        "발바닥","발바닥털","패드관리","항문주위","위생컷",
    ]},
    {"code": "etc_fee", "group": "etc", "aliases": [
        "재료비","재료대","소모품","소모품비","위생비","위생용품","일회용품",
        "material fee","supply fee","수납","수납비","부대비","부대비용",
        "소모품대","위생재료","일회용","일회용재료","기타재료",
    ]},
    {"code": "etc_discount", "group": "etc", "aliases": [
        "할인","절사할인","절사","단수할인","단수절사","단수조정","절사조정",
        "할인금액","할인액","환급","감면","조정금액","조정","금액조정",
        "discount","절사 할인","단수 할인","반올림할인","끝전할인",
        "쿠폰할인","쿠폰","회원할인","보호자할인","다두할인","소개할인",
    ]},
    {"code": "etc_other", "group": "etc", "aliases": ["기타","etc","other","기타비용","기타비"]},
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

# =========================================================
# ✅ 영수증 항목명 → 매칭 가능한 변형 자동 생성 (경우의수 최소화)
# =========================================================

# 1) 접두어: "검사 - ", "치과(별도 추가)", "수술:", "처방 " 등
_PREFIX_WORDS = [
    "검사","처치","수술","치과","예방","처방","주사","약제","진료","기본",
    "추가","별도","특수","응급","야간","일반","정밀","정기","긴급","외래",
    "입원","퇴원","재진","초진","당일","익일","심화","종합","기초","확인",
    "재검","추적","경과","관찰","상담","의뢰","협진","세부","상세","간이",
    "혈액","영상","임상","병리","처방전","처방전달","약국","조제","약국조제",
    "수납","청구","보험","비보험","급여","비급여","본인부담","감면",
    "소동물","대동물","강아지","고양이","소형견","중형견","대형견",
    "cat","dog","canine","feline",
    # 영수증 카테고리 접두어 (구체 항목이 우선되도록)
    "초음파","방사선","엑스레이","혈액검사","소변","대변","분변",
    "예방접종","접종","백신","수액","링거","내복","외용","안과","피부과",
    "정형","재활","미용","위생",
]
_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(re.escape(w) for w in _PREFIX_WORDS) + r")\s*[-·:()（）\s]*",
    re.IGNORECASE,
)

# 2) 접미어: 괄호, 체중, 수량, 시간, 횟수 등
_SUFFIX_PATTERNS = [
    r"\s*\(.*?\)\s*$",                          # (10kg이하), (~5kg)
    r"\s*（.*?）\s*$",                            # 전각 괄호
    r"\s*\[.*?\]\s*$",                           # [참고], [추가]
    r"\s*[-·:]\s*\d+\s*$",                       # 끝에 숫자
    r"\s+\d+\s*kg.*$",                           # 10kg이하
    r"\s+~?\d+\s*kg.*$",                         # ~10kg
    r"\s+\d+\s*[회건개번차일분시].*$",            # 1회, 2건, 3개, 1일
    r"\s+[0-9]+\s*시간.*$",                      # 1시간 미만
    r"\s+\d+\s*[mM][lL].*$",                     # 100ml
    r"\s+\d+\s*[cC][cC].*$",                     # 100cc
    r"\s+x\s*\d+.*$",                            # x3, x 2
    r"\s+\d+[차].*$",                            # 1차, 2차
    r"\s*#\d+.*$",                               # #1, #2
    r"\s+\d+\s*[tT].*$",                         # 1T, 2T (정제 수량)
    r"\s+\d+\s*[cC].*$",                         # 1C, 2C (캡슐 수량)
    r"\s+\d+\s*[eE][aA].*$",                     # 1ea, 2EA
    r"\s+\d+\s*매.*$",                           # 1매, 2매
    r"\s+\d+\s*[pP][aA][cC][kK].*$",            # 1pack
    r"\s+\d+\s*세트.*$",                         # 1세트
    r"\s+\d+\s*[bB][oO][xX].*$",                # 1box
    r"\s+이상$",                                  # 이상
    r"\s+이하$",                                  # 이하
    r"\s+미만$",                                  # 미만
    r"\s+초과$",                                  # 초과
]
_SUFFIX_RE = re.compile("|".join(_SUFFIX_PATTERNS))

# 3) 분리 기호: 하이픈, 슬래시, 플러스, 쉼표, 공백+대시
_SPLIT_RE = re.compile(r"\s*[-–—·/+,&]\s*")

# 4) 무의미한 수식어 (제거해도 의미 유지)
_FILLER_WORDS = re.compile(
    r"(?:별도|추가|기본|일반|특수|정밀|간이|긴급|당일|익일|재|소형|중형|대형"
    r"|소동물|대동물|강아지|고양이|dog|cat|canine|feline"
    r"|좌|우|양측|단측|전신|국소|부분|전체|좌측|우측|앞|뒤|상|하"
    r"|경구|외용|주사용|주사제|제제|액제|정제|캡슐|연고|크림|용액"
    r"|1차|2차|3차|4차|5차|1회|2회|3회)\s*",
    re.IGNORECASE,
)


def _generate_variants(s: str) -> List[str]:
    """
    영수증 항목명 하나로부터 매칭 가능한 변형 문자열들을 자동 생성.
    예: "치과(별도 추가)스케일링 추가+폴리싱 (-10kg이하)"
    → ["치과(별도 추가)스케일링 추가+폴리싱 (-10kg이하)",
       "스케일링 추가+폴리싱",
       "스케일링", "폴리싱",
       ...]
    """
    raw = (s or "").strip()
    if not raw:
        return []

    seen = set()
    variants = []

    def _add(v):
        v = v.strip()
        if v and len(v) >= 2 and v not in seen:
            seen.add(v)
            variants.append(v)

    # 원본
    _add(raw)

    # 접두어 제거 (반복: "검사 - 혈액 - CBC" → "혈액 - CBC" → "CBC")
    cur = raw
    for _ in range(4):
        prev = cur
        cur = _PREFIX_RE.sub("", cur).strip()
        _add(cur)
        if cur == prev:
            break
    no_prefix = cur

    # 접미어 제거 (반복 적용)
    for base in [raw, no_prefix]:
        cleaned = base
        for _ in range(5):
            prev = cleaned
            cleaned = _SUFFIX_RE.sub("", cleaned).strip()
            if cleaned == prev:
                break
        _add(cleaned)

    # 접두어+접미어 동시 제거
    combined = no_prefix
    for _ in range(5):
        prev = combined
        combined = _SUFFIX_RE.sub("", combined).strip()
        if combined == prev:
            break
    _add(combined)

    # 괄호 내용 모두 제거: "치과(별도 추가)스케일링" → "치과스케일링" → "스케일링"
    no_parens = re.sub(r"\(.*?\)|（.*?）|\[.*?\]", "", raw).strip()
    _add(no_parens)
    no_parens_no_prefix = no_parens
    for _ in range(4):
        prev = no_parens_no_prefix
        no_parens_no_prefix = _PREFIX_RE.sub("", no_parens_no_prefix).strip()
        _add(no_parens_no_prefix)
        if no_parens_no_prefix == prev:
            break

    # 분리 기호 기준 각 파트
    for base in [raw, no_prefix, no_parens, no_parens_no_prefix]:
        parts = _SPLIT_RE.split(base)
        clean_parts = []
        for part in parts:
            part = part.strip()
            part = re.sub(r"\(.*?\)|（.*?）|\[.*?\]", "", part).strip()
            part = _SUFFIX_RE.sub("", part).strip()
            _add(part)
            # 파트에서도 접두어 제거
            part_no_prefix = _PREFIX_RE.sub("", part).strip()
            _add(part_no_prefix)
            if part:
                clean_parts.append(part)
        # 역순 연결: "초음파 - 복부" → "복부초음파"
        if len(clean_parts) == 2:
            _add(clean_parts[1] + clean_parts[0])
            _add(clean_parts[0] + clean_parts[1])

    # 수식어 제거 버전
    for base in list(variants)[:15]:
        no_filler = _FILLER_WORDS.sub("", base).strip()
        _add(no_filler)

    # 콜론(:) 뒤 부분: "발치-영구치 발치 (1개당) : 앞니" → "앞니"
    if ":" in raw or "：" in raw:
        after_colon = re.split(r"\s*[:：]\s*", raw)[-1].strip()
        after_colon = re.sub(r"\(.*?\)", "", after_colon).strip()
        _add(after_colon)
        # 콜론 앞 부분도 추가
        before_colon = re.split(r"\s*[:：]\s*", raw)[0].strip()
        before_colon = re.sub(r"\(.*?\)", "", before_colon).strip()
        _add(before_colon)
        bc_no_prefix = _PREFIX_RE.sub("", before_colon).strip()
        _add(bc_no_prefix)

    # 공백으로 분리된 한글 토큰 중 2글자 이상인 것들 (마지막 수단)
    for base in [no_prefix, combined]:
        kr_tokens = re.findall(r"[가-힣]{2,}", base)
        for tk in kr_tokens:
            _add(tk)

    return variants


# 접두어 단어 자체는 너무 일반적 → 매칭에서 큰 감점
_PREFIX_WORD_SET = set(_PREFIX_WORDS)

def _match_score(tag: Dict[str, Any], query: str) -> Tuple[int, Dict[str, Any]]:
    q_raw = (query or "").strip()
    if not q_raw: return 0, {}
    if _is_single_latin_char(q_raw): return 0, {}

    # ✅ 원본 + 접두어/접미어 제거 변형 모두에 대해 매칭 시도
    variants = _generate_variants(q_raw)
    if not variants:
        variants = [q_raw]

    # 구체적(비접두어) variant가 1개라도 있으면 접두어 variant 억제
    has_non_prefix = any(v.strip() not in _PREFIX_WORD_SET for v in variants)

    best_overall = 0
    best_why: List[str] = []

    for variant in variants:
        score, ev = _match_score_single(tag, variant)
        # 접두어 단어 자체("검사","처치","수술","처방" 등)는 너무 generic
        # → 구체적 variant가 있을 때만 감점
        v_stripped = variant.strip()
        if v_stripped in _PREFIX_WORD_SET and has_non_prefix:
            score = min(score, 80)  # threshold(90) 미만으로 억제
        if score > best_overall:
            best_overall = score
            best_why = ev.get("why", [])

    return best_overall, {"why": best_why[:10]}


def _match_score_single(tag: Dict[str, Any], q_raw: str) -> Tuple[int, Dict[str, Any]]:
    """단일 문자열에 대한 태그 매칭 점수 계산."""
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
            # ✅ 짧은 한글 별명(2~3글자)이 더 긴 단어의 일부로만 매칭되는 경우 감점
            # 예: "사료" in "검사료" → 잘못된 매칭 (별도 단어가 아님)
            idx = q_norm.find(a_norm)
            is_false_substring = False
            if len(a_norm) <= 3 and any("\uac00" <= c <= "\ud7a3" for c in a_norm):
                # 앞 글자가 한글이면 → 독립 단어가 아님
                if idx > 0 and "\uac00" <= q_norm[idx - 1] <= "\ud7a3":
                    is_false_substring = True
            if is_false_substring:
                s = min(80, 60 + len(a_norm) * 2)  # threshold 미만으로 억제
                best = max(best, s); why.append(f"falseSubstr:{a}")
            else:
                s = 120 + min(60, len(a_norm) * 2)
                best = max(best, s); hit += 1; strong = True; why.append(f"inQuery:{a}")
        elif a_norm.find(q_norm) >= 0:
            kr_bonus = 10 if len(q_norm) <= 4 and any("가" <= c <= "힣" for c in q_norm) else 0
            s = 90 + min(40, len(q_norm) * 2) + kr_bonus
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

def resolve_record_tags(*, items, hospital_name=None, ocr_text=None, record_thresh=120, item_thresh=100, max_tags=8, return_item_tags=True, **kw):
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


# =========================================================
# main.py 호환 함수: classify_item / classify_record
# =========================================================
_PRESCRIPTION_KEYWORDS = {"처방", "rx", "prescription", "therapeutic", "치료용"}

def classify_item(item_name: str, threshold: int = 100) -> Optional[str]:
    """단일 항목명 → 최적 태그 코드 반환. 매칭 안 되면 None."""
    if not (item_name or "").strip():
        return None
    best_code = None
    best_score = 0
    second_code = None
    second_score = 0
    for tag in TAG_CATALOG:
        s, _ = _match_score(tag, item_name)
        if s > best_score:
            second_code, second_score = best_code, best_score
            best_score = s
            best_code = tag["code"]
        elif s > second_score:
            second_score = s
            second_code = tag["code"]

    # 처방식 우선: supply_food가 1등이지만 "처방" 키워드가 있으면 care_prescription_diet 우선
    if best_code == "supply_food":
        nm_lower = _normalize(item_name)
        if any(_normalize(kw) in nm_lower for kw in _PRESCRIPTION_KEYWORDS):
            # care_prescription_diet가 후보에 있으면 교체
            if second_code == "care_prescription_diet" and second_score >= threshold:
                best_code = second_code
                best_score = second_score
            else:
                # 직접 검색
                for tag in TAG_CATALOG:
                    if tag["code"] == "care_prescription_diet":
                        s, _ = _match_score(tag, item_name)
                        if s >= threshold:
                            best_code = "care_prescription_diet"
                            best_score = s
                        break

    if best_code and best_score >= threshold:
        if best_code == "etc_other":
            return None
        return best_code
    return None


def classify_record(items: list, ocr_text: str = "", threshold: int = 120) -> List[str]:
    """항목 리스트 + OCR 텍스트 → 레코드 수준 태그 코드 리스트 반환."""
    converted = []
    for it in (items or []):
        nm = (it.get("name") or it.get("itemName") or "").strip()
        if nm:
            converted.append({"itemName": nm})
    result = resolve_record_tags(
        items=converted,
        ocr_text=ocr_text,
        record_thresh=threshold,
        item_thresh=max(80, threshold - 20),
        return_item_tags=False,
    )
    return result.get("tags", [])

