"""
반려동물 질환/케어 태그 정의 파일.

•  AI 케어 분석(통계/요약)에서 record.tags(서버 코드) 매칭에 사용됩니다.
•  iOS MedicalTagPresets.swift / ReceiptTags.swift 와 완전 동기화.

✅ 권장: iOS에서 보내는 tags 는 아래의 "code(=snake_case)" 를 사용하세요.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Literal

SpeciesType = Literal["dog", "cat", "both"]


@dataclass(frozen=True)
class ConditionTagConfig:
    code: str
    label: str
    species: SpeciesType
    group: str
    keywords: List[str]
    guide: List[str]


CONDITION_TAGS: Dict[str, ConditionTagConfig] = {

    # ===================================================
    # 1) 피부 · 알레르기 / 귀
    # ===================================================
    "skin_atopy": ConditionTagConfig(
        code="skin_atopy", label="피부 · 아토피/알레르기", species="both", group="dermatology",
        keywords=["skin_atopy","아토피","피부 알레르기","알레르기","알러지","알레르기성 피부염","가려움","소양감","발적","홍반","습진","피부염","atopy","allergic dermatitis","dermatitis","pruritus","itch"],
        guide=["저자극 샴푸를 사용해 주세요.","알레르기를 유발할 수 있는 간식·사료는 피해주세요.","규칙적으로 빗질해 주면서 피부 상태를 확인해 주세요."],
    ),
    "skin_food_allergy": ConditionTagConfig(
        code="skin_food_allergy", label="피부 · 식이 알레르기", species="both", group="dermatology",
        keywords=["skin_food_allergy","식이 알레르기","음식 알레르기","사료 알레르기","food allergy","dietary allergy","elimination diet"],
        guide=["문제가 된 음식/간식을 메모해 두고 식단에서 제외해 주세요.","수의사와 식이 제한 테스트 계획을 상의해 보세요."],
    ),
    "skin_pyoderma": ConditionTagConfig(
        code="skin_pyoderma", label="피부 · 세균성 피부염(농피증)", species="both", group="dermatology",
        keywords=["skin_pyoderma","농피증","세균성 피부염","pyoderma","pustule","고름","농포","모낭염","folliculitis"],
        guide=["처방 받은 약욕/약을 정해진 기간 동안 꾸준히 사용해 주세요.","피부가 계속 젖어 있지 않도록 잘 말려 주세요."],
    ),
    "skin_malassezia": ConditionTagConfig(
        code="skin_malassezia", label="피부 · 곰팡이성 피부염(말라세지아)", species="both", group="dermatology",
        keywords=["skin_malassezia","말라세지아","곰팡이","진균","yeast","malassezia","fungal"],
        guide=["항진균 샴푸·약을 빼먹지 않고 사용해 주세요.","귀·발 사이 등 습한 부위를 자주 확인해 주세요."],
    ),
    "ear_otitis": ConditionTagConfig(
        code="ear_otitis", label="귀 · 외이염/귓병", species="both", group="dermatology",
        keywords=["ear_otitis","외이염","귓병","귀 염증","otitis","otitis externa","ear infection"],
        guide=["귀 세정제를 주기적으로 사용해 주세요.","귀를 심하게 긁거나 머리를 흔드는 행동이 늘면 병원에 내원해 주세요."],
    ),

    # ===================================================
    # 2) 심장
    # ===================================================
    "heart_murmur": ConditionTagConfig(
        code="heart_murmur", label="심장 · 심잡음", species="both", group="cardiology",
        keywords=["heart_murmur","심잡음","심장잡음","잡음","heart murmur","murmur"],
        guide=["정기적인 심장 초음파/흉부 방사선 검사 주기를 수의사와 상의해 주세요.","호흡이 갑자기 빨라지거나 기침이 늘면 바로 병원에 문의해 주세요."],
    ),
    "heart_mitral_valve": ConditionTagConfig(
        code="heart_mitral_valve", label="심장 · 승모판 질환(MVD/MMVD)", species="dog", group="cardiology",
        keywords=["heart_mitral_valve","승모판","승모판폐쇄부전","mitral valve","MVD","MMVD","MR","myxomatous"],
        guide=["수의사가 안내한 주기로 심장 초음파를 추적 검사해 주세요.","잠자는 동안 호흡수가 늘어나면 기록해 두고 상담해 주세요."],
    ),
    "heart_hcm": ConditionTagConfig(
        code="heart_hcm", label="심장 · 비대성심근증(HCM)", species="cat", group="cardiology",
        keywords=["heart_hcm","비대성심근증","비대심근증","HCM","hcm","hypertrophic cardiomyopathy","심근비대","좌심실비대"],
        guide=["정기적인 심장 초음파 검사로 진행 상태를 확인하는 것이 중요해요.","호흡이 빨라지거나 입으로 숨을 쉬면 즉시 병원에 방문해 주세요.","스트레스를 최소화하고 조용한 환경을 유지해 주세요."],
    ),

    # ===================================================
    # 3) 관절/정형
    # ===================================================
    "ortho_patella": ConditionTagConfig(
        code="ortho_patella", label="관절 · 슬개골 탈구", species="dog", group="orthopedics",
        keywords=["ortho_patella","슬개골","슬개골탈구","무릎 탈구","슬탈","patella","patellar","patellar luxation","MPL","LPL","PL","파행","절뚝","limping","lameness"],
        guide=["집 안 바닥에는 미끄럽지 않은 매트를 깔아 주세요.","계단 오르내리기/높은 점프는 최대한 제한해 주세요.","체중 관리와 관절 영양제 급여는 수의사와 상의해 보세요."],
    ),
    "ortho_arthritis": ConditionTagConfig(
        code="ortho_arthritis", label="관절 · 관절염", species="both", group="orthopedics",
        keywords=["ortho_arthritis","관절염","골관절염","퇴행성","arthritis","osteoarthritis","OA","DJD","degenerative joint disease"],
        guide=["체중 조절이 관절 관리에서 가장 중요해요.","짧고 잦은 산책으로 관절에 무리가 가지 않게 운동량을 조절해 주세요."],
    ),

    # ===================================================
    # 4) 비뇨기계
    # ===================================================
    "urinary_stones": ConditionTagConfig(
        code="urinary_stones", label="비뇨기 · 요로결석", species="both", group="urology",
        keywords=["urinary_stones","요로결석","결석","방광결석","신장결석","요석","urolithiasis","urolith","bladder stone","kidney stone","struvite","스트루바이트","calcium oxalate","옥살레이트","CaOx"],
        guide=["충분한 수분 섭취가 가장 중요해요. 음수량을 늘릴 수 있는 방법을 시도해 보세요.","처방식(요로 처방사료)을 먹고 있다면 수의사 지시대로 유지해 주세요.","소변량/색/빈도 변화를 관찰하고 기록해 두면 진료에 도움이 돼요."],
    ),
    "urinary_cystitis": ConditionTagConfig(
        code="urinary_cystitis", label="비뇨기 · 방광염/FLUTD", species="both", group="urology",
        keywords=["urinary_cystitis","방광염","방광","혈뇨","빈뇨","배뇨곤란","FLUTD","flutd","FIC","fic","cystitis","hematuria","dysuria","stranguria","feline lower urinary tract disease","idiopathic cystitis"],
        guide=["스트레스가 원인일 수 있어요. 화장실 환경과 수를 점검해 주세요.","음수량을 늘리는 것이 중요해요. 습식 사료 급여를 고려해 보세요.","소변을 자주 보거나 피가 섞여 나오면 바로 병원에 문의해 주세요."],
    ),
    "renal_ckd": ConditionTagConfig(
        code="renal_ckd", label="신장 · 만성신장병(CKD)", species="both", group="urology",
        keywords=["renal_ckd","만성신장병","만성신부전","신부전","신장병","신장질환","CKD","chronic kidney disease","renal failure","renal insufficiency","BUN","크레아티닌","creatinine","SDMA","IRIS stage","iris"],
        guide=["수분 섭취를 충분히 유지하는 것이 가장 중요해요.","신장 처방식을 수의사와 상의해 보세요.","정기적인 혈액검사/소변검사로 진행 단계를 확인해 주세요."],
    ),

    # ===================================================
    # 5) 예방 (백신/기생충)
    # ===================================================
    "prevent_vaccine_comprehensive": ConditionTagConfig(
        code="prevent_vaccine_comprehensive", label="예방접종 · 종합백신(DHPPL/FVRCP)", species="both", group="preventive",
        keywords=["prevent_vaccine_comprehensive","종합백신","혼합백신","DHPPL","DHPP","DHPPi","DHPPI","DHLPP","5종","6종","FVRCP","fvrcp"],
        guide=["종합백신은 일정 주기로 반복 접종이 필요해요.","접종한 날짜와 다음 접종 예정일을 캘린더에 기록해 두세요.","접종 후 24시간 정도는 컨디션·식욕 변화를 잘 관찰해 주세요."],
    ),
    "prevent_vaccine_corona": ConditionTagConfig(
        code="prevent_vaccine_corona", label="예방접종 · 코로나 장염(개)", species="dog", group="preventive",
        keywords=["prevent_vaccine_corona","코로나 백신","코로나장염","corona vaccine","coronavirus","corona enteritis"],
        guide=["접종 간격은 병원 안내에 맞춰 주세요.","접종 후 붓기/통증이 심하면 병원에 문의해 주세요."],
    ),
    "prevent_vaccine_kennel": ConditionTagConfig(
        code="prevent_vaccine_kennel", label="예방접종 · 켄넬코프(기관지염)", species="dog", group="preventive",
        keywords=["prevent_vaccine_kennel","켄넬코프","켄넬 코프","기관지염 백신","bordetella","kennel cough","보르데텔라"],
        guide=["접종 후 1~2일은 컨디션 변화를 관찰해 주세요."],
    ),
    "prevent_vaccine_rabies": ConditionTagConfig(
        code="prevent_vaccine_rabies", label="예방접종 · 광견병", species="both", group="preventive",
        keywords=["prevent_vaccine_rabies","광견병","광견","rabies"],
        guide=["지역/상황에 따라 접종 주기가 달라질 수 있어요. 병원 안내에 맞춰 기록해 주세요."],
    ),
    "prevent_vaccine_lepto": ConditionTagConfig(
        code="prevent_vaccine_lepto", label="예방접종 · 렙토(Lepto)", species="dog", group="preventive",
        keywords=["prevent_vaccine_lepto","렙토","렙토스피라","렙토2","렙토4","lepto","leptospira","leptospirosis"],
        guide=["접종 주기/추가 접종 여부는 병원 안내에 맞춰 기록해 주세요."],
    ),
    "prevent_vaccine_parainfluenza": ConditionTagConfig(
        code="prevent_vaccine_parainfluenza", label="예방접종 · 파라인플루엔자(개)", species="dog", group="preventive",
        keywords=["prevent_vaccine_parainfluenza","파라인플루엔자","파라인","parainfluenza","CPiV","CPI"],
        guide=["호흡기 증상이 있거나 컨디션이 떨어지면 병원에 문의해 주세요."],
    ),
    "prevent_vaccine_fip": ConditionTagConfig(
        code="prevent_vaccine_fip", label="예방접종 · FIP(고양이)", species="cat", group="preventive",
        keywords=["prevent_vaccine_fip","FIP","fip","전염성복막염","복막염","feline infectious peritonitis"],
        guide=["접종 기록과 아이의 컨디션을 함께 관찰해 주세요."],
    ),
    "prevent_heartworm": ConditionTagConfig(
        code="prevent_heartworm", label="예방 · 심장사상충", species="both", group="preventive",
        keywords=["prevent_heartworm","사상충","심장사상충","heartworm","dirofilaria","하트가드","heartgard","넥스가드스펙트라","nexgard spectra","심파리카트리오","simparica trio","리볼루션","revolution"],
        guide=["매달 같은 날짜에 예방약을 챙기면 잊기 쉬운 달을 줄일 수 있어요.","구토/설사 등 이상 반응이 있으면 복용 중단 후 병원에 문의해 주세요."],
    ),
    "prevent_external": ConditionTagConfig(
        code="prevent_external", label="예방 · 외부기생충(진드기/벼룩)", species="both", group="preventive",
        keywords=["prevent_external","외부기생충","진드기","벼룩","tick","flea","브라벡토","bravecto","넥스가드","nexgard","프론트라인","frontline","심파리카","simparica"],
        guide=["산책이 잦은 계절엔 외부기생충 예방을 꾸준히 유지해 주세요."],
    ),
    "prevent_deworming": ConditionTagConfig(
        code="prevent_deworming", label="예방 · 구충(내부기생충)", species="both", group="preventive",
        keywords=["prevent_deworming","구충","구충제","내부기생충","회충","deworm","deworming","drontal","밀베맥스","milbemax"],
        guide=["구충 주기는 아이의 생활 환경에 따라 달라질 수 있어요. 병원 안내에 맞춰 기록해 주세요."],
    ),

    # ===================================================
    # 6) 검사
    # ===================================================
    "exam_xray": ConditionTagConfig(
        code="exam_xray", label="검사 · 엑스레이", species="both", group="exam",
        keywords=["exam_xray","엑스레이","X-ray","x-ray","xray","XR","방사선","radiograph","radiography"],
        guide=["검사 기록이에요. 결과(정상/이상 소견)를 한 줄로 함께 남기면 다음 요약이 더 정확해져요."],
    ),
    "exam_ct": ConditionTagConfig(
        code="exam_ct", label="검사 · CT", species="both", group="exam",
        keywords=["exam_ct","CT","ct","ct촬영","ct검사","ct조영","ct scan","computed tomography","컴퓨터단층촬영","씨티"],
        guide=["CT 검사 결과(소견)를 함께 기록해 두면 추적 관리에 도움이 돼요."],
    ),
    "exam_mri": ConditionTagConfig(
        code="exam_mri", label="검사 · MRI", species="both", group="exam",
        keywords=["exam_mri","MRI","mri","mri촬영","mri검사","magnetic resonance","자기공명","엠알아이"],
        guide=["MRI 검사 결과(소견)를 함께 기록해 두면 추적 관리에 도움이 돼요."],
    ),
    "exam_endoscope": ConditionTagConfig(
        code="exam_endoscope", label="검사 · 내시경", species="both", group="exam",
        keywords=["exam_endoscope","내시경","endoscopy","endoscope","gastroscopy","위내시경","장내시경","관절경","arthroscopy"],
        guide=["내시경 검사 소견을 기록해 두면 추적에 도움이 돼요."],
    ),
    "exam_biopsy": ConditionTagConfig(
        code="exam_biopsy", label="검사 · 조직검사(생검)", species="both", group="exam",
        keywords=["exam_biopsy","생검","조직검사","biopsy","fna","fine needle","세침흡인","병리검사","histopathology"],
        guide=["조직검사 결과를 받으면 병원에서 설명해 주는 소견을 메모해 두세요."],
    ),
    "exam_echo": ConditionTagConfig(
        code="exam_echo", label="검사 · 심장초음파", species="both", group="exam",
        keywords=["exam_echo","심장초음파","심초음파","cardiac ultrasound","echocardiogram","echocardiography","echo","에코"],
        guide=["심장초음파 수치(LA/Ao 비율 등)를 기록해 두면 추적에 도움이 돼요."],
    ),
    "exam_us_abdomen": ConditionTagConfig(
        code="exam_us_abdomen", label="검사 · 복부초음파", species="both", group="exam",
        keywords=["exam_us_abdomen","복부초음파","abdominal ultrasound","abdominal us","abd us","abd sono"],
        guide=["복부초음파 소견을 기록해 두면 추적에 도움이 돼요."],
    ),
    "exam_us_general": ConditionTagConfig(
        code="exam_us_general", label="검사 · 초음파(일반)", species="both", group="exam",
        keywords=["exam_us_general","초음파","ultrasound","sono","sonography","US"],
        guide=["초음파 검사 소견을 기록해 두면 추적에 도움이 돼요."],
    ),
    "exam_blood_cbc": ConditionTagConfig(
        code="exam_blood_cbc", label="검사 · CBC(혈구검사)", species="both", group="exam",
        keywords=["exam_blood_cbc","CBC","cbc","complete blood count","혈구검사","혈구","blood count"],
        guide=["CBC 수치 변화를 추적하면 건강 상태 변화를 확인할 수 있어요."],
    ),
    "exam_blood_chem": ConditionTagConfig(
        code="exam_blood_chem", label="검사 · 생화학검사", species="both", group="exam",
        keywords=["exam_blood_chem","생화학","생화학검사","chemistry","biochem","간수치","신장수치","간기능","chemistry panel","chem"],
        guide=["생화학 수치는 추세가 중요해요. 이전 결과와 비교하면 좋아요."],
    ),
    "exam_blood_general": ConditionTagConfig(
        code="exam_blood_general", label="검사 · 혈액검사", species="both", group="exam",
        keywords=["exam_blood_general","혈액검사","혈검","피검","blood test","profile","혈액","blood work"],
        guide=["혈액검사 결과 소견을 한 줄로 기록해 두면 좋아요."],
    ),
    "exam_blood": ConditionTagConfig(
        code="exam_blood", label="검사 · 혈액검사", species="both", group="exam",
        keywords=["exam_blood","혈액검사","혈검","피검","CBC","Chemistry","blood test","생화학","전해질"],
        guide=["혈액검사는 상태를 확인하는 검사 기록이에요. 병원 소견을 짧게 메모해 두면 좋아요."],
    ),
    "exam_blood_type": ConditionTagConfig(
        code="exam_blood_type", label="검사 · 혈액형검사", species="both", group="exam",
        keywords=["exam_blood_type","혈액형","혈액형검사","blood type","blood typing","crossmatch","교차시험"],
        guide=["혈액형 결과를 기록해 두면 응급 시 수혈에 도움이 돼요."],
    ),
    "exam_coagulation": ConditionTagConfig(
        code="exam_coagulation", label="검사 · 응고검사", species="both", group="exam",
        keywords=["exam_coagulation","응고검사","응고","coagulation","PT","aPTT","프로트롬빈","피브리노겐"],
        guide=["응고 검사 수치를 기록해 두면 추적에 도움이 돼요."],
    ),
    "exam_electrolyte": ConditionTagConfig(
        code="exam_electrolyte", label="검사 · 전해질검사", species="both", group="exam",
        keywords=["exam_electrolyte","전해질","전해질검사","electrolyte","나트륨","칼륨","칼슘","calcium","phosphorus"],
        guide=["전해질 수치 변화를 기록해 두면 관리에 도움이 돼요."],
    ),
    "exam_crp": ConditionTagConfig(
        code="exam_crp", label="검사 · CRP(염증)", species="both", group="exam",
        keywords=["exam_crp","CRP","crp","c-reactive protein","염증수치","염증검사","염증마커"],
        guide=["CRP 수치 변화를 추적하면 염증 상태를 확인할 수 있어요."],
    ),
    "exam_ecg": ConditionTagConfig(
        code="exam_ecg", label="검사 · 심전도(ECG)", species="both", group="exam",
        keywords=["exam_ecg","심전도","ECG","EKG","electrocardiogram","심전도검사"],
        guide=["심전도 소견을 기록해 두면 심장 상태 추적에 도움이 돼요."],
    ),
    "exam_heart_general": ConditionTagConfig(
        code="exam_heart_general", label="검사 · 심장검사(일반)", species="both", group="exam",
        keywords=["exam_heart_general","심장검사","cardiac","heart","heart check","심장"],
        guide=["심장검사 소견을 기록해 두면 추적에 도움이 돼요."],
    ),
    "exam_heart": ConditionTagConfig(
        code="exam_heart", label="검사 · 심장검사(심초음파/심전도)", species="both", group="exam",
        keywords=["exam_heart","심장검사","심초음파","심전도","ECG","EKG","echo","echocardiography","cardiac exam"],
        guide=["심장검사는 추적이 중요해요. 검사 수치/단계(등급) 메모를 남기면 좋아요."],
    ),
    "exam_hormone": ConditionTagConfig(
        code="exam_hormone", label="검사 · 호르몬검사", species="both", group="exam",
        keywords=["exam_hormone","호르몬","호르몬검사","hormone","T4","T3","fT4","TSH","갑상선","thyroid","cortisol","코르티솔","ACTH","부신"],
        guide=["호르몬 수치 변화를 추적하면 내분비 질환 관리에 도움이 돼요."],
    ),
    "exam_general": ConditionTagConfig(
        code="exam_general", label="검사 · 정밀검사/검진", species="both", group="exam",
        keywords=["exam_general","정밀검사","검사","검진","진단검사","health check","checkup","screening","기본검사"],
        guide=["검사 결과 소견을 한 줄로 기록해 두면 추적에 도움이 돼요."],
    ),
    "exam_ultrasound": ConditionTagConfig(
        code="exam_ultrasound", label="검사 · 초음파", species="both", group="exam",
        keywords=["exam_ultrasound","초음파","복부초음파","심장초음파","ultrasound","sono","sonography","US"],
        guide=["초음파는 진단을 위한 검사 기록이에요. 검사 목적/소견을 함께 남겨두면 도움이 돼요."],
    ),
    "exam_lab_panel": ConditionTagConfig(
        code="exam_lab_panel", label="검사 · 종합검사", species="both", group="exam",
        keywords=["exam_lab_panel","종합검사","종합검진","패널","lab panel","screening panel","profile"],
        guide=["종합검사는 변화 추적이 중요해요. 이전 결과와 비교 메모를 남겨두면 좋아요."],
    ),
    "exam_urine": ConditionTagConfig(
        code="exam_urine", label="검사 · 소변검사(요검사)", species="both", group="exam",
        keywords=["exam_urine","소변검사","요검사","urinalysis","UA","urine test","요비중","요침사","요단백","UPC"],
        guide=["소변검사는 요로/신장 상태 확인에 도움이 돼요."],
    ),
    "exam_fecal": ConditionTagConfig(
        code="exam_fecal", label="검사 · 대변검사", species="both", group="exam",
        keywords=["exam_fecal","대변검사","분변검사","fecal","stool test","기생충 검사","giardia"],
        guide=["대변검사는 장 상태/기생충 확인에 도움이 돼요."],
    ),
    "exam_fecal_pcr": ConditionTagConfig(
        code="exam_fecal_pcr", label="검사 · 대변 PCR(GI PCR)", species="both", group="exam",
        keywords=["exam_fecal_pcr","대변 PCR","분변 PCR","fecal pcr","stool pcr","gi pcr","GI panel"],
        guide=["PCR은 원인 추정에 도움 될 수 있어요."],
    ),
    "exam_allergy": ConditionTagConfig(
        code="exam_allergy", label="검사 · 알러지 검사", species="both", group="exam",
        keywords=["exam_allergy","알러지검사","알레르기검사","allergy test","IgE","atopy test"],
        guide=["알러지 검사는 결과 해석이 중요해요. 수의사 설명 요점을 함께 기록해 두면 좋아요."],
    ),
    "exam_eye": ConditionTagConfig(
        code="exam_eye", label="검사 · 안과검사", species="both", group="exam",
        keywords=["exam_eye","안과검사","안압","형광염색","schirmer","fluorescein","IOP","ophthalmic exam"],
        guide=["안과 검사는 증상 변화(눈곱/충혈/통증) 기록이 도움이 돼요."],
    ),
    "exam_skin": ConditionTagConfig(
        code="exam_skin", label="검사 · 피부검사", species="both", group="exam",
        keywords=["exam_skin","피부검사","피부스크래핑","skin scraping","cytology","진균검사","말라세지아"],
        guide=["피부 검사는 사진 기록이 특히 도움이 돼요."],
    ),
    "exam_sdma": ConditionTagConfig(
        code="exam_sdma", label="검사 · SDMA(신장마커)", species="both", group="exam",
        keywords=["exam_sdma","SDMA","sdma","신장마커","신장검사","idexx sdma","renal sdma"],
        guide=["SDMA는 신장 기능을 조기에 확인하는 데 도움 될 수 있어요."],
    ),
    "exam_probnp": ConditionTagConfig(
        code="exam_probnp", label="검사 · proBNP(심장마커)", species="both", group="exam",
        keywords=["exam_probnp","proBNP","probnp","NT-proBNP","ntprobnp","bnp","심장마커","프로비엔피"],
        guide=["심장마커 검사는 다른 검사와 함께 해석돼요."],
    ),
    "exam_fructosamine": ConditionTagConfig(
        code="exam_fructosamine", label="검사 · 당화알부민/FRU", species="both", group="exam",
        keywords=["exam_fructosamine","당화알부민","fructosamine","fru","glycated albumin","당뇨","혈당"],
        guide=["혈당 관련 검사는 추세가 중요해요."],
    ),
    "exam_glucose_curve": ConditionTagConfig(
        code="exam_glucose_curve", label="검사 · 혈당곡선", species="both", group="exam",
        keywords=["exam_glucose_curve","혈당곡선","혈당커브","glucose curve","bg curve","연속혈당"],
        guide=["혈당곡선은 시간대별 변화가 핵심이에요."],
    ),
    "exam_blood_gas": ConditionTagConfig(
        code="exam_blood_gas", label="검사 · 혈액가스(BGA)", species="both", group="exam",
        keywords=["exam_blood_gas","혈액가스","blood gas","BGA","i-stat","istat","가스분석"],
        guide=["혈액가스는 응급/호흡/산염기 상태 평가에 쓰일 수 있어요."],
    ),

    # ===================================================
    # 7) 약(처방)
    # ===================================================
    "medicine_antibiotic": ConditionTagConfig(
        code="medicine_antibiotic", label="처방 · 항생제", species="both", group="medication",
        keywords=["medicine_antibiotic","항생제","antibiotic","abx","amoxicillin","clavamox","cephalexin","convenia","doxycycline","metronidazole","baytril"],
        guide=["처방 받은 기간을 지켜 복용해 주세요."],
    ),
    "medicine_anti_inflammatory": ConditionTagConfig(
        code="medicine_anti_inflammatory", label="처방 · 소염/항염(NSAID)", species="both", group="medication",
        keywords=["medicine_anti_inflammatory","소염","항염","소염제","NSAIDs","NSAID","meloxicam","metacam","carprofen","rimadyl","onsior","galliprant"],
        guide=["식욕 저하/구토 등 이상 증상이 있으면 복용을 멈추고 병원에 문의해 주세요."],
    ),
    "medicine_allergy": ConditionTagConfig(
        code="medicine_allergy", label="처방 · 알러지 약", species="both", group="medication",
        keywords=["medicine_allergy","알러지","알레르기","항히스타민","apoquel","cytopoint","cetirizine","zyrtec"],
        guide=["증상이 심해지거나 졸림이 심하면 병원에 문의해 주세요."],
    ),
    "medicine_gi": ConditionTagConfig(
        code="medicine_gi", label="처방 · 위장/장", species="both", group="medication",
        keywords=["medicine_gi","위장","장","설사","구토","장염","famotidine","omeprazole","cerenia","ondansetron"],
        guide=["물/식사량 변화를 함께 관찰해 주세요."],
    ),
    "medicine_ear": ConditionTagConfig(
        code="medicine_ear", label="처방 · 귀", species="both", group="medication",
        keywords=["medicine_ear","귀약","점이","이염","ear drops","otic","otomax","surolan","posatex"],
        guide=["귀 세정/점이 지침이 있다면 횟수를 지켜 사용해 주세요."],
    ),
    "medicine_skin": ConditionTagConfig(
        code="medicine_skin", label="처방 · 피부", species="both", group="medication",
        keywords=["medicine_skin","피부약","연고","약욕","ointment","topical","chlorhexidine","ketoconazole"],
        guide=["피부가 악화되는 부위가 있으면 사진으로 기록해 두면 도움이 돼요."],
    ),
    "medicine_eye": ConditionTagConfig(
        code="medicine_eye", label="처방 · 안약", species="both", group="medication",
        keywords=["medicine_eye","안약","점안","eye drops","ophthalmic","tobramycin","ofloxacin","cyclosporine"],
        guide=["점안 횟수/간격을 지켜 주세요."],
    ),
    "medicine_painkiller": ConditionTagConfig(
        code="medicine_painkiller", label="처방 · 진통", species="both", group="medication",
        keywords=["medicine_painkiller","진통","진통제","painkiller","analgesic","tramadol","gabapentin"],
        guide=["통증 징후(절뚝/숨김/예민함)가 줄어드는지 함께 관찰해 주세요."],
    ),
    "medicine_steroid": ConditionTagConfig(
        code="medicine_steroid", label="처방 · 스테로이드", species="both", group="medication",
        keywords=["medicine_steroid","스테로이드","steroid","prednisone","prednisolone","dexamethasone"],
        guide=["스테로이드는 용량/기간 관리가 중요해요. 갑자기 중단하지 말고 병원 지침에 따라 주세요."],
    ),
    "drug_general": ConditionTagConfig(
        code="drug_general", label="처방 · 약(일반)", species="both", group="medication",
        keywords=["drug_general","약","약값","내복약","처방","처방약","medication","drug","Rx"],
        guide=["처방 약은 지시된 기간/용량을 지켜 주세요."],
    ),
    "drug_oral": ConditionTagConfig(
        code="drug_oral", label="처방 · 내복약(경구)", species="both", group="medication",
        keywords=["drug_oral","내복약","경구약","먹는약","oral","oral med","po"],
        guide=["내복약은 정해진 시간에 복용하면 효과가 좋아요."],
    ),

    # ===================================================
    # 8) 처치 / 수액 / 응급 / 입원
    # ===================================================
    "care_injection": ConditionTagConfig(
        code="care_injection", label="처치 · 주사/주사제", species="both", group="procedure",
        keywords=["care_injection","주사","주사제","주사료","injection","shot","SC","IM","IV"],
        guide=["주사 후 부종/통증/무기력 등이 지속되면 병원에 문의해 주세요."],
    ),
    "care_fluid": ConditionTagConfig(
        code="care_fluid", label="처치 · 수액/링거", species="both", group="procedure",
        keywords=["care_fluid","수액","링거","iv fluid","fluid therapy","수액처치","피하수액","정맥수액","lactated ringer","생리식염수"],
        guide=["수액 처치 후 부종이나 호흡 변화가 있으면 병원에 문의해 주세요."],
    ),
    "care_transfusion": ConditionTagConfig(
        code="care_transfusion", label="처치 · 수혈", species="both", group="procedure",
        keywords=["care_transfusion","수혈","transfusion","blood transfusion","전혈","packed rbc","혈장","plasma","ffp"],
        guide=["수혈 후 이상 반응(발열/구토/호흡곤란)이 있으면 즉시 병원에 알려주세요."],
    ),
    "care_oxygen": ConditionTagConfig(
        code="care_oxygen", label="처치 · 산소치료", species="both", group="procedure",
        keywords=["care_oxygen","산소","산소치료","산소방","산소텐트","oxygen","o2","산소케이지"],
        guide=["산소치료가 필요한 상태라면 호흡 상태를 꾸준히 관찰해 주세요."],
    ),
    "care_emergency": ConditionTagConfig(
        code="care_emergency", label="처치 · 응급처치", species="both", group="procedure",
        keywords=["care_emergency","응급","응급처치","응급진료","emergency","ER","야간진료","야간응급","CPR"],
        guide=["응급 상황 시 증상과 시간을 기록해 두면 진료에 도움이 돼요."],
    ),
    "care_catheter": ConditionTagConfig(
        code="care_catheter", label="처치 · 카테터/도뇨", species="both", group="procedure",
        keywords=["care_catheter","카테터","도뇨관","유치도뇨관","catheter","urinary catheter","정맥카테터","도뇨","방광세척"],
        guide=["카테터 삽입 부위를 깨끗하게 유지해 주세요."],
    ),
    "care_procedure_fee": ConditionTagConfig(
        code="care_procedure_fee", label="처치 · 처치료/시술료", species="both", group="procedure",
        keywords=["care_procedure_fee","처치료","처치비","시술료","시술비","처치","시술","procedure fee"],
        guide=["처치/시술 항목은 병원마다 표기가 달라요. 가능하면 상세 내용을 메모해 두면 좋아요."],
    ),
    "care_dressing": ConditionTagConfig(
        code="care_dressing", label="처치 · 드레싱/붕대/소독", species="both", group="procedure",
        keywords=["care_dressing","드레싱","붕대","거즈","소독","세척","dressing","bandage","gauze"],
        guide=["상처 부위를 핥지 않도록 주의해 주세요."],
    ),
    "care_anal_gland": ConditionTagConfig(
        code="care_anal_gland", label="처치 · 항문낭", species="both", group="procedure",
        keywords=["care_anal_gland","항문낭","항문낭짜기","항문낭세척","anal gland","anal sac","항문선"],
        guide=["항문낭이 자주 차면 주기적으로 짜주는 것이 좋아요."],
    ),
    "care_ear_flush": ConditionTagConfig(
        code="care_ear_flush", label="처치 · 귀세척", species="both", group="procedure",
        keywords=["care_ear_flush","귀세척","이도세척","ear flush","ear cleaning","귀청소"],
        guide=["귀세척 후 분비물 색/양 변화를 관찰해 주세요."],
    ),
    "care_e_collar": ConditionTagConfig(
        code="care_e_collar", label="소모품 · 넥카라", species="both", group="procedure",
        keywords=["care_e_collar","넥카라","엘리자베스카라","보호카라","e-collar","cone"],
        guide=["상처 보호 목적이라면 착용 시간을 지켜 주세요."],
    ),
    "care_prescription_diet": ConditionTagConfig(
        code="care_prescription_diet", label="소모품 · 처방식(사료)", species="both", group="procedure",
        keywords=["care_prescription_diet","처방식","처방사료","prescription diet","hill's","royal canin","k/d","c/d","i/d"],
        guide=["처방식은 목적에 맞게 급여하는 기간/방법이 중요해요."],
    ),
    "hospitalization": ConditionTagConfig(
        code="hospitalization", label="입원", species="both", group="procedure",
        keywords=["hospitalization","입원","입원비","입원료","ICU","중환자","중환자실","입원관리","케이지"],
        guide=["입원 중 상태 변화(식욕/활력/배변)를 기록해 두면 좋아요."],
    ),

    # ===================================================
    # 9) 치과
    # ===================================================
    "dental_tartar": ConditionTagConfig(
        code="dental_tartar", label="치과 · 치석", species="both", group="procedure",
        keywords=["dental_tartar","치석","치석제거","tartar","calculus","dental calculus","plaque"],
        guide=["양치질을 꾸준히 해주면 치석 예방에 큰 도움이 돼요."],
    ),
    "dental_scaling": ConditionTagConfig(
        code="dental_scaling", label="치과 · 스케일링", species="both", group="procedure",
        keywords=["dental_scaling","스케일링","치석 제거","scaling","dental cleaning"],
        guide=["시술 후 며칠간은 식욕/통증 반응을 관찰해 주세요."],
    ),
    "dental_extraction": ConditionTagConfig(
        code="dental_extraction", label="치과 · 발치", species="both", group="procedure",
        keywords=["dental_extraction","발치","extraction","dental extraction"],
        guide=["처방 약을 잘 챙겨 주세요. 출혈/통증이 심하면 병원에 문의해 주세요."],
    ),
    "dental_forl": ConditionTagConfig(
        code="dental_forl", label="치과 · 치아흡수성병변(FORL)", species="cat", group="procedure",
        keywords=["dental_forl","치아흡수","치아흡수성","흡수성병변","FORL","forl","tooth resorption","feline odontoclastic","resorptive lesion","TR"],
        guide=["고양이에게 흔한 치과 질환이에요. 식욕 변화를 잘 관찰해 주세요.","발치가 필요할 수 있어요. 수의사와 치료 계획을 상의해 주세요."],
    ),
    "dental_gingivitis": ConditionTagConfig(
        code="dental_gingivitis", label="치과 · 치은염/잇몸염증", species="both", group="procedure",
        keywords=["dental_gingivitis","치은염","잇몸염증","잇몸","치주염","치주","gingivitis","periodontitis","periodontal","gum disease"],
        guide=["양치질과 구강 관리를 꾸준히 해주세요.","잇몸이 붓거나 출혈이 있으면 병원에 문의해 주세요."],
    ),
    "dental_treatment": ConditionTagConfig(
        code="dental_treatment", label="치과 · 잇몸/치주치료", species="both", group="procedure",
        keywords=["dental_treatment","잇몸치료","치주치료","periodontal","불소","불소도포","fluoride","신경치료"],
        guide=["치주치료 후 양치 루틴을 천천히 시작해 보세요."],
    ),

    # ===================================================
    # 10) 수술
    # ===================================================
    "surgery_general": ConditionTagConfig(
        code="surgery_general", label="수술 · 일반", species="both", group="procedure",
        keywords=["surgery_general","수술","operation","surgery","봉합","마취","중성화","spay","neuter","castration"],
        guide=["수술 후 회복은 아이마다 달라요. 활력/식욕/상처 상태를 함께 기록해 두면 좋아요."],
    ),
    "surgery_spay_neuter": ConditionTagConfig(
        code="surgery_spay_neuter", label="수술 · 중성화", species="both", group="procedure",
        keywords=["surgery_spay_neuter","중성화","spay","neuter","castration","ovariohysterectomy","자궁적출","중성화수술"],
        guide=["수술 후 실밥 제거 시기까지 상처 부위를 잘 관찰해 주세요."],
    ),
    "surgery_tumor": ConditionTagConfig(
        code="surgery_tumor", label="수술 · 종양수술", species="both", group="procedure",
        keywords=["surgery_tumor","종양","종양제거","종양수술","tumor","tumor removal","mass removal","혹","종괴"],
        guide=["조직검사 결과를 확인하고, 추후 관리 계획을 수의사와 상의해 주세요."],
    ),
    "surgery_foreign_body": ConditionTagConfig(
        code="surgery_foreign_body", label="수술 · 이물제거", species="both", group="procedure",
        keywords=["surgery_foreign_body","이물","이물제거","foreign body","위절개","장절개","gastrotomy","enterotomy"],
        guide=["수술 후 식사 재개 시기는 수의사 지시에 따라 주세요."],
    ),
    "surgery_cesarean": ConditionTagConfig(
        code="surgery_cesarean", label="수술 · 제왕절개", species="both", group="procedure",
        keywords=["surgery_cesarean","제왕절개","cesarean","c-section","caesarean"],
        guide=["수술 후 어미와 새끼 모두의 상태를 잘 관찰해 주세요."],
    ),
    "surgery_hernia": ConditionTagConfig(
        code="surgery_hernia", label="수술 · 탈장수술", species="both", group="procedure",
        keywords=["surgery_hernia","탈장","탈장수술","hernia","회음부탈장","서혜부탈장","배꼽탈장"],
        guide=["수술 후 재발 여부를 주기적으로 확인해 주세요."],
    ),
    "surgery_eye": ConditionTagConfig(
        code="surgery_eye", label="수술 · 안과수술", species="both", group="procedure",
        keywords=["surgery_eye","안과수술","eye surgery","체리아이","cherry eye","백내장","cataract","안구적출","enucleation"],
        guide=["수술 후 점안약 사용 지침을 잘 지켜 주세요."],
    ),

    # ===================================================
    # 11) 재활
    # ===================================================
    "rehab_therapy": ConditionTagConfig(
        code="rehab_therapy", label="재활 · 물리치료", species="both", group="procedure",
        keywords=["rehab_therapy","재활","재활치료","물리치료","rehabilitation","physical therapy","수중런닝머신","hydrotherapy","레이저치료","laser therapy","침치료","acupuncture"],
        guide=["재활은 꾸준히 하는 것이 중요해요. 아이의 컨디션에 맞춰 강도를 조절해 주세요."],
    ),

    # ===================================================
    # 12) 웰니스/기타
    # ===================================================
    "wellness_checkup": ConditionTagConfig(
        code="wellness_checkup", label="웰니스 · 건강검진", species="both", group="wellness",
        keywords=["wellness_checkup","건강검진","종합검진","health check","check-up","checkup","screening"],
        guide=["성견/성묘는 1년에 한 번, 노령 아이는 6개월마다 건강검진을 권장해요."],
    ),
    "checkup_general": ConditionTagConfig(
        code="checkup_general", label="기본진료 · 상담/진찰", species="both", group="wellness",
        keywords=["checkup_general","기본진료","진료","진찰","상담","초진","재진","consult","checkup","opd"],
        guide=["기본진료 기록은 증상을 함께 남기면 다음 진료에 도움이 돼요."],
    ),
    "grooming_basic": ConditionTagConfig(
        code="grooming_basic", label="미용 · 목욕/관리", species="both", group="wellness",
        keywords=["grooming_basic","미용","목욕","클리핑","가위컷","발톱","귀청소","grooming","bath","trim"],
        guide=["피부가 민감한 아이는 미용 후 가려움/홍반이 생기지 않는지 관찰해 주세요."],
    ),
    "microchip": ConditionTagConfig(
        code="microchip", label="마이크로칩", species="both", group="wellness",
        keywords=["microchip","마이크로칩","칩","내장형칩","동물등록","pet registration"],
        guide=["마이크로칩 번호를 기록해 두면 만약의 경우에 도움이 돼요."],
    ),
    "euthanasia": ConditionTagConfig(
        code="euthanasia", label="안락사", species="both", group="wellness",
        keywords=["euthanasia","안락사","임종"],
        guide=[],
    ),
    "funeral": ConditionTagConfig(
        code="funeral", label="장례/화장", species="both", group="wellness",
        keywords=["funeral","장례","화장","cremation","장례비","화장비"],
        guide=[],
    ),
    "etc_other": ConditionTagConfig(
        code="etc_other", label="기타", species="both", group="wellness",
        keywords=["etc_other","기타","etc","other"],
        guide=[],
    ),
}


# ---------------------------------------------------
# ALIASES
# ---------------------------------------------------

ALIASES: Dict[str, str] = {
    # orthopedics
    "orthoPatella": "ortho_patella",
    "orthoArthritis": "ortho_arthritis",
    # dermatology / ear
    "skinAtopy": "skin_atopy",
    "skinFoodAllergy": "skin_food_allergy",
    "skinPyoderma": "skin_pyoderma",
    "skinMalassezia": "skin_malassezia",
    "earOtitis": "ear_otitis",
    # cardiology
    "heartMurmur": "heart_murmur",
    "heartMitralValve": "heart_mitral_valve",
    "heartHCM": "heart_hcm",
    "heartHcm": "heart_hcm",
    # urology
    "urinaryStones": "urinary_stones",
    "urinaryCystitis": "urinary_cystitis",
    "renalCKD": "renal_ckd",
    "renalCkd": "renal_ckd",
    # preventive vaccines
    "preventVaccineComprehensive": "prevent_vaccine_comprehensive",
    "vaccineComprehensive": "prevent_vaccine_comprehensive",
    "vaccine_comprehensive": "prevent_vaccine_comprehensive",
    "preventVaccineCorona": "prevent_vaccine_corona",
    "vaccineCorona": "prevent_vaccine_corona",
    "vaccine_corona": "prevent_vaccine_corona",
    "preventVaccineKennel": "prevent_vaccine_kennel",
    "vaccineKennel": "prevent_vaccine_kennel",
    "vaccine_kennel": "prevent_vaccine_kennel",
    "preventVaccineRabies": "prevent_vaccine_rabies",
    "vaccineRabies": "prevent_vaccine_rabies",
    "vaccine_rabies": "prevent_vaccine_rabies",
    "preventVaccineLepto": "prevent_vaccine_lepto",
    "vaccineLepto": "prevent_vaccine_lepto",
    "vaccine_lepto": "prevent_vaccine_lepto",
    "preventVaccineParainfluenza": "prevent_vaccine_parainfluenza",
    "vaccineParainfluenza": "prevent_vaccine_parainfluenza",
    "vaccine_parainfluenza": "prevent_vaccine_parainfluenza",
    "preventVaccineFIP": "prevent_vaccine_fip",
    "vaccineFIP": "prevent_vaccine_fip",
    "vaccine_fip": "prevent_vaccine_fip",
    # preventive parasites
    "preventHeartworm": "prevent_heartworm",
    "preventExternal": "prevent_external",
    "preventDeworming": "prevent_deworming",
    # exam
    "examXray": "exam_xray",
    "examCt": "exam_ct",
    "examMri": "exam_mri",
    "examEndoscope": "exam_endoscope",
    "examBiopsy": "exam_biopsy",
    "examEcho": "exam_echo",
    "examUsAbdomen": "exam_us_abdomen",
    "examUsGeneral": "exam_us_general",
    "examBloodCbc": "exam_blood_cbc",
    "examBloodChem": "exam_blood_chem",
    "examBloodGeneral": "exam_blood_general",
    "examBloodType": "exam_blood_type",
    "examCoagulation": "exam_coagulation",
    "examElectrolyte": "exam_electrolyte",
    "examCrp": "exam_crp",
    "examEcg": "exam_ecg",
    "examHeartGeneral": "exam_heart_general",
    "examHormone": "exam_hormone",
    "examGeneral": "exam_general",
    "examBlood": "exam_blood",
    "examUltrasound": "exam_ultrasound",
    "examLabPanel": "exam_lab_panel",
    "examUrine": "exam_urine",
    "examFecal": "exam_fecal",
    "examFecalPCR": "exam_fecal_pcr",
    "examAllergy": "exam_allergy",
    "examHeart": "exam_heart",
    "examEye": "exam_eye",
    "examSkin": "exam_skin",
    "examSDMA": "exam_sdma",
    "examProBNP": "exam_probnp",
    "examFructosamine": "exam_fructosamine",
    "examGlucoseCurve": "exam_glucose_curve",
    "examBloodGas": "exam_blood_gas",
    # medication
    "medicineAntibiotic": "medicine_antibiotic",
    "medicineAntiInflammatory": "medicine_anti_inflammatory",
    "medicineAllergy": "medicine_allergy",
    "medicineGI": "medicine_gi",
    "medicineEar": "medicine_ear",
    "medicineSkin": "medicine_skin",
    "medicineEye": "medicine_eye",
    "medicinePainkiller": "medicine_painkiller",
    "medicineSteroid": "medicine_steroid",
    "drugGeneral": "drug_general",
    "drugOral": "drug_oral",
    # procedure / care
    "careInjection": "care_injection",
    "careFluid": "care_fluid",
    "careTransfusion": "care_transfusion",
    "careOxygen": "care_oxygen",
    "careEmergency": "care_emergency",
    "careCatheter": "care_catheter",
    "careProcedureFee": "care_procedure_fee",
    "careDressing": "care_dressing",
    "careAnalGland": "care_anal_gland",
    "careEarFlush": "care_ear_flush",
    "careECollar": "care_e_collar",
    "carePrescriptionDiet": "care_prescription_diet",
    # dental
    "dentalTartar": "dental_tartar",
    "dentalScaling": "dental_scaling",
    "dentalExtraction": "dental_extraction",
    "dentalFORL": "dental_forl",
    "dentalForl": "dental_forl",
    "dentalGingivitis": "dental_gingivitis",
    "dentalTreatment": "dental_treatment",
    # surgery
    "surgeryGeneral": "surgery_general",
    "surgerySpayNeuter": "surgery_spay_neuter",
    "surgeryTumor": "surgery_tumor",
    "surgeryForeignBody": "surgery_foreign_body",
    "surgeryCesarean": "surgery_cesarean",
    "surgeryHernia": "surgery_hernia",
    "surgeryEye": "surgery_eye",
    # rehab
    "rehabTherapy": "rehab_therapy",
    # wellness / misc
    "wellnessCheckup": "wellness_checkup",
    "checkupGeneral": "checkup_general",
    "groomingBasic": "grooming_basic",
    "etcOther": "etc_other",
}

for alias_key, canonical_key in ALIASES.items():
    if canonical_key in CONDITION_TAGS:
        CONDITION_TAGS[alias_key] = CONDITION_TAGS[canonical_key]


__all__ = ["ConditionTagConfig", "CONDITION_TAGS", "SpeciesType"]

