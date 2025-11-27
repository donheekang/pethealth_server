"""
반려동물 질환/케어 태그 정의 파일.

•⁠  ⁠이 태그들은 '진단'이 아니라
  진료기록/검사결과에 이미 적혀 있는 병명·소견을
  카테고리화해서 표시하기 위한 용도입니다.
•⁠  ⁠보호자에게는 "기록상 ○○ 관련 내용이 보인다" 수준으로만
  안내하는 것을 권장합니다.
"""

from dataclasses import dataclass
from typing import List, Dict, Literal

SpeciesType = Literal["dog", "cat", "both"]


@dataclass(frozen=True)
class ConditionTagConfig:
    code: str           # 내부 코드 (예: "skin_atopy")
    label: str          # 사용자 노출 이름 (예: "피부 · 아토피/알레르기")
    species: SpeciesType  # "dog" | "cat" | "both"
    group: str          # 상위 그룹 (dermatology / cardiology 등)
    keywords: List[str] # OCR/병명에서 매칭할 키워드 목록 (한/영 섞어서)

# ------------------------------------------------------------------
    # 10) 예방접종 및 웰니스 (Preventive / Wellness)
    # ------------------------------------------------------------------

    # 1. 종합백신 (가장 기본)
    "prevent_vaccine_comprehensive": ConditionTagConfig(
        code="prevent_vaccine_comprehensive",
        label="예방접종 · 종합백신(DHPPL/FVRCP)",
        species="both",
        group="preventive",
        keywords=[
            "종합백신", "혼합백신", "기초접종", "추가접종",
            "DHPPL", "DA2PP", "DHPP",  # 강아지
            "FVRCP", "3종 백신", "4종 백신", "5종 백신" # 고양이/강아지
        ],
    ),

    # 2. 코로나 (사용자 질문 반영)
    "prevent_vaccine_corona": ConditionTagConfig(
        code="prevent_vaccine_corona",
        label="예방접종 · 코로나 장염 백신",
        species="dog", # 고양이 전염성복막염(FIP) 백신은 드물므로 주로 강아지용
        group="preventive",
        keywords=[
            "코로나", "코로나 장염", "Corona", "Canine Coronavirus",
            "CV 백신", "장염 예방"
        ],
    ),

    # 3. 켄넬코프 / 인플루엔자 (호흡기 관련)
    "prevent_vaccine_respiratory": ConditionTagConfig(
        code="prevent_vaccine_respiratory",
        label="예방접종 · 켄넬코프/인플루엔자",
        species="dog",
        group="preventive",
        keywords=[
            "켄넬코프", "기관지염", "Kennel Cough", "Bordetella",
            "인플루엔자", "신종플루", "독감", "CIV", "Canine Influenza"
        ],
    ),

    # 4. 광견병 (법정 의무)
    "prevent_vaccine_rabies": ConditionTagConfig(
        code="prevent_vaccine_rabies",
        label="예방접종 · 광견병",
        species="both",
        group="preventive",
        keywords=[
            "광견병", "Rabies", "광견병 주사", "보강 접종"
        ],
    ),

    # 5. 심장사상충 예방 (약/주사)
    "prevent_heartworm": ConditionTagConfig(
        code="prevent_heartworm",
        label="예방 · 심장사상충 예방",
        species="both", # 고양이도 하긴 함
        group="preventive",
        keywords=[
            "심장사상충 예방", "하트가드", "넥스가드", "애드보킷", "레볼루션",
            "프로하트", "Heartworm Prevention"
        ],
    ),

    # 6. 외부기생충 (진드기)
    "prevent_external_parasite": ConditionTagConfig(
        code="prevent_external_parasite",
        label="예방 · 외부기생충(진드기)",
        species="both",
        group="preventive",
        keywords=[
            "외부기생충", "진드기", "벼룩", "브라벡토", "프론트라인",
            "Flea", "Tick"
        ],
    ),

CONDITION_TAGS: Dict[str, ConditionTagConfig] = {
    # ------------------------------------------------------------------
    # 1) 피부 · 알레르기 (Dermatology)
    # ------------------------------------------------------------------
    "skin_atopy": ConditionTagConfig(
        code="skin_atopy",
        label="피부 · 아토피/알레르기",
        species="both",
        group="dermatology",
        keywords=[
            "아토피", "아토피피부염", "알레르기성 피부염", "알레르기 피부염",
            "환경 알레르기", "알레르기", "atopy", "atopic dermatitis",
            "allergic dermatitis"
        ],
    ),
    "skin_food_allergy": ConditionTagConfig(
        code="skin_food_allergy",
        label="피부 · 식이 알레르기",
        species="both",
        group="dermatology",
        keywords=[
            "식이 알레르기", "음식 알레르기", "Food allergy",
            "food hypersensitivity", "식이과민"
        ],
    ),
    "skin_flea_allergy": ConditionTagConfig(
        code="skin_flea_allergy",
        label="피부 · 벼룩 알레르기",
        species="both",
        group="dermatology",
        keywords=[
            "벼룩 알레르기", "벼룩알레르기", "벼룩 피부염",
            "flea allergy", "fleabite", "flea dermatitis"
        ],
    ),
    "skin_pyoderma": ConditionTagConfig(
        code="skin_pyoderma",
        label="피부 · 세균성 피부염(농피증)",
        species="both",
        group="dermatology",
        keywords=[
            "농피증", "세균성 피부염", "pyoderma",
            "bacterial dermatitis"
        ],
    ),
    "skin_malassezia": ConditionTagConfig(
        code="skin_malassezia",
        label="피부 · 말라세지아/곰팡이",
        species="both",
        group="dermatology",
        keywords=[
            "말라세지아", "곰팡이 피부염", "yeast dermatitis",
            "Malassezia", "진균성 피부염"
        ],
    ),
    "skin_demodex": ConditionTagConfig(
        code="skin_demodex",
        label="피부 · 모낭충(데모덱스)",
        species="dog",
        group="dermatology",
        keywords=[
            "모낭충", "데모덱스", "demodex", "demodicosis",
            "Demodex canis"
        ],
    ),
    "skin_sarcoptes": ConditionTagConfig(
        code="skin_sarcoptes",
        label="피부 · 옴(사르코프테스)",
        species="both",
        group="dermatology",
        keywords=[
            "옴", "사르코프테스", "sarcoptes", "sarcoptic mange",
            "scabies"
        ],
    ),
    "skin_pruritus": ConditionTagConfig(
        code="skin_pruritus",
        label="피부 · 만성 가려움",
        species="both",
        group="dermatology",
        keywords=[
            "만성 가려움", "소양증", "가려움증", "pruritus",
            "pruritic", "itching"
        ],
    ),
    "skin_alopecia": ConditionTagConfig(
        code="skin_alopecia",
        label="피부 · 탈모/털빠짐",
        species="both",
        group="dermatology",
        keywords=[
            "탈모", "털빠짐", "alopecia", "hair loss",
            "모발 소실"
        ],
    ),
    "ear_otitis_external": ConditionTagConfig(
        code="ear_otitis_external",
        label="귀 · 외이염",
        species="both",
        group="dermatology",
        keywords=[
            "외이염", "귀염", "귓병", "otitis externa",
            "ear infection"
        ],
    ),
    "ear_otitis_chronic": ConditionTagConfig(
        code="ear_otitis_chronic",
        label="귀 · 만성/재발성 귀질환",
        species="both",
        group="dermatology",
        keywords=[
            "만성 외이염", "재발성 외이염", "chronic otitis",
            "recurrent otitis"
        ],
    ),
    "skin_recurrent_infection": ConditionTagConfig(
        code="skin_recurrent_infection",
        label="피부 · 재발성 피부염",
        species="both",
        group="dermatology",
        keywords=[
            "재발성 피부염", "반복성 피부염", "recurrent dermatitis",
            "recurrent skin infection"
        ],
    ),

    # ------------------------------------------------------------------
    # 2) 심장 · 혈압 (Cardiology)
    # ------------------------------------------------------------------
    "heart_hcm_cat": ConditionTagConfig(
        code="heart_hcm_cat",
        label="심장 · 비대성 심근증(HCM, 고양이)",
        species="cat",
        group="cardiology",
        keywords=[
            "비대성 심근증", "HCM", "hypertrophic cardiomyopathy",
            "좌심실 비대", "좌심실비대"
        ],
    ),
    "heart_dcm_dog": ConditionTagConfig(
        code="heart_dcm_dog",
        label="심장 · 확장성 심근증(DCM, 개)",
        species="dog",
        group="cardiology",
        keywords=[
            "확장성 심근증", "DCM", "dilated cardiomyopathy"
        ],
    ),
    "heart_murmur": ConditionTagConfig(
        code="heart_murmur",
        label="심장 · 심잡음",
        species="both",
        group="cardiology",
        keywords=[
            "심잡음", "심장 잡음", "heart murmur", "murmur"
        ],
    ),
    "heart_valve_mitral": ConditionTagConfig(
        code="heart_valve_mitral",
        label="심장 · 승모판 질환",
        species="dog",
        group="cardiology",
        keywords=[
            "승모판", "승모판막", "승모판 역류", "mitral regurgitation",
            "mitral valve disease", "MVD"
        ],
    ),
    "heart_arrhythmia": ConditionTagConfig(
        code="heart_arrhythmia",
        label="심장 · 부정맥",
        species="both",
        group="cardiology",
        keywords=[
            "부정맥", "심실성 부정맥", "심방세동", "심방조동",
            "arrhythmia", "AF", "atrial fibrillation"
        ],
    ),
    "heart_chf": ConditionTagConfig(
        code="heart_chf",
        label="심장 · 심부전/폐수종",
        species="both",
        group="cardiology",
        keywords=[
            "심부전", "울혈성 심부전", "폐수종", "CHF",
            "congestive heart failure", "pulmonary edema"
        ],
    ),
    "heart_pulmonary_hypertension": ConditionTagConfig(
        code="heart_pulmonary_hypertension",
        label="심장 · 폐고혈압",
        species="both",
        group="cardiology",
        keywords=[
            "폐고혈압", "pulmonary hypertension"
        ],
    ),
    "heart_cardiomegaly": ConditionTagConfig(
        code="heart_cardiomegaly",
        label="심장 · 심비대/심장확대",
        species="both",
        group="cardiology",
        keywords=[
            "심비대", "심장 비대", "cardiomegaly", "심장 확대",
            "heart enlargement"
        ],
    ),
    "heart_heartworm": ConditionTagConfig(
        code="heart_heartworm",
        label="심장 · 심장사상충(감염/치료력)",
        species="dog",
        group="cardiology",
        keywords=[
            "심장사상충", "사상충", "heartworm", "Dirofilaria",
            "HW 감염", "HW 양성"
        ],
    ),
    "bp_hypertension": ConditionTagConfig(
        code="bp_hypertension",
        label="혈압 · 고혈압",
        species="both",
        group="cardiology",
        keywords=[
            "고혈압", "혈압 상승", "hypertension",
            "systemic hypertension"
        ],
    ),

    # ------------------------------------------------------------------
    # 3) 신장 · 요로 (Nephrology / Urology)
    # ------------------------------------------------------------------
    "kidney_ckd": ConditionTagConfig(
        code="kidney_ckd",
        label="신장 · 만성 신부전(CKD)",
        species="both",
        group="nephrology",
        keywords=[
            "만성 신부전", "만성 신장질환", "CKD", "chronic kidney disease",
            "요독증", "uremia"
        ],
    ),
    "kidney_aki": ConditionTagConfig(
        code="kidney_aki",
        label="신장 · 급성 신부전(AKI)",
        species="both",
        group="nephrology",
        keywords=[
            "급성 신부전", "AKI", "acute kidney injury"
        ],
    ),
    "kidney_proteinuria": ConditionTagConfig(
        code="kidney_proteinuria",
        label="신장 · 단백뇨/신장단백뇨",
        species="both",
        group="nephrology",
        keywords=[
            "단백뇨", "신장단백뇨", "proteinuria",
            "UPC 증가", "UP/C"
        ],
    ),
    "urinary_uti": ConditionTagConfig(
        code="urinary_uti",
        label="요로 · 요로감염(UTI)",
        species="both",
        group="urology",
        keywords=[
            "요로감염", "UTI", "urinary tract infection", "방광 감염"
        ],
    ),
    "urinary_cystitis": ConditionTagConfig(
        code="urinary_cystitis",
        label="요로 · 방광염",
        species="both",
        group="urology",
        keywords=[
            "방광염", "cystitis", "하부요로질환", "FLUTD"
        ],
    ),
    "urinary_stone": ConditionTagConfig(
        code="urinary_stone",
        label="요로 · 결석(방광/요관/요도)",
        species="both",
        group="urology",
        keywords=[
            "결석", "방광결석", "요석", "요로결석",
            "urolith", "urolithiasis", "struvite", "옥살산칼슘"
        ],
    ),
    "urinary_obstruction_cat": ConditionTagConfig(
        code="urinary_obstruction_cat",
        label="요로 · 요도폐색(수컷 고양이)",
        species="cat",
        group="urology",
        keywords=[
            "요도폐색", "요폐", "urethral obstruction",
            "FUS", "plug", "요폐색"
        ],
    ),
    "urinary_incontinence": ConditionTagConfig(
        code="urinary_incontinence",
        label="요로 · 요실금",
        species="both",
        group="urology",
        keywords=[
            "요실금", "소변 실금", "incontinence", "urinary incontinence"
        ],
    ),
    "kidney_pyelonephritis": ConditionTagConfig(
        code="kidney_pyelonephritis",
        label="신장 · 신우신염",
        species="both",
        group="nephrology",
        keywords=[
            "신우신염", "pyelonephritis"
        ],
    ),
    "kidney_congenital": ConditionTagConfig(
        code="kidney_congenital",
        label="신장 · 선천성 이상/형태 이상",
        species="both",
        group="nephrology",
        keywords=[
            "선천성 신장", "신장 형성이상", "renal dysplasia",
            "단신장", "horseshoe kidney"
        ],
    ),

    # ------------------------------------------------------------------
    # 4) 치과 · 구강 (Dentistry / Oral)
    # ------------------------------------------------------------------
    "dental_periodontal": ConditionTagConfig(
        code="dental_periodontal",
        label="치과 · 치주질환/치주염",
        species="both",
        group="dentistry",
        keywords=[
            "치주염", "치주질환", "periodontal disease",
            "periodontitis", "치주병"
        ],
    ),
    "dental_gingivitis": ConditionTagConfig(
        code="dental_gingivitis",
        label="치과 · 치은염/잇몸 염증",
        species="both",
        group="dentistry",
        keywords=[
            "치은염", "잇몸염", "gingivitis", "잇몸 염증"
        ],
    ),
    "dental_tooth_resorption_cat": ConditionTagConfig(
        code="dental_tooth_resorption_cat",
        label="치과 · 치아흡수병변(TR, 고양이)",
        species="cat",
        group="dentistry",
        keywords=[
            "치아흡수", "흡수병변", "TR", "tooth resorption",
            "FORL", "feline odontoclastic resorptive lesion"
        ],
    ),
    "dental_stomatitis_cat": ConditionTagConfig(
        code="dental_stomatitis_cat",
        label="치과 · 구내염/만성 구강염(고양이)",
        species="cat",
        group="dentistry",
        keywords=[
            "구내염", "구강염", "stomatitis", "chronic stomatitis",
            "FCGS", "feline chronic gingivostomatitis"
        ],
    ),
    "dental_fractured_tooth": ConditionTagConfig(
        code="dental_fractured_tooth",
        label="치과 · 치아 파절",
        species="both",
        group="dentistry",
        keywords=[
            "치아파절", "치아 골절", "fractured tooth", "tooth fracture"
        ],
    ),
    "dental_tartar": ConditionTagConfig(
        code="dental_tartar",
        label="치과 · 치석/플라그",
        species="both",
        group="dentistry",
        keywords=[
            "치석", "치태", "플라그", "tartar", "plaque"
        ],
    ),
    "dental_halitosis": ConditionTagConfig(
        code="dental_halitosis",
        label="치과 · 구취/입냄새",
        species="both",
        group="dentistry",
        keywords=[
            "구취", "입냄새", "halitosis", "oral malodor"
        ],
    ),
    "dental_post_extraction": ConditionTagConfig(
        code="dental_post_extraction",
        label="치과 · 발치/치과 수술 후 관리",
        species="both",
        group="dentistry",
        keywords=[
            "발치", "치아 발치", "tooth extraction", "dental surgery"
        ],
    ),

    # ------------------------------------------------------------------
    # 5) 정형 · 신경 · 근골격 (Orthopedics / Neurology)
    # ------------------------------------------------------------------
    "ortho_patella_luxation": ConditionTagConfig(
        code="ortho_patella_luxation",
        label="정형 · 슬개골 탈구",
        species="dog",
        group="orthopedics",
        keywords=[
            "슬개골 탈구", "슬개골탈구", "patella luxation",
            "patellar luxation", "PL"
        ],
    ),
    "ortho_hip_dysplasia": ConditionTagConfig(
        code="ortho_hip_dysplasia",
        label="정형 · 고관절 형성이상(HD)",
        species="dog",
        group="orthopedics",
        keywords=[
            "고관절 형성이상", "고관절이형성", "HD",
            "hip dysplasia"
        ],
    ),
    "ortho_ivdd": ConditionTagConfig(
        code="ortho_ivdd",
        label="정형/신경 · 추간판 질환(디스크)",
        species="both",
        group="orthopedics",
        keywords=[
            "디스크", "추간판 질환", "IVDD", "intervertebral disc disease",
            "추간판탈출"
        ],
    ),
    "ortho_cruciate_rupture": ConditionTagConfig(
        code="ortho_cruciate_rupture",
        label="정형 · 십자인대 파열(CCL)",
        species="dog",
        group="orthopedics",
        keywords=[
            "십자인대 파열", "십자인대파열", "CCL rupture",
            "cranial cruciate ligament"
        ],
    ),
    "ortho_arthritis": ConditionTagConfig(
        code="ortho_arthritis",
        label="정형 · 관절염/퇴행성 관절질환",
        species="both",
        group="orthopedics",
        keywords=[
            "관절염", "퇴행성 관절질환", "DJD", "osteoarthritis",
            "degenerative joint disease", "arthrosis"
        ],
    ),
    "ortho_lameness": ConditionTagConfig(
        code="ortho_lameness",
        label="정형 · 파행/절뚝거림",
        species="both",
        group="orthopedics",
        keywords=[
            "파행", "절뚝", "절뚝거림", "다리 절뚝", "lameness",
            "gait abnormality"
        ],
    ),
    "ortho_elbow_dysplasia": ConditionTagConfig(
        code="ortho_elbow_dysplasia",
        label="정형 · 주관절 형성이상",
        species="dog",
        group="orthopedics",
        keywords=[
            "주관절 형성이상", "팔꿈치 형성이상", "elbow dysplasia"
        ],
    ),
    "ortho_panosteitis": ConditionTagConfig(
        code="ortho_panosteitis",
        label="정형 · 성장통/골막염(파노스)",
        species="dog",
        group="orthopedics",
        keywords=[
            "파노스", "성장통", "골막염", "panosteitis"
        ],
    ),
    "neuro_seizure": ConditionTagConfig(
        code="neuro_seizure",
        label="신경 · 발작/경련",
        species="both",
        group="neurology",
        keywords=[
            "발작", "경련", "간질", "seizure", "epilepsy",
            "epileptic"
        ],
    ),
    "neuro_vestibular": ConditionTagConfig(
        code="neuro_vestibular",
        label="신경 · 전정질환/평형장애",
        species="both",
        group="neurology",
        keywords=[
            "전정질환", "평형장애", "머리 기울임", "vestibular disease",
            "head tilt", "vestibular syndrome"
        ],
    ),

    # ------------------------------------------------------------------
    # 6) 내분비 · 대사 (Endocrine / Metabolic)
    # ------------------------------------------------------------------
    "endocrine_diabetes_dog": ConditionTagConfig(
        code="endocrine_diabetes_dog",
        label="내분비 · 당뇨병(개)",
        species="dog",
        group="endocrine",
        keywords=[
            "당뇨병", "diabetes mellitus", "DM", "고혈당"
        ],
    ),
    "endocrine_diabetes_cat": ConditionTagConfig(
        code="endocrine_diabetes_cat",
        label="내분비 · 당뇨병(고양이)",
        species="cat",
        group="endocrine",
        keywords=[
            "당뇨병", "diabetes mellitus", "DM", "고혈당"
        ],
    ),
    "endocrine_hyperthyroidism_cat": ConditionTagConfig(
        code="endocrine_hyperthyroidism_cat",
        label="내분비 · 갑상선기능항진증(고양이)",
        species="cat",
        group="endocrine",
        keywords=[
            "갑상선 기능항진증", "갑상선항진", "hyperthyroidism",
            "T4 상승"
        ],
    ),
    "endocrine_hypothyroidism_dog": ConditionTagConfig(
        code="endocrine_hypothyroidism_dog",
        label="내분비 · 갑상선기능저하증(개)",
        species="dog",
        group="endocrine",
        keywords=[
            "갑상선 기능저하증", "갑상선저하", "hypothyroidism",
            "T4 감소"
        ],
    ),
    "endocrine_cushings": ConditionTagConfig(
        code="endocrine_cushings",
        label="내분비 · 쿠싱(Cushing) 증후군",
        species="dog",
        group="endocrine",
        keywords=[
            "쿠싱", "부신피질기능항진증", "Cushing", "hyperadrenocorticism"
        ],
    ),
    "endocrine_addisons": ConditionTagConfig(
        code="endocrine_addisons",
        label="내분비 · 애디슨(Addison) 병",
        species="dog",
        group="endocrine",
        keywords=[
            "애디슨", "부신피질기능저하증", "Addison", "hypoadrenocorticism"
        ],
    ),
    "endocrine_obesity": ConditionTagConfig(
        code="endocrine_obesity",
        label="대사 · 비만/과체중",
        species="both",
        group="endocrine",
        keywords=[
            "비만", "과체중", "obesity", "overweight", "body condition score"
        ],
    ),
    "endocrine_hyperlipidemia": ConditionTagConfig(
        code="endocrine_hyperlipidemia",
        label="대사 · 고지혈증",
        species="both",
        group="endocrine",
        keywords=[
            "고지혈증", "hyperlipidemia", "hypertriglyceridemia",
            "고콜레스테롤"
        ],
    ),
    "endocrine_pancreatitis": ConditionTagConfig(
        code="endocrine_pancreatitis",
        label="소화 · 췌장염",
        species="both",
        group="gastroenterology",
        keywords=[
            "췌장염", "pancreatitis", "cPL", "fPL", "췌장 염증"
        ],
    ),
    "endocrine_insulin_resistance": ConditionTagConfig(
        code="endocrine_insulin_resistance",
        label="내분비 · 인슐린 저항/조절 어려움",
        species="both",
        group="endocrine",
        keywords=[
            "인슐린 저항", "인슐린 조절 어려움", "insulin resistance",
            "insulin dose adjustment"
        ],
    ),

    # ------------------------------------------------------------------
    # 7) 소화기 · 간 (GI / Liver)
    # ------------------------------------------------------------------
    "gi_gastritis": ConditionTagConfig(
        code="gi_gastritis",
        label="소화기 · 위염/위장 염증",
        species="both",
        group="gastroenterology",
        keywords=[
            "위염", "gastritis", "위장염"
        ],
    ),
    "gi_enteritis": ConditionTagConfig(
        code="gi_enteritis",
        label="소화기 · 장염/장 염증",
        species="both",
        group="gastroenterology",
        keywords=[
            "장염", "enteritis", "소장염", "대장염"
        ],
    ),
    "gi_ibd": ConditionTagConfig(
        code="gi_ibd",
        label="소화기 · 만성장염/IBD",
        species="both",
        group="gastroenterology",
        keywords=[
            "만성 장염", "IBD", "inflammatory bowel disease",
            "만성 설사"
        ],
    ),
    "gi_vomiting_diarrhea": ConditionTagConfig(
        code="gi_vomiting_diarrhea",
        label="소화기 · 반복적인 구토/설사",
        species="both",
        group="gastroenterology",
        keywords=[
            "반복 구토", "반복적인 구토", "만성 설사", "vomiting",
            "diarrhea", "gastroenteritis"
        ],
    ),
    "liver_enzyme_elevated": ConditionTagConfig(
        code="liver_enzyme_elevated",
        label="간 · 간수치 상승(ALT/AST/ALP/GGT)",
        species="both",
        group="hepatology",
        keywords=[
            "간수치 상승", "ALT 상승", "AST 상승", "ALP 상승",
            "GGT 상승", "liver enzyme elevation"
        ],
    ),
    "liver_cholangitis": ConditionTagConfig(
        code="liver_cholangitis",
        label="간 · 담관염/담즙 정체",
        species="both",
        group="hepatology",
        keywords=[
            "담관염", "담즙 정체", "cholestasis", "cholangitis",
            "cholangiohepatitis"
        ],
    ),
    "liver_hepatic_lipidosis_cat": ConditionTagConfig(
        code="liver_hepatic_lipidosis_cat",
        label="간 · 지방간(고양이)",
        species="cat",
        group="hepatology",
        keywords=[
            "지방간", "hepatic lipidosis", "fatty liver"
        ],
    ),
    "liver_portosystemic_shunt": ConditionTagConfig(
        code="liver_portosystemic_shunt",
        label="간 · 문맥전신단락(PSS)",
        species="both",
        group="hepatology",
        keywords=[
            "문맥전신단락", "PSS", "portosystemic shunt",
            "선천성 단락"
        ],
    ),
    "gi_colitis": ConditionTagConfig(
        code="gi_colitis",
        label="소화기 · 대장염/혈변",
        species="both",
        group="gastroenterology",
        keywords=[
            "대장염", "colitis", "혈변", "혈성 설사"
        ],
    ),
    "gi_constipation": ConditionTagConfig(
        code="gi_constipation",
        label="소화기 · 변비/배변곤란",
        species="both",
        group="gastroenterology",
        keywords=[
            "변비", "배변곤란", "constipation", "megacolon"
        ],
    ),

    # ------------------------------------------------------------------
    # 8) 호흡기 · 비강 (Respiratory)
    # ------------------------------------------------------------------
    "resp_uri_cat": ConditionTagConfig(
        code="resp_uri_cat",
        label="호흡기 · 상부호흡기감염(고양이 감기)",
        species="cat",
        group="respiratory",
        keywords=[
            "상부호흡기감염", "URI", "cat flu", "칼리시", "헤르페스",
            "재채기", "콧물", "고양이 감기"
        ],
    ),
    "resp_tracheal_collapse": ConditionTagConfig(
        code="resp_tracheal_collapse",
        label="호흡기 · 기관허탈(기관지 협착)",
        species="dog",
        group="respiratory",
        keywords=[
            "기관허탈", "기관 허탈", "tracheal collapse",
            "거위 울음", "goose honk"
        ],
    ),
    "resp_bronchitis": ConditionTagConfig(
        code="resp_bronchitis",
        label="호흡기 · 기관지염/만성 기관지염",
        species="both",
        group="respiratory",
        keywords=[
            "기관지염", "bronchitis", "만성 기관지염", "chronic bronchitis"
        ],
    ),
    "resp_pneumonia": ConditionTagConfig(
        code="resp_pneumonia",
        label="호흡기 · 폐렴",
        species="both",
        group="respiratory",
        keywords=[
            "폐렴", "pneumonia", "aspiration pneumonia"
        ],
    ),
    "resp_asthma_cat": ConditionTagConfig(
        code="resp_asthma_cat",
        label="호흡기 · 고양이 천식",
        species="cat",
        group="respiratory",
        keywords=[
            "천식", "고양이 천식", "asthma", "feline asthma",
            "브롱코콘스트릭션"
        ],
    ),
    "resp_chronic_cough": ConditionTagConfig(
        code="resp_chronic_cough",
        label="호흡기 · 만성 기침",
        species="both",
        group="respiratory",
        keywords=[
            "만성 기침", "기침 지속", "chronic cough", "coughing"
        ],
    ),
    "resp_rhinitis": ConditionTagConfig(
        code="resp_rhinitis",
        label="호흡기 · 비염/콧물",
        species="both",
        group="respiratory",
        keywords=[
            "비염", "rhinitis", "콧물", "재채기"
        ],
    ),

    # ------------------------------------------------------------------
    # 9) 종양 · 혈액 (Oncology / Hematology / Immune)
    # ------------------------------------------------------------------
    "onco_lymphoma": ConditionTagConfig(
        code="onco_lymphoma",
        label="종양 · 림프종",
        species="both",
        group="oncology",
        keywords=[
            "림프종", "lymphoma", "다발성 림프절 비대"
        ],
    ),
    "onco_mct": ConditionTagConfig(
        code="onco_mct",
        label="종양 · 비만세포종(MCT)",
        species="dog",
        group="oncology",
        keywords=[
            "비만세포종", "mast cell tumor", "MCT"
        ],
    ),
    "onco_mammary_tumor": ConditionTagConfig(
        code="onco_mammary_tumor",
        label="종양 · 유선종양/유암",
        species="both",
        group="oncology",
        keywords=[
            "유선종양", "유선암", "유방종양", "mammary tumor",
            "mammary carcinoma"
        ],
    ),
    "onco_soft_tissue_sarcoma": ConditionTagConfig(
        code="onco_soft_tissue_sarcoma",
        label="종양 · 연부조직육종",
        species="both",
        group="oncology",
        keywords=[
            "연부조직육종", "soft tissue sarcoma", "STS"
        ],
    ),
    "onco_osteosarcoma": ConditionTagConfig(
        code="onco_osteosarcoma",
        label="종양 · 골육종",
        species="dog",
        group="oncology",
        keywords=[
            "골육종", "osteosarcoma", "OS"
        ],
    ),
    "onco_hemangiosarcoma": ConditionTagConfig(
        code="onco_hemangiosarcoma",
        label="종양 · 혈관육종",
        species="dog",
        group="oncology",
        keywords=[
            "혈관육종", "hemangiosarcoma", "HSA"
        ],
    ),
    "onco_tumor_unspecified": ConditionTagConfig(
        code="onco_tumor_unspecified",
        label="종양 · 기타/비특이 종양",
        species="both",
        group="oncology",
        keywords=[
            "종양", "신생물", "mass", "neoplasia", "腫瘍"
        ],
    ),
    "onco_leukemia": ConditionTagConfig(
        code="onco_leukemia",
        label="혈액 · 백혈병/골수 질환",
        species="both",
        group="hematology",
        keywords=[
            "백혈병", "leukemia", "골수형성", "bone marrow",
            "myeloproliferative"
        ],
    ),
    "heme_anemia": ConditionTagConfig(
        code="heme_anemia",
        label="혈액 · 빈혈",
        species="both",
        group="hematology",
        keywords=[
            "빈혈", "anemia", "낮은 PCV", "낮은 HCT", "저혈구"
        ],
    ),
    "heme_thrombocytopenia": ConditionTagConfig(
        code="heme_thrombocytopenia",
        label="혈액 · 혈소판 감소증",
        species="both",
        group="hematology",
        keywords=[
            "혈소판 감소", "혈소판감소증", "thrombocytopenia",
            "낮은 PLT"
        ],
    ),
    "heme_leukopenia": ConditionTagConfig(
        code="heme_leukopenia",
        label="혈액 · 백혈구 감소",
        species="both",
        group="hematology",
        keywords=[
            "백혈구 감소", "leukopenia", "낮은 WBC"
        ],
    ),
    "immune_imha": ConditionTagConfig(
        code="immune_imha",
        label="면역 · 용혈성 빈혈(IMHA)",
        species="both",
        group="immune",
        keywords=[
            "용혈성 빈혈", "IMHA", "immune mediated hemolytic anemia"
        ],
    ),
    "immune_itp": ConditionTagConfig(
        code="immune_itp",
        label="면역 · 혈소판 파괴성 질환(ITP)",
        species="both",
        group="immune",
        keywords=[
            "면역매개성 혈소판감소", "ITP",
            "immune mediated thrombocytopenia"
        ],
    ),
    "immune_autoimmune_unspecified": ConditionTagConfig(
        code="immune_autoimmune_unspecified",
        label="면역 · 자가면역질환(기타)",
        species="both",
        group="immune",
        keywords=[
            "자가면역", "autoimmune", "면역매개성",
            "면역질환"
        ],
    ),

    # ------------------------------------------------------------------
    # 10) 안과 · 기타 (Ophthalmology + General / Preventive)
    # ------------------------------------------------------------------
    "eye_cataract": ConditionTagConfig(
        code="eye_cataract",
        label="안과 · 백내장",
        species="both",
        group="ophthalmology",
        keywords=[
            "백내장", "cataract"
        ],
    ),
    "eye_glaucoma": ConditionTagConfig(
        code="eye_glaucoma",
        label="안과 · 녹내장/안압 상승",
        species="both",
        group="ophthalmology",
        keywords=[
            "녹내장", "안압 상승", "glaucoma", "high IOP"
        ],
    ),
    "eye_kcs": ConditionTagConfig(
        code="eye_kcs",
        label="안과 · 건성각결막염(KCS)",
        species="dog",
        group="ophthalmology",
        keywords=[
            "건성각결막염", "KCS", "dry eye", "keratoconjunctivitis sicca"
        ],
    ),
    "eye_conjunctivitis": ConditionTagConfig(
        code="eye_conjunctivitis",
        label="안과 · 결막염/안검염",
        species="both",
        group="ophthalmology",
        keywords=[
            "결막염", "conjunctivitis", "눈곱", "충혈"
        ],
    ),
    "eye_corneal_ulcer": ConditionTagConfig(
        code="eye_corneal_ulcer",
        label="안과 · 각막궤양/각막 손상",
        species="both",
        group="ophthalmology",
        keywords=[
            "각막궤양", "corneal ulcer", "각막 손상"
        ],
    ),
    "eye_retinal_disease": ConditionTagConfig(
        code="eye_retinal_disease",
        label="안과 · 망막질환/시력 저하",
        species="both",
        group="ophthalmology",
        keywords=[
            "망막박리", "망막질환", "retinal", "시력저하"
        ],
    ),

    "wellness_senior_screen": ConditionTagConfig(
        code="wellness_senior_screen",
        label="웰니스 · 노령 건강검진",
        species="both",
        group="preventive",
        keywords=[
            "노령검진", "시니어검진", "senior checkup",
            "senior profile", "노령 건강검진"
        ],
    ),
    "wellness_vaccine_core": ConditionTagConfig(
        code="wellness_vaccine_core",
        label="웰니스 · 기본 예방접종(혼합/광견병 등)",
        species="both",
        group="preventive",
        keywords=[
            "예방접종", "백신", "혼합백신", "DHPP", "DA2PP",
            "FVRCP", "광견병", "rabies"
        ],
    ),
    "wellness_vaccine_noncore": ConditionTagConfig(
        code="wellness_vaccine_noncore",
        label="웰니스 · 추가 예방접종(켄넬코프 등)",
        species="both",
        group="preventive",
        keywords=[
            "켄넬코프", "kc 백신", "lepto", "렙토스피라",
            "non-core vaccine"
        ],
    ),
    "wellness_heartworm_prevent": ConditionTagConfig(
        code="wellness_heartworm_prevent",
        label="웰니스 · 심장사상충 예방",
        species="dog",
        group="preventive",
        keywords=[
            "심장사상충 예방", "심장사상충 예방약", "heartworm prevention",
            "HW 예방"
        ],
    ),
    "wellness_flea_tick_prevent": ConditionTagConfig(
        code="wellness_flea_tick_prevent",
        label="웰니스 · 외부기생충 예방(벼룩/진드기)",
        species="both",
        group="preventive",
        keywords=[
            "벼룩 약", "진드기 약", "외부 기생충", "flea", "tick",
            "flea & tick", "external parasite"
        ],
    ),
    "wellness_spay_neuter": ConditionTagConfig(
        code="wellness_spay_neuter",
        label="웰니스 · 중성화 수술 관련",
        species="both",
        group="preventive",
        keywords=[
            "중성화", "중성화수술", "spay", "neuter",
            "ovariohysterectomy", "orchiectomy"
        ],
    ),
    "wellness_weight_management": ConditionTagConfig(
        code="wellness_weight_management",
        label="웰니스 · 체중 관리/다이어트",
        species="both",
        group="preventive",
        keywords=[
            "체중 관리", "다이어트", "식이조절", "weight management",
            "weight control"
        ],
    ),
    "wellness_bloodwork_routine": ConditionTagConfig(
        code="wellness_bloodwork_routine",
        label="웰니스 · 정기 혈액검사",
        species="both",
        group="preventive",
        keywords=[
            "정기 혈액검사", "wellness profile", "screening test",
            "routine blood test"
        ],
    ),
    "wellness_urinalysis_routine": ConditionTagConfig(
        code="wellness_urinalysis_routine",
        label="웰니스 · 정기 소변검사",
        species="both",
        group="preventive",
        keywords=[
            "소변검사", "urinalysis", "UA", "요검사"
        ],
    ),
}

_all_ = ["ConditionTagConfig", "CONDITION_TAGS", "SpeciesType"]
