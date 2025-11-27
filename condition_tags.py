"""
반려동물 질환/케어 태그 정의 파일.
"""

from dataclasses import dataclass
from typing import List, Dict, Literal

SpeciesType = Literal["dog", "cat", "both"]


@dataclass(frozen=True)
class ConditionTagConfig:
    code: str            # 내부 코드
    label: str           # 표시 이름
    species: SpeciesType # 종 구분
    group: str           # 상위 그룹
    keywords: List[str]  # 매칭 키워드


# 딕셔너리 시작
CONDITION_TAGS: Dict[str, ConditionTagConfig] = {
    
    # 1) 피부 · 알레르기
    "skin_atopy": ConditionTagConfig(
        code="skin_atopy", label="피부 · 아토피/알레르기", species="both", group="dermatology",
        keywords=["아토피", "알레르기", "atopy", "allergic dermatitis"]
    ),
    "skin_food_allergy": ConditionTagConfig(
        code="skin_food_allergy", label="피부 · 식이 알레르기", species="both", group="dermatology",
        keywords=["식이 알레르기", "음식 알레르기", "Food allergy"]
    ),
    "skin_pyoderma": ConditionTagConfig(
        code="skin_pyoderma", label="피부 · 세균성 피부염(농피증)", species="both", group="dermatology",
        keywords=["농피증", "세균성 피부염", "pyoderma"]
    ),
    "skin_malassezia": ConditionTagConfig(
        code="skin_malassezia", label="피부 · 곰팡이성 피부염", species="both", group="dermatology",
        keywords=["말라세지아", "곰팡이", "yeast", "진균성"]
    ),
    "ear_otitis": ConditionTagConfig(
        code="ear_otitis", label="귀 · 외이염/귓병", species="both", group="dermatology",
        keywords=["외이염", "귓병", "otitis", "ear infection"]
    ),

    # 2) 심장
    "heart_murmur": ConditionTagConfig(
        code="heart_murmur", label="심장 · 심잡음", species="both", group="cardiology",
        keywords=["심잡음", "heart murmur", "murmur"]
    ),
    "heart_mitral_valve": ConditionTagConfig(
        code="heart_mitral_valve", label="심장 · 승모판 질환(MVD)", species="dog", group="cardiology",
        keywords=["승모판", "이첨판", "mitral valve", "MVD", "MR"]
    ),
    "heart_hcm": ConditionTagConfig(
        code="heart_hcm", label="심장 · 비대성 심근증(HCM)", species="cat", group="cardiology",
        keywords=["비대성 심근증", "HCM", "심근비대"]
    ),
    "heart_heartworm": ConditionTagConfig(
        code="heart_heartworm", label="심장 · 심장사상충 감염(치료중)", species="both", group="cardiology",
        keywords=["심장사상충 양성", "heartworm positive", "사상충 감염"]
    ),

    # 3) 신장/비뇨기
    "kidney_ckd": ConditionTagConfig(
        code="kidney_ckd", label="신장 · 만성 신부전(CKD)", species="both", group="nephrology",
        keywords=["만성 신부전", "CKD", "renal failure", "BUN 상승", "CREA 상승", "SDMA"]
    ),
    "urinary_stone": ConditionTagConfig(
        code="urinary_stone", label="요로 · 결석(방광/요도)", species="both", group="urology",
        keywords=["결석", "방광결석", "요석", "stone", "calculi", "struvite", "calcium oxalate"]
    ),
    "urinary_cystitis": ConditionTagConfig(
        code="urinary_cystitis", label="요로 · 방광염", species="both", group="urology",
        keywords=["방광염", "cystitis", "혈뇨", "오줌 소태"]
    ),
    "urinary_flutd": ConditionTagConfig(
        code="urinary_flutd", label="요로 · 하부요로질환(FLUTD/FIC)", species="cat", group="urology",
        keywords=["특발성 방광염", "FLUTD", "FIC", "배뇨곤란"]
    ),

    # 4) 관절/정형
    "ortho_patella": ConditionTagConfig(
        code="ortho_patella", label="관절 · 슬개골 탈구", species="dog", group="orthopedics",
        keywords=["슬개골", "patella", "무릎 탈구", "파행"]
    ),
    "ortho_arthritis": ConditionTagConfig(
        code="ortho_arthritis", label="관절 · 관절염", species="both", group="orthopedics",
        keywords=["관절염", "arthritis", "DJD", "퇴행성 관절"]
    ),
    "ortho_disk": ConditionTagConfig(
        code="ortho_disk", label="신경/관절 · 디스크(IVDD)", species="both", group="orthopedics",
        keywords=["디스크", "IVDD", "허리 통증", "마비"]
    ),

    # 5) 치과
    "dental_tartar": ConditionTagConfig(
        code="dental_tartar", label="치과 · 치석/치은염", species="both", group="dentistry",
        keywords=["치석", "스케일링", "치은염", "잇몸 염증", "scaling"]
    ),
    "dental_extraction": ConditionTagConfig(
        code="dental_extraction", label="치과 · 발치 치료", species="both", group="dentistry",
        keywords=["발치", "extraction", "이빨 뽑음"]
    ),
    "dental_resorption": ConditionTagConfig(
        code="dental_resorption", label="치과 · 치아흡수성병변(FORL)", species="cat", group="dentistry",
        keywords=["흡수병변", "FORL", "치아 흡수"]
    ),

    # 6) 소화기
    "gi_pancreatitis": ConditionTagConfig(
        code="gi_pancreatitis", label="소화기 · 췌장염", species="both", group="gastroenterology",
        keywords=["췌장염", "pancreatitis", "cPL", "fPL"]
    ),
    "gi_enteritis": ConditionTagConfig(
        code="gi_enteritis", label="소화기 · 장염/설사", species="both", group="gastroenterology",
        keywords=["장염", "설사", "enteritis", "diarrhea", "구토"]
    ),

    # 7) 예방접종 및 웰니스 (Preventive) - 상세 분리
    "prevent_vaccine_comprehensive": ConditionTagConfig(
        code="prevent_vaccine_comprehensive",
        label="예방접종 · 종합백신(DHPPL/FVRCP)",
        species="both",
        group="preventive",
        keywords=["종합백신", "혼합백신", "DHPPL", "DA2PP", "DHPP", "FVRCP", "4종백신", "5종백신"]
    ),
    "prevent_vaccine_corona": ConditionTagConfig(
        code="prevent_vaccine_corona",
        label="예방접종 · 코로나 장염",
        species="dog",
        group="preventive",
        keywords=["코로나 백신", "Corona", "Canine Coronavirus", "장염 예방"]
    ),
    "prevent_vaccine_kennel": ConditionTagConfig(
        code="prevent_vaccine_kennel",
        label="예방접종 · 켄넬코프",
        species="dog",
        group="preventive",
        keywords=["켄넬코프", "Kennel Cough", "기관지염 백신"]
    ),
    "prevent_vaccine_rabies": ConditionTagConfig(
        code="prevent_vaccine_rabies",
        label="예방접종 · 광견병",
        species="both",
        group="preventive",
        keywords=["광견병", "Rabies"]
    ),
    "prevent_heartworm": ConditionTagConfig(
        code="prevent_heartworm",
        label="예방 · 심장사상충 예방약",
        species="both",
        group="preventive",
        keywords=["심장사상충 예방", "하트가드", "넥스가드", "애드보킷", "레볼루션", "Heartworm Prev"]
    ),
    "prevent_external": ConditionTagConfig(
        code="prevent_external",
        label="예방 · 외부기생충(진드기)",
        species="both",
        group="preventive",
        keywords=["외부기생충", "진드기", "벼룩", "브라벡토", "프론트라인"]
    ),
    "wellness_checkup": ConditionTagConfig(
        code="wellness_checkup",
        label="웰니스 · 건강검진",
        species="both",
        group="preventive",
        keywords=["건강검진", "종합검진", "Checkup", "Health check"]
    ),
    "wellness_neuter": ConditionTagConfig(
        code="wellness_neuter",
        label="웰니스 · 중성화 수술",
        species="both",
        group="preventive",
        keywords=["중성화", "Castration", "Spay", "Neuter"]
    )
}

_all_ = ["ConditionTagConfig", "CONDITION_TAGS", "SpeciesType"]
