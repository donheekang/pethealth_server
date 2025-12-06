"""
반려동물 질환/케어 태그 정의 파일.
AI 케어 분석(Gemini)에서 태그 매칭 및 케어 가이드 제공에 사용됨.
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
    keywords: List[str]  # 태그 매칭 키워드
    guide: List[str]     # 케어 가이드 문구 리스트


# ---------------------------------------------------
# TAG DEFINITIONS
# ---------------------------------------------------

CONDITION_TAGS: Dict[str, ConditionTagConfig] = {

    # ---------------------------------------------------
    # 1) 피부 · 알레르기
    # ---------------------------------------------------
    "skin_atopy": ConditionTagConfig(
        code="skin_atopy",
        label="피부 · 아토피/알레르기",
        species="both",
        group="dermatology",
        keywords=[
            "skin_atopy", "아토피", "피부 알레르기", "알레르기", "atopy", "allergic dermatitis"
        ],
        guide=[
            "저자극 샴푸를 사용해주세요.",
            "알러지 유발 음식은 피해주세요.",
            "빗질을 규칙적으로 해주세요.",
        ],
    ),

    "skin_food_allergy": ConditionTagConfig(
        code="skin_food_allergy",
        label="피부 · 식이 알레르기",
        species="both",
        group="dermatology",
        keywords=[
            "skin_food_allergy", "식이 알레르기", "음식 알레르기", "food allergy"
        ],
        guide=[
            "문제가 되는 식재료를 기록하고 제거해주세요.",
            "수의사와 식이 테스트를 상의해보세요.",
        ],
    ),

    "skin_pyoderma": ConditionTagConfig(
        code="skin_pyoderma",
        label="피부 · 세균성 피부염(농피증)",
        species="both",
        group="dermatology",
        keywords=[
            "skin_pyoderma", "농피증", "세균성 피부염", "pyoderma"
        ],
        guide=[
            "약욕 처방을 꾸준히 따라주세요.",
            "피부가 젖지 않도록 관리해주세요.",
        ],
    ),

    "skin_malassezia": ConditionTagConfig(
        code="skin_malassezia",
        label="피부 · 곰팡이성 피부염",
        species="both",
        group="dermatology",
        keywords=[
            "skin_malassezia", "말라세지아", "곰팡이", "yeast", "진균성"
        ],
        guide=[
            "항진균 샴푸와 처방을 따라주세요.",
            "피부 상태를 꾸준히 관찰해주세요.",
        ],
    ),

    "ear_otitis": ConditionTagConfig(
        code="ear_otitis",
        label="귀 · 외이염/귓병",
        species="both",
        group="dermatology",
        keywords=[
            "ear_otitis", "외이염", "귓병", "otitis", "ear infection"
        ],
        guide=[
            "귀 세정제를 규칙적으로 사용해주세요.",
            "귀 털이 많은 경우 관리가 필요할 수 있어요.",
        ],
    ),

    # ---------------------------------------------------
    # 2) 심장
    # ---------------------------------------------------
    "heart_murmur": ConditionTagConfig(
        code="heart_murmur",
        label="심장 · 심잡음",
        species="both",
        group="cardiology",
        keywords=[
            "heart_murmur", "심잡음", "heart murmur", "murmur"
        ],
        guide=[
            "정기적인 심장초음파 검사가 필요해요.",
            "운동 강도는 무리가 가지 않게 조절해주세요.",
        ],
    ),

    "heart_mitral_valve": ConditionTagConfig(
        code="heart_mitral_valve",
        label="심장 · 승모판 질환(MVD)",
        species="dog",
        group="cardiology",
        keywords=[
            "heart_mitral_valve", "승모판", "mitral valve", "MVD", "MR"
        ],
        guide=[
            "정기적으로 심장 초음파 추적 검사를 진행하세요.",
            "기침, 호흡 변화가 생기면 즉시 병원 방문이 필요합니다.",
        ],
    ),

    # ---------------------------------------------------
    # 3) 관절/정형
    # ---------------------------------------------------
    "ortho_patella": ConditionTagConfig(
        code="ortho_patella",
        label="관절 · 슬개골 탈구",
        species="dog",
        group="orthopedics",
        keywords=[
            "ortho_patella", "슬개골", "patella", "무릎 탈구", "파행"
        ],
        guide=[
            "미끄럽지 않은 매트를 깔아주세요.",
            "계단이나 점프는 제한해주세요.",
            "관절 영양제를 고려해보세요.",
        ],
    ),

    "ortho_arthritis": ConditionTagConfig(
        code="ortho_arthritis",
        label="관절 · 관절염",
        species="both",
        group="orthopedics",
        keywords=[
            "ortho_arthritis", "관절염", "arthritis", "DJD", "퇴행성"
        ],
        guide=[
            "체중 조절이 가장 중요합니다.",
            "산책은 무리가 가지 않는 선에서 규칙적으로 진행해주세요.",
        ],
    ),

    # ---------------------------------------------------
    # 4) 예방접종
    # ---------------------------------------------------
    "prevent_vaccine_comprehensive": ConditionTagConfig(
        code="prevent_vaccine_comprehensive",
        label="예방접종 · 종합백신",
        species="both",
        group="preventive",
        keywords=[
            "prevent_vaccine_comprehensive", "종합백신", "DHPPL", "4종백신", "혼합백신"
        ],
        guide=[
            "정기적인 백신 스케줄을 확인해주세요.",
            "접종 후 24시간은 몸 상태를 관찰해주세요.",
        ],
    ),

    "prevent_vaccine_corona": ConditionTagConfig(
        code="prevent_vaccine_corona",
        label="예방접종 · 코로나 장염",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_corona", "코로나 백신", "Corona", "장염 예방"
        ],
        guide=[
            "접종 날짜를 놓치지 않도록 기록해두세요.",
        ],
    ),

    # 진료 코드용 alias (diagnosis: "vaccine_comprehensive")
    "vaccine_comprehensive": ConditionTagConfig(
        code="vaccine_comprehensive",
        label="예방접종 · 종합백신(DHPPL/FVRCP)",
        species="both",
        group="preventive",
        keywords=[
            "vaccine_comprehensive", "종합백신", "혼합백신", "DHPPL", "FVRCP"
        ],
        guide=[
            "정기적인 백신 스케줄을 캘린더에 기록해두세요.",
            "접종 후 1~2일 동안은 기력과 식욕을 잘 살펴봐 주세요.",
        ],
    ),

    # ---------------------------------------------------
    # 5) 기타 웰니스
    # ---------------------------------------------------
    "wellness_checkup": ConditionTagConfig(
        code="wellness_checkup",
        label="웰니스 · 건강검진",
        species="both",
        group="wellness",
        keywords=[
            "wellness_checkup", "건강검진", "종합검진", "health check"
        ],
        guide=[
            "성견/성묘는 1년에 한 번 건강검진을 권장해요.",
        ],
    ),
}

_all_ = ["ConditionTagConfig", "CONDITION_TAGS", "SpeciesType"]
