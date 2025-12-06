"""
반려동물 질환/케어 태그 정의 파일.
AI 케어 분석에서 태그 매칭 및 케어 가이드 제공에 사용됨.
"""

from dataclasses import dataclass
from typing import List, Dict, Literal

SpeciesType = Literal["dog", "cat", "both"]


@dataclass(frozen=True)
class ConditionTagConfig:
    code: str            # 내부 코드 (diagnosis 등에 들어가는 코드)
    label: str           # 화면에 보여줄 이름
    species: SpeciesType # dog / cat / both
    group: str           # 상위 카테고리 (dermatology, orthopedics 등)
    keywords: List[str]  # 이 단어가 진료 텍스트에 들어있으면 매칭
    guide: List[str]     # 관리 팁 문구 리스트


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
            "skin_atopy", "아토피", "피부 알레르기", "알레르기",
            "atopy", "allergic dermatitis"
        ],
        guide=[
            "저자극 샴푸를 사용해주세요.",
            "알러지 유발 음식을 피해주세요.",
            "규칙적으로 빗질해 주세요."
        ],
    ),

    "skin_food_allergy": ConditionTagConfig(
        code="skin_food_allergy",
        label="피부 · 식이 알레르기",
        species="both",
        group="dermatology",
        keywords=[
            "skin_food_allergy", "식이 알레르기", "음식 알레르기",
            "food allergy",
        ],
        guide=[
            "문제가 되는 식재료를 메모해 두고 식단에서 빼 주세요.",
            "수의사와 식이 테스트 계획을 상의해 보세요.",
        ],
    ),

    "skin_pyoderma": ConditionTagConfig(
        code="skin_pyoderma",
        label="피부 · 세균성 피부염(농피증)",
        species="both",
        group="dermatology",
        keywords=[
            "skin_pyoderma", "농피증", "세균성 피부염", "pyoderma",
        ],
        guide=[
            "처방받은 항생제와 약욕은 중간에 끊지 말고 끝까지 진행해 주세요.",
            "피부가 젖어 있는 시간이 길지 않도록 관리해 주세요.",
        ],
    ),

    "skin_malassezia": ConditionTagConfig(
        code="skin_malassezia",
        label="피부 · 곰팡이성 피부염(말라세지아)",
        species="both",
        group="dermatology",
        keywords=[
            "skin_malassezia", "말라세지아", "곰팡이", "진균성",
            "malassezia", "yeast",
        ],
        guide=[
            "항진균 샴푸를 수의사 지시에 맞춰 사용해 주세요.",
            "귀와 발 사이 등 주로 생기는 부위를 자주 확인해 주세요.",
        ],
    ),

    "ear_otitis": ConditionTagConfig(
        code="ear_otitis",
        label="귀 · 외이염/귓병",
        species="both",
        group="dermatology",
        keywords=[
            "ear_otitis", "외이염", "귓병", "otitis", "ear infection",
        ],
        guide=[
            "귀 세정제는 너무 자주보다는, 수의사가 권장한 주기로 사용해 주세요.",
            "악취, 심한 가려움, 분비물이 보이면 바로 병원에 내원해 주세요.",
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
            "heart_murmur", "심잡음", "heart murmur", "murmur",
        ],
        guide=[
            "정기적인 심장 청진과 필요 시 초음파 검사를 받는 것이 좋아요.",
            "격한 운동보다는 가벼운 산책 위주로 활동량을 조절해 주세요.",
        ],
    ),

    "heart_mitral_valve": ConditionTagConfig(
        code="heart_mitral_valve",
        label="심장 · 승모판 질환(MVD)",
        species="dog",
        group="cardiology",
        keywords=[
            "heart_mitral_valve", "승모판", "mitral valve",
            "MVD", "MR",
        ],
        guide=[
            "수의사가 안내한 주기로 심장초음파 추적 검사를 진행해 주세요.",
            "기침, 호흡 곤란, 기력 저하가 보이면 바로 병원에 내원해야 해요.",
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
            # 진단 코드 / 키워드
            "ortho_patella",
            "슬개골", "슬개골 탈구",
            "patella", "luxating patella",
            "무릎 탈구", "파행",
        ],
        guide=[
            "집 안에 미끄럽지 않은 매트를 깔아 주세요.",
            "계단, 높은 점프는 가능한 한 피하는 것이 좋아요.",
            "체중이 늘지 않도록 관리해 주는 것이 관절에 큰 도움을 줍니다.",
        ],
    ),

    "ortho_arthritis": ConditionTagConfig(
        code="ortho_arthritis",
        label="관절 · 관절염",
        species="both",
        group="orthopedics",
        keywords=[
            "ortho_arthritis", "관절염", "arthritis",
            "퇴행성", "DJD",
        ],
        guide=[
            "체중 조절이 가장 중요한 관리 포인트예요.",
            "짧고 잦은 산책 위주로, 무리가 가지 않는 범위에서 움직이게 해 주세요.",
        ],
    ),

    # ---------------------------------------------------
    # 4) 예방접종 (진단 코드용 alias 포함)
    # ---------------------------------------------------
    # 영수증/진단 코드에 들어가는 문자열: vaccine_comprehensive
    "vaccine_comprehensive": ConditionTagConfig(
        code="vaccine_comprehensive",
        label="예방접종 · 종합백신(DHPPL 등)",
        species="both",
        group="preventive",
        keywords=[
            "vaccine_comprehensive",           # 진단 코드
            "prevent_vaccine_comprehensive",   # 예전 코드/alias
            "종합백신", "혼합백신",
            "DHPPL", "FVRCP",
        ],
        guide=[
            "다음 종합백신 예정일을 캘린더나 앱에 기록해 두면 좋아요.",
            "접종 후 24시간 동안은 식욕, 컨디션, 주사 부위 상태를 유심히 봐 주세요.",
        ],
    ),

    "prevent_vaccine_corona": ConditionTagConfig(
        code="prevent_vaccine_corona",
        label="예방접종 · 코로나 장염",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_corona",
            "코로나 백신", "코로나 장염",
            "corona enteritis",
        ],
        guide=[
            "예방접종 스케줄이 끊기지 않도록, 다음 접종일을 미리 예약해 두면 좋아요.",
        ],
    ),

    # ---------------------------------------------------
    # 5) 웰니스/건강검진
    # ---------------------------------------------------
    "wellness_checkup": ConditionTagConfig(
        code="wellness_checkup",
        label="웰니스 · 건강검진",
        species="both",
        group="wellness",
        keywords=[
            "wellness_checkup", "건강검진", "종합검진",
            "health check", "웰니스 체크업",
        ],
        guide=[
            "성견·성묘는 1년에 한 번 정기 건강검진을 권장해요.",
            "노령기에는 혈액검사, 영상검사를 포함한 검진 주기를 더 자주 잡아 주세요.",
        ],
    ),
}

_all_ = ["ConditionTagConfig", "CONDITION_TAGS", "SpeciesType"]
