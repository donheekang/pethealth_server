"""
반려동물 질환/케어 태그 정의 파일.
AI 케어 분석에서 태그 매칭 및 케어 가이드 제공에 사용됨.
"""

from dataclasses import dataclass
from typing import List, Dict, Literal

# 종 구분 타입
SpeciesType = Literal["dog", "cat", "both"]


@dataclass(frozen=True)
class ConditionTagConfig:
    code: str            # 내부 코드 (예: "ortho_patella")
    label: str           # 표시 이름 (예: "관절 · 슬개골 탈구")
    species: SpeciesType # 종 구분
    group: str           # 상위 그룹 (dermatology / cardiology / orthopedics / preventive / wellness ...)
    keywords: List[str]  # 진단명/기록에서 매칭할 키워드
    guide: List[str]     # 간단 케어 가이드 문구 리스트


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
            "skin_atopy",
            "아토피",
            "피부 알레르기",
            "알레르기",
            "atopy",
            "allergic dermatitis",
        ],
        guide=[
            "저자극 샴푸를 사용해주세요.",
            "알레르기를 유발하는 간식/사료는 피해주세요.",
            "규칙적인 빗질로 피부를 깨끗하게 유지해 주세요.",
        ],
    ),

    "skin_food_allergy": ConditionTagConfig(
        code="skin_food_allergy",
        label="피부 · 식이 알레르기",
        species="both",
        group="dermatology",
        keywords=[
            "skin_food_allergy",
            "식이 알레르기",
            "음식 알레르기",
            "food allergy",
        ],
        guide=[
            "새로운 간식·사료를 바꾼 시점을 메모해 두면 도움이 됩니다.",
            "수의사와 식이 제한(엘리미네이션 다이어트)을 상의해 보세요.",
        ],
    ),

    "skin_pyoderma": ConditionTagConfig(
        code="skin_pyoderma",
        label="피부 · 세균성 피부염(농피증)",
        species="both",
        group="dermatology",
        keywords=[
            "skin_pyoderma",
            "농피증",
            "세균성 피부염",
            "pyoderma",
        ],
        guide=[
            "처방 받은 항생제·약욕 스케줄을 정확히 지켜주세요.",
            "피부가 축축하게 젖어 있지 않도록 잘 말려주세요.",
        ],
    ),

    "skin_malassezia": ConditionTagConfig(
        code="skin_malassezia",
        label="피부 · 곰팡이성 피부염",
        species="both",
        group="dermatology",
        keywords=[
            "skin_malassezia",
            "말라세지아",
            "곰팡이",
            "진균성",
            "yeast",
        ],
        guide=[
            "항진균 샴푸나 약을 처방대로 사용해 주세요.",
            "귀·피부에서 냄새가 심해지면 다시 진료를 권장드려요.",
        ],
    ),

    "ear_otitis": ConditionTagConfig(
        code="ear_otitis",
        label="귀 · 외이염/귓병",
        species="both",
        group="dermatology",
        keywords=[
            "ear_otitis",
            "외이염",
            "귓병",
            "otitis",
            "ear infection",
        ],
        guide=[
            "주 1~2회 정도 귀 세정제를 사용해 귀를 관리해 주세요.",
            "빨갛게 붓거나 냄새·분비물이 늘면 바로 병원 방문이 필요합니다.",
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
            "heart_murmur",
            "심잡음",
            "heart murmur",
            "murmur",
        ],
        guide=[
            "정기적인 심장 초음파·청진 검사를 받는 것이 좋아요.",
            "숨이 가빠지거나 기침이 늘면 바로 병원에 연락하세요.",
        ],
    ),

    "heart_mitral_valve": ConditionTagConfig(
        code="heart_mitral_valve",
        label="심장 · 승모판 질환(MVD)",
        species="dog",
        group="cardiology",
        keywords=[
            "heart_mitral_valve",
            "승모판",
            "mitral valve",
            "MVD",
            "MR",
        ],
        guide=[
            "담당 수의사와 정기적인 심장 초음파 추적 계획을 세워 주세요.",
            "격한 운동보다는 짧고 잦은 산책 위주로 활동량을 조절해 주세요.",
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
            "ortho_patella",
            "슬개골",
            "슬개골 탈구",
            "무릎 탈구",
            "patella",
            "파행",
        ],
        guide=[
            "미끄럽지 않은 매트를 깔아서 미끄럼을 줄여주세요.",
            "소파·침대 점프, 계단 오르내리기는 최대한 제한해 주세요.",
            "체중을 적정하게 유지하는 것이 관절 관리에 가장 중요해요.",
        ],
    ),

    "ortho_arthritis": ConditionTagConfig(
        code="ortho_arthritis",
        label="관절 · 관절염",
        species="both",
        group="orthopedics",
        keywords=[
            "ortho_arthritis",
            "관절염",
            "퇴행성 관절염",
            "arthritis",
            "DJD",
        ],
        guide=[
            "짧고 자주 걷는 산책으로 관절을 부드럽게 유지해 주세요.",
            "관절 영양제·진통제는 수의사 지시에 따라 꾸준히 급여해 주세요.",
        ],
    ),

    # ---------------------------------------------------
    # 4) 예방접종 (종합백신 / 코로나 등)
    # ---------------------------------------------------
    # 진단명이 "vaccine_comprehensive" 로 들어오는 케이스를 잡기 위한 태그
    "vaccine_comprehensive": ConditionTagConfig(
        code="vaccine_comprehensive",
        label="예방접종 · 종합백신(DHPPL/FVRCP)",
        species="both",
        group="preventive",
        keywords=[
            # 진단 코드 그대로
            "vaccine_comprehensive",
            # 예전 코드/표현도 같이 묶어줌
            "prevent_vaccine_comprehensive",
            "종합백신",
            "혼합백신",
            "DHPPL",
            "FVRCP",
        ],
        guide=[
            "예방접종 스케줄을 캘린더나 앱에 기록해두면 놓치지 않아요.",
            "접종 후 24시간 동안은 식욕·기력·부종 여부를 잘 관찰해 주세요.",
        ],
    ),

    "prevent_vaccine_corona": ConditionTagConfig(
        code="prevent_vaccine_corona",
        label="예방접종 · 코로나 장염",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_corona",
            "코로나 장염",
            "코로나 백신",
            "corona enteritis",
        ],
        guide=[
            "어린 강아지는 백신 간격을 정확히 지키는 것이 중요해요.",
        ],
    ),

    # ---------------------------------------------------
    # 5) 웰니스 · 건강검진
    # ---------------------------------------------------
    "wellness_checkup": ConditionTagConfig(
        code="wellness_checkup",
        label="웰니스 · 건강검진",
        species="both",
        group="wellness",
        keywords=[
            "wellness_checkup",
            "건강검진",
            "종합검진",
            "health check",
        ],
        guide=[
            "성견·성묘는 1년에 한 번, 노령 반려동물은 6개월마다 검진을 권장드려요.",
        ],
    ),
}

# 외부에서 import 할 때 노출할 이름들
_all_ = ["ConditionTagConfig", "CONDITION_TAGS", "SpeciesType"]
