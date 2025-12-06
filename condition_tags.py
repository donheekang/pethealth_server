"""
반려동물 질환/케어 태그 정의 파일.
AI 케어 분석에서 태그 매칭 및 케어 가이드 제공에 사용됨.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Literal

# 강아지/고양이 구분용 타입
SpeciesType = Literal["dog", "cat", "both"]


@dataclass(frozen=True)
class ConditionTagConfig:
    code: str            # 내부 코드 (ex. "ortho_patella")
    label: str           # 화면에 보이는 이름
    species: SpeciesType # "dog" / "cat" / "both"
    group: str           # 상위 그룹 (dermatology, orthopedics, preventive, wellness ...)
    keywords: List[str]  # 진단/메모에서 찾을 키워드들
    guide: List[str]     # 관리 가이드 문구 리스트


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
            "atopy", "allergic dermatitis",
        ],
        guide=[
            "저자극 샴푸를 사용해 주세요.",
            "알레르기를 유발할 수 있는 간식·사료는 피해주세요.",
            "규칙적으로 빗질해 주면서 피부 상태를 확인해 주세요.",
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
            "문제가 된 음식/간식을 메모해 두고 식단에서 빼 주세요.",
            "수의사와 식이 제한 테스트 계획을 상의해 보세요.",
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
            "처방 받은 약욕과 약을 정해진 기간 동안 꾸준히 사용해 주세요.",
            "피부가 계속 젖어 있지 않도록 잘 말려 주세요.",
        ],
    ),

    "skin_malassezia": ConditionTagConfig(
        code="skin_malassezia",
        label="피부 · 곰팡이성 피부염",
        species="both",
        group="dermatology",
        keywords=[
            "skin_malassezia", "말라세지아", "곰팡이", "yeast", "진균성",
        ],
        guide=[
            "항진균 샴푸·약을 빼먹지 않고 사용해 주세요.",
            "귀·발 사이 등 습한 부위를 자주 확인해 주세요.",
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
            "귀 세정제를 주기적으로 사용해 주세요.",
            "귀를 긁거나 머리를 흔드는 행동이 늘면 바로 병원에 내원해 주세요.",
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
            "정기적인 심장 초음파와 흉부 X-ray 검사를 권장해요.",
            "갑자기 호흡이 빠르거나 기침이 늘면 바로 병원에 내원해 주세요.",
        ],
    ),

    "heart_mitral_valve": ConditionTagConfig(
        code="heart_mitral_valve",
        label="심장 · 승모판 질환(MVD)",
        species="dog",
        group="cardiology",
        keywords=[
            "heart_mitral_valve", "승모판", "mitral valve", "MVD", "MR",
        ],
        guide=[
            "수의사가 안내한 주기로 심장 초음파를 추적 검사해 주세요.",
            "숨이 가빠지거나, 잠자는 동안 호흡수가 늘어나면 바로 병원에 연락해 주세요.",
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
            "ortho_patella", "슬개골", "무릎 탈구", "patella", "파행",
        ],
        guide=[
            "집 안 바닥에는 미끄럽지 않은 매트를 깔아 주세요.",
            "계단 오르내리기, 소파·침대 점프는 최대한 제한해 주세요.",
            "체중 관리와 관절 영양제 급여를 보호자와 상의해 보세요.",
        ],
    ),

    "ortho_arthritis": ConditionTagConfig(
        code="ortho_arthritis",
        label="관절 · 관절염",
        species="both",
        group="orthopedics",
        keywords=[
            "ortho_arthritis", "관절염", "arthritis", "DJD", "퇴행성",
        ],
        guide=[
            "체중 조절이 관절 관리에서 가장 중요해요.",
            "짧고 잦은 산책으로 관절에 무리가 가지 않도록 운동량을 조절해 주세요.",
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
            "prevent_vaccine_comprehensive", "종합백신", "혼합백신",
            "DHPPL", "FVRCP",
        ],
        guide=[
            "종합백신은 일정 주기로 반복 접종이 필요해요.",
            "접종한 날짜와 다음 접종 예정일을 캘린더에 기록해 두세요.",
            "접종 후 24시간 정도는 컨디션·식욕 변화를 잘 관찰해 주세요.",
        ],
    ),

    # iOS에서 진단 코드로 쓰는 "vaccine_comprehensive" 에 대한 alias
    "vaccine_comprehensive": ConditionTagConfig(
        code="vaccine_comprehensive",
        label="예방접종 · 종합백신(DHPPL/FVRCP)",
        species="both",
        group="preventive",
        keywords=[
            "vaccine_comprehensive", "종합백신", "혼합백신",
            "DHPPL", "FVRCP",
        ],
        guide=[
            "예방접종 기록을 앱이나 수첩에 정리해 두면 좋아요.",
            "접종 후 무기력·구토·얼굴 부종 등이 보이면 즉시 병원으로 연락해 주세요.",
        ],
    ),

    "prevent_vaccine_corona": ConditionTagConfig(
        code="prevent_vaccine_corona",
        label="예방접종 · 코로나 장염",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_corona",
            "코로나 백신",
            "corona vaccine",
            "장염 예방",
        ],
        guide=[
            "코로나 장염 백신은 병원에서 안내하는 간격에 맞춰 접종해 주세요.",
            "접종 부위를 만졌을 때 통증이 심하거나 붓기가 크면 병원에 문의해 주세요.",
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
            "wellness_checkup", "건강검진", "종합검진", "health check",
        ],
        guide=[
            "성견/성묘는 1년에 한 번, 노령 아이는 6개월마다 건강검진을 권장해요.",
        ],
    ),
}

_all_ = ["ConditionTagConfig", "CONDITION_TAGS", "SpeciesType"]
