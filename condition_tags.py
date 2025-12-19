"""
반려동물 질환/케어 태그 정의 파일.

•⁠  ⁠AI 케어 분석(통계/요약)에서 record.tags(서버 코드) 매칭에 사용됩니다.
•⁠  ⁠"검사(exam) / 처치(procedure) / 약(medication)" 태그는 '질환 진단'으로 추론하지 않도록
  main.py 쪽에서 그룹별로 분리해서 다루는 것을 권장합니다. (여기는 정의만 담당)

✅ 권장: iOS에서 보내는 tags 는 아래의 "code(=snake_case)" 를 사용하세요.
•⁠  ⁠예: "exam_xray", "ortho_patella", "prevent_heartworm" ...

✅ 호환: 기존 앱에서 camelCase를 쓰고 있다면, 파일 하단의 ALIASES 로 같이 지원합니다.
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
    group: str           # 상위 그룹 (dermatology, orthopedics, cardiology, preventive, exam, medication, procedure, wellness ...)
    keywords: List[str]  # (fallback 매칭용) diagnosis/clinic/memo 에서 찾을 키워드들
    guide: List[str]     # (선택) 관리 가이드 문구 리스트


# ---------------------------------------------------
# TAG DEFINITIONS (✅ canonical: snake_case)
# ---------------------------------------------------

CONDITION_TAGS: Dict[str, ConditionTagConfig] = {
    # ---------------------------------------------------
    # 1) 피부 · 알레르기 / 귀
    # ---------------------------------------------------
    "skin_atopy": ConditionTagConfig(
        code="skin_atopy",
        label="피부 · 아토피/알레르기",
        species="both",
        group="dermatology",
        keywords=[
            "skin_atopy", "아토피", "피부 알레르기", "알레르기",
            "atopy", "allergic dermatitis", "dermatitis",
            "가려움", "소양감", "발적", "홍반",
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
            "food allergy", "dietary allergy",
        ],
        guide=[
            "문제가 된 음식/간식을 메모해 두고 식단에서 제외해 주세요.",
            "수의사와 식이 제한(엘리미네이션) 테스트 계획을 상의해 보세요.",
        ],
    ),

    "skin_pyoderma": ConditionTagConfig(
        code="skin_pyoderma",
        label="피부 · 세균성 피부염(농피증)",
        species="both",
        group="dermatology",
        keywords=[
            "skin_pyoderma", "농피증", "세균성 피부염", "pyoderma",
            "pustule", "고름", "농포",
        ],
        guide=[
            "처방 받은 약욕/약을 정해진 기간 동안 꾸준히 사용해 주세요.",
            "피부가 계속 젖어 있지 않도록 잘 말려 주세요.",
        ],
    ),

    "skin_malassezia": ConditionTagConfig(
        code="skin_malassezia",
        label="피부 · 곰팡이성 피부염(말라세지아)",
        species="both",
        group="dermatology",
        keywords=[
            "skin_malassezia", "말라세지아", "곰팡이", "yeast",
            "진균성", "malassezia",
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
            "귀 염증", "귀 냄새", "귀 분비물",
        ],
        guide=[
            "귀 세정제를 주기적으로 사용해 주세요.",
            "귀를 심하게 긁거나 머리를 흔드는 행동이 늘면 병원에 내원해 주세요.",
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
            "잡음", "심장잡음",
        ],
        guide=[
            "정기적인 심장 초음파/흉부 방사선 검사 주기를 수의사와 상의해 주세요.",
            "호흡이 갑자기 빨라지거나 기침이 늘면 바로 병원에 문의해 주세요.",
        ],
    ),

    "heart_mitral_valve": ConditionTagConfig(
        code="heart_mitral_valve",
        label="심장 · 승모판 질환(MVD)",
        species="dog",
        group="cardiology",
        keywords=[
            "heart_mitral_valve", "승모판", "mitral valve", "MVD", "MR",
            "승모판폐쇄부전",
        ],
        guide=[
            "수의사가 안내한 주기로 심장 초음파를 추적 검사해 주세요.",
            "잠자는 동안 호흡수가 늘어나면 기록해 두고 상담해 주세요.",
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
            "ortho_patella", "슬개골", "슬개골탈구", "무릎 탈구",
            "patella", "파행", "절뚝",
        ],
        guide=[
            "집 안 바닥에는 미끄럽지 않은 매트를 깔아 주세요.",
            "계단 오르내리기/높은 점프는 최대한 제한해 주세요.",
            "체중 관리와 관절 영양제 급여는 수의사와 상의해 보세요.",
        ],
    ),

    "ortho_arthritis": ConditionTagConfig(
        code="ortho_arthritis",
        label="관절 · 관절염",
        species="both",
        group="orthopedics",
        keywords=[
            "ortho_arthritis", "관절염", "arthritis", "DJD", "퇴행성",
            "골관절염", "degenerative",
        ],
        guide=[
            "체중 조절이 관절 관리에서 가장 중요해요.",
            "짧고 잦은 산책으로 관절에 무리가 가지 않게 운동량을 조절해 주세요.",
        ],
    ),

    # ---------------------------------------------------
    # 4) 예방 (백신/기생충)
    # ---------------------------------------------------
    "prevent_vaccine_comprehensive": ConditionTagConfig(
        code="prevent_vaccine_comprehensive",
        label="예방접종 · 종합백신(DHPPL/FVRCP)",
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

    "prevent_vaccine_corona": ConditionTagConfig(
        code="prevent_vaccine_corona",
        label="예방접종 · 코로나 장염(개)",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_corona", "코로나 백신", "corona vaccine", "장염 예방",
        ],
        guide=[
            "접종 간격은 병원 안내에 맞춰 주세요.",
            "접종 후 붓기/통증이 심하거나 구토·무기력이 지속되면 병원에 문의해 주세요.",
        ],
    ),

    "prevent_vaccine_kennel": ConditionTagConfig(
        code="prevent_vaccine_kennel",
        label="예방접종 · 켄넬코프",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_kennel", "켄넬코프", "기관지염 백신", "kennel cough",
        ],
        guide=[
            "접종 후 1~2일은 컨디션 변화를 관찰해 주세요.",
            "기침이 늘거나 호흡이 불편해 보이면 병원에 문의해 주세요.",
        ],
    ),

    "prevent_vaccine_rabies": ConditionTagConfig(
        code="prevent_vaccine_rabies",
        label="예방접종 · 광견병",
        species="both",
        group="preventive",
        keywords=[
            "prevent_vaccine_rabies", "광견병", "rabies",
        ],
        guide=[
            "지역/상황에 따라 접종 주기가 달라질 수 있어요. 병원 안내에 맞춰 기록해 주세요.",
        ],
    ),

    "prevent_heartworm": ConditionTagConfig(
        code="prevent_heartworm",
        label="예방 · 심장사상충",
        species="both",
        group="preventive",
        keywords=[
            "prevent_heartworm", "사상충", "심장사상충", "heartworm",
        ],
        guide=[
            "매달 같은 날짜에 예방약을 챙기면 잊기 쉬운 달을 줄일 수 있어요.",
            "구토/설사 등 이상 반응이 있으면 복용 중단 후 병원에 문의해 주세요.",
        ],
    ),

    "prevent_external": ConditionTagConfig(
        code="prevent_external",
        label="예방 · 외부기생충(진드기/벼룩)",
        species="both",
        group="preventive",
        keywords=[
            "prevent_external", "외부기생충", "진드기", "벼룩",
            "tick", "flea",
        ],
        guide=[
            "산책이 잦은 계절엔 외부기생충 예방을 꾸준히 유지해 주세요.",
            "피부 가려움/발진이 심해지면 병원에 문의해 주세요.",
        ],
    ),

    "prevent_deworming": ConditionTagConfig(
        code="prevent_deworming",
        label="예방 · 구충",
        species="both",
        group="preventive",
        keywords=[
            "prevent_deworming", "구충", "내부기생충", "deworm",
        ],
        guide=[
            "구충 주기는 아이의 생활 환경에 따라 달라질 수 있어요. 병원 안내에 맞춰 기록해 주세요.",
        ],
    ),

    # ---------------------------------------------------
    # 5) 검사 (✅ 질환 ‘추론’ 금지 권장: main.py에서 group='exam' 분리)
    # ---------------------------------------------------
    "exam_xray": ConditionTagConfig(
        code="exam_xray",
        label="검사 · 엑스레이",
        species="both",
        group="exam",
        keywords=[
            "exam_xray", "엑스레이", "X-ray", "x-ray", "xray",
            "방사선", "radiograph", "radiography",
        ],
        guide=[
            "검사 기록이에요. 결과(정상/이상 소견)를 한 줄로 함께 남기면 다음 요약이 더 정확해져요.",
        ],
    ),

    "exam_blood": ConditionTagConfig(
        code="exam_blood",
        label="검사 · 혈액검사",
        species="both",
        group="exam",
        keywords=[
            "exam_blood", "혈액검사", "혈검", "CBC", "Chemistry", "blood test",
        ],
        guide=[
            "혈액검사는 상태를 확인하는 검사 기록이에요. 병원 소견을 짧게 메모해 두면 좋아요.",
        ],
    ),

    "exam_ultrasound": ConditionTagConfig(
        code="exam_ultrasound",
        label="검사 · 초음파",
        species="both",
        group="exam",
        keywords=[
            "exam_ultrasound", "초음파", "ultrasound", "sono",
        ],
        guide=[
            "초음파는 진단을 위한 검사 기록이에요. 검사 목적/소견을 함께 남겨두면 도움이 돼요.",
        ],
    ),

    "exam_lab_panel": ConditionTagConfig(
        code="exam_lab_panel",
        label="검사 · 종합검사",
        species="both",
        group="exam",
        keywords=[
            "exam_lab_panel", "종합검사", "종합검진", "패널", "panel", "lab panel",
        ],
        guide=[
            "종합검사는 변화 추적이 중요해요. 이전 결과와 비교 메모를 남겨두면 좋아요.",
        ],
    ),

    # ---------------------------------------------------
    # 6) 약(처방) / 처치 / 치과 / 수술  (✅ 질환과 분리 권장)
    # ---------------------------------------------------
    "medicine_antibiotic": ConditionTagConfig(
        code="medicine_antibiotic",
        label="처방 · 항생제",
        species="both",
        group="medication",
        keywords=["medicine_antibiotic", "항생제", "antibiotic"],
        guide=["처방 받은 기간을 지켜 복용해 주세요. 이상 반응이 있으면 병원에 문의해 주세요."],
    ),

    "medicine_anti_inflammatory": ConditionTagConfig(
        code="medicine_anti_inflammatory",
        label="처방 · 소염/항염",
        species="both",
        group="medication",
        keywords=["medicine_anti_inflammatory", "소염", "항염", "NSAIDs", "anti-inflammatory"],
        guide=["식욕 저하/구토 등 이상 증상이 있으면 복용을 멈추고 병원에 문의해 주세요."],
    ),

    "medicine_allergy": ConditionTagConfig(
        code="medicine_allergy",
        label="처방 · 알러지 약",
        species="both",
        group="medication",
        keywords=["medicine_allergy", "알러지", "항히스타민", "antihistamine"],
        guide=["증상이 심해지거나 졸림이 심하면 병원에 문의해 주세요."],
    ),

    "medicine_gi": ConditionTagConfig(
        code="medicine_gi",
        label="처방 · 위장/장",
        species="both",
        group="medication",
        keywords=["medicine_gi", "위장", "장", "설사", "구토", "probiotic"],
        guide=["물/식사량 변화를 함께 관찰해 주세요. 증상이 지속되면 병원에 문의해 주세요."],
    ),

    "medicine_ear": ConditionTagConfig(
        code="medicine_ear",
        label="처방 · 귀",
        species="both",
        group="medication",
        keywords=["medicine_ear", "귀약", "ear drops", "ear medication"],
        guide=["귀 세정/점이 지침이 있다면 횟수를 지켜 사용해 주세요."],
    ),

    "medicine_skin": ConditionTagConfig(
        code="medicine_skin",
        label="처방 · 피부",
        species="both",
        group="medication",
        keywords=["medicine_skin", "피부약", "ointment", "topical"],
        guide=["피부가 악화되는 부위가 있으면 사진으로 기록해 두면 진료에 도움이 돼요."],
    ),

    "medicine_eye": ConditionTagConfig(
        code="medicine_eye",
        label="처방 · 안약",
        species="both",
        group="medication",
        keywords=["medicine_eye", "안약", "eye drops", "ophthalmic"],
        guide=["점안 횟수/간격을 지켜 주세요. 눈을 심하게 비비면 병원에 문의해 주세요."],
    ),

    "medicine_painkiller": ConditionTagConfig(
        code="medicine_painkiller",
        label="처방 · 진통",
        species="both",
        group="medication",
        keywords=["medicine_painkiller", "진통", "painkiller", "analgesic"],
        guide=["통증 징후(절뚝/숨김/예민함)가 줄어드는지 함께 관찰해 주세요."],
    ),

    "dental_scaling": ConditionTagConfig(
        code="dental_scaling",
        label="치과 · 스케일링",
        species="both",
        group="procedure",
        keywords=["dental_scaling", "스케일링", "치석 제거", "scaling"],
        guide=["시술 후 며칠간은 식욕/통증 반응을 관찰해 주세요. 양치 루틴을 천천히 시작해 보세요."],
    ),

    "dental_extraction": ConditionTagConfig(
        code="dental_extraction",
        label="치과 · 발치",
        species="both",
        group="procedure",
        keywords=["dental_extraction", "발치", "extraction"],
        guide=["처방 약을 잘 챙겨 주세요. 출혈/통증이 심하면 병원에 문의해 주세요."],
    ),

    "surgery_general": ConditionTagConfig(
        code="surgery_general",
        label="수술 · 일반",
        species="both",
        group="procedure",
        keywords=["surgery_general", "수술", "operation", "surgery"],
        guide=["수술 후 회복은 아이마다 달라요. 활력/식욕/상처 상태를 함께 기록해 두면 좋아요."],
    ),

    # ---------------------------------------------------
    # 7) 웰니스
    # ---------------------------------------------------
    "wellness_checkup": ConditionTagConfig(
        code="wellness_checkup",
        label="웰니스 · 건강검진",
        species="both",
        group="wellness",
        keywords=[
            "wellness_checkup", "건강검진", "종합검진", "health check",
            "check-up", "checkup",
        ],
        guide=[
            "성견/성묘는 1년에 한 번, 노령 아이는 6개월마다 건강검진을 권장해요.",
        ],
    ),
}


# ---------------------------------------------------
# ALIASES (optional, for backward compatibility)
# - key: 들어올 수 있는 다른 코드(예: camelCase)
# - value: canonical snake_case key
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

    # preventive
    "preventVaccineComprehensive": "prevent_vaccine_comprehensive",
    "vaccineComprehensive": "prevent_vaccine_comprehensive",
    "vaccine_comprehensive": "prevent_vaccine_comprehensive",

    "preventVaccineCorona": "prevent_vaccine_corona",
    "vaccineCorona": "prevent_vaccine_corona",

    "vaccineKennel": "prevent_vaccine_kennel",
    "vaccineRabies": "prevent_vaccine_rabies",

    "preventHeartworm": "prevent_heartworm",
    "preventExternal": "prevent_external",
    "preventDeworming": "prevent_deworming",

    # exam
    "examXray": "exam_xray",
    "examBlood": "exam_blood",
    "examUltrasound": "exam_ultrasound",
    "examLabPanel": "exam_lab_panel",

    # medication / procedure
    "medicineAntibiotic": "medicine_antibiotic",
    "medicineAntiInflammatory": "medicine_anti_inflammatory",
    "medicineAllergy": "medicine_allergy",
    "medicineGI": "medicine_gi",
    "medicineEar": "medicine_ear",
    "medicineSkin": "medicine_skin",
    "medicineEye": "medicine_eye",
    "medicinePainkiller": "medicine_painkiller",

    "dentalScaling": "dental_scaling",
    "dentalExtraction": "dental_extraction",
    "surgeryGeneral": "surgery_general",

    # wellness
    "wellnessCheckup": "wellness_checkup",
}

for alias_key, canonical_key in ALIASES.items():
    if canonical_key in CONDITION_TAGS:
        CONDITION_TAGS[alias_key] = CONDITION_TAGS[canonical_key]


_all_ = ["ConditionTagConfig", "CONDITION_TAGS", "SpeciesType"]
