"""
반려동물 질환/케어 태그 정의 파일.

•  AI 케어 분석(통계/요약)에서 record.tags(서버 코드) 매칭에 사용됩니다.
•  "검사(exam) / 처치(procedure) / 약(medication)" 태그는 '질환 진단'으로 추론하지 않도록
  main.py 쪽에서 그룹별로 분리해서 다루는 것을 권장합니다. (여기는 정의만 담당)

✅ 권장: iOS에서 보내는 tags 는 아래의 "code(=snake_case)" 를 사용하세요.
•  예: "exam_xray", "ortho_patella", "prevent_heartworm" ...

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
            "skin_atopy",
            # 한글
            "아토피", "피부 알레르기", "알레르기", "알러지", "알레르기성 피부염",
            "가려움", "소양감", "발적", "홍반", "습진", "피부염", "가려워", "긁음",
            # 영문/약어
            "atopy", "allergic dermatitis", "dermatitis", "pruritus", "itch",
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
            "skin_food_allergy",
            "식이 알레르기", "음식 알레르기", "사료 알레르기", "단백질 알레르기",
            "food allergy", "dietary allergy", "food hypersensitivity",
            "elimination diet", "limited ingredient",
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
            "skin_pyoderma",
            "농피증", "세균성 피부염", "세균성", "pyoderma",
            "pustule", "고름", "농포", "모낭염", "folliculitis",
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
            "skin_malassezia",
            "말라세지아", "곰팡이", "진균", "진균성",
            "yeast", "malassezia", "malassezia pachydermatis",
            "fungal", "fungus",
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
            "ear_otitis",
            "외이염", "귓병", "귀 염증", "귀냄새", "귀 냄새", "귀 분비물", "귀 진물",
            "otitis", "otitis externa", "ear infection",
            "ear discharge", "ear wax",
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
            "heart_murmur",
            "심잡음", "심장잡음", "잡음",
            "heart murmur", "murmur",
        ],
        guide=[
            "정기적인 심장 초음파/흉부 방사선 검사 주기를 수의사와 상의해 주세요.",
            "호흡이 갑자기 빨라지거나 기침이 늘면 바로 병원에 문의해 주세요.",
        ],
    ),

    "heart_mitral_valve": ConditionTagConfig(
        code="heart_mitral_valve",
        label="심장 · 승모판 질환(MVD/MMVD)",
        species="dog",
        group="cardiology",
        keywords=[
            "heart_mitral_valve",
            "승모판", "승모판폐쇄부전", "승모판 질환",
            "mitral valve", "MVD", "MMVD", "MR",
            "myxomatous", "degenerative mitral valve",
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
            "ortho_patella",
            "슬개골", "슬개골탈구", "무릎 탈구", "무릎탈구", "슬탈",
            "patella", "patellar", "patellar luxation", "patella luxation",
            "MPL", "LPL", "PL",
            "파행", "절뚝", "절뚝거림", "limping", "lameness",
            "stifle", "knee",
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
            "ortho_arthritis",
            "관절염", "골관절염", "퇴행성", "퇴행성관절", "관절통",
            "arthritis", "osteoarthritis", "OA",
            "DJD", "degenerative", "degenerative joint disease",
            "lameness", "joint pain",
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
            "prevent_vaccine_comprehensive",
            "종합백신", "혼합백신", "콤보백신", "콤보 백신",
            "DHPPL", "DHPP", "DHPPi", "DHPPI", "DHPPL", "DHLPP",
            "5종", "6종", "5in1", "6in1", "5-in-1", "6-in-1",
            "FVRCP", "fvrcp",
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
            "prevent_vaccine_corona",
            "코로나 백신", "코로나장염", "코로나 장염", "코로나 장염 백신",
            "corona vaccine", "coronavirus", "corona enteritis",
            "장염 예방",
        ],
        guide=[
            "접종 간격은 병원 안내에 맞춰 주세요.",
            "접종 후 붓기/통증이 심하거나 구토·무기력이 지속되면 병원에 문의해 주세요.",
        ],
    ),

    "prevent_vaccine_kennel": ConditionTagConfig(
        code="prevent_vaccine_kennel",
        label="예방접종 · 켄넬코프(기관지염)",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_kennel",
            "켄넬코프", "켄넬 코프", "기관지염 백신", "기관지염백신",
            "bordetella", "kennel cough",
            "보르데텔라",
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
            "prevent_vaccine_rabies",
            "광견병", "광견", "rabies",
        ],
        guide=[
            "지역/상황에 따라 접종 주기가 달라질 수 있어요. 병원 안내에 맞춰 기록해 주세요.",
        ],
    ),

    # ✅ 추가 백신: 렙토(Lepto)
    "prevent_vaccine_lepto": ConditionTagConfig(
        code="prevent_vaccine_lepto",
        label="예방접종 · 렙토(Lepto)",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_lepto",
            "렙토", "렙토스피라", "렙토2", "렙토4", "렙토 2종", "렙토 4종",
            "lepto", "leptospira", "leptospirosis",
            "lepto2", "lepto4", "L2", "L4",
        ],
        guide=[
            "접종 주기/추가 접종 여부는 병원 안내에 맞춰 기록해 주세요.",
        ],
    ),

    # ✅ 추가 백신: 파라인플루엔자(단독 표기 케이스)
    "prevent_vaccine_parainfluenza": ConditionTagConfig(
        code="prevent_vaccine_parainfluenza",
        label="예방접종 · 파라인플루엔자(개)",
        species="dog",
        group="preventive",
        keywords=[
            "prevent_vaccine_parainfluenza",
            "파라인플루엔자", "파라인", "파라", "파라 백신", "파라인 백신",
            "parainfluenza", "CPiV", "CPI", "PI",
        ],
        guide=[
            "호흡기 증상이 있거나 컨디션이 떨어지면 병원에 문의해 주세요.",
        ],
    ),

    # ✅ 추가 백신: FIP(고양이)
    "prevent_vaccine_fip": ConditionTagConfig(
        code="prevent_vaccine_fip",
        label="예방접종 · FIP(고양이)",
        species="cat",
        group="preventive",
        keywords=[
            "prevent_vaccine_fip",
            "FIP", "fip", "전염성복막염", "복막염",
            "feline infectious peritonitis",
            "primucell", "primucell fip",
            "FIP 백신",
        ],
        guide=[
            "접종 기록과 아이의 컨디션을 함께 관찰해 주세요.",
        ],
    ),

    "prevent_heartworm": ConditionTagConfig(
        code="prevent_heartworm",
        label="예방 · 심장사상충",
        species="both",
        group="preventive",
        keywords=[
            "prevent_heartworm",
            "사상충", "심장사상충", "heartworm", "dirofilaria",
            # 제품/표기
            "하트가드", "heartgard",
            "넥스가드스펙트라", "nexgard spectra",
            "심파리카트리오", "simparica trio",
            "리볼루션", "revolution",
            "인터셉터", "interceptor",
            "밀베마이신", "milbemycin",
            "이버멕틴", "ivermectin",
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
            "prevent_external",
            "외부기생충", "진드기", "벼룩",
            "tick", "flea",
            # 제품/표기
            "브라벡토", "bravecto",
            "넥스가드", "nexgard",
            "프론트라인", "frontline",
            "심파리카", "simparica",
            "크레델리오", "credelio",
            "레볼루션", "revolution",
        ],
        guide=[
            "산책이 잦은 계절엔 외부기생충 예방을 꾸준히 유지해 주세요.",
            "피부 가려움/발진이 심해지면 병원에 문의해 주세요.",
        ],
    ),

    "prevent_deworming": ConditionTagConfig(
        code="prevent_deworming",
        label="예방 · 구충(내부기생충)",
        species="both",
        group="preventive",
        keywords=[
            "prevent_deworming",
            "구충", "구충제", "내부기생충", "회충", "선충", "기생충",
            "deworm", "deworming", "dewormer", "internal parasite",
            # 제품/성분
            "드론탈", "drontal",
            "밀베맥스", "milbemax",
            "밀프로", "milpro",
            "파나쿠어", "panacur",
            "펜벤다졸", "fenbendazole",
            "피란텔", "pyrantel",
            "프라지콴텔", "praziquantel",
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
            "exam_xray",
            "엑스레이", "X-ray", "x-ray", "xray", "XR",
            "방사선", "x선", "X선",
            "radiograph", "radiography", "radiology",
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
            "exam_blood",
            "혈액검사", "혈검", "피검", "CBC", "cbC",
            "Chemistry", "chemistry", "biochem", "biochemistry",
            "blood test", "혈액", "생화학", "전해질", "electrolyte",
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
            "exam_ultrasound",
            "초음파", "복부초음파", "심장초음파",
            "ultrasound", "sono", "sonography", "US",
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
            "exam_lab_panel",
            "종합검사", "종합검진", "패널", "패널검사",
            "panel", "lab panel", "screening panel", "profile",
        ],
        guide=[
            "종합검사는 변화 추적이 중요해요. 이전 결과와 비교 메모를 남겨두면 좋아요.",
        ],
    ),

    # ✅ 추가: 소변검사
    "exam_urine": ConditionTagConfig(
        code="exam_urine",
        label="검사 · 소변검사(요검사)",
        species="both",
        group="exam",
        keywords=[
            "exam_urine",
            "소변검사", "요검사", "요 검사", "소변 검사",
            "urinalysis", "UA", "ua", "urine test",
            "요비중", "요침사", "요단백", "UPC",
        ],
        guide=[
            "소변검사는 요로/신장 상태 확인에 도움이 돼요. 결과 소견을 한 줄로 메모해 두면 좋아요.",
        ],
    ),

    # ✅ 추가: 대변검사
    "exam_fecal": ConditionTagConfig(
        code="exam_fecal",
        label="검사 · 대변검사",
        species="both",
        group="exam",
        keywords=[
            "exam_fecal",
            "대변검사", "분변검사", "배변검사",
            "fecal", "stool test", "stool",
            "기생충 검사", "giardia", "지아르디아",
        ],
        guide=[
            "대변검사는 장 상태/기생충 확인에 도움이 돼요. 증상(설사/혈변)도 같이 기록해 주세요.",
        ],
    ),

    # ✅ 추가: 알러지 검사
    "exam_allergy": ConditionTagConfig(
        code="exam_allergy",
        label="검사 · 알러지 검사",
        species="both",
        group="exam",
        keywords=[
            "exam_allergy",
            "알러지검사", "알레르기검사", "알러지 검사", "알레르기 검사",
            "IgE", "IGE", "ige",
            "allergy test", "atopy test",
        ],
        guide=[
            "알러지 검사는 결과 해석이 중요해요. 수의사 설명 요점을 함께 기록해 두면 좋아요.",
        ],
    ),

    # ✅ 추가: 심장 검사(검사 태그로 분리)
    "exam_heart": ConditionTagConfig(
        code="exam_heart",
        label="검사 · 심장검사(심초음파/심전도)",
        species="both",
        group="exam",
        keywords=[
            "exam_heart",
            "심장검사", "심장 검사", "심초음파", "심장초음파",
            "심전도", "ECG", "EKG", "ekg", "ecg",
            "echo", "echocardiography",
            "cardiac exam", "cardiac",
        ],
        guide=[
            "심장검사는 추적이 중요해요. 검사 수치/단계(등급) 메모를 남기면 좋아요.",
        ],
    ),

    # ✅ 추가: 안과 검사
    "exam_eye": ConditionTagConfig(
        code="exam_eye",
        label="검사 · 안과검사",
        species="both",
        group="exam",
        keywords=[
            "exam_eye",
            "안과검사", "안과 검사", "눈검사", "눈 검사",
            "안압", "형광염색", "쉬르머", "schirmer", "fluorescein", "IOP", "iop",
            "ophthalmic exam", "eye exam",
        ],
        guide=[
            "안과 검사는 증상 변화(눈곱/충혈/통증) 기록이 도움이 돼요.",
        ],
    ),

    # ✅ 추가: 피부 검사(스크래핑/진균/세포검사 등)
    "exam_skin": ConditionTagConfig(
        code="exam_skin",
        label="검사 · 피부검사",
        species="both",
        group="exam",
        keywords=[
            "exam_skin",
            "피부검사", "피부 검사", "피부스크래핑", "피부 스크래핑",
            "skin scraping", "scraping",
            "진균검사", "fungal test", "곰팡이검사",
            "세포검사", "cytology",
            "말라세지아", "malassezia",
        ],
        guide=[
            "피부 검사는 사진 기록이 특히 도움이 돼요. 악화/호전 시점을 함께 남겨보세요.",
        ],
    ),

    # ✅ 추가 검사: SDMA
    "exam_sdma": ConditionTagConfig(
        code="exam_sdma",
        label="검사 · SDMA(신장마커)",
        species="both",
        group="exam",
        keywords=[
            "exam_sdma",
            "SDMA", "sdma", "신장마커", "신장 표지자", "신장검사", "신장수치",
            "symmetrical dimethylarginine",
            "idexx sdma", "renal sdma",
        ],
        guide=[
            "SDMA는 신장 기능을 조기에 확인하는 데 도움 될 수 있어요. 추적 수치를 메모해 두면 좋아요.",
        ],
    ),

    # ✅ 추가 검사: proBNP
    "exam_probnp": ConditionTagConfig(
        code="exam_probnp",
        label="검사 · proBNP(심장마커)",
        species="both",
        group="exam",
        keywords=[
            "exam_probnp",
            "proBNP", "probnp", "pro bnp", "pro-bnp",
            "NT-proBNP", "ntprobnp", "nt-probnp", "bnp",
            "cardiopet", "cardio pet",
            "심장마커", "심장 표지자", "BNP검사", "프로비엔피",
        ],
        guide=[
            "심장마커 검사는 다른 검사(초음파/심전도)와 함께 해석돼요. 소견을 같이 기록해 주세요.",
        ],
    ),

    # ✅ 추가 검사: 당화알부민/FRU
    "exam_fructosamine": ConditionTagConfig(
        code="exam_fructosamine",
        label="검사 · 당화알부민/FRU",
        species="both",
        group="exam",
        keywords=[
            "exam_fructosamine",
            "당화알부민", "당화", "당화검사",
            "fructosamine", "fru",
            "glycated albumin", "ga",
            "당뇨", "혈당",
        ],
        guide=[
            "혈당 관련 검사는 추세가 중요해요. 이전 결과 대비 변화가 있으면 같이 메모해 주세요.",
        ],
    ),

    # ✅ 추가 검사: 혈당곡선
    "exam_glucose_curve": ConditionTagConfig(
        code="exam_glucose_curve",
        label="검사 · 혈당곡선",
        species="both",
        group="exam",
        keywords=[
            "exam_glucose_curve",
            "혈당곡선", "혈당커브", "혈당 커브",
            "glucose curve", "blood glucose curve", "bg curve",
            "혈당측정", "혈당 체크", "연속혈당", "glucose monitoring",
        ],
        guide=[
            "혈당곡선은 시간대별 변화가 핵심이에요. 측정 시간/식사/투약 정보를 같이 기록하면 좋아요.",
        ],
    ),

    # ✅ 추가 검사: 혈액가스
    "exam_blood_gas": ConditionTagConfig(
        code="exam_blood_gas",
        label="검사 · 혈액가스(BGA)",
        species="both",
        group="exam",
        keywords=[
            "exam_blood_gas",
            "혈액가스", "혈가", "가스분석",
            "blood gas", "BGA", "bga", "bgas",
            "i-stat", "istat", "i stat",
        ],
        guide=[
            "혈액가스는 응급/호흡/산염기 상태 평가에 쓰일 수 있어요. 소견을 함께 메모해 주세요.",
        ],
    ),

    # ✅ 추가 검사: 대변 PCR
    "exam_fecal_pcr": ConditionTagConfig(
        code="exam_fecal_pcr",
        label="검사 · 대변 PCR(GI PCR)",
        species="both",
        group="exam",
        keywords=[
            "exam_fecal_pcr",
            "대변 PCR", "대변pcr", "분변 PCR", "분변pcr", "배설물 PCR", "배설물pcr",
            "fecal pcr", "stool pcr", "gi pcr", "panel pcr",
            "GI panel", "지아이 패널",
            "장염 pcr", "설사 pcr",
        ],
        guide=[
            "PCR은 원인 추정에 도움 될 수 있어요. 검사 전후 증상 변화도 같이 남겨보세요.",
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
        keywords=[
            "medicine_antibiotic", "항생제", "antibiotic", "abx",
            "amoxicillin", "amoxi", "amox",
            "amoxiclav", "clavulanate", "augmentin", "clavamox",
            "cephalexin", "cefovecin", "convenia",
            "doxycycline", "doxy",
            "clindamycin", "metronidazole", "metro",
            "enrofloxacin", "baytril",
        ],
        guide=["처방 받은 기간을 지켜 복용해 주세요. 이상 반응이 있으면 병원에 문의해 주세요."],
    ),

    "medicine_anti_inflammatory": ConditionTagConfig(
        code="medicine_anti_inflammatory",
        label="처방 · 소염/항염(NSAID)",
        species="both",
        group="medication",
        keywords=[
            "medicine_anti_inflammatory",
            "소염", "항염", "소염제", "항염제",
            "NSAIDs", "NSAID", "nsaid",
            "anti-inflammatory", "anti inflammatory",
            "meloxicam", "metacam",
            "carprofen", "rimadyl",
            "robenacoxib", "onsior",
            "grapiprant", "galliprant",
        ],
        guide=["식욕 저하/구토 등 이상 증상이 있으면 복용을 멈추고 병원에 문의해 주세요."],
    ),

    "medicine_allergy": ConditionTagConfig(
        code="medicine_allergy",
        label="처방 · 알러지 약",
        species="both",
        group="medication",
        keywords=[
            "medicine_allergy",
            "알러지", "알레르기", "항히스타민",
            "antihistamine",
            "apoquel", "cytopoint",
            "cetirizine", "zyrtec",
            "loratadine", "claritin",
            "diphenhydramine", "benadryl",
        ],
        guide=["증상이 심해지거나 졸림이 심하면 병원에 문의해 주세요."],
    ),

    "medicine_gi": ConditionTagConfig(
        code="medicine_gi",
        label="처방 · 위장/장",
        species="both",
        group="medication",
        keywords=[
            "medicine_gi",
            "위장", "장", "설사", "구토", "장염",
            "probiotic", "유산균",
            "famotidine", "pepcid",
            "omeprazole", "pantoprazole",
            "sucralfate",
            "metoclopramide", "reglan",
            "maropitant", "cerenia",
            "ondansetron",
        ],
        guide=["물/식사량 변화를 함께 관찰해 주세요. 증상이 지속되면 병원에 문의해 주세요."],
    ),

    "medicine_ear": ConditionTagConfig(
        code="medicine_ear",
        label="처방 · 귀",
        species="both",
        group="medication",
        keywords=[
            "medicine_ear",
            "귀약", "점이", "이염",
            "ear drops", "ear medication", "otic", "otitis",
            "otomax", "surolan", "posatex", "easotic", "claro",
        ],
        guide=["귀 세정/점이 지침이 있다면 횟수를 지켜 사용해 주세요."],
    ),

    "medicine_skin": ConditionTagConfig(
        code="medicine_skin",
        label="처방 · 피부",
        species="both",
        group="medication",
        keywords=[
            "medicine_skin",
            "피부약", "연고", "약욕", "샴푸처방",
            "ointment", "topical", "derm",
            "chlorhexidine", "ketoconazole", "miconazole",
        ],
        guide=["피부가 악화되는 부위가 있으면 사진으로 기록해 두면 진료에 도움이 돼요."],
    ),

    "medicine_eye": ConditionTagConfig(
        code="medicine_eye",
        label="처방 · 안약",
        species="both",
        group="medication",
        keywords=[
            "medicine_eye",
            "안약", "점안", "eye drops", "ophthalmic", "eyedrop",
            "tobramycin", "ofloxacin", "ciprofloxacin",
            "chloramphenicol", "erythromycin",
            "atropine", "cyclosporine", "tacrolimus",
        ],
        guide=["점안 횟수/간격을 지켜 주세요. 눈을 심하게 비비면 병원에 문의해 주세요."],
    ),

    "medicine_painkiller": ConditionTagConfig(
        code="medicine_painkiller",
        label="처방 · 진통",
        species="both",
        group="medication",
        keywords=[
            "medicine_painkiller",
            "진통", "진통제", "painkiller", "analgesic",
            "tramadol", "gabapentin",
            "buprenorphine", "codeine",
        ],
        guide=["통증 징후(절뚝/숨김/예민함)가 줄어드는지 함께 관찰해 주세요."],
    ),

    # ✅ 추가: 스테로이드
    "medicine_steroid": ConditionTagConfig(
        code="medicine_steroid",
        label="처방 · 스테로이드",
        species="both",
        group="medication",
        keywords=[
            "medicine_steroid",
            "스테로이드", "steroid",
            "pred", "prednisone", "prednisolone",
            "methylpred", "methylprednisolone",
            "dexamethasone", "dex",
            "triamcinolone",
        ],
        guide=[
            "스테로이드는 용량/기간 관리가 중요해요. 갑자기 중단하지 말고 병원 지침에 따라 주세요.",
        ],
    ),

    # ✅ 추가: 처치/진료/소모품 (질환 추론 금지 권장: group='procedure')
    "care_injection": ConditionTagConfig(
        code="care_injection",
        label="처치 · 주사/주사제",
        species="both",
        group="procedure",
        keywords=[
            "care_injection",
            "주사", "주사제", "주사료", "주사 처치",
            "injection", "inj", "shot",
            "SC", "S/C", "IM", "I/M", "IV", "I/V",
            "피하주사", "근육주사", "정맥주사",
        ],
        guide=[
            "주사 후 부종/통증/무기력 등이 지속되면 병원에 문의해 주세요.",
        ],
    ),

    "care_procedure_fee": ConditionTagConfig(
        code="care_procedure_fee",
        label="처치 · 처치료/시술료",
        species="both",
        group="procedure",
        keywords=[
            "care_procedure_fee",
            "처치료", "처치비", "시술료", "시술비", "처치", "시술",
            "procedure fee", "treatment fee", "handling fee",
            "procedure", "treatment",
        ],
        guide=[
            "처치/시술 항목은 병원마다 표기가 달라요. 가능하면 상세 내용을 메모해 두면 좋아요.",
        ],
    ),

    "care_dressing": ConditionTagConfig(
        code="care_dressing",
        label="처치 · 드레싱/붕대/소독",
        species="both",
        group="procedure",
        keywords=[
            "care_dressing",
            "드레싱", "붕대", "거즈", "랩", "소독", "세척", "상처처치",
            "dressing", "bandage", "bandaging", "wrap", "gauze",
            "cleaning", "disinfection",
        ],
        guide=[
            "상처 부위를 핥지 않도록 주의해 주세요. 붓기/열감/악취가 나면 병원에 문의해 주세요.",
        ],
    ),

    "care_e_collar": ConditionTagConfig(
        code="care_e_collar",
        label="소모품 · 넥카라",
        species="both",
        group="procedure",
        keywords=[
            "care_e_collar",
            "넥카라", "엘리자베스카라", "엘리자베스", "보호카라", "보호대",
            "e-collar", "ecollar", "cone",
            "elizabethan collar", "elizabeth collar",
        ],
        guide=[
            "상처 보호 목적이라면 착용 시간을 지켜 주세요. 피부 쓸림이 있으면 병원에 문의해 주세요.",
        ],
    ),

    "care_prescription_diet": ConditionTagConfig(
        code="care_prescription_diet",
        label="소모품 · 처방식(사료)",
        species="both",
        group="procedure",
        keywords=[
            "care_prescription_diet",
            "처방식", "처방사료", "처방캔", "병원사료", "병원 사료",
            "prescription diet", "rx diet", "therapeutic diet",
            "hill's", "hills", "로얄캐닌", "royal canin",
            "k/d", "c/d", "s/d", "i/d", "z/d", "h/d",
            "renal diet", "urinary diet", "gastrointestinal diet", "hypoallergenic diet",
            "신장 처방식", "요로 처방식", "위장 처방식", "알러지 처방식",
        ],
        guide=[
            "처방식은 목적에 맞게 급여하는 기간/방법이 중요해요. 수의사 지침을 따라 주세요.",
        ],
    ),

    "dental_scaling": ConditionTagConfig(
        code="dental_scaling",
        label="치과 · 스케일링",
        species="both",
        group="procedure",
        keywords=["dental_scaling", "스케일링", "치석 제거", "scaling", "dental cleaning", "tartar"],
        guide=["시술 후 며칠간은 식욕/통증 반응을 관찰해 주세요. 양치 루틴을 천천히 시작해 보세요."],
    ),

    "dental_extraction": ConditionTagConfig(
        code="dental_extraction",
        label="치과 · 발치",
        species="both",
        group="procedure",
        keywords=["dental_extraction", "발치", "extraction", "dental extraction"],
        guide=["처방 약을 잘 챙겨 주세요. 출혈/통증이 심하면 병원에 문의해 주세요."],
    ),

    "surgery_general": ConditionTagConfig(
        code="surgery_general",
        label="수술 · 일반",
        species="both",
        group="procedure",
        keywords=["surgery_general", "수술", "operation", "surgery", "봉합", "마취", "중성화", "spay", "neuter", "castration"],
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
            "check-up", "checkup", "screening",
        ],
        guide=[
            "성견/성묘는 1년에 한 번, 노령 아이는 6개월마다 건강검진을 권장해요.",
        ],
    ),

    # ✅ (옵션) iOS에서 태그가 올 수 있는 것들: 빠지지 않게 최소 정의
    "checkup_general": ConditionTagConfig(
        code="checkup_general",
        label="기본진료 · 상담/진찰",
        species="both",
        group="wellness",
        keywords=[
            "checkup_general",
            "기본진료", "기본 진료", "진료", "진찰", "상담", "초진", "재진",
            "consult", "checkup", "opd",
        ],
        guide=[
            "기본진료 기록은 증상(언제부터/빈도/악화요인)을 함께 남기면 다음 진료에 도움이 돼요.",
        ],
    ),

    "grooming_basic": ConditionTagConfig(
        code="grooming_basic",
        label="미용 · 목욕/관리",
        species="both",
        group="wellness",
        keywords=[
            "grooming_basic",
            "미용", "목욕", "클리핑", "가위컷", "발톱", "귀청소",
            "grooming", "bath", "trim", "clipping",
        ],
        guide=[
            "피부가 민감한 아이는 미용 후 가려움/홍반이 생기지 않는지 관찰해 주세요.",
        ],
    ),

    "etc_other": ConditionTagConfig(
        code="etc_other",
        label="기타",
        species="both",
        group="wellness",
        keywords=["etc_other", "기타", "etc", "other"],
        guide=[],
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

    # preventive vaccines (canonical: prevent_vaccine_*)
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

    # ✅ 추가 백신
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
    "examBlood": "exam_blood",
    "examUltrasound": "exam_ultrasound",
    "examLabPanel": "exam_lab_panel",

    # ✅ 추가 exam
    "examUrine": "exam_urine",
    "examFecal": "exam_fecal",
    "examAllergy": "exam_allergy",
    "examHeart": "exam_heart",
    "examEye": "exam_eye",
    "examSkin": "exam_skin",
    "examSDMA": "exam_sdma",
    "examProBNP": "exam_probnp",
    "examFructosamine": "exam_fructosamine",
    "examGlucoseCurve": "exam_glucose_curve",
    "examBloodGas": "exam_blood_gas",
    "examFecalPCR": "exam_fecal_pcr",

    # medication / procedure
    "medicineAntibiotic": "medicine_antibiotic",
    "medicineAntiInflammatory": "medicine_anti_inflammatory",
    "medicineAllergy": "medicine_allergy",
    "medicineGI": "medicine_gi",
    "medicineEar": "medicine_ear",
    "medicineSkin": "medicine_skin",
    "medicineEye": "medicine_eye",
    "medicinePainkiller": "medicine_painkiller",
    "medicineSteroid": "medicine_steroid",

    # ✅ care/procedure
    "careInjection": "care_injection",
    "careProcedureFee": "care_procedure_fee",
    "careDressing": "care_dressing",
    "careECollar": "care_e_collar",
    "carePrescriptionDiet": "care_prescription_diet",

    "dentalScaling": "dental_scaling",
    "dentalExtraction": "dental_extraction",
    "surgeryGeneral": "surgery_general",

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


