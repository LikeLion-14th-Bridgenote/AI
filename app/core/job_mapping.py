"""직무 매핑 — 프론트 세부 직무(17개) → 백엔드 category(6개).

회의록 '관점별 요약' 탭이 폭발하지 않게 category로 묶는다.
세부 직무 원본은 버리지 않고 각주/회의록 프롬프트에 힌트로 따로 넘긴다.
(수민/진수님과 합의: 프론트는 세밀하게, AI는 큰 범주로)
"""

# 세부 직무(한글 라벨 또는 코드) → category
_JOB_MAP = {
    # dev
    "개발 IT": "dev", "개발IT": "dev", "데이터 분석": "dev", "dev": "dev",
    # design
    "디자인": "design", "design": "design",
    # pm
    "기획 PM": "pm", "기획PM": "pm", "경영 관리": "pm", "경영관리": "pm", "pm": "pm",
    # sales
    "영업 사업개발": "sales", "영업": "sales", "마케팅 광고": "sales", "마케팅": "sales", "sales": "sales",
    # research
    "연구": "research", "research": "research",
    # etc (인사·재무·법무·생산·의료·교육·공공·학생·기타)
}

CATEGORIES = ["dev", "design", "pm", "sales", "research", "etc"]


def to_category(job: str) -> str:
    """세부 직무 → 6개 category. 미매핑은 etc."""
    if not job:
        return "etc"
    return _JOB_MAP.get(job.strip(), "etc")


def categories_of(jobs) -> list:
    """직무 리스트 → 중복 제거된 category 리스트 (CATEGORIES 순서 유지)."""
    got = {to_category(j) for j in jobs}
    return [c for c in CATEGORIES if c in got]
