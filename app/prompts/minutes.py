"""회의록 생성 프롬프트 — 세션 로그 → 결정/논의/액션, 직무 관점별 + 언어 렌더링.

직무 category당 기준 언어로 1회 생성하고, 나머지 언어는 그 결과를 번역해서 채운다.
언어별 회의록은 "다른 요약"이 아니라 "같은 회의록의 번역본"이어야 하기 때문.
"""

import json
from typing import List, Optional

from app.schemas.minutes import MinutesUtterance

MINUTES_SYSTEM_PROMPT = (
    "You summarize a multilingual meeting transcript into a structured minutes section. "
    "Separate the content into decisions, discussions, and action_items. "
    "Write the OUTPUT in the requested language, and emphasize what matters to the requested job perspective.\n"
    "- decisions / discussions: array of strings.\n"
    '- action_items: array of objects {"task": str, "owner": str|null, "deadline": str|null}. '
    "Use the exact keys task/owner/deadline (English keys, values in the requested language). "
    "owner/deadline null if not stated.\n"
    "Output ONLY a single JSON object: "
    '{"decisions": [], "discussions": [], "action_items": []}. No prose, no fences.'
)

_JOB_LABEL = {
    "dev": "개발/엔지니어링",
    "design": "디자인",
    "pm": "기획/PM",
    "sales": "영업/마케팅",
    "research": "연구",
    "etc": "일반",
}


def build_minutes_prompt(
    utterances: List[MinutesUtterance], language: str, job_category: Optional[str]
) -> str:
    log = "\n".join(
        f"  - [{u.lang}] {u.speaker or '?'}: {u.text}" for u in utterances
    ) or "  (no utterances)"
    perspective = _JOB_LABEL.get(job_category or "etc", "일반")
    return (
        f"# Transcript\n{log}\n\n"
        f"# Output language\n  {language}\n"
        f"# Job perspective (요약 관점)\n  {perspective}\n\n"
        f"# Task\n"
        f"Summarize into decisions / discussions / action_items in [{language}], "
        f"prioritizing what the [{perspective}] role needs. "
        f'Return JSON: {{"decisions": [], "discussions": [], '
        f'"action_items": [{{"task": "...", "owner": "...", "deadline": "..."}}]}}'
    )


MINUTES_TRANSLATE_SYSTEM_PROMPT = (
    "You translate already-written meeting minutes into another language. "
    "The input is a JSON array of strings; return a JSON array of the SAME length, "
    "same order, each element translated. "
    "Do NOT merge, split, add, drop, summarize, or reorder elements. "
    "Keep person names, product names, and numbers as they are.\n"
    "Output ONLY the JSON array. No prose, no fences."
)


def build_minutes_translate_prompt(texts: List[str], language: str) -> str:
    """기준 언어 섹션의 문자열들을 language로 번역 — 개수·순서가 곧 구조라 그대로 유지시킨다."""
    # ponytail: 평탄화한 문자열 배열로만 번역. 항목별 맥락(결정/논의/액션 구분)은 안 주는데,
    #           번역 품질이 아쉬우면 라벨 붙인 객체 배열로 올리면 됨.
    return (
        f"# Translate\n"
        f"# Target language\n  {language}\n\n"
        f"# Items ({len(texts)} strings, JSON array)\n"
        f"{json.dumps(texts, ensure_ascii=False)}\n\n"
        f"# Task\n"
        f"Translate every element into [{language}]. "
        f"Return ONLY a JSON array of exactly {len(texts)} strings in the same order."
    )
