"""회의록 생성 프롬프트 — 세션 로그 → 결정/논의/액션, 언어·직무 관점별.

한 (언어 × 직무 category) 조합당 1회 호출. 언어=출력 언어, 직무=요약 관점.
"""

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
