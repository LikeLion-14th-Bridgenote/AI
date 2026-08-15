"""Erin Meyer 'Culture Map' 8축 기반 문화 각주 프롬프트.

근거 이론: Edward Hall 고맥락/저맥락, Erin Meyer Culture Map.
※ AI는 단정하지 않는다 — "오해 가능성 + 대체 표현 제안"까지만, 확정은 사용자.

게이트 통과분에 대해 화자↔청자 문화·직무 + RAG 규칙(코퍼스)을 근거로
cultural_note(각주) + 참가자 언어별 번역을 한 번의 LLM 호출로 뽑는다.
"""

import json
from typing import List

from app.schemas.analyze import AnalyzeRequest

CULTURE_MAP_AXES = [
    "Communicating (low-context ↔ high-context)",
    "Evaluating (direct ↔ indirect negative feedback)",
    "Persuading (principles-first ↔ applications-first)",
    "Leading (egalitarian ↔ hierarchical)",
    "Deciding (consensual ↔ top-down)",
    "Trusting (task-based ↔ relationship-based)",
    "Disagreeing (confrontational ↔ avoids confrontation)",
    "Scheduling (linear-time ↔ flexible-time)",
]

ANALYZE_SYSTEM_PROMPT = (
    "You are a cross-cultural communication analyst for a multilingual meeting tool. "
    "Using Erin Meyer's Culture Map framework and Edward Hall's high/low-context theory, "
    "you judge whether an utterance is likely to be MISREAD across cultures — not whether it is good or bad.\n"
    "Culture Map axes:\n- " + "\n- ".join(CULTURE_MAP_AXES) + "\n\n"
    "Rules:\n"
    "- Ground your judgment in the provided culture rules (RAG). Do not rely only on stereotypes.\n"
    "- Never assert intent as fact. Frame as 'likely / may be' and offer a clearer rewrite; the user decides.\n"
    "- Especially for hedged expressions (KR '검토해보겠습니다', JP '検討します'): do NOT call them a refusal. "
    "They mean the speaker has NOT committed — the outcome is still open. "
    "The risk is that the listener hears them as a commitment.\n"
    "- If the utterance is actually low-risk once you see the context, return has_risk=false.\n"
    "- Write speaker_intent / listener_misread / advice in the SPEAKER's language.\n"
    "- has_risk MUST be a JSON boolean (true/false), NOT the string \"true\"/\"false\".\n"
    "- risk_level MUST be exactly one of: High, Med, Low.\n"
    "- note_type MUST be exactly one of these three Korean labels: 문화 이해, 커뮤니케이션, 업무 스타일.\n"
    "  · 문화 이해 — 완곡/직설, 맥락 의존 등 표현 방식의 문화 차이\n"
    "  · 커뮤니케이션 — 피드백·이견 제시·동의 표현 방식\n"
    "  · 업무 스타일 — 일정·담당·의사결정 등 업무 진행 방식\n"
    "- Output ONLY a single JSON object, no prose, no markdown fences."
)

# 프론트(회의록 문화 가이드 탭)가 이 세 값으로 칩·통계를 그린다. 다른 값이 오면 집계에서 빠진다.
NOTE_TYPES = ("문화 이해", "커뮤니케이션", "업무 스타일")

# LLM이 반드시 이 형태로만 반환하도록 지시하는 출력 스키마
_OUTPUT_SCHEMA = {
    "has_risk": "true|false",
    "risk_level": "High|Med|Low",
    "note_type": "문화 이해|커뮤니케이션|업무 스타일 중 하나 (이 셋 외 금지)",
    "speaker_intent": "화자의 실제 의도 (추정, 단정 금지)",
    "listener_misread": "청자 문화에서 오독할 수 있는 지점",
    "advice": "화자에게 줄 한 줄 조언",
    "rewrite_text": "청자 문화에 맞게 다시 쓴 문장 (청자 언어로)",
    "translations": [{"lang": "청자 언어코드", "text": "그 언어로 번역"}],
}


def _culture_line(scores: dict, code: str) -> str:
    c = scores.get(code)
    if not c:
        return f"{code}: (no reference data)"
    cm = c.get("culturemap", {})
    axes = ", ".join(f"{k}={v}" for k, v in cm.items())
    return f"{code} ({c.get('name', code)}): {axes}"


def build_user_prompt(
    req: AnalyzeRequest, rules: List[dict], country_scores: dict
) -> str:
    """게이트 통과 발화 → 각주 생성용 유저 프롬프트."""
    listeners = "\n".join(
        f"  - participant_id={l.participant_id}, lang={l.lang}, "
        f"culture={l.culture}, job={l.job}"
        for l in req.listeners
    )
    rule_lines = "\n".join(f"  - [{r.get('culture')}] {r['rule_text']}" for r in rules) or "  (none)"
    cultures = {req.speaker.culture} | {l.culture for l in req.listeners}
    cmap = "\n".join(f"  - {_culture_line(country_scores, c)}" for c in sorted(cultures))
    context = "\n".join(f"  - ({c.lang}) {c.text}" for c in req.context) or "  (none)"

    return (
        f"# Utterance\n"
        f'  text: "{req.source_text}"\n'
        f"  source_lang: {req.source_lang}\n"
        f"  speaker: culture={req.speaker.culture}, job={req.speaker.job}\n"
        f"  meeting_context: {req.meeting_context or '(none)'}\n\n"
        f"# Listeners\n{listeners}\n\n"
        f"# Prior turns (판단용 직전 맥락)\n{context}\n\n"
        f"# Culture Map reference (참가 문화권)\n{cmap}\n\n"
        f"# Culture rules retrieved for listener cultures (RAG 근거)\n{rule_lines}\n\n"
        f"# Task\n"
        f"Judge misread risk of the utterance for each listener culture, then produce ONE note "
        f"(worst-case listener) and translations for every listener language.\n"
        f"Return JSON exactly in this shape:\n{json.dumps(_OUTPUT_SCHEMA, ensure_ascii=False)}"
    )
