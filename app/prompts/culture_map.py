"""
Erin Meyer 'Culture Map' 8축 기반 문화 각주 프롬프트 템플릿.
근거 이론: Edward Hall 고맥락/저맥락, Erin Meyer Culture Map.
※ AI는 단정하지 않는다 — "오해 가능성 + 대체 표현 제안"까지만, 확정은 사용자.
"""

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

# TODO: 실제 시스템/유저 프롬프트 작성 (provider 확정 후 튜닝).
ANALYZE_SYSTEM_PROMPT = ""
