"""각주 오케스트레이션 테스트 (게이트·RAG·LLM 목킹, 네트워크·DB 없이).

파이프라인 배선만 검증: 게이트 통과 → 프롬프트 → LLM JSON → AnalyzeResponse 매핑,
그리고 LLM/파싱 실패 시 fail-open(각주 생략).
"""

import asyncio
import json

from app.schemas.analyze import AnalyzeRequest
from app.services.analyze import service


def _req():
    return AnalyzeRequest(
        sentence_id="u1", meeting_id="m1",
        source_text="네, 검토해보겠습니다", source_lang="ko",
        speaker={"culture": "KR", "job": "pm"},
        listeners=[
            {"participant_id": "p1", "lang": "en", "culture": "US", "job": "dev"},
            {"participant_id": "p2", "lang": "vi", "culture": "VN", "job": "design"},
        ],
    )


def _patch(monkeypatch, *, gate=True, llm_out="", raises=False):
    async def fake_gate(_req):
        return gate

    async def fake_rules(_req, top_k=5):
        return [{"culture": "US", "rule_text": "low-context; takes words literally"}]

    async def fake_complete(prompt, system=""):
        if raises:
            raise RuntimeError("llm down")
        return llm_out

    monkeypatch.setattr(service, "passes_gate", fake_gate)
    monkeypatch.setattr(service, "_gather_rules", fake_rules)
    monkeypatch.setattr(service, "get_llm_client",
                        lambda: type("C", (), {"complete": staticmethod(fake_complete)})())
    monkeypatch.setattr(service, "_scores", lambda: {})


def test_gate_miss_returns_no_risk(monkeypatch):
    _patch(monkeypatch, gate=False)
    r = asyncio.run(service.analyze(_req()))
    assert r.has_risk is False and len(r.translations) == 2


def test_note_generated_from_llm_json(monkeypatch):
    out = json.dumps({
        "has_risk": True, "risk_level": "High", "note_type": "직설성",
        "speaker_intent": "완곡한 유보", "listener_misread": "US는 승인으로 오독",
        "advice": "기한을 되물어 확인",
        "rewrite_text": "I'll need to review this—can we confirm the deadline?",
        "translations": [{"lang": "en", "text": "Yes, I'll review it."},
                         {"lang": "vi", "text": "Vâng, tôi sẽ xem xét."}],
    }, ensure_ascii=False)
    _patch(monkeypatch, llm_out=out)
    r = asyncio.run(service.analyze(_req()))
    assert r.has_risk is True and r.risk_level == "High"
    assert r.listener_misread.startswith("US")
    # 번역이 participant_id에 lang으로 매핑됐는지
    en = next(t for t in r.translations if t.participant_id == "p1")
    assert en.text == "Yes, I'll review it."


def test_llm_says_low_risk_drops_note(monkeypatch):
    _patch(monkeypatch, llm_out=json.dumps({"has_risk": False, "translations": []}))
    r = asyncio.run(service.analyze(_req()))
    assert r.has_risk is False


def test_string_false_is_coerced(monkeypatch):
    # LLM이 boolean 대신 문자열 "false"를 줘도 위험 없음으로 처리 (truthy 함정 방지)
    _patch(monkeypatch, llm_out=json.dumps({"has_risk": "false", "translations": []}))
    r = asyncio.run(service.analyze(_req()))
    assert r.has_risk is False


def test_llm_failure_is_fail_open(monkeypatch):
    _patch(monkeypatch, raises=True)
    r = asyncio.run(service.analyze(_req()))
    assert r.has_risk is False and len(r.translations) == 2  # 500 아님


def test_malformed_json_is_fail_open(monkeypatch):
    _patch(monkeypatch, llm_out="not json {{{")
    r = asyncio.run(service.analyze(_req()))
    assert r.has_risk is False
