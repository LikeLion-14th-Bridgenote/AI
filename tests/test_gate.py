"""게이트 로직 오프라인 테스트 (임베딩·DB 없이 monkeypatch).

문화 거리는 실제 country_scores.json으로 검증하고,
게이트 판정은 임베딩/코퍼스 검색을 가짜로 갈아끼워 순수 로직만 확인한다.
"""

import asyncio

from app.rag.culture_distance import cultural_distance
from app.schemas.analyze import AnalyzeRequest
from app.services.analyze import gate


def _req(speaker_culture="KR", listener_culture="US", text="네, 검토해보겠습니다"):
    return AnalyzeRequest(
        utterance_id="u1",
        meeting_id="m1",
        source_text=text,
        source_lang="ko",
        speaker={"culture": speaker_culture, "job_role": "pm"},
        listeners=[{"participant_id": "p1", "lang": "en", "culture": listener_culture, "job_role": "dev"}],
    )


def test_cultural_distance_ordering():
    assert cultural_distance("KR", "KR") == 0.0
    # 한↔미(먼 조합)가 한↔베(가까운 조합)보다 문화 거리가 커야 한다.
    assert cultural_distance("KR", "US") > cultural_distance("KR", "VN")
    # 미지 문화는 중립값.
    assert cultural_distance("KR", "ZZ") == 0.5


def _patch(monkeypatch, similarity):
    monkeypatch.setattr(gate, "embed_one", lambda text: [0.0])

    async def fake_search(vec, culture, top_k=5):
        return [{"rule_text": "r", "note_type": "직설성", "culture": culture, "similarity": similarity}]

    monkeypatch.setattr(gate, "search_by_vector", fake_search)


def test_gate_triggers_on_high_similarity(monkeypatch):
    _patch(monkeypatch, similarity=0.9)
    assert asyncio.run(gate.passes_gate(_req())) is True


def test_gate_skips_on_low_similarity(monkeypatch):
    _patch(monkeypatch, similarity=0.1)
    assert asyncio.run(gate.passes_gate(_req())) is False


def test_distant_culture_lowers_threshold(monkeypatch):
    # 설정값에 무관하게: eff(KR,US) < sim < base 인 유사도를 잡으면
    # 한↔미(먼 조합)는 문턱이 낮아져 통과, 한↔한(거리 0, 문턱=base)은 미통과여야 한다.
    s = gate.get_settings()
    eff_us = s.gate_base_threshold - s.gate_distance_sensitivity * cultural_distance("KR", "US")
    sim = (eff_us + s.gate_base_threshold) / 2  # eff_us < sim < base
    _patch(monkeypatch, similarity=sim)
    assert asyncio.run(gate.passes_gate(_req(listener_culture="US"))) is True
    assert asyncio.run(gate.passes_gate(_req(speaker_culture="KR", listener_culture="KR"))) is False
