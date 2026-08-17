from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_stub_no_risk(monkeypatch):
    # 게이트를 갈아끼워 임베딩·DB 없이 엔드포인트 계약(200/has_risk/translations)만 검증.
    async def no_risk(_req):
        return False

    monkeypatch.setattr("app.services.analyze.service.passes_gate", no_risk)
    payload = {
        "sentence_id": "u1",
        "meeting_id": "m1",
        "source_text": "네, 검토해보겠습니다",
        "source_lang": "ko",
        "speaker": {"culture": "KR", "job": "pm"},
        "listeners": [
            {"participant_id": "p1", "lang": "en", "culture": "US", "job": "dev"}
        ],
    }
    r = client.post("/ai/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["sentence_id"] == "u1"
    assert body["has_risk"] is False
    assert len(body["translations"]) == 1


def test_analyze_missing_field_returns_400():
    r = client.post("/ai/analyze", json={"sentence_id": "u1"})
    assert r.status_code == 400
    assert "message" in r.json()


def test_translate_stub():
    payload = {
        "sentence_id": "u2",
        "source_text": "금요일까지 가능할까요?",
        "source_lang": "ko",
        "targets": [{"participant_id": "p1", "lang": "en"}],
    }
    r = client.post("/ai/translate", json=payload)
    assert r.status_code == 200
    assert r.json()["sentence_id"] == "u2"


def _patch_deep(monkeypatch, *, corpus_rows=None, corpus_raises=False, embed_dim=384):
    """deep 헬스체크의 외부 의존(임베딩·DB)을 갈아끼운다.

    핸들러가 함수 안에서 import하므로 원본 모듈 속성을 갈아끼워야 한다.
    """
    async def fake_embed(text):
        return [0.0] * embed_dim

    monkeypatch.setattr("app.core.embeddings.embed_one", fake_embed)

    async def fake_search(vec, culture, top_k=5, entry_type=None):
        if corpus_raises:
            raise RuntimeError("relation does not exist")
        return corpus_rows if corpus_rows is not None else [{"rule_text": "r"}]

    monkeypatch.setattr("app.rag.corpus.search_by_vector", fake_search)


def test_health_deep_ok(monkeypatch):
    from app.core import health as health_mod
    from app.core.config import get_settings

    health_mod.reset()
    s = get_settings()
    monkeypatch.setattr(s, "llm_api_key", "sk-test", raising=False)
    monkeypatch.setattr(s, "supabase_db_url", "postgresql://x", raising=False)
    _patch_deep(monkeypatch)

    body = client.get("/health/deep").json()
    assert body["status"] == "ok"
    assert all(c["ok"] for c in body["checks"].values())
    assert body["fail_open"] == {}


def test_health_deep_reports_dead_corpus(monkeypatch):
    # 각주가 조용히 꺼진 상태를 여기서는 잡아내야 한다 (/health는 못 잡는다).
    from app.core import health as health_mod
    from app.core.config import get_settings

    health_mod.reset()
    s = get_settings()
    monkeypatch.setattr(s, "llm_api_key", "sk-test", raising=False)
    monkeypatch.setattr(s, "supabase_db_url", "postgresql://x", raising=False)
    _patch_deep(monkeypatch, corpus_raises=True)

    body = client.get("/health/deep").json()
    assert body["status"] == "degraded"
    assert body["checks"]["corpus"]["ok"] is False
    assert "RuntimeError" in body["checks"]["corpus"]["detail"]
    assert client.get("/health").json()["status"] == "ok"  # 얕은 쪽은 여전히 ok


def test_health_deep_shows_fail_open_counts(monkeypatch):
    from app.core import health as health_mod
    from app.core.config import get_settings

    health_mod.reset()
    health_mod.record_fail_open("gate")
    health_mod.record_fail_open("gate")
    health_mod.record_fail_open("note")

    s = get_settings()
    monkeypatch.setattr(s, "llm_api_key", "sk-test", raising=False)
    monkeypatch.setattr(s, "supabase_db_url", "postgresql://x", raising=False)
    _patch_deep(monkeypatch)

    body = client.get("/health/deep").json()
    assert body["fail_open"] == {"gate": 2, "note": 1}
    health_mod.reset()
