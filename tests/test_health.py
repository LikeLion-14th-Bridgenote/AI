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
