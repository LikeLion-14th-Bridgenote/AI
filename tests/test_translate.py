"""번역 서비스 테스트 (LLM 목킹, 네트워크 없이)."""

import asyncio
import json

from app.schemas.translate import TranslateRequest
from app.services.translate import service


def _req():
    return TranslateRequest(
        sentence_id="u1", source_text="금요일까지 가능할까요?", source_lang="ko",
        targets=[
            {"participant_id": "p1", "lang": "en"},
            {"participant_id": "p2", "lang": "vi"},
        ],
    )


def _patch(monkeypatch, out=None, raises=False):
    async def fake(prompt, max_tokens=1024):
        if raises:
            raise RuntimeError("down")
        return out
    monkeypatch.setattr(service, "get_llm_client",
                        lambda: type("C", (), {"complete": staticmethod(fake)})())


def test_translations_mapped_to_participants(monkeypatch):
    _patch(monkeypatch, json.dumps({"translations": [
        {"lang": "en", "text": "Is it possible by Friday?"},
        {"lang": "vi", "text": "Có thể xong trước thứ Sáu không?"},
    ]}))
    r = asyncio.run(service.translate(_req()))
    en = next(t for t in r.translations if t.participant_id == "p1")
    assert en.text == "Is it possible by Friday?"
    assert len(r.translations) == 2


def test_failure_is_fail_open(monkeypatch):
    _patch(monkeypatch, raises=True)
    r = asyncio.run(service.translate(_req()))
    assert len(r.translations) == 2 and all(t.text == "" for t in r.translations)  # 500 아님
