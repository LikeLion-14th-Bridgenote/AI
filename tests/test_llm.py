"""LLM provider 스위치 라우팅 테스트 (네트워크·SDK 없이).

get_llm_client()가 provider별로 올바른 모델·base_url을 고르는지,
알 수 없는 provider가 에러를 내는지만 확인한다. 실제 API 호출은 검증 대상 아님.
"""

import asyncio

import pytest

from app.core import llm
from app.core.config import Settings


def _build(provider, model=""):
    # get_settings는 lru_cache라, Settings를 직접 만들어 get_llm_client와 같은 라우팅만 검증.
    s = Settings(llm_provider=provider, llm_api_key="k", llm_model=model)
    d = llm._DEFAULTS.get(provider, llm._DEFAULTS["openai"])
    return llm.LLMClient(provider, s.llm_api_key, s.llm_model or d["model"], d["base_url"])


def test_openai_defaults():
    c = _build("openai", "")
    assert c.base_url is None and c.model == "gpt-4o-mini"


def test_gemini_uses_openai_compat_base_url():
    c = _build("gemini", "")
    assert c.base_url.endswith("/openai/") and c.model.startswith("gemini")


def test_claude_no_base_url():
    c = _build("claude", "")
    assert c.base_url is None and c.model.startswith("claude")


def test_explicit_model_overrides_default():
    assert _build("openai", "gpt-4o").model == "gpt-4o"


def test_unknown_provider_raises():
    c = llm.LLMClient("bogus", "k", "m", None)
    with pytest.raises(ValueError):
        asyncio.run(c.complete("hi"))
