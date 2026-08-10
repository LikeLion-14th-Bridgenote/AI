"""LLM provider 스위치 — openai | gemini | claude.

- openai/gemini: OpenAI 호환 SDK 공용. base_url만 분기(gemini는 구글 호환 엔드포인트).
- claude: anthropic SDK 별도(system이 top-level, max_tokens 필수 등 차이).

.env의 LLM_PROVIDER / LLM_API_KEY / LLM_MODEL로 결정.
문화 각주는 구조화 JSON을 뽑아야 하므로 complete()는 문자열(JSON 문자열)을 반환한다.
SDK import는 무거우므로 실제 호출 시점까지 지연한다(테스트가 패키지 없이 돈다).
"""

from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings

# provider별 기본 모델 + OpenAI 호환 base_url
_DEFAULTS = {
    "openai": {"model": "gpt-4o-mini", "base_url": None},
    "gemini": {
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "claude": {"model": "claude-sonnet-4-6", "base_url": None},
}


@dataclass
class LLMClient:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str] = None

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        if self.provider in ("openai", "gemini"):
            return await self._openai_complete(prompt, system, max_tokens)
        if self.provider == "claude":
            return await self._claude_complete(prompt, system, max_tokens)
        raise ValueError(f"알 수 없는 LLM provider: {self.provider}")

    async def _openai_complete(self, prompt: str, system: str, max_tokens: int) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = await client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""

    async def _claude_complete(self, prompt: str, system: str, max_tokens: int) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.api_key)
        resp = await client.messages.create(  # anthropic: system은 top-level, max_tokens 필수
            model=self.model,
            system=system or None,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


def get_llm_client() -> LLMClient:
    s = get_settings()
    d = _DEFAULTS.get(s.llm_provider, _DEFAULTS["openai"])
    return LLMClient(
        provider=s.llm_provider,
        api_key=s.llm_api_key,
        model=s.llm_model or d["model"],
        base_url=d["base_url"],
    )
