from app.core.config import get_settings


class LLMClient:
    """
    LLM provider 스위치 (claude / openai — 미확정).
    실제 SDK 호출은 provider 확정 후 구현. 지금은 자리표시자.
    """

    def __init__(self, provider: str, api_key: str) -> None:
        self.provider = provider
        self.api_key = api_key

    async def complete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            f"LLM 호출 미구현 (provider={self.provider}). provider 확정 후 SDK 연동 필요."
        )


def get_llm_client() -> LLMClient:
    s = get_settings()
    return LLMClient(s.llm_provider, s.llm_api_key)
