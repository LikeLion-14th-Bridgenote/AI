from app.core.config import get_settings


class STTClient:
    """
    Deepgram STT 클라이언트 자리표시자.
    (실시간 오디오 스트리밍 자체는 Spring BE realtime 모듈이 담당. 이 자리는 확장 대비.)
    """

    def __init__(self, provider: str, api_key: str) -> None:
        self.provider = provider
        self.api_key = api_key


def get_stt_client() -> STTClient:
    s = get_settings()
    return STTClient(s.stt_provider, s.stt_api_key)
