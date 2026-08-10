from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수(.env) 로드 및 검증. 값 목록은 .env.example 참고."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (미확정. provider 스위치로 처리 — openai | gemini | claude)
    #  - openai/gemini: OpenAI 호환 SDK 공용(base_url만 분기)
    #  - claude: anthropic SDK 별도
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_model: str = ""  # 빈 값이면 provider 기본 모델 사용 (app/core/llm.py)

    # STT (Deepgram)
    stt_provider: str = "deepgram"
    stt_api_key: str = ""

    # Supabase pgvector (culture_corpus 조회용)
    supabase_db_url: str = ""

    # 임베딩 (로컬 fastembed, 다국어) — 게이트·코퍼스 공용
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = 384

    # 오해 감지 게이트 튜닝
    # ⚠️ 캘리브레이션 값 — 코퍼스 적재 후 테스트셋(#8)으로 F1 최적점을 찾아 확정해야 함.
    #    현 다국어 MiniLM은 교차언어 절대 유사도가 낮음(관련≈0.18 vs 무관≈0.15)이라
    #    변별 마진이 얇다. 재현율/정밀도 부족 시 multilingual-e5-large(query:/passage: 프리픽스)로 교체 검토.
    gate_base_threshold: float = 0.30  # 기본 유사도 문턱(잠정)
    gate_distance_sensitivity: float = 0.05  # 문화가 멀수록 문턱 낮춤(민감)
    gate_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
