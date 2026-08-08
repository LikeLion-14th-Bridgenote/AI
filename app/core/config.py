from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수(.env) 로드 및 검증. 값 목록은 .env.example 참고."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (Claude vs OpenAI — 미확정. provider 스위치로 처리)
    llm_provider: str = "claude"
    llm_api_key: str = ""

    # STT (Deepgram)
    stt_provider: str = "deepgram"
    stt_api_key: str = ""

    # Supabase pgvector (culture_corpus 조회용)
    supabase_db_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
