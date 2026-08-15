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

    # 오해 감지 게이트 튜닝 — 값 근거는 scripts/eval_gate.py (누수 없는 42건 기준)
    #
    # 판정식: risk_sim >= base - sensitivity * cultural_distance  AND  risk_sim >= neutral_sim
    #
    # 실측하면 판정을 실제로 가르는 건 뒤쪽(risk vs neutral) 비교다. 42건 중 탈락 21건이
    # 전부 그 비교에서 걸리고 문턱에서 걸리는 건 0건이다. 즉 아래 두 값은 지금
    # 사실상 작동하지 않는다. sens를 0.8까지 올려도 F1이 0.88~0.90에서 움직이지 않는다.
    # 값을 지우지 않는 이유는, 코퍼스가 커져 유사도 분포가 벌어지면 다시 의미를 갖기 때문이다.
    # 자세한 진단은 이슈 #10 참고.
    gate_base_threshold: float = 0.35  # 스윕 최고점 (F1 0.93, P 1.00 / R 0.86)
    gate_distance_sensitivity: float = 0.05  # 문화가 멀수록 문턱을 낮춤 — 현재 영향 없음
    gate_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
