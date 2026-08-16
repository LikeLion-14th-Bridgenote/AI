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
    db_pool_max: int = 10  # 코퍼스 조회용 커넥션 풀 상한

    # 임베딩 (로컬 fastembed, 다국어) — 게이트·코퍼스 공용
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = 384

    # 오해 감지 게이트 튜닝 — 값 근거는 scripts/eval_gate.py (누수 없는 58건 기준)
    #
    # 판정식: risk_sim >= base - sensitivity * cultural_distance  AND  risk_sim >= neutral_sim - neutral_margin
    #
    # 판정을 실제로 가르는 건 뒤쪽(risk vs neutral) 비교다. 문턱은 거의 작동하지 않는데,
    # 이유가 코퍼스 크기가 아니라 **문화별 유사도 스케일 차이**로 밝혀졌다:
    # 한국어는 평범한 발화도 risk_seed와 0.67 닮게 나오고(영어는 0.26), 그래서 같은
    # base=0.35가 KR에서는 23건 중 22건을 통과시키고 US에서는 12건 중 6건만 통과시킨다.
    # 전역 상수 하나로는 두 언어를 동시에 맞출 수 없다.
    #
    # 문화별 문턱(그 문화의 neutral 시드가 risk 시드와 닮은 정도를 기준선으로)도 재봤는데
    # F1 0.909로 오히려 나빴다. 문화당 표본이 3~23건으로 얇아 상수 하나로 대표가 안 된다.
    # 같은 실험을 반복하지 않도록 적어둔다. 근본 해결은 한국어 변별력이 나은 임베딩 모델이다.
    gate_base_threshold: float = 0.35  # 격자 최고점은 0.30 (margin 0.05와 함께 F1 0.941)
    gate_distance_sensitivity: float = 0.05  # 문화가 멀수록 문턱을 낮춤 — 현재 영향 없음
    # neutral 비교 완화 마진: risk_sim이 neutral_sim보다 이 값 이내로만 낮으면 통과시킨다.
    # 다국어 MiniLM은 마진이 얇아 STT 변형(띄어쓰기 등) 완곡표현이 neutral에 살짝 밀려 놓치는 일이 잦음.
    # margin=0으로 두면 recall이 0.75(FN 8건)로 무너지므로 마진 자체는 필요하다.
    # 다만 58건 기준 0.05가 0.10을 모든 축에서 이긴다(recall 동일 0.97, FP 4 vs 7).
    # 42건 시절엔 observed만 0.10이 유리해 보였는데 관찰 사례가 늘자 뒤집혔다 — 표본 착시였다.
    gate_neutral_margin: float = 0.10  # 0.05 제안 (F1 0.886 -> 0.925). 변경은 팀 합의 후
    gate_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
