from typing import List

from app.core.config import get_settings


async def search_rules(
    text: str, source_lang: str, target_lang: str, top_k: int = 5
) -> List[dict]:
    """
    culture_corpus (Supabase pgvector) 유사도 검색 (스텁).
    테이블: (id, rule_text, source_lang, target_lang, embedding VECTOR)
    """
    # TODO: 임베딩 생성 후 pgvector <=> 연산으로 top_k 규칙 검색.
    _ = get_settings().supabase_db_url
    return []
