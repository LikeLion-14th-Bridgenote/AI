"""culture_corpus (Supabase pgvector) 유사도 검색.

테이블: culture_corpus(id, rule_text, culture, note_type, source, entry_type, embedding vector)
청자 문화(culture)로 필터한 뒤 코사인 유사도 상위 top_k 규칙을 반환한다.
적재는 scripts/ingest_corpus.py 참고.
"""

from typing import List

import psycopg
from pgvector.psycopg import register_vector

from app.core.config import get_settings
from app.core.embeddings import embed_one


async def search_by_vector(
    vec: List[float], target_culture: str, top_k: int = 5
) -> List[dict]:
    """미리 계산한 임베딩으로 검색 (게이트가 발화당 1회만 임베딩하도록 분리)."""
    # ponytail: 동기 psycopg를 async 함수 안에서 호출 → 이벤트 루프 블로킹.
    #           MVP엔 충분, 부하 커지면 psycopg AsyncConnection 또는 스레드풀로 교체.
    dsn = get_settings().supabase_db_url
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        rows = conn.execute(
            "SELECT rule_text, note_type, culture, 1 - (embedding <=> %s) AS similarity "
            "FROM culture_corpus WHERE culture = %s "
            "ORDER BY embedding <=> %s LIMIT %s",
            (vec, target_culture, vec, top_k),
        ).fetchall()
    return [
        {"rule_text": r[0], "note_type": r[1], "culture": r[2], "similarity": float(r[3])}
        for r in rows
    ]


async def search_rules(text: str, target_culture: str, top_k: int = 5) -> List[dict]:
    """텍스트를 임베딩해서 검색 (단건 편의 함수)."""
    return await search_by_vector(embed_one(text), target_culture, top_k)
