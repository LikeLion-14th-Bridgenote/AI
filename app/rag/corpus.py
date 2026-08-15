"""culture_corpus (Supabase pgvector) 유사도 검색.

테이블: culture_corpus(id, rule_text, culture, note_type, source, entry_type, embedding vector)
문화(culture)로 필터한 뒤 코사인 유사도 상위 top_k를 반환한다.
적재는 scripts/ingest_corpus.py 참고.

발화 1건마다 게이트에서 2회(risk_seed·neutral_seed), 각주 RAG에서 청자 문화 수만큼
검색이 일어난다. 매번 새 커넥션을 열면 TCP+TLS 핸드셰이크가 그만큼 쌓이고 Supabase
커넥션 한도에도 먼저 부딪히므로 풀을 하나 두고 재사용한다.

psycopg의 async 드라이버는 Windows 기본 이벤트 루프(Proactor)에서 동작하지 않는다.
팀원 로컬이 대부분 Windows라, 동기 풀을 쓰되 조회를 스레드로 넘겨 이벤트 루프를
막지 않는 방식을 택했다. 스레드 왕복 비용은 있지만 네트워크 대기에 비하면 무시할 수준이고
플랫폼을 가리지 않는다.
"""

import logging
from typing import List, Optional

import anyio.to_thread
import numpy as np
from psycopg_pool import ConnectionPool

from app.core.config import get_settings
from app.core.embeddings import embed_one

logger = logging.getLogger(__name__)

_pool: Optional[ConnectionPool] = None


def _configure(conn):
    """커넥션마다 vector 타입 어댑터 등록 (풀이 새 커넥션을 열 때 호출)."""
    from pgvector.psycopg import register_vector

    register_vector(conn)


def _get_pool() -> ConnectionPool:
    """지연 생성 — import 시점에 DB에 붙지 않도록(테스트·오프라인 실행 대비)."""
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = ConnectionPool(
            s.supabase_db_url,
            min_size=1,
            max_size=s.db_pool_max,
            configure=_configure,
            timeout=10,
            open=True,
        )
        logger.info("culture_corpus 커넥션 풀 생성 (max_size=%s)", s.db_pool_max)
    return _pool


def close_pool() -> None:
    """앱 종료 시 정리."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def _search_sync(vec, target_culture: str, top_k: int, entry_type: str) -> List[dict]:
    where = "culture = %s" + (" AND entry_type = %s" if entry_type else "")
    params = [vec, target_culture] + ([entry_type] if entry_type else []) + [vec, top_k]
    with _get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT rule_text, note_type, culture, 1 - (embedding <=> %s) AS similarity "
            f"FROM culture_corpus WHERE {where} "
            "ORDER BY embedding <=> %s LIMIT %s",
            tuple(params),
        ).fetchall()
    return [
        {"rule_text": r[0], "note_type": r[1], "culture": r[2], "similarity": float(r[3])}
        for r in rows
    ]


async def search_by_vector(
    vec: List[float], target_culture: str, top_k: int = 5, entry_type: str = None
) -> List[dict]:
    """미리 계산한 임베딩으로 검색 (게이트가 발화당 1회만 임베딩하도록 분리).

    entry_type=None이면 전체, 'rule'/'risk_seed'/'neutral_seed'면 해당 종류만.
    게이트는 화자 문화의 risk_seed·neutral_seed로, 각주 RAG는 청자 문화의 rule로 검색한다.
    """
    v = np.asarray(vec, dtype=np.float32)  # 리스트는 float8[]로 어댑트돼 <=> 연산 불가
    return await anyio.to_thread.run_sync(_search_sync, v, target_culture, top_k, entry_type)


async def search_rules(
    text: str, target_culture: str, top_k: int = 5, entry_type: str = None
) -> List[dict]:
    """텍스트를 임베딩해서 검색 (단건 편의 함수)."""
    return await search_by_vector(await embed_one(text), target_culture, top_k, entry_type)
