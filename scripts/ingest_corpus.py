"""culture_corpus 적재 — 시드(JSONL)를 임베딩해서 Supabase pgvector에 upsert.

    python scripts/ingest_corpus.py

시드: data/seed/culture_corpus_seed.jsonl
필요: .env 의 SUPABASE_DB_URL, 최초 실행 시 임베딩 모델 다운로드(약 0.22GB).
재실행하면 id 기준 upsert(덮어쓰기)라 안전하다.
"""

import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.core.config import get_settings
from app.core.embeddings import embed_texts

SEED = Path(__file__).resolve().parents[1] / "data" / "seed" / "culture_corpus_seed.jsonl"


def main() -> None:
    settings = get_settings()
    rows = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"시드 {len(rows)}건 임베딩 중...")
    vectors = embed_texts([r["rule_text"] for r in rows])

    with psycopg.connect(settings.supabase_db_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS culture_corpus (
                id         TEXT PRIMARY KEY,
                rule_text  TEXT NOT NULL,
                culture    TEXT NOT NULL,
                note_type  TEXT,
                source     TEXT,
                entry_type TEXT,
                embedding  vector({settings.embed_dim})
            )
            """
        )
        register_vector(conn)
        with conn.cursor() as cur:
            for row, vec in zip(rows, vectors):
                cur.execute(
                    """
                    INSERT INTO culture_corpus
                        (id, rule_text, culture, note_type, source, entry_type, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        rule_text = EXCLUDED.rule_text,
                        culture   = EXCLUDED.culture,
                        note_type = EXCLUDED.note_type,
                        source    = EXCLUDED.source,
                        embedding = EXCLUDED.embedding
                    """,
                    (row["id"], row["rule_text"], row["culture"], row.get("note_type"),
                     row.get("source"), row.get("entry_type"), vec),
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS culture_corpus_embed_idx "
            "ON culture_corpus USING hnsw (embedding vector_cosine_ops)"
        )
    print(f"적재 완료: {len(rows)}건 → culture_corpus")


if __name__ == "__main__":
    main()
