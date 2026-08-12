"""오해 감지 게이트 — 임베딩 유사도 + 문화 거리 가중 1차 필터.

발화를 임베딩해 청자 문화의 위험표현 코퍼스와 유사도를 재고,
화자↔청자 문화 거리로 문턱을 조정한다(먼 문화일수록 민감).
통과분만 LLM 상세 분석(각주)으로 넘어간다.

    effective_threshold = base - distance_sensitivity * cultural_distance
    유사도 >= effective_threshold 인 청자가 하나라도 있으면 통과.
"""

import logging

from app.core.config import get_settings
from app.core.embeddings import embed_one
from app.rag.corpus import search_by_vector
from app.rag.culture_distance import cultural_distance
from app.schemas.analyze import AnalyzeRequest

logger = logging.getLogger(__name__)


async def passes_gate(req: AnalyzeRequest) -> bool:
    s = get_settings()
    try:
        vec = embed_one(req.source_text)  # 발화당 1회만 임베딩

        for listener in req.listeners:
            rules = await search_by_vector(vec, listener.culture, top_k=s.gate_top_k)
            if not rules:
                continue
            top_sim = max(r["similarity"] for r in rules)
            dist = cultural_distance(req.speaker.culture, listener.culture)
            effective = s.gate_base_threshold - s.gate_distance_sensitivity * dist
            if top_sim >= effective:
                return True

        return False
    except Exception:  # noqa: BLE001
        # fail-open: 코퍼스/임베딩 인프라 오류 시 각주는 건너뛰되(위험 없음 처리)
        # 번역 등 나머지 응답은 정상 반환되도록 500을 내지 않는다.
        logger.warning("게이트 판정 실패 → 각주 생략(fail-open)", exc_info=True)
        return False
