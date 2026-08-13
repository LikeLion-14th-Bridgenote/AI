"""오해 감지 게이트 — 위험표현(risk_seed) 유사도 + 문화 거리 가중 1차 필터.

발화를 임베딩해 **화자 문화의 위험표현 시드**(실제 완곡 표현들)와 유사도를 잰다.
위험표현일수록(=오해를 유발하는 화자측 표현) 유사도가 높다.
그 위험도를 화자↔청자 문화 거리로 조정한다(먼 문화일수록 민감).
통과분만 LLM 상세 분석(각주)으로 넘어간다.

    effective_threshold = base - distance_sensitivity * cultural_distance(화자, 청자)
    유사도 >= effective_threshold 인 청자가 하나라도 있으면 통과.

※ eval: scripts/eval_gate.py 로 F1 재고 base_threshold 튜닝 (risk_seed 매칭 기준).
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

        # 화자 문화의 위험표현 시드와 유사도 (발화가 알려진 완곡표현을 닮았나)
        seeds = await search_by_vector(
            vec, req.speaker.culture, top_k=s.gate_top_k, entry_type="risk_seed"
        )
        if not seeds:
            return False
        top_sim = max(r["similarity"] for r in seeds)

        # 정상표현 시드보다 더 닮아야 통과 (사실전달·인사 등 오탐 제거)
        neutrals = await search_by_vector(
            vec, req.speaker.culture, top_k=s.gate_top_k, entry_type="neutral_seed"
        )
        top_neutral = max((r["similarity"] for r in neutrals), default=0.0)
        if top_sim < top_neutral:
            return False  # 위험표현보다 정상표현에 가까움 → 통과 안 함

        # 청자 중 한 명이라도 문화 거리 기준 문턱을 넘으면 통과
        for listener in req.listeners:
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
