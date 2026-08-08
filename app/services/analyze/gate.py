from app.schemas.analyze import AnalyzeRequest


async def passes_gate(req: AnalyzeRequest) -> bool:
    """
    오해 감지 게이트 (스텁).

    실제 로직: pgvector 임베딩 유사도 + 문화 거리 가중으로 1차 필터.
    게이트를 통과한 발화만 LLM 상세 분석으로 넘긴다.
    지금은 항상 False(위험 없음) 를 반환하는 자리표시자.
    """
    # TODO: rag.corpus.search_rules + 문화 거리 가중으로 판정
    return False
