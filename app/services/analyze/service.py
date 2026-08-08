from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.schemas.common import Translation
from app.services.analyze.gate import passes_gate


async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    /ai/analyze 오케스트레이션 (스텁).
    1) 게이트 판정 → 2) 통과 시 LLM 각주·리라이트·번역, 미통과 시 번역만.
    현재는 번역/각주 로직이 자리표시자다.
    """
    has_risk = await passes_gate(req)

    # TODO: 실제 번역(LLM/번역 API)으로 교체. 지금은 빈 텍스트 자리표시자.
    translations = [
        Translation(participant_id=l.participant_id, lang=l.lang, text="")
        for l in req.listeners
    ]

    if not has_risk:
        return AnalyzeResponse(
            utterance_id=req.utterance_id,
            has_risk=False,
            translations=translations,
        )

    # TODO: 게이트 통과분 — LLM으로 Culture Map 8축 기반 각주 생성.
    return AnalyzeResponse(
        utterance_id=req.utterance_id,
        has_risk=True,
        risk_level="Low",
        note_type="",
        speaker_intent="",
        listener_misread="",
        advice="",
        rewrite_text="",
        translations=translations,
    )
