from app.schemas.common import Translation
from app.schemas.translate import TranslateRequest, TranslateResponse


async def translate(req: TranslateRequest) -> TranslateResponse:
    """/ai/translate (스텁). 게이트 미통과 발화의 번역만 단독 처리."""
    # TODO: 실제 번역(LLM/번역 API)으로 교체.
    translations = [
        Translation(participant_id=t.participant_id, lang=t.lang, text="")
        for t in req.targets
    ]
    return TranslateResponse(utterance_id=req.utterance_id, translations=translations)
