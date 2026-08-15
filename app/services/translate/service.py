"""/ai/translate — 게이트 미통과 발화의 순수 번역.

타겟 언어를 한 번의 LLM 호출로 번역 → lang을 participant_id에 매핑.
실패 시 빈 텍스트로 두고 계속(fail-open).
"""

import json
import logging

from app.core.llm import get_llm_client
from app.prompts.translate import build_translate_prompt
from app.schemas.common import Translation
from app.schemas.translate import TranslateRequest, TranslateResponse

logger = logging.getLogger(__name__)


async def translate(req: TranslateRequest) -> TranslateResponse:
    by_lang = {}
    try:
        prompt = build_translate_prompt(req, context=[])
        raw = await get_llm_client().complete(prompt, max_tokens=1024)
        data = json.loads(raw)
        by_lang = {t.get("lang"): t.get("text", "") for t in data.get("translations", [])}
    except Exception:  # noqa: BLE001
        logger.warning("번역 실패 → 빈 텍스트", exc_info=True)

    translations = [
        Translation(participant_id=t.participant_id, lang=t.lang, text=by_lang.get(t.lang, ""))
        for t in req.targets
    ]
    return TranslateResponse(sentence_id=req.sentence_id, translations=translations)
