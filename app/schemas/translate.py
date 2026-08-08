from typing import List

from pydantic import BaseModel

from app.schemas.common import Translation


class TranslateTarget(BaseModel):
    participant_id: str
    lang: str


class TranslateRequest(BaseModel):
    """POST /ai/translate 요청. 게이트 미통과(각주 불필요) 발화의 번역만 단독 처리."""

    utterance_id: str
    source_text: str
    source_lang: str
    targets: List[TranslateTarget]


class TranslateResponse(BaseModel):
    utterance_id: str
    translations: List[Translation]
