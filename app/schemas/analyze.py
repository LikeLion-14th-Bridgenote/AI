from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import Translation


class Speaker(BaseModel):
    culture: str
    job_role: str


class Listener(BaseModel):
    participant_id: str
    lang: str
    culture: str
    job_role: str


class ContextItem(BaseModel):
    text: str
    lang: str


class AnalyzeRequest(BaseModel):
    """POST /ai/analyze 요청. Spring BE가 STT 확정 발화를 담아 호출한다."""

    utterance_id: str
    meeting_id: str
    source_text: str
    source_lang: str
    meeting_context: Optional[str] = None
    speaker: Speaker
    listeners: List[Listener]
    context: List[ContextItem] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """
    POST /ai/analyze 응답.
    - 게이트 통과(오해 소지 있음): 각주 필드 + translations
    - 게이트 미통과(오해 없음): has_risk=false + translations 만
    (None 필드는 응답에서 제외 — 라우터에서 response_model_exclude_none=True)
    """

    utterance_id: str
    has_risk: bool
    risk_level: Optional[str] = None  # High | Med | Low
    note_type: Optional[str] = None
    speaker_intent: Optional[str] = None
    listener_misread: Optional[str] = None
    advice: Optional[str] = None
    rewrite_text: Optional[str] = None
    translations: List[Translation] = Field(default_factory=list)
