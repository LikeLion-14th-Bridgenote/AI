from pydantic import BaseModel


class Translation(BaseModel):
    """참가자별 번역 결과."""

    participant_id: str
    lang: str
    text: str


class Message(BaseModel):
    """에러 응답 규약: { "message": "..." }"""

    message: str
