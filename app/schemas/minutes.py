from typing import List, Optional

from pydantic import BaseModel, Field


class MinutesUtterance(BaseModel):
    """회의록 생성 입력: 세션 로그의 발화 한 건."""

    utterance_id: str
    speaker: Optional[str] = None
    lang: str
    text: str


class MinutesRequest(BaseModel):
    """POST /ai/minutes 요청 (배치). 상세 스키마 미확정 — 최소 형태."""

    meeting_id: str
    languages: List[str] = Field(default_factory=list)
    job_roles: List[str] = Field(default_factory=list)
    utterances: List[MinutesUtterance] = Field(default_factory=list)


class ActionItem(BaseModel):
    """액션 아이템 — 동균 UI '액션 탭'(담당자/기한 컬럼)과 정합."""

    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None


class MinutesSection(BaseModel):
    """언어·직무별 회의록 한 섹션 (결정/논의/액션)."""

    language: str
    job_role: Optional[str] = None
    decisions: List[str] = Field(default_factory=list)
    discussions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)


class MinutesResponse(BaseModel):
    meeting_id: str
    minutes: List[MinutesSection] = Field(default_factory=list)
