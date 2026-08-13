"""/ai/minutes — 회의 종료 후 배치. 세션 로그 → 언어 × 직무 category별 회의록.

직무는 job_mapping으로 6개 category로 묶어 탭 폭발을 막는다.
조합마다 1회 LLM 호출. 한 섹션이 실패해도 빈 섹션으로 두고 계속(fail-open).
"""

import json
import logging

from app.core.job_mapping import categories_of
from app.core.llm import get_llm_client
from app.prompts.minutes import MINUTES_SYSTEM_PROMPT, build_minutes_prompt
from app.schemas.minutes import ActionItem, MinutesRequest, MinutesResponse, MinutesSection

logger = logging.getLogger(__name__)

# LLM이 액션아이템을 문자열/영문키/한글키 어느 형태로 줘도 받아낸다
_OWNER_KEYS = ("owner", "담당자", "담당")
_DEADLINE_KEYS = ("deadline", "마감일", "기한", "마감")
_TASK_KEYS = ("task", "내용", "할일", "action")


def _to_action(item) -> ActionItem:
    if isinstance(item, str):
        return ActionItem(task=item)
    if isinstance(item, dict):
        pick = lambda keys: next((item[k] for k in keys if item.get(k)), None)
        return ActionItem(
            task=pick(_TASK_KEYS) or str(item),
            owner=pick(_OWNER_KEYS),
            deadline=pick(_DEADLINE_KEYS),
        )
    return ActionItem(task=str(item))


async def _section(req: MinutesRequest, language: str, category) -> MinutesSection:
    try:
        prompt = build_minutes_prompt(req.utterances, language, category)
        raw = await get_llm_client().complete(prompt, system=MINUTES_SYSTEM_PROMPT, max_tokens=2048)
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.warning("회의록 섹션 생성 실패 → 빈 섹션 (%s/%s)", language, category, exc_info=True)
        data = {}
    return MinutesSection(
        language=language,
        job_role=category,
        decisions=data.get("decisions", []) or [],
        discussions=data.get("discussions", []) or [],
        action_items=[_to_action(a) for a in (data.get("action_items", []) or [])],
    )


async def generate_minutes(req: MinutesRequest) -> MinutesResponse:
    languages = req.languages or _langs_from_utterances(req)
    categories = categories_of(req.job_roles) or [None]  # 직무 없으면 일반 섹션 1개

    sections = []
    for language in languages:
        for category in categories:
            sections.append(await _section(req, language, category))
    return MinutesResponse(meeting_id=req.meeting_id, minutes=sections)


def _langs_from_utterances(req: MinutesRequest) -> list:
    """languages 미지정 시 발화에 등장한 언어로 대체 (없으면 ko)."""
    seen = list(dict.fromkeys(u.lang for u in req.utterances if u.lang))
    return seen or ["ko"]
