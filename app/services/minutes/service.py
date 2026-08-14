"""/ai/minutes — 회의 종료 후 배치. 세션 로그 → 언어 × 직무 category별 회의록.

직무는 job_mapping으로 6개 category로 묶어 탭 폭발을 막는다.
언어는 "다른 요약"이 아니라 "같은 회의록의 렌더링"이므로, category당 기준 언어로 1회만
생성하고 나머지 언어는 그 결과를 번역해서 채운다(항목 개수·순서가 언어 간 항상 일치).
생성 실패는 빈 섹션, 번역 실패는 기준 언어 내용으로 두고 계속(fail-open).
"""

import json
import logging

from app.core.job_mapping import categories_of
from app.core.llm import get_llm_client
from app.prompts.minutes import (
    MINUTES_SYSTEM_PROMPT,
    MINUTES_TRANSLATE_SYSTEM_PROMPT,
    build_minutes_prompt,
    build_minutes_translate_prompt,
)
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


def _translatable(section: MinutesSection) -> list:
    """번역 대상 문자열을 순서대로 평탄화. owner는 사람 이름이라 제외, deadline("금요일")은 포함."""
    texts = [*section.decisions, *section.discussions]
    for a in section.action_items:
        texts.append(a.task)
        if a.deadline:
            texts.append(a.deadline)
    return texts


def _rebuild(section: MinutesSection, language: str, texts: list) -> MinutesSection:
    """_translatable과 같은 순서로 번역문을 되꽂는다."""
    it = iter(texts)
    decisions = [next(it) for _ in section.decisions]
    discussions = [next(it) for _ in section.discussions]
    action_items = []
    for a in section.action_items:
        task = next(it)
        action_items.append(
            ActionItem(task=task, owner=a.owner, deadline=next(it) if a.deadline else None)
        )
    return MinutesSection(
        language=language,
        job_role=section.job_role,
        decisions=decisions,
        discussions=discussions,
        action_items=action_items,
    )


async def _translated(section: MinutesSection, language: str) -> MinutesSection:
    """기준 언어 섹션을 language로 렌더링. 실패·개수 불일치면 기준 언어 내용 그대로(fail-open)."""
    texts = _translatable(section)
    try:
        if not texts:
            return section.model_copy(update={"language": language})
        prompt = build_minutes_translate_prompt(texts, language)
        raw = await get_llm_client().complete(
            prompt, system=MINUTES_TRANSLATE_SYSTEM_PROMPT, max_tokens=2048
        )
        out = json.loads(raw)
        if not isinstance(out, list) or len(out) != len(texts):
            # 개수가 어긋나면 어느 항목이 어디로 갔는지 알 수 없어 통째로 신뢰 불가
            raise ValueError(f"번역 항목 수 불일치: {len(texts)} 요청 / 응답 {out!r:.120}")
        return _rebuild(section, language, [str(t) for t in out])
    except Exception:  # noqa: BLE001
        logger.warning(
            "회의록 번역 실패 → 기준 언어 내용 유지 (%s/%s)", language, section.job_role, exc_info=True
        )
        return section.model_copy(update={"language": language})


async def generate_minutes(req: MinutesRequest) -> MinutesResponse:
    languages = req.languages or _langs_from_utterances(req)
    categories = categories_of(req.job_roles) or [None]  # 직무 없으면 일반 섹션 1개
    base_language = languages[0]  # 기준 언어: 여기서만 생성하고 나머지는 번역

    bases = {category: await _section(req, base_language, category) for category in categories}
    sections = []
    for language in languages:
        for category in categories:
            base = bases[category]
            sections.append(
                base if language == base_language else await _translated(base, language)
            )
    return MinutesResponse(meeting_id=req.meeting_id, minutes=sections)


def _langs_from_utterances(req: MinutesRequest) -> list:
    """languages 미지정 시 발화에 등장한 언어로 대체 (없으면 ko)."""
    seen = list(dict.fromkeys(u.lang for u in req.utterances if u.lang))
    return seen or ["ko"]
