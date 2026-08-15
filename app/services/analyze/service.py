"""/ai/analyze 오케스트레이션.

1) 임베딩 게이트로 오해 소지 판정
2) 통과분: RAG(청자 문화 코퍼스) → Culture Map 프롬프트 → LLM → 각주 JSON 파싱
   같은 호출에서 참가자 언어별 번역까지 받아 채운다.
3) 미통과분: 각주 없이 번역만 (translate 서비스가 담당, 여기선 빈 값 자리표시).
"""

import json
import logging

from app.core.embeddings import embed_one
from app.core.llm import get_llm_client
from app.prompts.culture_map import ANALYZE_SYSTEM_PROMPT, NOTE_TYPES, build_user_prompt
from app.rag.corpus import search_by_vector
from app.rag.culture_distance import _scores  # country_scores.json 로더 재사용
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.schemas.common import Translation
from app.services.analyze.gate import passes_gate

logger = logging.getLogger(__name__)


def _as_bool(v) -> bool:
    """LLM이 boolean 대신 문자열('true'/'false')을 줄 때를 방어."""
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _note_type(v) -> str:
    """프론트 문화 가이드 탭이 이 세 값으로 칩·통계를 그린다. 벗어나면 집계에서 빠지므로 보정."""
    s = (v or "").strip()
    if s in NOTE_TYPES:
        return s
    # ponytail: 부분일치만 본다. 더 정교한 매핑이 필요해지면 라벨 사전을 두자.
    for t in NOTE_TYPES:
        if t in s or s in t:
            return t
    return NOTE_TYPES[0]  # 판단 불가 시 가장 포괄적인 "문화 이해"


def _empty_translations(req: AnalyzeRequest):
    return [
        Translation(participant_id=l.participant_id, lang=l.lang, text="")
        for l in req.listeners
    ]


async def _gather_rules(req: AnalyzeRequest, top_k: int = 5):
    """청자 문화별로 관련 규칙을 모아 각주 근거로 쓴다 (발화당 1회 임베딩)."""
    vec = embed_one(req.source_text)
    seen, rules = set(), []
    for listener in req.listeners:
        if listener.culture in seen:
            continue
        seen.add(listener.culture)
        rules += await search_by_vector(vec, listener.culture, top_k=top_k)
    return rules


def _map_translations(req: AnalyzeRequest, raw: list) -> list:
    """LLM이 lang별로 준 번역을 participant_id에 매핑 (같은 lang은 공유)."""
    by_lang = {t.get("lang"): t.get("text", "") for t in raw if isinstance(t, dict)}
    return [
        Translation(participant_id=l.participant_id, lang=l.lang, text=by_lang.get(l.lang, ""))
        for l in req.listeners
    ]


async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    has_risk = await passes_gate(req)
    if not has_risk:
        # 게이트 미통과: 각주 불필요. 번역은 translate 서비스 몫(여기선 빈 값).
        return AnalyzeResponse(
            sentence_id=req.sentence_id, has_risk=False,
            translations=_empty_translations(req),
        )

    try:
        rules = await _gather_rules(req)
        prompt = build_user_prompt(req, rules, _scores())
        raw = await get_llm_client().complete(prompt, system=ANALYZE_SYSTEM_PROMPT)
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        # LLM/파싱 실패: 각주는 못 달지만 500 대신 안전 응답(번역만).
        logger.warning("각주 생성 실패 → 각주 생략", exc_info=True)
        return AnalyzeResponse(
            sentence_id=req.sentence_id, has_risk=False,
            translations=_empty_translations(req),
        )

    # LLM이 재검토 후 위험 없다고 판단하면 각주 생략
    # 방어: LLM이 boolean 대신 문자열 "false"를 줄 수 있음 (truthy 함정)
    if not _as_bool(data.get("has_risk", True)):
        return AnalyzeResponse(
            sentence_id=req.sentence_id, has_risk=False,
            translations=_map_translations(req, data.get("translations", [])),
        )

    return AnalyzeResponse(
        sentence_id=req.sentence_id,
        has_risk=True,
        risk_level=data.get("risk_level", "Med"),
        note_type=_note_type(data.get("note_type")),
        speaker_intent=data.get("speaker_intent", ""),
        listener_misread=data.get("listener_misread", ""),
        advice=data.get("advice", ""),
        rewrite_text=data.get("rewrite_text", ""),
        translations=_map_translations(req, data.get("translations", [])),
    )
