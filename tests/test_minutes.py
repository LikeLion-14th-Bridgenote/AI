"""회의록 + 직무 매핑 테스트 (LLM 목킹, 네트워크 없이)."""

import asyncio
import json

from app.core import job_mapping
from app.schemas.minutes import MinutesRequest
from app.services.minutes import service


# --- 직무 매핑 ---

def test_job_mapping_detail_to_category():
    assert job_mapping.to_category("데이터 분석") == "dev"
    assert job_mapping.to_category("기획 PM") == "pm"
    assert job_mapping.to_category("법무") == "etc"  # MVP는 etc
    assert job_mapping.to_category("") == "etc"


def test_categories_dedup_and_order():
    # 개발IT·데이터분석 둘 다 dev로 → 한 번만, CATEGORIES 순서 유지
    cats = job_mapping.categories_of(["데이터 분석", "개발 IT", "디자인"])
    assert cats == ["dev", "design"]


# --- 회의록 ---

def _req(job_roles=None):
    return MinutesRequest(
        meeting_id="m1",
        languages=["ko", "en"],
        job_roles=job_roles or [],
        utterances=[
            {"sentence_id": "u1", "speaker": "종윤", "lang": "ko", "text": "금요일까지 초안 공유할게요"},
        ],
    )


def _patch(monkeypatch, out):
    async def fake_complete(prompt, system="", max_tokens=2048):
        return out
    monkeypatch.setattr(service, "get_llm_client",
                        lambda: type("C", (), {"complete": staticmethod(fake_complete)})())


def test_sections_are_language_times_category(monkeypatch):
    _patch(monkeypatch, json.dumps({"decisions": ["d"], "discussions": [],
           "action_items": [{"task": "API 마무리", "owner": "종윤", "deadline": "금요일"}]}))
    r = asyncio.run(service.generate_minutes(_req(job_roles=["개발 IT", "디자인"])))
    # 언어 2 × category 2(dev,design) = 4 섹션
    assert len(r.minutes) == 4
    langs = {s.language for s in r.minutes}
    cats = {s.job_role for s in r.minutes}
    assert langs == {"ko", "en"} and cats == {"dev", "design"}
    assert r.minutes[0].decisions == ["d"]
    assert r.minutes[0].action_items[0].owner == "종윤"


def test_action_item_accepts_string_and_korean_keys(monkeypatch):
    # LLM이 문자열, 한글키 dict 어느 형태로 줘도 정규화
    _patch(monkeypatch, json.dumps({"decisions": [], "discussions": [], "action_items": [
        "그냥 문자열 할일",
        {"내용": "디자인 시안", "담당자": "수민", "마감일": "수요일"},
    ]}))
    r = asyncio.run(service.generate_minutes(_req(job_roles=["개발 IT"])))
    items = r.minutes[0].action_items
    assert items[0].task == "그냥 문자열 할일" and items[0].owner is None
    assert items[1].task == "디자인 시안" and items[1].deadline == "수요일"


def test_no_job_roles_makes_one_section_per_language(monkeypatch):
    _patch(monkeypatch, json.dumps({"decisions": [], "discussions": [], "action_items": []}))
    r = asyncio.run(service.generate_minutes(_req(job_roles=[])))
    assert len(r.minutes) == 2  # ko, en × (일반)
    assert all(s.job_role is None for s in r.minutes)


def test_llm_failure_yields_empty_section(monkeypatch):
    async def boom(prompt, system="", max_tokens=2048):
        raise RuntimeError("down")
    monkeypatch.setattr(service, "get_llm_client",
                        lambda: type("C", (), {"complete": staticmethod(boom)})())
    r = asyncio.run(service.generate_minutes(_req(job_roles=["개발 IT"])))
    assert len(r.minutes) == 2  # 500 아님
    assert r.minutes[0].decisions == [] and r.minutes[0].job_role == "dev"
