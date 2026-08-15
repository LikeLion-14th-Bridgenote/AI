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


def _patch(monkeypatch, out, translated=None, calls=None):
    """생성/번역 두 종류의 호출을 구분해 목킹. 번역 기본값은 각 항목에 'EN:'을 붙여 되돌려줌."""
    async def fake_complete(prompt, system="", max_tokens=2048):
        is_translate = prompt.startswith("# Translate")
        if calls is not None:
            calls.append(("translate" if is_translate else "generate", prompt))
        if not is_translate:
            return out
        if translated is not None:
            return translated
        items = json.loads(next(l for l in prompt.splitlines() if l.startswith("[")))
        return json.dumps([f"EN:{t}" for t in items], ensure_ascii=False)
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
    # 같은 회의록의 번역본이므로 언어가 달라도 항목 개수는 동일
    by_cat = {}
    for s in r.minutes:
        by_cat.setdefault(s.job_role, []).append(s)
    for ko, en in by_cat.values():
        assert len(ko.decisions) == len(en.decisions)
        assert len(ko.discussions) == len(en.discussions)
        assert len(ko.action_items) == len(en.action_items)


def test_base_language_generates_and_others_translate(monkeypatch):
    calls = []
    _patch(monkeypatch, json.dumps({"decisions": ["금요일까지 초안"], "discussions": [],
           "action_items": []}), calls=calls)
    r = asyncio.run(service.generate_minutes(_req(job_roles=["개발 IT"])))
    # 기준 언어(ko) 1회 생성 + 나머지 언어(en) 1회 번역 — 언어마다 새로 요약하지 않는다
    assert [k for k, _ in calls] == ["generate", "translate"]
    assert "ko" in calls[0][1] and "금요일까지 초안 공유할게요" in calls[0][1]  # 생성엔 전사가 들어감
    assert "en" in calls[1][1] and "금요일까지 초안 공유할게요" not in calls[1][1]  # 번역엔 결과만
    assert r.minutes[0].language == "ko" and r.minutes[0].decisions == ["금요일까지 초안"]
    assert r.minutes[1].language == "en" and r.minutes[1].decisions == ["EN:금요일까지 초안"]


def test_translation_length_mismatch_falls_back_to_base(monkeypatch):
    # 번역 결과가 2개인데 원본은 3개 → 어느 항목인지 알 수 없으므로 기준 언어 내용 유지
    _patch(monkeypatch, json.dumps({"decisions": ["결정1", "결정2"], "discussions": ["논의1"],
           "action_items": []}), translated=json.dumps(["EN:1", "EN:2"]))
    r = asyncio.run(service.generate_minutes(_req(job_roles=["개발 IT"])))
    ko, en = r.minutes
    assert en.language == "en"  # 섹션 자체는 나온다(예외·500 아님)
    assert en.decisions == ko.decisions == ["결정1", "결정2"]
    assert en.discussions == ko.discussions == ["논의1"]


def test_owner_is_not_translated(monkeypatch):
    _patch(monkeypatch, json.dumps({"decisions": [], "discussions": [], "action_items": [
        {"task": "API 마무리", "owner": "종윤", "deadline": "금요일"},
    ]}))
    r = asyncio.run(service.generate_minutes(_req(job_roles=["개발 IT"])))
    en = r.minutes[1].action_items[0]
    assert en.owner == "종윤"  # 사람 이름은 그대로
    assert en.task == "EN:API 마무리" and en.deadline == "EN:금요일"


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
