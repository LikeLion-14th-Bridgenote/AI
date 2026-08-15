"""각주 스모크 테스트 — 실제 LLM(gpt-4o)에 붙여 각주 품질을 눈으로 확인.

Supabase 없이 로컬 country_scores + 인라인 규칙으로 프롬프트를 만들어
get_llm_client().complete()를 실제 호출한다. .env의 LLM_API_KEY만 있으면 동작.

    python scripts/smoke_analyze.py

여러 발화를 돌려 각주 JSON을 출력한다. openai/gemini/claude 어느 provider든
.env만 바꾸면 그대로 비교 가능.
"""

import asyncio
import json

from app.core.llm import get_llm_client
from app.prompts.culture_map import ANALYZE_SYSTEM_PROMPT, build_user_prompt
from app.rag.culture_distance import _scores
from app.schemas.analyze import AnalyzeRequest

# 데모 케이스 (게이트는 건너뛰고 각주 생성만 검증)
CASES = [
    dict(source_text="네, 한번 검토해보겠습니다", source_lang="ko",
         speaker={"culture": "KR", "job": "pm"},
         listeners=[{"participant_id": "p1", "lang": "en", "culture": "US", "job": "dev"}],
         meeting_context="법률 자문 미팅"),
    dict(source_text="That's an interesting idea, I'll bear it in mind", source_lang="en",
         speaker={"culture": "GB", "job": "pm"},
         listeners=[{"participant_id": "p1", "lang": "en", "culture": "US", "job": "dev"}],
         meeting_context="스타트업 킥오프"),
    dict(source_text="I will try my best to get it done", source_lang="en",
         speaker={"culture": "IN", "job": "dev"},
         listeners=[{"participant_id": "p1", "lang": "ko", "culture": "KR", "job": "pm"}],
         meeting_context="오프쇼어 개발 미팅"),
]

# 인라인 규칙 (DB 없이) — 실제로는 RAG가 청자 문화 코퍼스에서 검색
INLINE_RULES = [
    {"culture": "US", "rule_text": "Low-context; takes words literally. 'I'll review it' reads as commitment."},
    {"culture": "US", "rule_text": "A polite deferral is heard as agreement unless a clear 'no' is stated."},
    {"culture": "KR", "rule_text": "High-context; a soft 'I'll try' may be a face-saving decline."},
]


async def main() -> None:
    client = get_llm_client()
    print(f"provider={client.provider} model={client.model}\n")
    for i, c in enumerate(CASES, 1):
        req = AnalyzeRequest(sentence_id=f"u{i}", meeting_id="m1", **c)
        prompt = build_user_prompt(req, INLINE_RULES, _scores())
        raw = await client.complete(prompt, system=ANALYZE_SYSTEM_PROMPT)
        print(f"=== 케이스 {i}: [{req.speaker.culture}→{req.listeners[0].culture}] {req.source_text}")
        try:
            print(json.dumps(json.loads(raw), ensure_ascii=False, indent=2))
        except Exception:
            print("(JSON 파싱 실패, 원문 출력)\n" + raw)
        print()


if __name__ == "__main__":
    asyncio.run(main())
