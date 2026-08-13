"""번역 + 회의록 스모크 — 실제 LLM으로 두 기능 품질 확인.

.env의 LLM_API_KEY만 있으면 동작 (Supabase·게이트 불필요).

    python scripts/smoke_others.py
"""

import asyncio
import json

from app.schemas.minutes import MinutesRequest
from app.schemas.translate import TranslateRequest
from app.services.minutes.service import generate_minutes
from app.services.translate.service import translate


async def main() -> None:
    print("=== 번역 (게이트 미통과 발화) ===")
    treq = TranslateRequest(
        sentence_id="u1", source_text="금요일까지 자료 공유 부탁드립니다", source_lang="ko",
        targets=[{"participant_id": "p1", "lang": "en"}, {"participant_id": "p2", "lang": "vi"}],
    )
    tr = await translate(treq)
    for t in tr.translations:
        print(f"  [{t.lang}] {t.text}")

    print("\n=== 회의록 (언어 × 직무) ===")
    mreq = MinutesRequest(
        meeting_id="m1", languages=["ko", "en"], job_roles=["개발 IT", "기획 PM"],
        utterances=[
            {"sentence_id": "u1", "speaker": "종윤", "lang": "ko", "text": "로그인 API는 금요일까지 제가 마무리하겠습니다"},
            {"sentence_id": "u2", "speaker": "Minh", "lang": "vi", "text": "Tôi sẽ lo phần giao diện đăng nhập"},
            {"sentence_id": "u3", "speaker": "수민", "lang": "ko", "text": "디자인 시안은 수요일에 공유할게요"},
            {"sentence_id": "u4", "speaker": "종윤", "lang": "ko", "text": "그럼 다음 회의는 금요일 오후 3시로 하겠습니다"},
        ],
    )
    mr = await generate_minutes(mreq)
    for s in mr.minutes:
        print(f"\n  [{s.language} / {s.job}]")
        print("   결정:", s.decisions)
        print("   논의:", s.discussions)
        print("   액션:", s.action_items)


if __name__ == "__main__":
    asyncio.run(main())
