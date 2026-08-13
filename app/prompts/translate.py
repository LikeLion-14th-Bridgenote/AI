"""번역 프롬프트 — 게이트 미통과(각주 불필요) 발화의 순수 번역.

여러 타겟 언어를 한 번의 LLM 호출로 처리한다(발화당 1콜).
직전 맥락은 대명사·생략어 보정용으로만 참고.
"""

import json
from typing import List

from app.schemas.translate import TranslateRequest


def build_translate_prompt(req: TranslateRequest, context: List[str]) -> str:
    langs = ", ".join(sorted({t.lang for t in req.targets}))
    ctx = "\n".join(f"  - {c}" for c in context) or "  (none)"
    return (
        "Translate the utterance into each target language. "
        "Keep meeting register; use prior turns only to resolve pronouns/ellipsis.\n\n"
        f"# Utterance ({req.source_lang})\n  {req.source_text}\n\n"
        f"# Prior turns\n{ctx}\n\n"
        f"# Target languages\n  {langs}\n\n"
        '# Output\nReturn ONLY JSON: {"translations": [{"lang": "en", "text": "..."}]}. '
        "One entry per target language, no prose, no fences.\n"
        f"Shape: {json.dumps({'translations': [{'lang': 'en', 'text': '...'}]}, ensure_ascii=False)}"
    )
