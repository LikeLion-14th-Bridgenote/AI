"""평가셋이 시드 코퍼스와 겹치는지 검사.

게이트는 시드와의 유사도로 판정하므로, 평가셋 문장이 시드에 들어 있으면 유사도가
1.0이 나와 무조건 맞힌다. 그만큼 F1이 부풀려진다. 실제로 처음 24건 중 7건이 이렇게
새고 있었고, 대소문자·쉼표만 다른 2건은 눈으로 못 잡아 한 번 더 놓쳤다.

    python scripts/check_eval_leakage.py

시드나 평가셋을 건드린 뒤에는 항상 돌릴 것. 누수가 있으면 종료코드 1.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "culture_corpus_seed.jsonl"
TESTSET = ROOT / "data" / "eval" / "gate_testset.jsonl"

# 공백·문장부호·대소문자 차이는 임베딩상 거의 같은 문장이므로 정규화해서 비교한다.
_STRIP = re.compile(r"""[\s.,!?…·\-—'"()]""")


def _norm(s: str) -> str:
    return _STRIP.sub("", s.lower())


def _load(path: Path, key: str):
    return [json.loads(line)[key] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    seeds = _load(SEED, "rule_text")
    tests = _load(TESTSET, "text")
    seed_raw = {s.strip() for s in seeds}
    seed_norm = {_norm(s) for s in seeds}

    exact = [t for t in tests if t.strip() in seed_raw]
    normalized = [t for t in tests if _norm(t) in seed_norm and t.strip() not in seed_raw]
    partial = [
        (t, s) for t in tests for s in seed_raw
        if len(s) > 5 and t.strip() != s and (s in t or t.strip() in s)
    ]

    print(f"시드 {len(seeds)}건 / 평가셋 {len(tests)}건")
    print(f"  완전 일치      : {len(exact)}")
    print(f"  정규화 후 일치 : {len(normalized)}   (대소문자·문장부호만 다른 것)")
    print(f"  부분 포함      : {len(partial)}")

    for t in exact + normalized:
        print(f"    누수: {t!r}")
    for t, s in partial:
        print(f"    부분: {t!r} ↔ {s!r}")

    total = len(exact) + len(normalized) + len(partial)
    print("\n" + ("누수 없음" if total == 0 else f"누수 {total}건 — 평가셋 문장을 바꾸세요"))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
