"""평가셋이 시드 코퍼스와 겹치는지 검사.

게이트는 시드와의 유사도로 판정하므로, 평가셋 문장이 시드에 들어 있으면 유사도가
1.0이 나와 무조건 맞힌다. 그만큼 F1이 부풀려진다. 실제로 처음 24건 중 7건이 이렇게
새고 있었고, 대소문자·쉼표만 다른 2건은 눈으로 못 잡아 한 번 더 놓쳤다.

    python scripts/check_eval_leakage.py           # 문자열 검사 (빠름)
    python scripts/check_eval_leakage.py --embed   # + 임베딩 유사도 (모델 로드, 느림)

문자열 검사만으로는 "생각 좀 해보겠습니다" ↔ "고민해보겠습니다"(유사도 0.995) 같은
근접 중복을 못 잡는다. 게이트는 문자열이 아니라 임베딩으로 판정하므로, 시드를
추가한 뒤에는 --embed로 한 번 더 보는 게 안전하다.

종료코드: 문자열 누수가 있으면 1. --embed의 근접 중복은 경고만 하고 0
(대소관계를 뒤집지 않으면 F1에 영향이 없는 경우가 많아 판단이 필요하다).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "culture_corpus_seed.jsonl"
TESTSET = ROOT / "data" / "eval" / "gate_testset.jsonl"

NEAR_THRESHOLD = 0.95  # 이 이상이면 사실상 같은 문장으로 본다

# 공백·문장부호·대소문자 차이는 임베딩상 거의 같은 문장이므로 정규화해서 비교한다.
_STRIP = re.compile(r"""[\s.,!?…·\-—'"()]""")


def _norm(s: str) -> str:
    return _STRIP.sub("", s.lower())


def _load(path: Path, key: str):
    return [json.loads(line)[key] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_strings(seeds, tests) -> int:
    seed_raw = {s.strip() for s in seeds}
    seed_norm = {_norm(s) for s in seeds}

    exact = [t for t in tests if t.strip() in seed_raw]
    normalized = [t for t in tests if _norm(t) in seed_norm and t.strip() not in seed_raw]
    partial = [
        (t, s) for t in tests for s in seed_raw
        if len(s) > 5 and t.strip() != s and (s in t or t.strip() in s)
    ]

    print(f"  완전 일치      : {len(exact)}")
    print(f"  정규화 후 일치 : {len(normalized)}   (대소문자·문장부호만 다른 것)")
    print(f"  부분 포함      : {len(partial)}")

    for t in exact + normalized:
        print(f"    누수: {t!r}")
    for t, s in partial:
        print(f"    부분: {t!r} ↔ {s!r}")

    return len(exact) + len(normalized) + len(partial)


def check_embeddings(seeds, tests) -> int:
    """임베딩 유사도로 근접 중복을 찾는다. 모델 로드가 있어 느리다."""
    import math

    from app.core.embeddings import embed_texts

    print(f"\n임베딩 유사도 검사 (임계 {NEAR_THRESHOLD})")
    sv = embed_texts(seeds)
    tv = embed_texts(tests)

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b) + 1e-9)

    hits = []
    for t, v in zip(tests, tv):
        sim, src = max((cos(v, x), s) for s, x in zip(seeds, sv))
        if sim >= NEAR_THRESHOLD:
            hits.append((sim, t, src))

    print(f"  근접 중복      : {len(hits)} / {len(tests)}")
    for sim, t, src in sorted(hits, reverse=True):
        print(f"    {sim:.3f} {t[:36]!r} ↔ {src[:36]!r}")

    if hits:
        print("\n  근접 중복은 F1에 영향이 없을 수도 있다 —")
        print("  risk·neutral 양쪽 유사도를 함께 올리면 대소관계가 안 바뀌기 때문이다.")
        print("  영향을 보려면 해당 항목을 빼고 eval_gate.py를 돌려 비교할 것.")
    return len(hits)


def main() -> int:
    use_embed = "--embed" in sys.argv
    seeds = _load(SEED, "rule_text")
    tests = _load(TESTSET, "text")

    print(f"시드 {len(seeds)}건 / 평가셋 {len(tests)}건")
    total = check_strings(seeds, tests)

    if use_embed:
        check_embeddings(seeds, tests)
    else:
        print("\n  (임베딩 검사는 --embed 로 실행)")

    print("\n" + ("문자열 누수 없음" if total == 0 else f"문자열 누수 {total}건 — 평가셋 문장을 바꾸세요"))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
