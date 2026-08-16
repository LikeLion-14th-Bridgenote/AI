"""게이트 오프라인 평가 — 키·DB 없이 게이트가 오해 발화를 거르는지 검증.

data/seed 코퍼스를 로컬 임베딩(fastembed)해 인메모리 검색하고,
data/eval/gate_testset.jsonl(라벨: misread/normal)에 게이트 로직을 적용해
정밀도/재현율/F1과 문턱 스윕을 출력한다. Supabase·LLM 불필요.

    python scripts/eval_gate.py

app/services/analyze/gate.py의 실제 판정식과 동일하게 맞춰야 의미가 있다:
    predict misread  ⇔  risk_sim >= base - sensitivity * cultural_distance(source, listener)
                    and  risk_sim >= neutral_sim - gate_neutral_margin

주의: 평가셋 문장이 시드 코퍼스에 들어가면 유사도가 1.0이라 무조건 맞힌다.
시드나 평가셋을 건드린 뒤에는 scripts/check_eval_leakage.py로 누수를 확인할 것.
"""

import json
import math
from pathlib import Path

from app.core.config import get_settings
from app.core.embeddings import embed_texts
from app.rag.culture_distance import cultural_distance

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed" / "culture_corpus_seed.jsonl"
TESTSET = ROOT / "data" / "eval" / "gate_testset.jsonl"


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.hypot(*a) * math.hypot(*b) + 1e-9)


def main() -> None:
    s = get_settings()
    rules = [json.loads(l) for l in SEED.read_text(encoding="utf-8").splitlines() if l.strip()]
    tests = [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"코퍼스 {len(rules)}개 + 테스트 {len(tests)}개 임베딩 중...")
    rule_vecs = embed_texts([r["rule_text"] for r in rules])
    test_vecs = embed_texts([t["text"] for t in tests])
    by_culture = {}  # (culture, entry_type) → [vec]
    for r, v in zip(rules, rule_vecs):
        by_culture.setdefault((r["culture"], r["entry_type"]), []).append(v)

    # 각 발화의 (화자 risk_seed 최대 유사도, 문화거리, 정답) 미리 계산
    # gate.py와 동일: 화자 risk_seed 최대 유사도 + neutral_seed 최대 유사도.
    rows = []
    for t, tv in zip(tests, test_vecs):
        risk = max((_cos(tv, rv) for rv in by_culture.get((t["source"], "risk_seed"), [])), default=0.0)
        neut = max((_cos(tv, rv) for rv in by_culture.get((t["source"], "neutral_seed"), [])), default=0.0)
        dist = cultural_distance(t["source"], t["listener"])
        rows.append((risk, neut, dist, t["label"] == "misread", t))

    def predict(top, neut, dist, base, sens):
        # gate.py와 동일: 문턱 통과 AND 정상표현에 "뚜렷하게" 밀리지 않을 것(마진)
        return (top >= base - sens * dist) and (top >= neut - s.gate_neutral_margin)

    def score(base, sens, subset=None):
        tp = fp = fn = tn = 0
        for top, neut, dist, is_mis, _ in subset if subset is not None else rows:
            pred = predict(top, neut, dist, base, sens)
            tp += pred and is_mis
            fp += pred and not is_mis
            fn += (not pred) and is_mis
            tn += (not pred) and not is_mis
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1, (tp, fp, fn, tn)

    def report(label, subset):
        p, r, f1, cm = score(s.gate_base_threshold, s.gate_distance_sensitivity, subset)
        print(f"  {label:<22} n={len(subset):<3} P={p:.2f} R={r:.2f} F1={f1:.3f}"
              f"  (TP{cm[0]} FP{cm[1]} FN{cm[2]} TN{cm[3]})")

    print("\n=== 현재 설정 ===")
    print(f"base={s.gate_base_threshold} sens={s.gate_distance_sensitivity}"
          f" margin={s.gate_neutral_margin}")
    report("전체", rows)

    # origin별 분리 측정 — authored(합성)와 observed(실제 사례)의 성능 차이가
    # "합성 데이터에만 맞는 모델"인지 판별하는 근거다.
    origins = sorted({r[4].get("origin", "authored") for r in rows})
    if len(origins) > 1:
        print("\n=== origin별 ===")
        for o in origins:
            report(o, [r for r in rows if r[4].get("origin", "authored") == o])

    print("\n=== 문턱 스윕 (F1 최고점 찾기) ===")
    best = None
    for base_i in range(30, 75, 5):
        base = base_i / 100
        _, _, f1s, _ = score(base, s.gate_distance_sensitivity)
        mark = ""
        if best is None or f1s > best[1]:
            best = (base, f1s)
            mark = " ←"
        print(f"  base={base:.2f}: F1={f1s:.2f}{mark}")
    # ponytail: '≈'는 윈도우 기본 콘솔(cp949)에서 UnicodeEncodeError로 스크립트를 죽인다
    print(f"\n권장 base_threshold ~ {best[0]:.2f} (F1={best[1]:.2f})")

    print("\n=== 오분류 (틀린 것만) ===")
    for top, neut, dist, is_mis, t in rows:
        pred = predict(top, neut, dist, s.gate_base_threshold, s.gate_distance_sensitivity)
        if pred != is_mis:
            kind = "FP(정상→오해)" if pred else "FN(오해→놓침)"
            print(f"  [{kind}] risk={top:.2f} neut={neut:.2f} {t.get('origin', '?')}"
                  f" {t['source']}→{t['listener']}: {t['text'][:30]}")


if __name__ == "__main__":
    main()
