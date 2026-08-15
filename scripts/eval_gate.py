"""게이트 오프라인 평가 — 키·DB 없이 게이트가 오해 발화를 거르는지 검증.

data/seed 코퍼스를 로컬 임베딩(fastembed)해 인메모리 검색하고,
data/eval/gate_testset.jsonl(라벨: misread/normal)에 게이트 로직을 적용해
정밀도/재현율/F1과 문턱 스윕을 출력한다. Supabase·LLM 불필요.

    python scripts/eval_gate.py

app/services/analyze/gate.py의 실제 판정식과 동일하게 맞춰야 의미가 있다:
    predict misread  ⇔  max_sim >= base - sensitivity * cultural_distance(source, listener)

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

    def score(base, sens):
        tp = fp = fn = tn = 0
        for top, neut, dist, is_mis, _ in rows:
            pred = (top >= base - sens * dist) and (top >= neut)  # gate.py와 동일
            tp += pred and is_mis
            fp += pred and not is_mis
            fn += (not pred) and is_mis
            tn += (not pred) and not is_mis
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1, (tp, fp, fn, tn)

    print("\n=== 현재 설정 ===")
    p, r, f1, cm = score(s.gate_base_threshold, s.gate_distance_sensitivity)
    print(f"base={s.gate_base_threshold} sens={s.gate_distance_sensitivity}"
          f" → P={p:.2f} R={r:.2f} F1={f1:.2f}  (TP{cm[0]} FP{cm[1]} FN{cm[2]} TN{cm[3]})")

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
    print(f"\n권장 base_threshold ≈ {best[0]:.2f} (F1={best[1]:.2f})")

    print("\n=== 오분류 (틀린 것만) ===")
    for top, neut, dist, is_mis, t in rows:
        pred = (top >= s.gate_base_threshold - s.gate_distance_sensitivity * dist) and (top >= neut)
        if pred != is_mis:
            kind = "FP(정상→오해)" if pred else "FN(오해→놓침)"
            print(f"  [{kind}] risk={top:.2f} neut={neut:.2f} {t['source']}→{t['listener']}: {t['text'][:26]}")


if __name__ == "__main__":
    main()
