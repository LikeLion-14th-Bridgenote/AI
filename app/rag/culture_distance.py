"""문화 거리 — Hofstede 6차원 유클리드 거리(0~1 정규화).

게이트에서 "문화가 멀수록 오해 소지에 민감(문턱 낮춤)"하게 쓰는 가중치.
예: 한↔미(먼 조합)는 문턱을 낮춰 더 잘 잡고, 한↔베(가까운 조합)는 완화.
값은 app/rag/country_scores.json (Hofstede Insights 기반)에서 로드.
"""

import json
from functools import lru_cache
from pathlib import Path

_DIMS = ("PDI", "IDV", "MAS", "UAI", "LTO", "IVR")
_MAX = len(_DIMS) ** 0.5  # 각 차원을 [0,1]로 정규화했을 때의 최대 유클리드 거리


@lru_cache(maxsize=1)
def _scores() -> dict:
    path = Path(__file__).parent / "country_scores.json"
    return json.loads(path.read_text(encoding="utf-8"))


def cultural_distance(a: str, b: str) -> float:
    """문화권 a·b 사이 거리를 0(동일)~1(최대)로 반환. 미지 조합은 중립값 0.5."""
    if a == b:
        return 0.0
    scores = _scores()
    if a not in scores or b not in scores:
        return 0.5
    ha, hb = scores[a]["hofstede"], scores[b]["hofstede"]
    dist = sum(((ha[k] - hb[k]) / 100.0) ** 2 for k in _DIMS) ** 0.5
    return min(dist / _MAX, 1.0)
