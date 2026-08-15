"""fail-open 발생 집계.

게이트·각주·번역은 실패해도 500을 내지 않고 조용히 넘어간다(fail-open). 서비스가
죽지 않는다는 점에서는 의도한 설계지만, 밖에서 보면 "오해 소지가 없는 발화"와
"인프라가 죽어서 판정을 못 한 발화"가 똑같아 보인다.

실제로 pgvector 타입 불일치로 모든 벡터 검색이 예외였을 때도 API는 계속 200에
has_risk=false를 반환했고, 스모크 테스트를 돌려보고서야 발견했다.

그래서 어디서 몇 번 삼켰는지만 세어둔다. /health/deep에서 확인한다.
"""

from collections import Counter
from typing import Dict

# ponytail: 프로세스 로컬 카운터. 인스턴스가 여러 대면 각자 센다.
#           집계가 필요해지면 로그 수집이나 메트릭 익스포터로 옮기자.
_counts: Counter = Counter()


def record_fail_open(where: str) -> None:
    """fail-open 1회 기록. where는 'gate' / 'note' / 'translate' 등 지점 이름."""
    _counts[where] += 1


def snapshot() -> Dict[str, int]:
    return dict(_counts)


def reset() -> None:
    """테스트용."""
    _counts.clear()
