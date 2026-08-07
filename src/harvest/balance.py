"""적응증별 코호트 균형 맞추기 — 적응증 교란 제거.

`expand.py` 가 중단군만 수집해서 신규 10개 적응증의 완주군이 0건이 됐다.
그러면 그 적응증들은 참조 분포가 없어 `__ALL__` 로 폴백하고, 그 완주군은
기존 5개 적응증(아토피·당뇨·류마티스 등)으로 채워져 있다.

결과적으로 **모집이 원래 어려운 종양 적응증을 모집이 쉬운 적응증의 완주군과 비교**하게 된다.
이 상태의 AUC 상승은 프로토콜 품질을 잡은 것이 아니라 적응증을 맞힌 것이다.
적응증별로 두 코호트가 모두 있어야 층내 비교가 성립한다.

`ctgov.py` 하베스터가 기존 jsonl 의 nct_id 로 중복을 건너뛰므로 그대로 append 된다.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.benchmarks import cohort_of  # noqa: E402
from harvest.ctgov import harvest  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")

# 적응증별 완주군 목표치. 모집실패군과 같은 자릿수는 되어야 층내 분위수가 의미를 갖는다.
TARGET_COMPLETED = 40


def missing_completed() -> list[str]:
    counts: Counter[str] = Counter()
    buckets: set[str] = set()
    with RAW.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            buckets.add(r["condition_bucket"])
            if cohort_of(r) == "completed":
                counts[r["condition_bucket"]] += 1
    need = [b for b in sorted(buckets) if counts[b] < TARGET_COMPLETED]
    for b in sorted(buckets):
        print(f"  {b:<34} 완주군 {counts[b]:>3}건" + ("  ← 보충 필요" if b in need else ""))
    return need


if __name__ == "__main__":
    print("적응증별 완주군 현황:")
    need = missing_completed()
    if not need:
        print("\n보충할 적응증이 없습니다.")
        raise SystemExit(0)
    print(f"\n{len(need)}개 적응증에서 COMPLETED 수집 시작 (목표 각 {TARGET_COMPLETED}건)\n")
    harvest(
        buckets=need,
        statuses=["COMPLETED"],
        per_cell=TARGET_COMPLETED,
        out_path=RAW,
    )
