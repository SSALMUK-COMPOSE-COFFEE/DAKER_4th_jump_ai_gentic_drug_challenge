"""일정 지연 라벨 검증.

일정 지연 = 실제 1차완료일 − v0 예상 1차완료일 (개월).

이 라벨이 쓸 만한지 확인할 것:
1. 중단군과 완료군이 갈리는가
2. 등록 달성률과 독립적인 정보를 담고 있는가 (상관이 너무 높으면 중복 라벨)
3. 계획 기간 자체가 비현실적이었는지 볼 수 있는가 (계획 기간 대비 지연 비율)
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.amendment_confound import parse_date, spearman  # noqa: E402
from labels.taxonomy import classify, enrollment_attainment  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")


def months(a, b) -> float | None:
    if not a or not b:
        return None
    return round((b - a).days / 30.44, 1)


def describe(name: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {name:<24} n=0")
        return
    vs = sorted(vals)
    print(
        f"  {name:<24} n={len(vs):<4} 중앙값={st.median(vs):>7.1f} "
        f"25%={vs[len(vs)//4]:>7.1f} 75%={vs[3*len(vs)//4]:>7.1f}"
    )


def main() -> None:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    recs = []
    for r in rows:
        start = parse_date(r["v0"].get("start_date"))
        planned_end = parse_date(r["v0"].get("planned_primary_completion"))
        actual_end = parse_date(r["labels"].get("actual_primary_completion"))
        planned_dur = months(start, planned_end)
        slip = months(planned_end, actual_end)
        if slip is None or planned_dur is None or planned_dur <= 0:
            continue
        recs.append(
            {
                "slip": slip,
                "planned_dur": planned_dur,
                "slip_ratio": round(slip / planned_dur, 3),
                "att": enrollment_attainment(
                    r["v0"].get("planned_enrollment"), r["labels"].get("actual_enrollment")
                ),
                "status": r["labels"]["final_status"],
                "cat": classify(r["labels"].get("why_stopped"))[0],
            }
        )

    print(f"분석 대상 {len(recs)}건 (일정 계산 가능)\n")

    print("[1] 일정 지연(개월) — 상태별")
    for status in ("COMPLETED", "TERMINATED", "WITHDRAWN"):
        describe(status, [r["slip"] for r in recs if r["status"] == status])

    print("\n[2] 계획 기간 대비 지연 비율 — 상태별")
    print("    (1.0 = 계획한 기간만큼 더 걸렸다는 뜻)")
    for status in ("COMPLETED", "TERMINATED", "WITHDRAWN"):
        describe(status, [r["slip_ratio"] for r in recs if r["status"] == status])

    print("\n[3] v0 계획 기간 자체 — 상태별 (계획이 애초에 짧았는가)")
    for status in ("COMPLETED", "TERMINATED", "WITHDRAWN"):
        describe(status, [r["planned_dur"] for r in recs if r["status"] == status])

    print("\n[4] 등록 달성률과의 중복성 점검")
    both = [r for r in recs if r["att"] is not None]
    print(f"  Spearman(지연, 달성률)       = {spearman([r['slip'] for r in both], [r['att'] for r in both])}")
    print(f"  Spearman(지연비율, 달성률)    = {spearman([r['slip_ratio'] for r in both], [r['att'] for r in both])}")
    print("  → 절댓값이 낮으면 서로 독립적인 정보 = 라벨 두 개를 같이 쓸 가치가 있다.")

    print("\n[5] 중단 사유 분류별 지연 비율 (A=교정가능, C=외생)")
    for cat in ("A", "B", "C", "D"):
        describe(f"{cat}분류", [r["slip_ratio"] for r in recs if r["cat"] == cat])


if __name__ == "__main__":
    main()
