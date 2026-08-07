"""적격기준 개정 횟수와 등록 달성률의 관계 — 생존 편향 검증.

최초 가설: 적격기준을 여러 번 고친 트라이얼은 설계가 처음부터 잘못됐으므로 달성률이 낮을 것이다.
실측 결과: 정반대였다 (개정 0회 중앙값 0.16 < 1회 0.49 < 2회+ 0.74).

의심되는 원인은 생존 편향이다. 일찍 죽은 트라이얼은 개정할 기회 자체가 없었다.
개정 횟수는 "설계 품질"이 아니라 "살아있던 기간"을 재고 있을 가능성이 크다.

이 스크립트는 그 의심을 검증한다. 검증되면 개정 횟수는 정답 라벨에서 빼고
공변량으로만 쓴다 — 라벨로 쓰면 인과를 거꾸로 학습한다.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels.taxonomy import enrollment_attainment  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")


def parse_date(d: str | None) -> date | None:
    if not d:
        return None
    parts = d.split("-")
    try:
        return date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1,
                    int(parts[2]) if len(parts) > 2 else 1)
    except (ValueError, IndexError):
        return None


def months_alive(row: dict) -> float | None:
    """개시일부터 실제 1차완료일까지의 개월 수 = 개정할 기회가 있던 기간."""
    start = parse_date(row["v0"].get("start_date"))
    end = parse_date(row["labels"].get("actual_primary_completion"))
    if not start or not end:
        return None
    days = (end - start).days
    return round(days / 30.44, 1) if days > 0 else None


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """순위 상관. scipy 없이 계산한다 (동순위는 평균 순위로 처리)."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def describe(name: str, vals: list[float]) -> None:
    if not vals:
        print(f"    {name:<16} n=0")
        return
    vs = sorted(vals)
    print(f"    {name:<16} n={len(vs):<4} 중앙값={st.median(vs):.2f}")


def main() -> None:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]

    recs = []
    for r in rows:
        att = enrollment_attainment(
            r["v0"].get("planned_enrollment"), r["labels"].get("actual_enrollment")
        )
        alive = months_alive(r)
        if att is None:
            continue
        recs.append(
            {
                "att": att,
                "amend": r["labels"].get("amend_eligibility", 0),
                "amend_all": r["labels"].get("n_versions", 0),
                "alive": alive,
                "status": r["labels"]["final_status"],
            }
        )

    print(f"분석 대상 {len(recs)}건 (달성률 계산 가능)\n")

    # --- 1. 개정 횟수가 생존 기간을 재고 있는가 ---
    with_alive = [r for r in recs if r["alive"]]
    print("[1] 개정 횟수 vs 생존 기간 (개월)")
    print(f"  Spearman(적격기준 개정, 생존기간) = "
          f"{spearman([r['amend'] for r in with_alive], [r['alive'] for r in with_alive])}")
    print(f"  Spearman(전체 버전 수,   생존기간) = "
          f"{spearman([r['amend_all'] for r in with_alive], [r['alive'] for r in with_alive])}")
    print("  → 강한 양의 상관이면 개정 횟수는 설계 품질이 아니라 생존 기간의 대리변수다.\n")

    # --- 2. 생존 기간을 고정하면 관계가 사라지는가 ---
    print("[2] 생존 기간 층화 후 개정 횟수별 달성률")
    strata = [("~12개월", 0, 12), ("12-24개월", 12, 24), ("24개월+", 24, 10**6)]
    for label, lo, hi in strata:
        band = [r for r in with_alive if lo < r["alive"] <= hi]
        if len(band) < 6:
            print(f"  {label}: n={len(band)} (표본 부족)")
            continue
        print(f"  {label} (n={len(band)})")
        describe("개정 0회", [r["att"] for r in band if r["amend"] == 0])
        describe("개정 1회+", [r["att"] for r in band if r["amend"] >= 1])
        sub = [r for r in band if r["att"] is not None]
        print(f"    층내 Spearman(개정, 달성률) = "
              f"{spearman([r['amend'] for r in sub], [r['att'] for r in sub])}")

    # --- 3. 완료군만 봤을 때 (생존 편향 제거) ---
    print("\n[3] COMPLETED 만 (모두 끝까지 살아남음 → 생존 편향 제거)")
    comp = [r for r in recs if r["status"] == "COMPLETED"]
    describe("개정 0회", [r["att"] for r in comp if r["amend"] == 0])
    describe("개정 1회", [r["att"] for r in comp if r["amend"] == 1])
    describe("개정 2회+", [r["att"] for r in comp if r["amend"] >= 2])
    print(f"  Spearman(개정, 달성률 | COMPLETED) = "
          f"{spearman([r['amend'] for r in comp], [r['att'] for r in comp])}")


if __name__ == "__main__":
    main()
