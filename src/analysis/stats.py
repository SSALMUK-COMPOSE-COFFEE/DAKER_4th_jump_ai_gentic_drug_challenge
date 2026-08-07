"""수집 데이터의 기술통계 — 제안서에 넣을 대표 숫자를 뽑는다.

확인하려는 것:
1. 등록 달성률이 중단군과 완료군에서 실제로 갈리는가 (갈리지 않으면 이 라벨은 쓸 수 없다)
2. 중단 사유 A/B/C 분포 — A(교정 가능)가 의미 있는 비중인가
3. 적격기준 개정 횟수가 달성률과 상관이 있는가
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels.taxonomy import CATEGORY_NAMES, classify, enrollment_attainment  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")


def load() -> list[dict]:
    with RAW.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def describe(name: str, values: list[float]) -> None:
    if not values:
        print(f"  {name:<22} n=0")
        return
    vs = sorted(values)
    print(
        f"  {name:<22} n={len(vs):<4} 중앙값={st.median(vs):<7.2f} "
        f"평균={st.fmean(vs):<7.2f} 25%={vs[len(vs)//4]:<7.2f} 75%={vs[3*len(vs)//4]:<7.2f}"
    )


def main() -> None:
    rows = load()
    print(f"총 {len(rows)}건\n")

    by_status: Counter[str] = Counter(r["labels"]["final_status"] for r in rows)
    print("최종 상태 분포:", dict(by_status), "\n")

    # --- 1. 등록 달성률 ---
    print("[1] 등록 달성률 (실제/목표) — 상태별")
    groups: dict[str, list[float]] = {}
    for r in rows:
        att = enrollment_attainment(
            r["v0"].get("planned_enrollment"), r["labels"].get("actual_enrollment")
        )
        if att is None:
            continue
        groups.setdefault(r["labels"]["final_status"], []).append(att)
    for status, vals in sorted(groups.items()):
        describe(status, vals)

    term = groups.get("TERMINATED", [])
    comp = groups.get("COMPLETED", [])
    if term and comp:
        under_t = sum(1 for v in term if v < 0.8) / len(term)
        under_c = sum(1 for v in comp if v < 0.8) / len(comp)
        print(f"\n  목표 80% 미달 비율:  중단군 {under_t:.1%}  vs  완료군 {under_c:.1%}")
        print(f"  → 분리도 {under_t - under_c:+.1%}p (클수록 이 라벨이 유효)")

    # --- 2. 중단 사유 3분류 ---
    print("\n[2] 중단 사유 4분류 (TERMINATED/WITHDRAWN)")
    cats: Counter[str] = Counter()
    reasons: Counter[tuple[str, str]] = Counter()
    unmatched: list[str] = []
    stopped = [r for r in rows if r["labels"]["final_status"] in ("TERMINATED", "WITHDRAWN")]
    for r in stopped:
        cat, reason, _ = classify(r["labels"].get("why_stopped"))
        cats[cat] += 1
        reasons[(cat, reason)] += 1
        if cat == "?" and r["labels"].get("why_stopped"):
            unmatched.append(r["labels"]["why_stopped"])
    total = sum(cats.values()) or 1
    for cat in ("A", "B", "C", "D", "?"):
        n = cats.get(cat, 0)
        print(f"  {cat} {CATEGORY_NAMES[cat]:<10} {n:>4}건 ({n/total:5.1%})")
    print("\n  세부 사유 상위:")
    for (cat, reason), n in reasons.most_common(10):
        print(f"    {cat} {reason:<24} {n}건")
    if unmatched:
        print(f"\n  미분류 {len(unmatched)}건 (LLM 2차 패스 대상) 예시:")
        for u in unmatched[:5]:
            print(f"    - {u[:90]}")

    # --- 3. 적격기준 개정 vs 달성률 ---
    print("\n[3] 적격기준 개정 횟수별 등록 달성률")
    buckets: dict[str, list[float]] = {}
    for r in rows:
        att = enrollment_attainment(
            r["v0"].get("planned_enrollment"), r["labels"].get("actual_enrollment")
        )
        if att is None:
            continue
        n = r["labels"].get("amend_eligibility", 0)
        key = "0회" if n == 0 else "1회" if n == 1 else "2회+"
        buckets.setdefault(key, []).append(att)
    for key in ("0회", "1회", "2회+"):
        describe(key, buckets.get(key, []))

    # --- 4. 블라인드 입력 완결성 점검 ---
    print("\n[4] v0 입력 완결성 (결측이면 피처로 쓸 수 없다)")
    n = len(rows)
    for field in (
        "planned_enrollment",
        "planned_primary_completion",
        "eligibility_criteria",
        "n_sites",
        "primary_outcomes",
    ):
        missing = sum(1 for r in rows if not r["v0"].get(field))
        print(f"  {field:<28} 결측 {missing:>4}/{n} ({missing/n:5.1%})")
    est = sum(1 for r in rows if r["v0"].get("planned_enrollment_type") == "ESTIMATED")
    print(f"  planned_enrollment_type==ESTIMATED  {est}/{n} ({est/n:.1%}) ← 결과 정보 미포함 확인")


if __name__ == "__main__":
    main()
