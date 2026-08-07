"""전례 공급량 실측 — 시간 컷오프를 걸면 인용할 전례가 실제로 남는가.

## 왜 이 측정이 필요했나

Tier 2 레드팀은 모든 공격에 전례 NCT ID 를 인용해야 한다 (`agent-architecture.md` §4).
그런데 2012년 개시 프로토콜을 심사하면서 2018년 개시 시험을 전례로 들면, 그 시점에
존재하지 않던 정보를 쓰는 것이다. Tier 1 은 LLM 을 안 쓰므로 v0 입력만으로 블라인드가
성립하지만, **Tier 2 의 전례 검색은 코퍼스 전체를 뒤지므로 여기서 미래 정보가 새어든다.**

따라서 `precedent.start_date < target.start_date` 를 하드 필터로 강제한다.
문제는 그러면 전례가 남지 않을 수 있다는 것이다. 코퍼스가 2006–2020 개시라
초기 연도 대상은 인용할 과거 시험이 코퍼스 안에 거의 없다.

**이 스크립트는 그 공급량을 측정해서, 전례 기반 공격을 몇 %의 대상에 대해 포기해야 하는지
숫자로 만든다.** 포기해야 한다면 A1 Planner 가 사전에 기각하고 그 사실을 리포트에 노출한다.

## 층 정의

검색 키는 `agent-architecture.md` §3 의 2순위(적응증 + phase_stratum)를 기준으로 한다.
1순위(개입 유형까지)는 더 좁아지므로 여기 숫자가 상한이다.

실행: python3 src/analysis/precedent_supply.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.amendment_confound import parse_date  # noqa: E402
from audit.benchmarks import cohort_of, phase_stratum  # noqa: E402
from labels.taxonomy import classify  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")

# A1 Planner 가 전례 기반 공격을 기각하는 임계값 (agent-architecture.md §2)
MIN_PRECEDENTS = 5


def load() -> list[dict]:
    rows = []
    for line in RAW.open():
        if not line.strip():
            continue
        r = json.loads(line)
        start = parse_date(r["v0"].get("start_date"))
        if not start:
            continue
        rows.append(
            {
                "nct": r["nct_id"],
                "start": start,
                "stratum": f"{r['condition_bucket']}|{phase_stratum(r['v0'])}",
                "ind": r["condition_bucket"],
                "cohort": cohort_of(r),
                "cls": classify(r["labels"].get("why_stopped"))[0]
                if r["labels"]["final_status"] in ("TERMINATED", "WITHDRAWN")
                else None,
            }
        )
    return rows


def supply(rows: list[dict], key: str) -> list[dict]:
    """대상마다 자기 개시일 이전에 개시된 같은 층 전례 수를 센다."""
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r[key], []).append(r)
    for pool in by_key.values():
        pool.sort(key=lambda x: x["start"])

    out = []
    for r in rows:
        pool = by_key[r[key]]
        # 자기보다 먼저 개시된 것만. 동일 개시일은 제외한다 (그 시점에 결과가 없다).
        prior = [p for p in pool if p["start"] < r["start"] and p["nct"] != r["nct"]]
        out.append(
            {
                **r,
                "n_prior": len(prior),
                # 실패군 전례가 핵심 — whyStopped 를 인용할 수 있는 것은 중단군뿐이다
                "n_prior_failed_A": sum(1 for p in prior if p["cls"] == "A"),
                "n_prior_completed": sum(1 for p in prior if p["cohort"] == "completed"),
            }
        )
    return out


def report(rows: list[dict], key: str, label: str) -> None:
    data = supply(rows, key)
    targets = [d for d in data if d["cohort"]]  # 감사 대상이 되는 코호트만

    print(f"\n{'='*82}\n{label}  (층 키: {key}, 감사 대상 {len(targets)}건)\n{'='*82}")

    print(f"\n  {'개시연도':<10}{'대상수':>7}{'전례 중앙값':>13}{'A실패전례 중앙값':>18}"
          f"{'전례<5 비율':>13}{'A실패전례=0':>13}")
    by_year: dict[int, list[dict]] = {}
    for d in targets:
        by_year.setdefault(d["start"].year, []).append(d)

    for year in sorted(by_year):
        g = by_year[year]
        med = st.median([d["n_prior"] for d in g])
        med_a = st.median([d["n_prior_failed_A"] for d in g])
        thin = sum(1 for d in g if d["n_prior"] < MIN_PRECEDENTS) / len(g)
        no_a = sum(1 for d in g if d["n_prior_failed_A"] == 0) / len(g)
        print(f"  {year:<10}{len(g):>7}{med:>13.0f}{med_a:>18.0f}{thin:>12.0%}{no_a:>13.0%}")

    thin_all = sum(1 for d in targets if d["n_prior"] < MIN_PRECEDENTS)
    no_a_all = sum(1 for d in targets if d["n_prior_failed_A"] == 0)
    print(f"\n  전체: 전례 {MIN_PRECEDENTS}건 미만 = {thin_all}/{len(targets)} ({thin_all/len(targets):.1%})"
          f"  |  A분류 실패 전례 0건 = {no_a_all}/{len(targets)} ({no_a_all/len(targets):.1%})")

    # 코호트별로 갈리는지 — 한쪽 코호트만 전례가 부족하면 평가가 편향된다
    print(f"\n  {'코호트':<16}{'n':>6}{'전례 중앙값':>13}{'전례<5':>10}{'A실패전례 중앙값':>18}")
    for c in ("completed", "recruit_failed"):
        g = [d for d in targets if d["cohort"] == c]
        if not g:
            continue
        print(f"  {c:<16}{len(g):>6}{st.median([d['n_prior'] for d in g]):>13.0f}"
              f"{sum(1 for d in g if d['n_prior'] < MIN_PRECEDENTS)/len(g):>9.0%}"
              f"{st.median([d['n_prior_failed_A'] for d in g]):>18.0f}")


def main() -> None:
    rows = load()
    print(f"{RAW} — 개시일 파싱 성공 {len(rows)}건")
    print(f"개시연도 범위: {min(r['start'] for r in rows).year}–{max(r['start'] for r in rows).year}")
    print(f"층 수: 적응증+Phase {len({r['stratum'] for r in rows})}개 / 적응증 {len({r['ind'] for r in rows})}개")

    report(rows, "stratum", "2순위 검색 — 적응증 + Phase 계층")
    report(rows, "ind", "3순위 폴백 — 적응증만")

    print(f"\n{'='*82}\n층별 총 표본 (전례 공급의 상한)\n{'='*82}")
    cnt = Counter(r["stratum"] for r in rows)
    for k, n in cnt.most_common():
        print(f"  {k:<48}{n:>5}")


if __name__ == "__main__":
    main()
