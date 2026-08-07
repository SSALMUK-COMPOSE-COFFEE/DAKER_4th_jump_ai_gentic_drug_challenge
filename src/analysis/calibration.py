"""감사 점수 캘리브레이션 — 파이프라인이 실제로 실패를 구분하는가.

NCT03601897 사례에서 false positive 를 확인했다. 목표 30명에 실제 177명(590%)을 등록한
트라이얼에 감사 점수 100/100 을 줬고, 그 트라이얼의 실제 중단 사유는 경영 판단(C분류)이었다.

따라서 점수를 신뢰하기 전에 전체 코호트에서 분포를 봐야 한다. 확인할 것:

1. 모집실패군(A) 점수가 완주군보다 높은가 — 안 높으면 파이프라인이 무의미하다
2. 경영 중단군(C) 점수가 완주군과 비슷한가 — C에 높은 점수를 주면 그게 false positive 다
3. 점수 임계값별 정밀도·재현율

주의: 참조 분포를 만든 데이터로 그 참조 분포를 평가하므로 **낙관적으로 편향된다**(in-sample).
따라서 여기서 잘 나와도 성능 주장을 할 수 없고, 여기서 나쁘면 확실히 못 쓴다는 의미만 갖는다.
제안서에 넣을 숫자는 holdout 분할 후 다시 측정해야 한다.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit import benchmarks as bm  # noqa: E402
from audit.pipeline import audit_v0  # noqa: E402
from labels.taxonomy import classify, enrollment_attainment  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")


def group_of(row: dict, *, a_priority: bool = False) -> str:
    """평가 집단 이름. `a_priority` 는 복합 사유에서 A 를 우선할지 결정한다 (taxonomy.classify 참조)."""
    status = row["labels"]["final_status"]
    if status == "COMPLETED":
        return "완주"
    cat = classify(row["labels"].get("why_stopped"), a_priority=a_priority)[0]
    if status == "WITHDRAWN":
        return "철회(개시전)"
    return {
        "A": "중단-모집실패",
        "B": "중단-과학적",
        "C": "중단-경영",
        "D": "중단-경쟁환경",
    }.get(cat, "중단-미분류")


def describe(name: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {name:<18} n=0")
        return
    vs = sorted(vals)
    print(
        f"  {name:<18} n={len(vs):<4} 중앙값={st.median(vs):>6.1f} "
        f"평균={st.fmean(vs):>6.1f} 25%={vs[len(vs)//4]:>6.1f} 75%={vs[3*len(vs)//4]:>6.1f}"
    )


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney U 기반 AUC. 동점은 0.5로 센다."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return round(wins / (len(pos) * len(neg)), 3)


def main() -> None:
    bench = bm.load()
    rows = [json.loads(l) for l in RAW.open() if l.strip()]

    scored: list[dict] = []
    for r in rows:
        # 네트워크를 쓰지 않는다 — 경쟁 밀도는 적응증 풀 조회가 필요해 배치에 부적합.
        # 즉 여기서 재는 것은 "피처 기반 감사만"의 구분력이다.
        rep = audit_v0(r["v0"], r["condition_bucket"], nct_id=r["nct_id"], bench=bench, http=None)
        scored.append(
            {
                "nct": r["nct_id"],
                "score": rep.risk_score(),
                "n_high": sum(1 for f in rep.findings if f.severity == "high"),
                "group": group_of(r),
                "att": enrollment_attainment(
                    r["v0"].get("planned_enrollment"), r["labels"].get("actual_enrollment")
                ),
            }
        )

    print(f"채점 대상 {len(scored)}건 (경쟁 밀도 제외, 피처 감사만)\n")
    print("[1] 집단별 감사 점수 분포")
    order = ["완주", "중단-모집실패", "중단-경쟁환경", "중단-과학적", "중단-경영", "중단-미분류", "철회(개시전)"]
    groups = {g: [s["score"] for s in scored if s["group"] == g] for g in order}
    for g in order:
        describe(g, groups[g])

    print("\n[2] 구분력 (AUC, 0.5 = 무작위)")
    comp = groups["완주"]
    for g in ("중단-모집실패", "중단-경쟁환경", "중단-과학적", "중단-경영"):
        a = auc(groups[g], comp)
        verdict = ""
        if a is not None:
            if g == "중단-모집실패":
                verdict = " ← 높아야 함 (표적)"
            elif g == "중단-경영":
                verdict = " ← 0.5 근처여야 함 (기권 대상)"
        print(f"  {g:<18} vs 완주 : AUC {a}{verdict}")

    print("\n[3] 임계값별 성능 — 양성=중단-모집실패, 음성=완주")
    pos, neg = groups["중단-모집실패"], comp
    print(f"  {'임계값':>6}{'재현율':>9}{'정밀도':>9}{'완주군 오탐률':>14}")
    for th in (25, 37, 50, 62, 75):
        tp = sum(1 for v in pos if v >= th)
        fp = sum(1 for v in neg if v >= th)
        rec = tp / len(pos) if pos else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        fpr = fp / len(neg) if neg else 0
        print(f"  {th:>6}{rec:>9.1%}{prec:>9.1%}{fpr:>14.1%}")

    print("\n[4] false positive 점검 — 감사 점수가 높은데 모집은 성공한 사례")
    fps = [s for s in scored if s["score"] >= 50 and s["att"] is not None and s["att"] >= 1.0]
    print(f"  점수 50 이상 & 등록 달성률 100% 이상: {len(fps)}건")
    for s in sorted(fps, key=lambda x: -x["score"])[:8]:
        print(f"    {s['nct']}  점수 {s['score']:>3}  달성률 {s['att']:>5.2f}  {s['group']}")

    print("\n[5] high finding 개수 분포 (점수 포화 여부)")
    print("  ", dict(sorted(Counter(s["n_high"] for s in scored).items())))


if __name__ == "__main__":
    main()
