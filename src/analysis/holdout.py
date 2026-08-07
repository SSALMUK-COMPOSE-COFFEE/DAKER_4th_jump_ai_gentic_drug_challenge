"""Holdout 평가 — 제안서에 넣을 수 있는 유일한 숫자.

`calibration.py` 는 참조 분포를 만든 데이터로 그 분포를 평가하므로 낙관 편향된다(in-sample).
여기서는 train 으로만 참조 분포를 만들고 test 에서만 채점한다.

분할은 NCT ID 해시 기반이라 실행마다 동일하다 — 재현 가능성이 심사 대상이므로 난수를 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from pathlib import Path

# 부트스트랩 재표본 수. 난수 시드를 고정해 재현 가능하게 한다.
N_BOOT = 2000
BOOT_SEED = 20260730

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import auc, group_of  # noqa: E402
from audit import benchmarks as bm  # noqa: E402
from audit.pipeline import audit_v0  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
TEST_FRACTION = 0.30


def in_test(nct_id: str) -> bool:
    """ID 해시로 결정론적 분할. 같은 트라이얼은 항상 같은 쪽에 간다."""
    h = hashlib.sha256(nct_id.encode()).hexdigest()
    return (int(h[:8], 16) % 1000) / 1000.0 < TEST_FRACTION


def build_benchmarks(rows: list[dict]) -> dict:
    """benchmarks.build 와 동일한 로직을 주어진 행 집합에만 적용한다."""
    acc: dict = {}
    for r in rows:
        cohort = bm.cohort_of(r)
        if not cohort:
            continue
        feats = bm.extract(r)  # r 에 nct_id 포함 → 경쟁 피처 조인됨
        for bucket in (r["condition_bucket"], "__ALL__"):
            per = acc.setdefault(bucket, {}).setdefault(cohort, {})
            for f in bm.FEATURES:
                per.setdefault(f, []).append(feats[f])
    return {
        "_meta": {"note": "holdout train split"},
        "buckets": {
            b: {c: {f: bm.quantiles(v) for f, v in feats.items()} for c, feats in co.items()}
            for b, co in acc.items()
        },
    }


def score_all(rows: list[dict], bench: dict) -> list[dict]:
    out = []
    for r in rows:
        rep = audit_v0(r["v0"], r["condition_bucket"], nct_id=r["nct_id"], bench=bench, http=None)
        out.append({"score": rep.risk_score(), "group": group_of(r)})
    return out


def auc_ci(pos: list[float], neg: list[float]) -> str:
    """부트스트랩 95% 신뢰구간.

    양성 표본이 적으면 AUC 점추정치가 거의 무의미하므로 구간을 반드시 함께 보고한다.
    구간이 0.5 를 포함하면 "구분력이 있다"고 말할 수 없다.
    """
    import random

    if not pos or not neg:
        return "n/a"
    point = auc(pos, neg)
    rng = random.Random(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        p = [rng.choice(pos) for _ in pos]
        n = [rng.choice(neg) for _ in neg]
        a = auc(p, n)
        if a is not None:
            boots.append(a)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    flag = "" if lo > 0.5 else "  ⚠ 0.5 포함 → 구분력 주장 불가"
    return f"{point}  95% CI [{lo:.3f}, {hi:.3f}]{flag}"


def report(name: str, scored: list[dict]) -> None:
    groups: dict[str, list[float]] = {}
    for s in scored:
        groups.setdefault(s["group"], []).append(s["score"])
    comp = groups.get("완주", [])
    fail = groups.get("중단-모집실패", [])
    sci = groups.get("중단-과학적", [])
    biz = groups.get("중단-경영", [])

    print(f"[{name}] n={len(scored)}  완주 {len(comp)} / 모집실패 {len(fail)}")
    if comp and fail:
        print(f"  중앙값       완주 {st.median(comp):.1f}  vs  모집실패 {st.median(fail):.1f}")
        print(f"  AUC 모집실패 vs 완주  = {auc_ci(fail, comp)}")
    if comp and sci:
        print(f"  AUC 과학적   vs 완주  = {auc_ci(sci, comp)}")
    if comp and biz:
        print(f"  AUC 경영     vs 완주  = {auc_ci(biz, comp)}")
    if comp and fail:
        print(f"  {'임계값':>6}{'재현율':>9}{'정밀도':>9}{'오탐률':>9}")
        for th in (37, 50, 62, 75):
            tp = sum(1 for v in fail if v >= th)
            fp = sum(1 for v in comp if v >= th)
            rec = tp / len(fail)
            prec = tp / (tp + fp) if (tp + fp) else 0
            print(f"  {th:>6}{rec:>9.1%}{prec:>9.1%}{fp/len(comp):>9.1%}")
    print()


def main() -> None:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    train = [r for r in rows if not in_test(r["nct_id"])]
    test = [r for r in rows if in_test(r["nct_id"])]
    print(f"전체 {len(rows)}건 → train {len(train)} / test {len(test)}\n")

    bench_train = build_benchmarks(train)

    # 참조 분포는 train 으로만 만들고, test 에서만 채점한다
    report("TEST (train 참조 분포)", score_all(test, bench_train))

    # 비교용: 같은 참조 분포로 train 을 채점하면 얼마나 낙관적인가
    report("TRAIN (참고 — 낙관 편향 확인용)", score_all(train, bench_train))


if __name__ == "__main__":
    main()
