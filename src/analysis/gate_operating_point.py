"""검증 게이트 적용 후 Tier 1 의 실제 동작점 — holdout 기준.

## 왜 이 측정이 필요했나

`pipeline.py` 에 `RISK_ELIGIBLE` 게이트를 넣어, 적응증 층화 검증을 통과한 피처
(현재 `monthly_enrollment_burden` 하나)만 위험 판정을 낼 수 있게 바꿨다. 그 결과
철회된 경쟁 밀도 피처와 방향이 반대인 2차 평가변수 피처가 더 이상 HIGH 를 발화하지 않는다.

그러면 **게이트를 통과한 감사가 실제로 얼마나 발화하는지**가 새 질문이 된다.
아무것도 발화하지 않으면 도구가 침묵하는 것이고, 너무 자주 발화하면 경고가 무의미하다.

`holdout.py` 는 종합 점수의 AUC 를 재고, 이 스크립트는 **이진 발화(위험 판정 1건 이상)의
민감도·오탐률**을 잰다. 사용자가 실제로 보는 것이 점수가 아니라 finding 목록이므로
(validated-numbers.md 3.3 에서 점수 제시를 폐기했다) 이쪽이 실사용 동작점이다.

## in-sample 을 쓰면 안 되는 이유

심각도 임계값이 참조 분포의 분위수(p25/median)에서 나온다. 즉 **fitting 이 있다.**
같은 코퍼스로 분포를 만들고 채점하면 낙관 편향된다. 실측하니 차이가 났다:

    in-sample : 민감도 65.2% / 오탐률 37.1%
    holdout   : 이 스크립트의 출력값 ← 제안서에는 이것만 쓴다

분할은 `holdout.in_test()` 재사용 — NCT ID SHA-256 해시 기반이라 실행마다 동일하다.

실행: python3 src/analysis/gate_operating_point.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.holdout import in_test  # noqa: E402
from audit import benchmarks as bm  # noqa: E402
from audit.pipeline import RISK_ELIGIBLE, audit_features  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
N_BOOT = 2000
BOOT_SEED = 20260730


def build_benchmarks(rows: list[dict]) -> dict:
    """`benchmarks.build()` 와 **동일한 4단 버킷 키**로 만든다.

    `holdout.py` 의 build_benchmarks 는 적응증과 `__ALL__` 만 써서 Phase 계층이 빠져 있다.
    실제 파이프라인은 `stratum_key()` 로 Phase 계층을 먼저 찾으므로, 그것을 재현하지 않으면
    다른 파이프라인을 측정하는 셈이 된다.
    """
    acc: dict = {}
    for r in rows:
        cohort = bm.cohort_of(r)
        if not cohort:
            continue
        feats = bm.extract(r)
        ph = bm.phase_stratum(r["v0"])
        for bucket in (
            bm.stratum_key(r["condition_bucket"], r["v0"]),
            r["condition_bucket"],
            f"__ALL__|{ph}",
            "__ALL__",
        ):
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


def fires(row: dict, bench: dict) -> bool:
    risk, _ = audit_features(row["v0"], row["condition_bucket"], bench, row["nct_id"])
    return bool(risk)


def prop_ci(hits: int, n: int) -> tuple[float, float]:
    """부트스트랩 비율 신뢰구간. 표본이 작아 정규근사를 쓰지 않는다."""
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(BOOT_SEED)
    obs = [1] * hits + [0] * (n - hits)
    boots = sorted(sum(rng.choice(obs) for _ in obs) / n for _ in range(N_BOOT))
    return boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]


def measure(rows: list[dict], bench: dict, label: str) -> dict:
    by_cohort: dict[str, list[bool]] = {}
    for r in rows:
        c = bm.cohort_of(r)
        if not c:
            continue
        by_cohort.setdefault(c, []).append(fires(r, bench))

    fail = by_cohort.get("recruit_failed", [])
    comp = by_cohort.get("completed", [])
    if not fail or not comp:
        print(f"[{label}] 표본 부족 — 모집실패 {len(fail)} / 완주 {len(comp)}")
        return {}

    tp, fp = sum(fail), sum(comp)
    sens, fpr = tp / len(fail), fp / len(comp)
    s_lo, s_hi = prop_ci(tp, len(fail))
    f_lo, f_hi = prop_ci(fp, len(comp))

    print(f"[{label}]  모집실패 {len(fail)}건 / 완주 {len(comp)}건")
    print(f"  민감도 (모집실패에서 발화)  {sens:>7.1%}   95% CI [{s_lo:.1%}, {s_hi:.1%}]")
    print(f"  오탐률 (완주군에서 발화)    {fpr:>7.1%}   95% CI [{f_lo:.1%}, {f_hi:.1%}]")
    print(f"  특이도                      {1-fpr:>7.1%}")
    lift = sens / fpr if fpr else float("inf")
    print(f"  발화 리프트 (민감도/오탐률) {lift:>7.2f}배   ← 1.0 이면 무작위\n")
    return {
        "n_fail": len(fail), "n_comp": len(comp),
        "sensitivity": round(sens, 3), "sensitivity_ci95": [round(s_lo, 3), round(s_hi, 3)],
        "false_positive_rate": round(fpr, 3), "fpr_ci95": [round(f_lo, 3), round(f_hi, 3)],
        "lift": round(lift, 2),
    }


def main() -> None:
    rows = [json.loads(line) for line in RAW.open() if line.strip()]
    train = [r for r in rows if not in_test(r["nct_id"])]
    test = [r for r in rows if in_test(r["nct_id"])]

    print(f"게이트 통과 피처: {sorted(RISK_ELIGIBLE)}")
    print(f"전체 {len(rows)}건 → train {len(train)} / test {len(test)}")
    print(f"부트스트랩 {N_BOOT}회, seed {BOOT_SEED}\n")

    bench_train = build_benchmarks(train)

    print("── 참조 분포를 train 으로만 만들고 test 에서만 채점 ──\n")
    result = {"test": measure(test, bench_train, "TEST — 제안서 인용값")}
    result["train"] = measure(train, bench_train, "TRAIN — 낙관 편향 확인용")

    out = Path("data/gate_operating_point.json")
    out.write_text(json.dumps(
        {"_meta": {"n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                   "risk_eligible": sorted(RISK_ELIGIBLE),
                   "note": "이진 발화(위험 판정 1건 이상)의 동작점. TEST 만 인용 가능."},
         **result}, ensure_ascii=False, indent=1))
    print(f"{out} 저장 — 제안서는 TEST 행만 인용한다")


if __name__ == "__main__":
    main()
