"""코호트 정의 민감도 분석 — 결론이 라벨 정의에 의존하는가.

## 왜 이 분석이 필요했나

`whyStopped` 에 중단 사유가 **복합으로 기재**되는 경우가 사유 기재분의 7.9% 있다.
예: `"Sponsor decision due to slow accrual"`, `"poor recruitment, funding"`,
`"Standard of care changed to FOLFIRINOX; poor accrual"`.

`taxonomy.PATTERNS` 는 D→C→B→A 순서로 첫 매칭을 채택하므로, 이런 경우 **A(모집 실패)가
가장 뒤로 밀린다.** 실측하니 A 가 매칭되지만 다른 분류로 배정된 사례가 33건 —
**A 코호트(176건)의 19%** 였다.

어느 쪽이 옳은지 데이터만으로 결정할 수 없다:
- 현행(C/D/B 우선): "자금 부족으로 모집 중단"은 외생 요인이다 — `taxonomy.py` docstring 의 근거
- A 우선: 모집 실패가 **원인**이고 스폰서 결정은 종료 **절차**다

그리고 A 우선으로 바꾸면 `"pandemic-related slow accrual"` 처럼 **등록 시점에 예견 불가능한**
사례가 표적 코호트에 섞인다 — 4분류의 자체 기준("에이전트가 등록 시점에 예견할 수 있었는가")과
충돌한다.

→ **한쪽을 고르지 않고 양쪽으로 측정한다.** 두 값이 비슷하면 결론이 라벨 정의에 강건하다는
증거가 되고, 크게 다르면 그 사실 자체를 한계로 보고한다. 이 프로젝트는 이미 검증 없이 통과
판정한 피처를 철회한 이력이 있어, 정의 선택으로 결과가 바뀌는 것을 숨기지 않는다.

실행: python3 src/analysis/cohort_sensitivity.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import group_of  # noqa: E402
from analysis.stratified_signal import (  # noqa: E402
    BOOT_SEED,
    STRATA,
    boot_ci,
    indication_only_control,
    stratified_auc,
)
from audit.benchmarks import FEATURES, extract, phase_stratum  # noqa: E402
from labels.taxonomy import classify, classify_all  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
OUT = Path("data/cohort_sensitivity.json")

# 표적과 특이도 집단. 표적은 높아야 하고 특이도 집단은 0.5 근처여야 한다.
COMPARISONS = {
    "recruit_vs_completed": ("중단-모집실패", "완주"),
    "scientific_vs_completed": ("중단-과학적", "완주"),
    "business_vs_completed": ("중단-경영", "완주"),
}

# 제안서 인용값은 적응증 층화다. 5축 전부를 재면 실행이 오래 걸리므로 핵심 3축만 본다.
FOCUS_STRATA = ("none", "indication", "indication+phase")


def load_records(a_priority: bool) -> list[dict]:
    recs = []
    for line in RAW.open():
        if not line.strip():
            continue
        r = json.loads(line)
        recs.append(
            {
                "nct": r["nct_id"],
                "ind": r["condition_bucket"],
                "phase": phase_stratum(r["v0"]),
                "sponsor": r["v0"].get("sponsor_class"),
                "group": group_of(r, a_priority=a_priority),
                "f": extract(r),
            }
        )
    return recs


def ambiguity_report() -> dict:
    """복합 사유의 규모를 먼저 확인한다."""
    rows = [json.loads(line) for line in RAW.open() if line.strip()]
    stopped = [r for r in rows if r["labels"]["final_status"] in ("TERMINATED", "WITHDRAWN")]
    described = [r for r in stopped if (r["labels"].get("why_stopped") or "").strip()]

    multi = shadowed = 0
    examples = []
    for r in described:
        ws = r["labels"]["why_stopped"]
        cats = [c for c, _, _ in classify_all(ws)]
        if len(cats) > 1:
            multi += 1
        if "A" in cats and cats[0] != "A":
            shadowed += 1
            if len(examples) < 8:
                examples.append({"primary": cats[0], "why_stopped": ws[:120]})

    print(f"중단·철회 {len(stopped)}건 / 사유 기재 {len(described)}건")
    print(f"  복합 사유(2개 이상 분류 매칭)  {multi}건 ({multi/len(described):.1%})")
    print(f"  A 가 우선순위에 밀린 사례        {shadowed}건 ({shadowed/len(described):.1%})\n")
    return {
        "n_stopped": len(stopped), "n_described": len(described),
        "n_multi_match": multi, "n_a_shadowed": shadowed, "examples": examples,
    }


def run(a_priority: bool, label: str) -> dict:
    recs = load_records(a_priority)
    rng = random.Random(BOOT_SEED)
    result: dict = {}

    counts = {g: sum(1 for r in recs if r["group"] == g) for g in
              ("완주", "중단-모집실패", "중단-과학적", "중단-경영", "중단-경쟁환경", "중단-미분류")}
    print(f"{'='*88}\n[{label}]  a_priority={a_priority}")
    print("  집단 크기: " + "  ".join(f"{g} {n}" for g, n in counts.items() if n))
    result["group_counts"] = counts

    for comp, (pos_g, neg_g) in COMPARISONS.items():
        ctrl = indication_only_control(recs, pos_g, neg_g)
        print(f"\n  {comp}   음성 대조군(적응증 라벨만) = {ctrl}")
        print(f"    {'피처':<30}" + "".join(f"{s:>22}" for s in FOCUS_STRATA))
        per_feature: dict = {}
        for key in FEATURES:
            cells, row = [], {}
            for s in FOCUS_STRATA:
                a, _, _ = stratified_auc(recs, key, STRATA[s], pos_g, neg_g)
                if a is None:
                    cells.append(f"{'—':>22}")
                    row[s] = None
                    continue
                ci = boot_ci(recs, key, STRATA[s], pos_g, neg_g, rng)
                mark = "*" if ci and ci[0] > 0.5 else " "
                cells.append(f"{a:>9}{mark} [{ci[0]:.2f},{ci[1]:.2f}]".rjust(22) if ci
                             else f"{a:>22}")
                row[s] = {"auc": a, "ci95": list(ci) if ci else None}
            per_feature[key] = row
            print(f"    {key:<30}" + "".join(cells))
        robust = [k for k, v in per_feature.items()
                  if all(s and s["ci95"] and s["ci95"][0] > 0.5 for s in v.values())]
        print(f"    ★ 3축 전부 CI 하한>0.5: {robust or '없음'}")
        result[comp] = {"indication_only_control": ctrl, "features": per_feature,
                        "robust_features": robust}
    print()
    return result


def main() -> None:
    print(f"게이트 대상 피처 {len(FEATURES)}개, 부트스트랩 seed {BOOT_SEED}\n")
    amb = ambiguity_report()

    out = {
        "_meta": {
            "source": str(RAW), "boot_seed": BOOT_SEED, "focus_strata": list(FOCUS_STRATA),
            "note": "정의1=현행(C/D/B 우선), 정의2=A 우선. 제안서는 두 값을 함께 인용한다.",
        },
        "ambiguity": amb,
        "def1_current": run(False, "정의 1 — 현행 (D→C→B→A 순서, 순수 A만)"),
        "def2_a_priority": run(True, "정의 2 — A 우선 (복합 사유에서 A 채택)"),
    }

    # 핵심 비교: 월 모집부담의 적응증 층화 AUC 가 정의에 따라 얼마나 움직이는가
    print(f"{'='*88}\n결론 — 월 모집부담 (적응증 층화), 모집실패 vs 완주\n{'='*88}")
    for name, key in (("정의 1 (현행)", "def1_current"), ("정의 2 (A 우선)", "def2_a_priority")):
        d = out[key]["recruit_vs_completed"]["features"]["monthly_enrollment_burden"]["indication"]
        n = out[key]["group_counts"]["중단-모집실패"]
        if d:
            lo, hi = d["ci95"]
            print(f"  {name:<16} A={n:>4}건   AUC {d['auc']}  95% CI [{lo:.3f}, {hi:.3f}]")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n{OUT} 저장")


if __name__ == "__main__":
    main()
