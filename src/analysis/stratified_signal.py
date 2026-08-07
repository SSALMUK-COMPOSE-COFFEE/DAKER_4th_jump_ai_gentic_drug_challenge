"""층화 신호 검증 — 제안서에 인용할 숫자를 확정하는 단일 소스.

## 왜 이 스크립트가 필요했나

`competition_signal.py` 는 층화 없이 통합 AUC 만 재서 경쟁 시험 밀도를 0.678 로 보고했고,
CI 하한이 0.5 를 넘는다는 이유로 "구분력 입증" 판정을 내렸다. **그 판정은 틀렸다.**
적응증으로 층화하면 0.472(우연 이하)이고, 프로토콜과 날짜를 전부 버리고 적응증 라벨만 쓰는
대조 예측기가 0.722 로 더 높다. 즉 그 피처는 "어느 질환인가"를 재인코딩한 것이었다.

※ 위 수치는 라벨러 정규식 수정(2026-07-30, A 코호트 112→121) 후 재측정값이다.
  단일 소스는 `docs/memory/validated-numbers.md` §3.1 이다.

실수의 원인은 **음성 대조군이 없었다**는 것이다. 그래서 이 스크립트는 모든 피처에 대해
대조군을 강제로 함께 출력한다.

## 방법론 — 홀드아웃이 필요한 곳과 필요 없는 곳

- **피처 수준 질문** ("이 원시 값이 두 집단을 구분하는가"): 아무것도 fitting 하지 않으므로
  홀드아웃이 불필요하다. 전체 코퍼스 AUC 가 이미 비편향 추정치이고, 필요한 것은 신뢰구간뿐이다.
  표본을 30% 버리면 검정력만 잃는다.
- **파이프라인 수준 질문** ("우리 감사 점수가 구분하는가"): 임계값이 참조 분위수에서 나오므로
  fitting 이 있다. 반드시 홀드아웃이 필요하다 → `holdout.py` 가 담당.

이 스크립트는 전자를 다룬다.

## 층화

교란 후보마다 층을 고정하고 층내 AUC 를 쌍 수로 가중 평균한다.
- `indication` — 확인된 교란원 (경쟁 밀도를 죽인 원인)
- `phase` — Phase 1/2 바스켓은 목표 등록수가 작고 적격기준이 촘촘한 것이 정상
- `sponsor_class` — 모집실패군 학술(OTHER) 74.4% vs 완주군 산업 50.1% 로 강하게 갈림
- `indication+phase` — 두 개 동시 고정

한 층에서만 살아남는 피처는 다른 층의 대리변수일 수 있으므로, **모든 층에서 CI 하한이 0.5 를
넘는 피처만 채택**한다.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import auc, group_of  # noqa: E402
from audit.benchmarks import FEATURES, HIGHER_IS_RISKIER, extract, phase_stratum  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
OUT = Path("data/validation.json")

N_BOOT = 400
BOOT_SEED = 20260730

# 층화 축 → 행에서 층 키를 뽑는 함수
STRATA = {
    "none": lambda r: "_",
    "indication": lambda r: r["ind"],
    "phase": lambda r: r["phase"],
    "sponsor": lambda r: r["sponsor"] or "UNKNOWN",
    "indication+phase": lambda r: f"{r['ind']}|{r['phase']}",
}

# 표적 집단 vs 대조 집단. 표적은 높아야 하고, 특이도 집단은 0.5 근처여야 한다.
COMPARISONS = {
    "recruit_vs_completed": ("중단-모집실패", "완주"),
    "scientific_vs_completed": ("중단-과학적", "완주"),
    "business_vs_completed": ("중단-경영", "완주"),
}


def load_records() -> list[dict]:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    recs = []
    for r in rows:
        recs.append(
            {
                "nct": r["nct_id"],
                "ind": r["condition_bucket"],
                "phase": phase_stratum(r["v0"]),
                "sponsor": r["v0"].get("sponsor_class"),
                "group": group_of(r),
                "f": extract(r),
            }
        )
    return recs


def stratified_auc(
    data: list[dict], key: str, stratum, pos_group: str, neg_group: str
) -> tuple[float | None, int, int]:
    """층내 AUC 를 쌍 수로 가중 평균. 방향은 HIGHER_IS_RISKIER 로 통일한다."""
    num = den = 0.0
    n_pos = n_neg = 0
    buckets: dict[str, tuple[list, list]] = {}
    for r in data:
        v = r["f"].get(key)
        if v is None:
            continue
        if r["group"] == pos_group:
            buckets.setdefault(stratum(r), ([], []))[0].append(float(v))
        elif r["group"] == neg_group:
            buckets.setdefault(stratum(r), ([], []))[1].append(float(v))
    for p, n in buckets.values():
        if not p or not n:
            continue
        n_pos += len(p)
        n_neg += len(n)
        a = auc(p, n)
        if key not in HIGHER_IS_RISKIER:
            a = 1 - a  # 낮을수록 위험한 피처는 방향을 뒤집어 "높으면 위험"으로 통일
        num += a * len(p) * len(n)
        den += len(p) * len(n)
    return (round(num / den, 3) if den else None), n_pos, n_neg


def boot_ci(
    data: list[dict], key: str, stratum, pos_group: str, neg_group: str, rng: random.Random
) -> tuple[float, float] | None:
    boots = []
    for _ in range(N_BOOT):
        sample = [rng.choice(data) for _ in data]
        a, _, _ = stratified_auc(sample, key, stratum, pos_group, neg_group)
        if a is not None:
            boots.append(a)
    if len(boots) < N_BOOT // 2:
        return None
    boots.sort()
    return boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def indication_only_control(data: list[dict], pos_group: str, neg_group: str) -> float | None:
    """음성 대조군: 프로토콜과 날짜를 전부 버리고 적응증 라벨만 쓴다.

    이 값보다 낮은 피처는 "질환을 맞히는 것"에 지나지 않는다.
    경쟁 밀도가 여기서 걸렸다 (미층화 0.678 → 적응증 층화 0.472 < 대조군 0.722).
    """
    by_ind: dict[str, list[int]] = {}
    for r in data:
        if r["group"] in (pos_group, neg_group):
            by_ind.setdefault(r["ind"], []).append(1 if r["group"] == pos_group else 0)
    rate = {ind: st.fmean(ys) for ind, ys in by_ind.items() if ys}
    pos = [rate[r["ind"]] for r in data if r["group"] == pos_group and r["ind"] in rate]
    neg = [rate[r["ind"]] for r in data if r["group"] == neg_group and r["ind"] in rate]
    return auc(pos, neg) if pos and neg else None


def main() -> None:
    recs = load_records()
    rng = random.Random(BOOT_SEED)
    result: dict = {"_meta": {"source": str(RAW), "n_rows": len(recs), "n_boot": N_BOOT,
                              "boot_seed": BOOT_SEED}, "comparisons": {}}

    for comp_name, (pos_g, neg_g) in COMPARISONS.items():
        n_p = sum(1 for r in recs if r["group"] == pos_g)
        n_n = sum(1 for r in recs if r["group"] == neg_g)
        print(f"\n{'='*78}\n{comp_name}  —  {pos_g} {n_p}건  vs  {neg_g} {n_n}건\n{'='*78}")

        ctrl = indication_only_control(recs, pos_g, neg_g)
        print(f"  ※ 음성 대조군 (적응증 라벨만) AUC = {ctrl}  ← 피처가 이보다 못하면 질환 판별기다\n")

        header = f"  {'피처':<28}" + "".join(f"{s:>17}" for s in STRATA)
        print(header)
        comp_result: dict = {"n_pos": n_p, "n_neg": n_n, "indication_only_control": ctrl,
                             "features": {}}

        for key in FEATURES:
            cells = []
            per_stratum: dict = {}
            for s_name, s_fn in STRATA.items():
                a, _, _ = stratified_auc(recs, key, s_fn, pos_g, neg_g)
                if a is None:
                    cells.append(f"{'—':>17}")
                    per_stratum[s_name] = None
                    continue
                ci = boot_ci(recs, key, s_fn, pos_g, neg_g, rng)
                mark = ""
                if ci:
                    mark = "*" if ci[0] > 0.5 else ""
                cells.append(f"{a:>10}{mark:<2}{'':>5}" if not ci else f"{a:>9}{mark:<1} {ci[0]:.2f}-{ci[1]:.2f}"[:17].rjust(17))
                per_stratum[s_name] = {"auc": a, "ci95": list(ci) if ci else None}
            print(f"  {key:<28}" + "".join(cells))
            comp_result["features"][key] = per_stratum

        # 모든 층에서 CI 하한 > 0.5 인 피처만 채택
        robust = [
            k for k, v in comp_result["features"].items()
            if all(s and s["ci95"] and s["ci95"][0] > 0.5 for s in v.values())
        ]
        comp_result["robust_features"] = robust
        if comp_name == "recruit_vs_completed":
            print(f"\n  ★ 모든 층에서 CI 하한>0.5 인 피처: {robust or '없음'}")
        result["comparisons"][comp_name] = comp_result

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n{OUT} 저장 — 제안서는 이 파일만 인용한다 (* = CI 하한 > 0.5)")


if __name__ == "__main__":
    main()
