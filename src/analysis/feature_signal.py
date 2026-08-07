"""Tier 1 계량 감사 피처가 실제로 신호를 갖는지 검증.

과제 정의는 design-v2.md 2.2 에 따른다:
  양성군 = A분류(모집 실패)로 중단된 트라이얼
  음성군 = COMPLETED
  목표변수 = 등록 달성률

v0에만 존재하는 피처만 쓴다. 여기서 신호가 없으면 Tier 1(계량 감사) 계층의 전제가 무너지고,
LLM 레드팀만 남게 되므로 설계를 다시 봐야 한다.
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

MASKING_RANK = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}


def v0_features(row: dict) -> dict[str, float | None]:
    """v0에서만 뽑는 피처. 결과 정보가 섞이면 안 된다."""
    v0 = row["v0"]
    start = parse_date(v0.get("start_date"))
    end = parse_date(v0.get("planned_primary_completion"))
    planned_dur = round((end - start).days / 30.44, 1) if start and end and end > start else None
    n_sites = v0.get("n_sites")
    enroll = v0.get("planned_enrollment")

    return {
        "계획_등록수": enroll,
        "계획_기간_개월": planned_dur,
        "포함기준_항목수": v0.get("n_inclusion_items") or None,
        "제외기준_항목수": v0.get("n_exclusion_items") or None,
        "적격기준_총항목": (
            (v0.get("n_inclusion_items") or 0) + (v0.get("n_exclusion_items") or 0)
        ) or None,
        "적격기준_글자수": len(v0.get("eligibility_criteria") or "") or None,
        "사이트_수": n_sites if n_sites else None,
        "국가_수": len(v0.get("countries") or []) or None,
        "군_수": v0.get("n_arms") or None,
        "2차평가변수_수": v0.get("n_secondary_outcomes") or None,
        "맹검_수준": MASKING_RANK.get(v0.get("masking") or ""),
        # 파생: 사이트당 월 모집 부담 — 일정 현실성의 핵심 지표
        "사이트당_월모집부담": (
            round(enroll / n_sites / planned_dur, 3)
            if enroll and n_sites and planned_dur else None
        ),
        "월_모집부담": round(enroll / planned_dur, 2) if enroll and planned_dur else None,
    }


def main() -> None:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]

    cohort = []
    for r in rows:
        att = enrollment_attainment(
            r["v0"].get("planned_enrollment"), r["labels"].get("actual_enrollment")
        )
        if att is None:
            continue
        status = r["labels"]["final_status"]
        cat = classify(r["labels"].get("why_stopped"))[0]
        if status == "COMPLETED":
            group = 0  # 음성
        elif status == "TERMINATED" and cat == "A":
            group = 1  # 양성 (모집 실패)
        else:
            continue
        cohort.append({"att": att, "group": group, **v0_features(r)})

    n_pos = sum(c["group"] for c in cohort)
    print(f"코호트 {len(cohort)}건 (모집실패 {n_pos} / 완료 {len(cohort)-n_pos})\n")
    if n_pos < 8:
        print("양성군 표본 부족 — 수집을 더 늘려야 한다.")
        return

    feature_names = [k for k in cohort[0] if k not in ("att", "group")]

    print("[1] 피처별 신호 — 모집실패군 vs 완료군 중앙값 비교")
    print(f"  {'피처':<22}{'실패군':>10}{'완료군':>10}{'차이배수':>10}{'ρ(달성률)':>12}")
    results = []
    for f in feature_names:
        pos = [c[f] for c in cohort if c["group"] == 1 and c[f] is not None]
        neg = [c[f] for c in cohort if c["group"] == 0 and c[f] is not None]
        if len(pos) < 6 or len(neg) < 6:
            print(f"  {f:<22}{'표본부족':>10} (실패군 n={len(pos)}, 완료군 n={len(neg)})")
            continue
        mp, mn = st.median(pos), st.median(neg)
        ratio = round(mp / mn, 2) if mn else None
        paired = [(c[f], c["att"]) for c in cohort if c[f] is not None]
        rho = spearman([p[0] for p in paired], [p[1] for p in paired])
        results.append((abs(rho) if rho else 0, f, mp, mn, ratio, rho))
        print(f"  {f:<22}{mp:>10.2f}{mn:>10.2f}{str(ratio):>10}{str(rho):>12}")

    print("\n[2] 신호 강한 순 (|ρ| 기준)")
    for _, f, mp, mn, ratio, rho in sorted(results, reverse=True)[:6]:
        print(f"  {f:<22} ρ={rho:<8} 실패군 {mp:.2f} vs 완료군 {mn:.2f} ({ratio}배)")

    print("\n해석: |ρ| 0.2 이상 피처가 여러 개면 Tier 1 계량 감사만으로도 baseline이 성립한다.")
    print("      LLM 계층은 그 위에서 얼마나 더 올리는지로 평가하면 된다.")


if __name__ == "__main__":
    main()
