"""참조 분포 산출 — 감사 결과에 "많다/적다"를 말할 근거를 만든다.

에이전트가 "포함기준 13개는 많습니다"라고 주장하려면 비교 대상이 있어야 한다.
수집한 1,463건에서 **완주군(COMPLETED)** 과 **모집실패군(A분류 TERMINATED)** 의 분포를 각각 뽑아
두 집단 중 어느 쪽에 가까운지로 판정한다.

완주군만 기준으로 삼지 않는 이유: "완주군 중앙값보다 크다"만으로는 위험 신호가 약하다.
"실패군 중앙값에 도달했다"가 훨씬 강한 경고이고, 실제로 두 분포는 뚜렷하게 떨어져 있다.

출력: data/benchmarks.json
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.amendment_confound import parse_date  # noqa: E402
from labels.taxonomy import classify  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
OUT = Path("data/benchmarks.json")
COMPETITION = Path("data/competition_features.json")

_comp_cache: dict[str, dict] | None = None


def competition_features(nct_id: str | None) -> dict:
    """사전 계산된 경쟁 밀도 피처를 읽는다.

    적응증 풀 조회가 느려서 배치로 미리 계산해 두고 여기서 조인한다.
    파일이 없으면 빈 dict — 경쟁 피처 없이도 나머지 감사는 동작해야 한다.
    """
    global _comp_cache
    if _comp_cache is None:
        _comp_cache = json.loads(COMPETITION.read_text()) if COMPETITION.exists() else {}
    return (_comp_cache.get(nct_id or "") or {}) if nct_id else {}

# 참조 분포를 만들 v0 피처. 전부 등록 시점에 계산 가능해야 한다.
FEATURES = (
    # 경쟁 시험 밀도 — ⚠ **철회된 피처군.** 한때 "holdout 에서 유일하게 구분력 입증"으로
    # 판정했으나(미층화 AUC 0.678), 적응증으로 층화하면 0.472 [0.411, 0.530] 로 우연 이하다.
    # 적응증 라벨만 쓰는 음성 대조군이 0.722 로 더 높아, "어느 질환인가"의 재인코딩이었다.
    # 참조 분포는 계속 만들되 위험 판정에는 쓰지 않는다 (pipeline.FEATURE_STATUS = retracted).
    "n_concurrent_phase2",
    "n_concurrent_trials",
    "total_competing_seats",
    "n_inclusion_items",
    "n_exclusion_items",
    "n_eligibility_items",
    "eligibility_chars",
    "planned_enrollment",
    "planned_duration_months",
    "monthly_enrollment_burden",
    "n_secondary_outcomes",
)

# 값이 클수록 위험한 피처 / 작을수록 위험한 피처를 구분해야 판정 방향이 맞는다.
HIGHER_IS_RISKIER = {
    "n_concurrent_phase2",
    "n_concurrent_trials",
    "total_competing_seats",
    "n_inclusion_items",
    "n_exclusion_items",
    "n_eligibility_items",
    "eligibility_chars",
    "planned_duration_months",
    "n_secondary_outcomes",
}


def extract(row: dict) -> dict[str, float | None]:
    v0 = row["v0"]
    comp = competition_features(row.get("nct_id"))
    start = parse_date(v0.get("start_date"))
    end = parse_date(v0.get("planned_primary_completion"))
    dur = round((end - start).days / 30.44, 1) if start and end and end > start else None
    enroll = v0.get("planned_enrollment")
    inc = v0.get("n_inclusion_items") or None
    exc = v0.get("n_exclusion_items") or None
    return {
        "n_concurrent_phase2": comp.get("n_concurrent_phase2"),
        "n_concurrent_trials": comp.get("n_concurrent_trials"),
        "total_competing_seats": comp.get("total_competing_seats"),
        "n_inclusion_items": inc,
        "n_exclusion_items": exc,
        "n_eligibility_items": (inc or 0) + (exc or 0) or None,
        "eligibility_chars": len(v0.get("eligibility_criteria") or "") or None,
        "planned_enrollment": enroll,
        "planned_duration_months": dur,
        "monthly_enrollment_burden": round(enroll / dur, 2) if enroll and dur else None,
        "n_secondary_outcomes": v0.get("n_secondary_outcomes") or None,
    }


def quantiles(vals: list[float]) -> dict | None:
    vs = sorted(v for v in vals if v is not None)
    if len(vs) < 6:
        return None
    return {
        "n": len(vs),
        "p25": round(vs[len(vs) // 4], 2),
        "median": round(st.median(vs), 2),
        "p75": round(vs[3 * len(vs) // 4], 2),
        "p90": round(vs[min(len(vs) - 1, int(len(vs) * 0.9))], 2),
    }


def phase_stratum(v0: dict) -> str:
    """Phase 계층. 순수 Phase 2 와 Phase 1/2 바스켓은 설계 관행이 달라 같이 비교하면 안 된다.

    Phase 1/2 는 용량 증량이 목적이라 목표 등록수가 작고 적격기준이 촘촘한 것이 정상이다.
    이를 순수 Phase 2 완주군과 비교하면 정상 설계가 위험으로 오판된다 (NCT03601897 오탐 원인).
    """
    ph = set(v0.get("phases") or [])
    if "PHASE1" in ph or "EARLY_PHASE1" in ph:
        return "p1p2"
    if "PHASE3" in ph:
        return "p2p3"
    return "p2"


def stratum_key(bucket: str, v0: dict) -> str:
    return f"{bucket}|{phase_stratum(v0)}"


def cohort_of(row: dict, *, a_priority: bool = False) -> str | None:
    """참조 분포용 코호트. `a_priority` 는 복합 사유에서 A 를 우선할지 결정한다.

    두 정의가 33건에서 갈린다 (A 코호트 121건의 약 27%). `src/analysis/cohort_sensitivity.py` 참조.
    """
    status = row["labels"]["final_status"]
    if status == "COMPLETED":
        return "completed"
    if (
        status == "TERMINATED"
        and classify(row["labels"].get("why_stopped"), a_priority=a_priority)[0] == "A"
    ):
        return "recruit_failed"
    return None


def build() -> dict:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]

    # bucket → cohort → feature → [값]
    acc: dict[str, dict[str, dict[str, list]]] = {}
    for r in rows:
        cohort = cohort_of(r)
        if not cohort:
            continue
        feats = extract(r)
        ph = phase_stratum(r["v0"])
        for bucket in (
            stratum_key(r["condition_bucket"], r["v0"]),
            r["condition_bucket"],
            f"__ALL__|{ph}",
            "__ALL__",
        ):
            per_cohort = acc.setdefault(bucket, {}).setdefault(cohort, {})
            for f in FEATURES:
                per_cohort.setdefault(f, []).append(feats[f])

    out: dict = {
        "_meta": {
            "source": str(RAW),
            "n_rows": len(rows),
            "features": list(FEATURES),
            "higher_is_riskier": sorted(HIGHER_IS_RISKIER),
            "note": (
                "cohort=completed 는 완주군, recruit_failed 는 whyStopped 가 모집 실패(A분류)인 중단군. "
                "표본이 6건 미만인 조합은 null 로 남긴다."
            ),
        },
        "buckets": {},
    }
    for bucket, cohorts in acc.items():
        out["buckets"][bucket] = {
            cohort: {f: quantiles(vals) for f, vals in feats.items()}
            for cohort, feats in cohorts.items()
        }
    return out


def load() -> dict:
    if not OUT.exists():
        raise SystemExit(f"{OUT} 가 없습니다. 먼저 `python src/audit/benchmarks.py` 를 실행하세요.")
    with OUT.open() as f:
        return json.load(f)


def reference(bench: dict, bucket: str, feature: str, v0: dict | None = None) -> dict | None:
    """참조 분포. 적응증+Phase 계층 → 적응증 → 전체+Phase → 전체 순으로 폴백한다.

    Phase 계층을 적응증보다 먼저 포기하지 않는 이유: Phase 1/2 를 순수 Phase 2 와 섞는 것이
    적응증을 섞는 것보다 오판을 크게 만든다 (설계 관행 자체가 다르다).
    """
    ph = phase_stratum(v0) if v0 else None
    chain = (
        [stratum_key(bucket, v0), f"__ALL__|{ph}", bucket, "__ALL__"] if v0
        else [bucket, "__ALL__"]
    )
    for key in chain:
        b = bench["buckets"].get(key)
        if not b:
            continue
        comp = (b.get("completed") or {}).get(feature)
        fail = (b.get("recruit_failed") or {}).get(feature)
        if comp and fail:
            return {"bucket_used": key, "completed": comp, "recruit_failed": fail}
        if comp and key == "__ALL__":
            return {"bucket_used": key, "completed": comp, "recruit_failed": None}
    return None


if __name__ == "__main__":
    bench = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(bench, f, ensure_ascii=False, indent=1)
    print(f"{OUT} 생성 — 버킷 {len(bench['buckets'])}개\n")

    all_b = bench["buckets"]["__ALL__"]
    print("전체(__ALL__) 참조 분포 — 완주군 vs 모집실패군 중앙값")
    print(f"  {'피처':<28}{'완주군':>10}{'실패군':>10}{'방향':>8}")
    for f in FEATURES:
        c = (all_b.get("completed") or {}).get(f)
        r = (all_b.get("recruit_failed") or {}).get(f)
        if not c or not r:
            print(f"  {f:<28}{'표본부족':>10}")
            continue
        arrow = "높을수록↑위험" if f in HIGHER_IS_RISKIER else "낮을수록↓위험"
        print(f"  {f:<28}{c['median']:>10}{r['median']:>10}{arrow:>8}")

    print("\n적응증별 표본 수 (완주군 / 모집실패군)")
    for bucket, co in sorted(bench["buckets"].items()):
        if bucket == "__ALL__":
            continue
        c = (co.get("completed") or {}).get("n_inclusion_items")
        r = (co.get("recruit_failed") or {}).get("n_inclusion_items")
        print(f"  {bucket:<32}{(c or {}).get('n', 0):>4} / {(r or {}).get('n', 0):<4}")
