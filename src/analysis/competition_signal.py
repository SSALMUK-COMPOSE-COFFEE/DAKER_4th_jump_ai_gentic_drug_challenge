"""경쟁 시험 밀도 피처의 구분력 검증 — 아직 한 번도 안 재본 항목.

이 피처가 프로젝트에서 가장 독창적인 부분인데, 배치 채점에서는 적응증 풀 조회(네트워크)가
필요해서 계속 제외돼 있었다. 나머지 피처가 holdout AUC 0.587 로 미입증이므로,
이 피처가 신호를 갖는지가 Tier 1 존속 여부를 결정한다.

계산: 대상 트라이얼 개시 시점에 같은 적응증에서 모집 창이 열려 있던 트라이얼 수와 목표 등록수 합.
적응증 풀은 한 번만 받아 캐시한다 (15개 적응증 × 최대 3000건).

출력: data/competition_features.json  (nct_id → 피처)
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.amendment_confound import parse_date  # noqa: E402
from analysis.calibration import auc, group_of  # noqa: E402
from analysis.holdout import auc_ci  # noqa: E402
from audit.competition import competition_at, fetch_indication_pool  # noqa: E402
from harvest.ctgov import Http  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
OUT = Path("data/competition_features.json")


def build() -> dict[str, dict]:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    cached: dict[str, dict] = {}
    if OUT.exists():
        cached = json.loads(OUT.read_text())
        print(f"[resume] 기존 {len(cached)}건 재사용")

    http = Http(delay=0.15)
    pools: dict[str, list] = {}
    todo = [r for r in rows if r["nct_id"] not in cached]
    print(f"계산 대상 {len(todo)}건 / 전체 {len(rows)}건")

    for i, r in enumerate(todo, 1):
        bucket = r["condition_bucket"]
        start = parse_date(r["v0"].get("start_date"))
        if not start:
            cached[r["nct_id"]] = {}
            continue
        if bucket not in pools:
            pools[bucket] = fetch_indication_pool(http, bucket)
            dated = sum(1 for t in pools[bucket] if t["start"] and t["end"])
            print(f"  [pool] {bucket}: {len(pools[bucket])}건 (기간 있음 {dated})")
        cached[r["nct_id"]] = competition_at(pools[bucket], r["nct_id"], start)
        if i % 200 == 0:
            OUT.write_text(json.dumps(cached, ensure_ascii=False))
            print(f"  ... {i}/{len(todo)}")

    OUT.write_text(json.dumps(cached, ensure_ascii=False, indent=1))
    return cached


def evaluate(feats: dict[str, dict]) -> None:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    by_group: dict[str, dict[str, list]] = {}
    for r in rows:
        f = feats.get(r["nct_id"]) or {}
        if not f:
            continue
        g = group_of(r)
        for k in ("n_concurrent_trials", "n_concurrent_phase2", "total_competing_seats"):
            v = f.get(k)
            if v is not None:
                by_group.setdefault(k, {}).setdefault(g, []).append(float(v))

    order = ["완주", "중단-모집실패", "중단-경쟁환경", "중단-과학적", "중단-경영"]
    for k, groups in by_group.items():
        print(f"\n[{k}]")
        for g in order:
            vals = groups.get(g) or []
            if not vals:
                continue
            vs = sorted(vals)
            print(
                f"  {g:<14} n={len(vs):<4} 중앙값={st.median(vs):>10,.0f} "
                f"25%={vs[len(vs)//4]:>10,.0f} 75%={vs[3*len(vs)//4]:>10,.0f}"
            )
        comp = groups.get("완주") or []
        fail = groups.get("중단-모집실패") or []
        land = groups.get("중단-경쟁환경") or []
        if comp and fail:
            print(f"  AUC 모집실패 vs 완주 = {auc_ci(fail, comp)}")
        if comp and land:
            print(f"  AUC 경쟁환경 vs 완주 = {auc(land, comp)}  (n={len(land)}, 참고용)")


if __name__ == "__main__":
    feats = build()
    print(f"\n{OUT} — {len(feats)}건\n")
    print("=" * 70)
    print("경쟁 밀도 피처의 구분력")
    print("=" * 70)
    evaluate(feats)
