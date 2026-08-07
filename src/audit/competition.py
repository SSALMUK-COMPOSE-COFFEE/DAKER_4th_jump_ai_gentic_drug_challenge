"""경쟁 시험 밀도 — Tier 1 계량 감사의 핵심 피처.

대상 트라이얼이 개시된 시점에, 같은 적응증에서 동시에 모집 중이던 트라이얼이 몇 개였고
그들이 합쳐서 몇 명을 뽑으려 했는지 센다. 환자 풀을 두고 경쟁한 정도가 모집 난항의 직접 원인이다.

순수 계산 피처라서 LLM leakage와 무관하고, 값이 재현 가능하다.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harvest.ctgov import V2, Http  # noqa: E402

FIELDS = ",".join(
    ["NCTId", "StartDate", "PrimaryCompletionDate", "CompletionDate", "EnrollmentCount", "Phase"]
)


def _parse(d: str | None) -> date | None:
    if not d:
        return None
    parts = d.split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(y, m, day)
    except (ValueError, IndexError):
        return None


def _get(study: dict, *path: str):
    cur = study
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def fetch_indication_pool(http: Http, condition: str, cap: int = 3000) -> list[dict]:
    """해당 적응증의 모든 인터벤셔널 트라이얼을 기간·등록수와 함께 가져온다."""
    out: list[dict] = []
    token = None
    while len(out) < cap:
        params = {
            "query.cond": condition,
            "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
            "fields": FIELDS,
            "pageSize": 100,
            "countTotal": "true",
        }
        if token:
            params["pageToken"] = token
        data = http.get_json(f"{V2}/studies", params)
        if not data:
            break
        studies = data.get("studies") or []
        if not studies:
            break
        for s in studies:
            p = s.get("protocolSection", {})
            out.append(
                {
                    "nct_id": _get(p, "identificationModule", "nctId"),
                    "start": _parse(_get(p, "statusModule", "startDateStruct", "date")),
                    "end": _parse(
                        _get(p, "statusModule", "primaryCompletionDateStruct", "date")
                        or _get(p, "statusModule", "completionDateStruct", "date")
                    ),
                    "enrollment": _get(p, "designModule", "enrollmentInfo", "count"),
                    "phases": _get(p, "designModule", "phases") or [],
                }
            )
        token = data.get("nextPageToken")
        if not token:
            break
    return out


def competition_at(pool: list[dict], target_nct: str, at: date) -> dict:
    """`at` 시점에 모집 창이 열려 있던 경쟁 트라이얼 집계."""
    concurrent = [
        t
        for t in pool
        if t["nct_id"] != target_nct
        and t["start"]
        and t["end"]
        and t["start"] <= at <= t["end"]
    ]
    seats = [t["enrollment"] for t in concurrent if t["enrollment"]]
    same_phase = [t for t in concurrent if any(p.startswith("PHASE2") for p in t["phases"])]
    return {
        "n_concurrent_trials": len(concurrent),
        "n_concurrent_phase2": len(same_phase),
        "total_competing_seats": sum(seats),
        "median_competitor_size": sorted(seats)[len(seats) // 2] if seats else None,
    }


if __name__ == "__main__":
    http = Http(delay=0.2)
    cases = [
        ("atopic dermatitis", "NCT03738423", date(2018, 11, 13)),
        ("atopic dermatitis", "NCT04220411", date(2020, 1, 1)),
        ("non-small cell lung cancer", "NCT03601897", date(2018, 8, 1)),
    ]
    pools: dict[str, list[dict]] = {}
    for cond, nct, start in cases:
        if cond not in pools:
            pools[cond] = fetch_indication_pool(http, cond)
            dated = sum(1 for t in pools[cond] if t["start"] and t["end"])
            print(f"[pool] {cond}: {len(pools[cond])}건 (기간 정보 있음 {dated}건)")
        stats = competition_at(pools[cond], nct, start)
        print(f"  {nct} @ {start} → {stats}")
