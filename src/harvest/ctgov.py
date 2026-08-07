"""ClinicalTrials.gov 하베스터.

핵심 아이디어: `/api/int/studies/{nct}/history/0` 이 트라이얼 개시 전 최초 등록본을 그대로 준다.
v0 = 블라인드 입력(당시 심사자가 볼 수 있던 정보 전부), 최종 버전 = 정답 라벨.
따라서 라벨을 수동으로 만들 필요가 없고 leakage 통제가 구조적으로 보장된다.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import requests

V2 = "https://clinicaltrials.gov/api/v2"
INT = "https://clinicaltrials.gov/api/int"
UA = {"User-Agent": "jump-ai-4th-protocol-audit/0.1 (research)"}

SEARCH_FIELDS = ",".join(
    [
        "NCTId",
        "BriefTitle",
        "OverallStatus",
        "WhyStopped",
        "Phase",
        "Condition",
        "StartDate",
        "EnrollmentCount",
        "EnrollmentType",
        "LeadSponsorClass",
    ]
)


class Http:
    """재시도·레이트리밋을 갖춘 얇은 GET 래퍼."""

    def __init__(self, delay: float = 0.34, retries: int = 4) -> None:
        self.session = requests.Session()
        self.session.headers.update(UA)
        self.delay = delay
        self.retries = retries

    def get_json(self, url: str, params: dict | None = None) -> dict[str, Any] | None:
        for attempt in range(self.retries):
            time.sleep(self.delay)
            try:
                r = self.session.get(url, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if 400 <= r.status_code < 500 and r.status_code != 429:
                # 잘못된 쿼리는 재시도해도 낫지 않는다. 조용히 0건으로 보이면 디버깅이 불가능하므로 즉시 터뜨린다.
                raise RuntimeError(f"{r.status_code} {url}\n{r.text[:300]}")
            # 429/5xx → 백오프
            time.sleep(3 * (attempt + 1))
        return None


def search(
    http: Http,
    condition: str,
    status: str,
    phase: str = "PHASE2",
    start_from: str = "2010-01-01",
    start_to: str = "2020-12-31",
    limit: int = 200,
) -> Iterator[dict[str, Any]]:
    """조건·상태별 인터벤셔널 트라이얼 검색. 개시일 범위를 제한해 결과가 확정된 것만 받는다."""
    token = None
    seen = 0
    while seen < limit:
        params = {
            "query.cond": condition,
            "filter.overallStatus": status,
            "filter.advanced": (
                f"AREA[Phase]{phase} AND AREA[StudyType]INTERVENTIONAL "
                f"AND AREA[StartDate]RANGE[{start_from},{start_to}]"
            ),
            "fields": SEARCH_FIELDS,
            "pageSize": min(100, limit - seen),
            "countTotal": "true",
        }
        if token:
            params["pageToken"] = token
        data = http.get_json(f"{V2}/studies", params)
        if not data:
            return
        studies = data.get("studies") or []
        if not studies:
            return
        for s in studies:
            yield s
            seen += 1
        token = data.get("nextPageToken")
        if not token:
            return


# --- v0(블라인드 입력) 추출 ------------------------------------------------


def _strip_html(t: str | None) -> str:
    if not t:
        return ""
    t = re.sub(r"</(p|li|div)>", "\n", t)
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _count_criteria(criteria: str) -> tuple[int, int]:
    """포함/제외 기준 항목 수를 센다. 적격기준 제약도(restrictiveness) 프록시."""
    low = criteria.lower()
    split = re.split(r"exclusion criteria", low, maxsplit=1)
    def n_items(chunk: str) -> int:
        return len([l for l in chunk.splitlines() if len(l.strip()) > 15])
    inc = n_items(split[0]) if split else 0
    exc = n_items(split[1]) if len(split) > 1 else 0
    return inc, exc


@dataclass
class Record:
    nct_id: str
    condition_bucket: str
    # --- 블라인드 입력 (v0에서만) ---
    v0: dict[str, Any] = field(default_factory=dict)
    # --- 정답 라벨 (최종 버전에서만) ---
    labels: dict[str, Any] = field(default_factory=dict)


def extract_v0(proto: dict[str, Any]) -> dict[str, Any]:
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    elig = proto.get("eligibilityModule", {})
    arms = proto.get("armsInterventionsModule", {})
    outcomes = proto.get("outcomesModule", {})
    loc = proto.get("contactsLocationsModule", {})
    sponsor = proto.get("sponsorCollaboratorsModule", {})
    desc = proto.get("descriptionModule", {})

    criteria = _strip_html(elig.get("eligibilityCriteria"))
    inc, exc = _count_criteria(criteria)
    locations = loc.get("locations") or []
    info = design.get("designInfo") or {}

    return {
        "brief_title": ident.get("briefTitle"),
        "start_date": (status.get("startDateStruct") or {}).get("date"),
        "planned_primary_completion": (status.get("primaryCompletionDateStruct") or {}).get("date"),
        "planned_enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "planned_enrollment_type": (design.get("enrollmentInfo") or {}).get("type"),
        "phases": design.get("phases"),
        "allocation": info.get("allocation"),
        "masking": ((info.get("maskingInfo") or {}).get("masking")),
        "primary_purpose": info.get("primaryPurpose"),
        "n_arms": len(arms.get("armGroups") or []),
        "interventions": [
            {"type": i.get("type"), "name": i.get("name")}
            for i in (arms.get("interventions") or [])
        ],
        "primary_outcomes": [
            {"measure": o.get("measure"), "timeframe": o.get("timeFrame")}
            for o in (outcomes.get("primaryOutcomes") or [])
        ],
        "n_secondary_outcomes": len(outcomes.get("secondaryOutcomes") or []),
        "eligibility_criteria": criteria,
        "n_inclusion_items": inc,
        "n_exclusion_items": exc,
        "min_age": elig.get("minimumAge"),
        "max_age": elig.get("maximumAge"),
        "sex": elig.get("sex"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "n_sites": len(locations),
        "countries": sorted({l.get("country") for l in locations if l.get("country")}),
        "sponsor_class": (sponsor.get("leadSponsor") or {}).get("class"),
        "brief_summary": _strip_html(desc.get("briefSummary"))[:2000],
    }


def extract_labels(proto: dict[str, Any], changes: list[dict]) -> dict[str, Any]:
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})

    def module_amendments(label: str) -> int:
        # version 0 은 최초 등록이므로 개정 횟수에서 제외
        return sum(
            1 for c in changes if c.get("version", 0) > 0 and label in (c.get("moduleLabels") or [])
        )

    return {
        "final_status": status.get("overallStatus"),
        "why_stopped": status.get("whyStopped"),
        "actual_enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "actual_enrollment_type": (design.get("enrollmentInfo") or {}).get("type"),
        "actual_primary_completion": (status.get("primaryCompletionDateStruct") or {}).get("date"),
        "n_versions": len(changes),
        "amend_eligibility": module_amendments("Eligibility"),
        "amend_outcomes": module_amendments("Outcome Measures"),
        "amend_design": module_amendments("Study Design"),
        "amend_arms": module_amendments("Arms and Interventions"),
        "amend_locations": module_amendments("Contacts/Locations"),
    }


def fetch_record(http: Http, nct_id: str, bucket: str) -> Record | None:
    hist = http.get_json(f"{INT}/studies/{nct_id}/history")
    if not hist or not hist.get("changes"):
        return None
    changes = hist["changes"]
    v0 = http.get_json(f"{INT}/studies/{nct_id}/history/0")
    last = http.get_json(f"{INT}/studies/{nct_id}/history/{changes[-1]['version']}")
    if not v0 or not last:
        return None
    return Record(
        nct_id=nct_id,
        condition_bucket=bucket,
        v0=extract_v0(v0["study"]["protocolSection"]),
        labels=extract_labels(last["study"]["protocolSection"], changes),
    )


def harvest(
    buckets: list[str],
    statuses: list[str],
    per_cell: int,
    out_path: Path,
) -> list[Record]:
    http = Http()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {json.loads(l)["nct_id"] for l in f if l.strip()}
        print(f"[resume] 기존 {len(done)}건 건너뜀")

    records: list[Record] = []
    with out_path.open("a") as f:
        for bucket in buckets:
            for status in statuses:
                hits = list(search(http, bucket, status, limit=per_cell))
                print(f"[search] {bucket} / {status}: {len(hits)}건")
                for s in hits:
                    nct = s["protocolSection"]["identificationModule"]["nctId"]
                    if nct in done:
                        continue
                    rec = fetch_record(http, nct, bucket)
                    if not rec:
                        print(f"  ! {nct} 이력 없음, 건너뜀")
                        continue
                    f.write(
                        json.dumps(
                            {"nct_id": rec.nct_id, "condition_bucket": bucket,
                             "v0": rec.v0, "labels": rec.labels},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()
                    done.add(nct)
                    records.append(rec)
                print(f"  → 누적 {len(done)}건")
    return records


if __name__ == "__main__":
    BUCKETS = [
        "non-small cell lung cancer",
        "atopic dermatitis",
        "type 2 diabetes",
        "rheumatoid arthritis",
        "alzheimer disease",
    ]
    harvest(
        buckets=BUCKETS,
        statuses=["TERMINATED", "WITHDRAWN", "COMPLETED"],
        per_cell=40,
        out_path=Path("data/raw/ctgov_phase2.jsonl"),
    )
