"""모수 통계 — 우리 코퍼스가 아니라 ClinicalTrials.gov 전수를 센다.

## 왜 이 스크립트가 필요했나

`data/raw/ctgov_phase2.jsonl` (1,463건) 은 **의도적으로 편향 수집**되었다.
완주군은 적응증당 40건 균일 수집, 중단군은 존재하는 만큼 수집했다
(`docs/memory/validated-numbers.md` §4.1). 따라서 그 코퍼스에서 나온
"중단 사유 A분류가 단일 최다" 같은 진술은 **코퍼스의 성질이지 임상연구의 성질이 아니다.**

제안서 항목 1(필요성·문제정의 20점)과 항목 5(비즈니스 가치 20점)는 모수 비율을 요구한다.
그래서 이 스크립트는 검색 필터를 제외한 어떤 표본추출도 하지 않고 **전수를 센다.**

## 방법

1. `countTotal=true` 카운트 쿼리로 Phase × 상태 조합의 총 건수만 싸게 얻는다
   (레코드를 받지 않는다. 조합당 요청 1회).
2. `whyStopped` 분류는 전수가 가능하다 — 필요한 필드가 3개뿐이라
   `pageSize=1000` 페이징으로 Phase 1/2/3 의 TERMINATED+WITHDRAWN 전체
   (약 1.5만 건)를 15~20 요청으로 받는다. **표본추출이 아니라 전수다.**
3. 분류는 `labels.taxonomy.classify()` 를 그대로 재사용한다 (규칙 기반 1차 패스).
   규칙이 매칭하지 못한 건은 `?` 로 남고, A분류 비율은 상·하한으로 구간을 준다.

## 실행

    .venv/bin/python3 src/analysis/census.py

출력: `data/raw/ctgov_census.jsonl` (중단·철회 전수, 3필드), `data/census.json` (집계)
기존 jsonl 이 있으면 재수집하지 않는다. 강제 재수집은 파일 삭제 후 재실행.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harvest.ctgov import V2, Http  # noqa: E402
from labels.taxonomy import CATEGORY_NAMES, classify  # noqa: E402

OUT_RAW = Path("data/raw/ctgov_census.jsonl")
OUT_JSON = Path("data/census.json")

# 코퍼스와 동일한 창을 쓴다 — 비교 가능성이 목적이다.
# 상한 2020: 2020년 개시분까지는 결과가 확정될 시간이 있었다고 본다.
START_FROM, START_TO = "2010-01-01", "2020-12-31"
BASE = f"AREA[StudyType]INTERVENTIONAL AND AREA[StartDate]RANGE[{START_FROM},{START_TO}]"

PHASES = ("PHASE1", "PHASE2", "PHASE3")
# 결과가 확정된 상태 3개 + 참고용
DETERMINED = ("COMPLETED", "TERMINATED", "WITHDRAWN")
OTHER_STATUS = ("UNKNOWN", "ACTIVE_NOT_RECRUITING", "RECRUITING", "SUSPENDED",
                "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING")

# 코퍼스의 15개 적응증. 모수 적응증 구성비를 얻어 코퍼스 균일수집과 대조한다.
BUCKETS = [
    "non-small cell lung cancer", "atopic dermatitis", "type 2 diabetes",
    "rheumatoid arthritis", "alzheimer disease", "pancreatic cancer", "glioblastoma",
    "amyotrophic lateral sclerosis", "idiopathic pulmonary fibrosis",
    "systemic lupus erythematosus", "sickle cell disease", "cystic fibrosis",
    "acute myeloid leukemia", "ovarian cancer", "heart failure",
]


# --- 통계 유틸 (numpy/scipy 없음) -----------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항비율 Wilson 95% 신뢰구간. 전수 집계에는 필요 없고 표본 추정에만 쓴다."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


# --- 1. 카운트 쿼리 --------------------------------------------------------


def count(http: Http, adv: str, status: str | None = None) -> int | None:
    params = {"filter.advanced": adv, "countTotal": "true", "pageSize": 1, "fields": "NCTId"}
    if status:
        params["filter.overallStatus"] = status
    data = http.get_json(f"{V2}/studies", params)
    return data.get("totalCount") if data else None


def phase_status_counts(http: Http) -> dict:
    out: dict = {}
    for ph in PHASES:
        adv = f"AREA[Phase]{ph} AND {BASE}"
        row = {"all": count(http, adv)}
        for st in DETERMINED + OTHER_STATUS:
            row[st] = count(http, adv, st)
        out[ph] = row
    # 순수 Phase 2 (PHASE1/2·PHASE2/3 바스켓 제외) — 코퍼스와 필터를 맞춘 값이 주값이고 이건 대조용
    adv_pure = f"AREA[Phase]PHASE2 AND NOT AREA[Phase]PHASE1 AND NOT AREA[Phase]PHASE3 AND {BASE}"
    out["PHASE2_PURE"] = {"all": count(http, adv_pure)} | {
        st: count(http, adv_pure, st) for st in DETERMINED
    }
    return out


def indication_counts(http: Http) -> dict:
    """15개 적응증별 Phase 2 상태 분포. 코퍼스 균일수집과 모수 구성비를 대조한다."""
    out: dict = {}
    for b in BUCKETS:
        adv = f"AREA[Phase]PHASE2 AND {BASE} AND AREA[ConditionSearch]({b})"
        out[b] = {"all": count(http, adv)} | {st: count(http, adv, st) for st in DETERMINED}
    return out


# --- 2. 중단·철회 전수 수집 (whyStopped 3필드만) ---------------------------

FIELDS = "NCTId,OverallStatus,WhyStopped,Phase,StartDate,LeadSponsorClass"


def page_all(http: Http, adv: str, status: str) -> list[dict]:
    rows, token = [], None
    while True:
        params = {
            "filter.advanced": adv,
            "filter.overallStatus": status,
            "fields": FIELDS,
            "pageSize": 1000,
        }
        if token:
            params["pageToken"] = token
        data = http.get_json(f"{V2}/studies", params)
        if not data or not data.get("studies"):
            break
        for s in data["studies"]:
            p = s.get("protocolSection", {})
            stat = p.get("statusModule", {})
            rows.append(
                {
                    "nct_id": p.get("identificationModule", {}).get("nctId"),
                    "phase_filter": adv.split("AREA[Phase]")[1].split(" ")[0],
                    "phases": (p.get("designModule") or {}).get("phases"),
                    "status": stat.get("overallStatus"),
                    "why_stopped": stat.get("whyStopped"),
                    "start_date": (stat.get("startDateStruct") or {}).get("date"),
                    "sponsor_class": ((p.get("sponsorCollaboratorsModule") or {})
                                      .get("leadSponsor") or {}).get("class"),
                }
            )
        token = data.get("nextPageToken")
        if not token:
            break
    return rows


def harvest_stopped(http: Http) -> list[dict]:
    if OUT_RAW.exists():
        rows = [json.loads(l) for l in OUT_RAW.open() if l.strip()]
        print(f"[resume] {OUT_RAW} 에서 {len(rows)}건 로드 (재수집 안 함)")
        return rows
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ph in PHASES:
        adv = f"AREA[Phase]{ph} AND {BASE}"
        for st in ("TERMINATED", "WITHDRAWN"):
            got = page_all(http, adv, st)
            print(f"  [census] {ph}/{st}: {len(got)}건")
            rows += got
    with OUT_RAW.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{OUT_RAW} 저장 — {len(rows)}건")
    return rows


# --- 3. 분류 집계 ----------------------------------------------------------


def classify_block(rows: list[dict]) -> dict:
    """규칙 기반 1차 패스 전수 적용. A비율의 상·하한을 함께 낸다."""
    cats: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    empty = unmatched = 0
    for r in rows:
        cat, reason, _ = classify(r.get("why_stopped"))
        cats[cat] += 1
        reasons[f"{cat}/{reason}"] += 1
        if cat == "?":
            if reason == "empty":
                empty += 1
            else:
                unmatched += 1
    n = len(rows)
    stated = n - empty                      # 사유가 기재된 건
    matched = stated - unmatched            # 규칙이 잡은 건
    a = cats.get("A", 0)
    return {
        "n": n,
        "n_reason_empty": empty,
        "n_reason_stated": stated,
        "n_rule_matched": matched,
        "n_rule_unmatched": unmatched,
        "rule_coverage_of_stated": round(matched / stated, 4) if stated else None,
        "by_category": {c: cats.get(c, 0) for c in ("A", "B", "C", "D", "?")},
        # A비율의 세 가지 분모 — 어느 분모를 쓰는지가 해석을 완전히 바꾼다
        "share_A_of_all": round(a / n, 4) if n else None,
        "share_A_of_stated": round(a / stated, 4) if stated else None,
        "share_A_of_matched": round(a / matched, 4) if matched else None,
        # 미분류를 전부 A가 아니라고 보는 하한 / 전부 A라고 보는 상한
        "share_A_of_stated_bounds": [
            round(a / stated, 4) if stated else None,
            round((a + unmatched) / stated, 4) if stated else None,
        ],
        "top_reasons": reasons.most_common(12),
    }


def main() -> None:
    http = Http()
    print(f"창: 개시 {START_FROM} ~ {START_TO}, INTERVENTIONAL\n")

    print("[1] Phase × 상태 전수 카운트 (countTotal 쿼리)")
    psc = phase_status_counts(http)
    print(f"  {'Phase':<12}{'전체':>8}{'완료':>8}{'중단':>8}{'철회':>8}{'확정계':>9}"
          f"{'중단율':>9}{'철회율':>9}")
    for ph, row in psc.items():
        det = sum(row[s] or 0 for s in DETERMINED)
        t, w = row["TERMINATED"] or 0, row["WITHDRAWN"] or 0
        print(f"  {ph:<12}{row['all']:>8}{row['COMPLETED']:>8}{t:>8}{w:>8}{det:>9}"
              f"{t/det:>9.2%}{w/det:>9.2%}")

    print("\n[2] 중단·철회 전수 수집 후 whyStopped 4분류")
    rows = harvest_stopped(http)

    blocks: dict = {}
    for ph in PHASES:
        sub = [r for r in rows if r["phase_filter"] == ph]
        blocks[ph] = {
            "ALL_STOPPED": classify_block(sub),
            "TERMINATED": classify_block([r for r in sub if r["status"] == "TERMINATED"]),
            "WITHDRAWN": classify_block([r for r in sub if r["status"] == "WITHDRAWN"]),
        }

    for ph in PHASES:
        b = blocks[ph]["TERMINATED"]
        print(f"\n  {ph} TERMINATED n={b['n']} (사유 기재 {b['n_reason_stated']}, "
              f"규칙 커버리지 {b['rule_coverage_of_stated']:.1%})")
        for c in ("A", "B", "C", "D", "?"):
            k = b["by_category"][c]
            print(f"    {c} {CATEGORY_NAMES[c]:<10} {k:>5}건  전체대비 {k/b['n']:6.2%}  "
                  f"기재분대비 {k/b['n_reason_stated']:6.2%}")
        lo, hi = b["share_A_of_stated_bounds"]
        print(f"    → A비율 (사유 기재분 기준) 하한 {lo:.2%} ~ 상한 {hi:.2%}")

    print("\n[3] 모수 A분류 유병률 — PPV 계산에 넣을 값")
    prev: dict = {}
    for ph in PHASES:
        row = psc[ph]
        det = sum(row[s] or 0 for s in DETERMINED)
        bt = blocks[ph]["TERMINATED"]
        bw = blocks[ph]["WITHDRAWN"]
        # 카운트 쿼리의 총건수와 전수 수집 건수가 어긋나면 즉시 드러나야 한다
        a_t, a_w = bt["by_category"]["A"], bw["by_category"]["A"]
        prev[ph] = {
            "n_determined": det,
            "n_terminated": row["TERMINATED"],
            "n_withdrawn": row["WITHDRAWN"],
            "n_harvested_terminated": bt["n"],
            "n_harvested_withdrawn": bw["n"],
            "n_A_terminated": a_t,
            "n_A_withdrawn": a_w,
            # 주 정의: A분류 TERMINATED / 결과 확정 전체 (코퍼스 라벨 정의와 일치)
            "prevalence_A_terminated": round(a_t / det, 5),
            # 철회까지 포함한 넓은 정의
            "prevalence_A_incl_withdrawn": round((a_t + a_w) / det, 5),
            # 미분류가 전부 A였다면의 상한
            "prevalence_A_terminated_upper": round(
                (a_t + bt["n_rule_unmatched"]) / det, 5),
        }
        p = prev[ph]
        print(f"  {ph}: A중단 {a_t}건 / 확정 {det}건 = {p['prevalence_A_terminated']:.3%}"
              f"  (철회포함 {p['prevalence_A_incl_withdrawn']:.3%},"
              f" 상한 {p['prevalence_A_terminated_upper']:.3%})")

    print("\n[4] 적응증 15개 모수 구성비 vs 코퍼스 균일수집")
    ind = indication_counts(http)
    print(f"  {'적응증':<34}{'Phase2 전체':>12}{'완료':>8}{'중단':>8}{'철회':>8}{'중단율':>9}")
    for b, row in ind.items():
        det = sum(row[s] or 0 for s in DETERMINED)
        print(f"  {b:<34}{row['all']:>12}{row['COMPLETED']:>8}{row['TERMINATED']:>8}"
              f"{row['WITHDRAWN']:>8}{(row['TERMINATED'] or 0)/det:>9.2%}" if det else
              f"  {b:<34}{row['all']:>12}")

    print("\n[5] Phase 2 중단군 하위분해 — 바스켓·스폰서 유형별 A비율")
    p2t = [r for r in rows if r["phase_filter"] == "PHASE2" and r["status"] == "TERMINATED"]
    subsets: dict[str, list[dict]] = {
        "phase2_pure": [r for r in p2t if set(r["phases"] or []) == {"PHASE2"}],
        "phase1_2_basket": [r for r in p2t if "PHASE1" in (r["phases"] or [])],
        "phase2_3_basket": [r for r in p2t if "PHASE3" in (r["phases"] or [])],
    }
    for sc in ("INDUSTRY", "OTHER", "NIH", "OTHER_GOV", "NETWORK"):
        sel = [r for r in p2t if r["sponsor_class"] == sc]
        if len(sel) >= 50:
            subsets[f"sponsor_{sc}"] = sel
    breakdown = {k: classify_block(v) for k, v in subsets.items()}
    print(f"  {'하위집단':<22}{'n':>7}{'사유기재':>9}{'A건수':>7}{'A/기재분':>10}")
    for k, b in breakdown.items():
        print(f"  {k:<22}{b['n']:>7}{b['n_reason_stated']:>9}"
              f"{b['by_category']['A']:>7}{b['share_A_of_stated']:>10.2%}")

    out = {
        "_meta": {
            "window": [START_FROM, START_TO],
            "study_type": "INTERVENTIONAL",
            "phase_filter_note": (
                "AREA[Phase]PHASE2 는 PHASE1|PHASE2, PHASE2|PHASE3 바스켓을 포함한다. "
                "코퍼스(ctgov_phase2.jsonl)와 동일한 필터이므로 비교 가능하다. "
                "PHASE2_PURE 는 바스켓을 제외한 대조값."
            ),
            "classifier": "labels.taxonomy.classify (규칙 기반 1차 패스, LLM 미사용)",
            "census_not_sample": True,
            "generated_by": "src/analysis/census.py",
        },
        "phase_status_counts": psc,
        "why_stopped_blocks": blocks,
        "prevalence": prev,
        "indication_counts": ind,
        "phase2_terminated_breakdown": breakdown,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n{OUT_JSON} 저장")


if __name__ == "__main__":
    main()
