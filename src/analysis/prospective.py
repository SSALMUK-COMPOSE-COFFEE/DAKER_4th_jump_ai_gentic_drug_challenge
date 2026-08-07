"""전향 예측 사전등록 — 아직 결과가 존재하지 않는 시험에 예측을 걸고 해시로 봉인한다.

## 왜 이것이 필요한가

후향 백테스트(`history/0` 입력)는 누출을 구조적으로 막았지만, 심사자 입장에서 남는 의심이
하나 있다. **"결과를 이미 아는 사람이 만든 규칙 아닌가."** 코퍼스를 보며 피처를 고르고
임계값을 정한 것은 사실이므로, 이 의심은 후향 설계로는 원리적으로 해소되지 않는다.

전향 예측은 그 의심을 시간으로 해소한다. **지금 모집 중인 시험의 결과는 아직 세상에
존재하지 않는다.** 우리도, 우리가 쓴 LLM 의 사전학습 데이터도 알 수 없다. 예측을 걸고
그 목록을 해시로 봉인해 두면, 나중에 결과가 나왔을 때 **사후에 목록을 고치지 않았음**을
누구나 검산할 수 있다.

## 무엇을 예측하는가 — 개별이 아니라 군(群)이다

이 도구의 PPV 는 10.7% 다(`scale-and-value.md` §1.2). 경고받은 시험 10건 중 9건은
모집 실패로 중단되지 **않는다.** 따라서 개별 시험의 운명을 맞히겠다는 예측은 설계상
틀린 주장이고, 여기서 하지 않는다.

사전등록하는 가설은 **군 대비**다 — 경고군의 모집 실패 중단율이 비경고군보다 높은가.
이것이 트리아지 도구가 실제로 주장할 수 있는 유일한 형태의 예측이다.

## 봉인되는 것

1. **판정 규칙** — 어떤 피처가 위험 판정 자격을 갖는지(`RISK_ELIGIBLE`), 심각도 임계값
   로직, 그리고 참조 분포 자체. 규칙을 먼저 고정하지 않으면 예측 목록만 봉인해도
   사후에 "그때 규칙은 이랬다"고 바꿀 수 있다.
2. **예측 목록** — NCT ID → 경고/무경고 판정.
3. **검증 절차** — 언제, 무엇으로, 어떤 기준으로 성패를 가릴지. 이것을 미리 못 박지 않으면
   결과를 보고 유리한 기준을 고르게 된다.

## 실행

    .venv/bin/python src/analysis/prospective.py            # 수집 → 예측 → 봉인
    .venv/bin/python src/analysis/prospective.py --verify   # 저장된 파일의 해시 재검산

산출물: `data/prospective_commit.json`
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit import benchmarks as bm  # noqa: E402
from audit.pipeline import FEATURE_STATUS, RISK_ELIGIBLE, audit_features  # noqa: E402
from harvest.ctgov import INT, V2, Http, extract_v0  # noqa: E402

OUT = Path("data/prospective_commit.json")
CORPUS = Path("data/raw/ctgov_phase2.jsonl")

# 참조 분포가 존재하는 적응증. 코퍼스와 같아야 층화 비교가 성립한다 (`harvest/expand.py`).
BUCKETS = [
    "non-small cell lung cancer", "atopic dermatitis", "type 2 diabetes",
    "rheumatoid arthritis", "alzheimer disease", "pancreatic cancer", "glioblastoma",
    "amyotrophic lateral sclerosis", "idiopathic pulmonary fibrosis",
    "systemic lupus erythematosus", "sickle cell disease", "cystic fibrosis",
    "acute myeloid leukemia", "ovarian cancer", "heart failure",
]

# --- 사전등록 포함/제외 기준 (예측 실행 전에 고정한다) ----------------------
#
# ⚠ 이 필터를 **개시일(StartDate)** 이 아니라 **최초 등록일(StudyFirstPostDate)** 로
# 잡은 것이 이 설계의 핵심이다. 첫 시도에서 개시일로 걸었더니 NCT03362606 처럼
# **최초 완료예정이 2019년인데 아직도 모집 중인** 시험이 표본에 들어왔다. 개시일까지
# 개정해 미뤄 온 시험들이다. 그런 시험의 현재 등록본은 목표 등록수·기간이 이미 모집 부진에
# 맞춰 조정된 뒤라, 그것으로 "예측"을 하면 **이미 일어난 결과를 관측하는 것**이 된다.
# 최초 등록일이 최근인 시험만 받고, 입력도 최초 등록본(history/0)만 쓴다.
FIRST_POST_FROM = "2025-01-01"   # 최초 등록이 최근일수록 개정 누적이 적다
FIRST_POST_TO = "MAX"
PER_BUCKET = 40             # 적응증당 상한. 대형 적응증이 표본을 지배하지 않게 한다
REQUIRED_STATUS = "RECRUITING"
MAX_AMENDMENTS = 6          # 개정이 이보다 많으면 최초 계획이 이미 폐기된 것으로 보고 제외

SEARCH_FIELDS = ",".join([
    "NCTId", "BriefTitle", "OverallStatus", "WhyStopped", "Phase", "Condition",
    "StartDate", "StudyFirstPostDate", "PrimaryCompletionDate", "EnrollmentCount",
    "EnrollmentType", "LeadSponsorClass",
])


def canonical(obj) -> str:
    """해시 대상 직렬화. 키 순서·공백이 실행마다 같아야 재검산이 성립한다."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 수집

def fetch_v0(http: Http, nct: str) -> tuple[dict | None, int]:
    """최초 등록본(history/0)과 개정 횟수. 후향 백테스트와 **동일한 입력 정의**다.

    현재 등록본을 쓰면 안 되는 이유는 `FIRST_POST_FROM` 주석 참조 — 모집이 부진하면
    목표 등록수를 낮추거나 기간을 늘리는데, 그 조정이 그대로 우리 피처(월 모집부담)를
    끌어내려 "예측"이 결과의 그림자가 된다.
    """
    hist = http.get_json(f"{INT}/studies/{nct}/history")
    if not hist or not hist.get("changes"):
        return None, 0
    n_amend = len(hist["changes"]) - 1
    v0 = http.get_json(f"{INT}/studies/{nct}/history/0")
    if not v0:
        return None, n_amend
    try:
        return extract_v0(v0["study"]["protocolSection"]), n_amend
    except (KeyError, TypeError):
        return None, n_amend


def collect(http: Http) -> tuple[list[dict], list[dict]]:
    """현재 모집 중이면서 최근에 처음 등록된 Phase 2 시험을 모은다."""
    rows: list[dict] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    for bucket in BUCKETS:
        got, token, cand = 0, None, []
        while len(cand) < PER_BUCKET:
            params = {
                "query.cond": bucket,
                "filter.overallStatus": REQUIRED_STATUS,
                "filter.advanced": (
                    f"AREA[Phase]PHASE2 AND AREA[StudyType]INTERVENTIONAL "
                    f"AND AREA[StudyFirstPostDate]RANGE[{FIRST_POST_FROM},{FIRST_POST_TO}]"
                ),
                "fields": SEARCH_FIELDS,
                "pageSize": min(100, PER_BUCKET - len(cand)),
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
                proto = s.get("protocolSection") or {}
                nct = (proto.get("identificationModule") or {}).get("nctId")
                st = proto.get("statusModule") or {}
                if not nct or nct in seen:
                    continue
                # 사전등록 무결성: 예측 시점에 결과가 존재하지 않아야 한다
                if st.get("overallStatus") != REQUIRED_STATUS or st.get("whyStopped"):
                    continue
                seen.add(nct)
                cand.append((nct, (st.get("studyFirstPostDateStruct") or {}).get("date")))
            token = data.get("nextPageToken")
            if not token:
                break

        for nct, first_post in cand:
            v0, n_amend = fetch_v0(http, nct)
            if v0 is None:
                dropped.append({"nct_id": nct, "reason": "최초 등록본 조회 실패"})
                continue
            if n_amend > MAX_AMENDMENTS:
                dropped.append({"nct_id": nct, "reason": f"개정 {n_amend}회 (>{MAX_AMENDMENTS})"})
                continue
            rows.append({
                "nct_id": nct, "condition_bucket": bucket, "v0": v0,
                "first_posted": first_post, "n_amendments": n_amend,
            })
            got += 1
        print(f"  {bucket:36} {got:>3}건 (후보 {len(cand)})")
    return rows, dropped


# ---------------------------------------------------------------- 예측

def predict(rows: list[dict], bench: dict) -> tuple[list[dict], list[dict]]:
    """감사를 돌려 경고/무경고를 가른다. 판정 불가 건은 분리해 따로 기록한다."""
    preds, excluded = [], []
    for r in rows:
        v0, nct = r["v0"], r["nct_id"]
        feats = bm.extract({"v0": v0, "nct_id": nct})
        burden = feats.get("monthly_enrollment_burden")
        if burden is None:
            excluded.append({"nct_id": nct, "reason": "월 모집부담 계산 불가(등록수 또는 완료예정일 누락)"})
            continue
        if not bm.reference(bench, r["condition_bucket"], "monthly_enrollment_burden", v0):
            excluded.append({"nct_id": nct, "reason": "참조 분포 없음"})
            continue

        findings, _ctx = audit_features(v0, r["condition_bucket"], bench, nct)
        risky = [f for f in findings if f.severity in ("high", "medium")]
        preds.append({
            "nct_id": nct,
            "bucket": r["condition_bucket"],
            "phase_stratum": bm.phase_stratum(v0),
            "start_date": v0.get("start_date"),
            "planned_primary_completion": v0.get("planned_primary_completion"),
            "first_posted": r.get("first_posted"),
            "n_amendments": r.get("n_amendments"),
            "monthly_enrollment_burden": burden,
            "verdict": "WARN" if risky else "CLEAR",
            "severity": max((f.severity for f in risky), default=None),
            "codes": sorted(f.code for f in risky),
        })
    preds.sort(key=lambda p: p["nct_id"])
    excluded.sort(key=lambda e: e["nct_id"])
    return preds, excluded


def freeze_rule(bench: dict) -> dict:
    """판정 규칙을 스냅샷으로 뜬다. 예측보다 **먼저** 고정되어야 의미가 있다."""
    src = Path("src/audit/pipeline.py").read_bytes()
    return {
        "risk_eligible": sorted(RISK_ELIGIBLE),
        "feature_status": {k: v[0] for k, v in sorted(FEATURE_STATUS.items())},
        "severity_rule": (
            "관측값이 완주군 참조분포의 p25 미만(또는 higher_is_riskier 피처는 p75 초과)이면 "
            "medium, 실패군 중앙값을 넘어서면 high. 임계값은 참조 분포의 분위수에서 나오며 "
            "참조 분포는 아래 해시로 고정된다."
        ),
        "warn_definition": "위험 판정(severity ∈ {high, medium}) 이 1건 이상이면 WARN",
        "pipeline_sha256": hashlib.sha256(src).hexdigest(),
        "benchmarks_sha256": sha256(bench),
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "corpus_note": "2010-2020 개시 Phase 2 코퍼스 1,463건. 전향 예측 대상과 시간축이 겹치지 않는다.",
    }


def commitment() -> dict:
    """검증 절차를 미리 못 박는다. 결과를 보고 기준을 고르는 것을 막는 장치다."""
    return {
        "primary_hypothesis": (
            "H1: WARN 군의 '모집 실패 중단'(라벨 A) 발생률이 CLEAR 군보다 높다. "
            "개별 시험의 결과를 예측하지 않는다 — 모수 PPV 10.7% 이므로 개별 예측은 "
            "설계상 대부분 빗나가며, 그것이 이 도구의 정상 동작이다."
        ),
        "effect_measure": "위험비(RR) = P(A분류 중단 | WARN) / P(A분류 중단 | CLEAR)",
        "success_criterion": "RR 의 95% CI 하한 > 1.0 이면 가설 지지, 1.0 을 포함하면 미지지",
        "prespecified_null_result": (
            "RR ≤ 1 또는 CI 가 1 을 포함하면 '전향 검증 실패'로 그대로 보고한다. "
            "사후에 하위군을 나누거나 기준을 바꾸지 않는다."
        ),
        "outcome_ascertainment": (
            "CT.gov v2 의 overallStatus 와 whyStopped 를 받아 "
            "src/labels/taxonomy.py 의 classify() 로 분류한다. 분류기도 이 커밋 시점의 "
            "버전을 쓴다(pipeline_sha256 과 동일 저장소 상태)."
        ),
        "checkpoints": [
            {"date": "2027-08-07", "purpose": "1차 중간 판정 — 이 시점까지 종료된 건만으로 산출"},
            {"date": "2029-08-07", "purpose": "최종 판정 — 대다수 시험의 계획 완료일 경과 후"},
        ],
        "censoring": (
            "판정 시점에 아직 RECRUITING/ACTIVE 인 시험은 중도절단으로 처리하고 분모에서 뺀다. "
            "종료됐으나 사유 미기재인 건은 '분류 불가'로 별도 보고한다."
        ),
        "falsifiability": (
            "이 커밋의 해시는 제안서에 인쇄된다. 목록·규칙·기준을 사후에 바꾸면 해시가 달라지므로 "
            "누구나 검산으로 적발할 수 있다."
        ),
        "known_biases": [
            {
                "name": "생존 편향 — RECRUITING 만 뽑은 데서 오는 구조적 편향",
                "detail": (
                    "예측이 성립하려면 결과가 아직 없어야 하므로 모집 중인 시험만 대상이 된다. "
                    "그런데 모집이 빠른 시험은 이미 등록을 마치고 ACTIVE_NOT_RECRUITING 으로 "
                    "넘어가 표본에서 빠진다. 따라서 이 표본은 모집이 느린 쪽으로 기울어 있고, "
                    "실제로 경고율이 모수 유병률로 계산한 기대치(약 31%)보다 높게 나온다."
                ),
                "why_it_does_not_invalidate_h1": (
                    "H1 은 절대 경고율이 아니라 **WARN 군과 CLEAR 군의 대비**를 본다. 표본 전체가 "
                    "한쪽으로 기울어도 두 군 사이의 위험비는 여전히 해석 가능하다. 다만 이 "
                    "표본에서 얻은 경고율을 모집단의 경고율로 일반화해서는 안 된다."
                ),
            },
            {
                "name": "시대 차이 — 참조 분포는 2010–2020 개시 코퍼스에서 나왔다",
                "detail": (
                    "예측 대상은 2025년 이후 최초 등록분이라 15년의 간극이 있다. 설계 관행이 "
                    "변했다면 임계값이 체계적으로 어긋날 수 있다. 참고로 전향 대상의 월 모집부담 "
                    "중앙값은 코퍼스 완주군과 가까웠고 모집실패군과는 뚜렷이 달랐다."
                ),
                "why_it_does_not_invalidate_h1": (
                    "참조 분포를 예측 대상으로 다시 추정하면 사전등록이 무너지므로 하지 않는다. "
                    "시대 차이는 검증 시점에 '이 도구가 2020년대 시험에도 통하는가'라는 질문의 "
                    "답으로 함께 보고한다."
                ),
            },
            {
                "name": "개정 누적 — 최초 등록본을 입력으로 쓴다",
                "detail": (
                    "모집이 부진하면 목표 등록수를 낮추거나 기간을 늘리므로, 현재 등록본을 쓰면 "
                    "이미 일어난 결과가 피처에 새어든다. 첫 설계에서 개시일 기준으로 표본을 "
                    "뽑았다가 최초 완료예정이 2019년인데 아직 모집 중인 시험이 섞여 들어온 것을 "
                    "발견하고, 최초 등록일 기준 + 최초 등록본(history/0) + 개정 상한으로 바꿨다."
                ),
                "why_it_does_not_invalidate_h1": (
                    "후향 백테스트와 입력 정의가 같아졌으므로 두 결과를 같은 잣대로 비교할 수 있다."
                ),
            },
        ],
    }


# ---------------------------------------------------------------- 실행

def build() -> None:
    print("[1/4] 참조 분포 (기존 코퍼스 1,463건) …")
    bench = bm.build()
    rule = freeze_rule(bench)
    print(f"      규칙 해시 {rule['pipeline_sha256'][:16]}… · 분포 해시 {rule['benchmarks_sha256'][:16]}…")

    print(f"[2/4] CT.gov 수집 — {REQUIRED_STATUS} · Phase 2 · 최초등록 {FIRST_POST_FROM} 이후 …")
    http = Http()
    rows, dropped = collect(http)
    print(f"      총 {len(rows)}건 (수집 단계 제외 {len(dropped)}건)")

    print("[3/4] 감사 실행 …")
    preds, excluded = predict(rows, bench)
    excluded = sorted(excluded + dropped, key=lambda e: e["nct_id"])
    warn = [p for p in preds if p["verdict"] == "WARN"]
    print(f"      판정 {len(preds)}건 (제외 {len(excluded)}건) · WARN {len(warn)}건 "
          f"({len(warn) / len(preds) * 100:.1f}%)" if preds else "      판정 0건")

    print("[4/4] 봉인 …")
    now = datetime.now(timezone.utc)
    payload = {
        "predictions": preds,
        "rule": rule,
        "commitment": commitment(),
    }
    doc = {
        "_meta": {
            "purpose": "전향 예측 사전등록 (JUMP AI 4th, 분야 3)",
            "committed_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_predictions": len(preds),
            "n_warn": len(warn),
            "n_clear": len(preds) - len(warn),
            "n_excluded": len(excluded),
            "integrity": (
                f"수집 시점 전 건이 overallStatus={REQUIRED_STATUS} 이고 whyStopped 가 비어 있음을 "
                "확인했다. 즉 예측 대상의 결과는 이 시각 현재 존재하지 않는다."
            ),
            "inclusion": {
                "status": REQUIRED_STATUS, "phase": "PHASE2", "study_type": "INTERVENTIONAL",
                "first_post_date_range": f"{FIRST_POST_FROM}..{FIRST_POST_TO}",
                "max_amendments": MAX_AMENDMENTS,
                "input_version": "history/0 (최초 등록본) — 후향 백테스트와 동일 정의",
                "buckets": BUCKETS, "per_bucket_cap": PER_BUCKET,
            },
            "how_to_verify": (
                "python src/analysis/prospective.py --verify — predictions/rule/commitment 세 블록을 "
                "canonical JSON 으로 직렬화해 SHA-256 을 다시 계산하고 아래 값과 대조한다."
            ),
        },
        "commit_sha256": sha256(payload),
        "prediction_sha256": sha256(preds),
        "excluded": excluded,
        **payload,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1))

    print(f"\n      → {OUT}")
    print(f"      커밋 해시 (제안서에 인쇄할 값)\n        {doc['commit_sha256']}")
    print(f"      예측목록 해시\n        {doc['prediction_sha256']}")
    _summary(preds)


def _summary(preds: list[dict]) -> None:
    if not preds:
        return
    from collections import Counter
    by_bucket = Counter(p["bucket"] for p in preds if p["verdict"] == "WARN")
    tot = Counter(p["bucket"] for p in preds)
    print("\n  적응증별 경고율")
    for b, n in sorted(tot.items(), key=lambda x: -x[1]):
        w = by_bucket.get(b, 0)
        print(f"    {b:36} {w:>3}/{n:<3} ({w / n * 100:>4.0f}%)")


def verify() -> None:
    if not OUT.exists():
        raise SystemExit(f"{OUT} 없음")
    doc = json.loads(OUT.read_text())
    payload = {k: doc[k] for k in ("predictions", "rule", "commitment")}
    ok_commit = sha256(payload) == doc["commit_sha256"]
    ok_pred = sha256(doc["predictions"]) == doc["prediction_sha256"]
    print(f"  커밋 해시   {'✓ 일치' if ok_commit else '✗ 불일치'}  {doc['commit_sha256']}")
    print(f"  예측 해시   {'✓ 일치' if ok_pred else '✗ 불일치'}  {doc['prediction_sha256']}")
    m = doc["_meta"]
    print(f"  봉인 시각   {m['committed_at_utc']}")
    print(f"  예측 {m['n_predictions']}건 — WARN {m['n_warn']} / CLEAR {m['n_clear']}")
    if not (ok_commit and ok_pred):
        raise SystemExit(1)


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else build()
