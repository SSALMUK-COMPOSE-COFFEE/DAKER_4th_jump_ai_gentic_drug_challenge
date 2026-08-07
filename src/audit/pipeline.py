"""Tier 1 계량 감사 파이프라인.

프로토콜(v0 스냅샷)을 받아 결정론적 감사 findings 를 만든다. LLM을 쓰지 않으므로
비용이 0이고, 같은 입력에 항상 같은 결과가 나온다 — 심사에서 재현 가능성을 주장할 수 있다.

각 finding 은 반드시 다음을 갖는다:
  - 수치 근거 (관측값 + 완주군/실패군 참조 분포)
  - 판정 방향과 심각도
  - 수정 가능한 프로토콜 요소를 겨냥한 제안

스폰서 유형은 **근거로 쓰지 않는다.** 모집실패군 학술(OTHER) 74.4% vs 완주군 산업 50.1% 로 강하게 갈리지만,
그걸 근거로 삼으면 "학술 스폰서니까 실패한다"는 교정 불가능한 조언이 되고 프로토콜 감사가 아니라
스폰서 신분 판별기가 된다. design-v2.md 6절 참조.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.amendment_confound import parse_date  # noqa: E402
from audit import benchmarks as bm  # noqa: E402
from audit.competition import competition_at, fetch_indication_pool  # noqa: E402
from harvest.ctgov import Http, fetch_record  # noqa: E402

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}

# 완주군 중앙값에서 이 비율 이상 벗어나야 finding 으로 올린다.
# 3명/월 vs 3.09명/월 같은 무의미한 차이를 경고로 띄우면 신뢰를 잃는다.
MIN_RELATIVE_MARGIN = 0.15


def josa(word: str, pair: str) -> str:
    """한글 조사 선택. pair 는 '이/가', '을/를', '은/는' 형식.

    받침 유무로 결정한다. 숫자·영문으로 끝나는 경우가 있어 한글이 아니면 앞쪽(받침 있음)을 쓴다.
    """
    with_final, without_final = pair.split("/")
    if not word:
        return without_final
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_final if (ord(ch) - 0xAC00) % 28 else without_final
    if ch.isdigit():
        # 0·1·3·6·7·8 은 받침 있음 (영/일/삼/육/칠/팔)
        return with_final if ch in "013678" else without_final
    return with_final

# 피처별 사람이 읽을 이름과 단위
LABELS = {
    "n_concurrent_phase2": ("개시 시점 동시 모집 Phase 2 시험 수", "건"),
    "n_concurrent_trials": ("개시 시점 동시 모집 전체 시험 수", "건"),
    "total_competing_seats": ("경쟁 시험 목표 등록수 합계", "명"),
    "n_inclusion_items": ("포함기준 항목수", "개"),
    "n_exclusion_items": ("제외기준 항목수", "개"),
    "n_eligibility_items": ("적격기준 총항목", "개"),
    "eligibility_chars": ("적격기준 분량", "자"),
    "planned_enrollment": ("목표 등록수", "명"),
    "planned_duration_months": ("계획 기간", "개월"),
    "monthly_enrollment_burden": ("월 모집 목표", "명/월"),
    "n_secondary_outcomes": ("2차 평가변수 수", "개"),
}


# 피처별 검증 상태 — `data/validation.json` 의 recruit_vs_completed / indication 층 기준.
# (생성: src/analysis/stratified_signal.py, 문서: docs/memory/validated-numbers.md)
#
# **위험 판정(high/medium)을 낼 수 있는 것은 `validated` 뿐이다.** 나머지는 맥락 정보로만 제시한다.
# 이 구분을 코드에 박아두는 이유: 검증에서 죽은 피처가 계속 HIGH 를 발화하고 있었고,
# 그중 하나는 방향이 반대라서 **거꾸로 된 교정안**을 출력하고 있었다. design-v2.md 2.1 이
# 경계한 실패 모드가 실제로 발생한 것이다.
FEATURE_STATUS: dict[str, tuple[str, str]] = {
    "monthly_enrollment_burden": (
        "validated",
        "적응증 층화 AUC 0.632 [0.559, 0.701]. 층화 5축 전부에서 CI 하한 > 0.5 로 유지된 "
        "유일한 피처. 단 방향이 역설적이다 — 실패군이 부담이 더 낮다(1.54 vs 3.37명/월).",
    ),
    "planned_duration_months": (
        "nonspecific",
        "층화 후에도 살아남지만(0.581) 세 집단 모두에 발화한다 "
        "(모집실패 0.581 / 효능실패 0.591 / 경영중단 0.571). '문제 있는 시험' 일반 지표이며 "
        "모집 실패에 특화되지 않았다. 월 모집부담의 구성요소로만 사용한다.",
    ),
    "n_secondary_outcomes": (
        "reversed",
        "적응증 층화 AUC 0.454 — **방향이 반대다.** 완주군·모집실패군 중앙값이 모두 4개이고 AUC 가 0.5 미만으로 "
        "2차 평가변수가 많은 쪽이 오히려 완주한다. 줄이라는 조언은 데이터와 반대 방향이다.",
    ),
    "n_concurrent_phase2": (
        "retracted",
        "적응증 층화 AUC 0.472 (우연 이하). 프로토콜과 날짜를 버리고 적응증 라벨만 쓰는 "
        "음성 대조군이 0.722 로 더 높다 — 이 피처는 '어느 질환인가'의 재인코딩이었다. "
        "게다가 프로토콜로 변경 불가능하다. validated-numbers.md 3.1",
    ),
    "n_concurrent_trials": ("retracted", "적응증 층화 AUC 0.515. 위와 같은 적응증 교란."),
    "total_competing_seats": ("retracted", "적응증 층화 AUC 0.515. 위와 같은 적응증 교란."),
    "n_inclusion_items": (
        "unvalidated",
        "적응증 층화 AUC 0.555 [0.493, 0.621] 로 구분력 미입증. **코호트 정의에 따라 판정이 뒤집힌다** — A 우선 정의에서는 0.570 [0.513] 으로 통과한다 (cohort_sensitivity.py). 정의 의존적이므로 채택하지 않는다. 분포 차이 자체는 실측이므로 "
        "(실패군 13.5개 vs 완주군 9개) 참고 정보로 제시하되 위험 판정에는 쓰지 않는다.",
    ),
    "n_exclusion_items": ("unvalidated", "적응증 층화 AUC 0.498. 구분력 미입증 (0.5 미만)."),
    "n_eligibility_items": ("unvalidated", "적응증 층화 AUC 0.525. 구분력 미입증."),
    "eligibility_chars": ("unvalidated", "적응증 층화 AUC 0.508. 구분력 미입증."),
    "planned_enrollment": (
        "unvalidated",
        "적응증 층화 0.582 로 CI 하한이 0.5 를 넘지만 스폰서 층에서 무너져 5축 전부를 통과하지 못했다. "
        "방향도 주의 — 효능실패군에서 0.348 로 강한 역방향이다.",
    ),
}

# 위험 판정(high/medium)을 낼 자격이 있는 피처.
RISK_ELIGIBLE = {k for k, (status, _) in FEATURE_STATUS.items() if status == "validated"}


@dataclass
class Finding:
    code: str
    severity: str
    title: str
    observed: float | None
    ref_completed: float | None
    ref_failed: float | None
    bucket_used: str | None
    evidence: str
    suggestion: str
    validation_status: str = "validated"
    validation_note: str = ""


@dataclass
class AuditReport:
    nct_id: str | None
    bucket: str
    title: str | None
    start_date: str | None
    findings: list[Finding] = field(default_factory=list)
    # 검증되지 않은 피처의 관측값. 위험 판정이 아니라 맥락 정보이며 점수에 들어가지 않는다.
    context: list[Finding] = field(default_factory=list)
    competition: dict | None = None
    tier1_only: bool = True
    notes: list[str] = field(default_factory=list)

    def risk_score(self) -> int:
        """심각도 가중 합 (0~100). **내부 진단·평가용이며 위험 확률이 아니다.**

        holdout 검증에서 이 점수의 구분력이 입증되지 않았다
        (모집실패 vs 완주 AUC 0.587, 95% CI [0.484, 0.693] — 0.5 포함).
        따라서 사용자에게 "위험도 74점" 처럼 제시하면 캘리브레이션된 확률이라는 잘못된 인상을 준다.
        `to_dict()` 는 이 값을 `diagnostic_score` 로 내보내고 미검증 사실을 함께 붙인다.
        """
        w = {"high": 25, "medium": 12, "low": 5, "info": 0}
        raw = sum(w[f.severity] for f in self.findings)
        return min(100, raw)

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: SEVERITY_ORDER[f.severity])

    def to_dict(self) -> dict:
        return {
            "nct_id": self.nct_id,
            "bucket": self.bucket,
            "title": self.title,
            "start_date": self.start_date,
            "diagnostic_score": self.risk_score(),
            "score_calibration": {
                "calibrated": False,
                "holdout_auc": 0.587,
                "holdout_ci95": [0.484, 0.693],
                "note": (
                    "심각도 가중 합이며 위험 확률이 아니다. holdout 신뢰구간이 0.5를 포함해 "
                    "구분력이 입증되지 않았다. 개별 finding 의 수치 근거를 보고 판단할 것."
                ),
            },
            "tier1_only": self.tier1_only,
            "competition": self.competition,
            "findings": [asdict(f) for f in self.sorted_findings()],
            "context": [asdict(f) for f in self.context],
            "notes": self.notes,
        }


def _severity(observed: float, comp_q: dict, fail_q: dict | None, higher_riskier: bool) -> str:
    """완주군 분포에서 얼마나 극단에 있는지로 심각도를 정한다.

    처음 구현은 **완주군 중앙값 초과**를 위험 신호로 삼았는데 이게 설계 오류였다.
    정의상 어느 집단이든 절반이 중앙값을 넘으므로, 피처 8개를 독립적으로 판정하면
    대부분의 프로토콜이 여러 개를 발동시킨다. 실측에서 완주군 감사 점수 중앙값이 49.5,
    임계값 50에서 완주군 오탐률 50% 가 나왔다 (`src/analysis/calibration.py`).

    그래서 기준선을 분위수로 올렸다:
      - high   : 완주군 p90 를 넘고 **동시에** 모집실패군 중앙값도 넘음
      - medium : 완주군 p75 를 넘음
      - info   : 그 외

    두 조건을 AND 로 묶는 이유는, 완주군에서 드문 값이라도 실패군에서도 드물면
    실패의 표지가 아니기 때문이다.
    """
    if not comp_q:
        return "info"
    p75, p90 = comp_q.get("p75"), comp_q.get("p90")
    fail_med = (fail_q or {}).get("median")
    if p75 is None or p90 is None:
        return "info"

    if higher_riskier:
        beyond_p90 = observed > p90
        beyond_fail = fail_med is not None and observed >= fail_med
        if beyond_p90 and beyond_fail:
            return "high"
        if observed > p75:
            return "medium"
        return "info"

    # 낮을수록 위험한 피처는 하위 분위수를 쓴다
    p25 = comp_q.get("p25")
    if p25 is None:
        return "info"
    below_p25 = observed < p25
    beyond_fail = fail_med is not None and observed <= fail_med
    if below_p25 and beyond_fail:
        return "high"
    if observed < comp_q["median"] and abs(observed - comp_q["median"]) / abs(
        comp_q["median"] or 1
    ) >= MIN_RELATIVE_MARGIN:
        return "medium"
    return "info"


def features_from_v0(v0: dict, nct_id: str | None = None) -> dict[str, float | None]:
    """benchmarks.extract 와 동일한 정의를 써야 비교가 성립한다."""
    return bm.extract({"v0": v0, "nct_id": nct_id})


# 서로 강하게 상관된 피처 묶음. 묶음 안에서는 가장 심각한 것 하나만 finding 으로 올린다.
# 적격기준 항목수·총항목·분량은 사실상 같은 특성(적격기준이 촘촘한가)을 세 번 재고 있어서,
# 독립 가산하면 한 가지 문제로 점수가 75점 오른다. 캘리브레이션에서 점수 포화의 주 원인이었다.
FEATURE_GROUPS: dict[str, str] = {
    # 경쟁 밀도 3종도 같은 것을 재므로 묶는다. 그중 구분력이 가장 높은 것이 대표로 뽑힌다.
    "n_concurrent_phase2": "competition",
    "n_concurrent_trials": "competition",
    "total_competing_seats": "competition",
    "n_inclusion_items": "eligibility",
    "n_exclusion_items": "eligibility",
    "n_eligibility_items": "eligibility",
    "eligibility_chars": "eligibility",
    # 목표 등록수와 월 모집부담은 분모만 다른 같은 계획을 본다
    "planned_enrollment": "enrollment_plan",
    "monthly_enrollment_burden": "enrollment_plan",
}


def _dedupe_correlated(findings: list[Finding]) -> list[Finding]:
    """상관 묶음마다 가장 심각한 finding 하나만 남긴다. 묶음 밖 피처는 그대로 통과."""
    best: dict[str, Finding] = {}
    passthrough: list[Finding] = []
    for f in findings:
        key = f.code.removeprefix("feature.")
        group = FEATURE_GROUPS.get(key)
        if group is None:
            passthrough.append(f)
            continue
        cur = best.get(group)
        if cur is None or SEVERITY_ORDER[f.severity] < SEVERITY_ORDER[cur.severity]:
            best[group] = f
    return passthrough + list(best.values())


def audit_features(
    v0: dict, bucket: str, bench: dict, nct_id: str | None = None
) -> tuple[list[Finding], list[Finding]]:
    """(위험 판정 findings, 맥락 context) 를 분리해서 돌려준다.

    `RISK_ELIGIBLE` 에 없는 피처는 관측값이 아무리 극단이어도 위험 판정을 내지 않는다.
    적응증 층화에서 구분력이 입증되지 않았거나(unvalidated), 철회됐거나(retracted),
    방향이 반대(reversed)이기 때문이다. 그런 피처로 심각도를 매기면 근거 없는 경고가 되고,
    `reversed` 의 경우 **데이터와 반대 방향의 조언**을 출력한다.
    """
    feats = features_from_v0(v0, nct_id)
    findings: list[Finding] = []
    context: list[Finding] = []

    for key, observed in feats.items():
        if observed is None:
            continue
        ref = bm.reference(bench, bucket, key, v0)
        if not ref:
            continue
        comp_q = ref["completed"]
        fail_q = ref["recruit_failed"]
        comp = comp_q["median"]
        fail = (fail_q or {}).get("median")
        higher = key in bm.HIGHER_IS_RISKIER
        sev = _severity(observed, comp_q, fail_q, higher)
        if sev == "info":
            continue

        status, note = FEATURE_STATUS.get(key, ("unvalidated", "검증 기록이 없다."))
        name, unit = LABELS.get(key, (key, ""))
        cmp_word = "많습니다" if higher else "낮습니다"
        evidence = (
            f"이 프로토콜의 {name}{josa(name, '은/는')} {observed:g}{unit}입니다. "
            f"같은 적응증 완주군 중앙값은 {comp:g}{unit}"
            + (f", 모집 실패군 중앙값은 {fail:g}{unit}입니다." if fail is not None else "입니다.")
        )

        if key in RISK_ELIGIBLE:
            if fail is not None and sev == "high":
                evidence += " 실패군 쪽에 서 있습니다."
            findings.append(
                Finding(
                    code=f"feature.{key}",
                    severity=sev,
                    title=f"{name}{josa(name, '이/가')} 완주군보다 {cmp_word}",
                    observed=round(float(observed), 2),
                    ref_completed=comp,
                    ref_failed=fail,
                    bucket_used=ref["bucket_used"],
                    evidence=evidence,
                    suggestion=_suggest(key, observed, comp),
                    validation_status=status,
                    validation_note=note,
                )
            )
            continue

        # 검증 미통과 — 관측값과 분포만 제시하고 판정·조언은 하지 않는다.
        context.append(
            Finding(
                code=f"context.{key}",
                severity="info",
                title=f"[참고] {name} {observed:g}{unit}",
                observed=round(float(observed), 2),
                ref_completed=comp,
                ref_failed=fail,
                bucket_used=ref["bucket_used"],
                evidence=evidence,
                suggestion=(
                    "이 항목은 위험 판정 근거로 쓰지 않습니다. "
                    + ("방향이 데이터와 반대이므로 조언을 생성하지 않습니다. " if status == "reversed" else "")
                    + f"({note})"
                ),
                validation_status=status,
                validation_note=note,
            )
        )

    return _dedupe_correlated(findings), context


def _suggest(key: str, observed: float, comp: float) -> str:
    if key in ("n_inclusion_items", "n_exclusion_items", "n_eligibility_items"):
        n = name_of(key)
        return (
            f"{n}{josa(n, '을/를')} 완주군 수준({comp:g}개)까지 줄이는 것을 검토하십시오. "
            "임상적으로 필수적이지 않은 수치 컷오프와 병용약 금지 항목이 우선 후보입니다."
        )
    if key == "eligibility_chars":
        return "적격기준 서술을 간결화하고, 해석 여지가 있는 조건을 명확한 판정 기준으로 대체하십시오."
    if key == "planned_duration_months":
        return (
            f"계획 기간 {observed:g}개월은 완주군({comp:g}개월)보다 깁니다. "
            "기간을 늘려 모집 부담을 낮추는 방식은 데이터상 실패군의 특징입니다. "
            "사이트 수를 늘려 기간을 단축하는 설계를 검토하십시오."
        )
    if key == "monthly_enrollment_burden":
        return (
            "월 모집 목표가 완주군보다 낮습니다. 여유로운 계획이 아니라, "
            "모집 가능 환자가 애초에 적다는 신호일 수 있습니다. 적격 환자 풀 추정치를 근거로 제시하십시오."
        )
    if key == "planned_enrollment":
        return "목표 등록수가 완주군보다 작습니다. 검정력 계산 근거를 명시하십시오."
    if key == "n_secondary_outcomes":
        return "2차 평가변수가 많으면 사이트 부담과 데이터 수집 비용이 커집니다. 핵심만 남기십시오."
    if key in ("n_concurrent_phase2", "n_concurrent_trials", "total_competing_seats"):
        # 경쟁 밀도 자체는 프로토콜로 바꿀 수 없다. 바꿀 수 있는 것은 어디서·누구를 모집하느냐다.
        #
        # 이 피처는 **철회되었다.** 한때 "holdout 에서 구분력이 확인된 유일한 신호"로 판정했으나,
        # 적응증으로 층화하면 AUC 0.472 (CI 0.41-0.53) 으로 우연 이하다. 적응증 라벨만 쓰는
        # 음성 대조군이 0.722 로 더 높아, 이 피처는 "어느 질환인가"의 재인코딩이었다.
        # 오판 원인은 음성 대조군을 만들지 않고 미층화 CI 하한이 0.5 를 넘는 것만 본 것이다.
        # validated-numbers.md 3.1 참조. 위험 판정 근거로 쓰지 말고 맥락 정보로만 제시한다.
        return (
            "경쟁 시험 수는 프로토콜로 바꿀 수 없으므로, 겹치지 않는 환자군을 확보하는 방향으로 대응하십시오. "
            "① 경쟁 시험이 제외하는 하위 집단(고령·동반질환·이전 치료 이력)을 오히려 포함, "
            "② 경쟁이 덜한 지역·국가로 사이트 확장, "
            "③ 동일 기전 경쟁 시험 대비 차별화 근거를 프로토콜에 명시. "
            "※ 이 항목은 적응증 층화 후 구분력이 사라져(AUC 0.472) 위험 판정 근거에서 제외되었습니다. "
            "참고 맥락으로만 보십시오."
        )
    return "해당 항목의 설정 근거를 프로토콜에 명시하십시오."


def name_of(key: str) -> str:
    return LABELS.get(key, (key, ""))[0]


def audit_competition(
    http: Http, bucket: str, nct_id: str | None, start: date | None, pool_cache: dict
) -> tuple[dict | None, Finding | None]:
    if not start:
        return None, None
    if bucket not in pool_cache:
        pool_cache[bucket] = fetch_indication_pool(http, bucket)
    stats = competition_at(pool_cache[bucket], nct_id or "", start)

    # 경쟁 밀도는 이제 참조 분포 기반 피처 감사(feature.n_concurrent_*)가 담당한다.
    # 여기서 다시 finding 을 만들면 같은 내용이 두 번 보고되고, 하드코딩 임계값(n>=100)은
    # 데이터 근거가 없다. 통계값만 리포트에 실어 화면에서 맥락으로 쓴다.
    return stats, None


def audit_v0(
    v0: dict,
    bucket: str,
    *,
    nct_id: str | None = None,
    bench: dict | None = None,
    http: Http | None = None,
    pool_cache: dict | None = None,
) -> AuditReport:
    bench = bench or bm.load()
    report = AuditReport(
        nct_id=nct_id,
        bucket=bucket,
        title=v0.get("brief_title"),
        start_date=v0.get("start_date"),
    )
    risk, ctx = audit_features(v0, bucket, bench, nct_id)
    report.findings.extend(risk)
    report.context.extend(ctx)

    if http is not None:
        stats, finding = audit_competition(
            http, bucket, nct_id, parse_date(v0.get("start_date")), pool_cache if pool_cache is not None else {}
        )
        report.competition = stats
        if finding:
            report.findings.append(finding)
    else:
        report.notes.append("경쟁 시험 밀도 감사를 건너뛰었습니다 (네트워크 미사용 모드).")

    if not v0.get("n_sites"):
        report.notes.append(
            "등록 시점 사이트 목록이 비어 있어 사이트당 모집 부담을 계산하지 못했습니다 "
            "(실측 결측률 33%)."
        )
    report.notes.append("Tier 1 결정론적 감사만 수행했습니다. LLM 호출 0회, 비용 $0.")
    return report


def audit_nct(nct_id: str, bucket: str, *, with_competition: bool = True) -> AuditReport:
    """실제 트라이얼의 v0 를 받아 감사한다. 백테스트 데모의 진입점."""
    http = Http(delay=0.2)
    rec = fetch_record(http, nct_id, bucket)
    if not rec:
        raise SystemExit(f"{nct_id} 이력을 가져오지 못했습니다.")
    report = audit_v0(
        rec.v0,
        bucket,
        nct_id=nct_id,
        http=http if with_competition else None,
        pool_cache={},
    )
    report.notes.append(
        f"[정답 — 감사 후 공개] 최종 상태 {rec.labels['final_status']}, "
        f"실제 등록 {rec.labels['actual_enrollment']}명 "
        f"(목표 {rec.v0.get('planned_enrollment')}명), "
        f"중단 사유: {rec.labels.get('why_stopped') or '기재 없음'}"
    )
    return report


SCOPE_NOTICE = (
    "이 감사는 **모집 실현성**만 다룹니다. 동일 피처의 적응증 층화 AUC 가 "
    "모집 실패 0.632 (CI 0.559-0.701), 효능·독성 실패 0.400 (CI 0.33-0.47, 역방향), "
    "경영 판단 중단 0.543 (CI 0.48-0.61, 기권)입니다. "
    "약효·독성 위험과 경영 판단 중단은 이 도구의 판정 범위가 아닙니다."
)


def render(report: AuditReport) -> str:
    """리포트 출력.

    **종합 점수를 표시하지 않는다.** holdout 에서 구분력이 입증되지 않았고
    (AUC 0.587, CI [0.484, 0.693]), 점수로 제시하면 캘리브레이션된 위험 확률이라는
    잘못된 인상을 준다. validated-numbers.md 3.3 의 결정이다.
    내부 진단값은 `to_dict()["diagnostic_score"]` 에만 남는다.
    """
    out = [
        f"■ {report.nct_id or '(신규 프로토콜)'} — {report.title or ''}",
        f"  적응증: {report.bucket} | 개시(계획): {report.start_date}",
        "",
        f"  ▣ 판정 범위: {SCOPE_NOTICE}",
        "",
    ]
    out.append(f"  ── 위험 판정 ({len(report.findings)}건) — 층화 검증을 통과한 피처만 ──\n")
    for f in report.sorted_findings():
        out.append(f"  [{f.severity.upper():<6}] {f.title}")
        out.append(f"           근거: {f.evidence}")
        out.append(f"           제안: {f.suggestion}")
        out.append("")
    if not report.findings:
        out.append("  검증된 피처에서 발견된 위험 신호 없음.\n")

    if report.context:
        out.append(f"  ── 참고 관측값 ({len(report.context)}건) — 위험 판정 근거가 아님 ──\n")
        for f in report.context:
            out.append(f"  [{f.validation_status:<12}] {f.title}")
            out.append(f"           근거: {f.evidence}")
            out.append(f"           주의: {f.suggestion}")
            out.append("")

    for n in report.notes:
        out.append(f"  · {n}")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        nct, bucket = sys.argv[1], sys.argv[2]
        print(render(audit_nct(nct, bucket)))
    else:
        print(__doc__)
        print("사용법: python src/audit/pipeline.py <NCT_ID> <적응증>")
        print("예시:   python src/audit/pipeline.py NCT03601897 'non-small cell lung cancer'")
