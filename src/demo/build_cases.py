"""백테스트 데모 케이스 생성 — 웹이 소비할 JSON 을 만든다.

데모의 핵심은 챗봇이 아니라 **블라인드 백테스트**다. 화면 흐름:

  1. 실제 트라이얼의 v0(등록 시점 스냅샷)만 보여준다 — 결과 정보 0
  2. 에이전트의 감사 findings 와 교정 제안
  3. [실제 결과 공개] → 최종 상태, 실제 등록수, 중단 사유

따라서 케이스 JSON 은 `blind`(공개 가능) 와 `reveal`(버튼 클릭 후) 를 **구조적으로 분리**한다.
프론트에서 reveal 을 먼저 렌더하는 실수를 막기 위한 분리이고, 사전계산이므로 서버·키가 필요 없다.

케이스 선정은 결과를 보고 고르면 체리피킹이 된다. 그래서 선정 기준을 코드로 고정하고
선정에 쓴 규칙을 JSON 에 함께 남긴다 — 심사에서 재현 가능해야 한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import group_of  # noqa: E402
from analysis.holdout import in_test  # noqa: E402
from audit import benchmarks as bm  # noqa: E402
from audit.pipeline import audit_v0  # noqa: E402
from labels.taxonomy import CATEGORY_NAMES, classify, enrollment_attainment  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
OUT = Path("data/demo_cases.json")

# 케이스 구성 — 각 집단에서 몇 건씩 뽑을지. 성공 사례만 넣으면 데모가 거짓이 된다.
# 완주군과 경영중단군을 반드시 포함해 오탐도 그대로 보여준다.
QUOTA = {
    "중단-모집실패": 6,
    "완주": 4,
    "중단-경영": 2,
    "중단-과학적": 2,
}

SELECTION_RULE = (
    "holdout test 분할(NCT ID SHA-256 해시 기반)에 속한 트라이얼만 후보로 삼고, "
    "집단별 정해진 수만큼 NCT ID 사전순으로 앞에서부터 선택한다. "
    "감사 결과를 보고 고르지 않는다 — 체리피킹을 배제하기 위한 규칙."
)


def blind_view(v0: dict) -> dict:
    """등록 시점에 볼 수 있던 것만. 결과 관련 필드는 절대 넣지 않는다."""
    return {
        "brief_title": v0.get("brief_title"),
        "start_date": v0.get("start_date"),
        "planned_primary_completion": v0.get("planned_primary_completion"),
        "planned_enrollment": v0.get("planned_enrollment"),
        "planned_enrollment_type": v0.get("planned_enrollment_type"),
        "phases": v0.get("phases"),
        "allocation": v0.get("allocation"),
        "masking": v0.get("masking"),
        "n_arms": v0.get("n_arms"),
        "primary_outcomes": v0.get("primary_outcomes"),
        "n_secondary_outcomes": v0.get("n_secondary_outcomes"),
        "n_inclusion_items": v0.get("n_inclusion_items"),
        "n_exclusion_items": v0.get("n_exclusion_items"),
        "eligibility_criteria": v0.get("eligibility_criteria"),
        "n_sites": v0.get("n_sites"),
        "countries": v0.get("countries"),
        "brief_summary": v0.get("brief_summary"),
    }


def reveal_view(row: dict) -> dict:
    labels = row["labels"]
    cat, reason, evidence = classify(labels.get("why_stopped"))
    att = enrollment_attainment(
        row["v0"].get("planned_enrollment"), labels.get("actual_enrollment")
    )
    return {
        "final_status": labels.get("final_status"),
        "actual_enrollment": labels.get("actual_enrollment"),
        "enrollment_attainment": att,
        "actual_primary_completion": labels.get("actual_primary_completion"),
        "why_stopped": labels.get("why_stopped"),
        "category": cat,
        "category_name": CATEGORY_NAMES.get(cat),
        "category_reason": reason,
        "category_evidence": evidence,
        "amend_eligibility": labels.get("amend_eligibility"),
        "n_versions": labels.get("n_versions"),
    }


def select(rows: list[dict]) -> list[dict]:
    pool = [r for r in rows if in_test(r["nct_id"])]
    picked: list[dict] = []
    for group, n in QUOTA.items():
        cands = sorted(
            (r for r in pool if group_of(r) == group), key=lambda r: r["nct_id"]
        )
        if len(cands) < n:
            print(f"  ! {group}: 후보 {len(cands)}건 (요청 {n}건) — 있는 만큼만 사용")
        picked.extend(cands[:n])
    return picked


def main() -> None:
    bench = bm.load()
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    cases_rows = select(rows)

    cases = []
    for r in cases_rows:
        # 네트워크 미사용 — 경쟁 밀도는 별도 배치로 채운다 (적응증 풀 조회가 느리다)
        rep = audit_v0(
            r["v0"], r["condition_bucket"], nct_id=r["nct_id"], bench=bench, http=None
        )
        cases.append(
            {
                "nct_id": r["nct_id"],
                "condition": r["condition_bucket"],
                "group": group_of(r),
                "blind": blind_view(r["v0"]),
                "audit": rep.to_dict(),
                "reveal": reveal_view(r),
            }
        )

    payload = {
        "_meta": {
            "selection_rule": SELECTION_RULE,
            "quota": QUOTA,
            "n_cases": len(cases),
            "source_rows": len(rows),
            "warning": (
                "audit 는 Tier 1 결정론적 감사만 반영한다. 현재 holdout 구분력이 미입증 상태이므로 "
                "이 데모는 '작동하는 예측기'가 아니라 '감사 근거를 어떻게 제시하는가'를 보여주는 것이다."
            ),
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print(f"\n{OUT} 생성 — 케이스 {len(cases)}건\n")
    print(f"  {'NCT':<14}{'집단':<14}{'점수':>5}{'달성률':>8}  결과")
    for c in cases:
        rv = c["reveal"]
        att = rv["enrollment_attainment"]
        print(
            f"  {c['nct_id']:<14}{c['group']:<14}{c['audit']['diagnostic_score']:>5}"
            f"{(f'{att:.2f}' if att is not None else '—'):>8}  {rv['final_status']}"
        )


if __name__ == "__main__":
    main()
