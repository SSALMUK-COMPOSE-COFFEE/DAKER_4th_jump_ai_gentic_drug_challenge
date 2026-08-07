"""A6 Reviser 【에이전트】 — 교정안 생성과 회귀 자가 탐지.

T3 를 통과한 지적만 받아 프로토콜 수정안을 만든다. 그리고 **T1 을 재실행해 델타를 잰다.**

## 핵심은 개선이 아니라 악화 탐지다

수정이 한 피처를 좋게 만들면서 다른 피처를 나쁘게 만들 수 있다. 예를 들어 적격기준을 느슨하게
해서 항목 수는 줄었지만 목표 등록수를 그대로 두면, 월 모집부담이 오히려 실패군 방향으로
움직인다. **그 사실을 에이전트가 스스로 찾아내 보고한다.**

본선 평가 항목 "오류 발생 시 스스로 인지하고 수정하는가"(10점)에 대응하는 실물이며,
시연의 클라이맥스다.

## 교정 방향의 검산

수정안의 방향이 **같은 층 완주군의 실제 설계와 일치하는지** 확인한다. 완주군이 실제로
채택한 값에서 멀어지는 교정은 회귀로 표시한다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.engine import Engine  # noqa: E402
from audit.contracts import Attack  # noqa: E402
from audit.pipeline import FEATURE_STATUS, features_from_v0  # noqa: E402

SYSTEM = """당신은 임상시험 프로토콜 설계자다.
검증된 지적을 받아 **구체적이고 실행 가능한 수정안**을 만든다.

규칙:
- 수정은 프로토콜에서 실제로 바꿀 수 있는 값이어야 한다 (목표 등록수, 계획 기간,
  적격기준 임계값, 사이트 수 등).
- 근거 없이 값을 바꾸지 않는다. 지적에 붙은 인용의 범위 안에서만 제안한다.
- 수정이 다른 설계 요소에 미치는 영향을 함께 적는다."""

REVISION_SPEC = """JSON 만 출력한다:

{
  "revisions": [
    {"field": "planned_enrollment", "from": 120, "to": 80,
     "rationale": "왜 이 값인지 한 문장", "addresses": "겨냥한 지적의 target_element"}
  ],
  "side_effects": ["이 수정이 악화시킬 수 있는 것을 스스로 적는다"]
}

수정 가능한 field: planned_enrollment, planned_duration_months, n_sites,
eligibility_criteria(텍스트 수정 서술)."""


@dataclass
class Delta:
    feature: str
    before: float | None
    after: float | None
    direction: str          # improved | worsened | unchanged | unknown
    note: str = ""


@dataclass
class Revision:
    proposals: list[dict] = field(default_factory=list)
    declared_side_effects: list[str] = field(default_factory=list)
    narrative_only: list[dict] = field(default_factory=list)
    deltas: list[Delta] = field(default_factory=list)

    @property
    def regressions(self) -> list[Delta]:
        return [d for d in self.deltas if d.direction == "worsened"]

    def render(self) -> str:
        out = ["  [교정안]"]
        out += [f"    · {p.get('field')}: {p.get('from')} → {p.get('to')}  ({p.get('rationale','')})"
                for p in self.proposals] or ["    (없음)"]
        if self.narrative_only:
            out.append("  [서술형 제안 — 자동 재감사 대상 아님]")
            out += [f"    ~ {p.get('field')}: {str(p.get('to'))[:90]}"
                    for p in self.narrative_only]
        if self.declared_side_effects:
            out.append("  [에이전트가 스스로 밝힌 부작용]")
            out += [f"    ! {s}" for s in self.declared_side_effects]
        out.append("  [재감사 델타]")
        for d in self.deltas:
            mark = {"improved": "↑ 개선", "worsened": "↓ 악화",
                    "unchanged": "= 변화없음"}.get(d.direction, "? 불명")
            out.append(f"    {mark}  {d.feature}: {d.before} → {d.after}  {d.note}")
        if self.regressions:
            out.append(f"  ⚠ 이 교정안은 {len(self.regressions)}개 항목을 악화시킨다 "
                       f"— 에이전트가 스스로 탐지했다.")
        return "\n".join(out)


# 기계적으로 재감사할 수 있는 필드 — 수치로 치환 가능한 것만.
NUMERIC_FIELDS = {"planned_enrollment", "n_sites", "planned_duration_months"}

# 자연어 서술로 오는 필드. **치환하지 않는다.**
NARRATIVE_FIELDS = {"eligibility_criteria"}


def apply_revisions(v0: dict, proposals: list[dict]) -> tuple[dict, list[dict]]:
    """수정안을 v0 사본에 적용한다.

    ⚠ **자연어 서술은 치환하지 않는다.** A6 은 적격기준 수정을 "이중 바이오마커 → 단독으로
    완화" 같은 **서술**로 돌려주는데, 이를 본문으로 치환하면 4,801자가 50자가 되어
    `eligibility_chars` 델타가 완전히 거짓이 된다(실측에서 실제로 발생했다).
    자연어 기준 변경은 기계적 재감사의 대상이 아니며, 별도로 분리해 보고한다.

    반환: (수정된 v0, 기계 적용되지 않은 서술형 제안 목록)
    """
    out = dict(v0)
    narrative: list[dict] = []
    for p in proposals:
        f, to = p.get("field"), p.get("to")
        if f in NUMERIC_FIELDS and isinstance(to, (int, float)) and not isinstance(to, bool):
            out["_duration_override" if f == "planned_duration_months" else f] = (
                float(to) if f == "planned_duration_months" else int(to))
        else:
            narrative.append(p)
    return out, narrative


def measure_delta(v0: dict, v1: dict, *, completed_ref: dict | None = None) -> list[Delta]:
    """T1 을 재실행해 피처별 이동 방향을 잰다.

    방향 판정의 기준은 **완주군 중앙값에 가까워졌는가**다. 참조 분포가 없으면 `unknown`
    으로 남기고 개선이라고 주장하지 않는다.
    """
    before = features_from_v0(v0)
    after = features_from_v0(v1)
    out: list[Delta] = []
    for k in sorted(set(before) | set(after)):
        b, a = before.get(k), after.get(k)
        if b is None and a is None:
            continue
        if b == a:
            out.append(Delta(k, b, a, "unchanged"))
            continue
        ref = (completed_ref or {}).get(k, {}).get("median") if completed_ref else None
        if ref is None or b is None or a is None:
            out.append(Delta(k, b, a, "unknown", "참조 분포가 없어 방향을 판정하지 않는다"))
            continue
        moved = abs(a - ref) < abs(b - ref)
        status = FEATURE_STATUS.get(k, ("unvalidated", ""))[0]
        note = f"완주군 중앙값 {ref}" + ("" if status == "validated" else f" · 피처 상태 {status}")
        out.append(Delta(k, b, a, "improved" if moved else "worsened", note))
    return out


def revise(
    v0: dict, passed_attacks: list[Attack], *, engine: Engine,
    completed_ref: dict | None = None,
) -> Revision:
    if not passed_attacks:
        return Revision()

    findings = "\n".join(
        f"- [{a.target_element}] {a.claim}\n  제안: {a.suggestion}" for a in passed_attacks)
    prompt = (
        "## 검증을 통과한 지적\n" + findings +
        "\n\n## 현재 프로토콜 값\n"
        f"- planned_enrollment: {v0.get('planned_enrollment')}\n"
        f"- start_date: {v0.get('start_date')}\n"
        f"- planned_primary_completion: {v0.get('planned_primary_completion')}\n"
        f"- n_sites: {v0.get('n_sites')}\n"
        "\n## 출력 형식\n" + REVISION_SPEC)

    raw = engine.run_json("A6", prompt, role="reason", system=SYSTEM, max_tokens=1024)
    if isinstance(raw, list):
        raw = {"revisions": raw}
    proposals = [p for p in (raw or {}).get("revisions", []) if isinstance(p, dict)]
    side = [str(s) for s in (raw or {}).get("side_effects", [])]

    v1, narrative = apply_revisions(v0, proposals)
    return Revision(proposals=proposals, declared_side_effects=side,
                    narrative_only=narrative,
                    deltas=measure_delta(v0, v1, completed_ref=completed_ref))
