"""A1 Planner 【에이전트】 — 자율적 계획과 명시적 기권.

## 왜 이 컴포넌트가 정당한가

"모든 감사를 항상 돌린다"가 **실제로 틀린 결과를 냈다.** `phase_stratum()` docstring 이
기록하듯, Phase 1/2 용량증량 시험은 목표 등록수가 작고 적격기준이 촘촘한 것이 **정상 설계**인데,
순수 Phase 2 완주군과 비교해 NCT03601897 을 최고 위험으로 오판했다. 실제로는 목표 30명에
177명이 등록된 **모집 대성공** 사례였다.

A1 은 그 교훈의 구현이다. 그리고 **적용 불가 판정에 반드시 이유를 붙여 노출**한다.

## 규칙 우선, LLM 은 미해당 케이스만

아래 표는 규칙으로 충분하다. **LLM 이 필요한 부분은 표에 없는 케이스를 만났을 때**
적용 가능 여부를 스스로 판정하는 것이다. 따라서 규칙 테이블을 먼저 적용하고 미해당 항목만
LLM 에 맡기며, **규칙으로 커버된 비율을 함께 보고한다**(`taxonomy` 의 미분류율 보고 관행과 동일).

## 재계획 루프

전례 공급이 부족하면 A1 은 검색 층을 완화해 재시도하고, 그래도 0건이면 전례 기반 공격을
기각한다. 발동 빈도를 숫자로 말할 수 있다 — 감사 코호트·2순위 층 기준 **31.2%**
(`src/audit/precedent.py` docstring 의 대조표 참조).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.engine import Engine  # noqa: E402
from audit.benchmarks import phase_stratum  # noqa: E402
from audit.pipeline import RISK_ELIGIBLE  # noqa: E402
from audit.precedent import MIN_PRECEDENTS, PrecedentSet  # noqa: E402

# 감사 이름 → 사람이 읽을 설명
AUDITS = {
    "quant_burden": "월 모집부담 계량 감사 (검증 통과 피처)",
    "criteria_density": "적격기준 촘촘도 감사",
    "site_burden": "사이트당 모집부담 감사",
    "precedent_recruit": "전례 기반 모집 공격",
    "reference_dist": "적응증×Phase 참조 분포 비교",
}


@dataclass
class AuditPlan:
    run: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    replans: list[str] = field(default_factory=list)
    rule_covered: int = 0
    llm_decided: int = 0

    def label(self, name: str) -> str:
        return AUDITS.get(name, name)

    def render(self) -> str:
        out = ["  [실행할 감사]"]
        out += [f"    · {self.label(a)}" for a in self.run] or ["    (없음)"]
        out.append("  [돌리지 않는 감사와 그 이유]")
        out += [f"    × {self.label(a)} — {why}" for a, why in self.rejected] or ["    (없음)"]
        if self.replans:
            out.append("  [재계획]")
            out += [f"    ↻ {r}" for r in self.replans]
        total = self.rule_covered + self.llm_decided
        if total:
            out.append(f"  규칙 커버리지 {self.rule_covered}/{total} "
                       f"({self.rule_covered/total*100:.0f}%) · LLM 판정 {self.llm_decided}건")
        return "\n".join(out)


SYSTEM = """당신은 임상시험 프로토콜 감사 시스템의 계획 담당이다.
주어진 감사 후보가 이 프로토콜에 적용 가능한지 판단한다.
적용 불가로 판단하면 반드시 구체적 이유를 붙인다.
확신이 없으면 적용 불가 쪽으로 판단한다 — 근거 없는 감사를 돌리는 것보다 기권이 낫다."""


def plan(
    v0: dict,
    *,
    precedents: PrecedentSet,
    bench_bucket_n: int | None = None,
    extra_audits: list[str] | None = None,
    engine: Engine | None = None,
) -> AuditPlan:
    """규칙 테이블을 먼저 적용하고, 미해당 감사만 LLM 에 판정을 맡긴다."""
    p = AuditPlan()
    stratum = phase_stratum(v0)

    def rule(name: str, ok: bool, why: str) -> None:
        p.rule_covered += 1
        (p.run.append(name) if ok else p.rejected.append((name, why)))

    # --- 규칙 테이블 (agent-architecture.md §2) ---
    rule("quant_burden", "monthly_enrollment_burden" in RISK_ELIGIBLE,
         "월 모집부담이 위험 판정 자격을 잃었다")

    rule("criteria_density", stratum != "p1p2",
         "Phase 1/2 용량증량 시험은 적격기준이 촘촘한 것이 정상 설계다. "
         "순수 Phase 2 와 같은 분포로 비교하면 정상 설계를 위험으로 오판한다 (NCT03601897 오탐의 원인)")

    n_sites = v0.get("n_sites")
    rule("site_burden", bool(n_sites),
         "사이트 목록이 결측이다. 0으로 오해하면 계산이 망가진다")

    rule("reference_dist", (bench_bucket_n or 0) >= 6,
         f"해당 적응증×Phase 층의 표본이 {bench_bucket_n or 0}건으로 6건 미만이다. "
         "상위 층 폴백을 쓰며 층별 비교는 기각한다")

    # --- 전례 기반 공격: 공급량을 보고 판단하고, 부족하면 재계획 ---
    n_a = len(precedents.citable_a)
    if n_a >= MIN_PRECEDENTS:
        rule("precedent_recruit", True, "")
    elif n_a > 0:
        p.rule_covered += 1
        p.run.append("precedent_recruit")
        p.replans.append(
            f"A 분류 전례가 {n_a}건으로 임계값 {MIN_PRECEDENTS} 미만이라 검색 층을 "
            f"'{precedents.tier}'까지 완화했다. 인용 가능한 전례가 적음을 리포트에 명시한다")
    else:
        rule("precedent_recruit", False,
             "시간 컷오프 이전에 개시된 A 분류(모집 실패) 전례가 코퍼스에 0건이다. "
             "QUANT 인용만 허용하며, 이는 '위험이 없다'가 아니라 '인용할 근거가 없다'는 뜻이다")

    # --- 규칙에 없는 감사만 LLM 이 판정한다 ---
    for name in (extra_audits or []):
        if engine is None:
            p.rejected.append((name, "규칙에 없고 LLM 판정기가 없어 기각"))
            continue
        prompt = (
            f"프로토콜 요약:\n"
            f"- Phase 계층: {stratum}\n"
            f"- 목표 등록수: {v0.get('planned_enrollment')}\n"
            f"- 개시일: {v0.get('start_date')}\n"
            f"- 무작위배정: {v0.get('allocation')} / 눈가림: {v0.get('masking')}\n"
            f"- 팔 수: {v0.get('n_arms')} / 2차 평가변수 수: {v0.get('n_secondary_outcomes')}\n\n"
            f"감사 후보: {name}\n\n"
            f'JSON 만 출력: {{"applicable": true|false, "reason": "한 문장"}}')
        try:
            r = engine.run_json("A1", prompt, role="reason", system=SYSTEM, max_tokens=256)
        except Exception as e:                                   # noqa: BLE001
            p.rejected.append((name, f"LLM 판정 실패로 기각: {e}"))
            continue
        p.llm_decided += 1
        if isinstance(r, dict) and r.get("applicable"):
            p.run.append(name)
        else:
            reason = (r or {}).get("reason", "적용 불가") if isinstance(r, dict) else "판정 불가"
            p.rejected.append((name, reason))

    return p
