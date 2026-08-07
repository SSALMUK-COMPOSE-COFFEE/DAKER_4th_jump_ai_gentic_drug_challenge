"""A4 레드팀 【에이전트】 — 근거 결박 계약.

**계약: 모든 공격은 검증 가능한 인용을 최소 1개 가져야 한다. 없으면 공격이 아니다.**

A4 는 LLM 이 실제로 판단하는 지점이다 — 프로토콜의 **어디를** 칠지, 주어진 전례 중
**어느 것이** 이 프로토콜과 닮았는지, **어떤 문장으로** 지적할지.

그 판단의 검증은 A4 가 하지 않는다. T3(`audit/referee.py`)가 인용을 원문 대조하고,
T5(`audit/validator.py`)가 근거의 통계적 자격을 판정한다. **자율성은 LLM 이 갖고
검증은 코드가 갖는다** — 이 분리가 본 설계의 원칙이다.

## 반송 루프

T3 가 기각하면 기각 사유를 붙여 A4 에 되돌린다 (최대 `MAX_REPAIR` 회).
반송 횟수를 제한하는 이유는 비용이다 — 무한 재시도는 크레딧을 태운다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.engine import Engine  # noqa: E402
from audit.contracts import ATTACK_OUTPUT_SPEC, Attack  # noqa: E402
from audit.precedent import PrecedentSet  # noqa: E402

MAX_REPAIR = 2          # 반송 재시도 상한. 비용 통제.
MAX_PRECEDENTS_IN_PROMPT = 12
MAX_CRITERIA_CHARS = 2500

# ⚠ 프롬프트 설계 주의 — 실측으로 배운 것
#
# 초기 SYSTEM 의 마지막 줄이 "근거를 댈 수 없으면 빈 배열을 출력한다. 빈 배열은 실패가
# 아니다" 였다. 그러자 모델이 **인용 가능한 전례가 32건 있는데도 `[]` 만 반환**했다
# (출력 1토큰). 안전한 탈출구를 열어주면 모델은 그쪽으로 간다.
#
# 이 설계에서 A4 는 보수적일 이유가 없다. **틀린 공격을 걸러내는 것은 T3·T5 의 일**이고,
# A4 가 침묵하면 걸러낼 대상 자체가 없어져 기각률을 측정할 수 없다. 따라서 재료가 있으면
# 반드시 시도하도록 요구하고, 침묵은 재료가 없을 때로 한정한다.
SYSTEM = """당신은 임상시험 프로토콜을 심사하는 레드팀이다.
목표는 이 프로토콜이 **환자 모집에 실패할 수 있는 지점**을 찾아 지적하는 것이다.

작업 방식:
- 제공된 전례 목록에서 이 프로토콜과 가장 닮은 사례를 고르고, 그 시험이 왜 멈췄는지를
  근거로 이 프로토콜의 어느 부분이 같은 위험을 갖는지 지적한다.
- 계량 감사 수치가 완주군 중앙값과 크게 다르면 그것도 지적 대상이다.

절대 규칙:
- 모든 지적에는 제공된 자료에서 나온 인용을 붙인다. 인용 없는 지적은 만들지 않는다.
- 전례 인용문은 제공된 목록에서 글자 그대로 복사한다. 요약·의역·창작은 금지한다.
- 제공되지 않은 NCT ID 나 수치를 지어내지 않는다.
- 프로토콜을 고쳐서 바꿀 수 있는 요소만 겨냥한다.

분량:
- **전례 목록이나 계량 감사 수치가 제공되었다면 최소 1건, 최대 4건의 지적을 반드시 만든다.**
- 빈 배열은 **인용할 재료가 하나도 제공되지 않았을 때만** 허용된다.
  재료가 있는데 침묵하는 것은 임무 실패다. 확신이 부족한 지적이라도 인용을 붙여 제출하라 —
  근거의 타당성은 별도의 검증 단계가 판정한다."""


def _fmt_precedents(ps: PrecedentSet) -> str:
    if not ps.citable_a:
        return "(인용 가능한 모집 실패 전례가 없습니다. PRECEDENT 인용을 쓸 수 없습니다.)"
    lines = [f"검색 층: {ps.tier} · 시간 컷오프: {ps.cutoff} 이전 개시분만"]
    for p in ps.citable_a[:MAX_PRECEDENTS_IN_PROMPT]:
        att = ""
        if p.planned_enrollment and p.actual_enrollment is not None:
            att = f" (목표 {p.planned_enrollment} → 실제 {p.actual_enrollment})"
        lines.append(f'- {p.nct_id} | 개시 {p.start_date}{att}\n'
                     f'  why_stopped: "{(p.why_stopped or "").strip()}"')
    if len(ps.citable_a) > MAX_PRECEDENTS_IN_PROMPT:
        lines.append(f"… 외 {len(ps.citable_a)-MAX_PRECEDENTS_IN_PROMPT}건 (생략)")
    return "\n".join(lines)


def _fmt_quant(t1_values: dict, eligible: set[str], refs: dict | None) -> str:
    lines = []
    for k, v in t1_values.items():
        if v is None:
            continue
        tag = "위험 판정 가능" if k in eligible else "참고용 — 위험 근거로 쓸 수 없음"
        ref = ""
        if refs and (q := refs.get(k)):
            ref = f" | 완주군 중앙값 {q.get('median')} (n={q.get('n')})"
        lines.append(f"- {k} = {v}{ref}  [{tag}]")
    return "\n".join(lines) or "(계량 감사 결과 없음)"


def build_prompt(
    v0: dict, *, precedents: PrecedentSet, t1_values: dict,
    eligible: set[str], refs: dict | None = None,
    rejected_feedback: list[str] | None = None,
) -> str:
    crit = (v0.get("eligibility_criteria") or "")[:MAX_CRITERIA_CHARS]
    parts = [
        "## 심사 대상 프로토콜 (등록 시점 스냅샷 — 결과 정보 없음)",
        f"- 제목: {v0.get('brief_title')}",
        f"- 개시일: {v0.get('start_date')} · 목표 등록수: {v0.get('planned_enrollment')}"
        f" ({v0.get('planned_enrollment_type')})",
        f"- 배정: {v0.get('allocation')} · 눈가림: {v0.get('masking')} · 팔 수: {v0.get('n_arms')}",
        "",
        "### 적격기준 원문",
        crit,
        "",
        "## 계량 감사 결과 (T1)",
        _fmt_quant(t1_values, eligible, refs),
        "",
        "## 인용 가능한 과거 전례 (모집 실패로 중단된 것만)",
        _fmt_precedents(precedents),
        "",
    ]
    if rejected_feedback:
        parts += [
            "## 이전 시도가 기각되었다. 아래 사유를 고쳐서 다시 작성하라.",
            *(f"- {r}" for r in rejected_feedback),
            "",
        ]
    parts += ["## 출력 형식", ATTACK_OUTPUT_SPEC]
    return "\n".join(parts)


def generate(
    v0: dict, *, precedents: PrecedentSet, t1_values: dict, eligible: set[str],
    engine: Engine, refs: dict | None = None,
    rejected_feedback: list[str] | None = None,
) -> list[Attack]:
    prompt = build_prompt(v0, precedents=precedents, t1_values=t1_values,
                          eligible=eligible, refs=refs,
                          rejected_feedback=rejected_feedback)
    raw = engine.run_json("A4", prompt, role="reason", system=SYSTEM, max_tokens=2048)
    if isinstance(raw, dict):
        raw = raw.get("attacks") or [raw]
    return [Attack.from_obj(o) for o in raw if isinstance(o, dict)]
