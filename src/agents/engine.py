"""에이전트 공용 LLM 실행기 — dry-run 과 비용 통제.

## 왜 별도 계층인가

키를 넣는 순간부터 호출마다 돈이 나간다. 따라서 **키 없이도 프롬프트 조립·토큰 추정·
비용 예상까지 전부 검증할 수 있어야 한다.** 이 실행기가 그 경계다.

- `dry_run=True` — 실호출 0회. 프롬프트를 조립해 기록하고, 준비된 응답(fixture)이 있으면
  그것을 돌려준다. 없으면 빈 결과를 돌려주고 그 사실을 기록한다.
- `max_calls` — 폭주 방지 하드 캡. 초과하면 예외를 던진다.
- 모든 호출은 `Ledger` 에 적산된다 (제안서 항목 7의 실측 비용).

## 토큰 추정

dry-run 의 토큰 수는 **문자 수 기반 근사치**다(한글은 대략 문자당 1토큰 이상, 영문은 4자당
1토큰 수준이라 혼합 텍스트에서 오차가 크다). 실측이 아니며, **예상 비용을 보수적으로
잡기 위한 용도**다. 실제 값은 호출 후 `Ledger` 에 기록된 provider 보고값을 쓴다.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.client import LLM, Ledger, LLMError, _price_of  # noqa: E402


def estimate_tokens(text: str) -> int:
    """문자 수 기반 근사. 한글은 1자≈1토큰, 그 외는 4자≈1토큰으로 잡는다 (보수적)."""
    if not text:
        return 0
    hangul = len(re.findall(r"[가-힣]", text))
    return hangul + (len(text) - hangul) // 4


@dataclass
class CallRecord:
    agent: str
    role: str
    model: str
    prompt: str
    system: str | None
    est_in: int
    dry: bool
    response: str = ""


class StopRun(RuntimeError):
    """실행을 중단시키는 제어 예외.

    ⚠ **호출부는 반드시 이 예외를 잡아 지금까지의 결과를 저장해야 한다.**
    2026-07-31 에 호출 상한 예외를 잡지 않아 40회분($0.05)의 결과가 크래시로 유실됐다.
    돈은 나갔는데 데이터가 남지 않는 것이 최악이다.
    """


class BudgetExceeded(StopRun):
    """예산 상한 도달."""


class CallCapExceeded(StopRun):
    """호출 수 상한 도달."""


@dataclass
class Engine:
    llm: LLM
    dry_run: bool = False
    max_calls: int = 40
    use_fixtures: bool = False
    budget_usd: float | None = None      # 하드 상한. 도달하면 더 호출하지 않는다
    verbose_cost: bool = True            # 호출마다 토큰·비용을 찍는다
    fixtures: dict[str, str] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    exhausted: bool = False

    @property
    def ledger(self) -> Ledger:
        return self.llm.ledger

    def spent(self) -> float:
        return self.ledger.total_cost_usd()

    def remaining(self) -> float | None:
        return None if self.budget_usd is None else max(0.0, self.budget_usd - self.spent())

    def _check_budget(self) -> None:
        if self.budget_usd is None or self.dry_run:
            return
        if self.spent() >= self.budget_usd:
            self.exhausted = True
            raise BudgetExceeded(
                f"예산 상한 ${self.budget_usd:.4f} 도달 (사용 ${self.spent():.4f}). 중단합니다.")

    def _fixture_for(self, agent: str) -> str | None:
        return self.fixtures.get(agent)

    def run(self, agent: str, prompt: str, *, role: str, system: str | None = None,
            max_tokens: int = 2048) -> str:
        if len(self.calls) >= self.max_calls:
            self.exhausted = True
            raise CallCapExceeded(
                f"호출 상한 {self.max_calls} 도달 — 폭주 방지로 중단합니다. "
                f"(--max-calls 로 조정)")
        try:
            model = self.llm.model_for(role)
        except LLMError:
            # dry-run 은 프로바이더가 없어도 프롬프트 조립을 끝까지 검증할 수 있어야 한다.
            if not self.dry_run:
                raise
            model = f"(미설정:{role})"
        rec = CallRecord(agent=agent, role=role, model=model, prompt=prompt, system=system,
                         est_in=estimate_tokens(prompt) + estimate_tokens(system or ""),
                         dry=self.dry_run)
        self.calls.append(rec)

        if self.dry_run:
            rec.response = self._fixture_for(agent) or ""
            return rec.response

        self._check_budget()                     # 호출 **전** 확인
        before = self.spent()
        u_before = self.ledger.per_model.get(model)
        in_before = u_before.input_tokens if u_before else 0
        out_before = u_before.output_tokens if u_before else 0

        rec.response = self.llm.complete(prompt, role=role, system=system,
                                         max_tokens=max_tokens)

        u = self.ledger.per_model.get(model)
        d_in = (u.input_tokens - in_before) if u else 0
        d_out = (u.output_tokens - out_before) if u else 0
        d_cost = self.spent() - before
        if self.verbose_cost:
            rem = self.remaining()
            tail = f" · 잔여 ${rem:.4f}" if rem is not None else ""
            print(f"      💲 {agent} [{model}] 입력 {d_in:,} / 출력 {d_out:,} 토큰 · "
                  f"${d_cost:.5f} · 누적 ${self.spent():.4f}{tail}")
        self._check_budget()                     # 호출 **후** 확인
        return rec.response

    def run_json(self, agent: str, prompt: str, **kw):
        raw = self.run(agent, prompt, **kw)
        if not raw.strip():
            return []
        text = raw.strip()
        if m := re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S):
            text = m.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if m := re.search(r"[\[{].*[\]}]", text, re.S):
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"{agent}: JSON 파싱 실패. 응답 앞 200자: {text[:200]}")

    # --- 보고 ---------------------------------------------------------------

    def estimate_report(self) -> str:
        """dry-run 의 예상 비용. 출력 토큰은 알 수 없으므로 입력 기준 하한만 제시한다."""
        by: dict[str, list[int]] = {}
        for c in self.calls:
            by.setdefault(c.model, []).append(c.est_in)
        lines = [f"  {'모델':<34}{'호출':>7}{'추정 입력토큰':>16}{'입력측 비용':>14}"]
        total = 0.0
        for m, v in sorted(by.items()):
            p = _price_of(m)
            cost = sum(v) / 1e6 * p[0] if p else None
            total += cost or 0.0
            lines.append(f"  {m:<34}{len(v):>7}{sum(v):>16,}"
                         f"{('$%.4f' % cost) if cost is not None else '가격미등록':>14}")
        lines.append(f"  {'합계 (입력만)':<34}{len(self.calls):>7}"
                     f"{sum(sum(v) for v in by.values()):>16,}{'$%.4f' % total:>14}")
        lines.append("  ※ 문자 수 기반 근사치이며 출력 토큰은 포함되지 않은 하한이다.")
        return "\n".join(lines)

    def dump_prompts(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            [{"agent": c.agent, "role": c.role, "model": c.model, "est_in": c.est_in,
              "system": c.system, "prompt": c.prompt} for c in self.calls],
            ensure_ascii=False, indent=1))
