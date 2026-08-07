"""ID-only 음성 대조 — 사전학습 기억 누출량 측정.

## 왜 이 실험이 블라인드 주장의 전제인가

본 시스템은 "등록 시점 스냅샷만으로 심사한다"고 주장한다. Tier 1(LLM 0회)에서는 입력 구조가
그것을 보장한다. 그러나 **A4 는 LLM 이고, 프롬프트에 NCT ID 가 들어간다.** 유명한 트라이얼이면
모델이 결과를 이미 알고 있을 수 있다. 인용 의무 때문에 ID 를 지울 수도 없다.

→ **그러므로 측정한다.**

## 실험 설계

A4 에게 **프로토콜 본문·계량 감사 수치·전례 목록을 전부 빼고 NCT ID 만** 준다.
그 상태에서 나온 `PRECEDENT` 인용을 T3 로 검증한다.

- 인용이 **검증을 통과하면** — 모델이 주지 않은 전례를 기억에서 꺼내 정확히 맞힌 것이다.
  그만큼이 사전학습 누출이다.
- 인용이 **전부 기각되면** — 모델은 재료 없이 근거를 만들지 못한다. 블라인드 주장이 성립한다.

이 발상은 `stratified_signal.indication_only_control()` 과 같다. 그 함수가 경쟁 밀도 피처를
죽였다. 같은 도구를 LLM 계층에 적용한다.

**대조군 결과가 우연 수준이어야 블라인드 주장이 성립한다. 아니면 주장을 철회하고 그 사실을 보고한다.**

실행: python3 src/agents/idonly_control.py --limit 10 --budget-usd 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.engine import BudgetExceeded, Engine  # noqa: E402
from audit.contracts import ATTACK_OUTPUT_SPEC, Attack  # noqa: E402
from audit.precedent import load_corpus  # noqa: E402
from audit.referee import index_corpus, verify  # noqa: E402
from llm.client import LLM  # noqa: E402

OUT = Path("data/idonly_control.json")

SYSTEM = """당신은 임상시험 프로토콜을 심사하는 레드팀이다.
이 시험이 환자 모집에 실패할 위험을 지적하고, 근거가 되는 과거 전례를 인용하라.
인용에는 실제 NCT ID 와 그 시험의 whyStopped 원문을 정확히 적어야 한다."""


def build_prompt(nct_id: str, indication: str) -> str:
    return "\n".join([
        "## 심사 대상",
        f"- NCT ID: {nct_id}",
        f"- 적응증: {indication}",
        "",
        "프로토콜 본문과 참고 자료는 제공되지 않는다. 아는 범위에서 답하라.",
        "",
        "## 출력 형식",
        ATTACK_OUTPUT_SPEC,
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="ID-only 음성 대조")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--budget-usd", type=float, default=0.05)
    args = ap.parse_args()

    llm = LLM()
    if not llm.configured:
        raise SystemExit("LLM 미설정. .env 를 확인하세요.")
    engine = Engine(llm=llm, budget_usd=args.budget_usd, max_calls=args.limit + 5)

    rows = load_corpus()
    idx = index_corpus(rows)
    targets = [r for r in rows if (r.get("v0") or {}).get("start_date")][:args.limit]

    print(f"ID-only 음성 대조 — 대상 {len(targets)}건 · 예산 상한 ${args.budget_usd}")
    print("A4 에게 NCT ID 와 적응증만 주고 프로토콜·전례·수치는 전부 가린다.\n")

    results, reasons = [], Counter()
    n_gen = n_pass = n_prec = n_prec_pass = 0

    for r in targets:
        nct, bucket = r["nct_id"], r["condition_bucket"]
        cutoff = r["v0"]["start_date"]
        try:
            raw = engine.run_json("A4-idonly", build_prompt(nct, bucket),
                                  role="reason", system=SYSTEM, max_tokens=1024)
        except BudgetExceeded as e:
            print(f"\n⛔ {e}")
            break
        except Exception as e:                                    # noqa: BLE001
            print(f"  {nct}: 응답 파싱 실패 — {str(e)[:80]}")
            continue

        if isinstance(raw, dict):
            raw = raw.get("attacks") or [raw]
        atks = [Attack.from_obj(o) for o in raw if isinstance(o, dict)]
        n_gen += len(atks)

        passed = 0
        for a in atks:
            # 계량 인용은 재료가 없으므로 의미가 없다. 전례 인용만 센다.
            n_prec += sum(1 for c in a.citations if c.kind == "PRECEDENT")
            v = verify(a, corpus=idx, t1_values={}, cutoff=cutoff)
            if v.passed:
                passed += 1
                n_prec_pass += sum(1 for c in a.citations if c.kind == "PRECEDENT")
            else:
                reasons.update(v.reasons)
        n_pass += passed
        print(f"  {nct}  공격 {len(atks)}건 · 통과 {passed}건")
        results.append({"nct_id": nct, "generated": len(atks), "passed": passed})

    print(f"\n{'='*74}\nID-only 음성 대조 결과\n{'='*74}")
    if not n_gen:
        print("  공격이 생성되지 않았다.")
        return
    print(f"  공격 생성 {n_gen}건 · T3 통과 {n_pass}건 → 통과율 {n_pass/n_gen*100:.1f}%")
    print(f"  전례 인용 {n_prec}건 · 검증 통과 {n_prec_pass}건 "
          f"→ **누출 지표 {n_prec_pass/n_prec*100 if n_prec else 0:.1f}%**")
    print("\n  기각 사유:")
    for k, v in reasons.most_common():
        print(f"    {v:>3}  {k}")

    verdict = ("블라인드 주장 성립 — 모델은 재료 없이 검증 가능한 근거를 만들지 못했다."
               if n_prec_pass == 0 else
               f"⚠ 누출 있음 — 주지 않은 전례 {n_prec_pass}건을 기억에서 정확히 복원했다. "
               f"블라인드 주장을 이 범위만큼 약화시켜 보고해야 한다.")
    print(f"\n  판정: {verdict}")
    print(f"\n{engine.ledger.report()}")

    OUT.write_text(json.dumps({
        "_meta": {"n_targets": len(results), "llm_calls": len(engine.calls),
                  "measures": "프로토콜 본문 없이 NCT ID 만으로 검증 가능한 근거를 만드는가",
                  "verdict": verdict},
        "n_generated": n_gen, "n_passed": n_pass,
        "n_precedent_citations": n_prec, "n_precedent_verified": n_prec_pass,
        "leakage_rate": round(n_prec_pass / n_prec, 4) if n_prec else 0.0,
        "reject_reasons": dict(reasons), "per_target": results,
    }, ensure_ascii=False, indent=1))
    print(f"  {OUT} 저장")


if __name__ == "__main__":
    main()
