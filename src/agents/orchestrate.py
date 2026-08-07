"""전체 에이전트 루프 — A1 → T1·T2 → A4 ⇄ T3/T5 → A6 → T1 재실행.

## 실행

    # 키 없이 전 과정 검증 (실호출 0회, 프롬프트·예상비용 확인)
    python3 src/agents/orchestrate.py --dry-run --limit 2 --dump-prompts /tmp/prompts.json

    # 키를 넣은 뒤 실제 실행
    python3 src/agents/orchestrate.py --limit 2

## 세 개의 닫힌 루프

| 루프 | 구간 | 측정치 |
|---|---|---|
| 인용 검증 | A4 ⇄ T3 | 기각률 (사유별) |
| 근거 검증 | A4 ⇄ T5 | 기각률 (통계 사유별) |
| 교정 재감사 | A6 → T1 | 리스크 델타 · 악화 자가 탐지 |

**세 지표 전부 정답 라벨 없이 코퍼스만으로 측정된다.**

## 비용 통제

키를 넣는 순간부터 호출마다 비용이 발생하므로, 기본값을 보수적으로 잡았다 —
`--limit 2`, 반송 상한 2회, 전체 호출 상한 `--max-calls`. dry-run 에서 예상 비용을
먼저 확인한 뒤 실행할 것.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import redteam, reviser  # noqa: E402
from agents.engine import Engine, StopRun  # noqa: E402
from agents.planner import plan  # noqa: E402
from audit import benchmarks as bm  # noqa: E402
from audit.contracts import Attack  # noqa: E402
from audit.pipeline import RISK_ELIGIBLE, features_from_v0  # noqa: E402
from audit.precedent import load_corpus, search  # noqa: E402
from audit.referee import index_corpus, verify  # noqa: E402
from llm.client import LLM  # noqa: E402

OUT = Path("data/agent_run.json")


def demo_fixtures(ps, t1: dict) -> dict[str, str]:
    """dry-run 전용 — 실제 전례로 A4/A6 응답을 합성해 **루프 전 구간을 태운다.**

    키 없이도 T3 반송 루프와 A6 델타 측정이 실제로 동작하는지 확인해야 하므로,
    통과할 공격 1건과 기각될 공격 2건을 일부러 섞는다. 이 응답은 LLM 이 만든 것이
    아니며 성능 근거로 쓰지 않는다 — 배선 검증용이다.
    """
    burden = t1.get("monthly_enrollment_burden")
    atks: list[dict] = []
    if ps.citable_a:
        p = ps.citable_a[0]
        atks.append({
            "target_element": "eligibility.inclusion_criteria", "topic": "recruitment",
            "claim": "같은 적응증에서 모집 실패로 중단된 전례가 있다.",
            "suggestion": "적격기준 임계값을 완화해 모집 가능 모집단을 넓힌다.",
            "citations": [{"kind": "PRECEDENT", "nct_id": p.nct_id,
                           "quote": (p.why_stopped or "")[:60].strip()}],
        })
        atks.append({  # 위조 — T3 가 잡아야 한다
            "target_element": "eligibility.inclusion_criteria", "topic": "recruitment",
            "claim": "존재하지 않는 전례를 인용한 공격.",
            "citations": [{"kind": "PRECEDENT", "nct_id": "NCT09999999",
                           "quote": "terminated for slow accrual across all sites"}],
        })
    atks.append({  # 금지 근거 — T3 가 잡아야 한다
        "target_element": "sponsor_class", "topic": "recruitment",
        "claim": "학술 스폰서라 모집에 실패할 것이다.",
        "citations": [{"kind": "QUANT", "feature": "monthly_enrollment_burden",
                       "value": burden if burden is not None else 1.0}],
    })
    a6 = {
        "revisions": [{"field": "planned_enrollment",
                       "from": None, "to": 40,
                       "rationale": "모집 가능 규모에 맞춰 목표를 낮춘다.",
                       "addresses": "eligibility.inclusion_criteria"}],
        "side_effects": ["목표 등록수만 낮추면 월 모집부담이 실패군 방향으로 내려갈 수 있다."],
    }
    return {"A4": json.dumps(atks, ensure_ascii=False),
            "A6": json.dumps(a6, ensure_ascii=False)}


def refs_for(bench: dict, bucket: str, v0: dict) -> dict:
    out = {}
    for k in ("monthly_enrollment_burden", "planned_enrollment", "planned_duration_months",
              "n_inclusion_items", "n_exclusion_items"):
        if r := bm.reference(bench, bucket, k, v0):
            out[k] = r.get("completed") or r
    return out


def run_one(rec: dict, *, corpus_idx: dict, corpus_rows: list[dict], bench: dict,
            engine: Engine) -> dict:
    nct, bucket = rec["nct_id"], rec["condition_bucket"]
    v0 = rec["v0"]
    cutoff = v0.get("start_date") or ""

    print(f"\n{'='*86}\n{nct}  ·  {bucket}  ·  개시 {cutoff}\n{'='*86}")

    # --- T1 계량 감사 (LLM 0회) ---
    t1 = features_from_v0(v0, nct)
    refs = refs_for(bench, bucket, v0)

    # --- T2 전례 검색 (LLM 0회, 컷오프 하드 필터) ---
    ps = search(corpus_rows, indication=bucket, v0=v0, cutoff=cutoff, exclude_nct=nct)
    print(f"\n〈T2〉전례 검색 — 층={ps.tier} · 완주 {len(ps.completed)} / "
          f"A분류 실패 {len(ps.citable_a)}")

    # --- A1 Planner ---
    base = bm.reference(bench, bucket, "monthly_enrollment_burden", v0) or {}
    bucket_n = (base.get("completed") or {}).get("n")
    p = plan(v0, precedents=ps, bench_bucket_n=bucket_n, engine=engine)
    print(f"\n【A1 Planner】\n{p.render()}")

    if engine.dry_run and engine.use_fixtures:
        engine.fixtures = demo_fixtures(ps, t1)

    # --- A4 ⇄ T3 반송 루프 ---
    feedback: list[str] = []
    attempts: list[dict] = []
    passed: list[Attack] = []
    reject_reasons: Counter = Counter()

    for attempt in range(redteam.MAX_REPAIR + 1):
        if "precedent_recruit" not in p.run and not any(
                k in RISK_ELIGIBLE for k in t1 if t1.get(k) is not None):
            print("\n【A4】 A1 이 모든 공격 경로를 기각했다 — 생성하지 않는다.")
            break
        atks = redteam.generate(v0, precedents=ps, t1_values=t1, eligible=RISK_ELIGIBLE,
                                engine=engine, refs=refs, rejected_feedback=feedback or None)
        if not atks:
            print(f"\n【A4】 시도 {attempt+1}: 공격 0건"
                  f"{' (dry-run 이므로 응답 없음)' if engine.dry_run else ''}")
            break

        feedback = []
        kept = []
        print(f"\n【A4】 시도 {attempt+1}: 공격 {len(atks)}건 생성")
        for a in atks:
            v = verify(a, corpus=corpus_idx, t1_values=t1, cutoff=cutoff)
            if v.passed:
                kept.append(a)
                print(f"   ✓ [{a.target_element}] 통과 (인용 {v.checked}건 대조)")
            else:
                for r in v.reasons:
                    reject_reasons[r] += 1
                print(f"   ✗ [{a.target_element}] 기각 — {'; '.join(v.reasons)}")
                feedback.extend(v.detail)
        attempts.append({"attempt": attempt + 1, "generated": len(atks), "passed": len(kept)})
        passed = kept
        if kept or not feedback:
            break
        print(f"   ↻ 〈T3〉가 전부 기각 → A4 로 반송 (남은 재시도 "
              f"{redteam.MAX_REPAIR - attempt})")

    total_gen = sum(a["generated"] for a in attempts)
    if total_gen:
        print(f"\n〈T3〉인용 검증 기각률 — {total_gen - len(passed)}/{total_gen} = "
              f"{(total_gen - len(passed))/total_gen*100:.0f}%")

    # --- A6 교정 → T1 재실행 ---
    rev = reviser.revise(v0, passed, engine=engine, completed_ref=refs)
    if rev.proposals or rev.deltas:
        print(f"\n【A6 Reviser】\n{rev.render()}")
    elif not engine.dry_run:
        print("\n【A6 Reviser】 통과한 지적이 없어 교정안을 만들지 않는다.")

    return {
        "nct_id": nct, "indication": bucket, "cutoff": cutoff,
        "precedents": {"tier": ps.tier, "n_completed": len(ps.completed),
                       "n_citable_a": len(ps.citable_a), "note": ps.note},
        "plan": {"run": p.run, "rejected": p.rejected, "replans": p.replans,
                 "rule_covered": p.rule_covered, "llm_decided": p.llm_decided},
        "attempts": attempts,
        "n_passed": len(passed),
        "reject_reasons": dict(reject_reasons),
        "revision": {"proposals": rev.proposals,
                     "side_effects": rev.declared_side_effects,
                     "regressions": [d.feature for d in rev.regressions]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="에이전트 루프 실행")
    ap.add_argument("--nct", help="특정 NCT ID 만 실행")
    ap.add_argument("--limit", type=int, default=2, help="처리할 프로토콜 수 (기본 2)")
    ap.add_argument("--dry-run", action="store_true", help="실호출 0회. 프롬프트·예상비용만 확인")
    ap.add_argument("--fixtures", action="store_true",
                    help="dry-run 에서 합성 응답으로 T3 반송 루프·A6 델타까지 배선 검증")
    # 기본값을 대상 수에 맞춰 자동 산정한다. 고정 40 이면 --limit 을 올릴 때마다
    # 상한에 걸려 중간에 끊긴다 (2026-07-31 에 실제로 발생, $0.05 유실).
    ap.add_argument("--max-calls", type=int, default=None,
                    help="전체 LLM 호출 상한 (기본: 대상 수 × 3 + 10)")
    ap.add_argument("--budget-usd", type=float, default=0.10,
                    help="예산 하드 상한(USD). 도달하면 즉시 중단 (기본 0.10)")
    ap.add_argument("--dump-prompts", help="조립된 프롬프트를 JSON 으로 저장")
    args = ap.parse_args()

    llm = LLM()
    if not args.dry_run and not llm.configured:
        raise SystemExit(
            "LLM 이 설정되지 않았습니다. `cp .env.example .env` 후 LLM_API_KEY 를 넣거나,\n"
            "키 없이 검증하려면 --dry-run 을 쓰세요.")

    max_calls = args.max_calls if args.max_calls is not None else args.limit * 3 + 10
    engine = Engine(llm=llm, dry_run=args.dry_run, max_calls=max_calls,
                    use_fixtures=args.fixtures, budget_usd=args.budget_usd)
    print(f"설정: {llm.describe()}")
    print(f"모드: {'DRY-RUN (실호출 0회)' if args.dry_run else '실행'} · "
          f"호출 상한 {max_calls} · 예산 상한 ${args.budget_usd:.4f}")

    rows = load_corpus()
    idx = index_corpus(rows)
    bench = bm.load()

    targets = [r for r in rows if r["nct_id"] == args.nct] if args.nct else \
        [r for r in rows if (r.get("v0") or {}).get("start_date")][:args.limit]
    if not targets:
        raise SystemExit("대상 프로토콜을 찾지 못했습니다.")

    results: list[dict] = []
    stopped = ""

    def save() -> None:
        """**어떤 경로로 끝나든 반드시 저장한다.** 돈은 나갔는데 데이터가 없는 것이 최악이다."""
        OUT.write_text(json.dumps({
            "_meta": {"dry_run": args.dry_run, "n_targets": len(results),
                      "n_requested": len(targets), "llm_calls": len(engine.calls),
                      "provider": llm.provider_name or "(직접지정)",
                      "spent_usd": round(engine.spent(), 6), "stopped": stopped},
            "results": results,
        }, ensure_ascii=False, indent=1))

    try:
        for r in targets:
            try:
                results.append(run_one(r, corpus_idx=idx, corpus_rows=rows,
                                       bench=bench, engine=engine))
            except StopRun as e:
                stopped = str(e)
                print(f"\n⛔ {e}\n   완료 {len(results)}/{len(targets)}건.")
                break
            except Exception as e:                                # noqa: BLE001
                # 개별 프로토콜 실패로 전체를 잃지 않는다.
                stopped = f"{r['nct_id']} 처리 중 오류: {e}"
                print(f"\n  ⚠ {r['nct_id']} 실패 — {str(e)[:120]}")
                continue
            save()          # 프로토콜 하나 끝날 때마다 증분 저장
    finally:
        save()

    print(f"\n{'='*86}\n비용\n{'='*86}")
    if args.dry_run:
        print(engine.estimate_report())
        print("\n  실호출 0회 — 크레딧이 소모되지 않았습니다.")
    else:
        print(engine.ledger.report())

    if args.dump_prompts:
        engine.dump_prompts(args.dump_prompts)
        print(f"\n  프롬프트 {len(engine.calls)}건 → {args.dump_prompts}")

    print(f"  {OUT} 저장 — 프로토콜 {len(results)}건")


if __name__ == "__main__":
    main()
