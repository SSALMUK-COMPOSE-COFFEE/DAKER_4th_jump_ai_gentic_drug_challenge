"""T3 게이트 자가 시험 — 합성 위조 공격으로 검출력을 측정한다.

## 이 시험이 재는 것과 재지 않는 것

**재는 것**: T3 가 기계적 위조를 잡아내는 비율. 위조 유형을 우리가 만들어 넣으므로
정답을 알고 있고, 정답 라벨이 필요 없다.

**재지 않는 것**: LLM 이 실제로 인용을 얼마나 지어내는지. 그건 A4 가 돌아야만 알 수 있고,
제안서 항목 4의 "인용 검증 기각률"은 그쪽 숫자다. **두 수치를 같은 칸에 쓰지 않는다.**

또한 이 시험은 **맥락 왜곡**(인용문을 정확히 복사하되 다른 뜻으로 쓰는 것)을 다루지 못한다.
T3 의 구조적 한계이며, 검출 불가 항목으로 함께 보고한다.

실행: python3 src/audit/referee.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.contracts import Attack, Citation  # noqa: E402
from audit.referee import index_corpus, verify  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
OUT = Path("data/t3_selftest.json")


def build_cases(corpus: dict[str, dict]) -> list[dict]:
    """실제 코퍼스 레코드에서 정상 공격과 위조 공격을 합성한다."""
    # A 분류 실패 전례 중 인용문이 충분히 긴 것만 재료로 쓴다.
    mats = []
    for r in corpus.values():
        if r.get("_category") != "A":
            continue
        why = (r.get("labels") or {}).get("why_stopped") or ""
        sd = (r.get("v0") or {}).get("start_date") or ""
        if len(why) >= 30 and sd:
            mats.append((r["nct_id"], why, sd))
    mats.sort()

    nonA = [(r["nct_id"], (r.get("labels") or {}).get("why_stopped") or "",
             (r.get("v0") or {}).get("start_date") or "")
            for r in corpus.values()
            if r.get("_category") in ("B", "C") and (r.get("v0") or {}).get("start_date")
            and len((r.get("labels") or {}).get("why_stopped") or "") >= 30]
    nonA.sort()

    cases: list[dict] = []
    N = min(40, len(mats))

    def quote(w: str) -> str:
        return w[:60].strip()

    for i in range(N):
        nct, why, sd = mats[i]
        # 컷오프는 전례보다 뒤여야 정상이다.
        cutoff = "2099-01-01"

        cases.append(dict(
            kind="정상 (통과해야 함)", expect_pass=True, cutoff=cutoff,
            attack=Attack(target_element="eligibility.inclusion_criteria",
                          claim="같은 적응증에서 모집 실패로 중단된 전례가 있다.",
                          citations=[Citation(kind="PRECEDENT", nct_id=nct, quote=quote(why))])))

        cases.append(dict(
            kind="존재하지 않는 NCT ID", expect_pass=False, cutoff=cutoff,
            attack=Attack(target_element="eligibility.inclusion_criteria", claim="…",
                          citations=[Citation(kind="PRECEDENT", nct_id="NCT09999999",
                                              quote=quote(why))])))

        cases.append(dict(
            kind="인용문 변조", expect_pass=False, cutoff=cutoff,
            attack=Attack(target_element="eligibility.inclusion_criteria", claim="…",
                          citations=[Citation(kind="PRECEDENT", nct_id=nct,
                                              quote="terminated due to catastrophic toxicity in all arms")])))

        cases.append(dict(
            kind="시간 컷오프 위반 (미래 전례)", expect_pass=False, cutoff="2000-01-01",
            attack=Attack(target_element="eligibility.inclusion_criteria", claim="…",
                          citations=[Citation(kind="PRECEDENT", nct_id=nct, quote=quote(why))])))

        cases.append(dict(
            kind="인용문이 너무 짧음", expect_pass=False, cutoff=cutoff,
            attack=Attack(target_element="eligibility.inclusion_criteria", claim="…",
                          citations=[Citation(kind="PRECEDENT", nct_id=nct, quote=why[:6])])))

        if i < len(nonA):
            n2, w2, _ = nonA[i]
            cases.append(dict(
                kind="모집 공격에 비A분류 전례", expect_pass=False, cutoff=cutoff,
                attack=Attack(target_element="eligibility.inclusion_criteria", claim="…",
                              topic="recruitment",
                              citations=[Citation(kind="PRECEDENT", nct_id=n2, quote=quote(w2))])))

        cases.append(dict(
            kind="수치 조작 (QUANT)", expect_pass=False, cutoff=cutoff,
            t1={"monthly_enrollment_burden": 3.20},
            attack=Attack(target_element="planned_enrollment", claim="…",
                          citations=[Citation(kind="QUANT", feature="monthly_enrollment_burden",
                                              value=41.7)])))

        cases.append(dict(
            kind="검증 미통과 피처를 근거로", expect_pass=False, cutoff=cutoff,
            t1={"n_secondary_outcomes": 11.0},
            attack=Attack(target_element="secondary_outcomes", claim="…",
                          citations=[Citation(kind="QUANT", feature="n_secondary_outcomes",
                                              value=11.0)])))

        cases.append(dict(
            kind="금지 근거 (스폰서 유형)", expect_pass=False, cutoff=cutoff,
            attack=Attack(target_element="sponsor_class", claim="…",
                          citations=[Citation(kind="PRECEDENT", nct_id=nct, quote=quote(why))])))

        cases.append(dict(
            kind="무근거 공격 (인용 0)", expect_pass=False, cutoff=cutoff,
            attack=Attack(target_element="eligibility.inclusion_criteria",
                          claim="이 프로토콜은 모집이 어려워 보인다.", citations=[])))

    return cases


def main() -> None:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    corpus = index_corpus(rows)
    cases = build_cases(corpus)

    agg: dict[str, dict] = {}
    reason_counter: dict[str, int] = {}
    for c in cases:
        v = verify(c["attack"], corpus=corpus,
                   t1_values=c.get("t1", {"monthly_enrollment_burden": 3.20}),
                   cutoff=c["cutoff"])
        a = agg.setdefault(c["kind"], {"n": 0, "correct": 0, "expect_pass": c["expect_pass"]})
        a["n"] += 1
        a["correct"] += int(v.passed == c["expect_pass"])
        for r in v.reasons:
            reason_counter[r] = reason_counter.get(r, 0) + 1

    print(f"\n{'='*84}")
    print("T3 인용 대조 게이트 — 합성 위조 자가 시험")
    print(f"코퍼스 {len(corpus)}건 · 합성 공격 {len(cases)}건 · LLM 호출 0회")
    print(f"{'='*84}")
    print(f"  {'위조 유형':<34}{'건수':>6}{'기대':>8}{'적중률':>10}")
    print("  " + "-"*80)
    total = correct = 0
    for k, a in agg.items():
        rate = a["correct"] / a["n"] * 100
        exp = "통과" if a["expect_pass"] else "기각"
        flag = "" if rate == 100 else "   ← 미검출 있음"
        print(f"  {k:<34}{a['n']:>6}{exp:>8}{rate:>9.1f}%{flag}")
        total += a["n"]
        correct += a["correct"]
    print("  " + "-"*80)
    print(f"  {'전체':<34}{total:>6}{'':>8}{correct/total*100:>9.1f}%")

    forged = sum(a["n"] for a in agg.values() if not a["expect_pass"])
    caught = sum(a["correct"] for a in agg.values() if not a["expect_pass"])
    valid_n = sum(a["n"] for a in agg.values() if a["expect_pass"])
    valid_ok = sum(a["correct"] for a in agg.values() if a["expect_pass"])
    print(f"\n  위조 검출률   {caught}/{forged} = {caught/forged*100:.1f}%")
    print(f"  정상 통과율   {valid_ok}/{valid_n} = {valid_ok/valid_n*100:.1f}%  "
          f"(오기각률 {100-valid_ok/valid_n*100:.1f}%)")
    print("\n  기각 사유 분해:")
    for r, n in sorted(reason_counter.items(), key=lambda x: -x[1]):
        print(f"    {n:>4}  {r}")

    print("\n  ※ 이 시험이 재지 않는 것 — LLM 이 실제로 인용을 지어내는 비율."
          "\n     그것은 A4 가 돌아야 측정되며, 제안서 항목 4의 '인용 검증 기각률'은 그쪽 숫자다."
          "\n  ※ 검출 불가 (구조적 한계) — 인용문을 정확히 복사하되 맥락을 왜곡하는 공격.")

    OUT.write_text(json.dumps({
        "_meta": {"n_corpus": len(corpus), "n_cases": len(cases), "llm_calls": 0,
                  "measures": "게이트의 기계적 위조 검출력",
                  "does_not_measure": "LLM 의 실제 환각률 (A4 구현 후 별도 측정)"},
        "by_kind": agg,
        "reject_reasons": reason_counter,
        "forgery_detection_rate": round(caught / forged, 4),
        "false_rejection_rate": round(1 - valid_ok / valid_n, 4),
    }, ensure_ascii=False, indent=1))
    print(f"\n  {OUT} 저장\n")


if __name__ == "__main__":
    main()
