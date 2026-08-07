"""에이전트 실행 결과 집계 — 신뢰구간을 붙인다.

## 왜 공격 단위가 아니라 프로토콜 단위로 부트스트랩하는가

한 프로토콜에서 나온 공격들은 **같은 프롬프트·같은 전례 목록·같은 계량 수치**를 공유한다.
서로 독립이 아니다. 공격을 독립 표본으로 보고 이항 신뢰구간을 계산하면 **CI 가 부당하게
좁아지고**, 실제보다 안정된 숫자처럼 보인다.

→ 재표집 단위를 **프로토콜(클러스터)** 로 잡는다. 이것이 이 프로젝트가 세 번 기각한
   "과신한 숫자"를 또 만들지 않는 방법이다.

## 산출 지표

| 지표 | 정답 라벨 | 의미 |
|---|---|---|
| T3 인용 검증 기각률 | 불필요 | 환각·자격미달 통제. 사유별 분해 포함 |
| 환각 기각률 | 불필요 | 없는 ID / 인용문 불일치 / 수치 조작만 따로 |
| ID-only 누출 지표 | 불필요 | 재료 없이 검증 가능한 근거를 만드는 비율 |
| 프로토콜당 비용 | — | `Ledger` 실측 |

실행: python3 src/analysis/agent_metrics.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN = Path("data/agent_run.json")
IDONLY = Path("data/idonly_control.json")
OUT = Path("data/agent_metrics.json")

N_BOOT = 2000
SEED = 20260731

# 환각으로 분류하는 기각 사유 — 모델이 없는 것을 만들어낸 경우만.
# "검증을 통과하지 못한 피처를 위험 근거로 사용"은 환각이 아니라 자격 미달이므로 제외한다.
HALLUCINATION = {
    "전례 NCT ID 가 코퍼스에 없음 (환각)",
    "전례 인용문이 원문과 불일치 (환각)",
    "수치가 T1 출력과 불일치 (환각)",
}


def cluster_boot_ci(clusters: list[tuple[int, int]], rng: random.Random) -> tuple[float, float] | None:
    """clusters = [(분자, 분모), ...] 를 클러스터 단위로 재표집한다."""
    clusters = [c for c in clusters if c[1] > 0]
    if len(clusters) < 2:
        return None
    boots = []
    for _ in range(N_BOOT):
        s = [rng.choice(clusters) for _ in clusters]
        num = sum(x[0] for x in s)
        den = sum(x[1] for x in s)
        if den:
            boots.append(num / den)
    if len(boots) < N_BOOT // 2:
        return None
    boots.sort()
    return boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def main() -> None:
    rng = random.Random(SEED)
    d = json.loads(RUN.read_text())
    rs = d["results"]

    # --- 프로토콜별 (기각 수, 생성 수) ---
    per: list[tuple[int, int]] = []
    hall_per: list[tuple[int, int]] = []
    reasons: Counter = Counter()
    for r in rs:
        gen = sum(a["generated"] for a in r["attempts"])
        rej = gen - r["n_passed"]
        if gen:
            per.append((rej, gen))
            h = sum(v for k, v in r["reject_reasons"].items() if k in HALLUCINATION)
            hall_per.append((min(h, gen), gen))
        reasons.update(r["reject_reasons"])

    gen_t = sum(x[1] for x in per)
    rej_t = sum(x[0] for x in per)
    hall_t = sum(x[0] for x in hall_per)

    ci = cluster_boot_ci(per, rng)
    ci_h = cluster_boot_ci(hall_per, rng)

    print(f"\n{'='*78}\n에이전트 실행 지표 (프로토콜 단위 클러스터 부트스트랩 {N_BOOT}회)\n{'='*78}")
    print(f"  프로토콜 {len(per)}건 · 공격 생성 {gen_t}건 · LLM 호출 {d['_meta']['llm_calls']}회\n")

    def line(name, num, den, c):
        s = f"{num}/{den} = {num/den*100:.1f}%" if den else "—"
        cs = f"  95% CI [{c[0]*100:.1f}, {c[1]*100:.1f}]" if c else "  (CI 산출 불가)"
        print(f"  {name:<30}{s:>20}{cs}")

    line("T3 인용 검증 기각률", rej_t, gen_t, ci)
    line("└ 그중 환각 기각률", hall_t, gen_t, ci_h)

    print("\n  기각 사유 분해:")
    for k, v in reasons.most_common():
        tag = " [환각]" if k in HALLUCINATION else ""
        print(f"    {v:>4}건 ({v/max(rej_t,1)*100:>5.1f}%)  {k}{tag}")

    out: dict = {
        "_meta": {"n_boot": N_BOOT, "seed": SEED,
                  "resample_unit": "프로토콜 (클러스터). 공격은 같은 프롬프트를 공유해 독립이 아니다.",
                  "n_protocols": len(per), "n_attacks": gen_t,
                  "llm_calls": d["_meta"]["llm_calls"]},
        "t3_rejection": {"n": rej_t, "of": gen_t, "rate": round(rej_t/gen_t, 4) if gen_t else None,
                         "ci95": [round(x, 4) for x in ci] if ci else None},
        "hallucination": {"n": hall_t, "of": gen_t,
                          "rate": round(hall_t/gen_t, 4) if gen_t else None,
                          "ci95": [round(x, 4) for x in ci_h] if ci_h else None},
        "reject_reasons": dict(reasons),
    }

    # --- ID-only 대조 ---
    if IDONLY.exists():
        c = json.loads(IDONLY.read_text())
        pc, pv = c["n_precedent_citations"], c["n_precedent_verified"]
        print(f"\n  ID-only 음성 대조 (프로토콜 {c['_meta']['n_targets']}건)")
        print(f"    공격 생성 {c['n_generated']}건 · T3 통과 {c['n_passed']}건 "
              f"→ 통과율 {c['n_passed']/max(c['n_generated'],1)*100:.1f}%")
        print(f"    전례 인용 {pc}건 · 검증 통과 {pv}건 → 누출 지표 "
              f"{pv/pc*100 if pc else 0:.1f}%")
        full_pass = (gen_t - rej_t) / gen_t if gen_t else 0
        idonly_pass = c["n_passed"] / max(c["n_generated"], 1)
        print(f"    ★ 대비: 전체 맥락 {full_pass*100:.1f}% vs ID만 {idonly_pass*100:.1f}%")
        out["idonly"] = {"n_generated": c["n_generated"], "n_passed": c["n_passed"],
                         "leakage_rate": c["leakage_rate"],
                         "contrast": {"full_context_pass": round(full_pass, 4),
                                      "id_only_pass": round(idonly_pass, 4)}}

    print("\n  ⚠ 표본이 작다. 이 숫자는 성능 보증이 아니라 **관측된 동작 범위**로 보고한다.")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  {OUT} 저장\n")


if __name__ == "__main__":
    main()
