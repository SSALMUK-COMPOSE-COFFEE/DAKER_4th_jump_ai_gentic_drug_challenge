"""적격기준 의미 범주의 층화 판별력 실측 — "항목 단위 전례 대조" 가설의 통과/기각 판정.

## 왜 이 스크립트가 필요했나

`validated-numbers.md` 기준으로 남은 모순: 검증을 통과한 피처(월 모집부담)는 방향이 반대라
교정 조언을 만들 수 없고, 조언이 나오는 피처(적격기준 항목수)는 적응증 층화 AUC 0.568,
CI 하한 0.50 으로 구분력이 죽었다.

출구 가설: **항목 수(스칼라) 대신 의미 범주(어떤 기준을 두었는가)를 전례와 대조**하면
교정 가능한 근거가 나온다. 이 스크립트는 그 가설이 실측으로 성립하는지만 판정한다.

## 방법론 — 이 프로젝트가 이미 저지른 실수를 반복하지 않기 위한 장치 4개

1. **층화 필수.** `competition_signal.py` 는 미층화 AUC 0.682 만 보고 "통과" 판정을 내렸고,
   적응증 층화에서 0.463 으로 무너졌다(`validated-numbers.md` §3.1). 미층화 값은
   참고용으로만 출력하고 결론에 쓰지 않는다.
2. **음성 대조군 필수.** `stratified_signal.indication_only_control()` 를 그대로 재사용한다 —
   프로토콜과 날짜를 전부 버리고 적응증 라벨만 쓰는 예측기보다 못하면 질환 판별기다.
3. **분량 교란 층 추가.** 의미 범주 검출은 본질적으로 텍스트가 길면 더 많이 걸린다. 그리고
   실패군의 적격기준 글자수가 완주군보다 1.4배 길다(`design-v2.md` §2.4 계열 관측). 따라서
   `n_items` 3분위를 층으로 고정해 **"범주가 있다"가 "글이 길다"의 재인코딩인지** 확인한다.
   이 층을 넣지 않으면 이미 기각된 `eligibility_chars` 를 이름만 바꿔 되살리는 셈이 된다.
4. **다중검정 보정.** 범주 20개를 동시에 검정하므로 95% CI 하한 하나만으로 통과 판정하면
   기대 위양성이 1건이다. 부트스트랩 단측 p(= AUC ≤ 0.5 인 복제 비율)를 함께 내고
   Bonferroni 기준 α = 0.05/20 = 0.0025 와 비교한다.

## 이진 피처 AUC

범주 존재/부재는 이진값이므로 층내 AUC = 0.5 + (출현률차)/2 이다. 즉 출현률 차이 20%p 가
AUC 0.60 에 대응한다. 이 대응 때문에 AUC 와 출현률 차이를 둘 다 보고한다 — AUC 는 층화·가중을
기존 코드와 동일하게 처리하기 위해, 출현률 차이는 사용자에게 보여줄 문장을 만들기 위해.

계산은 이진 특성을 이용한 카운트 기반 구현(`_strat_auc_binary`)으로 하고,
`stratified_signal.stratified_auc()` 와 값이 일치하는지 실행 시마다 검산해서 출력한다
(부트스트랩 1,000회 × 범주 20개 × 층 6개를 쌍 순회로 돌리면 시간이 나오지 않는다).

출력: data/criteria_signal.json
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import group_of  # noqa: E402
from analysis.stratified_signal import indication_only_control, stratified_auc  # noqa: E402
from audit.benchmarks import HIGHER_IS_RISKIER, phase_stratum  # noqa: E402
from audit.criteria_parse import CONCEPT_NAMES, CONCEPTS, parse  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
OUT = Path("data/criteria_signal.json")

N_BOOT = 1000
BOOT_SEED = 20260730
ALPHA_BONF = 0.05 / len(CONCEPTS)

CONCEPT_KEYS = [c for c, _ in CONCEPTS]
EXTRA_KEYS = ["n_concepts", "n_items"]  # 분량 재인코딩 확인용 대조 피처

# `stratified_auc()` 는 HIGHER_IS_RISKIER 에 없는 키의 방향을 뒤집는다. 의미 범주는 전부
# "그 기준을 두었으면 위험 후보"라는 단일 방향으로 재므로 여기 등록해 둔다.
# (등록하지 않으면 재사용한 함수가 전부 1-AUC 를 돌려준다.)
HIGHER_IS_RISKIER.update(CONCEPT_KEYS)
HIGHER_IS_RISKIER.update(EXTRA_KEYS)

TAXONOMY = Path("src/labels/taxonomy.py")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"


POS = "중단-모집실패"
NEG = "완주"
SPECIFICITY = [("중단-과학적", "효능·독성 실패"), ("중단-경영", "경영 판단 중단")]


def load_records() -> list[dict]:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    recs = []
    for r in rows:
        p = parse(r["v0"].get("eligibility_criteria"))
        f: dict[str, float | None] = {k: (1.0 if k in p["concepts"] else 0.0) for k in CONCEPT_KEYS}
        f["n_concepts"] = float(len(p["concepts"]))
        f["n_items"] = float(p["n_items"])
        recs.append(
            {
                "nct": r["nct_id"],
                "ind": r["condition_bucket"],
                "phase": phase_stratum(r["v0"]),
                "sponsor": r["v0"].get("sponsor_class") or "UNKNOWN",
                "group": group_of(r),
                "why": r["labels"].get("why_stopped"),
                "parse": p,
                "f": f,
            }
        )
    # n_items 3분위 층 (분량 교란 통제)
    vals = sorted(r["f"]["n_items"] for r in recs)
    t1, t2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    for r in recs:
        v = r["f"]["n_items"]
        r["items_tertile"] = "T1" if v <= t1 else ("T2" if v <= t2 else "T3")
    return recs


STRATA = {
    "none": lambda r: "_",
    "indication": lambda r: r["ind"],
    "phase": lambda r: r["phase"],
    "indication+phase": lambda r: f"{r['ind']}|{r['phase']}",
    "items_tertile": lambda r: r["items_tertile"],
    "indication+items": lambda r: f"{r['ind']}|{r['items_tertile']}",
    # 스폰서 유형은 `design-v2.md` §6 이 지목한 최강 교란원이다 (모집실패군 학술 74.4% vs
    # 완주군 산업 70.0%). 이 층을 통과하지 못하면 "학술 스폰서면 실패한다"는
    # 교정 불가능한 조언을 적격기준 이름으로 포장한 것에 불과하다.
    "sponsor": lambda r: r["sponsor"],
    "indication+sponsor": lambda r: f"{r['ind']}|{r['sponsor']}",
}

# 결론에 쓰는 층. `none` 은 참고용, `phase` 는 적응증 교란을 못 잡으므로 판정에서 제외한다.
DECISIVE_STRATA = (
    "indication", "indication+phase", "items_tertile", "indication+items",
    "sponsor", "indication+sponsor",
)


def _strat_auc_binary(
    data: list[dict], key: str, stratum, pos_group: str, neg_group: str
) -> tuple[float | None, int, int, float | None]:
    """이진 피처용 층화 AUC. 층내 카운트만으로 계산하고 쌍 수로 가중 평균한다.

    반환: (AUC, n_pos, n_neg, 쌍수 가중 출현률차)
    """
    cnt: dict[str, list[int]] = {}
    for r in data:
        v = r["f"].get(key)
        if v is None:
            continue
        g = r["group"]
        if g != pos_group and g != neg_group:
            continue
        c = cnt.setdefault(stratum(r), [0, 0, 0, 0])  # pos1,pos0,neg1,neg0
        one = v >= 0.5
        if g == pos_group:
            c[0 if one else 1] += 1
        else:
            c[2 if one else 3] += 1
    num = den = dnum = 0.0
    n_pos = n_neg = 0
    for p1, p0, q1, q0 in cnt.values():
        np_, nn_ = p1 + p0, q1 + q0
        if not np_ or not nn_:
            continue
        n_pos += np_
        n_neg += nn_
        w = np_ * nn_
        a = (p1 * q0 + 0.5 * (p1 * q1 + p0 * q0)) / w
        num += a * w
        den += w
        dnum += (p1 / np_ - q1 / nn_) * w
    if not den:
        return None, 0, 0, None
    return round(num / den, 4), n_pos, n_neg, round(dnum / den, 4)


def strat_auc(
    data: list[dict], key: str, stratum, pos_group: str, neg_group: str
) -> tuple[float | None, int, int, float | None]:
    """이진 범주는 카운트 기반 고속 경로, 연속 대조 피처는 기존 `stratified_auc()` 를 쓴다.

    `n_items` / `n_concepts` 를 이진 경로에 넣으면 전부 1로 뭉개져 AUC 가 0.5 로 나온다.
    실행 시 검산(`crosscheck`)이 이 오류를 실제로 잡아냈다.
    """
    if key in CONCEPT_KEYS:
        return _strat_auc_binary(data, key, stratum, pos_group, neg_group)
    a, np_, nn_ = stratified_auc(data, key, stratum, pos_group, neg_group)
    return a, np_, nn_, None


def boot(
    data: list[dict], key: str, stratum, pos_group: str, neg_group: str, rng: random.Random
) -> tuple[tuple[float, float] | None, float | None]:
    """(95% CI, 단측 부트스트랩 p = AUC ≤ 0.5 인 복제 비율)."""
    boots = []
    n = len(data)
    for _ in range(N_BOOT):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        a, _, _, _ = strat_auc(sample, key, stratum, pos_group, neg_group)
        if a is not None:
            boots.append(a)
    if len(boots) < N_BOOT // 2:
        return None, None
    boots.sort()
    ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))])
    p = sum(1 for b in boots if b <= 0.5) / len(boots)
    return ci, p


def crosscheck(recs: list[dict]) -> list[str]:
    """카운트 기반 구현이 기존 `stratified_auc()` 와 같은 값을 내는지 검산.

    이 검산이 실제로 버그를 잡았다: 처음엔 `n_items` 같은 연속 대조 피처도 이진 경로로
    보냈고, `v >= 0.5` 이진화 때문에 전부 1이 되어 AUC 가 0.495 로 나왔다(참값 0.596).
    그래서 연속 피처는 `strat_auc()` 에서 기존 함수로 분기한다.
    """
    msgs = []
    for key in (CONCEPT_KEYS[0], CONCEPT_KEYS[5], CONCEPT_KEYS[13]):
        for s_name in ("none", "indication", "indication+phase", "indication+items"):
            a1, p1, n1, _ = _strat_auc_binary(recs, key, STRATA[s_name], POS, NEG)
            a2, p2, n2 = stratified_auc(recs, key, STRATA[s_name], POS, NEG)
            ok = a1 is not None and a2 is not None and abs(a1 - a2) < 1e-3 and (p1, n1) == (p2, n2)
            msgs.append(f"    {key:<26}{s_name:<18}fast={a1} ref={a2}  {'일치' if ok else '★불일치'}")
    return msgs


def prevalence(recs: list[dict], key: str, group: str) -> tuple[int, int]:
    sub = [r for r in recs if r["group"] == group and r["f"].get(key) is not None]
    return sum(1 for r in sub if r["f"][key] >= 0.5), len(sub)


# --- 실제 산출물 예시 ------------------------------------------------------

DEMO = Path("data/demo_cases.json")


def example_statements(recs: list[dict], verdict: dict[str, str], n_cases: int = 3) -> list[dict]:
    """데모 케이스 중 모집실패 건에 대해 실제 데이터로 서술을 생성한다.

    체리피킹 방지: `demo_cases.json` 의 선정 규칙(홀드아웃 test 분할 + NCT 사전순)을 그대로
    받아 **앞에서부터** n건을 쓴다. 감사 결과를 보고 고르지 않는다.

    범주 선택도 규칙으로 고정한다 — 해당 케이스가 실제로 가진 범주 중에서
    같은 적응증·Phase 층의 출현률 차이가 가장 큰 것. 자기 자신은 참조 분포에서 제외한다.
    각 서술에는 그 범주의 층화 검증 판정(`verdict`)을 반드시 함께 붙인다.
    """
    if not DEMO.exists():
        return []
    cases = [c for c in json.loads(DEMO.read_text())["cases"] if c["group"] == POS][:n_cases]
    by_nct = {r["nct"]: r for r in recs}
    out = []
    for c in cases:
        me = by_nct.get(c["nct_id"])
        if not me:
            continue
        peers = [
            r for r in recs
            if r["nct"] != me["nct"] and r["ind"] == me["ind"] and r["phase"] == me["phase"]
        ]
        have = [k for k in CONCEPT_KEYS if me["f"][k] >= 0.5]
        ranked = []
        for k in have:
            p1, pn = prevalence(peers, k, POS)
            c1, cn = prevalence(peers, k, NEG)
            if pn < 3 or cn < 5:
                continue
            ranked.append((p1 / pn - c1 / cn, k, p1, pn, c1, cn))
        ranked.sort(reverse=True)
        if not ranked:
            out.append({"nct_id": c["nct_id"], "error": "층 표본 부족으로 서술 생성 불가"})
            continue
        d, k, p1, pn, c1, cn = ranked[0]
        prec = [
            {"nct": r["nct"], "why_stopped": (r["why"] or "")[:200]}
            for r in peers if r["group"] == POS and r["f"][k] >= 0.5
        ][:4]
        out.append({
            "nct_id": c["nct_id"], "indication": me["ind"], "phase": me["phase"],
            "concept": k, "concept_name": CONCEPT_NAMES[k],
            "criterion_quoted": me["parse"]["concepts"][k]["item"],
            "match_evidence": me["parse"]["concepts"][k]["evidence"],
            "section": me["parse"]["concepts"][k]["section"],
            "completed": [c1, cn], "recruit_failed": [p1, pn], "rate_diff": round(d, 3),
            "precedents": prec,
            "verdict_of_concept": verdict.get(k, "판별력 미입증"),
            "actual_why_stopped": (c.get("reveal") or {}).get("why_stopped"),
            "actual_attainment": (c.get("reveal") or {}).get("enrollment_attainment"),
        })
    return out


def main() -> None:
    recs = load_records()
    rng = random.Random(BOOT_SEED)
    n_pos = sum(1 for r in recs if r["group"] == POS)
    n_neg = sum(1 for r in recs if r["group"] == NEG)

    print(f"코퍼스 {len(recs)}건 — {POS} {n_pos}건 / {NEG} {n_neg}건")
    print(f"부트스트랩 {N_BOOT}회 seed {BOOT_SEED}, Bonferroni α = {ALPHA_BONF:.4f} (범주 {len(CONCEPTS)}개)")

    print("\n=== 구현 검산 (카운트 기반 vs stratified_signal.stratified_auc) ===")
    for m in crosscheck(recs):
        print(m)

    ctrl = indication_only_control(recs, POS, NEG)
    print(f"\n=== 음성 대조군 ===\n  적응증 라벨만 쓰는 예측기 AUC = {ctrl}"
          "  ← 미층화 값이 이보다 낮으면 질환 판별기")

    result: dict = {
        "_meta": {
            "source": str(RAW), "n_rows": len(recs), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            # 코호트 정의가 `taxonomy.classify()` 에 의존한다. 이 파일이 세션 중 갱신되어
            # A분류 건수가 112 → 121 로 바뀐 일이 실제로 있었다. 숫자를 특정 상태에 못박는다.
            "corpus_sha256_16": _sha(RAW), "taxonomy_sha256_16": _sha(TAXONOMY),
            "n_concepts_tested": len(CONCEPTS), "alpha_bonferroni": ALPHA_BONF,
            "n_pos": n_pos, "n_neg": n_neg, "indication_only_control": ctrl,
            "decisive_strata": list(DECISIVE_STRATA),
            "note": "이진 범주 AUC = 0.5 + 출현률차/2. 결론은 DECISIVE_STRATA 만 사용한다.",
        },
        "concepts": {},
    }

    print("\n=== 출현률 (참고용, 미층화) ===")
    print(f"  {'범주':<28}{'완주':>14}{'모집실패':>14}{'차이':>9}")
    for key in CONCEPT_KEYS + EXTRA_KEYS:
        if key in EXTRA_KEYS:
            cv = [r["f"][key] for r in recs if r["group"] == NEG]
            pv = [r["f"][key] for r in recs if r["group"] == POS]
            print(f"  {key:<28}{'중앙 '+str(st.median(cv)):>14}{'중앙 '+str(st.median(pv)):>14}")
            continue
        c1, cn = prevalence(recs, key, NEG)
        p1, pn = prevalence(recs, key, POS)
        print(f"  {key:<28}{f'{c1}/{cn} ({c1/cn:.0%})':>14}"
              f"{f'{p1}/{pn} ({p1/pn:.0%})':>14}{(p1/pn - c1/cn):>+9.1%}")

    print("\n=== 층화 판별력 (모집실패 vs 완주) ===")
    hdr = f"  {'범주':<28}" + "".join(f"{s:>19}" for s in STRATA)
    print(hdr)
    print("  " + "-" * (28 + 19 * len(STRATA)))

    for key in CONCEPT_KEYS + EXTRA_KEYS:
        cells, per = [], {}
        for s_name, s_fn in STRATA.items():
            a, np_, nn_, dr = strat_auc(recs, key, s_fn, POS, NEG)
            if a is None:
                cells.append(f"{'—':>19}")
                per[s_name] = None
                continue
            ci, p = boot(recs, key, s_fn, POS, NEG, rng)
            mark = ""
            if ci and p is not None:
                mark = "**" if (ci[0] > 0.5 and p < ALPHA_BONF) else ("*" if ci[0] > 0.5 else "")
            txt = f"{a:.3f}{mark} {ci[0]:.2f}-{ci[1]:.2f}" if ci else f"{a:.3f}{mark}"
            cells.append(f"{txt:>19}")
            per[s_name] = {"auc": a, "ci95": list(ci) if ci else None, "boot_p": p,
                           "n_pos": np_, "n_neg": nn_, "rate_diff": dr}
        print(f"  {key:<28}" + "".join(cells))
        c1, cn = prevalence(recs, key, NEG) if key in CONCEPT_KEYS else (None, None)
        p1, pn = prevalence(recs, key, POS) if key in CONCEPT_KEYS else (None, None)
        result["concepts"][key] = {
            "name": CONCEPT_NAMES.get(key, key),
            "prevalence": {"completed": [c1, cn], "recruit_failed": [p1, pn]},
            "strata": per,
        }

    # 판정: 결론용 층 전부에서 CI 하한 > 0.5 이고 Bonferroni 통과
    passed = [
        k for k in CONCEPT_KEYS
        if all(
            (result["concepts"][k]["strata"].get(s) or {}).get("ci95")
            and result["concepts"][k]["strata"][s]["ci95"][0] > 0.5
            and result["concepts"][k]["strata"][s]["boot_p"] < ALPHA_BONF
            for s in DECISIVE_STRATA
        )
    ]
    weak = [
        k for k in CONCEPT_KEYS
        if k not in passed and all(
            (result["concepts"][k]["strata"].get(s) or {}).get("ci95")
            and result["concepts"][k]["strata"][s]["ci95"][0] > 0.5
            for s in DECISIVE_STRATA
        )
    ]
    print(f"\n  * = 95% CI 하한>0.5    ** = 그리고 Bonferroni p<{ALPHA_BONF:.4f}")
    print(f"  ★ 결론용 층 전부 통과 (Bonferroni 포함): {passed or '없음'}")
    print(f"  ☆ 결론용 층 전부 CI 하한>0.5 이나 Bonferroni 미달: {weak or '없음'}")

    # 적응증별 방향 점검. 경쟁 밀도는 15개 중 11개가 0.5 이하였다(validated-numbers §3.1).
    # 통합값이 소수 적응증에 끌려간 것이 아닌지 확인한다.
    print("\n=== 상위 범주의 적응증별 방향 (표본 부족 층 포함, 판정 보류 근거) ===")
    per_ind: dict = {}
    # 선정 기준을 `passed + weak` 로 두면 전부 기각일 때 이 표가 통째로 사라져
    # 문서가 인용한 숫자를 산출물에서 재현할 수 없다. 결정적 층 통과 개수로 뽑는다.
    n_pass_strata = {
        k: sum(1 for s in DECISIVE_STRATA
               if (result["concepts"][k]["strata"].get(s) or {}).get("ci95")
               and result["concepts"][k]["strata"][s]["ci95"][0] > 0.5)
        for k in CONCEPT_KEYS
    }
    result["_meta"]["n_decisive_strata_passed"] = n_pass_strata
    top = [k for k in CONCEPT_KEYS if n_pass_strata[k] >= 4]
    for key in top:
        print(f"\n  [{key}] {CONCEPT_NAMES[key]}")
        print(f"    {'적응증':<32}{'실패군':>14}{'완주군':>14}{'차이':>9}")
        rows_ = []
        for ind in sorted({r["ind"] for r in recs}):
            sub = [r for r in recs if r["ind"] == ind]
            p1, pn = prevalence(sub, key, POS)
            c1, cn = prevalence(sub, key, NEG)
            if not pn or not cn:
                continue
            d = p1 / pn - c1 / cn
            rows_.append({"indication": ind, "pos": [p1, pn], "neg": [c1, cn], "rate_diff": round(d, 3)})
            flag = "" if pn >= 5 else "  (실패군 n<5, 판정 보류)"
            print(f"    {ind:<32}{f'{p1}/{pn}':>14}{f'{c1}/{cn}':>14}{d:>+9.1%}{flag}")
        pos_dir = sum(1 for x in rows_ if x["rate_diff"] > 0)
        big = [x for x in rows_ if x["pos"][1] >= 5]
        pos_big = sum(1 for x in big if x["rate_diff"] > 0)
        print(f"    → 방향 일치 {pos_dir}/{len(rows_)} 적응증"
              f" (실패군 n≥5 인 층만: {pos_big}/{len(big)})")
        per_ind[key] = {"rows": rows_, "n_same_direction": pos_dir, "n_total": len(rows_),
                        "n_same_direction_n5": pos_big, "n_total_n5": len(big)}
    result["per_indication"] = per_ind

    # 특이도 — 예측해선 안 되는 집단에도 발화하는지. 적응증 층만 (계산량 제한).
    print("\n=== 특이도 (적응증 층화) — 표적 이외 집단에도 발화하면 '이상한 시험 감지기'다 ===")
    print(f"  {'범주':<28}{'모집실패':>22}{'효능·독성실패':>22}{'경영중단':>22}")
    spec: dict = {}
    for key in CONCEPT_KEYS:
        row = []
        a0 = result["concepts"][key]["strata"]["indication"]
        row.append(f"{a0['auc']:.3f} {a0['ci95'][0]:.2f}-{a0['ci95'][1]:.2f}" if a0 and a0["ci95"] else "—")
        spec[key] = {}
        for grp, _label in SPECIFICITY:
            a, np_, nn_, dr = strat_auc(recs, key, STRATA["indication"], grp, NEG)
            ci, p = boot(recs, key, STRATA["indication"], grp, NEG, rng) if a is not None else (None, None)
            spec[key][grp] = {"auc": a, "ci95": list(ci) if ci else None, "boot_p": p,
                              "n_pos": np_, "rate_diff": dr}
            row.append(f"{a:.3f} {ci[0]:.2f}-{ci[1]:.2f}" if ci else (f"{a:.3f}" if a else "—"))
        print(f"  {key:<28}" + "".join(f"{c:>22}" for c in row))
    result["specificity_indication_stratified"] = spec
    result["_meta"]["passed_all_decisive_strata"] = passed
    result["_meta"]["passed_ci_only"] = weak

    # 층별 표본 수 — 표본 부족으로 판정 보류할 범주를 명시하기 위한 근거
    strata_n = {}
    for s_name, s_fn in STRATA.items():
        cnt: dict[str, list[int]] = {}
        for r in recs:
            if r["group"] in (POS, NEG):
                c = cnt.setdefault(s_fn(r), [0, 0])
                c[0 if r["group"] == POS else 1] += 1
        usable = {k: v for k, v in cnt.items() if v[0] and v[1]}
        strata_n[s_name] = {
            "n_strata": len(cnt), "n_usable_strata": len(usable),
            "n_pos_usable": sum(v[0] for v in usable.values()),
            "n_neg_usable": sum(v[1] for v in usable.values()),
        }
    result["_meta"]["strata_coverage"] = strata_n
    print("\n=== 층별 사용 가능 표본 (양·음 둘 다 있는 층만 AUC 에 기여) ===")
    for s_name, v in strata_n.items():
        print(f"  {s_name:<20}층 {v['n_usable_strata']:>3}/{v['n_strata']:<3} "
              f"기여 표본 실패 {v['n_pos_usable']:>3} / 완주 {v['n_neg_usable']:>3}")

    # --- 실제 산출물 예시 --------------------------------------------------
    verdict = {}
    for k in CONCEPT_KEYS:
        if k in passed:
            verdict[k] = "결론용 층 전부 통과 (Bonferroni 포함)"
        elif k in weak:
            verdict[k] = "결론용 층 전부 CI 하한>0.5, Bonferroni 미달"
        else:
            fails = [s for s in DECISIVE_STRATA
                     if not ((result["concepts"][k]["strata"].get(s) or {}).get("ci95")
                             and result["concepts"][k]["strata"][s]["ci95"][0] > 0.5)]
            verdict[k] = f"판별력 미입증 (CI 하한≤0.5 인 층: {','.join(fails)})"
    ex = example_statements(recs, verdict)
    result["example_statements"] = ex
    print("\n=== 산출물 예시 (데모 케이스 모집실패 3건, 선정은 demo_cases.json 규칙 그대로) ===")
    for e in ex:
        if "error" in e:
            print(f"\n  [{e['nct_id']}] {e['error']}")
            continue
        c1, cn = e["completed"]
        p1, pn = e["recruit_failed"]
        print(f"\n  [{e['nct_id']}] {e['indication']} / {e['phase']}  범주={e['concept']}")
        print(f"    지목 기준({e['section']}): \"{e['criterion_quoted'][:150]}\"")
        print(f"    같은 적응증·Phase — 완주 {c1}/{cn}({c1/cn:.0%}) vs 모집실패 "
              f"{p1}/{pn}({p1/pn:.0%})  차이 {e['rate_diff']:+.0%}")
        for p in e["precedents"]:
            print(f"      전례 {p['nct']}  whyStopped: {p['why_stopped'][:110]!r}")
        print(f"    ※ 이 범주의 검증 판정: {e['verdict_of_concept']}")
        print(f"    (사후 확인용 — 실제 whyStopped: {str(e['actual_why_stopped'])[:100]!r}, "
              f"달성률 {e['actual_attainment']})")

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\n{OUT} 저장")


if __name__ == "__main__":
    main()
