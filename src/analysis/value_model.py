"""비즈니스 가치 정량화 — 모수 유병률과 실측 작동점을 곱해 PPV 를 낸다.

## 왜 이 스크립트가 필요했나

제안서 항목 5(비즈니스·사회적 가치 20점)는 "시간·비용 절감이 정량적으로 기대되는가"를 묻는다.
지금까지 있던 근거는 개별 사례(등록 달성률 43%)뿐이고 정량 추정이 없었다.

정량화에 필요한 것은 세 조각이다.

1. **모수 유병률 π** — Phase 2 시험 중 모집 실패로 중단되는 비율.
   `census.py` 가 ClinicalTrials.gov 전수로 구했다. 코퍼스 비율을 쓰면 안 된다
   (완주군을 적응증당 40건 균일 수집했으므로 유병률이 인위적이다).
2. **작동점(민감도·특이도)** — 월 모집부담 임계값에서 실측. `validated-numbers.md` §1 의 AUC 0.637
   을 만든 그 피처다.
3. **PPV** — π 와 작동점의 함수. π 가 낮으면 아무리 좋은 AUC 도 PPV 가 낮아진다.
   **이 스크립트의 존재 이유가 바로 그 계산을 숨기지 않는 것이다.**

## 하지 않는 것

"사전 경고로 회피 가능한 비율 Z" 는 우리가 실측한 것이 없다. 따라서 Z 를 가정해 단일 절감액을
내지 않는다. Z 를 자유 파라미터로 두고 민감도 표만 출력하고, 표에 "미검증 가정"을 붙인다.

## 특이도를 완주군만으로 재지 않는 이유

실제 세계의 음성군은 완주 시험만이 아니다. 다른 사유로 중단된 시험도 "모집 실패 경고를
받으면 오탐"이다. 그래서 음성군을 완주 + B(과학적) + C(경영) + D(경쟁) + 철회 로 구성하고,
각 집단의 오탐률을 **모수 구성비로 가중**한다. 완주군만 쓰면 특이도가 낙관 편향된다.

## 적응증 표준화

코퍼스의 적응증 구성비는 인위적이다(완주 균일 40건). 그래서 층별 민감도·오탐률을 구한 뒤
`census.json` 의 적응증별 Phase 2 실제 건수로 가중 평균한다. 표준화 전/후를 나란히 출력한다.

## 실행

    .venv/bin/python3 src/analysis/census.py      # 선행 (data/census.json 필요)
    .venv/bin/python3 src/analysis/value_model.py

출력: `data/value_model.json`
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import group_of  # noqa: E402
from audit.benchmarks import extract  # noqa: E402
from labels.taxonomy import classify  # noqa: E402

CORPUS = Path("data/raw/ctgov_phase2.jsonl")
CENSUS = Path("data/census.json")
CENSUS_RAW = Path("data/raw/ctgov_census.jsonl")
SECOND_PASS = Path("data/census_unmatched_sample_labels.json")
OUT = Path("data/value_model.json")

# 월 모집부담(명/월). 낮을수록 위험 → 임계값 이하면 경고.
FEATURE = "monthly_enrollment_burden"
THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0)

POS_GROUP = "중단-모집실패"
NEG_GROUPS = ("완주", "중단-과학적", "중단-경영", "중단-경쟁환경", "중단-미분류", "철회(개시전)")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, (c - h) / d), 4), round(min(1.0, (c + h) / d), 4))


# --- 1. 2차 패스 외삽: 미분류 구간의 분류 구성비 --------------------------


def second_pass_extrapolation() -> dict:
    """규칙 미매칭 건들의 분류 구성을 300건 라벨 표본에서 외삽한다.

    라벨은 **NCT ID 기준**으로 저장돼 있다. `taxonomy.py` 의 패턴이 갱신되면 미분류 풀이 줄어들어
    인덱스 기준 라벨이 무효가 되기 때문이다 (실제로 한 번 발생했다: 1,683 → 갱신 후 축소).

    단순무작위표본을 결정론적 부분집합(현재 미분류 풀)으로 제한하면 그 부분집합의
    단순무작위표본이 되므로 추정은 여전히 비편향이다. 표본 크기만 줄어든다.

    부수 효과로 **규칙 갱신분과 우리 라벨의 일치도**를 잴 수 있다 — 새 패턴이 잡게 된 건들에
    대해 규칙 라벨과 LLM 라벨을 대조한다.
    """
    meta = json.loads(SECOND_PASS.read_text())
    labels: dict[str, str] = meta["labels_by_nct"]
    rows = [json.loads(l) for l in CENSUS_RAW.open() if l.strip()]
    p2t = [r for r in rows if r["phase_filter"] == "PHASE2" and r["status"] == "TERMINATED"]
    pool = [r for r in p2t if classify(r.get("why_stopped")) == ("?", "unmatched", None)]
    pool_ncts = {r["nct_id"] for r in pool}

    # 라벨 표본 ∩ 현재 미분류 풀
    eff = {k: v for k, v in labels.items() if k in pool_ncts}
    counts: dict[str, int] = {c: 0 for c in ("A", "A?", "B", "C", "D", "?")}
    for v in eff.values():
        counts[v] += 1
    n = len(eff)
    n_unmatched = len(pool)

    # 규칙 갱신으로 매칭된 건들 — 규칙 라벨 vs LLM 라벨 일치도
    now_matched = {k: v for k, v in labels.items() if k not in pool_ncts}
    by_nct = {r["nct_id"]: r for r in p2t}
    agree = disagree = 0
    disagreements: list[dict] = []
    for nct, mine in now_matched.items():
        r = by_nct.get(nct)
        if not r:
            continue
        cat = classify(r.get("why_stopped"))[0]
        if cat == "?":
            continue
        mine_norm = "A" if mine == "A?" else mine
        if cat == mine_norm:
            agree += 1
        else:
            disagree += 1
            if len(disagreements) < 15:
                disagreements.append({"nct": nct, "rule": cat, "llm": mine,
                                      "why": (r.get("why_stopped") or "")[:120]})

    census = json.loads(CENSUS.read_text())
    blk = census["why_stopped_blocks"]["PHASE2"]["TERMINATED"]
    stated = blk["n_reason_stated"]
    rule = blk["by_category"]

    out: dict = {
        "taxonomy_sha256": hashlib.sha256(
            Path("src/labels/taxonomy.py").read_bytes()).hexdigest()[:16],
        "n_labeled_total": len(labels),
        "n_unmatched": n_unmatched,
        "n_sample_effective": n,
        "n_label_now_rule_matched": len(now_matched),
        "rule_vs_llm_agreement": {
            "n_compared": agree + disagree,
            "n_agree": agree,
            "agreement_rate": round(agree / (agree + disagree), 4) if agree + disagree else None,
            "examples_of_disagreement": disagreements,
        },
        "sample_counts": counts,
        "n_reason_stated": stated,
        "rule_only": {c: rule[c] for c in ("A", "B", "C", "D")},
        "extrapolated": {},
    }
    # 규칙 매칭분에서 넘어오는 기저 건수. rule["?"] 는 empty+unmatched 합이라 재사용하면
    # 이중계산이 된다 → "?" 와 "A?" 의 기저는 0 으로 두고 표본 외삽분만 센다.
    bases = {"A": rule["A"], "B": rule["B"], "C": rule["C"], "D": rule["D"], "A?": 0, "?": 0}
    for cat in ("A", "A?", "B", "C", "D", "?"):
        k = counts[cat]
        p = k / n
        lo, hi = wilson(k, n)
        base = bases[cat]
        out["extrapolated"][cat] = {
            "sample_rate": round(p, 4),
            "sample_ci95": [lo, hi],
            "n_added_point": round(n_unmatched * p, 1),
            "total_point": round(base + n_unmatched * p, 1),
            "share_of_stated_point": round((base + n_unmatched * p) / stated, 4),
            "share_of_stated_ci95": [
                round((base + n_unmatched * lo) / stated, 4),
                round((base + n_unmatched * hi) / stated, 4),
            ],
        }
    # A 의 상한 = A + A?
    a = out["extrapolated"]["A"]
    aq = out["extrapolated"]["A?"]
    out["A_share_of_stated_range"] = [
        a["share_of_stated_point"],
        round(a["share_of_stated_point"] + aq["n_added_point"] / stated, 4),
    ]
    return out


# --- 2. 모수 유병률 ---------------------------------------------------------


def prevalences(sp: dict) -> dict:
    census = json.loads(CENSUS.read_text())
    row = census["phase_status_counts"]["PHASE2"]
    det = row["COMPLETED"] + row["TERMINATED"] + row["WITHDRAWN"]
    blk = census["why_stopped_blocks"]["PHASE2"]["TERMINATED"]
    rule_a = blk["by_category"]["A"]
    a_pt = sp["extrapolated"]["A"]["total_point"]
    a_hi = a_pt + sp["extrapolated"]["A?"]["n_added_point"]
    return {
        "n_determined": det,
        "n_completed": row["COMPLETED"],
        "n_terminated": row["TERMINATED"],
        "n_withdrawn": row["WITHDRAWN"],
        # 하한: 규칙 매칭만. 사유 미기재 415건은 분자에서 제외 → 보수적.
        "pi_rule_only": round(rule_a / det, 5),
        # 주값: 2차 패스 외삽 점추정
        "pi_second_pass": round(a_pt / det, 5),
        # 상한: A? 를 전부 A로 인정
        "pi_second_pass_upper": round(a_hi / det, 5),
        "note": "분모 = 결과 확정(COMPLETED+TERMINATED+WITHDRAWN). 분자 = A분류 TERMINATED만 "
                "(코퍼스 라벨 정의와 일치). WITHDRAWN 의 A분류는 제외 → 보수적.",
    }


# --- 3. 작동점 (민감도·오탐률) ---------------------------------------------


def load_corpus() -> list[dict]:
    recs = []
    for l in CORPUS.open():
        if not l.strip():
            continue
        r = json.loads(l)
        v = extract(r).get(FEATURE)
        if v is None:
            continue
        recs.append({"nct": r["nct_id"], "ind": r["condition_bucket"],
                     "group": group_of(r), "x": float(v)})
    return recs


def operating_points(recs: list[dict], census_weights: dict) -> list[dict]:
    out = []
    for th in THRESHOLDS:
        row: dict = {"threshold": th, "rule": f"{FEATURE} <= {th} → 경고"}
        for g in (POS_GROUP,) + NEG_GROUPS:
            sub = [r for r in recs if r["group"] == g]
            k = sum(1 for r in sub if r["x"] <= th)
            row[g] = {"n": len(sub), "n_flag": k,
                      "rate": round(k / len(sub), 4) if sub else None,
                      "ci95": wilson(k, len(sub))}
        # 적응증 표준화 — 층별 비율을 모수 적응증 건수로 가중
        std: dict = {}
        for g in (POS_GROUP, "완주"):
            num = den = 0.0
            for ind, w in census_weights.items():
                sub = [r for r in recs if r["group"] == g and r["ind"] == ind]
                if len(sub) < 5:      # 층 표본이 너무 작으면 가중에서 제외
                    continue
                num += w * sum(1 for r in sub if r["x"] <= th) / len(sub)
                den += w
            std[g] = round(num / den, 4) if den else None
        row["indication_standardized"] = std
        out.append(row)
    return out


def composite_specificity(recs: list[dict], th: float, sp: dict, prev: dict) -> dict:
    """음성군을 모수 구성비로 가중해 오탐률을 합성한다.

    음성군 구성 (Phase 2, 결과 확정):
      완주            = COMPLETED 전수
      중단-과학적     = B분류 TERMINATED (2차 패스 외삽)
      중단-경영       = C분류 TERMINATED
      중단-경쟁환경   = D분류 TERMINATED
      중단-미분류     = ? (정보량 없음) TERMINATED
      철회(개시전)    = WITHDRAWN 전수
    """
    census = json.loads(CENSUS.read_text())
    row = census["phase_status_counts"]["PHASE2"]
    e = sp["extrapolated"]
    weights = {
        "완주": float(row["COMPLETED"]),
        "중단-과학적": e["B"]["total_point"],
        "중단-경영": e["C"]["total_point"],
        "중단-경쟁환경": e["D"]["total_point"],
        # 미분류 = 사유 미기재(empty) + 사유는 있으나 정보량 없음(2차 패스 "?")
        "중단-미분류": (census["why_stopped_blocks"]["PHASE2"]["TERMINATED"]["n_reason_empty"]
                       + e["?"]["total_point"]),
        "철회(개시전)": float(row["WITHDRAWN"]),
    }
    tot = sum(weights.values())
    fpr = num = 0.0
    detail = {}
    for g, w in weights.items():
        sub = [r for r in recs if r["group"] == g]
        if not sub:
            detail[g] = {"weight_share": round(w / tot, 4), "fpr": None, "n_corpus": 0}
            continue
        f = sum(1 for r in sub if r["x"] <= th) / len(sub)
        detail[g] = {"weight_share": round(w / tot, 4), "fpr": round(f, 4), "n_corpus": len(sub)}
        fpr += w * f
        num += w
    return {"composite_fpr": round(fpr / num, 4) if num else None,
            "negative_weights_total": round(tot, 1), "per_group": detail}


def ppv(pi: float, sens: float, fpr: float) -> float | None:
    d = pi * sens + (1 - pi) * fpr
    return round(pi * sens / d, 4) if d > 0 else None


def triage(pi: float, sens: float, fpr: float) -> dict:
    """경고군 / 비경고군의 실제 실패율과 그 비(lift). PPV 단독보다 이 비가 의사결정에 직결된다.

    PPV 는 유병률이 낮으면 자동으로 낮아진다. "경고를 받으면 실패 확률이 몇 배가 되는가" 는
    유병률에 덜 민감하고, 스크리닝 도구의 실제 효용에 더 가깝다.
    """
    flag = pi * sens + (1 - pi) * fpr                 # 경고 비율
    p_flag = pi * sens / flag if flag else None       # 경고군 실패율 = PPV
    unflag = pi * (1 - sens) + (1 - pi) * (1 - fpr)
    p_unflag = pi * (1 - sens) / unflag if unflag else None
    return {
        "flag_rate": round(flag, 4),
        "fail_rate_if_flagged": round(p_flag, 4) if p_flag is not None else None,
        "fail_rate_if_not_flagged": round(p_unflag, 4) if p_unflag is not None else None,
        "risk_ratio": round(p_flag / p_unflag, 2) if p_flag and p_unflag else None,
        "lift_over_base": round(p_flag / pi, 2) if p_flag else None,
        "n_screened_per_true_case": round(1 / (pi * sens), 1) if pi * sens else None,
    }


def main() -> None:
    print("[1] 규칙 미분류 구간 2차 패스 외삽")
    sp = second_pass_extrapolation()
    print(f"  taxonomy.py sha256[:16] = {sp['taxonomy_sha256']}")
    print(f"  미분류 {sp['n_unmatched']}건 중 유효 표본 {sp['n_sample_effective']}건 "
          f"(원 라벨 {sp['n_labeled_total']}건 ∩ 현재 미분류 풀)")
    ag = sp["rule_vs_llm_agreement"]
    if ag["n_compared"]:
        print(f"  규칙 갱신으로 매칭된 {sp['n_label_now_rule_matched']}건 중 {ag['n_compared']}건 대조 "
              f"→ 규칙·LLM 일치율 {ag['agreement_rate']:.1%}")
    print(f"  {'분류':<6}{'표본율':>9}{'95%CI':>18}{'전체 추정':>11}{'기재분 비율':>12}")
    for c in ("A", "A?", "B", "C", "D", "?"):
        e = sp["extrapolated"][c]
        ci = f"[{e['sample_ci95'][0]:.3f}, {e['sample_ci95'][1]:.3f}]"
        print(f"  {c:<6}{e['sample_rate']:>9.3f}{ci:>18}"
              f"{e['total_point']:>11.1f}{e['share_of_stated_point']:>12.2%}")
    lo, hi = sp["A_share_of_stated_range"]
    print(f"  → A분류 비율 (사유 기재분 기준): {lo:.1%} ~ {hi:.1%}")

    print("\n[2] 모수 유병률 π")
    prev = prevalences(sp)
    print(f"  결과 확정 {prev['n_determined']}건 "
          f"(완료 {prev['n_completed']} / 중단 {prev['n_terminated']} / 철회 {prev['n_withdrawn']})")
    print(f"  π (규칙만, 하한)     {prev['pi_rule_only']:.3%}")
    print(f"  π (2차 패스, 주값)   {prev['pi_second_pass']:.3%}")
    print(f"  π (A? 포함, 상한)    {prev['pi_second_pass_upper']:.3%}")

    print("\n[3] 작동점 — 월 모집부담 임계값별")
    recs = load_corpus()
    census = json.loads(CENSUS.read_text())
    cw = {b: float(v["all"]) for b, v in census["indication_counts"].items()}
    ops = operating_points(recs, cw)
    print(f"  {'임계':>5}{'민감도(A)':>11}{'완주 오탐':>11}{'표준화 민감도':>14}{'표준화 오탐':>12}")
    for r in ops:
        s = r["indication_standardized"]
        print(f"  {r['threshold']:>5}{r[POS_GROUP]['rate']:>11.1%}{r['완주']['rate']:>11.1%}"
              f"{(s[POS_GROUP] or 0):>14.1%}{(s['완주'] or 0):>12.1%}")

    print("\n[4] 합성 특이도 + PPV (모수 π 대입)")
    ppv_rows = []
    for r in ops:
        th = r["threshold"]
        comp = composite_specificity(recs, th, sp, prev)
        sens_raw = r[POS_GROUP]["rate"]
        sens_std = r["indication_standardized"][POS_GROUP]
        row = {
            "threshold": th,
            "sens_pooled": sens_raw,
            "sens_standardized": sens_std,
            "fpr_completed_only": r["완주"]["rate"],
            "fpr_composite": comp["composite_fpr"],
            "composite_detail": comp,
            "ppv": {
                "pi_rule_only": ppv(prev["pi_rule_only"], sens_std, comp["composite_fpr"]),
                "pi_second_pass": ppv(prev["pi_second_pass"], sens_std, comp["composite_fpr"]),
                "pi_upper": ppv(prev["pi_second_pass_upper"], sens_std, comp["composite_fpr"]),
                "corpus_prevalence_for_contrast": ppv(
                    len([r2 for r2 in recs if r2["group"] == POS_GROUP]) / len(recs),
                    sens_std, comp["composite_fpr"]),
            },
            "triage_at_pi_second_pass": triage(
                prev["pi_second_pass"], sens_std, comp["composite_fpr"]),
        }
        ppv_rows.append(row)
    print(f"  {'임계':>5}{'민감도*':>9}{'합성오탐':>10}{'PPV(π하한)':>12}{'PPV(π주값)':>12}"
          f"{'PPV(π상한)':>12}{'PPV(코퍼스π)':>14}")
    for r in ppv_rows:
        p = r["ppv"]
        print(f"  {r['threshold']:>5}{r['sens_standardized']:>9.1%}{r['fpr_composite']:>10.1%}"
              f"{p['pi_rule_only']:>12.1%}{p['pi_second_pass']:>12.1%}{p['pi_upper']:>12.1%}"
              f"{p['corpus_prevalence_for_contrast']:>14.1%}")
    print("  * 적응증 표준화 민감도. 합성오탐 = 완주+B+C+D+미분류+철회를 모수 구성비로 가중.")
    print("  코퍼스π 열은 편향 코퍼스 비율을 그대로 썼을 때의 값 — 얼마나 부풀려지는지 대조용.")

    print("\n[5] 트리아지 지표 (π = 2차 패스 주값)")
    print(f"  {'임계':>5}{'경고율':>9}{'경고군 실패율':>14}{'비경고군 실패율':>16}"
          f"{'위험비':>9}{'기저대비':>10}")
    for r in ppv_rows:
        t = r["triage_at_pi_second_pass"]
        print(f"  {r['threshold']:>5}{t['flag_rate']:>9.1%}{t['fail_rate_if_flagged']:>14.1%}"
              f"{t['fail_rate_if_not_flagged']:>16.1%}{t['risk_ratio']:>9.2f}"
              f"{t['lift_over_base']:>10.2f}")

    print("\n[5b] 교차 확인 — 홀드아웃 동작점(data/gate_operating_point.json)에 모수 π 대입")
    gate_path = Path("data/gate_operating_point.json")
    gate: dict = {}
    if gate_path.exists():
        g = json.loads(gate_path.read_text())
        for split in ("test", "train"):
            if split not in g:
                continue
            s, f = g[split]["sensitivity"], g[split]["false_positive_rate"]
            t = triage(prev["pi_second_pass"], s, f)
            gate[split] = {"sensitivity": s, "false_positive_rate": f,
                           "ppv_pi_second_pass": ppv(prev["pi_second_pass"], s, f),
                           "ppv_pi_rule_only": ppv(prev["pi_rule_only"], s, f),
                           "triage": t}
            print(f"  {split:<6} 민감도 {s:.1%} / 오탐 {f:.1%} → PPV {gate[split]['ppv_pi_second_pass']:.1%}"
                  f", 위험비 {t['risk_ratio']:.2f}")
        print("  ※ 이 동작점은 별도 워크스트림(src/analysis/gate_operating_point.py)이 만든 것이고"
              " 오탐률 분모가 완주군 단독이다. 합성 음성군보다 낙관적일 수 있다.")
    else:
        print("  (없음 — src/analysis/gate_operating_point.py 를 먼저 실행)")

    print("\n[6] 모수 규모 — 절감 계산의 분모")
    yr = 2020 - 2010 + 1
    n2 = json.loads(CENSUS.read_text())["phase_status_counts"]["PHASE2"]["all"]
    scale = {
        "n_phase2_registered_2010_2020": n2,
        "per_year_mean": round(n2 / yr, 1),
        "n_A_terminated_point": round(prev["pi_second_pass"] * prev["n_determined"], 1),
        "n_A_terminated_per_year": round(prev["pi_second_pass"] * prev["n_determined"] / yr, 1),
    }
    print(f"  Phase 2 등록 {n2}건 / {yr}년 = 연평균 {scale['per_year_mean']}건")
    print(f"  모집실패 중단 추정 {scale['n_A_terminated_point']}건 "
          f"= 연평균 {scale['n_A_terminated_per_year']}건")
    print("  → 시험 1건당 비용을 곱하면 연간 낭비 총액이 나온다 (문헌값은 별도 문서).")

    print("\n[7] 절감 민감도 — f(실제 지출 비율)·Z(회피 가능 비율)는 모두 미검증 가정")
    # 비용 X: Sertkaya et al., ASPE/ERG 2014 Table 1, Phase 2 전 치료영역 가중평균.
    # 물가조정 없음(원문 명시), 산업주도 시험 기준 → 학술 스폰서에는 과대적용.
    COST_PHASE2_USD = 13_350_000
    KR_APPROVALS_2025 = 783          # 식약처 2026-03-06 보도자료
    KR_PHASE2_SHARE = 0.205          # 2차 요약 출처, 식약처 1차 미확인
    op15 = next(r for r in ppv_rows if r["threshold"] == 1.5)
    pi, sens = prev["pi_second_pass"], op15["sens_standardized"]
    catch = pi * sens                # 스크리닝 1건당 실제로 잡는 모집실패 중단 건수
    saving: dict = {
        "assumptions": {
            "cost_phase2_usd": COST_PHASE2_USD,
            "cost_source": "Sertkaya et al., ASPE/ERG 2014 Table 1 (Phase 2 weighted mean). "
                           "물가조정 없음. 산업주도 기준.",
            "f_spent_fraction": "가정·미검증 (0.20/0.35/0.50)",
            "z_avoidable_fraction": "가정·미검증 (0.10/0.25/0.50)",
            "kr_phase2_share": f"{KR_PHASE2_SHARE} (2차 출처, 식약처 1차 미확인)",
        },
        "pi": pi, "sens": sens, "catch_per_screened": round(catch, 5),
        "identity_check": {
            "flag_rate_x_ppv": round(op15["triage_at_pi_second_pass"]["flag_rate"]
                                     * op15["ppv"]["pi_second_pass"], 5),
            "note": "pi*sens 와 같아야 한다 (반올림 오차 범위 내)",
        },
        "per_screened_protocol_usd": {}, "global_annual_usd": {}, "korea_annual_usd": {},
    }
    print(f"  {'f':>5}" + "".join(f"{'Z='+str(int(z*100))+'%':>12}" for z in (0.10, 0.25, 0.50)))
    for f in (0.20, 0.35, 0.50):
        cells = []
        for z in (0.10, 0.25, 0.50):
            v = catch * f * COST_PHASE2_USD * z
            saving["per_screened_protocol_usd"][f"f={f},Z={z}"] = round(v)
            saving["global_annual_usd"][f"f={f},Z={z}"] = round(v * scale["per_year_mean"])
            saving["korea_annual_usd"][f"f={f},Z={z}"] = round(
                v * KR_APPROVALS_2025 * KR_PHASE2_SHARE)
            cells.append(f"${v:>10,.0f}")
        print(f"  {f:>5}" + "".join(f"{c:>12}" for c in cells))
    # Z 없이 말할 수 있는 총량 — 낭비 규모
    saving["annual_waste_usd_no_Z"] = {
        f"f={f}": round(scale["n_A_terminated_per_year"] * COST_PHASE2_USD * f)
        for f in (0.20, 0.35, 0.50)
    }
    print("  Z 없이: 연간 낭비 총액 = 연 " + ", ".join(
        f"f={f}→${v/1e6:,.0f}M" for f, v in
        [(f, saving['annual_waste_usd_no_Z'][f'f={f}']) for f in (0.20, 0.35, 0.50)]))
    print(f"  Z·f 없이: 모집실패 중단 시험에 등록된 환자 = 연 "
          f"{scale['n_A_terminated_per_year'] * 10:.0f}명 "
          f"(A분류 실제 등록수 중앙값 10명, 코퍼스 실측)")

    out = {
        "_meta": {
            "generated_by": "src/analysis/value_model.py",
            "feature": FEATURE,
            "direction": "낮을수록 위험 (임계값 이하 = 경고)",
            "corpus": str(CORPUS),
            "census": str(CENSUS),
            "second_pass_labels": str(SECOND_PASS),
            "sample_seed": 20260730,
            "warning": "Z(사전 경고로 회피 가능한 비율)는 실측된 값이 없어 이 파일에 없다.",
        },
        "second_pass": sp,
        "prevalence": prev,
        "operating_points": ops,
        "ppv": ppv_rows,
        "holdout_gate_crosscheck": gate,
        "scale": scale,
        "saving_sensitivity": saving,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n{OUT} 저장")


if __name__ == "__main__":
    main()
