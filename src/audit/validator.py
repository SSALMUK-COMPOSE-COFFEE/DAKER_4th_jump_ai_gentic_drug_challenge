"""T5 근거 검증 — 판정부.

## 이 모듈이 하는 일

A4 레드팀이 `FEATURE_STATUS` 에 없는 근거로 공격하려 할 때, 그 근거가 **통계적 자격을
갖는지** 코퍼스로 판정한다. LLM 호출이 0회이고, 같은 입력에 항상 같은 판정이 나온다.

판정 등급은 `pipeline.FEATURE_STATUS` 의 어휘를 그대로 쓴다 —
`validated` / `nonspecific` / `unvalidated` / `reversed` / `retracted`.
**`validated` 만 위험 판정과 조언을 생성할 수 있다** (`pipeline.RISK_ELIGIBLE`).

## 왜 이 판정을 자동화하는가

`FEATURE_STATUS` 는 사람이 채운 표였다. 표를 만든 판단 — 층화하고, 음성 대조군과 비교하고,
미달이면 버리는 — 은 개발 과정에만 존재했고 시스템 안에는 없었다. 그런데 그 판단을 수행한
검사는 전부 `analysis/stratified_signal.py` 에 함수로 있다. 사람이 한 일은 그 함수를 호출한
것뿐이다. 이 모듈이 그 호출을 에이전트에게 넘긴다.

## 판정 규칙 (docs/memory/agent-architecture.md §12.2)

1. **표적 신호** — 결정적 층 전부에서 부트스트랩 CI 하한 > 0.5
2. **방향** — 층화 AUC 가 0.5 미만이고 CI 상한도 0.5 미만이면 `reversed`
3. **음성 대조군** — 적응증 라벨만 쓰는 대조 예측기를 넘는가
   ⚠ **이진 피처에는 이 검사를 결격 사유로 쓰지 않는다.** 이진 범주는 AUC 상한이
   `0.5 + 출현률차/2` 라 구조적으로 대조군(0.722)을 넘을 수 없다. 넘지 못한 사실은
   기록하되 판정 근거로는 층화 검사만 쓴다. 이 비대칭을 숨기지 않고 출력에 남긴다.
4. **특이도** — 효능·독성 실패군과 경영 판단 중단군에서 0.5 근처여야 한다.
   표적에서만 뜨지 않고 세 집단 모두에서 뜨면 `nonspecific` 이다.

## 실행

    python3 src/audit/validator.py          # data/criteria_signal.json → data/t5_first_run.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IN = Path("data/criteria_signal.json")
OUT = Path("data/t5_first_run.json")

# 특이도 집단에서 이 범위를 벗어나면 "표적 특이적이지 않다"고 본다.
SPECIFICITY_BAND = (0.45, 0.55)


@dataclass
class ValidationVerdict:
    """T5 의 판정 결과. agent-architecture.md §12.3 의 인터페이스."""

    feature: str
    name: str
    status: str
    target_auc: float | None
    ci95: list[float] | None
    control_auc: float | None
    beats_control: bool
    binary_ceiling: float | None      # 이진 피처의 구조적 AUC 상한
    strata_passed: list[str]
    strata_total: int
    specificity: dict
    prevalence: dict
    n_pos: int
    n_neg: int
    reason: str
    corpus_sha: str
    labeler_sha: str


def binary_auc_ceiling(prev: dict) -> float | None:
    """이진 범주의 구조적 AUC 상한 = 0.5 + 출현률차/2.

    52% vs 27% 처럼 출현률 차이가 커도 AUC 는 0.62 를 넘지 못한다. 연속형 피처와
    같은 잣대로 재면 이진 피처는 구조적으로 불리하다 — 이 값을 함께 보고해야
    "AUC 가 낮다"와 "신호가 없다"를 혼동하지 않는다.
    """
    try:
        c_hit, c_n = prev["completed"]
        f_hit, f_n = prev["recruit_failed"]
    except (KeyError, ValueError, TypeError):
        return None
    if not c_n or not f_n:
        return None
    return round(0.5 + abs(f_hit / f_n - c_hit / c_n) / 2, 4)


def classify(key: str, c: dict, meta: dict, spec: dict) -> ValidationVerdict:
    strata = c["strata"]
    decisive = meta["decisive_strata"]
    control = meta.get("indication_only_control")

    ind = strata.get("indication") or {}
    auc = ind.get("auc")
    ci = ind.get("ci95")

    passed = [ax for ax in decisive
              if (s := strata.get(ax)) and s.get("ci95") and s["ci95"][0] > 0.5]

    ceiling = binary_auc_ceiling(c.get("prevalence", {}))
    beats = bool(auc is not None and control is not None and auc > control)

    sc = (spec.get("중단-과학적") or {}).get("auc")
    bz = (spec.get("중단-경영") or {}).get("auc")
    off_target = [v for v in (sc, bz)
                  if v is not None and not (SPECIFICITY_BAND[0] <= v <= SPECIFICITY_BAND[1])]

    # --- 판정 ---
    if auc is None or ci is None:
        status, reason = "unvalidated", "적응증 층화 AUC 를 계산할 수 없다 (층 표본 부족)."
    elif auc < 0.5 and ci[1] < 0.5:
        status = "reversed"
        reason = (f"적응증 층화 AUC {auc} [CI 상한 {ci[1]}] — 방향이 반대다. "
                  f"이 근거로 공격하면 데이터와 거꾸로 된 조언이 된다.")
    elif len(passed) < len(decisive):
        failed = [ax for ax in decisive if ax not in passed]
        status = "unvalidated"
        reason = (f"결정적 {len(decisive)}축 중 {len(passed)}축만 통과. "
                  f"탈락: {', '.join(failed)}. 구분력 미입증이므로 위험 판정에 쓰지 않는다.")
        if len(passed) == len(decisive) - 1 and "indication+sponsor" in failed:
            reason += (" ※ 유일한 탈락 축이 `적응증+스폰서` 다 — 이 근거가 "
                       "스폰서 유형의 대리변수일 가능성을 시사한다.")
    elif off_target:
        status = "nonspecific"
        reason = (f"표적 {auc} 이지만 특이도 집단에서도 발화한다 "
                  f"(과학적 {sc} / 경영 {bz}). 모집 실패에 특화되지 않았다.")
    else:
        status = "validated"
        reason = (f"결정적 {len(decisive)}축 전부 통과, 특이도 정상 "
                  f"(과학적 {sc} / 경영 {bz}). 위험 판정 자격을 갖는다.")

    if ceiling is not None and not beats and status != "validated":
        reason += (f" ※ 음성 대조군(0.722) 미달이나, 이진 범주의 구조적 AUC 상한이 "
                   f"{ceiling} 이므로 이를 결격 사유로 쓰지 않았다.")

    return ValidationVerdict(
        feature=key, name=c.get("name", key), status=status,
        target_auc=auc, ci95=ci, control_auc=control, beats_control=beats,
        binary_ceiling=ceiling, strata_passed=passed, strata_total=len(decisive),
        specificity={"중단-과학적": sc, "중단-경영": bz},
        prevalence=c.get("prevalence", {}),
        n_pos=ind.get("n_pos", meta.get("n_pos", 0)),
        n_neg=ind.get("n_neg", meta.get("n_neg", 0)),
        reason=reason,
        corpus_sha=meta.get("corpus_sha256_16", ""),
        labeler_sha=meta.get("taxonomy_sha256_16", ""),
    )


def main() -> None:
    d = json.loads(IN.read_text())
    meta, concepts = d["_meta"], d["concepts"]
    spec_all = d.get("specificity_indication_stratified", {})

    # `concepts` 에는 의미 범주 20종 외에 개수 피처(`n_items`, `n_concepts`)가 섞여 있다.
    # 정본은 `_meta.n_decisive_strata_passed` 의 키 집합이다 (= 20종).
    canon = meta["n_decisive_strata_passed"]
    extra = [k for k in concepts if k not in canon]
    concepts = {k: v for k, v in concepts.items() if k in canon}
    assert len(concepts) == meta["n_concepts_tested"], (
        f"의미 범주 수 불일치: {len(concepts)} vs _meta {meta['n_concepts_tested']}")

    verdicts = [classify(k, c, meta, spec_all.get(k, {})) for k, c in concepts.items()]

    # 자체 검증 — 재계산한 통과 축 수가 저장값과 일치해야 한다.
    # 어긋나면 판정 규칙이 원 분석과 다르다는 뜻이므로 즉시 멈춘다.
    mismatch = {v.feature: (len(v.strata_passed), canon[v.feature])
                for v in verdicts if len(v.strata_passed) != canon[v.feature]}
    if mismatch:
        raise SystemExit(f"통과 축 수가 criteria_signal.json 과 불일치: {mismatch}")

    verdicts.sort(key=lambda v: (-len(v.strata_passed), -(v.target_auc or 0)))

    n = len(verdicts)
    rejected = [v for v in verdicts if v.status != "validated"]
    by_status: dict[str, int] = {}
    for v in verdicts:
        by_status[v.status] = by_status.get(v.status, 0) + 1

    print(f"\n{'='*94}")
    print(f"T5 근거 검증 — 첫 실행 결과   (코퍼스 {meta['corpus_sha256_16']} / "
          f"라벨러 {meta['taxonomy_sha256_16']})")
    print(f"표적 모집실패 {meta['n_pos']}건 vs 완주 {meta['n_neg']}건 · "
          f"결정적 층 {len(meta['decisive_strata'])}축 · 부트스트랩 {meta['n_boot']}회")
    print(f"의미 범주 {len(verdicts)}종 (개수 피처 {len(extra)}종 제외: {', '.join(extra)}) · "
          f"통과 축 수 재계산이 원 분석과 일치함을 확인")
    print(f"{'='*94}")
    print(f"  {'근거 (적격기준 의미 범주)':<34}{'층화AUC':>9}{'통과':>7}"
          f"{'출현률 완주→실패':>19}{'판정':>13}")
    print("  " + "-"*90)
    for v in verdicts:
        pv = v.prevalence
        try:
            cr = pv["completed"][0] / pv["completed"][1] * 100
            fr = pv["recruit_failed"][0] / pv["recruit_failed"][1] * 100
            rate = f"{cr:>5.1f}% → {fr:>5.1f}%"
        except Exception:
            rate = "—"
        auc = f"{v.target_auc:.3f}" if v.target_auc is not None else "—"
        print(f"  {v.name[:32]:<34}{auc:>9}{len(v.strata_passed):>4}/{v.strata_total}"
              f"{rate:>19}{v.status:>13}")

    print("  " + "-"*90)
    print(f"\n  검사한 근거 {n}종 · 기각 {len(rejected)}종 · "
          f"**기각률 {len(rejected)/n*100:.0f}%**")
    print(f"  판정 분포: " + " / ".join(f"{k} {v}" for k, v in sorted(by_status.items())))
    near = [v for v in verdicts if len(v.strata_passed) == v.strata_total - 1]
    if near:
        print(f"\n  ※ 1축 차이로 탈락한 근거 {len(near)}종 — "
              f"{', '.join(v.name.split('(')[0].strip() for v in near)}")
        print("     전부 `적응증+스폰서` 축에서 무너진다 → 스폰서 유형의 대리변수로 판단.")
        print("     판별기로 쓰지 않고 **전례 제시 근거로만** 사용한다.")

    OUT.write_text(json.dumps({
        "_meta": {
            "source": str(IN),
            "corpus_sha256_16": meta["corpus_sha256_16"],
            "taxonomy_sha256_16": meta["taxonomy_sha256_16"],
            "n_tested": n,
            "n_rejected": len(rejected),
            "rejection_rate": round(len(rejected) / n, 4),
            "decisive_strata": meta["decisive_strata"],
            "indication_only_control": meta.get("indication_only_control"),
            "note": "T5 판정. 정답 라벨 불필요 — 코퍼스만으로 측정된다.",
        },
        "by_status": by_status,
        "verdicts": [asdict(v) for v in verdicts],
    }, ensure_ascii=False, indent=1))
    print(f"\n  {OUT} 저장\n")


if __name__ == "__main__":
    main()
