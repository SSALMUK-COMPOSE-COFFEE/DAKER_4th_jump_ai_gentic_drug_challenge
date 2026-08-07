"""분야 4 전환 판단용 내부 검증 — v0 프로토콜의 "분자 바이오마커 환자 선별" 여부가
결과와 어떤 방향으로 연결되는지 측정한다.

## 왜 이걸 재는가

분야 3(프로토콜 감사)에 분야 1(가설·바이오마커) 요소를 얹는 융합 서사의 핵심 전제는
"교정안으로 바이오마커 기반 환자 선별을 제시하는 것이 타당하다" 이다.
그런데 우리 코퍼스에서 이미 확인된 사실은 **적격기준이 촘촘한 쪽이 모집에 실패한다**는 것이다
(포함기준 항목수 실패군 13.5 vs 완주군 9.0). 바이오마커 선별은 모집단을 더 좁힌다.

따라서 융합 서사를 제안서에 쓰기 전에 코퍼스 안에서 두 방향을 동시에 확인해야 한다:

- 방향 A (융합에 유리): 바이오마커 선별 시험은 **과학적(효능·독성) 실패**가 덜하다
- 방향 B (융합에 불리): 바이오마커 선별 시험은 **모집 실패**가 더 많다

두 방향이 동시에 성립하면 융합 서사는 "트레이드오프를 정량화하는 에이전트"로 정직하게
성립한다. 방향 A가 성립하지 않으면 융합 서사의 생물학적 근거가 코퍼스 안에 없다는 뜻이다.

## 입력
v0 만 사용한다 (brief_title / brief_summary / eligibility_criteria). 결과 정보 0.

## 한계
- 정규식 기반 탐지이므로 오분류가 있다. 탐지율을 적응증별로 함께 출력해 검산할 수 있게 했다.
- 이진 피처의 AUC 는 동점이 많아 0.5 쪽으로 압축된다. 비율·위험비를 함께 본다.
- 코퍼스의 적응증별 실패율은 수집 설계의 산물이므로(validated-numbers.md §4.1) 반드시 층화한다.
"""

from __future__ import annotations

import json
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import auc, group_of  # noqa: E402
from audit.benchmarks import extract, phase_stratum  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
N_BOOT = 300
BOOT_SEED = 20260730

# 마커 목록 설계 원칙 두 가지 (1차 시도에서 대량 오탐이 나와 고친 것):
#
# 1. **대문자 유전자 기호는 대소문자를 구분해 단어 경계로만 매칭한다.**
#    1차 시도에서 `MET` 411건, `FUS` 349건, `ANA` 115건이 탐지됐는데 전부 오탐이었다
#    ("...subjects who **met** the criteria", "in**fus**ion", "**ana**lysis").
# 2. **약물 표적은 마커가 아니다.** anti-IL13 항체 시험이 제목에 IL-13 을 쓰는 것은
#    환자 선별이 아니다. 그래서 사이토카인 표적명(IL-4/IL-13/TNF 등)을 목록에서 뺐고,
#    마커 탐색 범위를 `eligibility_criteria` 로 한정했다 — 환자 선별은 거기에만 쓰인다.

# (a) 대문자 유전자·항원 기호 — 대소문자 구분, 단어 경계 필수
SYMBOLS_CS = [
    "EGFR", "EGFRvIII", "ALK", "ROS1", "KRAS", "NRAS", "BRAF", "HER2", "ERBB2", "MET",
    "RET", "NTRK", "PIK3CA", "PTEN", "BRCA1", "BRCA2", "BRCA", "HRD", "MSI", "MMR", "TMB",
    "PD-L1", "PDL1", "PD-1", "IDH1", "IDH2", "FLT3", "NPM1", "CEBPA", "TP53", "TET2",
    "DNMT3A", "ASXL1", "RUNX1", "KMT2A", "MLL", "CD33", "CD123", "WT1", "MGMT", "ATRX",
    "H3K27M", "CFTR", "JAK2", "CALR", "SOD1", "C9orf72", "FUS", "TARDBP", "FGFR", "BCL2",
    "SMARCB1", "CEACAM5", "CA-125", "CA125", "APOE", "ACPA", "ANA", "HbS", "HbSS",
]
# (b) 대소문자 무관 표현 — 변이 표기·병리 마커·혈청학
PHRASES_CI = [
    r"f508del", r"delf508", r"g551d", r"r117h", r"exon\s*\d+",
    r"inv\(16\)", r"t\(8;\s*21\)", r"t\(15;\s*17\)", r"1p\s*/?\s*19q",
    r"amyloid", r"a\s?beta[\s-]?4?2?", r"p-?tau", r"csf biomarker",
    r"anti-?ccp", r"rheumatoid factor", r"anti-?dsdna",
    r"h(?:a)?emoglobin ss", r"sickle cell anemia genotype",
    r"cytogenetic (?:risk|abnormalit)", r"molecular (?:risk|subtype)",
]
MARKER_CS_RE = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(
    re.escape(s) for s in sorted(SYMBOLS_CS, key=len, reverse=True)) + r")(?![A-Za-z0-9])")
MARKER_CI_RE = re.compile("(" + "|".join(PHRASES_CI) + ")", re.I)

# 마커가 "선별 조건"으로 쓰였음을 나타내는 상태어
STATE_RE = re.compile(
    r"(mutat\w*|mutant|wild[- ]?type|amplif\w*|rearrang\w*|translocat\w*|fusion|"
    r"deletion|insertion|exon\s*\d+|positiv\w*|negativ\w*|overexpress\w*|express\w*|"
    r"status|genotype|carrier|documented|confirmed|detect\w*|elevat\w*|"
    r"\bhigh\b|\blow\b|≥|>=|\bat least\b)", re.I)

# 마커 없이도 분자 선별을 뜻하는 표현
GENERIC_RE = re.compile(
    r"(biomarker[- ]?(?:positive|selected|defined|based|driven|stratified)|"
    r"molecularly (?:selected|defined|characterized)|genomic\w* (?:selected|defined|profil\w*)|"
    r"companion diagnostic|next[- ]generation sequencing|NGS[- ]?(?:confirmed|based|selected)|"
    r"predictive biomarker|patient selection biomarker|enrichment (?:design|strategy)|"
    r"mutation[- ]positive|marker[- ]positive)", re.I)

WINDOW = 60  # 마커와 상태어가 같은 문맥에 있어야 한다고 볼 문자 거리


def enrichment_flag(v0: dict) -> tuple[int, list[str]]:
    """v0 에서 '분자 바이오마커 기반 환자 선별' 여부를 판정. (0/1, 근거 마커 목록)

    마커는 `eligibility_criteria` 에서만 찾는다(= 환자 선별 조건). 일반 표현은 전체 텍스트에서 찾는다.
    """
    elig = str(v0.get("eligibility_criteria") or "")
    whole = " \n ".join(
        str(v0.get(k) or "") for k in ("brief_title", "brief_summary", "eligibility_criteria")
    )
    hits: list[str] = []
    if GENERIC_RE.search(whole):
        hits.append("GENERIC")
    for rx in (MARKER_CS_RE, MARKER_CI_RE):
        for m in rx.finditer(elig):
            s, e = m.span()
            ctx = elig[max(0, s - WINDOW):min(len(elig), e + WINDOW)]
            if STATE_RE.search(ctx):
                hits.append(m.group(1).upper())
    hits = sorted(set(hits))
    return (1 if hits else 0), hits


def stratified_auc(data, key, stratum, pos_group, neg_group):
    num = den = 0.0
    buckets: dict[str, tuple[list, list]] = {}
    for r in data:
        v = r["f"].get(key)
        if v is None:
            continue
        if r["group"] == pos_group:
            buckets.setdefault(stratum(r), ([], []))[0].append(float(v))
        elif r["group"] == neg_group:
            buckets.setdefault(stratum(r), ([], []))[1].append(float(v))
    for p, n in buckets.values():
        if not p or not n:
            continue
        a = auc(p, n)
        num += a * len(p) * len(n)
        den += len(p) * len(n)
    return (round(num / den, 3) if den else None)


def boot_ci(data, key, stratum, pos_group, neg_group, rng):
    boots = []
    for _ in range(N_BOOT):
        sample = [rng.choice(data) for _ in data]
        a = stratified_auc(sample, key, stratum, pos_group, neg_group)
        if a is not None:
            boots.append(a)
    if len(boots) < N_BOOT // 2:
        return None
    boots.sort()
    return boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def main() -> None:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    recs = []
    marker_counter: Counter = Counter()
    for r in rows:
        flag, hits = enrichment_flag(r["v0"])
        marker_counter.update(hits)
        f = extract(r)
        recs.append({
            "nct": r["nct_id"],
            "ind": r["condition_bucket"],
            "phase": phase_stratum(r["v0"]),
            "group": group_of(r),
            "bm": flag,
            "hits": hits,
            "f": {"bm_enrichment": float(flag),
                  "n_inclusion_items": f["n_inclusion_items"],
                  "monthly_enrollment_burden": f["monthly_enrollment_burden"]},
        })

    print(f"대상 {len(recs)}건")
    print(f"바이오마커 선별 탐지: {sum(r['bm'] for r in recs)}건 "
          f"({sum(r['bm'] for r in recs)/len(recs):.1%})\n")

    print("[1] 적응증별 탐지율 (검산용 — 종양·CF·SCD 는 높고 IPF·심부전은 낮아야 정상)")
    by_ind = defaultdict(list)
    for r in recs:
        by_ind[r["ind"]].append(r["bm"])
    for ind, vs in sorted(by_ind.items(), key=lambda x: -st.fmean(x[1])):
        print(f"  {ind:<34} n={len(vs):<4} 탐지율 {st.fmean(vs):>6.1%}")

    print("\n[2] 결과 집단별 바이오마커 선별 비율 (미층화 — 참고용)")
    order = ["완주", "중단-모집실패", "중단-과학적", "중단-경영", "중단-경쟁환경",
             "중단-미분류", "철회(개시전)"]
    for g in order:
        vs = [r["bm"] for r in recs if r["group"] == g]
        if not vs:
            continue
        print(f"  {g:<14} n={len(vs):<4} 선별비율 {st.fmean(vs):>6.1%}")

    print("\n[3] 적응증 층화 — 층내 선별비율 (양성군 vs 완주군)")
    for pos in ("중단-모집실패", "중단-과학적"):
        print(f"\n  ── {pos} vs 완주")
        print(f"  {'적응증':<34}{'양성군 n/선별율':>20}{'완주군 n/선별율':>20}")
        tot_p = tot_pb = tot_n = tot_nb = 0
        for ind in sorted(by_ind):
            p = [r["bm"] for r in recs if r["group"] == pos and r["ind"] == ind]
            n = [r["bm"] for r in recs if r["group"] == "완주" and r["ind"] == ind]
            if not p or not n:
                continue
            tot_p += len(p); tot_pb += sum(p); tot_n += len(n); tot_nb += sum(n)
            print(f"  {ind:<34}{f'{len(p)} / {st.fmean(p):.0%}':>20}"
                  f"{f'{len(n)} / {st.fmean(n):.0%}':>20}")
        if tot_p and tot_n:
            print(f"  {'(층이 있는 적응증 합계)':<34}"
                  f"{f'{tot_p} / {tot_pb/tot_p:.0%}':>20}{f'{tot_n} / {tot_nb/tot_n:.0%}':>20}")

    print("\n[4] 적응증 층화 AUC (bm_enrichment, 1=선별 있음. >0.5 면 선별군이 그 실패에 더 많다)")
    rng = random.Random(BOOT_SEED)
    strata = {"미층화": lambda r: "_",
              "적응증": lambda r: r["ind"],
              "적응증+phase": lambda r: f"{r['ind']}|{r['phase']}"}
    for pos in ("중단-모집실패", "중단-과학적", "중단-경영"):
        line = f"  {pos:<14}"
        for sname, sfn in strata.items():
            a = stratified_auc(recs, "bm_enrichment", sfn, pos, "완주")
            ci = boot_ci(recs, "bm_enrichment", sfn, pos, "완주", rng) if a is not None else None
            cell = "—" if a is None else (f"{a} [{ci[0]:.2f},{ci[1]:.2f}]" if ci else str(a))
            line += f"{sname}={cell:<24}"
        print(line)

    print("\n[5] 선별 여부와 모집부담·포함기준 항목수 (완주군 안에서만 — 결과 교란 제거)")
    for key in ("monthly_enrollment_burden", "n_inclusion_items"):
        for g in ("완주", "중단-모집실패"):
            a = [r["f"][key] for r in recs if r["group"] == g and r["bm"] == 1
                 and r["f"][key] is not None]
            b = [r["f"][key] for r in recs if r["group"] == g and r["bm"] == 0
                 and r["f"][key] is not None]
            if not a or not b:
                continue
            print(f"  {key:<28} {g:<12} 선별有 n={len(a):<4} 중앙값={st.median(a):>7.2f} | "
                  f"선별無 n={len(b):<4} 중앙값={st.median(b):>7.2f}")

    print("\n[6] 가장 많이 탐지된 마커 25개 (정규식 검산용)")
    print("  ", ", ".join(f"{k}:{v}" for k, v in marker_counter.most_common(25)))


if __name__ == "__main__":
    main()
