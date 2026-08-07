"""분야 4(융합) 전환 판단용 실측 — 개입 약물에서 타겟을 역추적하고,
그 타겟-적응증 쌍의 Open Targets 근거 점수가 우리 코퍼스의 실패 유형을 구분하는지 본다.

## 검증할 두 가지

1. **역추적 실현성** — `v0.interventions[].name` (SAR408701 같은 사내 코드명 포함)에서
   ChEMBL ID → 작용기전 → 타겟까지 몇 %가 풀리는가. 이게 낮으면 융합 서사는 성립하지 않는다.
2. **신호 존재 여부** — 타겟-적응증 근거 점수가 "과학적(효능·독성) 중단"을 완주와 구분하는가.
   구분하면 분야 1 요소가 우리 라벨로 검증 가능하다. 구분하지 못하면 검증 불가다.

## 누출(leakage) 경고 — 이 스크립트가 재는 것의 한계

Open Targets 의 overall association score 에는 `clinical` datatype 이 들어간다. 이것은
**ChEMBL 의 임상시험 기록에서 나온다** — 즉 우리가 심사하려는 바로 그 시험이 점수에 기여한다.
게다가 점수는 2026-06 릴리스의 값이고 등록 시점(2010~2020)의 값이 아니다.
따라서 overall score 로 좋은 결과가 나와도 **예측력의 증거가 아니다**.
그래서 datatype 별 점수를 따로 뽑아 `clinical`/`literature` 를 제외한 값도 같이 평가한다.

출력: data/target_evidence.json (캐시 겸 결과)
"""

from __future__ import annotations

import html
import json
import random
import re
import statistics as st
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import auc, group_of  # noqa: E402
from audit.benchmarks import phase_stratum  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")
OUT = Path("data/target_evidence.json")
OT = "https://api.platform.opentargets.org/api/v4/graphql"

N_BOOT = 300
BOOT_SEED = 20260730

# condition_bucket → Open Targets 질병 ID. search(entityNames:["disease"]) 최상위 히트를 눈으로 확인해 고정.
DISEASE_ID = {
    "non-small cell lung cancer": "MONDO_0005233",
    "atopic dermatitis": "MONDO_0004980",          # OT 표기는 'atopic eczema'
    "type 2 diabetes": "MONDO_0005148",
    "rheumatoid arthritis": "MONDO_0008383",
    "alzheimer disease": "MONDO_0004975",
    "pancreatic cancer": "MONDO_0005192",          # OT 표기는 'exocrine pancreatic carcinoma'
    "glioblastoma": "MONDO_0018177",
    "amyotrophic lateral sclerosis": "MONDO_0004976",
    "idiopathic pulmonary fibrosis": "EFO_0000768",
    "systemic lupus erythematosus": "MONDO_0007915",
    "sickle cell disease": "MONDO_0011382",
    "cystic fibrosis": "MONDO_0009061",
    "acute myeloid leukemia": "MONDO_0018874",
    "ovarian cancer": "MONDO_0008170",
    "heart failure": "MONDO_0005252",
}

# 시험약이 아니라 병용 백본인 약. "가설의 타겟"으로 볼 수 없으므로 별도 집계한다.
BACKBONE = {
    "carboplatin", "cisplatin", "paclitaxel", "nab-paclitaxel", "docetaxel", "gemcitabine",
    "pemetrexed", "etoposide", "cyclophosphamide", "doxorubicin", "fluorouracil", "5-fu",
    "leucovorin", "oxaliplatin", "irinotecan", "cytarabine", "daunorubicin", "idarubicin",
    "azacitidine", "decitabine", "temozolomide", "methotrexate", "vinorelbine", "vincristine",
    "fludarabine", "melphalan", "busulfan", "hydroxyurea", "dexamethasone", "prednisone",
    "prednisolone", "metformin", "insulin", "insulin glargine", "bevacizumab", "rituximab",
    "capecitabine", "topotecan", "mitomycin", "bortezomib", "lenalidomide", "aspirin",
    "warfarin", "furosemide", "tacrolimus", "folinic acid", "thalidomide",
}

SKIP = {
    "placebo", "vehicle", "saline", "normal saline", "sham", "no intervention", "control",
    "standard of care", "soc", "best supportive care", "matching placebo", "water",
    "placebo oral tablet", "placebo comparator", "chemotherapy", "standard chemotherapy",
    "radiotherapy", "radiation", "surgery", "questionnaire", "moisturizer", "emollient",
    "physician's choice", "investigator's choice", "supportive care", "dextrose",
}
DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|u|%|w/w|w/v|mg/kg|mg/m2|units?)\b", re.I)
TAIL = re.compile(
    r"\b(tablet|tablets|capsule|capsules|injection|injectable|infusion|solution|suspension|cream|"
    r"ointment|gel|spray|topical|oral|iv|intravenous|subcutaneous|inhaled|arm|group|cohort|dose|"
    r"doses|placebo|combination|regimen|therapy|treatment)\b", re.I)
SPLIT = re.compile(r"\s*(?:/|\+|,|;|\band\b|\bplus\b|\bwith\b|\bin combination with\b|"
                   r"\bfollowed by\b|\bor\b)\s*", re.I)


def norm(s: str | None) -> str:
    s = html.unescape(s or "").replace("‐", "-").replace("‑", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def squash(s: str) -> str:
    return re.sub(r"[\s\-]", "", s)


def candidates(raw: str) -> list[str]:
    base = norm(raw)
    if not base or base in SKIP:
        return []
    out: list[str] = []
    parens = re.findall(r"\(([^)]*)\)", base)
    for piece in [re.sub(r"\([^)]*\)", " ", base)] + parens:
        piece = DOSE.sub(" ", piece)
        for part in SPLIT.split(piece):
            part = TAIL.sub(" ", part or "")
            part = re.sub(r"[^a-z0-9\-\s]", " ", part)
            part = re.sub(r"\s+", " ", part).strip(" -")
            if len(part) < 4 or part in SKIP or part.isdigit():
                continue          # 3글자 이하는 오탐이 많아 버린다 ('ipi' → IPILIMUMAB 같은 우연 매칭)
            out.append(part)
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res[:6]


def gql(query: str, tries: int = 4) -> dict:
    body = json.dumps({"query": query}).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(OT, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except Exception as e:                      # noqa: BLE001
            if i == tries - 1:
                return {"errors": [str(e)]}
            time.sleep(1.5 * (i + 1))
    return {}


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batch_search(names: list[str]) -> dict:
    parts = ['a%d: search(queryString:"%s", entityNames:["drug"], page:{index:0,size:5}){hits{id}}'
             % (i, esc(n)) for i, n in enumerate(names)]
    data = (gql("{" + " ".join(parts) + "}") or {}).get("data") or {}
    return {n: [h["id"] for h in ((data.get("a%d" % i) or {}).get("hits") or [])]
            for i, n in enumerate(names)}


def batch_drug(ids: list[str]) -> dict:
    parts = ['d%d: drug(chemblId:"%s"){id name drugType synonyms{label} tradeNames{label} '
             'mechanismsOfAction{rows{mechanismOfAction targets{id approvedSymbol}}}}'
             % (i, esc(c)) for i, c in enumerate(ids)]
    data = (gql("{" + " ".join(parts) + "}") or {}).get("data") or {}
    return {c: data.get("d%d" % i) for i, c in enumerate(ids)}


def batch_assoc(disease_id: str, ensembl_ids: list[str]) -> dict:
    q = ('{disease(efoId:"%s"){associatedTargets(Bs:%s, page:{index:0,size:%d})'
         '{rows{target{id} score datatypeScores{id score}}}}}'
         % (disease_id, json.dumps(ensembl_ids), max(len(ensembl_ids), 1)))
    d = gql(q)
    rows = (((d.get("data") or {}).get("disease") or {}).get("associatedTargets") or {}).get("rows") or []
    return {r["target"]["id"]: {"overall": r["score"],
                                "dt": {x["id"]: x["score"] for x in r["datatypeScores"]}}
            for r in rows}


# ─────────────────────────── 1. 약물명 → 타겟 ───────────────────────────

def resolve_targets(rows: list[dict]) -> tuple[dict, dict]:
    cand_of: dict[str, list[str]] = {}
    for r in rows:
        for iv in r["v0"].get("interventions") or []:
            if iv.get("type") in ("DRUG", "BIOLOGICAL"):
                nm = iv.get("name") or ""
                cand_of.setdefault(nm, candidates(nm))
    all_cands = sorted({c for v in cand_of.values() for c in v})
    print(f"  개입명 unique {len(cand_of)}건 → 후보 문자열 {len(all_cands)}개", file=sys.stderr)

    search_res: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(batch_search, list(chunks(all_cands, 12))):
            search_res.update(r)
    ids = sorted({cid for v in search_res.values() for cid in v})
    print(f"  ChEMBL 후보 {len(ids)}건 조회", file=sys.stderr)

    info: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(batch_drug, list(chunks(ids, 10))):
            info.update(r)

    def aliases(d: dict) -> set[str]:
        s = {norm(d.get("name"))}
        for k in ("synonyms", "tradeNames"):
            for v in d.get(k) or []:
                lab = norm(v.get("label"))
                s |= {lab, squash(lab)}
        return {x for x in s if x}

    resolved: dict[str, dict] = {}       # 후보 문자열 → 약물+타겟
    for c, hits in search_res.items():
        for cid in hits:
            d = info.get(cid)
            if not d:
                continue
            al = aliases(d)
            if c in al or squash(c) in al:
                mo = (d.get("mechanismsOfAction") or {}).get("rows") or []
                resolved[c] = {
                    "chembl": cid, "name": d["name"], "drugType": d.get("drugType"),
                    "targets": sorted({t["id"] for row in mo for t in (row.get("targets") or [])}),
                    "symbols": sorted({t["approvedSymbol"] for row in mo
                                       for t in (row.get("targets") or [])}),
                }
                break
    return cand_of, resolved


# ─────────────────────────── 2. 층화 AUC ───────────────────────────

def stratified_auc(data, key, stratum, pos, neg):
    num = den = 0.0
    buckets: dict[str, tuple[list, list]] = {}
    for r in data:
        v = r["f"].get(key)
        if v is None:
            continue
        if r["group"] == pos:
            buckets.setdefault(stratum(r), ([], []))[0].append(float(v))
        elif r["group"] == neg:
            buckets.setdefault(stratum(r), ([], []))[1].append(float(v))
    for p, n in buckets.values():
        if not p or not n:
            continue
        num += auc(p, n) * len(p) * len(n)
        den += len(p) * len(n)
    return round(num / den, 3) if den else None


def boot_ci(data, key, stratum, pos, neg, rng):
    boots = []
    for _ in range(N_BOOT):
        s = [rng.choice(data) for _ in data]
        a = stratified_auc(s, key, stratum, pos, neg)
        if a is not None:
            boots.append(a)
    if len(boots) < N_BOOT // 2:
        return None
    boots.sort()
    return boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def main() -> None:
    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    print(f"코퍼스 {len(rows)}건 — 약물명 역추적 시작", file=sys.stderr)
    cand_of, resolved = resolve_targets(rows)

    # 시험별 타겟 목록 (백본 제외분을 따로 보관)
    per_trial: list[dict] = []
    need: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        got, got_novel = [], []
        for iv in r["v0"].get("interventions") or []:
            if iv.get("type") not in ("DRUG", "BIOLOGICAL"):
                continue
            for c in cand_of.get(iv.get("name") or "", []):
                m = resolved.get(c)
                if not m:
                    continue
                got.append({"cand": c, **m})
                if c not in BACKBONE and m["targets"]:
                    got_novel.append({"cand": c, **m})
                break
        did = DISEASE_ID.get(r["condition_bucket"])
        # 음성 대조군을 만들기 위해 **모든** 적응증에 대해 같은 타겟의 점수를 받아온다.
        for m in got_novel:
            for other in DISEASE_ID.values():
                need[other].update(m["targets"])
        per_trial.append({"nct": r["nct_id"], "ind": r["condition_bucket"], "disease_id": did,
                          "phase": phase_stratum(r["v0"]),
                          "sponsor": r["v0"].get("sponsor_class") or "UNKNOWN",
                          "group": group_of(r),
                          "matched": got, "novel": got_novel})

    stats = {
        "n_trials": len(rows),
        "with_drug_intervention": sum(1 for t in per_trial
                                      if any(cand_of.get(x, []) for x in
                                             [iv.get("name") for iv in
                                              (next(r for r in rows if r["nct_id"] == t["nct"])
                                               ["v0"].get("interventions") or [])
                                              if iv.get("type") in ("DRUG", "BIOLOGICAL")])),
        "drug_resolved": sum(1 for t in per_trial if t["matched"]),
        "target_resolved": sum(1 for t in per_trial if any(m["targets"] for m in t["matched"])),
        "target_resolved_nonbackbone": sum(1 for t in per_trial if t["novel"]),
    }
    print(json.dumps(stats, ensure_ascii=False), file=sys.stderr)

    # 타겟-질병 근거 점수 수집
    assoc: dict[tuple[str, str], dict] = {}
    for did, tset in need.items():
        if not did or not tset:
            continue
        tl = sorted(tset)
        for part in chunks(tl, 120):
            got = batch_assoc(did, part)
            for tid in part:
                assoc[(did, tid)] = got.get(tid) or {"overall": 0.0, "dt": {}}
        print(f"  {did}: 타겟 {len(tl)}건 근거 조회 완료", file=sys.stderr)

    NONCLIN = ("genetic_association", "genetic_literature", "somatic_mutation",
               "affected_pathway", "animal_model", "known_drug", "rna_expression")
    recs = []
    for t in per_trial:
        did = t["disease_id"]
        ov, gen, nonclin, ctrl = [], [], [], []
        for m in t["novel"]:
            for tid in m["targets"]:
                a = assoc.get((did, tid))
                if not a:
                    continue
                ov.append(a["overall"])
                gen.append(a["dt"].get("genetic_association", 0.0))
                nonclin.append(max([a["dt"].get(k, 0.0) for k in NONCLIN] or [0.0]))
                # 음성 대조군: **다른 14개 적응증**에 대한 같은 타겟의 점수 중앙값.
                # 이 대조군이 표적 피처만큼 구분하면, 재고 있는 것은 "타겟-적응증 적합도"가
                # 아니라 "그 타겟이 유명한가"다. (경쟁 밀도 오판을 반복하지 않기 위한 장치)
                others = [assoc[(o, tid)] for o in DISEASE_ID.values()
                          if o != did and (o, tid) in assoc]
                if others:
                    ctrl.append((
                        st.median([a["overall"] for a in others]),
                        st.median([max([a["dt"].get(k, 0.0) for k in NONCLIN] or [0.0])
                                   for a in others]),
                    ))
        recs.append({
            "nct": t["nct"], "ind": t["ind"], "phase": t["phase"], "group": t["group"],
            "sponsor": t["sponsor"],
            "symbols": sorted({s for m in t["novel"] for s in m["symbols"]}),
            "f": {
                "ot_overall_max": max(ov) if ov else None,
                "ot_genetic_max": max(gen) if gen else None,
                "ot_nonclinical_max": max(nonclin) if nonclin else None,
                "CONTROL_fame_overall": max(c[0] for c in ctrl) if ctrl else None,
                "CONTROL_fame_nonclinical": max(c[1] for c in ctrl) if ctrl else None,
            },
        })

    have = [r for r in recs if r["f"]["ot_overall_max"] is not None]
    print(f"\n근거 점수까지 확보한 시험 {len(have)} / {len(recs)}건 "
          f"({len(have)/len(recs):.1%})")

    print("\n[1] 적응증별 역추적 성공률 (백본약 제외, 타겟까지)")
    per_ind = defaultdict(lambda: [0, 0])
    for r, t in zip(recs, per_trial):
        per_ind[r["ind"]][0] += 1
        per_ind[r["ind"]][1] += 1 if t["novel"] else 0
    for ind, (n, k) in sorted(per_ind.items(), key=lambda x: -x[1][1] / x[1][0]):
        print(f"  {ind:<34} {k:>4}/{n:<4} {k/n:>6.0%}")

    print("\n[2] 집단별 Open Targets 근거 점수 중앙값 (근거 확보분만)")
    order = ["완주", "중단-모집실패", "중단-과학적", "중단-경영", "중단-미분류", "철회(개시전)"]
    print(f"  {'집단':<14}{'n':>5}{'overall':>10}{'genetic':>10}{'nonclin':>10}")
    for g in order:
        sub = [r for r in have if r["group"] == g]
        if not sub:
            continue
        def med(k):
            vs = [r["f"][k] for r in sub if r["f"][k] is not None]
            return st.median(vs) if vs else float("nan")
        print(f"  {g:<14}{len(sub):>5}{med('ot_overall_max'):>10.3f}"
              f"{med('ot_genetic_max'):>10.3f}{med('ot_nonclinical_max'):>10.3f}")

    print("\n[3] 층화 AUC — 양성군 vs 완주. **낮을수록** 근거가 강한 쪽이 안전하다는 뜻")
    print("    (0.5 포함 = 신호 없음. 문헌 주장이 맞다면 '중단-과학적'에서 0.5 미만이어야 한다)")
    rng = random.Random(BOOT_SEED)
    strata = {"미층화": lambda r: "_", "적응증": lambda r: r["ind"],
              "적응증+phase": lambda r: f"{r['ind']}|{r['phase']}",
              "적응증+스폰서": lambda r: f"{r['ind']}|{r['sponsor']}"}
    for key in ("ot_overall_max", "ot_genetic_max", "ot_nonclinical_max",
                "CONTROL_fame_overall", "CONTROL_fame_nonclinical"):
        print(f"\n  ── {key}")
        for pos in ("중단-과학적", "중단-모집실패", "중단-경영"):
            line = f"    {pos:<12}"
            for sname, sfn in strata.items():
                a = stratified_auc(recs, key, sfn, pos, "완주")
                ci = boot_ci(recs, key, sfn, pos, "완주", rng) if a is not None else None
                cell = "—" if a is None else (f"{a} [{ci[0]:.2f},{ci[1]:.2f}]" if ci else str(a))
                line += f"{sname}={cell:<24}"
            print(line)

    print("\n[3b] 적응증 라벨만 쓰는 음성 대조군 (프로토콜·타겟 정보 전부 버림)")
    for pos in ("중단-과학적", "중단-모집실패", "중단-경영"):
        by_ind: dict[str, list[int]] = {}
        for r in have:
            if r["group"] in (pos, "완주"):
                by_ind.setdefault(r["ind"], []).append(1 if r["group"] == pos else 0)
        rate = {i: st.fmean(v) for i, v in by_ind.items() if v}
        p = [rate[r["ind"]] for r in have if r["group"] == pos and r["ind"] in rate]
        n = [rate[r["ind"]] for r in have if r["group"] == "완주" and r["ind"] in rate]
        print(f"    {pos:<12} 적응증라벨 단독 AUC = {auc(p, n)}  (n={len(p)} vs {len(n)})")

    print("\n[4] 가장 많이 나온 타겟 20개 (검산용)")
    print("  ", ", ".join(f"{k}:{v}" for k, v in
                          Counter(s for r in recs for s in r["symbols"]).most_common(20)))

    OUT.write_text(json.dumps({
        "_meta": {"source": str(RAW), "ot_api": OT, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                  "disease_id_map": DISEASE_ID,
                  "warning": "overall score 에는 clinical(=ChEMBL 임상시험) 근거가 포함되어 "
                             "심사 대상 시험 자체가 점수에 기여한다. 예측력 주장에 쓸 수 없다."},
        "stats": stats, "records": recs,
    }, ensure_ascii=False, indent=1))
    print(f"\n{OUT} 저장")


if __name__ == "__main__":
    main()
