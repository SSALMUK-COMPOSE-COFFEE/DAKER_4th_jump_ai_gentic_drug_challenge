"""적격기준 항목 파싱 + 의미 범주(concept) 정규화.

## 왜 이 모듈이 필요했나

`validated-numbers.md` 기준으로 검증을 통과한 피처는 **월 모집부담** 하나다. 그런데 방향이
"실패군이 부담이 더 낮다"(1.50 vs 3.37명/월)여서 **교정 조언을 만들 수 없다**. 반대로 교정
조언이 나오는 피처(적격기준 항목수)는 적응증 층화 AUC 0.568, CI 하한 0.50 으로 구분력이 죽었다.

가설: **숫자 하나(항목 수)로 위험을 판정하는 대신 적격기준을 항목 단위로 전례와 대조하면**
약한 AUC 에 의존하지 않고도 교정 가능한 근거를 만들 수 있다. 즉

> "이 프로토콜은 '사전 전신치료 이력 제외' 기준을 두고 있다. 같은 적응증·Phase 완주 시험
>  N건 중 x건(x%)만 이 기준을 두었고, 모집 실패 시험 M건 중 y건(y%)이 두었다. 전례: NCT...".

이 서술은 (1) 특정 기준 하나를 지목하므로 교정 가능하고, (2) 전례 NCT ID 와 `whyStopped`
원문을 인용하므로 근거 결박이며, (3) 원시 빈도 비교이므로 fitting 이 없고, (4) 폐기된
종합 점수에 의존하지 않는다.

이 모듈은 그 서술의 **입력**(항목 파싱 + 의미 범주)만 만든다. 판별력 판정은
`src/analysis/criteria_signal.py` 가 한다.

## 방법론 — 파싱

1. `v0.eligibility_criteria` 는 자유 텍스트 전문이고 HTML 엔티티가 이중 이스케이프된 채
   들어있다(`&#x27;`, `&amp;#x2F;`). `html.unescape` 를 2회 적용한다.
2. 섹션 분리는 **줄 시작에 붙은 `inclusion criteria` / `exclusion criteria` 헤더**를
   전부 찾아 그 위치로 텍스트를 자른다. 헤더가 줄 전체를 차지하지 않고
   `Inclusion Criteria:Patients with ...` 처럼 첫 항목과 붙어 있는 경우가 있어
   "헤더만 있는 줄"을 찾는 방식은 26건을 놓쳤다.
3. 항목 분리는 줄바꿈 기준이다. 실측 1,463건 중 줄바꿈이 3개 미만인 문서는 2건뿐이라
   문장 단위 재분할은 하지 않는다(한계로 보고).
4. 소제목 줄(`Haematology:`, `PATIENT CHARACTERISTICS:` 등)은 항목이 아니므로 제거한다.
   기존 하베스터(`harvest/ctgov.py::_count_criteria`)는 이것을 항목으로 센다 — 즉
   대조 기준값 자체가 정확한 정답이 아니다. 일치율은 그 점을 감안해 읽어야 한다.

## 방법론 — 의미 범주

`labels/taxonomy.py` 와 같은 방식이다: (범주, 정규식) 목록을 위에서부터 매칭하고
**매칭된 근거 문구를 함께 돌려준다**. 심사 대응과 사용자 표시에 원문 인용이 필요하다.

taxonomy 와 다른 점 하나: taxonomy 는 첫 매칭 하나만 채택하지만(중단 사유는 하나),
적격기준 한 항목은 여러 범주에 동시에 속할 수 있어(예: "ECOG 0-1 이고 ANC ≥1500")
**모든 매칭을 채택**한다.

규칙 기반 1차 패스만 한다. LLM 호출은 없다. 규칙이 못 잡는 롱테일은
`unmatched_items` 로 세어 미분류율을 정직하게 보고한다 — **LLM 2차 패스 필요** 지점이다.

출력 스키마 (`parse()`):
    {"mode": str, "inclusion": [item...], "exclusion": [item...],
     "n_inclusion": int, "n_exclusion": int, "dropped_headers": int,
     "concepts": {concept: {"section": "inclusion"|"exclusion"|"both",
                            "evidence": str, "item": str}}, ...}
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RAW = Path("data/raw/ctgov_phase2.jsonl")

# --- 섹션 헤더 -------------------------------------------------------------
# 줄 시작(선행 불릿/번호/공백 허용)에 붙은 헤더만 인정한다. 항목 본문 안의
# "does not meet all inclusion criteria" 같은 언급을 헤더로 오인하지 않기 위한 앵커링.
# 헤더 앞에 수식어가 최대 4단어 붙는 실측 서식이 있다 ("ALS Subject Inclusion Criteria:",
# "Part 1 Inclusion Criteria:", "Key Inclusion Criteria (main ones):").
_HDR = re.compile(
    r"(?m)^[\W\d]{0,6}(?P<pre>(?:[A-Za-z0-9/()'\-]+[ \t]+){0,4})"
    r"(?P<kind>in|ex)clusion[ \t]+(?:criteri\w*|requirements?)",
    re.I,
)

# 헤더 직전 단어가 이것들이면 헤더가 아니라 문장 안의 언급이다.
_HDR_STOPWORDS = {
    "meet", "meets", "meeting", "met", "the", "all", "any", "other", "above", "below",
    "following", "per", "and", "or", "of", "to", "in", "with", "satisfies", "satisfy",
    "fulfil", "fulfils", "fulfill", "fulfills", "fulfilling", "listed", "same", "study",
    "these", "this", "protocol", "either", "both", "one", "no", "not",
}

# NCI/CTEP 서식. exclusion 헤더가 없고 부정 표현("No prior ...")으로 제외를 쓴다.
_NCI_HDR = re.compile(
    r"(?m)^\W{0,4}(disease characteristics|patient characteristics|"
    r"prior concurrent therapy|prior\/concurrent therapy)\s*:?\s*$",
    re.I,
)

_MIN_ITEM_CHARS = 16  # 하베스터의 `len(l.strip()) > 15` 와 동일 임계값 (대조 가능하게 유지)


def clean_text(t: str | None) -> str:
    """이중 이스케이프된 HTML 엔티티를 풀고 개행을 정규화한다."""
    t = html.unescape(html.unescape(t or ""))
    return t.replace("\r\n", "\n").replace("\r", "\n")


def _norm_line(l: str) -> str:
    l = re.sub(r"\s+", " ", l).strip()
    l = re.sub(r"^[\-*•·●‣⁃o]\s+", "", l)          # 불릿
    l = re.sub(r"^\(?\d{1,2}[.)]\s+", "", l)         # 1. / 1) / (1)
    l = re.sub(r"^\(?[a-z][.)]\s+", "", l)           # a. / a)
    return l.strip()


def _is_subheader(l: str) -> bool:
    """소제목 판정. 두 규칙만 쓴다 — 과잉 판정이 실제 기준을 지웠기 때문이다.

    첫 시도에서 "짧고 대문자로 시작하는 줄"까지 소제목으로 봤더니
    `Multiple myeloma`, `Lactating female subject` 같은 **실제 기준 항목**이 삭제됐다.
    남긴 규칙:
      1. 콜론으로 끝나는 줄 — 하위 목록을 여는 도입문이다 (`Adequate end-organ function:`).
      2. 소문자가 전혀 없는 줄 — NCI 서식의 대문자 섹션명 (`DISEASE CHARACTERISTICS`).
    """
    if l.endswith(":"):
        return True
    alpha = [c for c in l if c.isalpha()]
    return len(alpha) >= 2 and not any(c.islower() for c in alpha)


def split_items(chunk: str) -> tuple[list[str], int]:
    """(항목 목록, 버린 소제목 수). 줄바꿈 단위 + 소제목 제거 + 최소 길이."""
    items, dropped = [], 0
    for raw in chunk.split("\n"):
        l = _norm_line(raw)
        if len(l) < _MIN_ITEM_CHARS:
            continue
        if _is_subheader(l):
            dropped += 1
            continue
        items.append(l)
    return items, dropped


def header_hits(t: str) -> list[tuple[int, int, str]]:
    """(시작, 끝, 'in'|'ex') 목록. 문장 안의 언급은 stopword 로 걸러낸다."""
    hits = []
    for m in _HDR.finditer(t):
        pre = (m.group("pre") or "").strip().split()
        if pre and pre[-1].lower().strip(".,;:") in _HDR_STOPWORDS:
            continue  # "does not meet the inclusion criteria" 류
        hits.append((m.start(), m.end(), m.group("kind").lower()))
    return hits


def split_sections(text: str) -> tuple[str, dict[str, str]]:
    """(모드, {"inclusion": 본문, "exclusion": 본문}).

    모드: both / inclusion_only / exclusion_only / nci_style / unsectioned / empty
    헤더가 여러 번 나오면(Part 1 / Part 2, Cohort A/B) 같은 종류끼리 이어붙인다.
    """
    t = clean_text(text)
    if not t.strip():
        return "empty", {"inclusion": "", "exclusion": ""}

    hits = header_hits(t)
    if not hits:
        if _NCI_HDR.search(t):
            return "nci_style", {"inclusion": t, "exclusion": ""}
        return "unsectioned", {"inclusion": t, "exclusion": ""}

    parts: dict[str, list[str]] = {"inclusion": [], "exclusion": []}
    for i, (_s, e, kind) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(t)
        body = t[e:end]
        # 헤더 뒤에 남은 `(main ones):` `:` `-` 등을 털어낸다
        body = re.sub(r"^\s*(\([^)]{0,40}\))?\s*[:\-–.]?\s*", "", body)
        parts["inclusion" if kind == "in" else "exclusion"].append(body)

    has_i, has_e = bool(parts["inclusion"]), bool(parts["exclusion"])
    mode = "both" if (has_i and has_e) else ("inclusion_only" if has_i else "exclusion_only")
    return mode, {k: "\n".join(v) for k, v in parts.items()}


# --- 의미 범주 -------------------------------------------------------------
# (범주, 정규식). 모든 매칭을 채택한다 — 한 항목이 여러 범주에 속할 수 있다.
# 소문자화된 항목 텍스트에 대해 매칭한다.
CONCEPTS: list[tuple[str, str]] = [
    # 사전 치료 이력 — 제한(치료 이력 없어야 함) 과 요구(치료 이력 있어야 함)를 분리한다.
    # 같은 "prior therapy" 어휘를 쓰지만 프로토콜에 주는 함의가 정반대다.
    ("prior_tx_naive_required", (
        r"\b(no|without|never (had|received)|free of|absence of)\s+(any\s+|previous\s+|prior\s+)*"
        r"(prior\s+)?(systemic\s+|cytotoxic\s+|anti-?cancer\s+|anti-?neoplastic\s+|"
        r"disease-?modifying\s+|biologic\s+)?"
        r"(chemotherap\w*|systemic therap\w*|systemic treatment|treatment for|therapy for)"
        r"|\b(treatment|therapy|chemotherapy)-?na[iï]ve\b"
        r"|\bpreviously untreated\b|\buntreated\b.{0,25}(disease|patients?|subjects?)"
        r"|\bfirst-?line\b.{0,20}(only|setting|eligible)|\bno previous\s+(treatment|therapy)"
    )),
    ("prior_tx_required", (
        r"\b(progress\w*|relaps\w*|refractor\w*|fail\w*|intoleran\w*)\s+(on|after|to|following)\b"
        r"|\bat least (one|two|1|2|\d)\s+(prior|previous)\s+(line|regimen|therap|treatment)"
        r"|\breceived (at least )?(one|two|1|2|\d+)\s+(prior|previous)"
        r"|\bmust have (received|had)\s+(prior|previous|at least)"
        r"|\bpreviously treated\b|\bpretreated\b|\bsecond-?line\b|\bthird-?line\b"
        r"|\bprior (platinum|anthracycline|tki|chemotherapy) (is |was )?(required|mandatory)"
    )),
    # 성능 상태
    ("performance_status", (
        r"\b(ecog|zubrod)\b|\bkarnofsky\b|\bkps\b|\blansky\b"
        r"|\b(who|world health organization)\s+performance\b|\bperformance (status|score)\b"
        r"|\bps\s*(of\s*)?[0-9<>=≤≥]"
    )),
    # 수치 검사 컷오프
    ("lab_hematology", (
        r"\b(anc|absolute neutrophil|neutrophil count|neutrophils?)\b.{0,40}[0-9≥≤<>]"
        r"|\bplatelets?\b.{0,40}[0-9≥≤<>]|\bplatelet count\b"
        r"|\bh(a)?emoglobin\b.{0,40}[0-9≥≤<>]|\bhgb\b|\bh(a)?ematocrit\b"
        r"|\b(wbc|white blood cell|leu[ck]ocytes?)\b.{0,40}[0-9≥≤<>]"
        r"|\babsolute lymphocyte count\b"
    )),
    ("lab_renal", (
        r"\b(serum )?creatinine\b|\bcreatinine clearance\b|\bcr?cl\b|\begfr\b"
        r"|\bglomerular filtration\b|\bgfr\b.{0,30}[0-9≥≤<>]"
    )),
    ("lab_hepatic", (
        r"\b(ast|alt|sgot|sgpt|aspartate aminotransferase|alanine aminotransferase|transaminase)\w*\b"
        r"|\b(total |serum |direct )?bilirubin\b|\balkaline phosphatase\b|\balp\b"
        r"|\b(inr|international normalized ratio)\b|\balbumin\b.{0,30}[0-9≥≤<>]"
    )),
    # 나이
    ("age_bound", (
        r"\bage[ds]?\b.{0,25}[0-9]{1,3}\s*(years?|yrs?|y\.?o\.?)"
        r"|\b(≥|>=|>|at least|older than|minimum of)\s*1?[0-9]{1,2}\s*years?\s*(of age|old)?"
        r"|\b(≤|<=|<|no (older|more) than|younger than|up to|maximum of)\s*[0-9]{2,3}\s*years?"
        r"|\b[0-9]{1,2}\s*(to|-|–)\s*[0-9]{2,3}\s*years?\s*(of age|old|inclusive)"
        r"|\b[0-9]{1,3}\s*years?\s*(of age|old)\b"
        r"|\badults?\b.{0,15}\b1[6-9]\s*years?"
    )),
    # 동반질환 제외
    ("comorbid_cardiac", (
        r"\b(myocardial infarction|congestive heart failure|\bchf\b|unstable angina|"
        r"cardiac arrhythmi\w*|arrhythmi\w*|qtc?\b|\bnyha\b|left ventricular ejection|\blvef\b|"
        r"cardiomyopath\w*|cerebrovascular accident|\bstroke\b|coronary artery disease|"
        r"uncontrolled hypertension|cardiovascular (disease|event))\b"
    )),
    ("comorbid_hepatic_renal", (
        r"\b(cirrhosis|hepatic (failure|impairment|insufficiency)|liver (failure|disease)|"
        r"end-?stage renal|renal (failure|impairment|insufficiency)|dialysis|nephrotic)\b"
    )),
    ("comorbid_autoimmune", (
        r"\b(auto-?immune|autoimmunity)\b|\bimmunodeficien\w*\b"
        r"|\bimmunosuppress\w*\b.{0,25}(therapy|treatment|agent|drug|medication)"
        r"|\b(systemic )?corticosteroid\w*\b.{0,30}(chronic|ongoing|require)"
    )),
    ("infection_viral", (
        r"\b(hiv|human immunodeficiency)\b|\bhepatitis\s*(a|b|c|b or c|b/c)?\b"
        r"|\bhbv\b|\bhcv\b|\bhbs?ag\b|\btuberculosis\b|\blatent tb\b"
        r"|\bactive infection\b|\bsystemic (bacterial|fungal|viral) infection\b"
    )),
    # 뇌전이 / CNS
    ("cns_involvement", (
        r"\b(brain|cerebral|cns|central nervous system|intracranial)\s*"
        r"(metasta\w*|involvement|disease|lesion|lymphoma)\b"
        r"|\bleptomeningeal\b|\bcarcinomatous meningitis\b|\bspinal cord compression\b"
    )),
    # 워시아웃
    ("washout_period", (
        r"\bwash-?out\b"
        r"|\bwithin\s+\d+\s*(day|week|month|hour|hr)s?\s*(prior|before|of|preceding|"
        r"prior to)\b"
        r"|\bat least\s+\d+\s*(day|week|month)s?\s*(prior|before|since|must have elapsed|"
        r"have elapsed|after)\b"
        r"|\b\d+\s*(day|week|month)s?\s*(wash-?out|since (the )?last)\b"
        r"|\b(\d+|five|four|three|two)\s*half-?li(fe|ves)\b"
    )),
    # 병용약 금지
    ("concomitant_med_prohibited", (
        r"\bconcomitant\b.{0,40}(prohibit\w*|not (allowed|permitted)|forbidden|excluded)"
        r"|\b(prohibited|disallowed|not permitted|not allowed)\s+(concomitant\s+)?"
        r"(medication|therap|drug|treatment)"
        r"|\b(cyp\s?[0-9][a-z]?[0-9]*)\b.{0,60}(inhibitor|inducer|substrate)"
        r"|\b(strong|potent|moderate)\s+(inhibitor|inducer)s?\b"
        r"|\bp-?gp\b.{0,30}(inhibitor|inducer)"
        r"|\bconcurrent\b.{0,30}(investigational|anti-?cancer|chemotherap|immunosuppress)"
        r"|\b(no|not receiving|must not (be )?(receiv|tak))\w*\s+(any\s+)?other\s+"
        r"(investigational|study)\s+(agent|drug|product)"
    )),
    # 임신·수유·피임
    ("pregnancy_contraception", (
        r"\bpregnan\w*\b|\bbreast-?feed\w*\b|\bnursing\b|\blactat\w*\b"
        r"|\bcontracept\w*\b|\bchild-?bearing potential\b|\bwocbp\b"
        r"|\b(negative|serum|urine)\s+(beta-?h?cg|pregnancy test)\b"
    )),
    # 이전 임상시험 참여
    ("prior_trial_participation", (
        r"\b(participat\w*|enroll\w*|enrol\w*)\b.{0,45}\b(another|other|any other|previous|prior|"
        r"concurrent|different)\s+(clinical\s+)?(trial|stud[yi]|investigation)"
        r"|\b(another|other|prior|previous)\s+(clinical\s+)?(trial|stud[yi]\w*)\b.{0,40}"
        r"\b(within|particip|enroll|enrol)"
        r"|\bprior (exposure to|treatment with) (an )?investigational\b"
        r"|\bpreviously (received|treated with)\b.{0,30}investigational"
    )),
    # 조직학적 확진 / 측정가능 병변
    ("histologic_confirmation", (
        r"\b(histolog\w*|cytolog\w*|patholog\w*|biops[yi]\w*)\w*\s*(and\s*\/?\s*or\s*"
        r"cytolog\w*\s*)?(confirm\w*|proven|document\w*|verified|diagnos\w*)\b"
        r"|\b(confirmed|proven)\s+(by\s+)?(histolog|cytolog|biops)"
    )),
    ("measurable_disease", (
        r"\bmeasurable\s+(disease|lesion|target)\b|\brecist\b|\bircist\b|\birecist\b"
        r"|\bevaluable disease\b|\brano\b|\bcheson\b|\bwho criteria\b.{0,20}(measur|respons)"
    )),
    # 바이오마커·유전형 요구
    ("biomarker_requirement", (
        r"\b(egfr|alk|ros1?|kras|nras|braf|her-?2|erbb2|met\b|ret\b|ntrk|pik3ca|"
        r"pd-?l1|pd-?1|msi-?h|dmmr|brca[12]?|idh[12]?|flt3|npm1|jak2|bcr-?abl|"
        r"tp53|cd20|cd19|cd33|cftr|apoe|hla-?[a-z]?[0-9]*)\b"
        r"|\b(mutation|mutant|amplification|translocation|rearrangement|fusion|"
        r"overexpression|expression level)\b.{0,30}(positive|required|documented|present|confirm)"
        r"|\b(biomarker|genotype|mutational status)\b.{0,30}(required|positive|confirmed|documented)"
    )),
    # 2차 악성종양 제외 — 종양 시험에서 매우 흔하고 모집 폭을 크게 줄인다
    ("second_malignancy", (
        r"\b(second|other|another|prior|previous|concurrent|additional)\s+"
        r"(primary\s+|invasive\s+|active\s+)?(malignanc\w*|cancer|neoplas\w*|tumo?r)\b"
        r"|\bhistory of (any )?(other )?(malignanc\w*|cancer)\b"
    )),
]

CONCEPT_NAMES = {
    "prior_tx_naive_required": "사전 치료 이력 없어야 함 (treatment-naive 요구)",
    "prior_tx_required": "사전 치료 이력 있어야 함 (2차 이상 라인)",
    "performance_status": "ECOG/KPS 성능 상태 컷오프",
    "lab_hematology": "혈액 수치 컷오프 (호중구·혈소판·헤모글로빈)",
    "lab_renal": "신기능 컷오프 (크레아티닌·eGFR)",
    "lab_hepatic": "간기능 컷오프 (AST/ALT/빌리루빈)",
    "age_bound": "나이 상·하한",
    "comorbid_cardiac": "심장 동반질환 제외",
    "comorbid_hepatic_renal": "간·신 동반질환 제외",
    "comorbid_autoimmune": "자가면역·면역억제 제외",
    "infection_viral": "감염 제외 (HIV/HBV/HCV/TB)",
    "cns_involvement": "뇌전이·CNS 침범",
    "washout_period": "워시아웃 기간",
    "concomitant_med_prohibited": "병용약 금지",
    "pregnancy_contraception": "임신·수유·피임 요구",
    "prior_trial_participation": "이전 임상시험 참여 제외",
    "histologic_confirmation": "조직학적 확진 요구",
    "measurable_disease": "측정가능 병변(RECIST) 요구",
    "biomarker_requirement": "바이오마커·유전형 요구",
    "second_malignancy": "2차·기왕 악성종양 제외",
}

_COMPILED = [(name, re.compile(pat, re.I)) for name, pat in CONCEPTS]


def detect_concepts(items: list[str], section: str) -> dict[str, dict]:
    """항목별로 전 범주를 매칭. {범주: {"section", "evidence", "item"}} 를 돌려준다.

    근거 문구(`evidence`)와 원문 항목(`item`)을 같이 담는 이유: 사용자에게 제시할 문장이
    "어느 기준을 지목하는지"를 원문으로 보여야 교정 가능한 조언이 되기 때문이다.
    """
    found: dict[str, dict] = {}
    for it in items:
        for name, rx in _COMPILED:
            m = rx.search(it)
            if not m:
                continue
            if name in found:
                continue
            found[name] = {"section": section, "evidence": m.group(0)[:80], "item": it[:300]}
    return found


def parse(text: str | None) -> dict:
    """적격기준 전문 → 섹션·항목·의미 범주."""
    mode, parts = split_sections(text or "")
    inc, d1 = split_items(parts["inclusion"])
    exc, d2 = split_items(parts["exclusion"])
    hits = header_hits(clean_text(text))
    n_inc_hdr = sum(1 for h in hits if h[2] == "in")

    concepts: dict[str, dict] = {}
    for c in (detect_concepts(exc, "exclusion"), detect_concepts(inc, "inclusion")):
        for k, v in c.items():
            if k in concepts:
                concepts[k]["section"] = "both"
            else:
                concepts[k] = v

    matched_items = 0
    for it in inc + exc:
        if any(rx.search(it) for _n, rx in _COMPILED):
            matched_items += 1

    return {
        "mode": mode,
        "inclusion": inc,
        "exclusion": exc,
        "n_inclusion": len(inc),
        "n_exclusion": len(exc),
        "dropped_inc": d1,
        "dropped_exc": d2,
        "n_inc_hdr": n_inc_hdr,
        "dropped_headers": d1 + d2,
        "n_items": len(inc) + len(exc),
        "n_items_matched": matched_items,
        "concepts": concepts,
    }


# --- 파서 검증 -------------------------------------------------------------


def validate(rows: list[dict]) -> dict:
    """하베스터의 `n_inclusion_items`/`n_exclusion_items` 와 대조한다.

    주의 — 이 대조는 "정답 대조"가 아니다. 하베스터 카운터는
    `exclusion criteria` 로 1회 split 하고 15자 초과 줄을 세는 것이 전부이므로
    소제목을 항목으로 세고, exclusion 헤더가 없는 서식에서 exc=0 이 된다.
    따라서 **불일치의 일부는 이쪽이 맞는 경우**다. 그 판단을 섞지 않기 위해
    모드별로 나눠 보고한다.
    """
    from collections import Counter

    stat = Counter()
    diffs = {"inc": [], "exc": [], "inc_raw": [], "exc_raw": []}
    per_mode: dict[str, Counter] = {}
    for r in rows:
        v0 = r["v0"]
        p = parse(v0.get("eligibility_criteria"))
        m = per_mode.setdefault(p["mode"], Counter())
        m["n"] += 1
        stat["n"] += 1
        stat["items"] += p["n_items"]
        stat["items_matched"] += p["n_items_matched"]
        if p["n_items"] == 0:
            stat["parse_fail"] += 1
            m["parse_fail"] += 1
            continue
        bi = v0.get("n_inclusion_items") or 0
        be = v0.get("n_exclusion_items") or 0
        # `_raw` = 소제목을 되살린 카운트. 하베스터는 소제목도 항목으로 세므로
        # 이쪽이 "항목 경계 판정이 하베스터와 같은가"를 직접 재는 값이다.
        cand = {
            "inc": p["n_inclusion"] - bi,
            "exc": p["n_exclusion"] - be,
            # 하베스터는 `Inclusion Criteria:` 헤더 줄 자체도 15자 초과라 항목으로 센다.
            # 그 +1 을 되살려야 "항목 경계가 같은가"만 남는다.
            "inc_raw": p["n_inclusion"] + p["dropped_inc"] + p["n_inc_hdr"] - bi,
            "exc_raw": p["n_exclusion"] + p["dropped_exc"] - be,
        }
        for tag, d in cand.items():
            diffs[tag].append(d)
            if d == 0:
                stat[f"{tag}_exact"] += 1
                m[f"{tag}_exact"] += 1
            if abs(d) <= 1:
                stat[f"{tag}_pm1"] += 1
            if abs(d) <= 2:
                stat[f"{tag}_pm2"] += 1
            base = max(bi if tag.startswith("inc") else be, 1)
            if abs(d) / base <= 0.10:
                stat[f"{tag}_pct10"] += 1
        if p["n_items_matched"] == 0:
            stat["no_concept"] += 1
    return {"stat": stat, "per_mode": per_mode, "diffs": diffs}


if __name__ == "__main__":
    import statistics as st
    from collections import Counter

    rows = [json.loads(l) for l in RAW.open() if l.strip()]
    v = validate(rows)
    s, n = v["stat"], v["stat"]["n"]

    print(f"=== 파서 검증 (n={n}) ===")
    print(f"  파싱 실패(항목 0개)        {s['parse_fail']:>5}건  {s['parse_fail']/n:6.1%}")
    ok = n - s["parse_fail"]
    print(f"  총 항목 {s['items']}개 / 범주 매칭 항목 {s['items_matched']}개 "
          f"({s['items_matched']/s['items']:.1%}) → 미분류 항목 {1-s['items_matched']/s['items']:.1%}")
    print(f"  범주 0개 검출 트라이얼      {s['no_concept']:>5}건  {s['no_concept']/ok:6.1%}")
    print("\n  하베스터 카운트 대조 (파싱 성공 %d건)" % ok)
    print(f"    {'':<22}{'정확':>8}{'±1':>8}{'±2':>8}{'±10%':>8}{'중앙차':>8}{'평균차':>8}")
    labels = {
        "inc": "포함(소제목 제거)", "inc_raw": "포함(소제목 포함)",
        "exc": "제외(소제목 제거)", "exc_raw": "제외(소제목 포함)",
    }
    for tag in ("inc", "inc_raw", "exc", "exc_raw"):
        d = v["diffs"][tag]
        print(f"    {labels[tag]:<20}{s[tag+'_exact']/ok:>8.1%}{s[tag+'_pm1']/ok:>8.1%}"
              f"{s[tag+'_pm2']/ok:>8.1%}{s[tag+'_pct10']/ok:>8.1%}"
              f"{st.median(d):>+8.1f}{st.fmean(d):>+8.2f}")

    print("\n  서식 모드별")
    print(f"    {'mode':<16}{'n':>6}{'inc_raw정확':>13}{'exc_raw정확':>13}{'실패':>7}")
    for mode, m in sorted(v["per_mode"].items(), key=lambda x: -x[1]["n"]):
        mo = m["n"] - m["parse_fail"]
        print(f"    {mode:<16}{m['n']:>6}"
              f"{(m['inc_raw_exact']/mo if mo else 0):>13.1%}"
              f"{(m['exc_raw_exact']/mo if mo else 0):>13.1%}{m['parse_fail']:>7}")

    print("\n=== 의미 범주 검출률 (전체 %d건) ===" % n)
    cnt: Counter = Counter()
    sec: dict[str, Counter] = {}
    for r in rows:
        p = parse(r["v0"].get("eligibility_criteria"))
        for k, meta in p["concepts"].items():
            cnt[k] += 1
            sec.setdefault(k, Counter())[meta["section"]] += 1
    print(f"  {'범주':<28}{'검출':>7}{'검출률':>8}   섹션분포(inc/exc/both)")
    for k, _p in CONCEPTS:
        c = cnt[k]
        sd = sec.get(k, Counter())
        print(f"  {k:<28}{c:>7}{c/n:>8.1%}   "
              f"{sd['inclusion']}/{sd['exclusion']}/{sd['both']}   {CONCEPT_NAMES[k]}")
