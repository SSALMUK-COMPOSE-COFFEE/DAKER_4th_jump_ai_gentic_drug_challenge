"""중단 사유 4분류 라벨러.

A(설계 교정 가능) / B(과학적) / C(외생적·경영) / D(경쟁·표준치료 변화) 로 나눈다.

분류를 나누는 기준은 **에이전트가 등록 시점에 예견할 수 있었는가**다.
- A: 프로토콜을 고쳐서 막을 수 있었다 → 예측 + 교정 제안
- B: 생물학적 결과. 전례 기반 예측만 시도
- C: 스폰서 경영 판단. 프로토콜과 무관 → **명시적 기권**
- D: 표준치료·경쟁 환경 변화. 등록 시점의 경쟁 시험 밀도로 **부분적으로 예견 가능** → 별도 취급

D를 C에 뭉개지 않고 분리한 이유: 실데이터에 "표준치료가 바뀌어 모집이 불가능해졌다",
"면역치료제가 나와 더 이상 모집이 타당하지 않다" 류가 뚜렷한 덩어리로 존재하고,
이것은 `audit/competition.py` 의 경쟁 시험 밀도 피처가 정확히 겨냥하는 대상이다.
C(경영 판단)와 섞으면 예견 가능한 실패를 예견 불가능으로 잘못 처리하게 된다.

규칙 기반 1차 패스를 먼저 두는 이유: 투명하고 재현 가능하며, 심사 시 근거를 그대로 보여줄 수 있다.
규칙으로 안 잡히는 것만 LLM에 넘긴다(2차 패스, 별도 모듈).
"""

from __future__ import annotations

import html
import re

# 순서가 중요하다. 위에서부터 먼저 매칭된 것을 채택한다.
# D와 C를 A보다 먼저 본다 — "자금 부족으로 모집 중단"은 모집 실패처럼 보여도 외생 요인이다.
PATTERNS: list[tuple[str, str, str]] = [
    # (분류, 세부 사유, 정규식)

    # D(경쟁·표준치료 변화) — 등록 시점 경쟁 밀도로 부분 예견 가능
    # 양방향으로 잡는다 — "Changing treatment landscape: ..." 처럼 수식어가 앞에 오는 형태가 많다.
    ("D", "landscape_shift", (
        r"(standard of care|treatment landscape|therapeutic landscape).{0,40}?(chang|evolv|shift)"
        r"|(chang\w*|evolv\w*|shift\w*)\s+(in\s+the\s+)?(standard of care|treatment landscape|therapeutic landscape)"
        r"|change in the standard of care"
    )),
    ("D", "competing_options", r"(new|other|alternative)\s*(studies|trials|therap|treatment|agent|option)\w*\s*(available|approved|emerged)|recruitment is no longer (acceptable|appropriate|feasible)|no longer (relevant|competitive)|approvals? of new (agents?|drugs?|therap\w+)|new (agents?|drugs?)\s+.{0,40}slow\w*\s+down\s+(the\s+)?recruit"),
    ("D", "external_data", r"(emerging|external|new)\s*(data|evidence|result)\w*\s*(with|from|on|showed|indicated)|based on emerging data|because of (the )?results? of\s+\w+\s+trial|results? of\s+.{0,20}(study|trial)\s+.{0,20}reached significance|pending additional phase\s*\d"),
    ("D", "rationale_obsolete", r"(scientific )?rationale\s*\w*\s*(deemed )?(obsolete|outdated|no longer valid)"),

    # C(외생적·경영) — 프로토콜과 무관, 기권 대상
    # 어간 뒤에 `\b` 를 붙이면 활용형이 전부 실패한다 (`fund(ing|er)?s?\b` 는 "funded" 를 놓쳤다).
    # 어간형은 `\w*` 로 열어둔다. 아래 B/safety 의 toxicit·tolerabilit 도 같은 버그였다.
    ("C", "funding", r"\bfund\w*|\b(financial|budget|monetary|money)\b|sponsor .*(withdrew|withdrawn)|withdrew\s+(the\s+)?(support|funding)|(support|funding) terminated"),
    ("C", "business_decision", r"(business|strategic|strategy|portfolio|commercial|company|corporate)\s*\w*\s*(decision|reason|priorit|reprioriti)|repriorit|(company|corporate|business)\'?s?\s+strategy|out-?licens|rights?\s+.{0,30}\breturned\s+to|business objectives?\s+.{0,20}chang|program\s+refocus|refocus\w*\s+.{0,20}program|internal review of the compan|future clinical development plans|decision to discontinue\s+.{0,40}\bdevelopment"),
    ("C", "program_terminated", r"(development|program|project)\s*(program\s*)?(was\s*)?(terminated|discontinued|halted|stopped|cancel)"),
    ("C", "sponsor_decision", r"\bsponsor'?s?\b.{0,25}\b(decision|choice|elected|decided|terminat\w*|cancel\w*|withdrew|withdrawn|closed)\b|(terminat|cancel|withdraw|recall)\w*\s*by (the )?sponsor|sponsor\s+recalled"),
    ("C", "covid", r"\bcovid|coronavirus|pandemic\b"),
    # `relocat\b` / `retire\b` 도 후행 `\b` 로 활용형("relocated","retired")을 놓쳤다.
    ("C", "pi_left", (
        r"\b(investigator|pi)\b.{0,25}\b(left|departed?|relocat\w*|retir\w*|transferr?ed|no longer at)"
        r"|change of (pi|investigator)"
        r"|departure of (the )?(study )?(team|staff|pi|investigator)"
        r"|\b(pi|investigator)\b.{0,15}departure"
        r"|investigator\'?s?\s+decision"
    )),
    ("C", "drug_supply", (
        r"(drug|study product|supply|manufactur|film|device|kit)\s*\w*\s*"
        r"(shortage|unavailab|supply issue|discontinued by)"
        r"|(expired|out of stock)\b.{0,30}(no longer )?availab|no longer availab"
        # "Test products expired." — 뒤에 availab 이 안 붙는 형태
        r"|\b(product|drug|test|kit|material)s?\s+.{0,15}expired|expired\s+\w{0,15}(product|drug|kit)"
        r"|(study\s+)?drug\s+.{0,25}(not (provided|supplied|available)|issues?)"
        r"|stability of\s+.{0,35}(could not|not) be established"
    )),
    ("C", "merger", r"\b(acquisition|acquired|merger|merged)\b"),
    ("C", "interest_withdrawn", r"\binterest\s*(was\s*)?withdrawn\b|lost interest|\babandon"),

    # B(과학적) — 프로토콜 설계로 막기 어려운 생물학적 결과
    ("B", "lack_of_efficacy", r"(lack|absence|insufficient|no evidence|failure)\s*of\s*(efficacy|effect|benefit|response)|futility|did not (meet|achieve|reach)|ineffective|not (reach|meet)\w*\s*the (targeted|target|required)|benefit-?\/?risk\s*(profile)?\s*.{0,30}(did not support|adverse change)|adverse change in the risk|missed endpoint|insufficient efficacy|inability to meet protocol objectives|insufficient\s+\w+\s+(engraftment|response)"),
    # agranulocytosis 처럼 구체 병명만 적힌 경우가 있어 "N cases of ~" 형태를 안전성으로 잡는다.
    #
    # ⚠ 과거 버그: `\b(safety|toxicit|...|tolerabilit|death)\b` 로 쓰여 있어 후행 `\b` 가
    # 어간형을 전부 죽였다 — "toxicity", "toxicities", "tolerability", "adverse events"(복수),
    # "deaths" 가 모두 매칭 실패했다. 단수 "adverse event" 만 잡혔다.
    # 결과적으로 B(과학적) 분류가 체계적으로 과소집계되어 있었다.
    ("B", "safety_toxicity", (
        r"\b(safety|toxic\w*|tolerabilit\w*|mortalit\w*)"
        r"|\badverse\s+(event|reaction|effect)s?"
        r"|\bdeaths?\b|\b(ae|sae|dlt)s?\b"
        r"|\d+\s*(cases?|events?|episodes?)\s*of\s*\w+"
        # 구체 독성 소견만 기재된 경우 (간효소 상승, 검사치 이상)
        r"|liver\s+(enzyme|biochemical|function)|(elevat\w*|abnormal)\s+.{0,25}(liver|hepatic|enzyme)"
    )),
    ("B", "interim_analysis", r"interim (analysis|review|result)|\b(dsmb|idmc|dmc)\b|data (safety )?monitoring"),

    # A(설계 교정 가능) — 프로토콜을 고쳐서 막을 수 있었던 것
    ("A", "recruitment", (
        # `low` 는 "lower than expected" 를, `accrual` 은 "accruals" 를 놓쳤다 → 활용형 허용
        r"\b(slow\w*|poor|low(er)?|insufficient|inadequate|lack of|limited|no)\s+(\w+\s+){0,2}"
        r"(accruals?|enrol(l)?ments?|recruit(ment|ing)?|randomi[sz]ations?|participants?|patients?|subjects?)\b"
        r"|(recruit|enroll|enrol|accru|randomi[sz])\w*\s*(was |were )?\w*\s*"
        r"(difficult|challeng|slow|insufficient|poor|fail|issue|problem|hard|paus|too slow)"
        # 역순 표현 — "difficult recruitement"(오타 포함), "difficulty in recruiting subjects"
        r"|(difficult\w*|slow\w*|poor|challeng\w*)\s+(\w+\s+){0,2}(recruit\w*|enrol\w*|accru\w*)"
        r"|unable to (recruit|enroll|enrol|accrue)"
        r"|failed to (recruit|enroll|enrol|accrue|meet its accrual)"
        r"|(not|never)\s+(accruing|enrolled|recruited)"
        r"|no (patients?|participants?|subjects?)\s*(were\s*)?(available|enrolled|recruited)"
        r"|insufficient patient population"
        r"|only \d+ patients? could be recruited"
        # "Accrual stopped", "enrollment closed early" — 선행 형용사 없이 동사만 오는 형태
        r"|\b(accruals?|enrol(l)?ments?|recruit(ment|ing)?)\s+(was\s+|were\s+)?(stopped|halted|closed|ceased|suspended)"
        r"|few (patients?|subjects?|participants?)\s+(would|were|could)"
        r"|before enrolling (the )?first patient|failure to meet\s+(the\s+)?recruit\w*\s*target|in advance of targeted enrollment"
    )),
    ("A", "eligibility_too_narrow", r"(eligib|inclusion|exclusion)\w*\s*(criteria)?\s*(too )?(strict|narrow|restrictive)|screen failure|no suitable patients|we need at least|hard to find (suitable )?(subject|patient|participant)"),
    ("A", "endpoint_issue", r"(endpoint|outcome measure|assay|biomarker)\s*\w*\s*(issue|problem|not (feasible|valid)|changed|inadequate)"),
    ("A", "protocol_design", r"(protocol|study design|design)\s*\w*\s*(amend|revis|redesign|change|flaw|error)|feasibilit\w+ (issue|concern)"),
    ("A", "site_logistics", r"(site|center|centre|staff|resource|logistic)\w*\s*(issue|problem|closure|closed|shortage|unavailab)"),

    # 개시 자체를 안 한 경우 — 사유가 더 특정되지 않으면 별도 취급
    ("C", "never_initiated", r"\b(trial|study)\s*not\s*(initiated|started|conducted)\b|not yet submitted"),
]

CATEGORY_NAMES = {
    "A": "설계 교정 가능",
    "B": "과학적",
    "C": "외생적·경영",
    "D": "경쟁·표준치료 변화",
    "?": "미분류",
}

# 에이전트가 예측을 시도해야 하는 분류. C는 기권 대상.
PREDICTABLE = ("A", "B", "D")


def _normalize(why_stopped: str) -> str:
    # 원문에 &#x27; 같은 HTML 엔티티가 그대로 들어있다 (company&#x27;s strategy)
    return re.sub(r"\s+", " ", html.unescape(why_stopped).lower())


def classify_all(why_stopped: str | None) -> list[tuple[str, str, str]]:
    """매칭된 **모든** 분류를 패턴 순서대로 반환 (분류당 첫 매칭 1개).

    `whyStopped` 에 사유가 복합으로 기재되는 경우가 실측 7.9% 있고
    ("Sponsor decision due to slow accrual"), `classify()` 는 우선순위상 첫 번째만 채택한다.
    코호트 정의의 민감도 분석을 하려면 매칭 전체가 필요하다.
    """
    if not why_stopped or not why_stopped.strip():
        return []
    text = _normalize(why_stopped)
    seen: dict[str, tuple[str, str, str]] = {}
    for cat, reason, pattern in PATTERNS:
        if cat in seen:
            continue
        m = re.search(pattern, text)
        if m:
            seen[cat] = (cat, reason, m.group(0))
    return list(seen.values())


def classify(why_stopped: str | None, *, a_priority: bool = False) -> tuple[str, str, str | None]:
    """(분류, 세부사유, 매칭된 근거 문구) 반환. 근거를 함께 돌려주는 게 심사 대응에 중요하다.

    `a_priority=False` (기본) — 패턴 순서 D→C→B→A 로 첫 매칭을 채택한다.
        근거: "자금 부족으로 모집 중단"은 모집 실패처럼 보여도 외생 요인이라는 판단.
    `a_priority=True` — A 가 매칭되면 다른 분류를 이기고 A 를 채택한다.
        근거: "Sponsor decision due to slow accrual" 은 모집 실패가 **원인**이고
        스폰서 결정은 종료 **절차**다.

    두 정의는 실측 33건(사유 기재분의 4.3%, A 코호트의 19%)에서 갈린다. 어느 쪽이 옳은지
    데이터만으로는 결정할 수 없으므로 **양쪽으로 결과를 내고 결론이 정의에 의존하는지 보고한다.**
    `src/analysis/cohort_sensitivity.py` 참조.
    """
    if not why_stopped or not why_stopped.strip():
        return "?", "empty", None
    if a_priority:
        for cat, reason, ev in classify_all(why_stopped):
            if cat == "A":
                return cat, reason, ev
    text = _normalize(why_stopped)
    for cat, reason, pattern in PATTERNS:
        m = re.search(pattern, text)
        if m:
            return cat, reason, m.group(0)
    return "?", "unmatched", None


def enrollment_attainment(planned: int | None, actual: int | None) -> float | None:
    """등록 달성률. 목표가 없거나 0이면 계산 불가."""
    if not planned or actual is None:
        return None
    return round(actual / planned, 3)


if __name__ == "__main__":
    samples = [
        "Lack of efficacy",
        "Recruitment difficulties",
        "Development program terminated.",
        "Due to the two AEs in Phase IIa study, recruiting was stopped for safety sake.",
        "Slow accrual",
        "Business reasons, not related to safety",
        "Study halted prematurely due to COVID-19 pandemic",
        "Sponsor decision",
        "Terminated due to lack of funding",
        "Phase 2 study results",
        "Inclusion criteria too restrictive, high screen failure rate",
    ]
    for s in samples:
        cat, reason, ev = classify(s)
        print(f"{cat} [{CATEGORY_NAMES[cat]:<10}] {reason:<22} ← {s!r}  (근거: {ev!r})")
