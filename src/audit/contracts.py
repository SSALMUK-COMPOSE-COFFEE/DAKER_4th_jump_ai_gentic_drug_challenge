"""에이전트–도구 간 계약 (자료형 + 인용 규약).

## 의존 방향

**도구 계층이 계약을 정의하고, 에이전트가 거기에 맞춘다.** 그 반대가 아니다.
A4 레드팀이 무엇을 출력하든 T3(`referee.py`)가 기계적으로 검증할 수 있어야 하므로,
검증 가능한 형태를 도구 쪽에서 먼저 못 박는다.

## 근거 결박 계약

**모든 공격은 아래 4종 중 최소 1개의 인용을 가져야 한다. 없으면 공격이 아니다.**

| 인용 유형 | 필수 필드 | T3 의 검증 |
|---|---|---|
| `PRECEDENT` | `nct_id`, `quote` | 코퍼스 조회 → ID 실존 · 인용문 원문 포함 · **개시일 < 컷오프** · A 분류 여부 |
| `QUANT` | `feature`, `value` | T1 출력과 수치 일치 · 해당 피처가 위험 판정 자격이 있는지 |
| `LITERATURE` | `pmid`, `quote`, `pub_date` | PMID 실존 · **발행일 ≤ 컷오프** · 원문 대조 |
| `REGULATORY` | `doc`, `section`, `quote` | 문서 원문 문자열 포함 |

## 공격 근거 금지 목록

수정 불가능하거나 검증에서 탈락한 근거는 공격에 쓸 수 없다. T3 가 강제한다.
특히 **스폰서 유형**은 모집실패군 학술 74.4% vs 완주군 산업 50.1% 로 강하게 갈리지만,
이를 학습하면 "학술 스폰서라 실패한다"는 교정 불가능한 조언이 되고 프로토콜 감사가 아니라
신분 판별기가 된다.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

# --- 공격이 겨냥할 수 없는 것 -------------------------------------------------

FORBIDDEN_GROUNDS: dict[str, str] = {
    "sponsor_class": "스폰서 유형은 수정 불가능한 신분이다. 프로토콜 감사가 아니라 신분 판별기가 된다.",
    "sponsor": "위와 같다.",
    "indication": "적응증 라벨만 쓰는 대조 예측기가 AUC 0.722 다. 적응증을 근거로 쓰면 질환 판별기다.",
    "condition": "위와 같다.",
    "n_concurrent_phase2": "경쟁 시험 밀도는 적응증 층화 후 0.472 로 철회되었고 프로토콜로 변경 불가능하다.",
    "n_concurrent_trials": "위와 같다.",
    "total_competing_seats": "위와 같다.",
    "composite_risk_score": "종합 위험 점수는 홀드아웃 0.587 [0.484, 0.693] 로 캘리브레이션 미입증이다.",
}

# 인용문이 이보다 짧으면 원문 대조가 무의미하다("the", "study" 등이 항상 통과한다).
MIN_QUOTE_CHARS = 12


class Reject:
    """기각 사유 코드. 기각률을 **사유별로 분해**해 보고하기 위해 문자열이 아니라 코드를 쓴다."""

    NO_CITATION = "무근거 공격 (인용 0개)"
    FORBIDDEN_GROUND = "금지된 공격 근거"
    UNKNOWN_KIND = "알 수 없는 인용 유형"
    QUOTE_TOO_SHORT = "인용문이 너무 짧아 대조 불가"

    PRECEDENT_NOT_FOUND = "전례 NCT ID 가 코퍼스에 없음 (환각)"
    PRECEDENT_QUOTE_MISMATCH = "전례 인용문이 원문과 불일치 (환각)"
    PRECEDENT_CUTOFF = "시간 컷오프 위반 (미래 전례 인용)"
    PRECEDENT_NOT_A = "모집 공격의 근거인데 A 분류(모집 실패) 전례가 아님"

    QUANT_UNKNOWN_FEATURE = "알 수 없는 피처"
    QUANT_MISMATCH = "수치가 T1 출력과 불일치 (환각)"
    QUANT_NOT_ELIGIBLE = "검증을 통과하지 못한 피처를 위험 근거로 사용"

    LITERATURE_UNVERIFIED = "PMID 를 확인할 수 없음"
    LITERATURE_CUTOFF = "발행일이 컷오프 이후 (미래 문헌 인용)"
    REGULATORY_NOT_FOUND = "규제 문서 원문에서 인용문을 찾을 수 없음"


@dataclass
class Citation:
    kind: str                       # PRECEDENT | QUANT | LITERATURE | REGULATORY
    nct_id: str | None = None
    quote: str | None = None
    feature: str | None = None
    value: float | None = None
    pmid: str | None = None
    pub_date: str | None = None
    doc: str | None = None
    section: str | None = None

    @classmethod
    def from_obj(cls, o: dict) -> "Citation":
        f = {k: o.get(k) for k in cls.__dataclass_fields__ if k != "kind"}
        v = f.get("value")
        if isinstance(v, str):
            try:
                f["value"] = float(re.sub(r"[^0-9.\-]", "", v) or "nan")
            except ValueError:
                f["value"] = None
        return cls(kind=str(o.get("kind", "")).upper(), **f)


@dataclass
class Attack:
    target_element: str             # 공격이 겨냥하는 프로토콜 요소 (수정 가능해야 한다)
    claim: str                      # 지적 문장
    citations: list[Citation] = field(default_factory=list)
    suggestion: str = ""            # 교정 방향 (선택)
    topic: str = "recruitment"      # 공격 주제. 모집 공격은 A 분류 전례만 허용된다

    @classmethod
    def from_obj(cls, o: dict) -> "Attack":
        return cls(
            target_element=str(o.get("target_element", "")).strip(),
            claim=str(o.get("claim", "")).strip(),
            citations=[Citation.from_obj(c) for c in (o.get("citations") or [])
                       if isinstance(c, dict)],
            suggestion=str(o.get("suggestion", "") or "").strip(),
            topic=str(o.get("topic", "recruitment") or "recruitment").strip(),
        )


@dataclass
class Verdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)      # 기각 사유 코드 (복수 가능)
    detail: list[str] = field(default_factory=list)       # 사람이 읽을 설명
    checked: int = 0                                      # 실제로 수행한 검사 수

    def reject(self, code: str, detail: str = "") -> "Verdict":
        self.passed = False
        self.reasons.append(code)
        if detail:
            self.detail.append(detail)
        return self


# --- A4 프롬프트에 그대로 넣을 출력 규약 ---------------------------------------

ATTACK_OUTPUT_SPEC = """반드시 아래 형태의 JSON 배열만 출력한다. 설명 문장을 붙이지 않는다.

[
  {
    "target_element": "수정 가능한 프로토콜 요소명 (예: eligibility.neutrophil_cutoff, planned_enrollment)",
    "topic": "recruitment",
    "claim": "지적 내용 한두 문장. 반드시 아래 인용으로 뒷받침되어야 한다.",
    "suggestion": "구체적 교정 방향 한 문장",
    "citations": [
      {"kind": "PRECEDENT", "nct_id": "NCT01234567", "quote": "why_stopped 원문에서 그대로 복사한 구절"},
      {"kind": "QUANT", "feature": "monthly_enrollment_burden", "value": 12.5}
    ]
  }
]

규칙:
- **재료가 제공되었다면 최소 1건, 최대 4건을 출력한다.** 빈 배열은 재료가 없을 때만 쓴다.
- 인용이 하나도 없는 공격은 금지한다. 근거를 댈 수 없으면 그 공격을 만들지 마라.
- PRECEDENT 의 quote 는 제공된 전례 목록의 why_stopped 원문에서 **글자 그대로 복사**한다.
  요약하거나 바꿔 쓰면 기각된다.
- 제공된 목록에 없는 NCT ID 를 지어내지 마라. 기각된다.
- QUANT 의 value 는 제공된 계량 감사 수치와 정확히 일치해야 한다.
- 다음은 공격 근거로 쓸 수 없다: 스폰서 유형, 적응증 자체의 난이도, 경쟁 시험 밀도, 종합 위험 점수.
- 공격은 **프로토콜을 고쳐서 바꿀 수 있는 요소**만 겨냥한다."""


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda o: asdict(o))
