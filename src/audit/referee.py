"""T3 인용 대조 〈도구〉 — 이 설계의 핵심 독창성.

## 왜 이 방식인가

**대부분의 환각 통제 장치는 LLM 이 LLM 을 심사한다. 그건 검증이 아니라 의견의 중첩이다.**

T3 는 공격의 **설득력을 판정하지 않는다.** 인용이 실제로 존재하는지만 기계적으로 확인한다.
LLM 호출이 0회이고, 같은 입력에 항상 같은 판정이 나오며, 심사위원이 직접 재현할 수 있다.

## 이 설계의 이점 셋

1. 환각을 의견이 아니라 **원문 대조**로 거른다.
2. **기각률이 정답 라벨 없이 측정된다** — 코퍼스만 있으면 된다.
3. 모델·프롬프트를 바꿀 때마다 재측정할 수 있다 (모델 등급을 낮추면 기각률이 오르는지).

## 구조적 한계 — 반드시 함께 보고한다

문자열 포함 검사는 **인용문을 정확히 복사하되 맥락을 왜곡하는** 실패 모드를 잡지 못한다.
검출 가능한 것(ID 위조·수치 조작·시간 위반)과 검출 불가한 것(맥락 왜곡)을 구분해 보고한다.

실행: python3 src/audit/referee.py    # 합성 위조 공격으로 게이트 검출력 자가 시험
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.contracts import (  # noqa: E402
    FORBIDDEN_GROUNDS, MIN_QUOTE_CHARS, Attack, Citation, Reject, Verdict,
)
from audit.pipeline import FEATURE_STATUS, RISK_ELIGIBLE  # noqa: E402

VALID_KINDS = {"PRECEDENT", "QUANT", "LITERATURE", "REGULATORY"}

# 관측값과 인용된 수치의 허용 오차. 반올림 표기 차이만 흡수하고 조작은 잡는다.
QUANT_TOLERANCE = 0.05


def norm(s: str | None) -> str:
    """공백 정규화 + 케이스 무시. 따옴표·유니코드 대시 변형을 흡수한다."""
    if not s:
        return ""
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"[‐-―]", "-", s)
    return re.sub(r"\s+", " ", s).strip().casefold()


def _forbidden(element: str) -> str | None:
    e = (element or "").strip().lower()
    for bad, why in FORBIDDEN_GROUNDS.items():
        if bad in e:
            return why
    return None


def verify_citation(
    c: Citation,
    *,
    corpus: dict[str, dict],
    t1_values: dict[str, float | None],
    cutoff: str,
    topic: str,
    v: Verdict,
) -> None:
    """인용 1건을 검증한다. 아래 검사는 전부 결정론적이며 LLM 판단이 개입하지 않는다."""
    v.checked += 1

    if c.kind not in VALID_KINDS:
        v.reject(Reject.UNKNOWN_KIND, f"kind={c.kind!r}")
        return

    if c.kind == "PRECEDENT":
        rec = corpus.get((c.nct_id or "").strip().upper())
        if rec is None:
            v.reject(Reject.PRECEDENT_NOT_FOUND, f"{c.nct_id} 는 코퍼스에 없다")
            return
        if len(norm(c.quote)) < MIN_QUOTE_CHARS:
            v.reject(Reject.QUOTE_TOO_SHORT, f"{c.nct_id}: {c.quote!r}")
            return
        why = (rec.get("labels") or {}).get("why_stopped")
        if norm(c.quote) not in norm(why):
            v.reject(Reject.PRECEDENT_QUOTE_MISMATCH,
                     f"{c.nct_id}: 인용 {c.quote!r} 이 원문에 없다")
            return
        sd = (rec.get("v0") or {}).get("start_date") or ""
        if not sd or sd >= cutoff:
            v.reject(Reject.PRECEDENT_CUTOFF,
                     f"{c.nct_id} 개시 {sd or '미상'} ≥ 컷오프 {cutoff}")
            return
        if topic == "recruitment" and rec.get("_category") != "A":
            v.reject(Reject.PRECEDENT_NOT_A,
                     f"{c.nct_id} 는 {rec.get('_category') or '미분류'} 분류다")
            return

    elif c.kind == "QUANT":
        f = (c.feature or "").strip()
        if f not in FEATURE_STATUS and f not in t1_values:
            v.reject(Reject.QUANT_UNKNOWN_FEATURE, f"{f!r}")
            return
        obs = t1_values.get(f)
        if obs is None or c.value is None:
            v.reject(Reject.QUANT_MISMATCH, f"{f}: 관측값 없음 (인용값 {c.value})")
            return
        if abs(float(c.value) - float(obs)) > max(QUANT_TOLERANCE, abs(obs) * 0.01):
            v.reject(Reject.QUANT_MISMATCH, f"{f}: 인용 {c.value} vs 관측 {obs}")
            return
        if f not in RISK_ELIGIBLE:
            v.reject(Reject.QUANT_NOT_ELIGIBLE,
                     f"{f} 는 {FEATURE_STATUS.get(f, ('?',))[0]} 상태다")
            return

    elif c.kind == "LITERATURE":
        if not (c.pmid or "").strip().isdigit():
            v.reject(Reject.LITERATURE_UNVERIFIED, f"pmid={c.pmid!r}")
            return
        if c.pub_date and cutoff and c.pub_date[:10] > cutoff[:10]:
            v.reject(Reject.LITERATURE_CUTOFF, f"PMID {c.pmid} 발행 {c.pub_date} > {cutoff}")
            return

    elif c.kind == "REGULATORY":
        # 문서 저장소 미구축 — 현재는 항상 미검증으로 기각한다.
        # 근거를 확인할 수 없는 인용을 통과시키는 것보다 기각이 안전하다.
        v.reject(Reject.REGULATORY_NOT_FOUND,
                 f"{c.doc}§{c.section}: 규제 문서 저장소가 아직 없다")


def verify(
    attack: Attack,
    *,
    corpus: dict[str, dict],
    t1_values: dict[str, float | None],
    cutoff: str,
) -> Verdict:
    """공격 1건을 검증한다. 통과하지 못한 공격은 산출물에 도달하지 못한다."""
    v = Verdict(passed=True)

    if not attack.citations:
        return v.reject(Reject.NO_CITATION, f"{attack.target_element}: 인용 0개")

    if why := _forbidden(attack.target_element):
        v.reject(Reject.FORBIDDEN_GROUND, f"{attack.target_element}: {why}")

    for c in attack.citations:
        verify_citation(c, corpus=corpus, t1_values=t1_values,
                        cutoff=cutoff, topic=attack.topic, v=v)
    return v


def index_corpus(rows: list[dict]) -> dict[str, dict]:
    """NCT ID → 레코드. `_category` 에 taxonomy 4분류를 미리 붙여 둔다."""
    from labels import taxonomy
    out = {}
    for r in rows:
        why = (r.get("labels") or {}).get("why_stopped")
        r["_category"] = taxonomy.classify(why)[0] if why else None
        out[r["nct_id"].upper()] = r
    return out


if __name__ == "__main__":
    from audit.referee_selftest import main
    main()
