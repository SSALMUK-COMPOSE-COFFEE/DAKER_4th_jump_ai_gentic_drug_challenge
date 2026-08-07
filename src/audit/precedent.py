"""T2 전례 검색 〈도구〉 — 시간 컷오프를 하드 필터로 강제한다.

## 왜 이 도구가 필요한가

A4 레드팀의 공격은 `PRECEDENT` 인용 — "같은 적응증에서 모집 실패로 중단된 과거 시험" — 을
근거로 삼는다. 그런데 대상 프로토콜보다 **나중에 개시된** 시험을 인용하면, 등록 시점에
존재하지 않던 정보를 쓰는 것이다. 후향 백테스트의 블라인드 주장이 여기서 깨진다.

→ **강제 규칙: `전례 개시일 < 대상 개시일` 인 것만 반환한다.** 이 필터는 검색 단계에서
   걸리고, T3(`referee.py`)가 통과한 지적마다 한 번 더 재검증한다. 이중 안전장치다.

## 검색 계층

    1순위  적응증 + phase_stratum + 개입 유형   (가장 엄격)
    2순위  적응증 + phase_stratum
    3순위  적응증

각 층에서 **완주군과 모집실패군을 분리해서** 반환한다. 실패군은 `why_stopped` 원문을 함께
싣고 `taxonomy.classify()` 로 4분류를 붙인다. **A 분류(모집 실패)만 모집 관련 공격의
근거로 허용한다.**

## 알려진 제약 — 전례 공급량 (실측)

대상의 상당수가 인용 가능한 A 분류 실패 전례를 **하나도 갖지 못한다.** 이때는 침묵하지 않고
"인용할 실패 전례가 코퍼스에 없습니다"를 명시한다. 근거 부족을 근거 없음으로 위장하지 않는다.

⚠ **비율이 문서마다 다른 이유** — 모수와 층 정의가 다르기 때문이다. 셋 다 같은 코퍼스에서
재현되며 서로 모순되지 않는다 (2026-07-31 대조 확인):

| 모수 | 검색 층 | A 분류 0건 |
|---|---|---|
| 감사 대상 코호트 705건 (`cohort_of()`) | 2순위 고정 | **220/705 = 31.2%** ← 제안서 인용값 |
| 감사 대상 코호트 705건 | 3순위 폴백 | 154/705 = 21.8% |
| 개시일 파싱 전체 1,446건 | 3순위 폴백 | 226/1,446 = 15.6% ← **이 모듈의 기본 동작** |

이 모듈은 A 분류 전례가 `MIN_PRECEDENTS` 미만이면 층을 완화하므로 실질 3순위이고, 모수도
감사 코호트가 아니라 전체다. **제안서는 보수적인 31.2% 를 인용한다** — 층을 완화하지 않았을
때의 값이므로 상한이고, 한계를 과소보고하지 않는 쪽이다.

## 라이브 모드

백테스트의 정본은 로컬 코퍼스다(재현성). 라이브 모드에서는 CT.gov v2 를 쓰며, 컷오프를
`filter.advanced=AREA[StartDate]RANGE[MIN,컷오프]` 로 **서버에 위임**한다 —
미래 전례가 응답에 애초에 들어오지 않는다. 자세한 근거는 agent-architecture.md §13.3.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.benchmarks import phase_stratum  # noqa: E402
from labels import taxonomy  # noqa: E402

RAW = Path("data/raw/ctgov_phase2.jsonl")

MIN_PRECEDENTS = 5   # 이 미만이면 A1 Planner 가 전례 기반 공격을 기각한다


@dataclass
class Precedent:
    nct_id: str
    indication: str
    phase: str
    start_date: str
    status: str
    why_stopped: str | None
    category: str | None          # taxonomy 4분류 (A/B/C/D)
    category_name: str | None
    planned_enrollment: int | None
    actual_enrollment: int | None


@dataclass
class PrecedentSet:
    tier: str                     # 실제로 사용된 검색 층
    tier_rank: int
    completed: list[Precedent] = field(default_factory=list)
    recruit_failed: list[Precedent] = field(default_factory=list)
    other_failed: list[Precedent] = field(default_factory=list)
    cutoff: str = ""
    note: str = ""

    @property
    def citable_a(self) -> list[Precedent]:
        """모집 관련 공격의 근거로 쓸 수 있는 전례 (A 분류 실패)."""
        return self.recruit_failed

    def supply_ok(self) -> bool:
        return len(self.citable_a) > 0

    def scarcity_note(self) -> str:
        if self.citable_a:
            return ""
        return ("이 프로토콜에는 시간 컷오프 이전에 개시된 A 분류(모집 실패) 전례가 "
                "코퍼스에 없습니다. 전례 기반 모집 공격을 생성하지 않으며, "
                "이는 '위험이 없다'는 뜻이 아니라 '인용할 근거가 없다'는 뜻입니다.")


def _intervention_types(v0: dict) -> set[str]:
    out = set()
    for iv in v0.get("interventions") or []:
        t = (iv or {}).get("type") if isinstance(iv, dict) else None
        if t:
            out.add(str(t).upper())
    return out


def load_corpus(path: Path = RAW) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _to_precedent(r: dict) -> Precedent | None:
    v0, lab = r.get("v0") or {}, r.get("labels") or {}
    sd = v0.get("start_date")
    if not sd:
        return None
    why = lab.get("why_stopped")
    cat, cname, _ = taxonomy.classify(why) if why else (None, None, None)
    return Precedent(
        nct_id=r["nct_id"], indication=r.get("condition_bucket", ""),
        phase=phase_stratum(v0), start_date=sd,
        status=lab.get("final_status", ""), why_stopped=why,
        category=cat, category_name=cname,
        planned_enrollment=v0.get("planned_enrollment"),
        actual_enrollment=lab.get("actual_enrollment"),
    )


def search(
    corpus: list[dict],
    *,
    indication: str,
    v0: dict,
    cutoff: str,
    exclude_nct: str | None = None,
) -> PrecedentSet:
    """시간 컷오프 이전에 개시된 전례만 계층적으로 검색한다.

    `cutoff` 는 대상 프로토콜의 개시일이다. **경계는 배타적(`<`)** 이다 — 같은 날 개시된
    시험은 등록 시점에 결과가 알려져 있지 않으므로 제외한다.
    """
    target_phase = phase_stratum(v0)
    target_iv = _intervention_types(v0)

    pool: list[tuple[Precedent, dict]] = []
    for r in corpus:
        if exclude_nct and r["nct_id"] == exclude_nct:
            continue
        if r.get("condition_bucket") != indication:
            continue
        p = _to_precedent(r)
        if p is None or p.start_date >= cutoff:   # ← 하드 컷오프
            continue
        pool.append((p, r.get("v0") or {}))

    tiers = [
        ("적응증+Phase+개입유형", 1,
         lambda p, v: p.phase == target_phase and bool(target_iv & _intervention_types(v))),
        ("적응증+Phase", 2, lambda p, v: p.phase == target_phase),
        ("적응증", 3, lambda p, v: True),
    ]

    chosen: PrecedentSet | None = None
    for name, rank, pred in tiers:
        sel = [p for p, v in pool if pred(p, v)]
        ps = PrecedentSet(tier=name, tier_rank=rank, cutoff=cutoff)
        for p in sel:
            if p.status == "COMPLETED":
                ps.completed.append(p)
            elif p.category == "A":
                ps.recruit_failed.append(p)
            elif p.status in ("TERMINATED", "WITHDRAWN"):
                ps.other_failed.append(p)
        chosen = ps
        # A 분류 전례가 충분히 나오면 그 층에서 멈춘다. 아니면 다음 층으로 완화한다.
        if len(ps.citable_a) >= MIN_PRECEDENTS:
            break

    ps = chosen or PrecedentSet(tier="없음", tier_rank=0, cutoff=cutoff)
    ps.note = ps.scarcity_note()
    return ps


def main() -> None:
    corpus = load_corpus()
    print(f"코퍼스 {len(corpus)}건 로드\n")
    shown = 0
    for r in corpus:
        v0 = r.get("v0") or {}
        if not v0.get("start_date"):
            continue
        ps = search(corpus, indication=r["condition_bucket"], v0=v0,
                    cutoff=v0["start_date"], exclude_nct=r["nct_id"])
        print(f"  {r['nct_id']}  {r['condition_bucket'][:28]:<30} 개시 {v0['start_date']}")
        print(f"    층={ps.tier}  완주 {len(ps.completed)} / A분류실패 {len(ps.citable_a)} / "
              f"기타실패 {len(ps.other_failed)}")
        for p in ps.citable_a[:2]:
            print(f"      · {p.nct_id} ({p.start_date}) \"{(p.why_stopped or '')[:60]}\"")
        if ps.note:
            print(f"      ⚠ {ps.note[:78]}")
        shown += 1
        if shown >= 5:
            break

    # 공급량 요약 — precedent_supply.py 의 31.2% 와 대조 가능해야 한다.
    n = miss = 0
    for r in corpus:
        v0 = r.get("v0") or {}
        if not v0.get("start_date"):
            continue
        ps = search(corpus, indication=r["condition_bucket"], v0=v0,
                    cutoff=v0["start_date"], exclude_nct=r["nct_id"])
        n += 1
        if not ps.supply_ok():
            miss += 1
    print(f"\n  전체 {n}건 중 A분류 전례 0건 = {miss}건 ({miss/n*100:.1f}%)  [층 완화 적용]")
    print("  → 이 비율만큼은 전례 기반 공격을 생성하지 않고 '근거 없음'을 명시한다.")
    print("  ※ 제안서 인용값 31.2% 는 감사 코호트 705건·2순위 층 고정 기준이다 (모듈 docstring 참조).")
    print("     세 값은 모수·층 정의 차이이며 서로 모순되지 않는다.")


if __name__ == "__main__":
    main()
