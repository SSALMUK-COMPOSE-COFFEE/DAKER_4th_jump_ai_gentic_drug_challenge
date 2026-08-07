"""중단군 집중 수집 — 모집실패(A분류) 표본을 늘린다.

1차 수집(471건)에서 완주군은 200건으로 충분했지만 모집실패군이 28건뿐이었다.
참조 분포와 층화 분석이 이 표본에 걸려 있으므로 TERMINATED/WITHDRAWN 만 더 긁는다.
COMPLETED 는 추가하지 않는다.

하베스터는 기존 jsonl 의 nct_id 를 읽어 중복을 건너뛰므로 그대로 append 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harvest.ctgov import harvest  # noqa: E402

# 1차의 5개 + 모집이 어려운 것으로 알려진 영역을 추가한다.
# 희귀질환·소아·중증 영역일수록 모집 실패가 많아 A분류 표본이 잘 나온다.
BUCKETS = [
    # 1차와 동일 (더 깊이 파기)
    "non-small cell lung cancer",
    "atopic dermatitis",
    "type 2 diabetes",
    "rheumatoid arthritis",
    "alzheimer disease",
    # 신규 — 모집 난항이 흔한 영역
    "pancreatic cancer",
    "glioblastoma",
    "amyotrophic lateral sclerosis",
    "idiopathic pulmonary fibrosis",
    "systemic lupus erythematosus",
    "sickle cell disease",
    "cystic fibrosis",
    "acute myeloid leukemia",
    "ovarian cancer",
    "heart failure",
]

if __name__ == "__main__":
    harvest(
        buckets=BUCKETS,
        statuses=["TERMINATED", "WITHDRAWN"],
        per_cell=60,
        out_path=Path("data/raw/ctgov_phase2.jsonl"),
    )
