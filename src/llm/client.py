"""프로바이더 중립 LLM 클라이언트 + 역할별 모델 계층화 + 비용 계량.

## 교체 방법 — `.env` 두 줄만 바꾼다

    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    LLM_API_KEY=...

Gemini · Anthropic · OpenAI 셋 다 **OpenAI Chat Completions 형식**을 지원하므로
전송 계층이 하나면 된다. `LLM_PROVIDER` 프리셋을 쓰면 BASE_URL 도 생략할 수 있다.

    LLM_PROVIDER=gemini      # 또는 anthropic / openai
    LLM_API_KEY=...

## 설계 원칙 — 세 가지

1. **도구 계층(T1~T5)은 LLM을 아예 쓰지 않는다.** 계량 감사·전례 검색·인용 대조·근거 검증은
   전부 계산이다. 따라서 파이프라인 비용은 에이전트(A1·A4·A6) 호출 수에만 비례한다.

2. **역할별로 모델 등급을 나눈다.** 대량 추출은 최저가 모델로, 레드팀 공격과 교정안 생성만
   추론 능력 있는 모델로 보낸다. 이 분리가 곧 제안서 항목 7(API·GPU 소요 규모)의 내용이 된다.

3. **모든 호출의 토큰을 적산한다.** "크레딧 대비 결과물의 질"이 본선 배점 15점이므로,
   비용을 추정이 아니라 실측으로 제시해야 한다.

## 의존성

표준 라이브러리만 쓴다. `.env` 파서도 직접 구현했다(python-dotenv 불필요).

## 환경변수

| 변수 | 설명 |
|---|---|
| `LLM_PROVIDER` | `gemini` / `anthropic` / `openai` 프리셋. BASE_URL·기본 모델을 채운다 |
| `LLM_BASE_URL` | 프리셋을 덮어쓴다. OpenAI 호환 엔드포인트의 루트(`/chat/completions` 앞까지) |
| `LLM_API_KEY` | 인증 키. 없으면 `GEMINI_API_KEY`/`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` 순으로 찾는다 |
| `LLM_API_STYLE` | `openai`(기본) / `anthropic`(네이티브 `/messages`) |
| `LLM_MODEL_EXTRACT` | 추출용 모델. 미설정 시 프로바이더 기본값 |
| `LLM_MODEL_REASON` | 추론용 모델. 미설정 시 프로바이더 기본값 |
| `LLM_PRICE_<MODEL>` | `입력단가,출력단가` (USD/1M). 표에 없는 모델의 비용을 재려면 지정 |
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- .env

def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """최소 `.env` 파서. `KEY=VALUE`, `#` 주석, 따옴표, `export ` 접두사를 처리한다.

    외부 의존성을 늘리지 않기 위해 직접 구현했다. 이미 환경에 있는 값은 기본적으로
    덮어쓰지 않는다 — 셸에서 준 값이 파일보다 우선한다.
    """
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        loaded[k] = v
        if override or k not in os.environ:
            os.environ[k] = v
    return loaded


# --------------------------------------------------------------------- 프로바이더

@dataclass(frozen=True)
class Provider:
    base_url: str
    style: str                      # "openai" | "anthropic"
    extract: str
    reason: str


# BASE_URL 은 `/chat/completions` 를 붙이기 직전까지의 루트다.
# Gemini·Anthropic 모두 OpenAI 호환 계층을 제공하므로 전송 코드는 하나로 충분하다.
PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        style="openai",
        extract="gemini-2.5-flash-lite",
        reason="gemini-3.5-flash-lite",
    ),
    "anthropic": Provider(
        base_url="https://api.anthropic.com/v1",
        style="openai",
        extract="claude-haiku-4-5-20251001",
        reason="claude-sonnet-5",
    ),
    "openai": Provider(
        base_url="https://api.openai.com/v1",
        style="openai",
        extract="gpt-5-mini",
        reason="gpt-5",
    ),
}

KEY_ENV_FALLBACK = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# USD / 1M 토큰. **여기 없는 모델은 '가격미등록'으로 표시되며 비용이 0으로 잡힌다.**
# 추정치를 넣지 않는다 — 틀린 비용을 제안서에 쓰느니 미등록이 낫다.
# 새 모델은 공식 가격 페이지를 확인해 추가하거나 `LLM_PRICE_<MODEL>` 로 주입한다.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.125, 0.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def _price_of(model: str) -> tuple[float, float] | None:
    """`LLM_PRICE_<MODEL>` 환경변수가 표보다 우선한다."""
    slug = re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")
    raw = os.environ.get(f"LLM_PRICE_{slug}")
    if raw:
        try:
            a, b = (float(x) for x in raw.split(","))
            return a, b
        except ValueError:
            pass
    return PRICING.get(model)


# ------------------------------------------------------------------------- 원장

@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, model: str) -> float | None:
        p = _price_of(model)
        if not p:
            return None
        return self.input_tokens / 1e6 * p[0] + self.output_tokens / 1e6 * p[1]


@dataclass
class Ledger:
    """모델별 사용량 원장. 파이프라인 끝에서 항목 7 표를 그대로 뽑는다."""

    per_model: dict[str, Usage] = field(default_factory=dict)

    def add(self, model: str, in_tok: int, out_tok: int) -> None:
        u = self.per_model.setdefault(model, Usage())
        u.calls += 1
        u.input_tokens += in_tok
        u.output_tokens += out_tok

    @property
    def calls(self) -> int:
        return sum(u.calls for u in self.per_model.values())

    def total_cost_usd(self) -> float:
        return sum(u.cost_usd(m) or 0.0 for m, u in self.per_model.items())

    def unpriced(self) -> list[str]:
        return [m for m in self.per_model if _price_of(m) is None]

    def report(self) -> str:
        lines = [f"  {'모델':<34}{'호출':>7}{'입력토큰':>12}{'출력토큰':>12}{'USD':>12}"]
        for m, u in sorted(self.per_model.items()):
            c = u.cost_usd(m)
            lines.append(
                f"  {m:<34}{u.calls:>7,}{u.input_tokens:>12,}{u.output_tokens:>12,}"
                f"{('$%.4f' % c) if c is not None else '가격미등록':>12}"
            )
        lines.append(f"  {'합계':<34}{self.calls:>7,}{'':>12}{'':>12}"
                     f"{'$%.4f' % self.total_cost_usd():>12}")
        if self.unpriced():
            lines.append(f"  ⚠ 가격 미등록: {', '.join(self.unpriced())} "
                         f"— LLM_PRICE_<MODEL> 로 주입하면 비용이 집계된다")
        return "\n".join(lines)


class LLMError(RuntimeError):
    pass


# ------------------------------------------------------------------------ 전송

def _post(url: str, payload: dict, headers: dict, retries: int = 4) -> dict:
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            # 4xx 는 재시도해도 낫지 않는다 (429 제외)
            if 400 <= e.code < 500 and e.code != 429:
                raise LLMError(f"HTTP {e.code}: {detail}") from e
            last = LLMError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise LLMError(f"재시도 {retries}회 실패: {last}")


class LLM:
    def __init__(self, ledger: Ledger | None = None, *, dotenv: str | Path = ".env") -> None:
        load_dotenv(dotenv)
        self.ledger = ledger if ledger is not None else Ledger()

        name = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
        self.provider_name = name if name in PROVIDERS else ""
        preset = PROVIDERS.get(self.provider_name)

        self.base_url = (os.environ.get("LLM_BASE_URL")
                         or (preset.base_url if preset else "")).rstrip("/")
        self.style = (os.environ.get("LLM_API_STYLE")
                      or (preset.style if preset else "openai")).lower()
        self.api_key = os.environ.get("LLM_API_KEY") or self._fallback_key()
        self._preset = preset

    def _fallback_key(self) -> str | None:
        if self.provider_name:
            return os.environ.get(KEY_ENV_FALLBACK[self.provider_name])
        for env in KEY_ENV_FALLBACK.values():
            if v := os.environ.get(env):
                return v
        return None

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def model_for(self, role: str) -> str:
        if v := os.environ.get(f"LLM_MODEL_{role.upper()}"):
            return v
        if self._preset:
            return getattr(self._preset, role)
        raise LLMError(
            f"모델을 결정할 수 없습니다. LLM_MODEL_{role.upper()} 또는 LLM_PROVIDER 를 설정하세요.")

    def describe(self) -> str:
        return (f"provider={self.provider_name or '(직접지정)'} style={self.style} "
                f"base_url={self.base_url or '(없음)'} key={'설정됨' if self.api_key else '없음'}")

    # ----------------------------------------------------------------- 공개 API

    def complete(
        self,
        prompt: str,
        *,
        role: str = "extract",
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        if not self.configured:
            raise LLMError(
                "LLM 이 설정되지 않았습니다. .env 에 LLM_PROVIDER(또는 LLM_BASE_URL)와 "
                "LLM_API_KEY 를 지정하세요. 예시는 .env.example 참조.")
        model = model or self.model_for(role)
        if self.style == "anthropic":
            return self._anthropic_native(model, prompt, system, max_tokens, temperature)
        return self._openai_compatible(model, prompt, system, max_tokens, temperature)

    def json_complete(self, prompt: str, **kw) -> dict | list:
        """JSON 응답 전용. 코드펜스로 감싸 오는 경우가 흔해서 벗겨낸다."""
        raw = self.complete(prompt, **kw)
        text = raw.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
        if fence:
            text = fence.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            m = re.search(r"[\[{].*[\]}]", text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"JSON 파싱 실패: {e}\n응답 앞 300자: {text[:300]}") from e

    # ------------------------------------------------------------- 전송 구현

    def _openai_compatible(self, model, prompt, system, max_tokens, temperature) -> str:
        """OpenAI Chat Completions 형식. Gemini·Anthropic·OpenAI 공통 경로."""
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        payload: dict = {"model": model, "messages": messages, "temperature": temperature,
                         "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"

        try:
            data = _post(url, payload, headers)
        except LLMError as e:
            # 일부 최신 모델은 max_tokens 대신 max_completion_tokens 를 요구한다.
            if "max_completion_tokens" not in str(e):
                raise
            payload.pop("max_tokens")
            payload["max_completion_tokens"] = max_tokens
            data = _post(url, payload, headers)

        u = data.get("usage") or {}
        self.ledger.add(model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"응답에 choices 가 없습니다: {json.dumps(data)[:300]}")
        return (choices[0].get("message") or {}).get("content") or ""

    def _anthropic_native(self, model, prompt, system, max_tokens, temperature) -> str:
        """Anthropic 네이티브 `/messages`. LLM_API_STYLE=anthropic 일 때만 쓴다."""
        payload: dict = {"model": model, "max_tokens": max_tokens,
                         "temperature": temperature,
                         "messages": [{"role": "user", "content": prompt}]}
        if system:
            payload["system"] = system
        data = _post(f"{self.base_url}/messages", payload, {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        })
        u = data.get("usage") or {}
        self.ledger.add(model, u.get("input_tokens", 0), u.get("output_tokens", 0))
        return "".join(b.get("text", "") for b in data.get("content", []))


def main() -> None:
    llm = LLM()
    print(f"설정: {llm.describe()}\n")
    print("프로바이더 프리셋:")
    for name, p in PROVIDERS.items():
        print(f"  {name:<10} {p.base_url}")
        for role in ("extract", "reason"):
            m = getattr(p, role)
            pr = _price_of(m)
            price = f"${pr[0]}/${pr[1]} per 1M" if pr else "가격미등록"
            print(f"    {role:<8} → {m:<32} {price}")

    if not llm.configured:
        print("\n⚠ 아직 설정되지 않았습니다. `cp .env.example .env` 후 키를 넣으세요.")
        return

    print(f"\n실호출 검사 ({llm.model_for('extract')}) …")
    try:
        out = llm.complete("Reply with exactly: OK", role="extract", max_tokens=16)
        print(f"  응답: {out.strip()[:60]}")
        print(llm.ledger.report())
    except LLMError as e:
        print(f"  실패: {e}")


if __name__ == "__main__":
    main()
