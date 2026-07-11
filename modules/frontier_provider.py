"""Optional OpenAI-compatible frontier-provider adapter.

No vendor is hard-coded. Configure an endpoint, model, and API key through
environment variables. The adapter is disabled when configuration is absent.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence


@dataclasses.dataclass(frozen=True)
class FrontierProviderConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    context_window: int
    max_output_tokens: int

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


def load_frontier_config() -> FrontierProviderConfig:
    def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, min(value, maximum))

    return FrontierProviderConfig(
        base_url=os.getenv("ELITE_FRONTIER_BASE_URL", "").strip(),
        api_key=os.getenv("ELITE_FRONTIER_API_KEY", "").strip(),
        model=os.getenv("ELITE_FRONTIER_MODEL", "").strip(),
        timeout_seconds=_integer(
            "ELITE_FRONTIER_TIMEOUT_SECONDS", 120, 10, 600
        ),
        context_window=_integer(
            "ELITE_FRONTIER_CONTEXT_WINDOW", 128000, 8192, 2000000
        ),
        max_output_tokens=_integer(
            "ELITE_FRONTIER_MAX_OUTPUT_TOKENS", 6000, 512, 64000
        ),
    )


def frontier_enabled() -> bool:
    return load_frontier_config().enabled


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(content)


def estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    # Conservative for code, JSON, and prose.
    chars = sum(
        len(_content_text(message.get("content", ""))) + 24
        for message in messages
    )
    return max(1, (chars + 2) // 3)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit < 160:
        return text[:limit]
    head = int(limit * 0.64)
    marker = "\n...[context compacted by Frontier V20]...\n"
    tail = max(0, limit - head - len(marker))
    return text[:head] + marker + text[-tail:]


def fit_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    context_window: int,
    output_tokens: int,
) -> list[dict[str, Any]]:
    normalized = [
        {
            **dict(message),
            "role": str(message.get("role", "user") or "user"),
            "content": _content_text(message.get("content", "")),
        }
        for message in messages
    ]
    if not normalized:
        return [{"role": "user", "content": ""}]

    safety = max(512, min(2048, context_window // 12))
    prompt_tokens = max(2048, context_window - output_tokens - safety)
    budget_chars = prompt_tokens * 3

    system_index = next(
        (
            index
            for index, message in enumerate(normalized)
            if message["role"] == "system"
        ),
        None,
    )
    latest_user = next(
        (
            index
            for index in range(len(normalized) - 1, -1, -1)
            if normalized[index]["role"] == "user"
        ),
        len(normalized) - 1,
    )

    protected = {latest_user}
    if system_index is not None:
        protected.add(system_index)

    kept: dict[int, dict[str, Any]] = {}
    system_budget = int(budget_chars * 0.42) if system_index is not None else 0
    user_budget = max(1200, int(budget_chars * 0.42))

    if system_index is not None:
        message = normalized[system_index]
        kept[system_index] = {
            **message,
            "content": _clip(message["content"], system_budget),
        }

    message = normalized[latest_user]
    kept[latest_user] = {
        **message,
        "content": _clip(message["content"], user_budget),
    }

    used = sum(len(message["content"]) for message in kept.values())
    remaining = max(0, budget_chars - used)

    for index in range(len(normalized) - 1, -1, -1):
        if index in protected or remaining < 240:
            continue
        message = normalized[index]
        clipped = _clip(message["content"], min(5000, remaining))
        if not clipped:
            continue
        kept[index] = {**message, "content": clipped}
        remaining -= len(clipped)

    result = [kept[index] for index in sorted(kept)]
    while estimate_tokens(result) > prompt_tokens and len(result) > 2:
        removable = next(
            (
                index
                for index, message in enumerate(result)
                if message["role"] != "system"
                and index != len(result) - 1
            ),
            None,
        )
        if removable is None:
            break
        result.pop(removable)

    return result


def _endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def frontier_generate(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int | None = None,
    temperature: float = 0.1,
) -> str:
    config = load_frontier_config()
    if not config.enabled:
        raise RuntimeError(
            "Frontier provider is not configured. Set "
            "ELITE_FRONTIER_BASE_URL, ELITE_FRONTIER_API_KEY, and "
            "ELITE_FRONTIER_MODEL."
        )

    output_tokens = min(
        max_tokens or config.max_output_tokens,
        config.max_output_tokens,
    )
    fitted = fit_messages(
        messages,
        context_window=config.context_window,
        output_tokens=output_tokens,
    )
    payload = json.dumps(
        {
            "model": config.model,
            "messages": fitted,
            "max_tokens": output_tokens,
            "temperature": max(0.0, min(float(temperature), 1.0)),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _endpoint(config.base_url),
        data=payload,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "EliteOmni-Frontier-V20/1.0",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(
                request,
                timeout=config.timeout_seconds,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                raise RuntimeError("Frontier provider returned no choices.")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            raise RuntimeError("Frontier provider returned empty content.")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1200]
            last_error = RuntimeError(
                f"Frontier provider HTTP {exc.code}: {detail}"
            )
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = exc

        if attempt < 2:
            time.sleep(1.5 * (2**attempt))

    raise RuntimeError(f"Frontier provider failed: {last_error}")
