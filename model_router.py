"""Deterministic, configurable model routing for EliteOmni.

The module keeps the original public API while fixing three reliability issues:
1. routing inspects the latest user message instead of stringifying the entire
   prompt payload;
2. fallback candidates are deduplicated, so aliases cannot create loops;
3. long system prompts retain both their opening policy and closing context.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

log = logging.getLogger(__name__)


def _env_model(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


MISTRAL_SMALL = _env_model("ELITE_MODEL_FAST", "cerebras/zai-glm-4.7")
MISTRAL_MEDIUM = _env_model("ELITE_MODEL_GENERAL", "cerebras/zai-glm-4.7")
MISTRAL_LARGE = _env_model("ELITE_MODEL_HARD", "cerebras/zai-glm-4.7")
MAGISTRAL = _env_model("ELITE_MODEL_REASONING", MISTRAL_LARGE)
CODESTRAL = _env_model("ELITE_MODEL_CODER", "cerebras/zai-glm-4.7")
REASONING_EFFORT = os.getenv("ELITE_REASONING_EFFORT", "high").strip().lower()

COMPLEXITY_MAP: dict[str, str] = {
    "easy": MISTRAL_SMALL,
    "medium": MISTRAL_MEDIUM,
    "hard": MISTRAL_LARGE,
    "coder": CODESTRAL,
    "coding": CODESTRAL,
    "reasoning": MAGISTRAL,
}

CODE_SIGNALS = [
    "```",
    "def ",
    "class ",
    "import ",
    "function ",
    "const ",
    "traceback",
    "syntaxerror",
    "typeerror",
]

_LANGUAGE_PATTERN = re.compile(
    r"\b(python|javascript|typescript|java|golang|go|rust|c\+\+|c#|sql|html|css|bash|powershell)\b",
    re.IGNORECASE,
)
_CODE_TASK_PATTERN = re.compile(
    r"\b(write|implement|debug|fix|refactor|optimi[sz]e|compile|test|code|script|function|class|api)\b",
    re.IGNORECASE,
)


def is_cerebras(model: str) -> bool:
    return str(model).startswith("cerebras/")


def cerebras_model_name(model: str) -> str:
    value = str(model)
    return value[len("cerebras/") :] if value.startswith("cerebras/") else value


def _last_user_text(messages_payload: Any) -> str:
    """Return only the latest user content from an OpenAI-style payload."""
    if messages_payload is None:
        return ""
    if isinstance(messages_payload, str):
        return messages_payload
    if isinstance(messages_payload, Mapping):
        content = messages_payload.get("content", "")
        return content if isinstance(content, str) else str(content)
    if isinstance(messages_payload, Sequence) and not isinstance(
        messages_payload, (bytes, bytearray)
    ):
        for message in reversed(messages_payload):
            if isinstance(message, Mapping) and message.get("role") == "user":
                content = message.get("content", "")
                return content if isinstance(content, str) else str(content)
    return str(messages_payload)


def _code_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    if "```" in text:
        score += 3
    if any(signal in lowered for signal in CODE_SIGNALS if signal != "```"):
        score += 2
    if _LANGUAGE_PATTERN.search(text):
        score += 1
    if _CODE_TASK_PATTERN.search(text):
        score += 1
    if re.search(r"\b\w+(error|exception)\b", lowered):
        score += 1
    return score


def select_model(complexity: str, messages_payload: Any = None) -> str:
    """Select a model deterministically from task type and complexity."""
    normalized = (complexity or "medium").strip().lower()
    user_text = _last_user_text(messages_payload)

    if normalized in {"coder", "coding", "easy_code", "code"}:
        return CODESTRAL
    if _code_score(user_text) >= 2:
        return CODESTRAL
    if normalized in {"reasoning", "deliberate", "math"}:
        return MAGISTRAL
    if normalized in {"hard", "high", "complex"}:
        return MISTRAL_LARGE
    if normalized in {"easy", "low", "fast"}:
        return MISTRAL_SMALL
    return MISTRAL_MEDIUM


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def get_token_budget(model_name: str) -> int:
    """Return a conservative output-token budget, configurable by environment."""
    model = str(model_name)
    if model == CODESTRAL:
        return _positive_int_env("ELITE_CODER_MAX_TOKENS", 8_000)
    if model in {MISTRAL_LARGE, MAGISTRAL}:
        return _positive_int_env("ELITE_HARD_MAX_TOKENS", 8_000)
    if model == MISTRAL_SMALL:
        return _positive_int_env("ELITE_FAST_MAX_TOKENS", 4_000)
    return _positive_int_env("ELITE_GENERAL_MAX_TOKENS", 8_000)


def record_outcome(model_name: str, outcome: Any) -> None:
    log.debug("[Telemetry] model=%s outcome=%s", model_name, outcome)


def trim_system(system_prompt: str, max_tokens: int = 4_000) -> str:
    """Trim approximately by characters while retaining prompt head and tail."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    max_chars = max_tokens * 4
    if len(system_prompt) <= max_chars:
        return system_prompt

    marker = "\n\n...[system prompt trimmed]...\n\n"
    usable = max(0, max_chars - len(marker))
    head_size = int(usable * 0.72)
    tail_size = usable - head_size
    return (
        system_prompt[:head_size].rstrip()
        + marker
        + system_prompt[-tail_size:].lstrip()
    )


class CircuitState:
    _lock = threading.Lock()
    _state: dict[str, dict[str, Any]] = {}
    THRESHOLD = _positive_int_env("ELITE_CIRCUIT_FAILURE_THRESHOLD", 3)
    RESET_S = _positive_int_env("ELITE_CIRCUIT_RESET_SECONDS", 60)

    @classmethod
    def record_failure(cls, model: str) -> None:
        now = time.monotonic()
        with cls._lock:
            state = cls._state.setdefault(
                model,
                {"failures": 0, "open": False, "opened_at": 0.0},
            )
            state["failures"] += 1
            if state["failures"] >= cls.THRESHOLD:
                state["open"] = True
                state["opened_at"] = now
                log.warning(
                    "[CircuitBreaker] %s OPEN after %d failures",
                    model,
                    state["failures"],
                )

    @classmethod
    def record_success(cls, model: str) -> None:
        with cls._lock:
            cls._state.pop(model, None)

    @classmethod
    def is_open(cls, model: str) -> bool:
        now = time.monotonic()
        with cls._lock:
            state = cls._state.get(model)
            if not state or not state.get("open"):
                return False
            if now - float(state.get("opened_at", 0.0)) >= cls.RESET_S:
                cls._state.pop(model, None)
                log.info("[CircuitBreaker] %s reset; allowing recovery probe", model)
                return False
            return True

    @classmethod
    def stats(cls) -> dict[str, dict[str, Any]]:
        with cls._lock:
            return {name: dict(state) for name, state in cls._state.items()}


def _unique_models(models: Sequence[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _configured_fallbacks() -> list[str]:
    custom = [
        value.strip()
        for value in os.getenv("ELITE_MODEL_FALLBACKS", "").split(",")
        if value.strip()
    ]
    return _unique_models(
        custom
        + [
            MISTRAL_LARGE,
            MISTRAL_MEDIUM,
            MISTRAL_SMALL,
            CODESTRAL,
            MAGISTRAL,
        ]
    )


_fallback_models = _configured_fallbacks()
FALLBACK_CHAIN: dict[str, str] = (
    {
        model: _fallback_models[min(index + 1, len(_fallback_models) - 1)]
        for index, model in enumerate(_fallback_models)
    }
    if _fallback_models
    else {}
)


def route_with_fallback(model: str) -> str:
    """Return the first healthy, unique configured model candidate."""
    candidates = _unique_models(
        [model, FALLBACK_CHAIN.get(model)] + _configured_fallbacks()
    )
    for candidate in candidates:
        if not CircuitState.is_open(candidate):
            if candidate != model:
                log.warning(
                    "[route_with_fallback] %s unavailable; routing to %s",
                    model,
                    candidate,
                )
            return candidate

    log.error("[route_with_fallback] all model circuits are open")
    return model
