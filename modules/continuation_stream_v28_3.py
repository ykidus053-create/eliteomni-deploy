"""Bounded token streaming and intelligent continuation for EliteOmni V28.3."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Sequence


_STOP_REASONS = {"stop", "end_turn", "eos", "completed", "complete"}
_LENGTH_REASONS = {"length", "max_tokens", "max_completion_tokens"}
_ABORT_REASONS = {
    "error",
    "cancelled",
    "canceled",
    "content_filter",
    "tool_calls",
}


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (AttributeError, TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def _env_float(
    name: str,
    default: float,
    low: float,
    high: float,
) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except (AttributeError, TypeError, ValueError):
        value = default
    return max(low, min(value, high))


@dataclass(frozen=True)
class ContinuationPolicy:
    coder_rounds: int = 3
    hard_rounds: int = 2
    medium_rounds: int = 1
    easy_rounds: int = 0
    continuation_tokens: int = 6000
    context_tail_chars: int = 24000
    max_total_chars: int = 120000
    min_novel_chars: int = 24
    max_overlap_chars: int = 4096
    queue_size: int = 128
    queue_put_timeout: float = 0.5
    disconnect_poll: float = 0.25

    @classmethod
    def from_env(cls) -> "ContinuationPolicy":
        return cls(
            coder_rounds=_env_int(
                "ELITE_CONTINUATION_CODER_ROUNDS", 3, 0, 6
            ),
            hard_rounds=_env_int(
                "ELITE_CONTINUATION_HARD_ROUNDS", 2, 0, 5
            ),
            medium_rounds=_env_int(
                "ELITE_CONTINUATION_MEDIUM_ROUNDS", 1, 0, 3
            ),
            easy_rounds=_env_int(
                "ELITE_CONTINUATION_EASY_ROUNDS", 0, 0, 1
            ),
            continuation_tokens=_env_int(
                "ELITE_CONTINUATION_MAX_TOKENS", 6000, 512, 12000
            ),
            context_tail_chars=_env_int(
                "ELITE_CONTINUATION_CONTEXT_CHARS", 24000, 4000, 60000
            ),
            max_total_chars=_env_int(
                "ELITE_CONTINUATION_MAX_TOTAL_CHARS",
                120000,
                12000,
                300000,
            ),
            min_novel_chars=_env_int(
                "ELITE_CONTINUATION_MIN_NOVEL_CHARS", 24, 1, 512
            ),
            max_overlap_chars=_env_int(
                "ELITE_CONTINUATION_MAX_OVERLAP_CHARS",
                4096,
                128,
                16000,
            ),
            queue_size=_env_int(
                "ELITE_STREAM_QUEUE_SIZE", 128, 8, 2048
            ),
            queue_put_timeout=_env_float(
                "ELITE_STREAM_QUEUE_PUT_TIMEOUT", 0.5, 0.05, 5.0
            ),
            disconnect_poll=_env_float(
                "ELITE_STREAM_DISCONNECT_POLL_SECONDS",
                0.25,
                0.05,
                2.0,
            ),
        )

    def rounds_for(self, skill: str, complexity: str) -> int:
        if (skill or "").strip().lower() == "coder":
            return self.coder_rounds
        normalized = (complexity or "").strip().lower()
        if normalized == "hard":
            return self.hard_rounds
        if normalized == "medium":
            return self.medium_rounds
        return self.easy_rounds


def _unclosed_code_fence(text: str) -> bool:
    return (text or "").count("```") % 2 == 1


def _looks_cut_mid_structure(text: str) -> bool:
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    if _unclosed_code_fence(stripped):
        return True
    return stripped.endswith(
        ("\\", "->", "=>", "=", "+", "-", "*", "/", "&&", "||", ",", "(", "[", "{")
    )


def _estimated_tokens(text: str) -> int:
    value = text or ""
    return max(len(value) // 4, int(len(value.split()) * 1.35))


def should_continue(
    *,
    segment: str,
    finish_reason: str | None,
    skill: str,
    complexity: str,
    max_tokens: int,
    round_index: int,
    total_chars: int,
    policy: ContinuationPolicy,
) -> tuple[bool, str]:
    """Decide from the newest segment rather than the accumulated answer."""
    if round_index >= policy.rounds_for(skill, complexity):
        return False, "round_limit"
    if total_chars >= policy.max_total_chars:
        return False, "total_character_limit"
    if not (segment or "").strip():
        return False, "empty_segment"

    reason = (finish_reason or "").strip().lower()

    if reason in _ABORT_REASONS:
        return False, f"finish_{reason}"
    if reason in _LENGTH_REASONS:
        return True, f"finish_{reason}"
    if reason in _STOP_REASONS:
        if _unclosed_code_fence(segment):
            return True, "stop_with_unclosed_code_fence"
        return False, f"finish_{reason}"

    if _unclosed_code_fence(segment):
        return True, "missing_finish_unclosed_code_fence"

    if max_tokens > 0 and _estimated_tokens(segment) >= int(max_tokens * 0.88):
        return True, "missing_finish_near_token_cap"

    if (
        (skill or "").strip().lower() == "coder"
        and _looks_cut_mid_structure(segment)
    ):
        return True, "missing_finish_incomplete_structure"

    return False, "segment_complete"


def continuation_prompt(round_number: int) -> str:
    return (
        "Continue exactly after the previous final character. "
        "Do not repeat, summarize, restart, add a new introduction, or reopen "
        "completed sections. Finish only the incomplete code block, file, test, "
        "or explanation. Preserve language, formatting, filenames, indentation, "
        "and API contracts. End normally as soon as the response is complete. "
        f"This is continuation round {round_number}."
    )


def build_continuation_messages(
    base_messages: Sequence[dict[str, Any]],
    accumulated: str,
    *,
    round_number: int,
    policy: ContinuationPolicy,
) -> list[dict[str, Any]]:
    tail = (accumulated or "")[-policy.context_tail_chars :]
    if len(accumulated or "") > len(tail):
        tail = "[Earlier assistant output omitted.]\n" + tail

    messages = [dict(item) for item in base_messages]
    messages.extend(
        [
            {"role": "assistant", "content": tail},
            {
                "role": "user",
                "content": continuation_prompt(round_number),
            },
        ]
    )
    return messages


def longest_suffix_prefix_overlap(
    existing: str,
    continuation: str,
    *,
    max_chars: int = 4096,
    minimum: int = 4,
) -> int:
    if not existing or not continuation:
        return 0

    ceiling = min(len(existing), len(continuation), max_chars)
    for size in range(ceiling, minimum - 1, -1):
        if existing[-size:] == continuation[:size]:
            return size
    return 0


def _remove_continuation_preamble(text: str) -> str:
    value = text or ""
    lowered = value.lower()
    for prefix in (
        "continuing exactly where i left off:",
        "continuing where i left off:",
        "continuing:",
        "here is the continuation:",
    ):
        if lowered.startswith(prefix):
            return value[len(prefix) :].lstrip("\r\n ")
    return value


def merge_continuation(
    existing: str,
    continuation: str,
    *,
    policy: ContinuationPolicy,
) -> tuple[str, str, int]:
    cleaned = _remove_continuation_preamble(continuation)
    overlap = longest_suffix_prefix_overlap(
        existing,
        cleaned,
        max_chars=policy.max_overlap_chars,
    )
    novel = cleaned[overlap:]
    return existing + novel, novel, overlap


def segment_fingerprint(text: str) -> str:
    normalized = " ".join((text or "").split())
    return hashlib.sha256(
        normalized.encode("utf-8", errors="ignore")
    ).hexdigest()[:16]


class OverlapAwareContinuation:
    """Suppress a repeated continuation prefix before it reaches the client."""

    def __init__(
        self,
        existing: str,
        *,
        policy: ContinuationPolicy,
        minimum_overlap: int = 4,
    ) -> None:
        self.existing = existing or ""
        self.policy = policy
        self.minimum_overlap = minimum_overlap
        self.pending = ""
        self.resolved = False
        self.overlap = 0
        self._candidate_sizes: list[int] | None = None

    def _candidates(self) -> list[int]:
        if self._candidate_sizes is not None:
            return self._candidate_sizes
        maximum = min(len(self.existing), self.policy.max_overlap_chars)
        first = self.pending[:1]
        self._candidate_sizes = [
            size
            for size in range(maximum, self.minimum_overlap - 1, -1)
            if not first or self.existing[-size : -size + 1] == first
        ]
        return self._candidate_sizes

    def feed(self, token: str) -> str:
        if not token:
            return ""
        if self.resolved:
            return token

        self.pending += token
        viable: list[int] = []
        completed: list[int] = []

        for size in self._candidates():
            suffix = self.existing[-size:]
            if len(self.pending) <= size:
                if suffix.startswith(self.pending):
                    viable.append(size)
                    if len(self.pending) == size:
                        completed.append(size)
            elif self.pending.startswith(suffix):
                completed.append(size)

        if viable and len(self.pending) < self.policy.max_overlap_chars:
            return ""

        self.overlap = max(completed, default=0)
        output = _remove_continuation_preamble(
            self.pending[self.overlap :]
        )
        self.pending = ""
        self.resolved = True
        return output

    def finish(self) -> str:
        if self.resolved:
            return ""

        cleaned = _remove_continuation_preamble(self.pending)
        self.overlap = longest_suffix_prefix_overlap(
            self.existing,
            cleaned,
            max_chars=self.policy.max_overlap_chars,
            minimum=self.minimum_overlap,
        )
        output = cleaned[self.overlap :]
        self.pending = ""
        self.resolved = True
        return output


_SENTINEL = object()


class AsyncTokenBridge:
    """Bounded thread-to-async queue with backpressure and disconnect stop."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        request: Any = None,
        policy: ContinuationPolicy | None = None,
    ) -> None:
        self.loop = loop
        self.request = request
        self.policy = policy or ContinuationPolicy.from_env()
        self.queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=self.policy.queue_size
        )
        self.stop_event = threading.Event()
        self.tokens_enqueued = 0
        self.rounds_closed = 0

    def put_from_thread(self, token: str) -> bool:
        if self.stop_event.is_set():
            return False

        future = asyncio.run_coroutine_threadsafe(
            self.queue.put(token),
            self.loop,
        )
        while not self.stop_event.is_set():
            try:
                future.result(timeout=self.policy.queue_put_timeout)
                self.tokens_enqueued += 1
                return True
            except TimeoutError:
                continue
            except Exception:
                future.cancel()
                return False

        future.cancel()
        return False

    def end_round_from_thread(self) -> bool:
        # Schedule the sentinel without blocking on event-loop acknowledgement.
        if self.stop_event.is_set():
            return False

        def _enqueue_end() -> None:
            if self.stop_event.is_set():
                return

            try:
                self.queue.put_nowait(_SENTINEL)
            except asyncio.QueueFull:
                task = self.loop.create_task(
                    self.queue.put(_SENTINEL)
                )

                def _consume_result(done: asyncio.Task[Any]) -> None:
                    try:
                        done.result()
                    except BaseException:
                        pass

                task.add_done_callback(_consume_result)

        try:
            self.loop.call_soon_threadsafe(_enqueue_end)
        except RuntimeError:
            return False

        self.rounds_closed += 1
        return True

    async def _disconnected(self) -> bool:
        checker = getattr(self.request, "is_disconnected", None)
        if not callable(checker):
            return False
        try:
            return bool(await checker())
        except Exception:
            return False

    async def get(self) -> str | None:
        while not self.stop_event.is_set():
            if await self._disconnected():
                self.cancel()
                return None
            try:
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.policy.disconnect_poll,
                )
            except asyncio.TimeoutError:
                continue
            if item is _SENTINEL:
                return None
            return str(item)
        return None

    def cancel(self) -> None:
        self.stop_event.set()

    def status(self) -> dict[str, Any]:
        return {
            "queue_size": self.policy.queue_size,
            "queue_depth": self.queue.qsize(),
            "tokens_enqueued": self.tokens_enqueued,
            "rounds_closed": self.rounds_closed,
            "cancelled": self.stop_event.is_set(),
        }


def runtime_status() -> dict[str, Any]:
    return {
        "version": "V28.3",
        "policy": asdict(ContinuationPolicy.from_env()),
        "continuation_decision": "newest_segment_only",
        "overlap_deduplication": True,
        "bounded_backpressure": True,
        "disconnect_cancellation": True,
    }
