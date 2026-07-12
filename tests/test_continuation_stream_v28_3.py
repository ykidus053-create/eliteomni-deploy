from __future__ import annotations

import asyncio

from modules.continuation_stream_v28_3 import (
    AsyncTokenBridge,
    ContinuationPolicy,
    OverlapAwareContinuation,
    build_continuation_messages,
    merge_continuation,
    should_continue,
)


def _policy(**changes):
    base = ContinuationPolicy(
        coder_rounds=3,
        hard_rounds=2,
        medium_rounds=1,
        easy_rounds=0,
        continuation_tokens=2000,
        context_tail_chars=100,
        max_total_chars=10000,
        min_novel_chars=8,
        max_overlap_chars=256,
        queue_size=2,
        queue_put_timeout=0.1,
        disconnect_poll=0.01,
    )
    return ContinuationPolicy(**{**base.__dict__, **changes})


def test_length_finish_continues():
    decision, reason = should_continue(
        segment="partial output",
        finish_reason="length",
        skill="coder",
        complexity="hard",
        max_tokens=2000,
        round_index=0,
        total_chars=100,
        policy=_policy(),
    )
    assert decision is True
    assert reason == "finish_length"


def test_normal_stop_ignores_accumulated_length():
    decision, reason = should_continue(
        segment="The continuation completed normally.",
        finish_reason="stop",
        skill="coder",
        complexity="hard",
        max_tokens=2000,
        round_index=1,
        total_chars=9000,
        policy=_policy(),
    )
    assert decision is False
    assert reason == "finish_stop"


def test_unclosed_code_fence_can_continue_after_stop():
    decision, reason = should_continue(
        segment="```python\ndef run():\n    return 1",
        finish_reason="stop",
        skill="coder",
        complexity="hard",
        max_tokens=2000,
        round_index=0,
        total_chars=100,
        policy=_policy(),
    )
    assert decision is True
    assert reason == "stop_with_unclosed_code_fence"


def test_round_limit_is_enforced():
    decision, reason = should_continue(
        segment="partial",
        finish_reason="length",
        skill="coder",
        complexity="hard",
        max_tokens=2000,
        round_index=3,
        total_chars=100,
        policy=_policy(),
    )
    assert decision is False
    assert reason == "round_limit"


def test_merge_removes_repeated_suffix_prefix():
    existing = "alpha\nbeta\ngamma\n"
    continuation = "beta\ngamma\ndelta\n"
    merged, novel, overlap = merge_continuation(
        existing,
        continuation,
        policy=_policy(),
    )
    assert merged == "alpha\nbeta\ngamma\ndelta\n"
    assert novel == "delta\n"
    assert overlap == len("beta\ngamma\n")


def test_overlap_streamer_suppresses_duplicate_prefix():
    joiner = OverlapAwareContinuation(
        "line one\nline two\n",
        policy=_policy(),
    )
    emitted = [
        joiner.feed("line "),
        joiner.feed("two\n"),
        joiner.feed("line three\n"),
        joiner.finish(),
    ]
    assert "".join(emitted) == "line three\n"


def test_continuation_context_is_tail_bounded():
    messages = build_continuation_messages(
        [{"role": "system", "content": "system"}],
        "x" * 500,
        round_number=2,
        policy=_policy(context_tail_chars=100),
    )
    assert messages[-2]["content"].endswith("x" * 100)
    assert len(messages[-2]["content"]) < 200
    assert "round 2" in messages[-1]["content"]


def test_bounded_bridge_delivers_round_sentinel():
    async def scenario():
        loop = asyncio.get_running_loop()
        bridge = AsyncTokenBridge(loop, policy=_policy(queue_size=2))

        import threading

        results = []

        def producer():
            results.append(bridge.put_from_thread("a"))
            results.append(bridge.put_from_thread("b"))
            results.append(bridge.end_round_from_thread())

        thread = threading.Thread(target=producer)
        thread.start()

        assert await bridge.get() == "a"
        assert await bridge.get() == "b"
        assert await bridge.get() is None

        thread.join(timeout=2)
        assert results == [True, True, True]

    asyncio.run(scenario())
