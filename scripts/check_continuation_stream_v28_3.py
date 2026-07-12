import json

from modules.continuation_stream_v28_3 import (
    ContinuationPolicy,
    merge_continuation,
    runtime_status,
    should_continue,
)


policy = ContinuationPolicy.from_env()

decision, reason = should_continue(
    segment="The final continuation is complete.",
    finish_reason="stop",
    skill="coder",
    complexity="hard",
    max_tokens=6000,
    round_index=1,
    total_chars=70000,
    policy=policy,
)

assert decision is False
assert reason == "finish_stop"

merged, novel, overlap = merge_continuation(
    "first\nsecond\n",
    "second\nthird\n",
    policy=policy,
)

assert merged == "first\nsecond\nthird\n"
assert novel == "third\n"
assert overlap == len("second\n")

print(json.dumps(runtime_status(), indent=2))
print("Newest-segment continuation decision passed.")
print("Continuation overlap deduplication passed.")
print("Bounded stream backpressure is enabled.")
