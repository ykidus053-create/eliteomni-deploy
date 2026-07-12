# Continuation and Streaming Runtime V28.3

V28.3 fixes runaway auto-continuation and improves browser token streaming.

## Auto-continuation

- Evaluates only the newest provider segment.
- Treats a normal provider `stop` as complete unless a code fence is open.
- Defaults coder continuation to three rounds instead of twelve.
- Caps continuation generation at 6,000 tokens per round.
- Sends only a bounded tail of prior output.
- Removes repeated suffix/prefix overlap before display.
- Stops on insufficient novel content or repeated segments.

## Streaming

- Uses a bounded thread-to-async queue with backpressure.
- Stops provider workers when the browser disconnects.
- Uses 32-byte target chunks, 128-byte maximum chunks, and 40 ms delay.
- Keeps the first four chunks immediate.
- No longer treats every comma as a mandatory flush boundary.

## Railway variables

```env
ELITE_CONTINUATION_CODER_ROUNDS=3
ELITE_CONTINUATION_HARD_ROUNDS=2
ELITE_CONTINUATION_MEDIUM_ROUNDS=1
ELITE_CONTINUATION_MAX_TOKENS=6000
ELITE_CONTINUATION_CONTEXT_CHARS=24000
ELITE_CONTINUATION_MAX_TOTAL_CHARS=120000
ELITE_CONTINUATION_MIN_NOVEL_CHARS=24
ELITE_CONTINUATION_MAX_OVERLAP_CHARS=4096

ELITE_STREAM_QUEUE_SIZE=128
ELITE_STREAM_QUEUE_PUT_TIMEOUT=0.5
ELITE_STREAM_DISCONNECT_POLL_SECONDS=0.25
ELITE_STREAM_TARGET_BYTES=32
ELITE_STREAM_MAX_BYTES=128
ELITE_STREAM_MAX_DELAY_MS=40
ELITE_STREAM_IMMEDIATE_CHUNKS=4
```
