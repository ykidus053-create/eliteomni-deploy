# EliteOmni Streaming Runtime V23

V23 improves the existing `/stream` protocol without converting it to SSE
and without changing the browser-visible response text.

## Repairs

- Removes application-supplied `Transfer-Encoding` and `Content-Length`.
  Uvicorn/Railway own response framing.
- Adds non-cacheable, no-transform headers so Railway's edge does not buffer
  or rewrite the token stream.
- Preserves EliteOmni metadata and NUL-delimited thinking/tool control frames.
- Flushes the first chunks immediately for low time-to-first-token.
- Coalesces later tiny provider fragments into smoother browser-sized chunks.
- Adds `/stream/status` metrics.
- Repairs the missing `delta` assignment in the legacy Groq SSE parser.

## Configuration

```env
ELITE_STREAM_SMOOTHING=1
ELITE_STREAM_TARGET_BYTES=48
ELITE_STREAM_MAX_BYTES=192
ELITE_STREAM_MAX_DELAY_MS=70
ELITE_STREAM_IMMEDIATE_CHUNKS=3
ELITE_STREAM_FLUSH_NEWLINE=1
ELITE_STREAM_FLUSH_PUNCTUATION=1
ELITE_STREAM_PATHS=/stream
```

Lower target/delay values feel more token-by-token. Higher values reduce DOM
work and look smoother on fast providers.

## Status

After deployment:

```text
GET /stream/status
```

The response reports active settings, request counts, chunk coalescing ratio,
last time-to-first-byte, and last stream duration.
