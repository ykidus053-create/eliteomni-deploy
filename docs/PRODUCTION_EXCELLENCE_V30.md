# Production Excellence V30

V30 closes the measurable engineering gap rather than adding another prompt.

## Runtime hardening

- Exact-origin CORS instead of wildcard CORS.
- No default debug or administration secret.
- Constant-time admin-token verification.
- Streaming-safe ASGI middleware.
- Request IDs and security response headers.
- Request-size, concurrency, and per-client rate limits.
- Liveness, readiness, and protected process metrics.
- Structured request logs without prompt or response bodies.
- Runtime SQLite files removed from Git tracking.

## Evidence gates

- A 20-case coding benchmark spanning API, SQL, security, concurrency,
  streaming, reliability, observability, and delivery.
- A static security gate.
- An architecture growth budget.
- A GitHub Actions quality gate.
- A deployment release-gate command.
- A dependency-free concurrent load-test command.

## Required Railway configuration

```env
ELITE_ADMIN_TOKEN=<random value of at least 24 characters>
ELITE_ALLOWED_ORIGINS=https://your-frontend.example
ELITE_ALLOWED_HOSTS=your-service.example
ELITE_REQUIRED_ENV=CEREBRAS_API_KEY

ELITE_MAX_REQUEST_BYTES=10485760
ELITE_MAX_CONCURRENT_REQUESTS=32
ELITE_CONCURRENCY_WAIT_SECONDS=2
ELITE_REQUESTS_PER_MINUTE=120
ELITE_RATE_LIMIT_BURST=30
ELITE_ENABLE_INTERNAL_METRICS=1
ELITE_TRUST_PROXY_HEADERS=1
ELITE_ALLOW_ADMIN_QUERY_TOKEN=0
```

## Release commands

```bash
python3 scripts/release_gate_v30.py
python3 scripts/release_gate_v30.py \
  --live-base-url https://your-service.example

python3 scripts/load_test_v30.py \
  --base-url https://your-service.example \
  --requests 50 \
  --concurrency 10
```

A static release-gate pass means the repository is deployable. A 10/10 claim
requires the live 20-case coding benchmark, load test, and deployed health
checks to pass. The gate intentionally refuses to manufacture that evidence.
