from modules.platform_excellence_v30 import (
    RuntimeMetrics,
    RuntimeSettings,
    TokenBucketLimiter,
    readiness_report,
)


def test_token_bucket_limits_burst():
    limiter = TokenBucketLimiter(per_minute=60, burst=2)
    assert limiter.allow("client", now=100.0)
    assert limiter.allow("client", now=100.0)
    assert not limiter.allow("client", now=100.0)
    assert limiter.allow("client", now=101.0)


def test_metrics_capture_latency_and_status():
    metrics = RuntimeMetrics()
    metrics.begin()
    metrics.finish(200, 12.5)
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["requests_total"] == 1
    assert snapshot["counters"]["status_200"] == 1
    assert snapshot["latency_ms"]["samples"] == 1


def test_readiness_reports_missing_environment(monkeypatch):
    monkeypatch.delenv("REQUIRED_FOR_TEST", raising=False)
    settings = RuntimeSettings(
        allowed_origins=(),
        allowed_hosts=(),
        max_request_bytes=1024,
        max_concurrency=1,
        concurrency_wait_seconds=1.0,
        requests_per_minute=10,
        rate_limit_burst=2,
        required_env=("REQUIRED_FOR_TEST",),
        enable_metrics=True,
        trust_proxy_headers=False,
    )
    report = readiness_report(settings)
    assert report["ok"] is False
    assert report["missing_required_environment"] == [
        "REQUIRED_FOR_TEST"
    ]
