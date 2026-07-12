#!/usr/bin/env python3
"""Dependency-free concurrent HTTP load test for EliteOmni deployments."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass


@dataclass
class Sample:
    ok: bool
    status: int
    total_ms: float
    bytes_received: int
    error: str = ""


def request_once(
    url: str,
    payload: bytes,
    headers: dict[str, str],
    timeout: float,
) -> Sample:
    started = time.perf_counter()
    try:
        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            body = response.read()
            status = int(response.status)
        return Sample(
            ok=200 <= status < 400,
            status=status,
            total_ms=(time.perf_counter() - started) * 1000,
            bytes_received=len(body),
        )
    except urllib.error.HTTPError as exc:
        return Sample(
            ok=False,
            status=int(exc.code),
            total_ms=(time.perf_counter() - started) * 1000,
            bytes_received=0,
            error=str(exc),
        )
    except Exception as exc:
        return Sample(
            ok=False,
            status=0,
            total_ms=(time.perf_counter() - started) * 1000,
            bytes_received=0,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_load(
    *,
    url: str,
    total: int,
    concurrency: int,
    payload: bytes,
    headers: dict[str, str],
    timeout: float,
) -> list[Sample]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one() -> Sample:
        async with semaphore:
            return await asyncio.to_thread(
                request_once,
                url,
                payload,
                headers,
                timeout,
            )

    return await asyncio.gather(*(one() for _ in range(total)))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int((len(ordered) - 1) * p)),
    )
    return round(ordered[index], 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint", default="/stream")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--max-p95-ms", type=float, default=30000)
    parser.add_argument("--token", default="")
    parser.add_argument(
        "--prompt",
        default="Reply with the word healthy.",
    )
    args = parser.parse_args()

    payload = json.dumps(
        {
            "message": args.prompt,
            "history": [],
            "skill": "general",
            "complexity": "easy",
        }
    ).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "accept": "text/plain",
    }
    if args.token:
        headers["authorization"] = f"Bearer {args.token}"

    samples = asyncio.run(
        run_load(
            url=args.base_url.rstrip("/") + args.endpoint,
            total=args.requests,
            concurrency=args.concurrency,
            payload=payload,
            headers=headers,
            timeout=args.timeout,
        )
    )
    latencies = [sample.total_ms for sample in samples]
    failures = [sample for sample in samples if not sample.ok]
    error_rate = len(failures) / max(1, len(samples))
    report = {
        "requests": len(samples),
        "concurrency": args.concurrency,
        "successes": len(samples) - len(failures),
        "failures": len(failures),
        "error_rate": round(error_rate, 4),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "approved": (
            error_rate <= args.max_error_rate
            and percentile(latencies, 0.95) <= args.max_p95_ms
        ),
        "failure_samples": [
            asdict(sample) for sample in failures[:5]
        ],
    }
    print(json.dumps(report, indent=2))
    return 0 if report["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
