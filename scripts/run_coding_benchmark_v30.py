#!/usr/bin/env python3
"""Run EliteOmni's coding benchmark against saved responses or a live endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from modules.coding_benchmark_v30 import (
    evaluate_response,
    load_cases,
    summarize,
)


def call_endpoint(
    base_url: str,
    endpoint: str,
    prompt: str,
    *,
    token: str,
    timeout: float,
    message_field: str,
) -> str:
    payload = json.dumps(
        {
            message_field: prompt,
            "history": [],
            "skill": "coder",
            "complexity": "hard",
        }
    ).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "accept": "text/plain",
    }
    if token:
        headers["authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="benchmarks/coding_v30.jsonl",
    )
    parser.add_argument("--fixtures")
    parser.add_argument("--base-url")
    parser.add_argument("--endpoint", default="/stream")
    parser.add_argument("--token", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--message-field", default="message")
    parser.add_argument(
        "--output",
        default="artifacts/coding_benchmark_v30.json",
    )
    args = parser.parse_args()

    if bool(args.fixtures) == bool(args.base_url):
        parser.error(
            "provide exactly one of --fixtures or --base-url"
        )

    cases = load_cases(args.suite)
    responses: dict[str, str] = {}

    if args.fixtures:
        responses = json.loads(
            Path(args.fixtures).read_text(encoding="utf-8")
        )

    evaluations = []
    for case in cases:
        if args.base_url:
            try:
                response = call_endpoint(
                    args.base_url,
                    args.endpoint,
                    case.prompt,
                    token=args.token,
                    timeout=args.timeout,
                    message_field=args.message_field,
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
            ) as exc:
                response = f"BENCHMARK REQUEST FAILED: {exc}"
        else:
            response = str(responses.get(case.case_id, ""))

        evaluation = evaluate_response(case, response)
        evaluations.append(evaluation)
        print(
            f"{case.case_id}: score={evaluation.score} "
            f"passed={evaluation.passed}"
        )

    report = summarize(evaluations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["release_approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
