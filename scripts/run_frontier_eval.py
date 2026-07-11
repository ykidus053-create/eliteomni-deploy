#!/usr/bin/env python3
"""Run a small live scorecard against EliteOmni's /chat endpoint."""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def score_response(task: dict[str, Any], response: str) -> dict[str, Any]:
    lowered = response.lower()
    checks = {}
    checks["nonempty"] = len(response.strip()) >= int(
        task.get("min_chars", 80)
    )
    checks["must_include"] = all(
        value.lower() in lowered for value in task.get("must_include", [])
    )
    checks["must_not_include"] = all(
        value.lower() not in lowered
        for value in task.get("must_not_include", [])
    )
    url_count = len(re.findall(r"https?://[^\s)\]>]+", response))
    checks["sources"] = url_count >= int(task.get("min_urls", 0))
    checks["code"] = (
        "```" in response if task.get("expect_code") else True
    )
    checks["tests"] = (
        bool(re.search(r"\btest_|pytest|unittest", response, re.I))
        if task.get("expect_tests")
        else True
    )
    passed = sum(bool(value) for value in checks.values())
    return {
        "score": round(100 * passed / len(checks), 1),
        "checks": checks,
        "url_count": url_count,
    }


def post_chat(base_url: str, prompt: str, timeout: int) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat"
    body = json.dumps({"message": prompt, "history": []}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default="benchmarks/frontier_gap_v20.json",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/frontier_gap_v20_results.json",
    )
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[: args.limit]

    if args.dry_run:
        print(f"Validated {len(tasks)} frontier-gap tasks.")
        return 0

    results = []
    for index, task in enumerate(tasks, start=1):
        started = time.time()
        try:
            payload = post_chat(
                args.base_url,
                task["prompt"],
                args.timeout,
            )
            response = str(payload.get("response", ""))
            scored = score_response(task, response)
            result = {
                "id": task["id"],
                "category": task["category"],
                "latency_seconds": round(time.time() - started, 3),
                "response": response,
                "metadata": {
                    key: value
                    for key, value in payload.items()
                    if key != "response"
                },
                **scored,
            }
        except Exception as exc:
            result = {
                "id": task["id"],
                "category": task["category"],
                "latency_seconds": round(time.time() - started, 3),
                "score": 0.0,
                "error": str(exc),
            }
        results.append(result)
        print(
            f"[{index}/{len(tasks)}] {result['id']}: "
            f"{result['score']}%"
        )

    summary = {
        "task_count": len(results),
        "average_score": round(
            sum(item["score"] for item in results) / max(1, len(results)),
            2,
        ),
        "by_category": {},
        "results": results,
    }
    categories = sorted({item["category"] for item in results})
    for category in categories:
        selected = [
            item["score"]
            for item in results
            if item["category"] == category
        ]
        summary["by_category"][category] = round(
            sum(selected) / max(1, len(selected)),
            2,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            key: value
            for key, value in summary.items()
            if key != "results"
        },
        indent=2,
    ))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
