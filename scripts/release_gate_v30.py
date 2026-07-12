#!/usr/bin/env python3
"""Single evidence-oriented release command for EliteOmni V30."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_step(name: str, command: list[str]) -> dict:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "returncode": process.returncode,
        "duration_seconds": round(
            time.perf_counter() - started,
            2,
        ),
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
        "passed": process.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--live-base-url")
    parser.add_argument("--live-token", default="")
    parser.add_argument(
        "--output",
        default="artifacts/release_gate_v30.json",
    )
    args = parser.parse_args()

    steps = [
        run_step(
            "compile",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "app.py",
                "modules",
                "scripts",
            ],
        ),
        run_step(
            "security",
            [sys.executable, "scripts/security_gate_v30.py"],
        ),
        run_step(
            "architecture",
            [sys.executable, "scripts/architecture_gate_v30.py"],
        ),
        run_step(
            "generated_code_quality",
            [
                sys.executable,
                "scripts/check_secure_code_gate_v31.py",
            ],
        ),
    ]

    if not args.skip_tests:
        steps.append(
            run_step(
                "pytest",
                [sys.executable, "-m", "pytest", "-q"],
            )
        )

    if args.live_base_url:
        command = [
            sys.executable,
            "scripts/run_coding_benchmark_v30.py",
            "--base-url",
            args.live_base_url,
        ]
        if args.live_token:
            command.extend(["--token", args.live_token])
        steps.append(run_step("live_coding_benchmark", command))

    report = {
        "gate": "EliteOmni Release Gate V30",
        "passed": all(step["passed"] for step in steps),
        "live_benchmark_required_for_10_of_10": True,
        "live_benchmark_executed": bool(args.live_base_url),
        "steps": steps,
    }
    if not args.live_base_url:
        report["passed_for_deployment"] = report["passed"]
        report["verified_10_of_10"] = False
        report["reason"] = (
            "Static and test gates passed, but a live 20-case coding "
            "benchmark is required before claiming 10/10."
        )
    else:
        report["passed_for_deployment"] = report["passed"]
        report["verified_10_of_10"] = report["passed"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
