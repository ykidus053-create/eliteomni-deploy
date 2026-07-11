"""CLI wrapper for EliteOmni's generated-artifact verifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.quality_kernel import (
    analyze_request,
    audit_answer,
    verify_python_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("response_file", type=Path)
    parser.add_argument(
        "--request",
        default="Build a complete production-ready Python implementation.",
    )
    args = parser.parse_args()

    response = args.response_file.read_text(encoding="utf-8")
    profile = analyze_request(args.request)
    audit = audit_answer(args.request, response, profile)
    payload = {
        "approved": audit.approved,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "severity": issue.severity,
            }
            for issue in audit.issues
        ],
        "artifact": (
            dataclasses.asdict(audit.artifact)
            if audit.artifact is not None
            else None
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0 if audit.approved else 1


if __name__ == "__main__":
    import dataclasses

    raise SystemExit(main())
