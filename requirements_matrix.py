"""Evidence-based requirements matrix for generated implementation claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from production_guard import audit_production_response


def enforce_requirements(
    response: str,
    request: str = "Provide a production-ready implementation",
) -> dict[str, Any]:
    """Return a machine-readable evidence report for generated code."""
    report = audit_production_response(request, response)
    return {
        "production_required": report.required,
        "approved": report.approved,
        "score": report.score,
        "violations": list(report.violations),
        "evidence": list(report.evidence),
    }


def main() -> int:
    """Audit a response file from the command line."""
    parser = argparse.ArgumentParser(
        description="Audit generated code for unsupported production claims."
    )
    parser.add_argument("response_file", type=Path)
    parser.add_argument(
        "--request",
        default="Provide a production-ready implementation",
        help="Original user request used to determine required rigor.",
    )
    args = parser.parse_args()

    response = args.response_file.read_text(encoding="utf-8")
    result = enforce_requirements(response, args.request)
    print(json.dumps(result, indent=2))
    return 0 if result["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
