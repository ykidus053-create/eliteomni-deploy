#!/usr/bin/env python3
"""Fail closed on EliteOmni's highest-risk production configuration defects."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def main() -> int:
    app = Path("app.py").read_text(encoding="utf-8")
    gitignore = (
        Path(".gitignore").read_text(encoding="utf-8")
        if Path(".gitignore").exists()
        else ""
    )

    failures: list[str] = []
    evidence: list[str] = []

    wildcard_cors = re.search(
        r"allow_origins\s*=\s*\[\s*['\"]\*['\"]\s*\]",
        app,
    )
    if wildcard_cors:
        failures.append("wildcard CORS remains enabled")
    else:
        evidence.append("wildcard CORS disabled")

    if 'get("DEBUG_SECRET", "changeme")' in app:
        failures.append("debug endpoint retains changeme fallback")
    else:
        evidence.append("debug endpoint has no default secret")

    if "# BEGIN PLATFORM EXCELLENCE V30" not in app:
        failures.append("V30 runtime hardening is not wired into app.py")
    else:
        evidence.append("V30 runtime hardening is wired")

    if "admin_token_valid(request)" not in app:
        failures.append("trace endpoint does not use constant-time admin auth")
    else:
        evidence.append("trace endpoint uses V30 admin auth")

    if "*.db" not in gitignore:
        failures.append("runtime database files are not ignored")
    else:
        evidence.append("runtime database files are ignored")

    tracked = subprocess.run(
        ["git", "ls-files", "*.db"],
        text=True,
        capture_output=True,
        check=False,
    )
    tracked_db = [
        line for line in tracked.stdout.splitlines() if line.strip()
    ]
    if tracked_db:
        failures.append(
            "runtime database files remain tracked: "
            + ", ".join(tracked_db)
        )
    else:
        evidence.append("no runtime database file is tracked")

    result = {
        "gate": "EliteOmni Security Gate V30",
        "approved": not failures,
        "failures": failures,
        "evidence": evidence,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
