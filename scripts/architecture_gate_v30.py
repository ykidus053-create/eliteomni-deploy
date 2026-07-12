#!/usr/bin/env python3
"""Architecture budgets that prevent new monolith and bypass-path growth."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path


def main() -> int:
    failures: list[str] = []
    evidence: list[str] = []

    app_path = Path("app.py")
    app = app_path.read_text(encoding="utf-8")
    max_app_bytes = int(os.getenv("ELITE_APP_MAX_BYTES", "400000"))

    try:
        ast.parse(app)
        evidence.append("app.py parses")
    except SyntaxError as exc:
        failures.append(f"app.py syntax error: {exc}")

    app_bytes = app_path.stat().st_size
    if app_bytes > max_app_bytes:
        failures.append(
            f"app.py exceeds byte budget: {app_bytes} > {max_app_bytes}"
        )
    else:
        evidence.append(
            f"app.py remains within transitional byte budget: {app_bytes}"
        )

    if app.count("# BEGIN PLATFORM EXCELLENCE V30") != 1:
        failures.append("V30 platform integration marker is not singular")
    else:
        evidence.append("V30 platform integration is singular")

    new_modules = [
        Path("modules/platform_excellence_v30.py"),
        Path("modules/coding_benchmark_v30.py"),
    ]
    for path in new_modules:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        import_star = any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        )
        if import_star:
            failures.append(f"{path} uses import-star")
        else:
            evidence.append(f"{path} has explicit imports")

        unsafe_findings: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in {"eval", "exec"}
                ):
                    unsafe_findings.append(
                        f"direct {node.func.id}() call"
                    )

                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "loads"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pickle"
                ):
                    unsafe_findings.append(
                        "pickle.loads() call"
                    )

                for keyword in node.keywords:
                    if (
                        keyword.arg == "shell"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        unsafe_findings.append(
                            "subprocess shell=True"
                        )

                    if (
                        keyword.arg == "verify"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False
                    ):
                        unsafe_findings.append(
                            "TLS verification disabled"
                        )

        if unsafe_findings:
            failures.append(
                f"{path} contains prohibited unsafe behavior: "
                + ", ".join(sorted(set(unsafe_findings)))
            )
        else:
            evidence.append(
                f"{path} has no executable unsafe primitives"
            )

        line_count = len(source.splitlines())
        if line_count > 700:
            failures.append(
                f"{path} exceeds focused-module line budget: {line_count}"
            )
        else:
            evidence.append(
                f"{path} stays within focused-module line budget"
            )

    result = {
        "gate": "EliteOmni Architecture Gate V30",
        "approved": not failures,
        "failures": failures,
        "evidence": evidence,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
