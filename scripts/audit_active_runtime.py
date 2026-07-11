"""Generate a static audit of EliteOmni's active Python runtime."""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import re
from pathlib import Path


DANGEROUS_PATTERNS = {
    "shell=True": re.compile(r"shell\s*=\s*True"),
    "runtime pip install": re.compile(
        r"subprocess\.(?:run|Popen)\([^)]*pip[^)]*install",
        re.DOTALL,
    ),
    "star import": re.compile(r"^\s*from\s+\S+\s+import\s+\*", re.MULTILINE),
    "bare except": re.compile(r"^\s*except\s*:", re.MULTILINE),
}


def python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        yield path


def imports_for(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/ACTIVE_RUNTIME_AUDIT.md"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    files = list(python_files(root))
    basenames: dict[str, list[Path]] = collections.defaultdict(list)
    findings: list[tuple[str, str, str]] = []
    longest: list[tuple[int, Path]] = []

    for path in files:
        relative = path.relative_to(root)
        basenames[path.name].append(relative)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        longest.append((max((len(line) for line in text.splitlines()), default=0), relative))
        for label, pattern in DANGEROUS_PATTERNS.items():
            count = len(pattern.findall(text))
            if count:
                findings.append((str(relative), label, str(count)))

    duplicates = {
        name: paths
        for name, paths in basenames.items()
        if len(paths) > 1
    }
    app_imports = imports_for(root / "app.py")
    longest.sort(reverse=True)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EliteOmni Active Runtime Audit",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Serving path",
        "",
        "- `app.py` is the live FastAPI composition root.",
        "- The active request path uses `pipeline_sync`, `build_system_prompt`, "
        "`build_chatml`, `modules.core.http_client`, and `modules.reliability`.",
        "- Quality V18 is installed at the final response boundary so later "
        "legacy enhancement passes cannot bypass it.",
        "",
        "## Direct imports from app.py",
        "",
    ]
    lines.extend(f"- `{name}`" for name in sorted(set(app_imports)))
    lines.extend((
        "",
        "## Duplicate Python basenames",
        "",
    ))
    for name, paths in sorted(duplicates.items()):
        lines.append(f"- `{name}`: " + ", ".join(f"`{path}`" for path in paths))

    lines.extend((
        "",
        "## Dangerous/static patterns",
        "",
        "| File | Pattern | Count |",
        "|---|---:|---:|",
    ))
    for file_name, label, count in sorted(findings):
        lines.append(f"| `{file_name}` | {label} | {count} |")

    lines.extend((
        "",
        "## Largest physical line lengths",
        "",
        "| File | Longest line |",
        "|---|---:|",
    ))
    for length, path in longest[:20]:
        lines.append(f"| `{path}` | {length} |")

    lines.extend((
        "",
        "## Priority recommendations",
        "",
        "1. Gradually split `app.py` into route, orchestration, research, and "
        "verification services while keeping Quality V18 as the invariant edge.",
        "2. Remove duplicate root/modules implementations after import-graph "
        "tests prove which copies are inactive.",
        "3. Eliminate runtime package installation and shell execution from the "
        "sandbox.",
        "4. Configure distinct models through `ELITE_MODEL_*` variables only "
        "after confirming provider availability.",
        "5. Add golden end-to-end evaluations for coding, research, reasoning, "
        "and tool-grounding before accepting future self-modifying prompts.",
        "",
    ))
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
