"""Dependency-aware repository intelligence for EliteOmni V28."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_SUPPORTED = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".cs", ".cpp", ".cc", ".cxx", ".c", ".sql", ".sh", ".ps1",
    ".yaml", ".yml", ".json", ".toml", ".html", ".css",
}
_SKIP = {
    ".git", ".venv", "venv", "__pycache__", "site-packages",
    "node_modules", "dist", "build", ".next", ".pytest_cache",
}
_FILE_RE = re.compile(
    r"(?<![\w])([A-Za-z0-9_.\-/\\]+\."
    r"(?:py|js|jsx|ts|tsx|go|rs|java|cs|cpp|cc|cxx|c|sql|sh|ps1|"
    r"yaml|yml|json|toml|html|css))(?![\w])",
    re.IGNORECASE,
)
_TRACE_RE = re.compile(
    r"(?:File\s+[\"'])?([A-Za-z0-9_.\-/\\]+\.py)"
    r"(?:[\"'],?\s*line\s*|:)(\d+)",
    re.IGNORECASE,
)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_STOP = {
    "about", "after", "again", "before", "change", "class", "code",
    "could", "error", "file", "from", "function", "have", "into",
    "make", "module", "please", "return", "should", "that", "then",
    "this", "using", "with", "without",
}
_CACHE_LOCK = threading.RLock()
_CACHE: dict[str, tuple[str, "RepositoryIndex"]] = {}


@dataclass(frozen=True)
class FileFacts:
    path: str
    language: str
    definitions: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()
    test_file: bool = False
    parse_error: str = ""


@dataclass
class RepositoryIndex:
    root: Path
    signature: str
    files: dict[str, FileFacts] = field(default_factory=dict)
    symbols: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    dependencies: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    dependents: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )


def repository_root() -> Path:
    configured = os.getenv("ELITE_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "app.py").exists():
        return candidate
    cwd = Path.cwd().resolve()
    return cwd if (cwd / "app.py").exists() else candidate


def _language(path: Path) -> str:
    return {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".go": "go",
        ".rs": "rust", ".java": "java", ".cs": "csharp",
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c": "c",
        ".sql": "sql", ".sh": "bash", ".ps1": "powershell",
        ".yaml": "yaml", ".yml": "yaml", ".json": "json",
        ".toml": "toml", ".html": "html", ".css": "css",
    }.get(path.suffix.lower(), "text")


def _is_test(relative: str) -> bool:
    name = Path(relative).name.lower()
    lowered = f"/{relative.lower()}"
    return (
        "/tests/" in lowered
        or name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
    )


def _iter_files(root: Path) -> Iterable[Path]:
    limit = max(50, min(int(os.getenv("ELITE_REPO_MAX_FILES", "800")), 5000))
    max_bytes = max(
        4096,
        min(int(os.getenv("ELITE_REPO_FILE_MAX_BYTES", "262144")), 2_000_000),
    )
    count = 0
    for path in sorted(root.rglob("*")):
        if count >= limit:
            return
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED:
            continue
        if any(part in _SKIP for part in path.parts):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        count += 1
        yield path


def _signature(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256(str(root).encode())
    for path in paths:
        try:
            stat = path.stat()
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
        except OSError:
            continue
    return digest.hexdigest()


def _python_facts(relative: str, source: str) -> FileFacts:
    definitions: set[str] = set()
    imports: set[str] = set()
    calls: set[str] = set()
    parse_error = ""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        tree = None
        parse_error = f"line {exc.lineno}: {exc.msg}"

    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                definitions.add(node.name)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add("." * node.level + (node.module or ""))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)

    return FileFacts(
        relative,
        "python",
        tuple(sorted(definitions)),
        tuple(sorted(imports)),
        tuple(sorted(calls)),
        _is_test(relative),
        parse_error,
    )


def _generic_facts(path: Path, relative: str, source: str) -> FileFacts:
    language = _language(path)
    definitions = set(
        re.findall(
            r"\b(?:function|class|interface|type|enum|struct|trait|fn|func|def)"
            r"\s+([A-Za-z_][A-Za-z0-9_]*)",
            source,
        )
    )
    imports = set(
        re.findall(
            r"(?:from|import|require|use|using)\s*(?:\(|)\s*[\"']?"
            r"([A-Za-z_./:@-][A-Za-z0-9_./:@-]*)",
            source,
        )
    )
    calls = set(
        re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", source)
    )

    if language == "sql":
        definitions.update(
            re.findall(
                r"\b(?:CREATE|ALTER)\s+"
                r"(?:PROCEDURE|PROC|FUNCTION|VIEW|TABLE|TRIGGER)\s+"
                r"(?:\[?[\w]+\]?\.)?\[?([\w]+)\]?",
                source,
                re.IGNORECASE,
            )
        )
        imports.update(
            re.findall(
                r"\b(?:FROM|JOIN|EXEC(?:UTE)?)\s+"
                r"(?:\[?[\w]+\]?\.)?\[?([\w]+)\]?",
                source,
                re.IGNORECASE,
            )
        )

    return FileFacts(
        relative,
        language,
        tuple(sorted(definitions)),
        tuple(sorted(imports)),
        tuple(sorted(calls)),
        _is_test(relative),
    )


def _module_candidates(name: str, current: str) -> list[str]:
    raw = name.strip()
    if not raw:
        return []
    leading = len(raw) - len(raw.lstrip("."))
    raw = raw.lstrip(".")
    base = Path(current).parent
    if leading:
        for _ in range(max(0, leading - 1)):
            base = base.parent
        prefix = base.as_posix().strip("/")
        raw = ".".join(part for part in (prefix, raw) if part)
    dotted = raw.replace("::", ".").replace("/", ".")
    pieces = [part for part in dotted.split(".") if part and part != "*"]
    if not pieces:
        return []
    stem = "/".join(pieces)
    return [
        f"{stem}.py", f"{stem}/__init__.py", f"{stem}.js",
        f"{stem}.ts", f"{stem}.tsx", f"{stem}.jsx",
        f"{stem}.go", f"{stem}.rs", f"{stem}.java", f"{stem}.cs",
    ]


def build_repository_index(
    root: Path | str | None = None,
    *,
    force: bool = False,
) -> RepositoryIndex:
    resolved = Path(root or repository_root()).expanduser().resolve()
    paths = list(_iter_files(resolved))
    signature = _signature(resolved, paths)
    key = str(resolved)

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == signature and not force:
            return cached[1]

    index = RepositoryIndex(resolved, signature)

    for path in paths:
        try:
            relative = path.relative_to(resolved).as_posix()
            source = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, ValueError):
            continue
        facts = (
            _python_facts(relative, source)
            if path.suffix.lower() == ".py"
            else _generic_facts(path, relative, source)
        )
        index.files[relative] = facts
        for symbol in facts.definitions:
            index.symbols[symbol].add(relative)

    all_paths = set(index.files)
    stems: dict[str, set[str]] = defaultdict(set)
    for relative in all_paths:
        stems[Path(relative).stem].add(relative)

    for relative, facts in index.files.items():
        for imported in facts.imports:
            targets: set[str] = set()
            for candidate in _module_candidates(imported, relative):
                if candidate in all_paths:
                    targets.add(candidate)
            tail = imported.replace("::", ".").split(".")[-1].strip()
            targets.update(stems.get(tail, set()))
            for target in targets:
                if target == relative:
                    continue
                index.dependencies[relative].add(target)
                index.dependents[target].add(relative)

    with _CACHE_LOCK:
        _CACHE[key] = (signature, index)
    return index


def _signals(query: str) -> tuple[list[str], list[tuple[str, int]], set[str]]:
    files = [
        value.replace("\\", "/").lstrip("./")
        for value in _FILE_RE.findall(query or "")
    ]
    traces = [
        (value.replace("\\", "/").lstrip("./"), int(line))
        for value, line in _TRACE_RE.findall(query or "")
    ]
    identifiers = {
        value
        for value in _IDENT_RE.findall(query or "")
        if value.lower() not in _STOP
    }
    return list(dict.fromkeys(files)), traces, identifiers


def _path_matches(reference: str, paths: Iterable[str]) -> set[str]:
    value = reference.replace("\\", "/").lstrip("./")
    return {
        path
        for path in paths
        if path == value
        or path.endswith("/" + value)
        or Path(path).name == Path(value).name
    }


def _tests_for(path: str, all_paths: set[str]) -> set[str]:
    stem = Path(path).stem.lower()
    return {
        candidate
        for candidate in all_paths
        if _is_test(candidate) and stem in Path(candidate).name.lower()
    }


def analyze_repository(
    query: str,
    *,
    root: Path | str | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    if os.getenv("ELITE_REPO_INTELLIGENCE", "1") != "1":
        return {
            "enabled": False,
            "summary": "Repository intelligence is disabled.",
            "files": [],
            "symbols": [],
            "tests": [],
            "risks": [],
        }

    index = build_repository_index(root)
    explicit, traces, identifiers = _signals(query)
    all_paths = set(index.files)
    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, set[str]] = defaultdict(set)
    lines: dict[str, set[int]] = defaultdict(set)

    for reference in explicit:
        for path in _path_matches(reference, all_paths):
            scores[path] += 120
            reasons[path].add(f"explicit file: {reference}")

    for reference, line in traces:
        for path in _path_matches(reference, all_paths):
            scores[path] += 180
            reasons[path].add(f"traceback line {line}")
            lines[path].add(line)

    for identifier in identifiers:
        for path in index.symbols.get(identifier, set()):
            scores[path] += 45
            reasons[path].add(f"defines {identifier}")
        for path, facts in index.files.items():
            if identifier.lower() in Path(path).stem.lower():
                scores[path] += 12
                reasons[path].add(f"filename matches {identifier}")
            if identifier in facts.calls:
                scores[path] += 8
                reasons[path].add(f"calls {identifier}")

    seeds = [
        path
        for path, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:8]
    ]
    queue: deque[tuple[str, int]] = deque((path, 0) for path in seeds)
    visited = set(seeds)

    while queue:
        current, depth = queue.popleft()
        if depth >= 2:
            continue
        dependencies = set(index.dependencies.get(current, set()))
        dependents = set(index.dependents.get(current, set()))
        for neighbor in dependencies | dependents:
            relation = "dependency" if neighbor in dependencies else "dependent"
            scores[neighbor] += 24 if depth == 0 else 10
            reasons[neighbor].add(f"{relation} of {current}")
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    tests: set[str] = set()
    for path in list(scores):
        for test_path in _tests_for(path, all_paths):
            tests.add(test_path)
            scores[test_path] += 30
            reasons[test_path].add(f"test for {path}")

    limit = max_files or int(os.getenv("ELITE_REPO_CONTEXT_FILES", "10"))
    limit = max(3, min(limit, 30))
    ranked = sorted(scores, key=lambda path: (-scores[path], path))[:limit]

    files = []
    for path in ranked:
        facts = index.files[path]
        files.append(
            {
                "path": path,
                "language": facts.language,
                "score": round(scores[path], 2),
                "reasons": sorted(reasons[path]),
                "definitions": list(facts.definitions[:15]),
                "dependencies": sorted(index.dependencies.get(path, set()))[:10],
                "dependents": sorted(index.dependents.get(path, set()))[:10],
                "traceback_lines": sorted(lines.get(path, set())),
                "test_file": facts.test_file,
                "parse_error": facts.parse_error,
            }
        )

    risks = []
    for item in files:
        if item["dependents"]:
            risks.append(
                f"{item['path']} has {len(item['dependents'])} indexed dependent(s)."
            )
        if item["parse_error"]:
            risks.append(
                f"{item['path']} parse issue: {item['parse_error']}."
            )

    return {
        "enabled": True,
        "root": str(index.root),
        "signature": index.signature[:16],
        "indexed_files": len(index.files),
        "summary": (
            f"Indexed {len(index.files)} source files and selected "
            f"{len(files)} change-impact files."
        ),
        "files": files,
        "symbols": sorted(
            name for name in identifiers if name in index.symbols
        ),
        "tests": sorted(tests),
        "risks": risks[:12],
    }


def format_repository_impact(
    query: str,
    *,
    root: Path | str | None = None,
    max_files: int | None = None,
) -> str:
    result = analyze_repository(query, root=root, max_files=max_files)
    if not result.get("enabled"):
        return ""

    output = [
        "[REPOSITORY CHANGE-IMPACT MAP]",
        result["summary"],
    ]
    for item in result["files"]:
        output.append(
            f"- {item['path']} [{item['language']}] — "
            + "; ".join(item["reasons"][:3])
        )
        if item["definitions"]:
            output.append(
                "  symbols: " + ", ".join(item["definitions"][:10])
            )
        if item["dependencies"]:
            output.append(
                "  depends on: " + ", ".join(item["dependencies"][:6])
            )
        if item["dependents"]:
            output.append(
                "  used by: " + ", ".join(item["dependents"][:6])
            )

    if result["tests"]:
        output.append(
            "Likely regression tests: " + ", ".join(result["tests"][:10])
        )
    if result["risks"]:
        output.append("Change risks:")
        output.extend(f"- {risk}" for risk in result["risks"][:8])
    output.append(
        "Use this map as evidence; do not invent unseen interfaces."
    )
    return "\n".join(output)


def runtime_status(root: Path | str | None = None) -> dict[str, Any]:
    index = build_repository_index(root)
    return {
        "version": "V28",
        "enabled": os.getenv("ELITE_REPO_INTELLIGENCE", "1") == "1",
        "root": str(index.root),
        "indexed_files": len(index.files),
        "indexed_symbols": len(index.symbols),
        "dependency_edges": sum(
            len(values) for values in index.dependencies.values()
        ),
        "signature": index.signature[:16],
    }
