"""Hybrid, symbol-aware repository retrieval for coding requests."""
from __future__ import annotations

import ast
import dataclasses
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "site-packages",
}
_MAX_FILE_BYTES = 1_500_000
_CACHE_LOCK = threading.RLock()
_CACHE_ROOT: str | None = None
_CACHE_EXPIRY = 0.0
_CACHE_SIGNATURE: tuple[int, int] | None = None
_CACHE_CHUNKS: tuple["CodeChunk", ...] = ()


@dataclasses.dataclass(frozen=True)
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    symbol: str
    kind: str
    text: str
    tokens: tuple[str, ...]


def _split_identifier(value: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", expanded)
        for token in token.split("_")
        if len(token) >= 2
    ]


def _tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_./:\-]{1,}", text or ""):
        tokens.extend(_split_identifier(raw.replace("/", "_").replace(".", "_")))
    return tokens


def _iter_python_files(root: Path) -> Iterable[Path]:
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [
            name
            for name in subdirs
            if name not in _IGNORE_DIRS and not name.startswith(".tox")
        ]
        base = Path(directory)
        for name in files:
            if not name.endswith(".py"):
                continue
            path = base / name
            try:
                if path.stat().st_size <= _MAX_FILE_BYTES:
                    yield path
            except OSError:
                continue


def _signature(files: list[Path]) -> tuple[int, int]:
    newest = 0
    for path in files:
        try:
            newest = max(newest, path.stat().st_mtime_ns)
        except OSError:
            continue
    return len(files), newest


def _node_end(node: ast.AST, fallback: int) -> int:
    return int(getattr(node, "end_lineno", fallback) or fallback)


def _slice(lines: list[str], start: int, end: int) -> str:
    start = max(1, start)
    end = min(len(lines), max(start, end))
    return "\n".join(lines[start - 1 : end])


def _chunk(
    relative: str,
    lines: list[str],
    start: int,
    end: int,
    symbol: str,
    kind: str,
) -> CodeChunk | None:
    text = _slice(lines, start, end).strip()
    if len(text) < 25:
        return None
    token_source = f"{relative} {symbol} {kind}\n{text}"
    return CodeChunk(
        path=relative,
        start_line=start,
        end_line=end,
        symbol=symbol,
        kind=kind,
        text=text,
        tokens=tuple(_tokenize(token_source)),
    )


def _file_chunks(path: Path, root: Path) -> list[CodeChunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    lines = text.splitlines()
    relative = path.relative_to(root).as_posix()
    chunks: list[CodeChunk] = []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None

    if tree is not None:
        imports_end = 0
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_end = max(imports_end, _node_end(node, node.lineno))
        if imports_end:
            header = _chunk(
                relative,
                lines,
                1,
                min(len(lines), imports_end + 8),
                "<module-header>",
                "module",
            )
            if header:
                chunks.append(header)

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            start = max(1, node.lineno - 3)
            end = min(len(lines), _node_end(node, node.lineno) + 3)
            item = _chunk(
                relative,
                lines,
                start,
                end,
                getattr(node, "name", "<anonymous>"),
                kind,
            )
            if item:
                chunks.append(item)

    # Fallback windows retain module-level constants and malformed source.
    window = 70
    stride = 55
    for start in range(1, len(lines) + 1, stride):
        end = min(len(lines), start + window - 1)
        item = _chunk(
            relative,
            lines,
            start,
            end,
            f"lines-{start}-{end}",
            "window",
        )
        if item:
            chunks.append(item)

    # Remove exact duplicate ranges emitted by AST + windows.
    seen = set()
    unique = []
    for item in chunks:
        key = (item.path, item.start_line, item.end_line, item.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _build_index(root: Path) -> tuple[CodeChunk, ...]:
    files = list(_iter_python_files(root))
    chunks = []
    for path in files:
        chunks.extend(_file_chunks(path, root))
    return tuple(chunks)


def _index(root: Path) -> tuple[CodeChunk, ...]:
    global _CACHE_ROOT, _CACHE_EXPIRY, _CACHE_SIGNATURE, _CACHE_CHUNKS
    now = time.monotonic()
    root = root.resolve()
    ttl = max(1.0, float(os.getenv("ELITE_CODE_RAG_CACHE_SECONDS", "20")))

    with _CACHE_LOCK:
        if (
            _CACHE_ROOT == str(root)
            and _CACHE_CHUNKS
            and now < _CACHE_EXPIRY
        ):
            return _CACHE_CHUNKS

        files = list(_iter_python_files(root))
        signature = _signature(files)
        if (
            _CACHE_ROOT == str(root)
            and _CACHE_CHUNKS
            and signature == _CACHE_SIGNATURE
        ):
            _CACHE_EXPIRY = now + ttl
            return _CACHE_CHUNKS

        chunks = []
        for path in files:
            chunks.extend(_file_chunks(path, root))
        _CACHE_ROOT = str(root)
        _CACHE_SIGNATURE = signature
        _CACHE_CHUNKS = tuple(chunks)
        _CACHE_EXPIRY = now + ttl
        return _CACHE_CHUNKS


def invalidate_index() -> None:
    global _CACHE_EXPIRY, _CACHE_CHUNKS, _CACHE_SIGNATURE
    with _CACHE_LOCK:
        _CACHE_EXPIRY = 0.0
        _CACHE_CHUNKS = ()
        _CACHE_SIGNATURE = None


def _traceback_targets(query: str) -> list[tuple[str, int]]:
    targets = []
    pattern = re.compile(
        r'(?:File\s+["\']|)([A-Za-z0-9_./\-]+\.py)["\']?'
        r'(?:,\s*line\s*|:)(\d+)',
        re.IGNORECASE,
    )
    for path, line in pattern.findall(query or ""):
        targets.append((path.replace("\\", "/"), int(line)))
    return targets


def _idf(chunks: tuple[CodeChunk, ...]) -> dict[str, float]:
    document_frequency = Counter()
    for chunk in chunks:
        document_frequency.update(set(chunk.tokens))
    count = max(1, len(chunks))
    return {
        token: math.log((count + 1) / (frequency + 0.5)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _score(
    query: str,
    query_tokens: list[str],
    query_counts: Counter,
    chunk: CodeChunk,
    idf: dict[str, float],
    targets: list[tuple[str, int]],
) -> float:
    chunk_counts = Counter(chunk.tokens)
    length = max(1, len(chunk.tokens))
    score = 0.0

    for token, qtf in query_counts.items():
        tf = chunk_counts.get(token, 0)
        if not tf:
            continue
        score += idf.get(token, 1.0) * qtf * (
            (tf * 2.2) / (tf + 1.2 + 0.002 * length)
        )

    lowered_query = query.lower()
    path_lower = chunk.path.lower()
    symbol_lower = chunk.symbol.lower()

    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query):
        exact = raw.lower()
        if exact in symbol_lower:
            score += 12.0
        if exact in path_lower:
            score += 7.0
        if re.search(rf"\b{re.escape(exact)}\b", chunk.text.lower()):
            score += 4.0

    if chunk.kind in {"function", "class"}:
        score += 1.5

    for target_path, line in targets:
        if path_lower.endswith(target_path.lower()) or target_path.lower() in path_lower:
            score += 30.0
            if chunk.start_line <= line <= chunk.end_line:
                score += 70.0

    if query_tokens and not any(token in chunk_counts for token in query_tokens):
        score *= 0.1
    return score


def get_relevant_code_context(
    query: str,
    top_k: int = 8,
    root: str | os.PathLike[str] = ".",
) -> str:
    """Return symbol-aware, traceback-aware code context with line numbers."""
    project_root = Path(root).resolve()
    chunks = _index(project_root)
    if not chunks:
        return ""

    query_tokens = _tokenize(query)
    query_counts = Counter(query_tokens)
    if not query_counts:
        return ""

    idf = _idf(chunks)
    targets = _traceback_targets(query)
    scored = [
        (
            _score(
                query,
                query_tokens,
                query_counts,
                chunk,
                idf,
                targets,
            ),
            chunk,
        )
        for chunk in chunks
    ]
    scored = [
        (score, chunk)
        for score, chunk in scored
        if score > 0.4
    ]
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].path,
            item[1].start_line,
        )
    )
    if not scored:
        return ""

    requested = max(1, min(int(top_k), 20))
    selected = []
    per_file = defaultdict(int)
    for score, chunk in scored:
        if per_file[chunk.path] >= 3:
            continue
        selected.append((score, chunk))
        per_file[chunk.path] += 1
        if len(selected) >= requested:
            break

    max_chars = max(
        4000,
        min(
            int(os.getenv("ELITE_CODE_RAG_MAX_CHARS", "26000")),
            80000,
        ),
    )
    parts = [
        "[RELEVANT CODEBASE CONTEXT — HYBRID SYMBOL INDEX V20]",
        f"Root: {project_root}",
    ]
    used = sum(len(part) for part in parts)
    for score, chunk in selected:
        header = (
            f"\n--- {chunk.path}:{chunk.start_line}-{chunk.end_line} "
            f"[{chunk.kind} {chunk.symbol}] score={score:.2f} ---\n"
        )
        body = chunk.text
        remaining = max_chars - used - len(header)
        if remaining <= 200:
            break
        if len(body) > remaining:
            body = body[:remaining] + "\n...[context limit reached]..."
        parts.extend([header, body])
        used += len(header) + len(body)

    parts.append("\n[END RELEVANT CODEBASE CONTEXT]")
    return "\n".join(parts)
