"""Unified coding and reasoning helpers for EliteOmni V27."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "js": "javascript",
    "javascript": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "typescript",
    "sh": "bash",
    "shell": "bash",
    "bash": "bash",
    "ps1": "powershell",
    "powershell": "powershell",
    "sql": "sql",
    "go": "go",
    "golang": "go",
    "rs": "rust",
    "rust": "rust",
    "java": "java",
    "cs": "csharp",
    "csharp": "csharp",
    "c#": "csharp",
    "cpp": "cpp",
    "c++": "cpp",
    "c": "c",
    "html": "html",
    "css": "css",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
}

_EXTENSION_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".sh": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}

_PLACEHOLDER_PATTERNS = (
    r"\bTODO\b",
    r"\bFIXME\b",
    r"NotImplementedError",
    r"\byour[_ -]?(?:api[_ -]?key|password|db[_ -]?url)\b",
    r"\brest of (?:the )?implementation\b",
    r"\bimplement (?:this|here)\b",
    r"\bplaceholder\b",
    r"\bpseudocode\b",
)


def repository_root() -> Path:
    configured = os.getenv("ELITE_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "app.py").exists():
        return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "app.py").exists():
        return cwd

    return candidate


def detect_language(text: str) -> str:
    value = text or ""

    fence = re.search(r"```([A-Za-z0-9_+#.-]+)", value)
    if fence:
        raw = fence.group(1).lower()
        if raw in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[raw]

    for file_ref in re.findall(
        r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+",
        value,
    ):
        language = _EXTENSION_LANGUAGE.get(Path(file_ref).suffix.lower())
        if language:
            return language

    lowered = value.lower()
    for alias in sorted(_LANGUAGE_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", lowered):
            return _LANGUAGE_ALIASES[alias]

    if any(token in value for token in ("def ", "import ", "Traceback")):
        return "python"
    if any(token in value for token in ("const ", "let ", "=>", "npm ")):
        return "javascript"
    if re.search(r"\bSELECT\b.+\bFROM\b", value, re.IGNORECASE | re.DOTALL):
        return "sql"

    return "python"


def trim_messages(
    messages: Sequence[Mapping[str, Any]] | None,
    max_chars: int | None = None,
) -> list[dict[str, str]]:
    """Preserve the system prompt and newest turns within an exact limit."""
    cap = max_chars or int(
        os.getenv("ELITE_MESSAGE_MAX_CHARS", "18000")
    )
    cap = max(2000, min(int(cap), 100000))

    normalized: list[dict[str, str]] = []

    for item in messages or []:
        if not isinstance(item, Mapping):
            continue

        role = str(item.get("role", "user") or "user")
        content = item.get("content", "")

        if not isinstance(content, str):
            content = str(content)

        if content.strip():
            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    if not normalized:
        return []

    system_message = next(
        (
            item
            for item in normalized
            if item["role"] == "system"
        ),
        None,
    )

    system_content = (
        system_message["content"]
        if system_message
        else ""
    )

    system_budget = min(
        len(system_content),
        min(4000, cap // 4),
    )

    conversation_budget = cap - system_budget
    selected: list[dict[str, str]] = []
    used = 0

    non_system = [
        item
        for item in normalized
        if item["role"] != "system"
    ]

    for item in reversed(non_system):
        remaining = conversation_budget - used

        if remaining <= 0:
            break

        content = item["content"]

        if len(content) > remaining:
            content = content[-remaining:]

        if not content:
            continue

        selected.append(
            {
                "role": item["role"],
                "content": content,
            }
        )

        used += len(content)

    selected.reverse()

    if system_budget:
        selected.insert(
            0,
            {
                "role": "system",
                "content": system_content[:system_budget],
            },
        )

    total = sum(
        len(item["content"])
        for item in selected
    )

    if total > cap:
        overflow = total - cap

        for item in selected:
            if item["role"] == "system":
                continue

            remove = min(
                overflow,
                len(item["content"]),
            )

            item["content"] = item["content"][remove:]
            overflow -= remove

            if overflow == 0:
                break

    return [
        item
        for item in selected
        if item["content"]
    ]

def _file_references(text: str) -> list[str]:
    refs = re.findall(
        r"\b[A-Za-z0-9_.\-/\\]+\."
        r"(?:py|js|jsx|ts|tsx|go|rs|java|cs|cpp|c|sql|sh|ps1|"
        r"yaml|yml|json|toml|md|html|css)\b",
        text or "",
        re.IGNORECASE,
    )
    return list(dict.fromkeys(ref.replace("\\", "/") for ref in refs))[:12]


def _traceback_references(text: str) -> list[str]:
    refs = re.findall(
        r"(?:File\s+[\"'])?([A-Za-z0-9_.\-/\\]+\.py)"
        r"(?:[\"'],?\s*line\s*|:)(\d+)",
        text or "",
        re.IGNORECASE,
    )
    return [f"{path.replace(chr(92), '/')}:{line}" for path, line in refs][:10]


def requirements_contract(text: str) -> str:
    clauses = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    selected: list[str] = []
    signal = re.compile(
        r"\b(must|need|should|required|include|without|only|exactly|"
        r"do not|don't|avoid|preserve|compatible|production|test|"
        r"fix|error|traceback|limit|security|performance)\b",
        re.IGNORECASE,
    )

    for clause in clauses:
        cleaned = " ".join(clause.split()).strip()
        if cleaned and signal.search(cleaned):
            selected.append(cleaned[:700])
        if len(selected) >= 10:
            break

    files = _file_references(text)
    traces = _traceback_references(text)
    language = detect_language(text)

    lines = [f"Target language: {language}."]
    lines.extend(f"- {item}" for item in selected)
    if files:
        lines.append("- Referenced files: " + ", ".join(files))
    if traces:
        lines.append("- Traceback locations: " + ", ".join(traces))
    if len(lines) == 1:
        lines.append("- Satisfy the request completely and preserve existing interfaces.")

    return "\n".join(lines)


def knowledge_boundary_check(message: str, skill: str = "general") -> str:
    """Return compact grounding instructions without making another model call."""
    lowered = (message or "").lower()
    files = _file_references(message)
    traces = _traceback_references(message)

    if skill == "coder" or files or traces:
        parts = [
            "[CODING KNOWLEDGE BOUNDARY]",
            "Use repository evidence before inventing functions, files, APIs, or schemas.",
            "Preserve public interfaces unless the request explicitly changes them.",
            "Distinguish syntax/static validation from actual execution and tests.",
        ]
        if files:
            parts.append("Files named by the user: " + ", ".join(files))
        if traces:
            parts.append("Traceback targets: " + ", ".join(traces))
        return "\n".join(parts)

    freshness = (
        "latest",
        "current",
        "today",
        "recent",
        "price",
        "version",
        "release",
        "law",
    )
    if any(token in lowered for token in freshness):
        return (
            "[FRESHNESS BOUNDARY]\n"
            "Current claims require live retrieval. Do not present model memory "
            "as current fact when retrieval is unavailable."
        )

    return ""


def prefetch_plan(message: str, skill: str = "general") -> dict[str, str]:
    """Build deterministic context; never consumes a provider request."""
    result = {
        "requirements": requirements_contract(message),
    }

    try:
        from modules.project_context import infer_project_context

        context = infer_project_context()
        if context:
            result["project"] = context[:2500]
    except Exception:
        pass

    if skill == "coder" or _file_references(message) or _traceback_references(message):
        try:
            from code_rag import get_relevant_code_context

            context = get_relevant_code_context(
                message,
                top_k=int(os.getenv("ELITE_CODE_RAG_TOP_K", "8")),
                root=repository_root(),
            )
            if context:
                result["repository"] = context[:12000]
        except Exception:
            pass

    return result


def architect_plan(message: str) -> str:
    """Create a repository-aware implementation plan without an extra LLM call."""
    language = detect_language(message)
    files = _file_references(message)
    traces = _traceback_references(message)
    target = ", ".join(files or traces) or "the smallest relevant module set"

    return "\n".join(
        (
            f"1. Locate the failure and active call path in {target}; reproduce it from the supplied evidence.",
            "2. Extract the required behavior, interfaces, data shapes, error semantics, and compatibility constraints.",
            f"3. Implement the smallest complete {language} change; avoid unrelated rewrites and invented dependencies.",
            "4. Add or update focused regression tests covering the failing path, edge cases, and error handling.",
            "5. Run syntax/static checks, then focused tests, then the full suite; report exactly what was and was not executed.",
            "6. Confirm deployment/runtime configuration separately from source correctness and keep rollback straightforward.",
        )
    )


def extract_code_blocks(response: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"```([A-Za-z0-9_+#.-]*)\s*\n(.*?)```",
        re.DOTALL,
    )
    return [
        ((_LANGUAGE_ALIASES.get(lang.lower(), lang.lower()) or "text"), code.strip())
        for lang, code in pattern.findall(response or "")
        if code.strip()
    ]


def verify_code_response(response: str, requested_language: str) -> dict[str, Any]:
    blocks = extract_code_blocks(response)
    issues: list[str] = []
    checks: list[str] = []

    if not blocks:
        issues.append("no fenced code block was returned")
        return {
            "approved": False,
            "language": requested_language,
            "checks": checks,
            "issues": issues,
        }

    combined = "\n\n".join(code for _, code in blocks)
    for pattern in _PLACEHOLDER_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            issues.append(f"placeholder signal matched: {pattern}")
            break

    languages = [lang for lang, _ in blocks]
    if requested_language == "python":
        python_blocks = [
            code for lang, code in blocks if lang in {"python", "py", ""}
        ]
        if not python_blocks:
            issues.append("requested Python but no Python block was returned")
        else:
            for index, code in enumerate(python_blocks, start=1):
                try:
                    ast.parse(code)
                    checks.append(f"python block {index}: syntax parsed")
                except SyntaxError as exc:
                    issues.append(
                        f"python block {index}: syntax error line {exc.lineno}: {exc.msg}"
                    )
    else:
        if requested_language not in languages:
            checks.append(
                "language label differs or is absent; no compiler was run"
            )
        else:
            checks.append(
                f"{requested_language} block detected; no compiler was run"
            )

    return {
        "approved": not issues,
        "language": requested_language,
        "checks": checks,
        "issues": issues,
    }


def editor_implement(
    plan: str,
    message: str,
    system: str,
    history: list,
    max_tokens: int,
    *,
    build_chatml: Callable[..., list],
    generate_sync: Callable[..., str],
) -> str:
    """Generate once, verify deterministically, and repair only on real evidence."""
    if not plan:
        return ""

    language = detect_language(message)
    contract = requirements_contract(message)
    context = prefetch_plan(message, "coder")
    repo_context = context.get("repository", "")[:10000]
    project_context = context.get("project", "")[:2000]

    coder_system = (
        f"{system}\n\n"
        "[ELITEOMNI CODING CORE V27]\n"
        f"Target language: {language}.\n"
        "Act as a senior repository engineer. Inspect the supplied interfaces and "
        "traceback locations. Produce a minimal, complete, compatible change. "
        "Do not invent files, APIs, package availability, command output, or test "
        "results. Do not reveal private chain-of-thought. Return the finished "
        "artifact or patch, followed by concise verification instructions. "
        "Use fenced code blocks labeled with the actual language. "
        "Never use TODO, pass, placeholders, omitted sections, or fake credentials."
    )

    prompt_text = (
        f"ORIGINAL TASK:\n{message}\n\n"
        f"IMPLEMENTATION PLAN:\n{plan}\n\n"
        f"REQUIREMENTS:\n{contract}\n\n"
        f"PROJECT FACTS:\n{project_context or '(not available)'}\n\n"
        f"RELEVANT REPOSITORY CONTEXT:\n{repo_context or '(not available)'}\n\n"
        "Return the complete implementation now."
    )

    prompt = build_chatml(coder_system, history or [], prompt_text)
    budget = min(
        max(1200, int(max_tokens or 8000)),
        int(os.getenv("ELITE_CODER_MAX_TOKENS", "8000")),
    )
    response = generate_sync(prompt, budget, "coder", len(message))
    audit = verify_code_response(response, language)

    repair_enabled = os.getenv("ELITE_CODER_REPAIR_ON_EVIDENCE", "1") == "1"
    if not audit["approved"] and repair_enabled:
        evidence = "\n".join(f"- {issue}" for issue in audit["issues"])
        repair_prompt = build_chatml(
            coder_system,
            [],
            (
                f"ORIGINAL TASK:\n{message}\n\n"
                f"THE PREVIOUS IMPLEMENTATION FAILED THESE DETERMINISTIC CHECKS:\n"
                f"{evidence}\n\n"
                "Return one complete corrected implementation. Do not discuss the "
                "failed attempt and do not claim tests were run."
            ),
        )
        repaired = generate_sync(repair_prompt, budget, "coder", len(message))
        repaired_audit = verify_code_response(repaired, language)
        if repaired_audit["approved"] or len(repaired) > len(response):
            response = repaired
            audit = repaired_audit

    if audit["approved"] and audit["checks"]:
        response = response.rstrip() + "\n\n**Static verification:** " + "; ".join(
            audit["checks"]
        )
    elif audit["issues"]:
        response = response.rstrip() + (
            "\n\n**Verification note:** Static checks did not establish correctness: "
            + "; ".join(audit["issues"][:3])
        )

    return response


def runtime_status() -> dict[str, Any]:
    return {
        "version": "V27",
        "repository_root": str(repository_root()),
        "coder_model": os.getenv(
            "ELITE_MODEL_CODER",
            "cerebras/zai-glm-4.7",
        ),
        "agent_team_enabled": os.getenv("ELITE_ENABLE_AGENT_TEAM", "0") == "1",
        "second_loop_enabled": (
            os.getenv("ELITE_ENABLE_SECOND_LOOP_ENGINE", "0") == "1"
        ),
        "buffered_verification_stream": (
            os.getenv("ELITE_BUFFERED_VERIFICATION_STREAM", "0") == "1"
        ),
    }

# BEGIN REPOSITORY INTELLIGENCE V28
_V28_BASE_PREFETCH_PLAN = prefetch_plan
_V28_BASE_ARCHITECT_PLAN = architect_plan
_V28_BASE_RUNTIME_STATUS = runtime_status


def prefetch_plan(message: str, skill: str = "general") -> dict[str, str]:
    result = _V28_BASE_PREFETCH_PLAN(message, skill)
    if (
        os.getenv("ELITE_REPO_INTELLIGENCE", "1") == "1"
        and (
            skill == "coder"
            or _file_references(message)
            or _traceback_references(message)
        )
    ):
        try:
            from modules.repository_intelligence_v28 import (
                format_repository_impact,
            )
            impact = format_repository_impact(
                message,
                root=repository_root(),
            )
            if impact:
                result["impact"] = impact[:12000]
        except Exception as exc:
            result["impact_warning"] = (
                f"Repository impact analysis was unavailable: {exc}"
            )
    return result


def architect_plan(message: str) -> str:
    base = _V28_BASE_ARCHITECT_PLAN(message)
    if os.getenv("ELITE_REPO_INTELLIGENCE", "1") != "1":
        return base
    try:
        from modules.repository_intelligence_v28 import analyze_repository
        analysis = analyze_repository(
            message,
            root=repository_root(),
            max_files=8,
        )
    except Exception:
        return base

    files = [
        item["path"]
        for item in analysis.get("files", [])
        if not item.get("test_file")
    ][:6]
    tests = analysis.get("tests", [])[:6]
    additions = []
    if files:
        additions.append("Impact files to inspect: " + ", ".join(files))
    if tests:
        additions.append("Likely regression tests: " + ", ".join(tests))
    return (
        base
        if not additions
        else base.rstrip() + "\n\nRepository impact:\n- "
        + "\n- ".join(additions)
    )


def runtime_status() -> dict[str, Any]:
    result = dict(_V28_BASE_RUNTIME_STATUS())
    try:
        from modules.repository_intelligence_v28 import (
            runtime_status as _repository_status,
        )
        result["repository_intelligence"] = _repository_status(
            root=repository_root()
        )
    except Exception as exc:
        result["repository_intelligence"] = {
            "version": "V28",
            "error": str(exc),
        }
    return result
# END REPOSITORY INTELLIGENCE V28

# BEGIN PRODUCTION SCOPE GUARD V28.1
_EDUCATIONAL_REQUEST_V281 = re.compile(
    r"\b("
    r"toy|educational|tutorial|teaching example|classroom|"
    r"learning exercise|demo(?:nstration)?|proof of concept|poc|"
    r"minimal example|simplified example"
    r")\b",
    re.IGNORECASE,
)


def explicit_educational_request(message: str) -> bool:
    """Return True only when non-production scope was explicitly requested."""
    return bool(_EDUCATIONAL_REQUEST_V281.search(message or ""))


def production_scope_contract(message: str) -> str:
    """Choose a realistic production scope unless education was explicit."""
    if explicit_educational_request(message):
        return (
            "[EXPLICIT EDUCATIONAL MODE]\n"
            "The user explicitly requested a tutorial, toy, demonstration, "
            "or classroom implementation. Label limitations accurately and "
            "do not claim production readiness."
        )

    if os.getenv("ELITE_PRODUCTION_SCOPE_DEFAULT", "1") != "1":
        return ""

    database_addendum = ""
    lowered = (message or "").lower()
    if any(
        token in lowered
        for token in (
            "database",
            "sql",
            "storage engine",
            "data store",
            "persistence",
            "repository",
        )
    ):
        database_addendum = (
            "\nDATABASE DEFAULTS:\n"
            "- Do not invent an in-memory SQL parser, miniature database, "
            "or storage engine unless the user explicitly asks to implement "
            "a database engine.\n"
            "- Prefer the requested mature engine. When none is named, use "
            "SQLite for a self-contained application or PostgreSQL for a "
            "networked service.\n"
            "- Include real schema constraints, transactions, indexes, "
            "migrations or initialization, parameterized queries, connection "
            "lifecycle, concurrency behavior, error handling, and tests.\n"
        )

    return (
        "[PRODUCTION SCOPE — DEFAULT]\n"
        "Treat this as code intended to run in a real application. "
        "Do not downgrade the request to a toy, educational exercise, demo, "
        "simplified implementation, proof of concept, or in-memory substitute "
        "unless the user explicitly requested that scope.\n"
        "Do not spend the answer debating several weaker interpretations. "
        "Choose the safest realistic production interpretation and implement "
        "it completely.\n"
        "Use mature standard-library or established dependencies for solved "
        "infrastructure problems. Build the requested product behavior around "
        "them instead of recreating foundational infrastructure without an "
        "explicit requirement.\n"
        "Include configuration, validation, durable state where required, "
        "failure handling, resource cleanup, security boundaries, and focused "
        "tests or exact validation commands."
        + database_addendum
    )


_V281_BASE_PREFETCH_PLAN = prefetch_plan
_V281_BASE_ARCHITECT_PLAN = architect_plan
_V281_BASE_EDITOR_IMPLEMENT = editor_implement


def prefetch_plan(message: str, skill: str = "general") -> dict[str, str]:
    result = dict(_V281_BASE_PREFETCH_PLAN(message, skill))
    if skill == "coder":
        contract = production_scope_contract(message)
        if contract:
            result["production_scope"] = contract
    return result


def architect_plan(message: str) -> str:
    base = _V281_BASE_ARCHITECT_PLAN(message)
    contract = production_scope_contract(message)

    if not contract or explicit_educational_request(message):
        return base

    return (
        "0. Lock production scope before implementation: do not substitute "
        "a toy, demo, educational parser, or in-memory imitation for the "
        "requested system.\n"
        + base
    )


def editor_implement(
    plan: str,
    message: str,
    system: str,
    history: list,
    max_tokens: int,
    *,
    build_chatml,
    generate_sync,
) -> str:
    contract = production_scope_contract(message)
    hardened_system = system

    if contract:
        hardened_system = (
            system.rstrip()
            + "\n\n"
            + contract
            + "\n\n"
            "Keep deliberation private. Output the implementation and concise "
            "verification evidence, not scope brainstorming or internal "
            "reasoning."
        )

    return _V281_BASE_EDITOR_IMPLEMENT(
        plan,
        message,
        hardened_system,
        history,
        max_tokens,
        build_chatml=build_chatml,
        generate_sync=generate_sync,
    )
# END PRODUCTION SCOPE GUARD V28.1
