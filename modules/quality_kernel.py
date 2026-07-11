"""Active request-quality kernel for EliteOmni.

This module intentionally sits at the serving boundary. It does not try to
replace every legacy experiment in the repository. Instead, it profiles each
request, adds a small task-specific contract, controls stale research routing,
and audits the *final* response after every legacy rewrite has completed.
"""
from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclasses.dataclass(frozen=True)
class TaskProfile:
    skill: str
    complexity: str
    needs_fresh_research: bool
    needs_calculation: bool
    needs_code_execution: bool
    production_claim: bool
    high_stakes: bool
    requires_sources: bool
    requires_buffered_verification: bool
    cache_allowed: bool


@dataclasses.dataclass(frozen=True)
class AuditIssue:
    code: str
    message: str
    severity: str = "error"


@dataclasses.dataclass(frozen=True)
class ArtifactVerification:
    attempted: bool
    syntax_ok: bool
    tests_found: bool
    tests_passed: bool
    summary: str
    details: str = ""


@dataclasses.dataclass(frozen=True)
class AnswerAudit:
    approved: bool
    issues: tuple[AuditIssue, ...]
    artifact: ArtifactVerification | None = None


_CURRENT_PROFILE: contextvars.ContextVar[TaskProfile | None] = (
    contextvars.ContextVar("eliteomni_quality_profile", default=None)
)
_INSTALLED = False

_FRESH_TERMS = re.compile(
    r"\b("
    r"latest|current|today|tonight|yesterday|this week|this month|"
    r"recent|recently|newly released|announced|breaking|live|"
    r"price|weather|forecast|score|standings|schedule|election|"
    r"president|prime minister|ceo|version|release notes|"
    r"law|regulation|policy|availability|outage"
    r")\b",
    re.IGNORECASE,
)
_RESEARCH_TERMS = re.compile(
    r"\b(research|paper|study|evidence|compare sources|fact[- ]check|"
    r"verify|investigate|literature review|deep dive)\b",
    re.IGNORECASE,
)
_CODE_TERMS = re.compile(
    r"\b(code|implement|build|debug|fix|refactor|repository|api|"
    r"database|sql|python|javascript|typescript|java|golang|rust|"
    r"docker|kubernetes|function|class|script|deployment)\b",
    re.IGNORECASE,
)
_PRODUCTION_TERMS = re.compile(
    r"\b(production(?:-grade|-ready)?|enterprise(?:-grade|-ready)?|"
    r"industrial-grade|secure|acid|durable|crash recovery|"
    r"write[- ]ahead|\bwal\b|high availability|fault tolerant|"
    r"thread[- ]safe|concurrent clients?)\b",
    re.IGNORECASE,
)
_HIGH_STAKES_TERMS = re.compile(
    r"\b(medical|diagnosis|treatment|legal|lawsuit|tax|investment|"
    r"financial advice|security vulnerability|credential|production "
    r"deployment|data loss|migration|backup|restore)\b",
    re.IGNORECASE,
)
_CALC_TERMS = re.compile(
    r"\b(calculate|compute|percentage|percent|sum|average|median|"
    r"convert|equation|interest|probability)\b|[0-9]\s*[\+\-\*/%]\s*[0-9]",
    re.IGNORECASE,
)


def _estimate_complexity(message: str, skill: str) -> str:
    words = len((message or "").split())
    hard_markers = (
        "architecture",
        "production",
        "distributed",
        "concurrency",
        "migration",
        "formal proof",
        "deep research",
        "end-to-end",
        "root cause",
        "trade-off",
        "benchmark",
    )
    lowered = (message or "").lower()
    if words > 180 or sum(marker in lowered for marker in hard_markers) >= 2:
        return "hard"
    if words > 55 or skill in {"coder", "researcher"}:
        return "medium"
    return "easy"


def needs_fresh_research(message: str) -> bool:
    """Return True only when facts are likely time-sensitive or verification is asked."""
    text = message or ""
    if not text.strip():
        return False

    # Explicit browsing requests always win.
    if re.search(
        r"\b(search|browse|look up|check online|use the web|find sources)\b",
        text,
        re.IGNORECASE,
    ):
        return True

    if _FRESH_TERMS.search(text):
        return True

    # A named year near the current era is a freshness signal.
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", text)]
    if years and max(years) >= 2024:
        return True

    # Research questions need browsing when they explicitly ask for evidence.
    if _RESEARCH_TERMS.search(text) and re.search(
        r"\b(source|citation|evidence|current|recent|latest|verify)\b",
        text,
        re.IGNORECASE,
    ):
        return True

    return False


def analyze_request(message: str, skill_hint: str | None = None) -> TaskProfile:
    text = message or ""
    lowered = text.lower()

    if skill_hint in {"coder", "researcher", "calculator", "general", "safety"}:
        skill = skill_hint
    elif _CODE_TERMS.search(text) or "```" in text:
        skill = "coder"
    elif _RESEARCH_TERMS.search(text) or needs_fresh_research(text):
        skill = "researcher"
    elif _CALC_TERMS.search(text):
        skill = "calculator"
    else:
        skill = "general"

    fresh = needs_fresh_research(text)
    production = bool(_PRODUCTION_TERMS.search(text) and skill == "coder")
    high_stakes = bool(_HIGH_STAKES_TERMS.search(text))
    complexity = _estimate_complexity(text, skill)
    sources = fresh or skill == "researcher"
    execute = skill == "coder" and (
        production
        or bool(re.search(r"\b(run|test|working|runnable|complete)\b", text, re.I))
    )
    buffered = production or high_stakes or (
        skill == "researcher" and fresh
    )

    return TaskProfile(
        skill=skill,
        complexity=complexity,
        needs_fresh_research=fresh,
        needs_calculation=bool(_CALC_TERMS.search(text)),
        needs_code_execution=execute,
        production_claim=production,
        high_stakes=high_stakes,
        requires_sources=sources,
        requires_buffered_verification=buffered,
        cache_allowed=not (fresh or production or high_stakes or skill == "coder"),
    )


def build_quality_directive(profile: TaskProfile) -> str:
    """Return one compact directive instead of stacking many unrelated prompts."""
    common = (
        "FINAL-ANSWER CONTRACT: Follow the user's constraints exactly. "
        "Do not invent evidence, execution results, sources, or capabilities. "
        "Keep internal reasoning private; provide the conclusion, assumptions, "
        "and a concise verification summary."
    )

    if profile.skill == "coder":
        coding = (
            " CODING CONTRACT: Inspect interfaces before changing them. Return "
            "complete runnable files or a precise patch, not pseudocode or "
            "placeholder bodies. Prefer proven libraries over reimplementing "
            "security, storage, parsers, transactions, or networking. Include "
            "tests for changed behavior and never call code production-ready "
            "unless the final artifact was executed and its claimed guarantees "
            "were tested."
        )
        if profile.production_claim:
            coding += (
                " PRODUCTION CLAIMS: Threat model failure paths, persistence, "
                "concurrency, rollback, recovery, input validation, observability, "
                "deployment, and upgrade behavior. Unsupported guarantees must be "
                "stated explicitly."
            )
        return common + coding

    if profile.skill == "researcher":
        return (
            common
            + " RESEARCH CONTRACT: Use current sources when freshness matters. "
            "State source dates, cite the source or URL near each material claim, "
            "separate sourced facts from inference, compare disagreements, and "
            "say when evidence is incomplete."
        )

    if profile.skill == "calculator":
        return (
            common
            + " CALCULATION CONTRACT: Show the governing equation, substitute "
            "values, compute with a tool when available, preserve units, and "
            "sanity-check the result."
        )

    return common


def _dedupe_sections(text: str) -> str:
    paragraphs = re.split(r"\n{2,}", text or "")
    seen: set[str] = set()
    output: list[str] = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        if not normalized:
            continue
        digest = hashlib.sha1(normalized[:500].encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        output.append(paragraph.strip())
    return "\n\n".join(output)


def compact_system_prompt(
    system_prompt: str,
    profile: TaskProfile,
    max_chars: int | None = None,
) -> str:
    """Deduplicate and cap prompt noise while retaining policy head and live tail."""
    cap = max_chars or int(os.getenv("ELITE_SYSTEM_PROMPT_MAX_CHARS", "32000"))
    cap = max(8000, min(cap, 80000))
    directive = build_quality_directive(profile)

    text = _dedupe_sections(system_prompt or "")
    if directive not in text:
        text = f"{text}\n\n{directive}".strip()

    if len(text) <= cap:
        return text

    marker = "\n\n...[legacy prompt context compacted by Quality V18]...\n\n"
    usable = cap - len(marker)
    head = int(usable * 0.62)
    tail = usable - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def _strip_private_reasoning(text: str) -> str:
    value = text or ""
    value = re.sub(
        r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
        "",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return value.strip()


def _code_blocks(response: str) -> list[tuple[str, str, int, int]]:
    pattern = re.compile(r"```([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)```", re.DOTALL)
    return [
        (match.group(1).lower(), match.group(2), match.start(), match.end())
        for match in pattern.finditer(response or "")
    ]


def _safe_filename(name: str, index: int) -> str:
    name = Path(name.strip()).name
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", name)
    if not name or name in {".", ".."}:
        name = f"snippet_{index}.py"
    return name[:120]


def _filename_before(response: str, start: int, index: int) -> str:
    lookback = response[max(0, start - 240):start]
    patterns = (
        r"(?:File|Path)\s*:\s*`?([A-Za-z0-9_./\-]+\.py)`?",
        r"\*\*([A-Za-z0-9_./\-]+\.py)\*\*",
        r"`([A-Za-z0-9_./\-]+\.py)`\s*$",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, lookback, re.IGNORECASE | re.MULTILINE))
        if matches:
            return _safe_filename(matches[-1].group(1), index)
    return f"snippet_{index}.py"


def _run_bounded(
    args: Sequence[str],
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(cwd),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def verify_python_artifact(response: str) -> ArtifactVerification:
    """Compile generated Python files and run included tests without installing deps."""
    blocks = _code_blocks(response)
    python_blocks = [
        (language, code, start, end)
        for language, code, start, end in blocks
        if language in {"", "py", "python", "python3"}
    ]
    if not python_blocks:
        return ArtifactVerification(
            attempted=False,
            syntax_ok=False,
            tests_found=False,
            tests_passed=False,
            summary="No verifiable Python code blocks were found.",
        )

    with tempfile.TemporaryDirectory(prefix="eliteomni_verify_") as tmp:
        root = Path(tmp)
        written: list[Path] = []
        for index, (_, code, start, _) in enumerate(python_blocks, start=1):
            filename = _filename_before(response, start, index)
            path = root / filename
            if path.exists():
                path = root / f"{path.stem}_{index}{path.suffix}"
            path.write_text(code.rstrip() + "\n", encoding="utf-8")
            written.append(path)

        compile_result = _run_bounded(
            [sys.executable, "-m", "compileall", "-q", str(root)],
            root,
            timeout=20,
        )
        if compile_result.returncode != 0:
            details = (compile_result.stderr or compile_result.stdout)[-3000:]
            return ArtifactVerification(
                attempted=True,
                syntax_ok=False,
                tests_found=False,
                tests_passed=False,
                summary="Generated Python did not compile.",
                details=details,
            )

        test_files = [
            path for path in written
            if path.name.startswith("test_")
            or "unittest.TestCase" in path.read_text(encoding="utf-8")
            or re.search(r"\bdef\s+test_", path.read_text(encoding="utf-8"))
        ]
        if not test_files:
            return ArtifactVerification(
                attempted=True,
                syntax_ok=True,
                tests_found=False,
                tests_passed=False,
                summary="Python syntax compiled, but no executable tests were included.",
            )

        pytest_result = _run_bounded(
            [sys.executable, "-m", "pytest", "-q"],
            root,
            timeout=35,
        )
        if pytest_result.returncode == 0:
            output = (pytest_result.stdout + "\n" + pytest_result.stderr).strip()
            summary_line = next(
                (
                    line.strip()
                    for line in reversed(output.splitlines())
                    if "passed" in line
                ),
                "Included tests passed.",
            )
            return ArtifactVerification(
                attempted=True,
                syntax_ok=True,
                tests_found=True,
                tests_passed=True,
                summary=summary_line[:300],
                details=output[-3000:],
            )

        # Some single-file programs intentionally expose tests through a `test`
        # command-line argument rather than pytest collection.
        if len(written) == 1 and re.search(
            r"sys\.argv.*[\"']test[\"']|unittest\.main",
            written[0].read_text(encoding="utf-8"),
            re.DOTALL,
        ):
            direct = _run_bounded(
                [sys.executable, str(written[0]), "test"],
                root,
                timeout=35,
            )
            if direct.returncode == 0:
                output = (direct.stdout + "\n" + direct.stderr).strip()
                return ArtifactVerification(
                    attempted=True,
                    syntax_ok=True,
                    tests_found=True,
                    tests_passed=True,
                    summary="Embedded test command executed successfully.",
                    details=output[-3000:],
                )

        output = (pytest_result.stdout + "\n" + pytest_result.stderr).strip()
        return ArtifactVerification(
            attempted=True,
            syntax_ok=True,
            tests_found=True,
            tests_passed=False,
            summary="Included tests failed.",
            details=output[-4000:],
        )


def _database_semantic_issues(response: str) -> list[AuditIssue]:
    code = "\n\n".join(block[1] for block in _code_blocks(response))
    combined = (response or "").lower()
    if not re.search(r"\b(database|sql|transaction|\bwal\b)\b", combined):
        return []

    issues: list[AuditIssue] = []
    if re.search(
        r"class\s+\w*RequestHandler\s*\(\s*socket\.socket\s*\)",
        code,
    ):
        issues.append(AuditIssue(
            "db.socket_subclass",
            "The request handler subclasses socket.socket instead of composing "
            "a socket or using socketserver.BaseRequestHandler.",
        ))

    shared_tx = bool(
        "self.current_tx_id" in code
        and "self.tx_buffer" in code
        and "threading.Thread" in code
        and not re.search(
            r"threading\.local|contextvars|session_state|connection_state|"
            r"transactions\s*:\s*(?:dict|Dict)",
            code,
            re.IGNORECASE,
        )
    )
    if shared_tx:
        issues.append(AuditIssue(
            "db.shared_transaction_state",
            "Transaction state is shared across client threads instead of being "
            "scoped per connection or transaction object.",
        ))

    insert = re.search(
        r"def\s+insert\s*\(.*?(?=\n\s*def\s+|\Z)",
        code,
        re.DOTALL,
    )
    rollback = re.search(
        r"def\s+rollback\s*\(.*?(?=\n\s*def\s+|\Z)",
        code,
        re.DOTALL,
    )
    if insert and rollback:
        exposes = "pk_index.insert" in insert.group(0)
        undo = re.search(
            r"\b(delete|remove|restore|undo|before_image)\b",
            rollback.group(0),
            re.IGNORECASE,
        )
        if exposes and not undo:
            issues.append(AuditIssue(
                "db.rollback_does_not_undo",
                "Uncommitted rows enter the shared index, but rollback has no "
                "corresponding undo operation.",
            ))

    if "two-phase commit" in combined and not re.search(
        r"\bprepare(?:d)?\b|PREPARE_TRANSACTION",
        code,
        re.IGNORECASE,
    ):
        issues.append(AuditIssue(
            "db.fake_two_phase_commit",
            "The answer claims two-phase commit but has no durable PREPARE "
            "phase or prepared-transaction recovery.",
        ))

    if re.search(r"\.recv\s*\(", code) and not re.search(
        r"length[_ -]?prefix|frame|delimiter|readline|"
        r"buffer.*(?:newline|\\n)|split\s*\(\s*b?[\"']\\n",
        code,
        re.IGNORECASE | re.DOTALL,
    ):
        issues.append(AuditIssue(
            "db.tcp_without_framing",
            "The TCP implementation treats recv() as a complete request and "
            "does not frame or buffer the byte stream.",
        ))

    if (
        "parts[6] != '='" in code
        and not re.search(r"re\.findall\([^)]*=", code, re.DOTALL)
    ):
        issues.append(AuditIssue(
            "db.tokenizer_missing_equals",
            "The parser expects an '=' token that its tokenizer does not emit.",
        ))

    if (
        re.search(r"json\.loads\s*\(\s*line\s*\)", code)
        and not re.search(
            r"checksum|crc|record[_ -]?length|length[_ -]?prefix",
            code,
            re.IGNORECASE,
        )
    ):
        issues.append(AuditIssue(
            "db.unframed_wal",
            "The WAL uses newline JSON without checksums or record framing, so "
            "a torn final write can break recovery.",
        ))

    return issues


def _has_sources(response: str) -> bool:
    if re.search(r"https?://\S+", response or ""):
        return True
    if re.search(r"\[[0-9]{1,3}\]", response or ""):
        return True
    return bool(re.search(
        r"(?im)^\s*(sources?|references?)\s*:",
        response or "",
    ))


def audit_answer(
    request: str,
    response: str,
    profile: TaskProfile | None = None,
) -> AnswerAudit:
    profile = profile or analyze_request(request)
    value = response or ""
    issues: list[AuditIssue] = []

    if not value.strip():
        issues.append(AuditIssue("answer.empty", "The model returned an empty response."))

    if value.count("```") % 2:
        issues.append(AuditIssue(
            "markdown.unclosed_fence",
            "The final response contains an unclosed Markdown code fence.",
        ))

    if profile.skill == "coder":
        placeholders = re.findall(
            r"(?im)^\s*(?:pass|\.\.\.|TODO\b|FIXME\b|"
            r"raise\s+NotImplementedError)\s*(?:#.*)?$",
            "\n".join(block[1] for block in _code_blocks(value)),
        )
        if placeholders:
            issues.append(AuditIssue(
                "code.placeholder",
                "The delivered code still contains placeholder implementation.",
            ))

        issues.extend(_database_semantic_issues(value))

    if profile.requires_sources and not _has_sources(value):
        issues.append(AuditIssue(
            "research.missing_sources",
            "The answer requires current research but contains no usable source "
            "citations or URLs.",
        ))

    artifact: ArtifactVerification | None = None
    if profile.skill == "coder" and _code_blocks(value):
        artifact = verify_python_artifact(value)
        if artifact.attempted and not artifact.syntax_ok:
            issues.append(AuditIssue(
                "code.syntax_failed",
                artifact.summary,
            ))
        if profile.production_claim:
            if not artifact.attempted:
                issues.append(AuditIssue(
                    "code.not_verifiable",
                    "The production implementation was not in a verifiable "
                    "Python artifact format.",
                ))
            elif not artifact.tests_found:
                issues.append(AuditIssue(
                    "code.tests_missing",
                    "The production implementation included no executable tests.",
                ))
            elif not artifact.tests_passed:
                issues.append(AuditIssue(
                    "code.tests_failed",
                    artifact.summary,
                ))

    return AnswerAudit(
        approved=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        artifact=artifact,
    )


def _failure_response(audit: AnswerAudit) -> str:
    lines = [
        "I withheld the generated answer because it failed final verification.",
        "",
        "Verification failures:",
    ]
    lines.extend(f"- {issue.message}" for issue in audit.issues[:10])
    if audit.artifact and audit.artifact.details:
        details = audit.artifact.details[-1800:]
        lines.extend(("", "Execution evidence:", "```text", details, "```"))
    lines.extend((
        "",
        "The next generation pass must correct these failures before the code "
        "or research is presented as usable.",
    ))
    return "\n".join(lines)


def finalize_response(
    request: str,
    response: str,
    profile: TaskProfile | None = None,
) -> tuple[str, AnswerAudit]:
    profile = profile or analyze_request(request)
    cleaned = _strip_private_reasoning(response)
    audit = audit_answer(request, cleaned, profile)

    if not audit.approved:
        return _failure_response(audit), audit

    if audit.artifact and audit.artifact.tests_passed:
        cleaned = (
            cleaned.rstrip()
            + "\n\n**Execution verification:** "
            + audit.artifact.summary
        )
    elif audit.artifact and audit.artifact.syntax_ok and not profile.production_claim:
        cleaned = (
            cleaned.rstrip()
            + "\n\n**Syntax verification:** Generated Python compiled successfully."
        )

    return cleaned, audit


def _install_stream_route(namespace: dict[str, Any]) -> None:
    app = namespace.get("app")
    pipeline = namespace.get("pipeline_sync")
    streaming_response = namespace.get("StreamingResponse")
    if app is None or pipeline is None or streaming_response is None:
        return

    try:
        from starlette.routing import request_response
    except Exception:
        return

    for route in getattr(app, "routes", []):
        if getattr(route, "path", None) != "/stream":
            continue
        original = getattr(route, "endpoint", None)
        if original is None or getattr(original, "_quality_v18_wrapped", False):
            continue

        async def verified_stream_endpoint(
            request,
            _original=original,
            _pipeline=pipeline,
        ):
            try:
                data = await request.json()
            except Exception:
                return await _original(request)

            message = str(data.get("message", "") or "").strip()
            history = data.get("history", []) or []
            hint = namespace.get("classify_skill")
            skill_hint = hint(message) if callable(hint) else None
            profile = analyze_request(message, skill_hint)

            if not profile.requires_buffered_verification:
                return await _original(request)

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: namespace["pipeline_sync"](message, history),
            )

            async def generator():
                yield json.dumps({
                    "skill": result.get("skill", profile.skill),
                    "mode": "verified-buffered-v18",
                    "quality": result.get("quality_v18", {}),
                }) + "\n"
                yield str(result.get("response", ""))

            return streaming_response(
                generator(),
                media_type="text/plain",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                },
            )

        verified_stream_endpoint._quality_v18_wrapped = True
        route.endpoint = verified_stream_endpoint
        route.app = request_response(verified_stream_endpoint)
        break


def install_runtime_hooks(namespace: dict[str, Any]) -> None:
    """Install wrappers around the active app globals after all routes exist."""
    global _INSTALLED
    if _INSTALLED:
        return

    original_pipeline = namespace.get("pipeline_sync")
    if callable(original_pipeline):
        def pipeline_sync_v18(message: str, history: list) -> dict:
            classifier = namespace.get("classify_skill")
            hint = classifier(message) if callable(classifier) else None
            profile = analyze_request(message, hint)
            token = _CURRENT_PROFILE.set(profile)
            try:
                result = original_pipeline(message, history)
            finally:
                _CURRENT_PROFILE.reset(token)

            if isinstance(result, Mapping):
                updated = dict(result)
                final, audit = finalize_response(
                    message,
                    str(updated.get("response", "")),
                    profile,
                )
                updated["response"] = final
                updated["quality_v18"] = {
                    "approved": audit.approved,
                    "skill": profile.skill,
                    "complexity": profile.complexity,
                    "fresh_research": profile.needs_fresh_research,
                    "issues": [issue.code for issue in audit.issues],
                    "artifact": (
                        dataclasses.asdict(audit.artifact)
                        if audit.artifact is not None
                        else None
                    ),
                }
                return updated

            final, _ = finalize_response(message, str(result), profile)
            return {"response": final, "skill": profile.skill}

        pipeline_sync_v18.__name__ = "pipeline_sync"
        pipeline_sync_v18._quality_v18_wrapped = True
        namespace["pipeline_sync"] = pipeline_sync_v18

    original_system = namespace.get("build_system_prompt")
    if callable(original_system):
        def build_system_prompt_v18(*args, **kwargs):
            result = original_system(*args, **kwargs)
            profile = _CURRENT_PROFILE.get()
            if profile is None:
                skill = str(args[0]) if args else str(kwargs.get("skill", "general"))
                profile = analyze_request("", skill)
            return compact_system_prompt(str(result), profile)

        namespace["build_system_prompt"] = build_system_prompt_v18

    original_chatml = namespace.get("build_chatml")
    if callable(original_chatml):
        def build_chatml_v18(system, history, user_msg, *args, **kwargs):
            profile = _CURRENT_PROFILE.get() or analyze_request(str(user_msg))
            system = compact_system_prompt(str(system), profile)
            return original_chatml(system, history, user_msg, *args, **kwargs)

        namespace["build_chatml"] = build_chatml_v18

    original_cache_get = namespace.get("cache_get")
    if callable(original_cache_get):
        def cache_get_v18(message, skill, *args, **kwargs):
            profile = _CURRENT_PROFILE.get() or analyze_request(str(message), str(skill))
            if not profile.cache_allowed:
                return None
            return original_cache_get(message, skill, *args, **kwargs)

        namespace["cache_get"] = cache_get_v18

    namespace["_needs_fresh_search"] = needs_fresh_research

    patterns = namespace.get("FORCE_TOOL_PATTERNS")
    if isinstance(patterns, dict):
        patterns["SEARCH"] = [
            "search",
            "look up",
            "latest",
            "news",
            "current",
            "today",
            "recent",
            "fact check",
            "verify online",
            "weather",
            "price",
            "score",
            "standings",
            "release notes",
        ]
        patterns["CALC"] = [
            "calculate",
            "compute",
            "percent",
            "sqrt",
            "sum",
            "multiply",
            "divide",
        ]

    _install_stream_route(namespace)
    namespace["QUALITY_KERNEL_V18_INSTALLED"] = True
    _INSTALLED = True
