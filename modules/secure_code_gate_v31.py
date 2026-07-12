"""Fail-closed syntax, security, and reliability gate for generated code."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


_FENCE_RE = re.compile(
    r"```(?P<info>[^\n`]*)\n(?P<code>.*?)```",
    re.DOTALL,
)
_CODE_REQUEST_RE = re.compile(
    r"\b("
    r"code|implement|build|create|write|develop|fix|debug|"
    r"refactor|function|class|script|service|api|database|"
    r"endpoint|worker|webhook|application|app"
    r")\b",
    re.IGNORECASE,
)
_PRODUCTION_RE = re.compile(
    r"\b("
    r"production|enterprise|secure|security|authentication|"
    r"authorization|database|api|service|worker|webhook|"
    r"deployment|distributed|concurrent|transaction|migration"
    r")\b",
    re.IGNORECASE,
)
_TEST_RE = re.compile(
    r"\b("
    r"pytest|unittest|test_[A-Za-z0-9_]+|describe\(|it\(|"
    r"npm (?:run )?test|pnpm (?:run )?test|go test|"
    r"cargo test|dotnet test|mvn test|gradle test"
    r")\b",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"\b("
    r"TODO|FIXME|NotImplementedError|your[_ -]code[_ -]here|"
    r"implementation omitted|rest of implementation|"
    r"left as an exercise|placeholder|changeme|your_api_key|"
    r"your_password|example\.com/api"
    r")\b",
    re.IGNORECASE,
)
_SECRET_NAME_RE = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"private[_-]?key|client[_-]?secret|auth[_-]?token)",
    re.IGNORECASE,
)
_SECRET_TEXT_RE = re.compile(
    r"(?im)^\s*(?:password|passwd|secret|api[_-]?key|"
    r"access[_-]?token|private[_-]?key|client[_-]?secret)"
    r"\s*[:=]\s*['\"](?P<value>[^'\"]{8,})['\"]"
)
_SQL_WORD_RE = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|MERGE|EXEC(?:UTE)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CodeBlock:
    index: int
    language: str
    info: str
    code: str


@dataclass(frozen=True)
class CodeIssue:
    code: str
    severity: str
    message: str
    block_index: int | None = None
    line: int | None = None


@dataclass(frozen=True)
class GateReport:
    passed: bool
    block_count: int
    errors: int
    warnings: int
    issues: tuple[CodeIssue, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "block_count": self.block_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "issues": [asdict(item) for item in self.issues],
        }


@dataclass(frozen=True)
class GateOutcome:
    text: str
    report: GateReport
    repair_rounds: int
    released: bool


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (AttributeError, TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def _normalize_language(info: str) -> str:
    first = (info or "").strip().split(maxsplit=1)
    language = first[0].lower() if first else ""
    aliases = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "node": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "sh": "bash",
        "shell": "bash",
        "zsh": "bash",
        "yml": "yaml",
        "postgres": "sql",
        "postgresql": "sql",
        "mysql": "sql",
        "tsql": "sql",
    }
    return aliases.get(language, language)


def extract_code_blocks(text: str) -> tuple[list[CodeBlock], bool]:
    blocks: list[CodeBlock] = []
    for index, match in enumerate(_FENCE_RE.finditer(text or ""), start=1):
        info = match.group("info").strip()
        blocks.append(
            CodeBlock(
                index=index,
                language=_normalize_language(info),
                info=info,
                code=match.group("code").strip("\n"),
            )
        )
    fence_count = (text or "").count("```")
    return blocks, fence_count % 2 == 0


def _issue(
    issues: list[CodeIssue],
    code: str,
    severity: str,
    message: str,
    block: CodeBlock | None = None,
    line: int | None = None,
) -> None:
    issues.append(
        CodeIssue(
            code=code,
            severity=severity,
            message=message,
            block_index=block.index if block else None,
            line=line,
        )
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keyword_bool(node: ast.Call, name: str, expected: bool) -> bool:
    for keyword in node.keywords:
        if (
            keyword.arg == name
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is expected
        ):
            return True
    return False


def _has_keyword(node: ast.Call, name: str) -> bool:
    return any(keyword.arg == name for keyword in node.keywords)


def _is_dynamic_sql(node: ast.AST) -> bool:
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Mod),
    ):
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return True
    return False


def _iter_assignments(
    tree: ast.AST,
) -> Iterable[tuple[str, ast.AST, int | None]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield target.id, node.value, getattr(node, "lineno", None)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                yield (
                    node.target.id,
                    node.value,
                    getattr(node, "lineno", None),
                )


def _scan_python(
    block: CodeBlock,
    issues: list[CodeIssue],
    production: bool,
) -> None:
    try:
        tree = ast.parse(block.code)
    except SyntaxError as exc:
        _issue(
            issues,
            "python.syntax",
            "error",
            f"Python syntax error: {exc.msg}",
            block,
            exc.lineno,
        )
        return

    async_function_depth = 0

    class Visitor(ast.NodeVisitor):
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            nonlocal async_function_depth
            async_function_depth += 1
            self._check_defaults(node)
            self.generic_visit(node)
            async_function_depth -= 1

        def visit_Call(self, node: ast.Call) -> None:
            name = _call_name(node.func)
            line = getattr(node, "lineno", None)

            if name in {"eval", "exec"}:
                _issue(
                    issues,
                    "python.dynamic_execution",
                    "error",
                    f"Unsafe dynamic execution through {name}().",
                    block,
                    line,
                )

            if name in {
                "os.system",
                "os.popen",
                "commands.getoutput",
                "commands.getstatusoutput",
            }:
                _issue(
                    issues,
                    "python.shell_execution",
                    "error",
                    f"Unsafe shell execution through {name}().",
                    block,
                    line,
                )

            if name in {
                "subprocess.run",
                "subprocess.Popen",
                "subprocess.call",
                "subprocess.check_call",
                "subprocess.check_output",
            } and _keyword_bool(node, "shell", True):
                _issue(
                    issues,
                    "python.subprocess_shell",
                    "error",
                    "subprocess shell=True enables command injection.",
                    block,
                    line,
                )

            if name in {
                "pickle.load",
                "pickle.loads",
                "dill.load",
                "dill.loads",
                "marshal.load",
                "marshal.loads",
            }:
                _issue(
                    issues,
                    "python.unsafe_deserialization",
                    "error",
                    f"Unsafe deserialization through {name}().",
                    block,
                    line,
                )

            if name == "yaml.load":
                loader_names = {
                    _call_name(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg == "Loader"
                }
                if not loader_names.intersection(
                    {"yaml.SafeLoader", "SafeLoader"}
                ):
                    _issue(
                        issues,
                        "python.yaml_unsafe_loader",
                        "error",
                        "yaml.load() must use SafeLoader or safe_load().",
                        block,
                        line,
                    )

            if name.startswith("requests.") or name in {
                "httpx.get",
                "httpx.post",
                "httpx.put",
                "httpx.patch",
                "httpx.delete",
                "httpx.request",
            }:
                if _keyword_bool(node, "verify", False):
                    _issue(
                        issues,
                        "python.tls_verification_disabled",
                        "error",
                        "TLS certificate verification is disabled.",
                        block,
                        line,
                    )
                if production and not _has_keyword(node, "timeout"):
                    _issue(
                        issues,
                        "python.http_timeout_missing",
                        "error",
                        "Production HTTP requests require an explicit timeout.",
                        block,
                        line,
                    )

            if name in {
                "ssl._create_unverified_context",
                "tempfile.mktemp",
            }:
                _issue(
                    issues,
                    "python.insecure_primitive",
                    "error",
                    f"Insecure primitive {name}() is prohibited.",
                    block,
                    line,
                )

            if name in {
                "hashlib.md5",
                "hashlib.sha1",
            }:
                _issue(
                    issues,
                    "python.weak_hash",
                    "warning",
                    f"{name} is unsuitable for passwords or signatures.",
                    block,
                    line,
                )

            if async_function_depth and name == "time.sleep":
                _issue(
                    issues,
                    "python.blocking_async_sleep",
                    "error",
                    "time.sleep() blocks the event loop; use await asyncio.sleep().",
                    block,
                    line,
                )

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"execute", "executemany", "query"}
                and node.args
                and _is_dynamic_sql(node.args[0])
            ):
                _issue(
                    issues,
                    "python.sql_injection",
                    "error",
                    "Dynamic SQL interpolation must be replaced with parameters.",
                    block,
                    line,
                )

            if name in {
                "jwt.decode",
                "jose.jwt.decode",
            }:
                for keyword in node.keywords:
                    if keyword.arg == "options" and isinstance(
                        keyword.value,
                        ast.Dict,
                    ):
                        pairs = zip(
                            keyword.value.keys,
                            keyword.value.values,
                        )
                        for key_node, value_node in pairs:
                            if (
                                _constant_string(key_node)
                                == "verify_signature"
                                and isinstance(
                                    value_node,
                                    ast.Constant,
                                )
                                and value_node.value is False
                            ):
                                _issue(
                                    issues,
                                    "python.jwt_verification_disabled",
                                    "error",
                                    "JWT signature verification is disabled.",
                                    block,
                                    line,
                                )

            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if (
                node.type is None
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
            ):
                _issue(
                    issues,
                    "python.swallowed_exception",
                    "error" if production else "warning",
                    "Bare except: pass hides failures and security errors.",
                    block,
                    getattr(node, "lineno", None),
                )
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._check_defaults(node)
            self.generic_visit(node)

        def _check_defaults(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            defaults = list(node.args.defaults) + [
                item for item in node.args.kw_defaults if item is not None
            ]
            if any(
                isinstance(item, (ast.List, ast.Dict, ast.Set))
                for item in defaults
            ):
                _issue(
                    issues,
                    "python.mutable_default",
                    "error" if production else "warning",
                    "Mutable default arguments cause cross-request state leakage.",
                    block,
                    getattr(node, "lineno", None),
                )

    Visitor().visit(tree)

    for name, value_node, line in _iter_assignments(tree):
        value = _constant_string(value_node)
        if (
            value
            and len(value) >= 8
            and _SECRET_NAME_RE.search(name)
            and not re.fullmatch(
                r"(?:change[-_ ]?me|example|placeholder|test|dummy).*",
                value,
                re.IGNORECASE,
            )
        ):
            _issue(
                issues,
                "python.hardcoded_secret",
                "error",
                f"Hardcoded credential assigned to {name!r}.",
                block,
                line,
            )

    if re.search(
        r"(?m)^\s*(?:debug|DEBUG)\s*=\s*True\s*$",
        block.code,
    ):
        _issue(
            issues,
            "python.debug_enabled",
            "error" if production else "warning",
            "Debug mode must not be enabled in production code.",
            block,
        )


def _run_syntax_command(
    command: list[str],
    code: str,
    suffix: str,
) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if not executable:
        return True, ""
    with tempfile.TemporaryDirectory(prefix="elite-code-gate-") as temp_dir:
        path = Path(temp_dir) / f"snippet{suffix}"
        path.write_text(code, encoding="utf-8")
        process = subprocess.run(
            [executable, *command[1:], str(path)],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": temp_dir,
                "TMPDIR": temp_dir,
            },
        )
        output = (process.stderr or process.stdout).strip()
        return process.returncode == 0, output[-800:]


def _scan_javascript(
    block: CodeBlock,
    issues: list[CodeIssue],
    production: bool,
) -> None:
    if block.language == "javascript":
        ok, output = _run_syntax_command(
            ["node", "--check"],
            block.code,
            ".js",
        )
        if not ok:
            _issue(
                issues,
                "javascript.syntax",
                "error",
                f"JavaScript syntax check failed: {output}",
                block,
            )

    patterns = [
        (
            r"\beval\s*\(",
            "javascript.eval",
            "eval() enables code injection.",
        ),
        (
            r"\bnew\s+Function\s*\(",
            "javascript.dynamic_function",
            "new Function() enables code injection.",
        ),
        (
            r"\b(?:child_process\.)?exec\s*\(",
            "javascript.shell_exec",
            "Shell exec must not receive untrusted input.",
        ),
        (
            r"rejectUnauthorized\s*:\s*false",
            "javascript.tls_disabled",
            "TLS certificate verification is disabled.",
        ),
        (
            r"verify_signature\s*:\s*false",
            "javascript.jwt_disabled",
            "JWT signature verification is disabled.",
        ),
    ]
    for pattern, code, message in patterns:
        if re.search(pattern, block.code, re.IGNORECASE):
            _issue(issues, code, "error", message, block)

    if production and re.search(
        r"\bfetch\s*\(",
        block.code,
    ) and not re.search(
        r"\b(?:AbortController|timeout|signal)\b",
        block.code,
        re.IGNORECASE,
    ):
        _issue(
            issues,
            "javascript.timeout_missing",
            "error",
            "Production fetch calls require cancellation or timeout handling.",
            block,
        )


def _scan_bash(
    block: CodeBlock,
    issues: list[CodeIssue],
) -> None:
    ok, output = _run_syntax_command(
        ["bash", "-n"],
        block.code,
        ".sh",
    )
    if not ok:
        _issue(
            issues,
            "bash.syntax",
            "error",
            f"Shell syntax check failed: {output}",
            block,
        )

    patterns = [
        (
            r"(?m)^\s*eval\s+",
            "bash.eval",
            "Shell eval enables command injection.",
        ),
        (
            r"\bcurl\b[^\n]*(?:\s-k\b|--insecure\b)",
            "bash.insecure_tls",
            "curl TLS verification is disabled.",
        ),
        (
            r"\bchmod\s+777\b",
            "bash.world_writable",
            "chmod 777 grants unsafe global write access.",
        ),
        (
            r"\brm\s+-rf\s+/(?:\s|$)",
            "bash.destructive_root_delete",
            "Destructive root deletion command is prohibited.",
        ),
    ]
    for pattern, code, message in patterns:
        if re.search(pattern, block.code, re.IGNORECASE):
            _issue(issues, code, "error", message, block)


def _scan_json(
    block: CodeBlock,
    issues: list[CodeIssue],
) -> None:
    try:
        json.loads(block.code)
    except json.JSONDecodeError as exc:
        _issue(
            issues,
            "json.syntax",
            "error",
            f"Invalid JSON: {exc.msg}",
            block,
            exc.lineno,
        )


def _scan_sql(
    block: CodeBlock,
    issues: list[CodeIssue],
    production: bool,
) -> None:
    text = block.code
    if text.count("'") % 2:
        _issue(
            issues,
            "sql.unbalanced_quote",
            "error",
            "SQL contains an unbalanced single quote.",
            block,
        )
    if text.count("(") != text.count(")"):
        _issue(
            issues,
            "sql.unbalanced_parenthesis",
            "error",
            "SQL contains unbalanced parentheses.",
            block,
        )
    if production and re.search(
        r"\bEXEC(?:UTE)?\s*\(\s*@",
        text,
        re.IGNORECASE,
    ):
        _issue(
            issues,
            "sql.dynamic_execution",
            "error",
            "Dynamic SQL execution requires strict parameterization.",
            block,
        )


def _scan_general(
    block: CodeBlock,
    issues: list[CodeIssue],
) -> None:
    if _PLACEHOLDER_RE.search(block.code):
        _issue(
            issues,
            "general.placeholder",
            "error",
            "Generated code contains a placeholder or omitted implementation.",
            block,
        )

    match = _SECRET_TEXT_RE.search(block.code)
    if match:
        value = match.group("value")
        if not re.fullmatch(
            r"(?:change[-_ ]?me|example|placeholder|test|dummy).*",
            value,
            re.IGNORECASE,
        ):
            _issue(
                issues,
                "general.hardcoded_secret",
                "error",
                "Generated code contains a hardcoded credential.",
                block,
            )

    if re.search(
        r"allow_origins\s*=\s*\[\s*['\"]\*['\"]\s*\]",
        block.code,
        re.IGNORECASE,
    ) and re.search(
        r"allow_credentials\s*=\s*True",
        block.code,
        re.IGNORECASE,
    ):
        _issue(
            issues,
            "general.cors_credentials_wildcard",
            "error",
            "Credentialed CORS cannot use a wildcard origin.",
            block,
        )


def inspect_generated_code(
    response: str,
    user_request: str,
) -> GateReport:
    blocks, fences_balanced = extract_code_blocks(response)
    issues: list[CodeIssue] = []
    expects_code = bool(_CODE_REQUEST_RE.search(user_request or ""))
    production = bool(_PRODUCTION_RE.search(user_request or ""))

    if not fences_balanced:
        _issue(
            issues,
            "markdown.unclosed_fence",
            "error",
            "The response contains an unclosed Markdown code fence.",
        )

    if expects_code and not blocks:
        _issue(
            issues,
            "general.code_missing",
            "error",
            "The coding response contains no fenced code block.",
        )

    max_blocks = _env_int(
        "ELITE_CODE_GATE_MAX_BLOCKS",
        12,
        1,
        40,
    )
    if len(blocks) > max_blocks:
        _issue(
            issues,
            "general.too_many_blocks",
            "error",
            f"Response contains {len(blocks)} code blocks; maximum is {max_blocks}.",
        )

    for block in blocks[:max_blocks]:
        _scan_general(block, issues)
        if block.language == "python":
            _scan_python(block, issues, production)
        elif block.language in {"javascript", "typescript"}:
            _scan_javascript(block, issues, production)
        elif block.language == "bash":
            _scan_bash(block, issues)
        elif block.language == "json":
            _scan_json(block, issues)
        elif block.language == "sql":
            _scan_sql(block, issues, production)
        elif block.language in {
            "text",
            "plaintext",
            "markdown",
            "md",
            "",
        }:
            if _SQL_WORD_RE.search(block.code):
                _scan_sql(block, issues, production)
            else:
                _issue(
                    issues,
                    "general.language_unspecified",
                    "warning",
                    "Code fence language is missing, so syntax could not be verified.",
                    block,
                )
        else:
            _issue(
                issues,
                "general.language_unverified",
                "warning",
                f"No local syntax verifier is configured for {block.language!r}.",
                block,
            )

    require_tests = (
        os.getenv("ELITE_CODE_GATE_REQUIRE_TESTS", "1") == "1"
    )
    if (
        expects_code
        and production
        and require_tests
        and not _TEST_RE.search(response or "")
    ):
        _issue(
            issues,
            "general.tests_missing",
            "error",
            "Production code must include tests or an exact test command.",
        )

    deduplicated: list[CodeIssue] = []
    seen: set[tuple] = set()
    for item in issues:
        key = (
            item.code,
            item.severity,
            item.block_index,
            item.line,
            item.message,
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)

    errors = sum(item.severity == "error" for item in deduplicated)
    warnings = sum(item.severity == "warning" for item in deduplicated)
    return GateReport(
        passed=errors == 0,
        block_count=len(blocks),
        errors=errors,
        warnings=warnings,
        issues=tuple(deduplicated),
    )


def build_repair_prompt(
    user_request: str,
    candidate: str,
    report: GateReport,
) -> str:
    issue_lines = []
    for item in report.issues:
        location = ""
        if item.block_index is not None:
            location = f" block={item.block_index}"
        if item.line is not None:
            location += f" line={item.line}"
        issue_lines.append(
            f"- [{item.severity.upper()}] {item.code}{location}: "
            f"{item.message}"
        )

    max_chars = _env_int(
        "ELITE_CODE_GATE_MAX_CANDIDATE_CHARS",
        80000,
        2000,
        200000,
    )
    return (
        "You are repairing a coding answer that failed mandatory release "
        "validation. Return a COMPLETE REPLACEMENT ANSWER, not a patch and "
        "not commentary about the audit.\n\n"
        "MANDATORY RULES:\n"
        "1. Preserve the user's requested language, framework, behavior, "
        "and public interfaces.\n"
        "2. Fix every listed syntax, security, and reliability issue.\n"
        "3. Use parameterized queries, explicit timeouts, safe "
        "deserialization, and externalized secrets where applicable.\n"
        "4. Include runnable tests for production code.\n"
        "5. Do not use TODO, pass, placeholders, omitted sections, mock "
        "implementations, or insecure shortcuts.\n"
        "6. Return balanced Markdown fences with a language on every code "
        "block.\n\n"
        f"ORIGINAL USER REQUEST:\n{user_request[:12000]}\n\n"
        "VALIDATION FAILURES:\n"
        + "\n".join(issue_lines)
        + "\n\nFAILED CANDIDATE:\n"
        + candidate[:max_chars]
    )


def _safe_failure(report: GateReport) -> str:
    failures = [
        item.message
        for item in report.issues
        if item.severity == "error"
    ][:8]
    detail = "\n".join(f"- {item}" for item in failures)
    return (
        "I could not safely release the generated code because it still "
        "failed mandatory syntax or security validation after automatic "
        "repair attempts.\n\n"
        "Remaining validation failures:\n"
        f"{detail or '- Unknown validation failure.'}\n\n"
        "No unvalidated code was returned."
    )


def enforce_secure_code_output(
    candidate: str,
    user_request: str,
    generate_fn: Callable[[str], str],
    *,
    max_rounds: int | None = None,
    fail_closed: bool | None = None,
) -> GateOutcome:
    rounds = (
        _env_int(
            "ELITE_CODE_GATE_MAX_REPAIR_ROUNDS",
            2,
            0,
            4,
        )
        if max_rounds is None
        else max(0, min(int(max_rounds), 4))
    )
    closed = (
        os.getenv("ELITE_CODE_GATE_FAIL_CLOSED", "1") == "1"
        if fail_closed is None
        else bool(fail_closed)
    )

    current = candidate or ""
    report = inspect_generated_code(current, user_request)
    if report.passed:
        return GateOutcome(
            text=current,
            report=report,
            repair_rounds=0,
            released=True,
        )

    used = 0
    for attempt in range(1, rounds + 1):
        prompt = build_repair_prompt(
            user_request,
            current,
            report,
        )
        try:
            replacement = generate_fn(prompt)
        except Exception:
            replacement = ""

        if not isinstance(replacement, str) or len(
            replacement.strip()
        ) < 40:
            break

        current = replacement.strip()
        used = attempt
        report = inspect_generated_code(
            current,
            user_request,
        )
        if report.passed:
            return GateOutcome(
                text=current,
                report=report,
                repair_rounds=used,
                released=True,
            )

    if closed:
        return GateOutcome(
            text=_safe_failure(report),
            report=report,
            repair_rounds=used,
            released=False,
        )
    return GateOutcome(
        text=current,
        report=report,
        repair_rounds=used,
        released=True,
    )


__all__ = [
    "CodeBlock",
    "CodeIssue",
    "GateOutcome",
    "GateReport",
    "build_repair_prompt",
    "enforce_secure_code_output",
    "extract_code_blocks",
    "inspect_generated_code",
]
