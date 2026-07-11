"""Evidence-based production-readiness checks for generated code responses."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Iterable

PRODUCTION_CODE_CONTRACT = r"""
## PRODUCTION EVIDENCE CONTRACT

Never call code "production-grade", "production-ready", "enterprise-grade",
"ACID", "durable", "thread-safe", or O(log N) unless the response includes
implementation evidence and tests for every claim.

For storage engines and databases:
- Define the exact supported SQL grammar and reject everything outside it.
- Do not claim ACID unless BEGIN, COMMIT, ROLLBACK, atomic commit records,
  isolation behavior, and crash recovery are actually implemented and tested.
- A WAL needs framed records, validation/checksums, transaction identifiers,
  commit markers, idempotent replay, torn-write handling, and recovery tests.
- A B-tree must implement node splitting and rebalancing. Otherwise call it a
  sorted in-memory index and do not claim B-tree complexity.
- Network servers need bounded concurrency, input limits, framing, timeouts,
  authentication/TLS strategy, graceful shutdown, structured logging, and
  configuration outside source code.
- Never deserialize untrusted data with eval, exec, pickle, or bare exceptions.
- Include executable tests for happy path, invalid input, concurrent access,
  crash recovery, truncated/corrupt persistence records, and regression cases.

If the requested scope cannot honestly be completed in one response, provide a
production-oriented reference implementation, clearly list what is missing,
and do not label it production-ready. Evidence beats confident wording.
""".strip()

_PRODUCTION_REQUEST = re.compile(
    r"\b("
    r"production(?:[- ]grade|[- ]ready)?|enterprise(?:[- ]grade)?|"
    r"ship(?:pable)?|deploy(?:able)?|industrial[- ]grade|"
    r"acid|durab(?:le|ility)|write[- ]ahead log|wal|"
    r"database engine|sql database|network server"
    r")\b",
    re.IGNORECASE,
)

_PRODUCTION_CLAIM = re.compile(
    r"\b("
    r"production(?:[- ]grade|[- ]ready)?|enterprise(?:[- ]grade)?|"
    r"industrial[- ]grade|fully durable|acid compliant|"
    r"ensures? durability|thread[- ]safe|o\(log\s*n\)"
    r")\b",
    re.IGNORECASE,
)

_DEMO_LANGUAGE = re.compile(
    r"\b("
    r"demo|toy|prototype|simplified|for simplicity|for brevity|"
    r"not implemented|doesn['’]?t split|no splitting|"
    r"in a real (?:database|system)|only .* supported|"
    r"keep (?:the )?code concise"
    r")\b",
    re.IGNORECASE,
)

_CODE_BLOCK = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ProductionAudit:
    """Result of checking whether production claims have implementation proof."""

    required: bool
    approved: bool
    score: int
    violations: tuple[str, ...]
    evidence: tuple[str, ...]


def production_requested(request: str) -> bool:
    """Return whether the request explicitly asks for production behavior."""
    return bool(_PRODUCTION_REQUEST.search(request or ""))


def _python_blocks(response: str) -> list[str]:
    blocks = [block.strip() for block in _CODE_BLOCK.findall(response or "")]
    if blocks:
        return blocks
    text = response or ""
    if re.search(r"(?m)^\s*(?:from|import|class|def)\s+", text):
        return [text]
    return []


def _ast_signals(blocks: Iterable[str]) -> tuple[set[str], list[str]]:
    signals: set[str] = set()
    errors: list[str] = []

    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError as exc:
            errors.append(
                f"Generated Python does not parse: line {exc.lineno}: {exc.msg}."
            )
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                signals.add("bare_except")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {"eval", "exec"}:
                        signals.add(node.func.id)
                    elif node.func.id == "print":
                        signals.add("print")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"start", "accept", "bind", "listen"}:
                        signals.add(node.func.attr)
                    elif node.func.attr in {"fsync", "flush"}:
                        signals.add(node.func.attr)
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == "Thread":
                        signals.add("thread_subclass")
                    elif isinstance(base, ast.Name) and base.id == "Thread":
                        signals.add("thread_subclass")
    return signals, errors


def _count_tests(text: str) -> int:
    return len(re.findall(r"(?m)^\s*def\s+test_[A-Za-z0-9_]+\s*\(", text))


def audit_production_response(request: str, response: str) -> ProductionAudit:
    """Audit generated code and prose for unsupported production claims."""
    request = request or ""
    response = response or ""
    required = production_requested(request) or bool(_PRODUCTION_CLAIM.search(response))
    if not required:
        return ProductionAudit(
            required=False,
            approved=True,
            score=100,
            violations=(),
            evidence=("No production-readiness claim was requested or made.",),
        )

    lower = response.lower()
    blocks = _python_blocks(response)
    code = "\n\n".join(blocks)
    signals, syntax_errors = _ast_signals(blocks)

    violations: list[str] = []
    evidence: list[str] = []
    penalty = 0

    def add(message: str, points: int) -> None:
        nonlocal penalty
        if message not in violations:
            violations.append(message)
            penalty += points

    for error in syntax_errors:
        add(error, 40)

    if not blocks:
        add(
            "Production implementation has no extractable executable code.",
            40,
        )
    else:
        evidence.append(f"Found {len(blocks)} executable Python block(s).")

    demo_match = _DEMO_LANGUAGE.search(response)
    if demo_match:
        add(
            "The response contradicts its production claim with demo/prototype "
            f"language ({demo_match.group(0)!r}).",
            35,
        )

    if "eval" in signals or re.search(r"\beval\s*\(", code):
        add(
            "Unsafe deserialization/execution: eval() appears in the implementation.",
            45,
        )
    if "exec" in signals or re.search(r"\bexec\s*\(", code):
        add(
            "Unsafe dynamic execution: exec() appears in the implementation.",
            45,
        )
    if "bare_except" in signals or re.search(r"(?m)^\s*except\s*:\s*$", code):
        add(
            "Bare except hides corruption and recovery failures.",
            25,
        )

    claims_acid = bool(re.search(r"\bacid\b|atomicity|rollback", response, re.I))
    if claims_acid:
        transaction_evidence = all(
            len(re.findall(term, lower)) >= 2
            for term in ("begin", "commit", "rollback")
        )
        has_commit_record = bool(
            re.search(r"commit[_ -]?(?:record|marker)|transaction[_ -]?id|\btxid\b", lower)
        )
        if not transaction_evidence or not has_commit_record:
            add(
                "ACID/transaction claim lacks an implemented transaction state "
                "machine with BEGIN, COMMIT, ROLLBACK, transaction IDs, and durable "
                "commit records.",
                40,
            )

    claims_wal = bool(re.search(r"write[- ]ahead|\bwal\b|durab", response, re.I))
    if claims_wal:
        framed = bool(
            re.search(r"checksum|crc|length[-_ ]prefix|record[_ -]length", lower)
        )
        torn_write = bool(re.search(r"torn|partial record|truncated|corrupt", lower))
        idempotent = bool(re.search(r"idempotent|sequence number|\blsn\b", lower))
        if not (framed and torn_write and idempotent):
            add(
                "WAL durability claim lacks framed/checksummed records, torn-write "
                "handling, and idempotent replay evidence.",
                35,
            )

    claims_btree = bool(re.search(r"\bb[- ]?tree\b|o\(log\s*n\)", response, re.I))
    if claims_btree:
        has_split = bool(re.search(r"\bsplit(?:_child|_node)?\b", code, re.I))
        has_rebalance = bool(re.search(r"\bmerge\b|\brebalanc", code, re.I))
        if not has_split or not has_rebalance:
            add(
                "B-tree/O(log N) claim is unsupported because node splitting and "
                "rebalancing are not implemented.",
                35,
            )

    network_server = bool(
        re.search(r"\bsocket\b|\bserver\b", request + "\n" + response, re.I)
        and re.search(r"\.accept\s*\(|\.listen\s*\(|\.bind\s*\(", code)
    )
    if network_server:
        bounded = bool(
            re.search(r"ThreadPoolExecutor|Semaphore|max_workers|connection_limit", code)
        )
        if (
            ("thread_subclass" in signals or ".start(" in code)
            and not bounded
        ):
            add(
                "The server creates unbounded per-connection threads; production "
                "concurrency must be bounded.",
                30,
            )
        has_security = bool(
            re.search(r"\bssl\b|\btls\b|auth|token|credential|certificate", lower)
        )
        if "0.0.0.0" in response and not has_security:
            add(
                "The server binds publicly without authentication or a TLS strategy.",
                30,
            )
        has_limits = bool(
            re.search(r"max_(?:request|frame|message|connection)|settimeout|timeout=", code)
        )
        if not has_limits:
            add(
                "The network protocol lacks explicit timeouts and input/frame limits.",
                25,
            )

    if ("print" in signals or re.search(r"\bprint\s*\(", code)) and not re.search(
        r"\blogging\b|structlog", code
    ):
        add(
            "Server/library code uses print() without structured logging.",
            15,
        )

    test_count = _count_tests(response)
    if test_count < 3:
        add(
            f"Only {test_count} executable test(s) found; production claims require "
            "at least happy-path, failure/recovery, and concurrency/regression tests.",
            30,
        )
    else:
        evidence.append(f"Found {test_count} executable test(s).")

    if re.search(r"\b(DATA_FILE|WAL_FILE|DEFAULT_PORT)\s*=", code) and not re.search(
        r"os\.environ|getenv|BaseSettings|argparse", code
    ):
        add(
            "Operational configuration is hardcoded instead of externalized.",
            15,
        )

    score = max(0, 100 - penalty)
    approved = not violations and score >= 90
    if approved:
        evidence.append("No unsupported production claims were detected.")

    return ProductionAudit(
        required=True,
        approved=approved,
        score=score,
        violations=tuple(violations),
        evidence=tuple(evidence),
    )


def format_audit_for_model(report: ProductionAudit) -> str:
    """Format concise correction instructions for a failed audit."""
    if report.approved:
        return "Production evidence gate: PASS"
    issues = "\n".join(
        f"- {item}" for item in report.violations[:8]
    )
    return (
        f"Production evidence gate: FAIL (score {report.score}/100)\n"
        f"{issues}\n"
        "Rewrite the implementation or explicitly downgrade its claims. Do not "
        "hide limitations behind confident wording."
    )
