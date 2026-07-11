"""Compatibility entry point for EliteOmni's hardened coding agent.

Historically this module parsed model-generated shell commands and executed
them with unsafe shell execution. It now delegates all repository changes to the
structured, repository-scoped executor in :mod:`agentic_loop`.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from agentic_loop import _call_mistral, run_agentic_task

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 180
_MAX_CAPTURE_CHARS = 20_000


def run_agent_turn_stream(prompt: str) -> str:
    """Generate one coding-agent turn without executing its output.

    The legacy name is retained for compatibility. The hardened agent loop
    is responsible for parsing and executing structured actions.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return _call_mistral(prompt.strip())


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run one fixed command without a shell and with bounded output."""
    if not command or not all(isinstance(part, str) and part for part in command):
        raise ValueError("command must contain non-empty string arguments")

    completed = subprocess.run(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(1, min(int(timeout), 600)),
        check=False,
    )

    if len(completed.stdout) > _MAX_CAPTURE_CHARS:
        completed.stdout = completed.stdout[:_MAX_CAPTURE_CHARS] + "\n...[truncated]"
    if len(completed.stderr) > _MAX_CAPTURE_CHARS:
        completed.stderr = completed.stderr[:_MAX_CAPTURE_CHARS] + "\n...[truncated]"
    return completed


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(result.stderr.rstrip())
    return "\n".join(parts)


def run_production_gates(
    repo_root: str | Path = ".",
) -> tuple[str, str, str, int]:
    """Run lint, tests, and Git diff checks without hiding failures."""
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")

    if shutil.which("ruff"):
        lint_command = [sys.executable, "-m", "ruff", "check", "."]
        lint = _run(lint_command, cwd=root)
        lint_text = _combined_output(lint)
    else:
        lint_text = "ruff not installed; lint skipped"

    tests = _run([sys.executable, "-m", "pytest", "-q"], cwd=root)
    diff = _run(["git", "diff", "--check"], cwd=root, timeout=60)

    test_text = _combined_output(tests)
    diff_text = _combined_output(diff)
    if diff.returncode != 0:
        diff_text = (
            f"{diff_text}\nGit diff validation failed with code "
            f"{diff.returncode}"
        ).strip()

    return lint_text, test_text, diff_text, tests.returncode


def enforce_anchoring_and_minimality(
    command: str,
    combined_results: str,
    task_desc: str,
    repo_root: str | Path = ".",
) -> str:
    """Append a warning when a nominal bug fix changes an excessive amount."""
    del command
    root = Path(repo_root).resolve()
    diff = _run(["git", "diff", "--numstat"], cwd=root, timeout=30)

    added = 0
    for line in diff.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit():
            added += int(fields[0])

    if added > 50 and "fix" in task_desc.lower():
        combined_results += (
            "\n[WARNING] This bug fix adds "
            f"{added} lines. Re-check whether the change can be smaller."
        )
    return combined_results


def initialize_agentic_environment(
    task_desc: str,
    repo_root: str | Path = ".",
) -> list[Path]:
    """Create non-destructive project guidance files when they are absent."""
    if not isinstance(task_desc, str) or not task_desc.strip():
        raise ValueError("task_desc must be a non-empty string")

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")

    created: list[Path] = []
    templates = {
        "CLAUDE.md": (
            "# Project directives\n\n"
            "## Validation\n"
            "- Lint: `python3 -m ruff check .`\n"
            "- Test: `python3 -m pytest -q`\n\n"
            "## Engineering rules\n"
            "- Inspect before editing.\n"
            "- Prefer the smallest correct change.\n"
            "- Add tests for behavior changes.\n"
            "- Do not hide command failures.\n"
        ),
        "plan.md": (
            "# Execution plan\n\n"
            f"Task: {task_desc.strip()}\n\n"
            "1. Inspect the relevant code and tests.\n"
            "2. Establish a failing or baseline test.\n"
            "3. Make a focused change.\n"
            "4. Run targeted validation.\n"
        ),
    }

    for filename, content in templates.items():
        path = root / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            created.append(path)
    return created


def compact_context_history(history_list: Sequence[str]) -> list[str]:
    """Keep context bounded while retaining the first and newest entries."""
    history = list(history_list)
    if len(history) <= 10:
        return history
    return [
        history[0],
        "[System context compacted]",
        *history[-4:],
    ]


def verify_correctness_under_concurrency(
    code_content: str,
    combined_results: str,
) -> str:
    """Add focused concurrency review prompts only when relevant."""
    patterns = ("time.sleep", "global ", "threading.Lock(", "asyncio.Lock(")
    detected = [pattern for pattern in patterns if pattern in code_content]
    if not detected:
        return combined_results
    return (
        combined_results
        + "\n[CONCURRENCY REVIEW] Check idempotency, ordering, shared-state "
        f"invariants, and timeout behavior for: {detected}"
    )


def enforce_formal_concurrency_proof(
    code_content: str,
    combined_results: str,
) -> str:
    """Request a concise invariant argument for concurrency-sensitive code."""
    if not any(
        marker in code_content
        for marker in ("threading", "asyncio", "concurrent.futures")
    ):
        return combined_results
    return (
        combined_results
        + "\n[INVARIANT CHECK] State the shared-state invariant and explain "
        "why every permitted interleaving preserves it."
    )


def enforce_stream_monotonicity_proof(
    code_content: str,
    combined_results: str,
) -> str:
    """Request ordering analysis for code that handles streams or watermarks."""
    lowered = code_content.lower()
    if not any(term in lowered for term in ("stream", "watermark", "event_time")):
        return combined_results
    return (
        combined_results
        + "\n[STREAM CHECK] Explain handling of duplicate, late, missing, "
        "and out-of-order events."
    )


def apply_invariant_first_discipline(
    task_desc: str,
    turn_number: int,
    history_turns: list[str],
) -> list[str]:
    """Add one concise invariant prompt at the beginning of difficult work."""
    if turn_number == 0:
        history_turns.append(
            "SYSTEM: Define the state invariant, choose one primary "
            f"abstraction, then test it against edge cases. Task: {task_desc}"
        )
    return history_turns


class DeterministicMachine:
    """Minimal deterministic state-transition compatibility class."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = dict(state)

    def apply_mutation(self, event: dict[str, Any]) -> dict[str, Any]:
        if not self._validate_invariants(event):
            raise ValueError("state transition violates invariants")
        self.state = self._execute_state_transition(self.state, event)
        return dict(self.state)

    def _validate_invariants(self, event: dict[str, Any]) -> bool:
        return isinstance(event, dict)

    def _execute_state_transition(
        self,
        current_state: dict[str, Any],
        event: dict[str, Any],
    ) -> dict[str, Any]:
        del event
        return dict(current_state)


def enforce_complexity_immune_system(
    task_desc: str,
    current_thought: str,
    history_turns: list[str],
) -> list[str]:
    """Warn when a simple task is being inflated into infrastructure work."""
    del task_desc
    infrastructure = ("kafka", "redis", "microservice", "cluster")
    detected = [word for word in infrastructure if word in current_thought.lower()]
    if detected:
        history_turns.append(
            "SYSTEM: Re-evaluate whether a local primitive solution is enough "
            f"before introducing: {detected}"
        )
    return history_turns


def enforce_hard_convergence_protocol(
    current_thought: str,
    history_turns: list[str],
) -> list[str]:
    """Encourage one verified implementation instead of unbounded variants."""
    variant_markers = ("alternative approach", "another option", "we could also")
    if any(marker in current_thought.lower() for marker in variant_markers):
        history_turns.append(
            "SYSTEM: Select one implementation, state why, and verify it."
        )
    return history_turns


def main(argv: Iterable[str] | None = None) -> int:
    """Run the hardened repository-scoped agent from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Coding task to perform")
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root; defaults to the current directory",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum structured agent turns",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = run_agentic_task(
            args.task,
            max_turns=args.max_turns,
            repo_root=args.repo,
        )
    except Exception as exc:
        log.exception("Agent execution failed")
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1

    print(json.dumps(result, indent=2))
    return 1 if result.get("escalation") else 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    raise SystemExit(main())
