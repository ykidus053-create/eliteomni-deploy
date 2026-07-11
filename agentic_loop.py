"""Safe, structured agentic coding loop.

The model may inspect and modify only files inside the configured repository root.
All actions are structured JSON; no model-generated shell command is executed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from task_state import get_global_state, reset_global_state
from self_verify import compounding_error_guard, confidence_score, validate_upstream

log = logging.getLogger(__name__)

MAX_ACTIONS_PER_TURN = 6
MAX_ACTION_OUTPUT = 12_000
MAX_FILE_BYTES = 1_000_000
DEFAULT_COMMAND_TIMEOUT = 120

SYSTEM_FRAMEWORK_TEMPLATE = """# ROLE
You are a production software-engineering agent operating inside one repository.
Use concise planning, inspect before editing, make the smallest correct change, and verify every modification.

# ACTION PROTOCOL
Return one or more lines beginning exactly with `ACTION: ` followed by one compact JSON object.
Do not emit shell commands. Do not wrap ACTION lines in Markdown fences.

Available tools:
1. {{"tool":"read","path":"relative/file.py","start_line":1,"end_line":240}}
2. {{"tool":"search","query":"symbol","path":".","glob":"*.py"}}
3. {{"tool":"write","path":"relative/file.py","content":"full content","overwrite":false}}
4. {{"tool":"replace","path":"relative/file.py","old":"exact text","new":"replacement","count":1}}
5. {{"tool":"test","args":["-q","tests/test_file.py"]}}
6. {{"tool":"lint","args":["relative/file.py"]}}
7. {{"tool":"git_diff"}}
8. {{"tool":"finish","summary":"what changed and what passed"}}

Rules:
- Paths must be relative to the repository. Never access parent directories or absolute paths.
- Read/search before write/replace.
- Use `replace` for surgical edits and `write` mainly for new files.
- After any modification, run test or lint successfully before finish.
- Do not issue `finish` with another action.
- Maximum {max_actions} actions per turn.

{task_state_section}

## RECENT EXECUTION HISTORY
{history}

## ORIGINAL TASK
{task}
"""

CODER_SYSTEM = """You are a careful autonomous coding agent. Use only the structured ACTION protocol supplied by the user prompt. Never output or request arbitrary shell execution. Preserve existing behavior unless the task requires a change. Verify modifications before finishing."""


class ActionError(RuntimeError):
    """Raised when an action is invalid or unsafe."""


@dataclass(frozen=True)
class ProcessResult:
    """Result of an allowed subprocess invocation."""

    returncode: int
    stdout: str
    stderr: str


class ActionExecutor:
    """Execute structured repository-scoped tools without using a shell."""

    def __init__(self, root: Path | str, timeout: int = DEFAULT_COMMAND_TIMEOUT) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repository root does not exist: {self.root}")
        self.timeout = max(1, min(int(timeout), 600))
        self.modified_files: set[str] = set()
        self.verification_passed = False

    def execute(self, actions: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Execute actions sequentially and return combined output plus statuses."""
        if not actions:
            raise ActionError("No actions supplied")
        if len(actions) > MAX_ACTIONS_PER_TURN:
            raise ActionError(f"Too many actions; maximum is {MAX_ACTIONS_PER_TURN}")
        if any(a.get("tool") == "finish" for a in actions) and len(actions) != 1:
            raise ActionError("finish must be the only action in its turn")

        statuses: List[Dict[str, Any]] = []
        blocks: List[str] = []
        for action in actions:
            tool = action.get("tool")
            try:
                output = self._dispatch(action)
                status = {
                    "tool": tool,
                    "success": True,
                    "output": output,
                    "returncode": 0,
                }
            except Exception as exc:
                output = f"{type(exc).__name__}: {exc}"
                status = {
                    "tool": tool,
                    "success": False,
                    "output": output,
                    "returncode": 1,
                }
            statuses.append(status)
            blocks.append(f"TOOL: {tool}\nSUCCESS: {status['success']}\nOUTPUT:\n{self._truncate(output)}")
        return "\n---\n".join(blocks), statuses

    def _dispatch(self, action: Dict[str, Any]) -> str:
        tool = action.get("tool")
        handlers = {
            "read": self._read,
            "search": self._search,
            "write": self._write,
            "replace": self._replace,
            "test": self._test,
            "lint": self._lint,
            "git_diff": self._git_diff,
            "finish": self._finish,
        }
        if tool not in handlers:
            raise ActionError(f"Unknown tool: {tool!r}")
        return handlers[tool](action)

    def _resolve(self, raw_path: Any, *, must_exist: bool = False) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ActionError("path must be a non-empty string")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise ActionError("absolute paths are forbidden")
        resolved = (self.root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ActionError("path escapes repository root") from exc
        if must_exist and not resolved.exists():
            raise ActionError(f"path does not exist: {raw_path}")
        return resolved

    @staticmethod
    def _require_int(value: Any, name: str, default: int) -> int:
        if value is None:
            return default
        if not isinstance(value, int):
            raise ActionError(f"{name} must be an integer")
        return value

    def _read(self, action: Dict[str, Any]) -> str:
        path = self._resolve(action.get("path"), must_exist=True)
        if not path.is_file():
            raise ActionError("read target must be a file")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ActionError(f"file exceeds {MAX_FILE_BYTES} bytes")
        start = max(1, self._require_int(action.get("start_line"), "start_line", 1))
        end = self._require_int(action.get("end_line"), "end_line", start + 239)
        if end < start or end - start > 500:
            raise ActionError("line range must be ordered and at most 501 lines")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start - 1 : end]
        return "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start=start))

    def _search(self, action: Dict[str, Any]) -> str:
        query = action.get("query")
        if not isinstance(query, str) or not query:
            raise ActionError("query must be a non-empty string")
        base = self._resolve(action.get("path", "."), must_exist=True)
        glob = action.get("glob", "*")
        if not isinstance(glob, str) or not glob:
            raise ActionError("glob must be a non-empty string")

        rg = shutil.which("rg")
        if rg:
            result = self._run_process(
                [rg, "-n", "--hidden", "--glob", "!.git/**", "--glob", glob, "--", query, str(base)],
                timeout=min(self.timeout, 60),
            )
            if result.returncode not in (0, 1):
                raise ActionError(result.stderr or "search failed")
            return result.stdout or "No matches"

        matches: List[str] = []
        paths = [base] if base.is_file() else base.rglob(glob)
        for path in paths:
            if len(matches) >= 200:
                break
            if not path.is_file() or ".git" in path.parts or path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if query in line:
                        matches.append(f"{path.relative_to(self.root)}:{number}:{line[:500]}")
                        if len(matches) >= 200:
                            break
            except OSError:
                continue
        return "\n".join(matches) if matches else "No matches"

    def _write(self, action: Dict[str, Any]) -> str:
        path = self._resolve(action.get("path"))
        content = action.get("content")
        overwrite = action.get("overwrite", False)
        if not isinstance(content, str):
            raise ActionError("content must be a string")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ActionError(f"content exceeds {MAX_FILE_BYTES} bytes")
        if path.exists() and not overwrite:
            raise ActionError("target exists; use replace or set overwrite=true")
        self._atomic_write(path, content)
        self._mark_modified(path)
        return f"Wrote {path.relative_to(self.root)} ({len(content.encode('utf-8'))} bytes)"

    def _replace(self, action: Dict[str, Any]) -> str:
        path = self._resolve(action.get("path"), must_exist=True)
        if not path.is_file():
            raise ActionError("replace target must be a file")
        old = action.get("old")
        new = action.get("new")
        count = self._require_int(action.get("count"), "count", 1)
        if not isinstance(old, str) or not old:
            raise ActionError("old must be a non-empty string")
        if not isinstance(new, str):
            raise ActionError("new must be a string")
        if count < 1 or count > 100:
            raise ActionError("count must be between 1 and 100")
        original = path.read_text(encoding="utf-8")
        occurrences = original.count(old)
        if occurrences != count:
            raise ActionError(f"expected {count} occurrence(s), found {occurrences}; re-read the file")
        updated = original.replace(old, new, count)
        self._atomic_write(path, updated)
        self._mark_modified(path)
        return f"Replaced {count} occurrence(s) in {path.relative_to(self.root)}"

    def _test(self, action: Dict[str, Any]) -> str:
        args = self._validate_args(action.get("args", ["-q"]))
        result = self._run_process([sys.executable, "-m", "pytest", *args])
        output = self._format_process(result)
        if result.returncode != 0:
            raise ActionError(output)
        self.verification_passed = True
        return output or "pytest passed"

    def _lint(self, action: Dict[str, Any]) -> str:
        args = self._validate_args(action.get("args", ["."]))
        result = self._run_process([sys.executable, "-m", "ruff", "check", *args])
        output = self._format_process(result)
        if result.returncode != 0:
            raise ActionError(output)
        self.verification_passed = True
        return output or "ruff passed"

    def _git_diff(self, action: Dict[str, Any]) -> str:
        del action
        status = self._run_process(["git", "status", "--short"], timeout=30)
        diff = self._run_process(["git", "diff", "--stat"], timeout=30)
        if status.returncode != 0 or diff.returncode != 0:
            raise ActionError(self._format_process(status) + self._format_process(diff))
        return f"STATUS:\n{status.stdout or '(clean)'}\nDIFF STAT:\n{diff.stdout or '(none)'}"

    def _finish(self, action: Dict[str, Any]) -> str:
        summary = action.get("summary", "").strip()
        if self.modified_files and not self.verification_passed:
            raise ActionError("cannot finish after modifications until test or lint passes")
        if not summary:
            raise ActionError("finish requires a non-empty summary")
        files = ", ".join(sorted(self.modified_files)) or "none"
        return f"FINISH\nSummary: {summary}\nModified files: {files}\nVerified: {self.verification_passed}"

    def _validate_args(self, value: Any) -> List[str]:
        if not isinstance(value, list) or len(value) > 30:
            raise ActionError("args must be a list with at most 30 items")
        args: List[str] = []
        for item in value:
            if not isinstance(item, str) or "\x00" in item or len(item) > 500:
                raise ActionError("each argument must be a safe string")
            args.append(item)
        return args

    def _run_process(self, command: Sequence[str], timeout: Optional[int] = None) -> ProcessResult:
        try:
            completed = subprocess.run(
                list(command),
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout or self.timeout,
                check=False,
                env={**os.environ, "PYTHONPATH": str(self.root)},
            )
        except subprocess.TimeoutExpired as exc:
            raise ActionError(f"command timed out after {timeout or self.timeout}s") from exc
        except OSError as exc:
            raise ActionError(f"unable to run {command[0]}: {exc}") from exc
        return ProcessResult(completed.returncode, completed.stdout, completed.stderr)

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = path.stat().st_mode if path.exists() else None
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        try:
            os.replace(temp_path, path)
            if mode is not None:
                os.chmod(path, mode)
        finally:
            temp_path.unlink(missing_ok=True)

    def _mark_modified(self, path: Path) -> None:
        self.modified_files.add(str(path.relative_to(self.root)))
        self.verification_passed = False

    @staticmethod
    def _format_process(result: ProcessResult) -> str:
        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        parts.append(f"RETURN CODE: {result.returncode}")
        return "\n".join(parts)

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_ACTION_OUTPUT:
            return text
        return text[:MAX_ACTION_OUTPUT] + "\n... [TRUNCATED]"


def extract_actions(output: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse only exact ACTION lines containing JSON objects."""
    actions: List[Dict[str, Any]] = []
    errors: List[str] = []
    for number, raw_line in enumerate(output.splitlines(), 1):
        if not raw_line.startswith("ACTION: "):
            continue
        payload = raw_line[len("ACTION: ") :].strip()
        try:
            action = json.loads(payload)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(action, dict):
            errors.append(f"line {number}: action must be a JSON object")
            continue
        actions.append(action)
    return actions, errors


def _call_mistral(
    prompt: str,
    system: str = CODER_SYSTEM,
    model: Optional[str] = None,
    max_tokens: int = 4000,
) -> str:
    """Call Mistral with bounded retries, timeouts, and response validation."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    chosen_model = model or os.getenv("AGENT_MODEL", "codestral-latest")
    bounded_tokens = max(512, min(int(max_tokens), 12_000))
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    response = session.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": bounded_tokens,
        },
        timeout=(10, 120),
    )
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Mistral returned an unexpected response shape") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Mistral returned an empty response")
    return content


def run_agentic_task(
    task_description: str,
    max_turns: int = 20,
    repo_root: Path | str = ".",
) -> Dict[str, Any]:
    """Run the structured agent loop and preserve the existing result contract."""
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("task_description must be a non-empty string")
    task_description = task_description.strip()
    if len(task_description) > 20_000:
        raise ValueError("task_description exceeds 20,000 characters")
    max_turns = max(1, min(int(max_turns), 50))

    state = reset_global_state()
    state.add_objective(task_description)
    executor = ActionExecutor(repo_root)
    history_turns: List[str] = []
    escalation: Optional[str] = None
    final_output = ""
    turns_used = 0

    for turn in range(max_turns):
        turns_used = turn + 1
        full_prompt = SYSTEM_FRAMEWORK_TEMPLATE.format(
            max_actions=MAX_ACTIONS_PER_TURN,
            task_state_section=state.to_prompt_section(),
            history="\n".join(history_turns[-10:]),
            task=task_description,
        )
        try:
            output = _call_mistral(full_prompt)
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            state.record_step("llm_call", success=False, confidence=0.0)
            history_turns.append(f"LLM ERROR: {type(exc).__name__}: {exc}")
            if state.should_escalate():
                escalation = f"LLM call failed repeatedly: {exc}"
                break
            continue

        actions, parse_errors = extract_actions(output)
        history_turns.append(f"AGENT OUTPUT:\n{output}")
        if parse_errors:
            history_turns.append("ACTION PARSE ERRORS:\n" + "\n".join(parse_errors))
        if not actions:
            state.record_step("no_valid_action", success=False, confidence=0.2)
            if state.should_escalate():
                escalation = "The model repeatedly failed to produce valid structured actions"
                break
            continue

        try:
            combined_results, statuses = executor.execute(actions)
        except ActionError as exc:
            combined_results = f"ActionError: {exc}"
            statuses = [{"tool": "batch", "success": False, "output": combined_results, "returncode": 1}]

        history_turns.append(f"RESULTS OF ACTIONS:\n{combined_results}")
        for status in statuses:
            is_safe, conf, reason = validate_upstream(status["output"], min_confidence=0.3)
            success = bool(status["success"]) and is_safe
            state.record_step(
                action=str(status.get("tool", "unknown"))[:80],
                success=success,
                confidence=conf if success else 0.2,
            )
            if not is_safe:
                log.warning("Low-confidence tool output for %s: %s", status.get("tool"), reason)

        finish_status = next((s for s in statuses if s.get("tool") == "finish"), None)
        if finish_status and finish_status["success"]:
            state.complete_objective(task_description)
            final_output = finish_status["output"]
            break

        if turn > 0 and turn % 3 == 0:
            result_history = [h for h in history_turns if h.startswith("RESULTS OF ACTIONS")]
            guard = compounding_error_guard(result_history, max_consecutive_low_confidence=3)
            if guard.get("should_escalate"):
                escalation = str(guard.get("reason", "Compounding errors detected"))
                state.add_open_question("Re-plan needed due to compounding errors")
                break

        if state.should_escalate():
            escalation = f"State escalation: failed_steps={len(state.failed_steps)}, confidence={state.confidence:.2f}"
            break

        if len(history_turns) > 24:
            history_turns = history_turns[:2] + ["[COMPACTED]"] + history_turns[-18:]

    if not final_output:
        final_output = history_turns[-1] if history_turns else ""
    return {
        "task_state": state.to_dict(),
        "history": history_turns,
        "escalation": escalation,
        "final_output": final_output,
        "turns_used": turns_used,
        "modified_files": sorted(executor.modified_files),
        "verification_passed": executor.verification_passed,
    }


def inject_state_into_system(system_prompt: str) -> str:
    """Append current task state to a system prompt."""
    try:
        state_section = get_global_state().to_prompt_section()
        return system_prompt + "\n\n" + state_section if state_section else system_prompt
    except Exception as exc:
        log.debug("inject_state_into_system failed: %s", exc)
        return system_prompt


def record_turn(
    user_msg: str,
    assistant_response: str,
    tool_outputs: Optional[List[str]] = None,
) -> None:
    """Record a chat turn in the global task state."""
    state = get_global_state()
    state.record_step(
        action=f"chat: {user_msg[:60]}",
        success=True,
        confidence=confidence_score(assistant_response),
    )
    for output in tool_outputs or []:
        is_safe, conf, _ = validate_upstream(output)
        state.record_step(action="tool_output", success=is_safe, confidence=conf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python3 agentic_loop.py <task_description>", file=sys.stderr)
        raise SystemExit(2)
    result = run_agentic_task(sys.argv[1])
    print(json.dumps(result, indent=2))
