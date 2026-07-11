from pathlib import Path

import pytest

from agentic_loop import (
    ActionExecutor,
    ProcessResult,
    SYSTEM_FRAMEWORK_TEMPLATE,
    extract_actions,
)


def test_extract_actions_is_strict() -> None:
    output = '\n'.join([
        'Thought ACTION: {"tool":"git_diff"}',
        'ACTION: {"tool":"read","path":"a.py"}',
        'ACTION: not-json',
    ])
    actions, errors = extract_actions(output)
    assert actions == [{"tool": "read", "path": "a.py"}]
    assert len(errors) == 1


def test_rejects_path_escape(tmp_path: Path) -> None:
    executor = ActionExecutor(tmp_path)
    combined, statuses = executor.execute([{"tool": "read", "path": "../secret"}])
    assert statuses[0]["success"] is False
    assert "escapes repository" in combined


def test_atomic_write_replace_and_finish_gate(tmp_path: Path) -> None:
    executor = ActionExecutor(tmp_path)
    _, statuses = executor.execute([
        {"tool": "write", "path": "pkg/a.py", "content": "x = 1\n"}
    ])
    assert statuses[0]["success"] is True
    assert (tmp_path / "pkg/a.py").read_text() == "x = 1\n"

    _, statuses = executor.execute([
        {"tool": "finish", "summary": "done"}
    ])
    assert statuses[0]["success"] is False
    assert "until test or lint passes" in statuses[0]["output"]

    _, statuses = executor.execute([
        {"tool": "replace", "path": "pkg/a.py", "old": "x = 1", "new": "x = 2", "count": 1}
    ])
    assert statuses[0]["success"] is True
    assert (tmp_path / "pkg/a.py").read_text() == "x = 2\n"


def test_successful_test_unlocks_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executor = ActionExecutor(tmp_path)
    executor.execute([{"tool": "write", "path": "a.py", "content": "x = 1\n"}])

    monkeypatch.setattr(
        executor,
        "_run_process",
        lambda command, timeout=None: ProcessResult(0, "1 passed\n", ""),
    )
    _, statuses = executor.execute([{"tool": "test", "args": ["-q"]}])
    assert statuses[0]["success"] is True
    assert executor.verification_passed is True

    _, statuses = executor.execute([{"tool": "finish", "summary": "implemented and tested"}])
    assert statuses[0]["success"] is True
    assert "FINISH" in statuses[0]["output"]


def test_replace_requires_exact_occurrence_count(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("x\nx\n")
    executor = ActionExecutor(tmp_path)
    _, statuses = executor.execute([
        {"tool": "replace", "path": "a.py", "old": "x", "new": "y", "count": 1}
    ])
    assert statuses[0]["success"] is False
    assert path.read_text() == "x\nx\n"


def test_finish_must_be_only_action(tmp_path: Path) -> None:
    executor = ActionExecutor(tmp_path)
    with pytest.raises(Exception, match="finish must be the only action"):
        executor.execute([
            {"tool": "git_diff"},
            {"tool": "finish", "summary": "done"},
        ])


def test_system_prompt_formats_with_json_examples() -> None:
    rendered = SYSTEM_FRAMEWORK_TEMPLATE.format(
        max_actions=6, task_state_section="", history="", task="fix bug"
    )
    assert '{"tool":"read"' in rendered
    assert "fix bug" in rendered
