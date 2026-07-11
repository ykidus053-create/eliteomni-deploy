from pathlib import Path
import subprocess

import pytest

import agent_core


def test_source_contains_no_shell_execution():
    source = Path(agent_core.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "subprocess.Popen" not in source


def test_run_agent_turn_stream_rejects_empty_prompt():
    with pytest.raises(ValueError):
        agent_core.run_agent_turn_stream("   ")


def test_main_delegates_to_hardened_loop(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run(task, max_turns, repo_root):
        captured.update(
            task=task,
            max_turns=max_turns,
            repo_root=repo_root,
        )
        return {"final_output": "done", "escalation": None}

    monkeypatch.setattr(agent_core, "run_agentic_task", fake_run)

    code = agent_core.main(
        ["fix parser", "--repo", str(tmp_path), "--max-turns", "4"]
    )

    assert code == 0
    assert captured == {
        "task": "fix parser",
        "max_turns": 4,
        "repo_root": str(tmp_path),
    }
    assert '"final_output": "done"' in capsys.readouterr().out


def test_production_gates_use_argument_lists(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, *, cwd, timeout=180):
        commands.append((command, cwd, timeout))
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(agent_core, "_run", fake_run)
    monkeypatch.setattr(agent_core.shutil, "which", lambda name: "/usr/bin/ruff")

    lint, tests, diff, returncode = agent_core.run_production_gates(tmp_path)

    assert returncode == 0
    assert lint == tests == diff == "ok"
    assert all(isinstance(command, list) for command, _, _ in commands)
    assert commands[0][0][:4] == [
        agent_core.sys.executable,
        "-m",
        "ruff",
        "check",
    ]
    assert commands[1][0][:4] == [
        agent_core.sys.executable,
        "-m",
        "pytest",
        "-q",
    ]
    assert commands[2][0] == ["git", "diff", "--check"]


def test_initialize_environment_never_overwrites_existing_file(tmp_path):
    existing = tmp_path / "CLAUDE.md"
    existing.write_text("keep me", encoding="utf-8")

    created = agent_core.initialize_agentic_environment(
        "fix a bug",
        repo_root=tmp_path,
    )

    assert existing.read_text(encoding="utf-8") == "keep me"
    assert created == [tmp_path / "plan.md"]
    assert "fix a bug" in (tmp_path / "plan.md").read_text(encoding="utf-8")


def test_compaction_keeps_first_and_latest_entries():
    history = [f"turn-{number}" for number in range(12)]
    compacted = agent_core.compact_context_history(history)

    assert compacted[0] == "turn-0"
    assert compacted[-4:] == history[-4:]
    assert len(compacted) == 6
