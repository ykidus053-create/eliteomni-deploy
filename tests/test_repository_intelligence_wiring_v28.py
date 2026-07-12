from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v28_wired_into_v27():
    source = (
        ROOT / "modules" / "coding_reasoning_v27.py"
    ).read_text(encoding="utf-8")
    assert "# BEGIN REPOSITORY INTELLIGENCE V28" in source


def test_v28_artifacts_exist():
    assert (ROOT / "modules" / "repository_intelligence_v28.py").exists()
    assert (
        ROOT / "scripts" / "check_repository_intelligence_v28.py"
    ).exists()


def test_v28_env_documented():
    source = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ELITE_REPO_INTELLIGENCE" in source
    assert "ELITE_REPO_CONTEXT_FILES" in source
