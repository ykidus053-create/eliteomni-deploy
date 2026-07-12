from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_uses_platform_excellence():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "# BEGIN PLATFORM EXCELLENCE V30" in source
    assert "configure_platform_excellence(app)" in source
    assert "admin_token_valid(request)" in source


def test_insecure_defaults_are_removed():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'allow_origins=["*"]' not in source
    assert 'get("DEBUG_SECRET", "changeme")' not in source


def test_runtime_database_is_ignored():
    source = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.db" in source


def test_ci_release_gate_exists():
    assert (
        ROOT / ".github" / "workflows" / "quality-gate-v30.yml"
    ).exists()
