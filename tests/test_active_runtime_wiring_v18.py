from pathlib import Path


def test_active_app_installs_quality_kernel_after_routes():
    text = Path("app.py").read_text(encoding="utf-8")
    marker = "BEGIN ACTIVE QUALITY KERNEL V18"
    assert marker in text
    assert text.rfind(marker) > text.find('@app.post("/stream")')
    assert "install_runtime_hooks(globals())" in text


def test_active_sandbox_has_no_runtime_auto_install_path():
    text = Path("modules/services/code_sandbox.py").read_text(
        encoding="utf-8"
    )
    assert "BEGIN SAFE SANDBOX OVERRIDES V18" in text
    tail = text.split("BEGIN SAFE SANDBOX OVERRIDES V18", 1)[1]
    assert "shell=True" not in tail
    assert "pip\", \"install" not in tail


def test_active_router_is_environment_configurable():
    text = Path("modules/reliability.py").read_text(encoding="utf-8")
    assert "BEGIN ACTIVE ROUTER V18" in text
    assert "ELITE_MODEL_CODER" in text
    assert "ELITE_MODEL_RESEARCH" in text


def test_audit_document_exists():
    assert Path("docs/ACTIVE_RUNTIME_AUDIT.md").is_file()
