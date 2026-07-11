from pathlib import Path


def test_frontier_v21_is_installed_after_quality_kernel():
    text = Path("app.py").read_text(encoding="utf-8")
    assert "BEGIN ACTIVE QUALITY KERNEL V18" in text
    assert "BEGIN FRONTIER RUNTIME V21" in text
    assert text.rfind("BEGIN FRONTIER RUNTIME V21") > text.rfind(
        "END ACTIVE QUALITY KERNEL V18"
    )
    assert "install_frontier_runtime_v21(globals())" in text


def test_agentic_loop_preserves_all_iterations():
    text = Path("app.py").read_text(encoding="utf-8")
    assert "response_parts: list[str] = []" in text
    assert "response_parts.append(response)" in text
    assert '"\\n\\n".join(response_parts)' in text
    assert "_agentic_iters" in text


def test_duplicate_loop_engine_block_removed():
    text = Path("app.py").read_text(encoding="utf-8")
    assert text.count(
        "# ── Loop Engineering (ReAct + Reflexion + Agentic + Search)"
    ) == 0
    assert text.count(
        "# ── Loop Engine (Plan+Search+ReAct+CAI+Reflexion)"
    ) == 1
