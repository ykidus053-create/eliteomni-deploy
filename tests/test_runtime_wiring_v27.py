from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_agents_exports_v27_helpers():
    source = (ROOT / "modules/services/agents.py").read_text(encoding="utf-8")
    assert "# BEGIN CODING REASONING CORE V27" in source
    assert "def knowledge_boundary_check" in source
    assert "def prefetch_plan" in source
    assert "def _truncate_msgs" in source


def test_app_has_coder_cost_guards():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "ELITE_ENABLE_AGENT_TEAM" in source
    assert "ELITE_ENABLE_SECOND_LOOP_ENGINE" in source
    assert "skill != \"coder\"" in source or "skill != 'coder'" in source


def test_true_streaming_is_default():
    source = (ROOT / "modules/quality_kernel.py").read_text(encoding="utf-8")
    assert "ELITE_BUFFERED_VERIFICATION_STREAM" in source


def test_frontier_coder_multiplication_is_opt_in():
    source = (ROOT / "modules/frontier_runtime_v21.py").read_text(
        encoding="utf-8"
    )
    assert "ELITE_FRONTIER_CODER" in source


def test_project_context_has_no_stale_home_path():
    source = (ROOT / "modules/project_context.py").read_text(encoding="utf-8")
    assert "~/eliteomni_app" not in source
    assert "ELITE_PROJECT_ROOT" in source

