from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_runs_v31_after_true_final_verification():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    marker = "# BEGIN SECURE CODE QUALITY GATE V31"
    assert marker in source
    assert "enforce_secure_code_output(" in source

    final_verification = source.index(
        "final = verification_pipeline("
        "final, msg, skill, complexity)"
    )
    gate = source.index(marker)
    persistence = source.index(
        'scratchpad_save(f"a_{int(time.time())}", final[:120])'
    )
    assert final_verification < gate < persistence


def test_every_streamed_coder_response_is_verified_first():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "# BEGIN STREAM CODER VERIFIED ROUTE V31" in source
    assert 'if ctx.get("skill", "") == "coder":' in source
    assert "lambda: pipeline_sync(msg, hist)" in source
    assert '"mode": "verified-v31"' in source


def test_legacy_coder_execution_results_are_not_appended():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert (
        "blocks = extract_code_blocks(final) "
        'if skill != "coder" else []'
    ) in source


def test_swarm_cannot_bypass_final_gate():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'return JSONResponse({"response": swarm_result})' not in source
    assert "final = swarm_result" in source


def test_v31_environment_is_documented():
    source = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ELITE_CODE_GATE_MAX_REPAIR_ROUNDS=2" in source
    assert "ELITE_CODE_GATE_FAIL_CLOSED=1" in source
    assert "ELITE_CODE_GATE_REQUIRE_TESTS=1" in source
