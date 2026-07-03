"""
Smoke test for core EliteOmni tools and prompt regressions.
Runs automatically via .git/hooks/pre-commit — a failing test blocks the commit.

Manual run: python3 smoke_test.py
Exit code 0 = all passed (or skipped). Non-zero = real failure, blocks commit.
"""
import sys
import traceback

sys.path.insert(0, ".")

FAILURES = []
SKIPPED = []

def check(name, fn, allow_skip_on=None):
    """allow_skip_on: tuple of exception types that should SKIP (not fail) the check —
    use for missing optional deps (e.g. playwright) unrelated to the actual code change."""
    print(f"[RUNNING] {name}...")
    try:
        result = fn()
        print(f"[PASS] {name}: {str(result)[:150]}")
        return True
    except Exception as e:
        if allow_skip_on and isinstance(e, allow_skip_on):
            print(f"[SKIP] {name}: {type(e).__name__}: {e}")
            SKIPPED.append(name)
            return True
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        FAILURES.append(name)
        return False


# ── NATIVE TOOLS (real dispatch path — what the model actually calls) ───────

def test_native_calc():
    from modules.services.tool_schemas import dispatch_tool_call
    result = dispatch_tool_call("calc", {"expression": "12 * (3 + 7) / 2"})
    assert "60" in str(result), f"calc gave wrong/unexpected result: {result}"
    return result

def test_native_search():
    from modules.services.tool_schemas import dispatch_tool_call
    result = dispatch_tool_call("search", {"query": "current weather"})
    assert result and len(str(result)) > 0, "search returned empty"
    assert "unknown tool" not in str(result).lower()
    return result

def test_native_weather():
    from modules.services.tool_schemas import dispatch_tool_call
    result = dispatch_tool_call("weather", {"location": "London"})
    assert result and len(str(result)) > 0, "weather returned empty"
    return result

def test_native_bash():
    from modules.services.tool_schemas import dispatch_tool_call
    result = dispatch_tool_call("bash", {"cmd": "echo smoke_test_ok"})
    assert "smoke_test_ok" in str(result), f"bash tool didn't echo correctly: {result}"
    return result

def test_native_fetch():
    from modules.services.tool_schemas import dispatch_tool_call
    result = dispatch_tool_call("fetch", {"url": "https://example.com"})
    assert result and len(str(result)) > 0, "fetch returned empty"
    return result

def test_native_rag_lookup():
    from modules.services.tool_schemas import dispatch_tool_call
    # Don't assert on content — just confirm it doesn't crash and returns a string
    result = dispatch_tool_call("rag_lookup", {"query": "test query smoke check"})
    assert isinstance(result, str), "rag_lookup must return a string"
    return result

def test_native_unknown_tool_handled_gracefully():
    from modules.services.tool_schemas import dispatch_tool_call
    result = dispatch_tool_call("totally_fake_tool_xyz", {})
    assert "unknown tool" in result.lower(), f"unexpected handling of unknown tool: {result}"
    return result


# ── DIRECT FUNCTION TESTS (vision, OCR, time — bypass dispatch, real API/logic) ──

def test_time_string():
    from modules.services.memory import tool_time
    s = tool_time()
    assert s and "UTC" in s, f"unexpected time string: {s!r}"
    return s

def test_vision_describe():
    from modules.core.http_client import vision_describe
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42"
        "YAAAAASUVORK5CYII="
    )
    result = vision_describe(tiny_png_b64, "what color is this image?")
    assert result and len(result) > 0, "empty vision_describe response"
    assert "error" not in result.lower()[:50], f"vision_describe returned error: {result[:200]}"
    return result

def test_ocr_document_handles_bad_input_gracefully():
    from modules.core.http_client import ocr_document
    result = ocr_document("not-valid-base64-data", "test.pdf")
    assert isinstance(result, str), "ocr_document must always return a string, never raise"
    return result

def test_dual_path_calc():
    from modules.services.tools import dual_path_calc
    result = dual_path_calc("2 + 2 * 3")
    assert isinstance(result, dict), "dual_path_calc must return a dict"
    return result

def test_tool_exec_safe_code():
    from modules.services.tools import tool_exec
    result = tool_exec("print(1 + 1)", timeout=5)
    assert "2" in str(result), f"tool_exec didn't execute correctly: {result}"
    return result

def test_tool_lint():
    from modules.services.tools import tool_lint
    result = tool_lint("x = 1\nprint(x)")
    assert isinstance(result, str), "tool_lint must return a string"
    return result


# ── PROMPT REGRESSION CHECKS (catch reverted fixes) ──────────────────────────

def test_streaming_tag_stripper_present():
    with open("app.py") as f:
        content = f.read()
    assert "<extended_thinking>" in content, "extended_thinking tag handling missing from app.py"
    assert "<extended_thinking_math>" in content, "extended_thinking_math tag handling missing"
    assert 'response = f"**🏗️ Architecture Plan:**' not in content, \
        "Architecture Plan preamble has regressed back into app.py"
    return "tag stripper + Architecture Plan removal both intact"

def test_enterprise_prompt_no_pseudocode():
    with open("modules/enterprise_code.py") as f:
        content = f.read()
    assert "write pseudocode or a outline first" not in content, \
        "pseudocode instruction has regressed back into enterprise_code.py"
    assert "SCOPE DISCIPLINE" in content, "SCOPE DISCIPLINE block missing"
    return "no pseudocode instruction, SCOPE DISCIPLINE present"

def test_session_no_brotli_without_package():
    """Guard against the Brotli regression: if br is re-added to
    Accept-Encoding, the brotli package MUST be importable, or every
    session-based call (vision, OCR, etc.) silently breaks again."""
    with open("modules/core/http_client.py") as f:
        content = f.read()
    if '"br"' in content or ", br" in content:
        try:
            import brotli  # noqa
        except ImportError:
            raise AssertionError(
                "Accept-Encoding includes 'br' but brotli package is not "
                "installed — this WILL break vision_describe/ocr_document "
                "the same way it did before. Either remove 'br' or "
                "pip install brotli."
            )
    return "Accept-Encoding/brotli consistency OK"

def test_max_t_not_accidentally_lowered():
    with open("app.py") as f:
        content = f.read()
    assert "max_t = 64000" in content or "max_t = 32000" in content, \
        "token ceiling appears to have been reverted to the old 16000-only value"
    return "token ceiling intact"


def main():
    print("=" * 70)
    print("ELITEOMNI SMOKE TEST")
    print("=" * 70)

    # Native tools — real dispatch path
    check("native tool: calc", test_native_calc)
    check("native tool: search", test_native_search)
    check("native tool: weather", test_native_weather)
    check("native tool: bash", test_native_bash)
    check("native tool: fetch", test_native_fetch)
    check("native tool: rag_lookup", test_native_rag_lookup)
    check("native tool: unknown-tool handling", test_native_unknown_tool_handled_gracefully)

    # Direct function tests
    check("time string generation", test_time_string)
    check("vision_describe (real API call)", test_vision_describe)
    check("ocr_document graceful failure on bad input", test_ocr_document_handles_bad_input_gracefully)
    check("dual_path_calc", test_dual_path_calc)
    check("tool_exec (safe code)", test_tool_exec_safe_code)
    check("tool_lint", test_tool_lint)

    # Prompt regression guards
    check("streaming tag stripper present", test_streaming_tag_stripper_present)
    check("enterprise prompt has no pseudocode instruction", test_enterprise_prompt_no_pseudocode)
    check("Brotli/Accept-Encoding consistency", test_session_no_brotli_without_package)
    check("token ceiling not reverted", test_max_t_not_accidentally_lowered)

    print("=" * 70)
    if SKIPPED:
        print(f"SKIPPED: {', '.join(SKIPPED)}")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
