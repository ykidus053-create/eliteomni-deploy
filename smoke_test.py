"""
Smoke test for core EliteOmni tools. Run this after ANY change to
app.py, modules/core/http_client.py, modules/services/memory.py,
modules/enterprise_code.py, or any prompt file.

Usage: python3 smoke_test.py
Exit code 0 = all passed. Non-zero = something broke.
"""
import sys
import base64
import traceback

sys.path.insert(0, ".")

FAILURES = []

def check(name, fn):
    print(f"[RUNNING] {name}...")
    try:
        result = fn()
        print(f"[PASS] {name}: {str(result)[:150]}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        FAILURES.append(name)
        return False


def test_time_string():
    from modules.services.memory import tool_time  # adjust if name differs
    s = tool_time()
    assert s and "UTC" in s, f"unexpected time string: {s!r}"
    return s


def test_vision_describe():
    from modules.core.http_client import vision_describe
    # 1x1 red pixel PNG, base64-encoded — minimal valid image input
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
    # Deliberately invalid base64/PDF — we just want it to fail CLEANLY,
    # not throw an unhandled exception that crashes the caller.
    result = ocr_document("not-valid-base64-data", "test.pdf")
    assert isinstance(result, str), "ocr_document must always return a string, never raise"
    return result


def test_streaming_tag_stripper():
    # Re-import the regex/tag-detection logic indirectly by checking
    # the actual patched constants exist where expected in app.py
    with open("app.py") as f:
        content = f.read()
    assert "<extended_thinking>" in content, "extended_thinking tag handling missing from app.py"
    assert "<extended_thinking_math>" in content, "extended_thinking_math tag handling missing"
    assert 'response = f"**🏗️ Architecture Plan:**' not in content, \
        "Architecture Plan preamble has regressed back into app.py"
    return "tag stripper constants present, Architecture Plan preamble absent"


def test_enterprise_prompt_has_no_pseudocode_instruction():
    with open("modules/enterprise_code.py") as f:
        content = f.read()
    assert "write pseudocode or a outline first" not in content, \
        "pseudocode instruction has regressed back into enterprise_code.py"
    assert "SCOPE DISCIPLINE" in content, "SCOPE DISCIPLINE block missing"
    return "no pseudocode instruction, SCOPE DISCIPLINE present"


def main():
    print("=" * 70)
    print("ELITEOMNI SMOKE TEST")
    print("=" * 70)

    check("time string generation", test_time_string)
    check("vision_describe (real API call)", test_vision_describe)
    check("ocr_document graceful failure on bad input", test_ocr_document_handles_bad_input_gracefully)
    check("streaming tag stripper code present", test_streaming_tag_stripper)
    check("enterprise prompt has no pseudocode instruction", test_enterprise_prompt_has_no_pseudocode_instruction)

    print("=" * 70)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
