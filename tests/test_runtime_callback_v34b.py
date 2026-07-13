from pathlib import Path

import constitutional_rlaif as cai


ROOT = Path(__file__).resolve().parents[1]


def test_keyword_callback():
    seen = {}

    def callback(prompt, **kwargs):
        seen.update(kwargs)
        return "ok"

    assert cai._invoke_generate_v34b(
        callback,
        "prompt",
        max_tokens=123,
    ) == "ok"
    assert seen["max_tokens"] == 123


def test_positional_callback():
    seen = {}

    def callback(prompt, budget):
        seen["budget"] = budget
        return "ok"

    assert cai._invoke_generate_v34b(
        callback,
        "prompt",
        max_tokens=321,
    ) == "ok"
    assert seen["budget"] == 321


def test_single_argument_callback():
    assert cai._invoke_generate_v34b(
        lambda prompt: "ok",
        "prompt",
        max_tokens=50,
    ) == "ok"


def test_callback_error_is_contained():
    def broken(prompt, **kwargs):
        raise RuntimeError("provider down")

    assert cai._invoke_generate_v34b(
        broken,
        "prompt",
        max_tokens=50,
    ) == ""


def test_gpt55_is_guarded():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "BEGIN OPTIONAL GPT55 ENHANCER V34B" in source
