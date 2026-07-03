import re
import logging
from typing import Optional, Callable

log = logging.getLogger(__name__)

_CONFIDENCE_HIGH_MARKERS = ["verified", "confirmed", "tested", "validated", "proven"]
_CONFIDENCE_LOW_MARKERS = ["uncertain", "maybe", "might", "possibly", "i think", "not sure", "guess", "approximate"]
_CONFIDENCE_ERROR_MARKERS = ["error", "failed", "exception", "traceback", "undefined", "null reference"]

def confidence_score(text: str) -> float:
    """NEW: Heuristic confidence 0.0-1.0. Used by compounding-error guard."""
    if not text or len(text) < 10:
        return 0.1
    lower = text.lower()

    error_hits = sum(1 for m in _CONFIDENCE_ERROR_MARKERS if m in lower)
    low_hits = sum(1 for m in _CONFIDENCE_LOW_MARKERS if m in lower)
    high_hits = sum(1 for m in _CONFIDENCE_HIGH_MARKERS if m in lower)

    if error_hits > 0:
        return max(0.1, 0.4 - (0.1 * error_hits))
    if low_hits > 2:
        return 0.4
    if low_hits > 0:
        return 0.6
    if high_hits > 0:
        return 0.9
    return 0.75

def validate_upstream(upstream_output: str, min_confidence: float = 0.5) -> tuple:
    """NEW: Compounding-error guard. Returns (is_safe_to_consume, confidence, reason)."""
    conf = confidence_score(upstream_output)
    if conf < min_confidence:
        return (False, conf, f"Upstream confidence {conf:.2f} < threshold {min_confidence}. Re-ground before proceeding.")
    return (True, conf, "OK")

def compounding_error_guard(step_history: list, max_consecutive_low_confidence: int = 2) -> dict:
    """NEW: Checks a sequence of steps for compounding errors."""
    if not step_history:
        return {"should_escalate": False, "reason": "No history", "last_good_step_index": -1}

    consecutive_low = 0
    last_good = -1
    for i, step in enumerate(step_history):
        step_text = step if isinstance(step, str) else step.get("output", "")
        conf = confidence_score(step_text)
        if conf >= 0.5:
            consecutive_low = 0
            last_good = i
        else:
            consecutive_low += 1
            if consecutive_low >= max_consecutive_low_confidence:
                return {
                    "should_escalate": True,
                    "reason": f"{consecutive_low} consecutive low-confidence steps. Re-plan from step {last_good}.",
                    "last_good_step_index": last_good
                }

    return {"should_escalate": False, "reason": "OK", "last_good_step_index": last_good}

def self_verify(answer: str, original_prompt: str, generate_fn, skill: str = "general", complexity: str = "medium") -> str:
    """Upgraded: structural code checks + confidence scoring before LLM critique."""
    if skill not in ("researcher", "coder", "agentic") and complexity != "hard":
        return answer

    if skill == "coder":
        if "```" not in answer and ("def " in answer or "class " in answer or "import " in answer):
            return "REVISED\nI need to provide the code in a proper markdown block.\n```python\n" + answer + "\n```"

    critique_prompt = f"""You previously answered this question:
<question>{original_prompt}</question>

<answer>{answer}</answer>

Critique this answer:
1. Are there factual errors or unsupported claims?
2. Is anything missing that the question required?
3. Is the structure clear and complete?
4. Did you drop any constraints from the original question?

If the answer is correct and complete, reply EXACTLY: VERIFIED
If it needs fixes, reply EXACTLY: REVISED\n<corrected answer here>"""

    try:
        result = generate_fn(critique_prompt) or ""
        if result.strip().startswith("REVISED"):
            revised_content = result.split("REVISED", 1)[-1].strip()
            if len(revised_content) > 20:
                return revised_content
        elif "VERIFIED" in result:
            return answer
    except Exception as e:
        log.debug("[self_verify] critique failed: %s", e)

    return answer

def verify_tool_result(tool_name: str, result: str, expected_schema: Optional[dict] = None) -> dict:
    """NEW: Verify a tool result matches expectations."""
    issues = []
    conf = confidence_score(result)

    if not result or result.strip() == "":
        issues.append("Empty result")
        return {"valid": False, "confidence": 0.0, "issues": issues}

    if expected_schema:
        try:
            parsed = __import__("json").loads(result)
            for key in expected_schema.get("required", []):
                if key not in parsed:
                    issues.append(f"Missing key: {key}")
        except Exception:
            if expected_schema.get("must_be_json"):
                issues.append("Result is not valid JSON")

    return {
        "valid": len(issues) == 0 and conf >= 0.5,
        "confidence": conf,
        "issues": issues
    }
