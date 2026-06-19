"""
swe_verifier.py - Post-generation code quality gate.
"""

import re
from typing import Tuple, List

_PLACEHOLDER_PATTERNS = [
    r"#\s*(TODO|FIXME|IMPLEMENT|ADD LOGIC|PLACEHOLDER|YOUR CODE HERE|fill in|complete this)",
    r"^\s*\.\.\.\s*$",
    r"#\s*(rest of (code|implementation)|add more|etc\.?)\s*$",
    r"^\s*pass\s*$",
    r"\[rest of code\]",
    r"except\s+Exception\s*:\s*pass",
]

def assess_code_quality(response: str) -> Tuple[bool, List[str], str]:
    failures = []
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    code = "\n".join(code_blocks) if code_blocks else response

    for pattern in _PLACEHOLDER_PATTERNS:
        matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE)
        if matches:
            failures.append(f"{pattern[:40]} → '{matches[0][:60]}'")

    if failures:
        print(f"[SWE-Verifier] ❌ caught {len(failures)} issues:")
        for f in failures:
            print(f"  • {f}")
    else:
        print(f"[SWE-Verifier] ✅ code passed quality gate ({len(code)} chars)")

    return (len(failures) == 0), failures, code

def build_rewrite_prompt(original_request: str, bad_code: str, failures: List[str]) -> list:
    return [
        {
            "role": "user",
            "content": f"""The following code failed quality review.

Issues found: {failures}

Original request: {original_request[:300]}

Bad code:
{bad_code[:2000]}

Rewrite it completely following these rules:
1. DOMAIN FIRST: identify the exact real-world system, name the standard algorithm experts use
2. Use the correct data structures practitioners actually use in production
3. Handle domain-specific edge cases (not just generic ones)
4. Full implementation — no placeholders, no TODO, no pass, no ellipsis
5. Would a senior engineer at a top tech company accept this code? If not, rewrite again."""
        }
    ]

def swe_verify_and_fix(response: str, original_request: str, stream_fn) -> Tuple[str, bool]:
    """Sync path — buffers rewrite. Used only when caller cannot stream."""
    passes, failures, bad_code = assess_code_quality(response)
    if passes:
        return response, False
    print(f"[SWE-Verifier] ❌ {len(failures)} issues found: {failures[:3]}")
    try:
        rewrite_msgs = build_rewrite_prompt(original_request, bad_code, failures)
        rewritten = stream_fn(rewrite_msgs, max_tokens=4000)
        passes2, failures2, _ = assess_code_quality(rewritten)
        if passes2:
            print("[SWE-Verifier] ✅ Rewrite passed quality gate")
        else:
            print(f"[SWE-Verifier] ⚠️ Rewrite still has {len(failures2)} issues — using best version")
        return rewritten, True
    except Exception as e:
        print(f"[SWE-Verifier] ❌ Rewrite failed: {e}")
        return response, False

def swe_verify_and_fix_stream(response: str, original_request: str, stream_iter_fn):
    """
    Rewrite disabled — original model output is better than rewrite model.
    Just strips hedging and returns original.
    Yields: (chunk: str, done: bool, fixed: bool)
    """
    import re
    cleaned = re.sub(
        r"(Would you like me to.*|Do you prefer.*|\(And yes.*?\)|I'll write.*?if you want.*?|Would you like to expand.*?)$",
        "", response.strip(), flags=re.IGNORECASE|re.DOTALL
    ).strip()
    fixed = cleaned != response.strip()
    yield cleaned, True, fixed
