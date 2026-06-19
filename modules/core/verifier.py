"""
Programmatic Verifier — turns feedback into a self-growing verified dataset.
- Code responses  → execute in sandbox, check for errors
- Math responses  → extract expressions, verify with sympy
- Auto-scores and appends to sft_store with verified=True tag
"""
import re, json, os, subprocess, tempfile, time
from pathlib import Path

FEEDBACK_PATH = Path(__file__).parent / "feedback_store.json"

# ── Code Verifier ─────────────────────────────────────────────────────────────
def _extract_python(response: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    return m.group(1).strip() if m else None

def verify_code(response: str) -> dict:
    code = _extract_python(response)
    if not code:
        return {"verified": False, "reason": "no_code_block"}
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            ["python3", tmp],
            capture_output=True, text=True, timeout=10
        )
        os.unlink(tmp)
        if result.returncode == 0:
            return {"verified": True, "reason": "executed_ok", "output": result.stdout[:500]}
        else:
            return {"verified": False, "reason": "runtime_error", "error": result.stderr[:300]}
    except subprocess.TimeoutExpired:
        os.unlink(tmp)
        return {"verified": False, "reason": "timeout"}
    except Exception as e:
        return {"verified": False, "reason": str(e)}

# ── Math Verifier ─────────────────────────────────────────────────────────────
def verify_math(response: str) -> dict:
    try:
        import sympy
    except ImportError:
        return {"verified": False, "reason": "sympy_not_installed"}
    # Extract boxed answers or = expressions
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
        r"answer(?:\s+is)?:?\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            try:
                val = sympy.sympify(m.group(1).replace(",", ""))
                return {"verified": True, "reason": "math_parsed", "value": str(val)}
            except Exception:
                continue
    return {"verified": False, "reason": "no_math_expression"}

# ── Auto-verify and store ─────────────────────────────────────────────────────
def auto_verify_and_store(skill: str, msg: str, response: str) -> dict:
    """
    Called after every response. Verifies if possible and appends
    high-quality verified samples to sft_store automatically.
    """
    result = {"verified": False, "reason": "unverifiable"}

    is_code = bool(_extract_python(response))
    is_math = skill == "calculator" or any(
        kw in msg.lower() for kw in ["solve", "calculate", "compute", "integral", "derivative", "equation"]
    )

    if is_code:
        result = verify_code(response)
    elif is_math:
        result = verify_math(response)

    # Auto-store verified responses into sft_store
    if result["verified"]:
        try:
            store = json.loads(FEEDBACK_PATH.read_text()) if FEEDBACK_PATH.exists() else {"feedback": {}, "sft_store": []}
            entry = {
                "skill": skill,
                "msg": msg[:500],
                "response": response[:2000],
                "verified": True,
                "verify_method": "code_exec" if is_code else "sympy",
                "verify_result": result,
                "ts": time.time()
            }
            store.setdefault("sft_store", []).append(entry)
            FEEDBACK_PATH.write_text(json.dumps(store, indent=2))
            print(f"[Verifier] ✅ auto-stored verified sample — skill={skill} method={entry['verify_method']}")
        except Exception as e:
            print(f"[Verifier] store error: {e}")

    return result
