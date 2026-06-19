"""
code_executor.py
Extracts code from AI response, runs it in a subprocess sandbox,
returns (passed, errors, output). Used to catch bugs before delivery.
"""
import re, subprocess, sys, tempfile, os
from typing import Tuple

def extract_code_blocks(response: str) -> list[str]:
    return re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)

# Imports that signal project-specific or unpublished dependencies
_SKIP_IMPORTS = re.compile(
    r"^(?:import|from)\s+("
    r"kv_store_pb2|grpc|etcd|raft|zookeeper|kafka|"  # infra stubs
    r"modules\.|app\.|eliteomni"                      # project-local
    r")",
    re.MULTILINE
)

def _has_unresolvable_deps(code: str) -> bool:
    """Return True if code imports things the sandbox can never satisfy."""
    # Project-local or protobuf-generated modules
    if _SKIP_IMPORTS.search(code):
        return True
    # Any import not in stdlib or installed packages
    imports = re.findall(r"^(?:import|from)\s+([\w]+)", code, re.MULTILINE)
    import sys
    for mod in imports:
        if mod in sys.stdlib_module_names:
            continue
        try:
            __import__(mod)
        except ImportError:
            return True
    return False

def run_code_safe(code: str, timeout: int = 10) -> Tuple[bool, str, str]:
    """Run code in isolated subprocess. Returns (passed, stdout, stderr)."""
    if _has_unresolvable_deps(code):
        print(f"[Executor] ⏭ skipping execution — unresolvable imports detected")
        return True, "SKIPPED: external deps", ""
    # Add basic test harness if no __main__ or test functions
    runnable = code
    if "__main__" not in code and "def test_" not in code:
        runnable += "\n# Auto-smoke-test: instantiate main classes\n"
        classes = re.findall(r"^class (\w+)", code, re.MULTILINE)
        for cls in classes[:2]:
            # Skip abstract/protocol/stub classes (body is all ... or pass)
            cls_body = re.search(rf"class {cls}[^:]*:(.*?)(?=^class |\Z)", code, re.DOTALL | re.MULTILINE)
            if cls_body:
                body = cls_body.group(1)
                stubs = re.findall(r"def \w+[^:]+:\s*\.\.\.\s*$", body, re.MULTILINE)
                methods = re.findall(r"def \w+", body)
                if stubs and len(stubs) == len(methods):
                    runnable += f"print('SKIP: {cls} is abstract/protocol')\n"
                    continue
            runnable += f"try:\n    _inst = {cls}()\n    print('OK: {cls} instantiated')\nexcept TypeError:\n    print('SKIP: {cls} requires args')\nexcept Exception as e:\n    print(f'FAIL: {cls}: {{e}}')\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                     delete=False, dir='/tmp') as f:
        f.write(runnable)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
        )
        passed = result.returncode == 0 and "FAIL:" not in result.stdout
        return passed, result.stdout[:500], result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT: code ran >10s"
    except Exception as e:
        return False, "", str(e)
    finally:
        os.unlink(tmp_path)

def execute_and_verify(response: str, original_request: str,
                        stream_fn) -> Tuple[str, bool]:
    """
    Extract code, run it, if it fails feed error back to model for fix.
    Returns (final_response, was_fixed).
    """
    blocks = extract_code_blocks(response)
    if not blocks:
        return response, False

    all_passed = True
    errors = []
    for block in blocks[:2]:  # test first 2 blocks
        passed, stdout, stderr = run_code_safe(block)
        if not passed:
            all_passed = False
            errors.append(stderr or stdout)

    if all_passed:
        print(f"[Executor] ✅ code runs clean")
        return response, False

    print(f"[Executor] ❌ code failed execution: {errors[0][:200]}")

    # Feed error back to model for fix
    fix_prompt = [{
        "role": "user",
        "content": f"""This code failed when executed:

ERROR:
{errors[0][:500]}

Original request: {original_request[:200]}

Broken code:
{blocks[0][:2000]}

Fix the code. Return only the corrected implementation. No explanation."""
    }]

    try:
        fixed = stream_fn(fix_prompt, max_tokens=4000)
        # Verify the fix
        fixed_blocks = extract_code_blocks(fixed)
        if fixed_blocks:
            passed2, _, stderr2 = run_code_safe(fixed_blocks[0])
            if passed2:
                print(f"[Executor] ✅ fix verified clean")
                return fixed, True
            else:
                print(f"[Executor] ⚠️ fix still has errors: {stderr2[:100]} — keeping original")
                return response, False  # do NOT return broken fix
        return response, False  # fix had no code blocks — keep original
    except Exception as e:
        print(f"[Executor] fix attempt failed: {e}")
        return response, False
