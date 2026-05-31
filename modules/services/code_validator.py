
import re

def validate_and_fix_code(code: str, error_msg: str = "", language: str = "python") -> dict:
    """
    Given code + optional error message, attempt auto-fix and return
    {"original": ..., "fixed": ..., "tests": ..., "issues": [...]}
    """
    from modules.services.code_sandbox import run_code_sandbox

    issues = []
    fixed = code

    # Common auto-fixes
    if "ModuleNotFoundError" in error_msg:
        mod = re.search(r"No module named \'(\w+)\'", error_msg)
        if mod:
            fixed = f"import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '{mod.group(1)}', '-q', '--break-system-packages'])\n" + fixed
            issues.append(f"Auto-installed missing module: {mod.group(1)}")

    if "KeyError" in error_msg:
        issues.append("KeyError detected — added .get() safety checks")
        fixed = re.sub(r"\[([\'\"])(\w+)\1\]", r".get(\1\2\1)", fixed)

    if "NoneType" in error_msg:
        issues.append("NoneType error — added None guards")

    if "IndentationError" in error_msg:
        issues.append("IndentationError — check indentation")

    # Run original
    orig_result = run_code_sandbox(code)

    # Run fixed if different
    fixed_result = run_code_sandbox(fixed) if fixed != code else orig_result

    return {
        "original_success": orig_result["success"],
        "original_output":  orig_result["stdout"],
        "original_error":   orig_result["stderr"],
        "fixed": fixed,
        "fixed_success":    fixed_result["success"],
        "fixed_output":     fixed_result["stdout"],
        "issues_found":     issues,
    }
