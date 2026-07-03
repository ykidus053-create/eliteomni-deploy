# modules/tools.py
import py_compile, tempfile, os

def tool_lint(code: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            fname = f.name
        py_compile.compile(fname, doraise=True)
        os.unlink(fname)
        return "OK"
    except Exception as e:
        return str(e)

def numpy_exec_safe(code: str) -> str:
    """Actually execute code in a sandboxed subprocess and return stdout/stderr.
    Used to verify generated code runs correctly before returning it to the user."""
    import subprocess, tempfile, os as _os
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                ["python3", fname],
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout.strip()
            error = result.stderr.strip()
            if result.returncode != 0:
                return f"[EXECUTION FAILED]\nSTDERR:\n{error[-1500:]}"
            return f"[EXECUTION OK]\nSTDOUT:\n{output[-1500:] if output else '(no output)'}"
        finally:
            _os.unlink(fname)
    except subprocess.TimeoutExpired:
        return "[EXECUTION FAILED] Timed out after 10s (possible infinite loop)"
    except Exception as e:
        return f"[EXECUTION FAILED] {e}"
