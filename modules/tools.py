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
    return "[numpy_exec_safe] not available in this context"
