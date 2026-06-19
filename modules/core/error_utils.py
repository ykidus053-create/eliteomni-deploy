"""
Structured error responses — never leak raw exception strings to callers.
Replaces: return {"error": str(e)} and return f"[Error reading {f}: {e}]"
"""
import logging, traceback

_log = logging.getLogger("eliteomni")

class AppError(Exception):
    def __init__(self, code: str, message: str, detail: str = ""):
        super().__init__(message)
        self.code    = code
        self.message = message
        self.detail  = detail

def safe_error(code: str, message: str, exc: Exception = None) -> dict:
    """Return a structured error dict; log the real exception internally."""
    if exc:
        _log.error("[%s] %s — %s", code, message,
                   traceback.format_exc()[-400:])
    return {"error": code, "message": message}

def safe_file_error(filename: str, exc: Exception) -> str:
    """Safe string for file-read failures — no raw exception text."""
    _log.warning("[file_read] %s — %s", filename, exc)
    return f"[Could not read {filename}]"

def safe_db_error(operation: str, exc: Exception) -> None:
    """Log DB errors; callers receive None / empty list."""
    _log.error("[db/%s] %s", operation, exc)
