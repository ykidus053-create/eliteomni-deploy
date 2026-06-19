import re, time, threading, logging
from typing import Optional
log = logging.getLogger(__name__)

from model_router import CircuitState, route_with_fallback, select_model, COMPLEXITY_MAP

_ROUTING_TABLE = {
    ("general",    "easy"):   "mistral-small-latest",
    ("general",    "medium"): "mistral-medium-3.5",
    ("general",    "hard"):   "mistral-large-latest",
    ("researcher", "easy"):   "mistral-medium-3.5",
    ("researcher", "medium"): "mistral-large-latest",
    ("researcher", "hard"):   "mistral-large-latest",
    ("coder",      "easy"):   "codestral-latest",
    ("coder",      "medium"): "codestral-latest",
    ("coder",      "hard"):   "codestral-latest",
    ("calculator", "easy"):   "mistral-small-latest",
    ("calculator", "medium"): "mistral-medium-3.5",
    ("calculator", "hard"):   "mistral-large-latest",
    ("safety",     "easy"):   "mistral-medium-3.5",
    ("safety",     "medium"): "mistral-large-latest",
    ("safety",     "hard"):   "mistral-large-latest",
}

def route_model_v3(skill: str, complexity: str) -> tuple:
    model = _ROUTING_TABLE.get((skill, complexity), select_model(complexity))
    model = route_with_fallback(model)
    log.debug("[route_model_v3] skill=%s complexity=%s -> %s", skill, complexity, model)
    return ("mistral", model)

def clean_history(history: list) -> list:
    if not history: return []
    try:
        clean = []
        for h in history:
            if not isinstance(h, dict): continue
            role    = h.get("role", "")
            content = h.get("content", "") or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
            content = content.strip()
            if not content or len(content) < 2: continue
            if role not in ("user", "assistant"): continue
            if clean and clean[-1]["role"] == role:
                if len(content) > len(clean[-1]["content"]):
                    clean[-1]["content"] = content[:800]
                continue
            clean.append({"role": role, "content": content[:800]})
        return clean[-8:]
    except Exception as e:
        log.error("[clean_history] %s", e); return []

class ToolResult:
    __slots__ = ("ok", "value", "error", "source")
    def __init__(self, ok: bool, value: str, error: str, source: str):
        self.ok = ok; self.value = value or ""; self.error = error or ""; self.source = source
    def to_context(self) -> Optional[str]:
        if self.ok and self.value:
            return f"[{self.source.upper()} RESULT]\n{self.value}\n[/{self.source.upper()}]"
        return None

def safe_tool_call(fn, *args, source: str = "tool", timeout: int = 10, **kwargs) -> ToolResult:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            val = fut.result(timeout=timeout)
        except FutureTimeout:
            log.warning("[safe_tool_call] %s timed out after %ds", source, timeout)
            return ToolResult(ok=False, value="", error=f"{source} timed out", source=source)
        except Exception as e:
            log.error("[safe_tool_call] %s error: %s", source, e)
            return ToolResult(ok=False, value="", error=str(e), source=source)
    if val is None or val == "" or (isinstance(val, list) and len(val) == 0):
        return ToolResult(ok=False, value="", error="empty result", source=source)
    return ToolResult(ok=True, value=str(val)[:2000], error="", source=source)

def call_llm(msgs: list, skill: str = "general", complexity: str = "medium",
             max_tokens: int = 1000, stream: bool = False):
    from modules.core.http_client import mistral_stream
    _, model = route_model_v3(skill, complexity)
    cleaned    = clean_history([m for m in msgs if m.get("role") != "system"])
    system     = next((m for m in msgs if m.get("role") == "system"), None)
    final_msgs = ([system] if system else []) + cleaned
    if not final_msgs: return ("" if not stream else iter([]))
    try:
        if stream: return mistral_stream(final_msgs, max_tokens=max_tokens, model=model)
        result = "".join(mistral_stream(final_msgs, max_tokens=max_tokens, model=model))
        CircuitState.record_success(model)
        return result
    except Exception as e:
        CircuitState.record_failure(model)
        log.error("[call_llm] model=%s failed: %s", model, e)
        raise

def check_tool_available(tool_name: str) -> bool:
    try:
        from modules.services.mcp import _MCP_TOOLS, _MCP_LOCK
        with _MCP_LOCK: return tool_name in _MCP_TOOLS
    except Exception: return False

def build_memory_context(msg: str) -> str:
    try:
        from memory import mem_get, episodic_get
    except Exception as e:
        log.debug("[build_memory_context] memory import failed: %s", e); return ""
    try:
        working  = mem_get(limit=3) or []
        episodic = episodic_get(limit=3) or []
    except Exception as e:
        log.error("[build_memory_context] fetch failed: %s", e); return ""
    facts, seen = [], set()
    for src in [working, episodic]:
        for m in src:
            txt = m if isinstance(m, str) else m.get("text", str(m))
            key = txt[:60].lower()
            if key not in seen and len(txt) > 10:
                seen.add(key); facts.append(txt[:200])
    if not facts: return ""
    lines = "\n".join(f"- {f}" for f in facts[:10])
    return f"\n[MEMORY - from past conversations]\n{lines}\n[/MEMORY]\n"
