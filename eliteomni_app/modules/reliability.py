
"""
EliteOmni Reliability Layer — 5 fixes mirroring Claude infrastructure
1. Tool calls never fail silently
2. Graceful degradation
3. Single codepath
4. Context always valid
5. Real model routing
"""
import re, time, threading
from typing import Optional

# ── FIX 5: REAL MODEL ROUTING ─────────────────────────────────────────────────
_ROUTING_TABLE = {
    ("general",    "easy"):   "mistral-small-latest",
    ("general",    "medium"): "mistral-small-latest",
    ("general",    "hard"):   "mistral-large-latest",
    ("researcher", "easy"):   "mistral-small-latest",
    ("researcher", "medium"): "mistral-large-latest",
    ("researcher", "hard"):   "mistral-large-latest",
    ("coder",      "easy"):   "mistral-small-latest",
    ("coder", "medium"): "codestral-latest",
    ("coder", "hard"): "codestral-latest",
    ("calculator", "easy"):   "mistral-small-latest",
    ("calculator", "medium"): "mistral-small-latest",
    ("calculator", "hard"):   "mistral-large-latest",
    ("safety",     "easy"):   "mistral-small-latest",
    ("safety",     "medium"): "mistral-small-latest",
    ("safety",     "hard"):   "mistral-small-latest",
}

def route_model_v3(skill: str, complexity: str) -> tuple:
    model = _ROUTING_TABLE.get(
        (skill, complexity),
        "mistral-small-latest" if complexity == "easy" else "mistral-large-latest"
    )
    print(f"[route_model_v3] skill={skill} complexity={complexity} -> {model}")
    return ("mistral", model)


# ── FIX 4: CONTEXT VALIDATION ─────────────────────────────────────────────────
def clean_history(history: list) -> list:
    if not history:
        return []
    try:
        clean = []
        for h in history:
            if not isinstance(h, dict):
                continue
            role    = h.get("role", "")
            content = h.get("content", "") or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
            content = content.strip()
            if not content or len(content) < 2:
                continue
            if role not in ("user", "assistant"):
                continue
            if clean and clean[-1]["role"] == role:
                if len(content) > len(clean[-1]["content"]):
                    clean[-1]["content"] = content[:800]
                continue
            clean.append({"role": role, "content": content[:800]})
        return clean[-8:]
    except Exception as e:
        print(f"[clean_history] error: {e}")
        return []


# ── FIX 1 + 2: TOOL WRAPPER ───────────────────────────────────────────────────
class ToolResult:
    __slots__ = ("ok", "value", "error", "source")
    def __init__(self, ok: bool, value: str, error: str, source: str):
        self.ok     = ok
        self.value  = value or ""
        self.error  = error or ""
        self.source = source

    def to_context(self) -> Optional[str]:
        if self.ok and self.value:
            return f"[{self.source.upper()} RESULT]\n{self.value}\n[/{self.source.upper()}]"
        return None


def safe_tool_call(fn, *args, source: str = "tool", timeout: int = 10, **kwargs) -> ToolResult:
    result_box = [None]
    error_box  = [None]
    def _run():
        try:
            result_box[0] = fn(*args, **kwargs)
        except Exception as e:
            error_box[0] = str(e)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f"[safe_tool_call] {source} timed out after {timeout}s - skipping")
        return ToolResult(ok=False, value="", error=f"{source} timed out", source=source)
    if error_box[0]:
        print(f"[safe_tool_call] {source} error: {error_box[0]}")
        return ToolResult(ok=False, value="", error=error_box[0], source=source)
    val = result_box[0]
    if val is None or val == "" or (isinstance(val, list) and len(val) == 0):
        return ToolResult(ok=False, value="", error="empty result", source=source)
    return ToolResult(ok=True, value=str(val)[:2000], error="", source=source)


# ── FIX 3: SINGLE CODEPATH ────────────────────────────────────────────────────
def call_llm(msgs: list, skill: str = "general", complexity: str = "medium",
             max_tokens: int = 1000, stream: bool = False):
    from modules.groq_client import mistral_stream
    _, model = route_model_v3(skill, complexity)
    cleaned  = clean_history([m for m in msgs if m.get("role") != "system"])
    system   = next((m for m in msgs if m.get("role") == "system"), None)
    final_msgs = ([system] if system else []) + cleaned
    if not final_msgs:
        return ("" if not stream else iter([]))
    if stream:
        return mistral_stream(final_msgs, max_tokens=max_tokens, model=model)
    return "".join(mistral_stream(final_msgs, max_tokens=max_tokens, model=model))


# ── FIX 2: MCP GUARD ──────────────────────────────────────────────────────────
def check_tool_available(tool_name: str) -> bool:
    try:
        from modules.mcp import _MCP_TOOLS, _MCP_LOCK
        with _MCP_LOCK:
            return tool_name in _MCP_TOOLS
    except Exception:
        return False


# ── MEMORY READ LOOP ──────────────────────────────────────────────────────────
def build_memory_context(msg: str) -> str:
    try:
        from modules.search import mem_get, mem_get_episodic, rag_get
    except Exception:
        return ""
    try:
        from modules.semantic_mem import semantic_mem_get
        sem = semantic_mem_get(msg, k=3) or []
    except Exception:
        sem = []
    try:
        from modules.memory import db_mem_get
        db = db_mem_get(msg, k=3) or []
    except Exception:
        db = []
    working  = mem_get(msg, k=3) or []
    episodic = mem_get_episodic(msg) or []
    rag      = rag_get(msg, k=2) or []
    facts = []
    seen  = set()
    for src in [working, db, sem, episodic]:
        for m in src:
            txt = m if isinstance(m, str) else m.get("text", str(m))
            key = txt[:60].lower()
            if key not in seen and len(txt) > 10:
                seen.add(key)
                facts.append(txt[:200])
    for r in rag:
        txt = r.get("text", "") if isinstance(r, dict) else str(r)
        key = txt[:60].lower()
        if key not in seen and len(txt) > 10:
            seen.add(key)
            facts.append(f"[KB] {txt[:200]}")
    if not facts:
        return ""
    lines = "\n".join(f"- {f}" for f in facts[:10])
    return f"\n[MEMORY - from past conversations]\n{lines}\n[/MEMORY]\n"
