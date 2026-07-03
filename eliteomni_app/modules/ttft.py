
"""
TTFT Optimization Layer for EliteOmni
Based on: trim input tokens = faster prefill = faster first token
"""
import time, threading, re

# ── 1. SYSTEM PROMPT TOKEN CAPS ───────────────────────────────────
# Every extra token in system prompt = extra prefill time before first token
# Measured caps based on complexity:
SYSTEM_PROMPT_CAPS = {
    "easy":   600,   # ~150 tokens — just personality + skill
    "medium": 1200,  # ~300 tokens — skill + tools + key rules
    "hard":   2400,  # ~600 tokens — full reasoning suite
}

def trim_system_prompt(system: str, complexity: str) -> str:
    """
    Hard-cap system prompt length by complexity.
    Single biggest TTFT win — fewer input tokens = faster prefill.
    Preserves the START of the prompt (identity + skill) and
    trims redundant rule blocks from the end.
    """
    cap = SYSTEM_PROMPT_CAPS.get(complexity, 1200)
    if len(system) <= cap:
        return system
    trimmed = system[:cap]
    # Don't cut mid-sentence
    last_newline = trimmed.rfind("\n")
    if last_newline > cap * 0.8:
        trimmed = trimmed[:last_newline]
    print(f"[TTFT] system prompt trimmed {len(system)} -> {len(trimmed)} chars")
    return trimmed


# ── 2. MAX TOKEN CAPS BY COMPLEXITY ──────────────────────────────
# Smaller max_tokens = model starts streaming sooner
# Model stops as soon as answer is complete anyway
MAX_TOKEN_CAPS = {
    "easy":   300,
    "medium": 800,
    "hard":   2000,
}

def cap_max_tokens(requested: int, complexity: str) -> int:
    return requested  # No more capping!


# ── 3. PARALLEL SEARCH + PROMPT BUILD ────────────────────────────
# Instead of: search() → build_prompt() → call_llm()
# Do:         search() ──┐
#             build()  ──┴→ call_llm()
# Hides search latency behind prompt construction

def parallel_search_and_build(msg: str, build_fn, search_fn):
    """
    Run search and prompt-build in parallel.
    Returns (search_context, built_prompt) together.
    Typically saves 200-600ms on researcher queries.
    """
    results = {"search": None, "error": None}

    def _do_search():
        try:
            results["search"] = search_fn(msg)
        except Exception as e:
            results["error"] = str(e)

    t = threading.Thread(target=_do_search, daemon=True)
    t.start()

    # Build prompt while search runs
    built = build_fn()

    # Wait for search (max 4s — dont block forever)
    t.join(timeout=4)
    return results.get("search"), built


# ── 4. TTFT MEASUREMENT ───────────────────────────────────────────
_ttft_log = []  # last 100 measurements

class TTFTTracker:
    """
    Wrap mistral_stream to measure and log actual TTFT.
    Usage: wrap your stream call, get real p50/p95 data.
    """
    def __init__(self, label: str = ""):
        self.label     = label
        self.t_start   = time.time()
        self.t_first   = None
        self.token_count = 0

    def on_token(self, tok: str):
        if tok and self.t_first is None:
            self.t_first = time.time()
            ttft_ms = (self.t_first - self.t_start) * 1000
            _ttft_log.append({"label": self.label, "ttft_ms": ttft_ms, "ts": self.t_start})
            if len(_ttft_log) > 100:
                _ttft_log.pop(0)
            print(f"[TTFT] {self.label} first token in {ttft_ms:.0f}ms")
        self.token_count += 1

    @staticmethod
    def report() -> dict:
        if not _ttft_log:
            return {}
        times = sorted(r["ttft_ms"] for r in _ttft_log)
        n = len(times)
        return {
            "p50": times[n // 2],
            "p95": times[int(n * 0.95)],
            "p99": times[min(int(n * 0.99), n - 1)],
            "count": n,
            "latest": times[-1],
        }


# ── 5. HISTORY TOKEN TRIMMER ──────────────────────────────────────
# Conversation history bloats input tokens fast
# Keep only what fits in a tight budget

def trim_history_for_ttft(hist_msgs: list, complexity: str) -> list:
    """
    Aggressively trim history based on complexity.
    Easy: last 2 turns only
    Medium: last 4 turns
    Hard: last 6 turns, each capped at 400 chars
    """
    limits = {"easy": 2, "medium": 4, "hard": 6}
    char_cap = {"easy": 200, "medium": 400, "hard": 600}
    n   = limits.get(complexity, 4)
    cap = char_cap.get(complexity, 400)
    trimmed = hist_msgs[-n * 2:] if hist_msgs else []
    return [{"role": m["role"], "content": m.get("content","")[:cap]} for m in trimmed]
