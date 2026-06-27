_TOKEN_BUDGET = {
    "easy":   {"history": 2000,  "system": 2000,  "rag": 1000,  "memory": 500},
    "medium": {"history": 6000,  "system": 4000,  "rag": 3000,  "memory": 1500},
    "hard":   {"history": 12000, "system": 8000,  "rag": 6000,  "memory": 3000},
}

def estimate_tokens(text: str) -> int:
    return max(1, len(str(text)) // 4)

def allocate_budget(complexity: str, available_ctx: int = 32000) -> dict:
    base = _TOKEN_BUDGET.get(complexity, _TOKEN_BUDGET["medium"])
    # Scale based on model context window
    scale = min(1.0, available_ctx / 32000.0)
    return {k: int(v * scale) for k, v in base.items()}

def trim_history_to_budget(history: list, budget_tokens: int) -> list:
    """Upgraded: Actually trims history to fit token budget, keeping most recent."""
    if not history: return []
    trimmed = []
    current_tokens = 0
    for msg in reversed(history):
        msg_tokens = estimate_tokens(str(msg.get("content", "")))
        if current_tokens + msg_tokens > budget_tokens:
            break
        trimmed.insert(0, msg)
        current_tokens += msg_tokens
    return trimmed

def trim_system_to_budget(system: str, budget_tokens: int) -> str:
    """Upgraded: Keeps the beginning and end of system prompt to preserve instructions."""
    sys_tokens = estimate_tokens(system)
    if sys_tokens <= budget_tokens:
        return system
    keep_chars = budget_tokens * 4
    keep_start = int(keep_chars * 0.6)
    keep_end = int(keep_chars * 0.3)
    return system[:keep_start] + "\n...[truncated]...\n" + system[-keep_end:]

def compress_rag_hits(hits: list, budget_tokens: int) -> str:
    if not hits: return ""
    result = []
    current_tokens = 0
    for h in hits:
        text = h.get("text", "") if isinstance(h, dict) else str(h)
        t = estimate_tokens(text)
        if current_tokens + t > budget_tokens:
            allowed_chars = (budget_tokens - current_tokens) * 4
            if allowed_chars > 100:
                result.append(text[:allowed_chars] + "...")
            break
        result.append(text)
        current_tokens += t
    return "\n".join(result)

def get_optimal_max_tokens(msg: str, skill: str, complexity: str) -> int:
    """Upgraded: Dynamic token allocation based on task."""
    if complexity == "hard" or skill == "coder":
        return 4000
    elif complexity == "medium":
        return 2000
    return 1000

allocate_context_budget = allocate_budget
trim_history_for_ttft = trim_history_to_budget
