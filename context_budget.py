_TOKEN_BUDGET = {
    "easy":   {"history": 4000,  "system": 3000,  "rag": 2000,  "memory": 1000, "state": 1000},
    "medium": {"history": 12000, "system": 6000,  "rag": 5000,  "memory": 3000, "state": 2500},
    "hard":   {"history": 30000, "system": 12000, "rag": 12000, "memory": 6000, "state": 5000},
}

def estimate_tokens(text: str) -> int:
    return max(1, len(str(text)) // 4)

def allocate_budget(complexity: str, available_ctx: int = 32000) -> dict:
    base = _TOKEN_BUDGET.get(complexity, _TOKEN_BUDGET["medium"])
    scale = min(1.0, available_ctx / 32000.0)
    return {k: int(v * scale) for k, v in base.items()}

def trim_history_to_budget(history: list, budget_tokens: int) -> list:
    """Upgraded: Preserves first user message (original instructions) + most recent."""
    if not history: return []

    first_user = None
    rest = []
    for msg in history:
        if first_user is None and msg.get("role") == "user":
            first_user = msg
        else:
            rest.append(msg)

    trimmed = []
    current_tokens = 0

    if first_user:
        msg_tokens = estimate_tokens(str(first_user.get("content", "")))
        if msg_tokens <= budget_tokens * 0.3:
            trimmed.append(first_user)
            current_tokens += msg_tokens

    recent = []
    for msg in reversed(rest):
        msg_tokens = estimate_tokens(str(msg.get("content", "")))
        if current_tokens + msg_tokens > budget_tokens:
            break
        recent.insert(0, msg)
        current_tokens += msg_tokens

    return trimmed + recent

def trim_history_preserving_instructions(history: list, budget_tokens: int) -> list:
    return trim_history_to_budget(history, budget_tokens)

def trim_system_to_budget(system: str, budget_tokens: int) -> str:
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
    if skill == "coder":
        return 6000
    elif skill == "agentic":
        return 5000
    elif complexity == "hard":
        return 4000
    elif complexity == "medium":
        return 2500
    return 1500

allocate_context_budget = allocate_budget
trim_history_for_ttft = trim_history_to_budget
