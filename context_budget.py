_TOKEN_BUDGET = {
    "easy":   {"history": 4000,  "system": 4000,  "rag": 2000,  "memory": 1000},
    "medium": {"history": 8000,  "system": 8000,  "rag": 4000,  "memory": 2000},
    "hard":   {"history": 16000, "system": 16000, "rag": 8000,  "memory": 4000},
}
def estimate_tokens(text):
    return max(1, len(text) // 4)
def allocate_budget(complexity, available_ctx=128000):
    base = _TOKEN_BUDGET.get(complexity, _TOKEN_BUDGET["medium"])
    return {k: v for k, v in base.items()}
def trim_history_to_budget(history, budget_tokens):
    return history
def trim_system_to_budget(system, budget_tokens):
    return system
def compress_rag_hits(hits, budget_tokens):
    if not hits:
        return ""
    return "\n".join([h.get("text","") if isinstance(h,dict) else str(h) for h in hits])
def get_optimal_max_tokens(msg, skill, complexity):
    return 32000
allocate_context_budget = allocate_budget
trim_history_for_ttft = trim_history_to_budget
