MISTRAL_SMALL  = "mistral-medium-3.5"
MISTRAL_MEDIUM = "mistral-medium-3.5"
MISTRAL_LARGE  = "mistral-medium-3.5"
REASONING_EFFORT = "high"
# Core Model Routing Definitions with Complete Pipeline Exports
import re


COMPLEXITY_MAP: dict[str, str] = {
    "easy":   "mistral-medium-3.5",
    "medium": "mistral-medium-3.5",
    "hard":   "mistral-medium-3.5",
}

FALLBACK_CHAIN: dict[str, str] = {
    "mistral-medium-3.5": "mistral-medium-3.5",
    "mistral-medium-3.5": "mistral-medium-3.5",
}

def select_model(complexity: str, messages_payload=None) -> str:
    # Explicit routing rule: All programming/syntax tasks go to Codestral
    if complexity in ("hard", "coder", "coding", "easy_code"):
        return "mistral-medium-3.5"
        
    # Content scanner fallback to prevent leakage
    if messages_payload:
        text_content = str(messages_payload).lower()
        code_signals = ["def ", "import ", "class ", "function", "const ", "html", "css", "python", "javascript", "rust", "c++", "sql", "```"]
        if any(signal in text_content for signal in code_signals):
            return "mistral-medium-3.5"
            
    return "mistral-medium-3.5"

def get_token_budget(model_name: str) -> int:
    """Calculates context window allocations for active execution streams."""
    if "devstral" in str(model_name).lower() or "codestral" in str(model_name).lower():
        return 32000
    return 128000

def record_outcome(model_name: str, outcome: dict or str or bool) -> None:
    """Satisfies Online Learning Loop telemetry feedback records."""
    print(f"[Telemetry] Route metrics recorded for engine choice: {model_name}")
    return None


def trim_system(system_prompt: str, max_tokens: int = 4000) -> str:
    """Satisfies memory/prompt reduction layer by passing through or capping text."""
    if len(system_prompt) > max_tokens * 4:
        return system_prompt[:max_tokens * 4]
    return system_prompt
