import re, time, logging
log = logging.getLogger(__name__)

MISTRAL_SMALL  = "mistral-medium-3.5"
MISTRAL_MEDIUM = "mistral-medium-3.5"
MISTRAL_LARGE  = "mistral-medium-3.5"
CODESTRAL      = "mistral-medium-3.5"
MAGISTRAL      = "mistral-medium-3.5"
REASONING_EFFORT = "high"

COMPLEXITY_MAP: dict[str, str] = {
    "easy":      MISTRAL_SMALL,
    "medium":    MISTRAL_MEDIUM,
    "hard":      MISTRAL_LARGE,
    "coder":     CODESTRAL,
    "reasoning": MAGISTRAL,
}

FALLBACK_CHAIN: dict[str, str] = {
    MISTRAL_LARGE:  MISTRAL_MEDIUM,
    MISTRAL_MEDIUM: MISTRAL_SMALL,
    CODESTRAL:      MISTRAL_MEDIUM,
    MAGISTRAL:      MISTRAL_LARGE,
    MISTRAL_SMALL:  MISTRAL_SMALL,
}

CODE_SIGNALS = ["def ", "import ", "class ", "function", "const ", "html", "css",
                "python", "javascript", "typescript", "rust", "c++", "sql", "```"]

def select_model(complexity: str, messages_payload=None) -> str:
    if complexity in ("coder", "coding", "easy_code"):
        return CODESTRAL
    if complexity == "reasoning":
        return MAGISTRAL
    if messages_payload:
        text_content = str(messages_payload).lower()
        if any(s in text_content for s in CODE_SIGNALS):
            return CODESTRAL
    return "mistral-medium-3.5"

def get_token_budget(model_name: str) -> int:
    if "codestral" in str(model_name).lower():
        return 32000
    if "small" in str(model_name).lower():
        return 32000
    return 128000

def record_outcome(model_name: str, outcome) -> None:
    log.debug("[Telemetry] model=%s outcome=%s", model_name, outcome)

def trim_system(system_prompt: str, max_tokens: int = 4000) -> str:
    if len(system_prompt) > max_tokens * 4:
        return system_prompt[:max_tokens * 4]
    return system_prompt

# ── Circuit Breaker ───────────────────────────────────────────────────────────
import threading as _th
class CircuitState:
    _lock  = _th.Lock()
    _state: dict[str, dict] = {}
    THRESHOLD = 3
    RESET_S   = 60

    @classmethod
    def record_failure(cls, model: str):
        with cls._lock:
            s = cls._state.setdefault(model, {"failures": 0, "open": False, "opened_at": 0})
            s["failures"] += 1
            if s["failures"] >= cls.THRESHOLD:
                s["open"] = True; s["opened_at"] = time.time()
                log.warning("[CircuitBreaker] %s OPEN after %d failures", model, s["failures"])

    @classmethod
    def record_success(cls, model: str):
        with cls._lock:
            cls._state.pop(model, None)

    @classmethod
    def is_open(cls, model: str) -> bool:
        with cls._lock:
            s = cls._state.get(model)
            if not s or not s["open"]: return False
            if time.time() - s["opened_at"] > cls.RESET_S:
                s["open"] = False; s["failures"] = 0; return False
            return True

    @classmethod
    def stats(cls) -> dict:
        with cls._lock:
            return {k: dict(v) for k, v in cls._state.items()}

def route_with_fallback(model: str) -> str:
    visited = set()
    while model and model not in visited:
        if not CircuitState.is_open(model):
            return model
        log.warning("[route_with_fallback] %s is open, trying fallback", model)
        visited.add(model)
        model = FALLBACK_CHAIN.get(model, MISTRAL_SMALL)
    return MISTRAL_SMALL
