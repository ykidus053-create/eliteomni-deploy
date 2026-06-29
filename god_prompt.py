import os, logging
log = logging.getLogger(__name__)
GOD_PROMPT_FILE = "GOD_PROMPT.md"

def get_god_prompt() -> str:
    """Upgraded: Reads the global, AI-editable ruleset."""
    if not os.path.exists(GOD_PROMPT_FILE):
        # Initialize with base rules
        base_rules = """# GOD PROMPT (AI EDITABLE)
## Core Directives
1. You are EliteOmni, an autonomous AGI framework.
2. NEVER write prototypes or stubs. All code must be production-ready.
3. You must use `logging` instead of `print()`.
4. You must handle edge cases and network timeouts.
"""
        with open(GOD_PROMPT_FILE, "w") as f:
            f.write(base_rules)
            
    try:
        with open(GOD_PROMPT_FILE, "r") as f:
            return f.read()
    except:
        return ""

def update_god_prompt(new_rules: str) -> bool:
    """Allows the AI to permanently learn a new global rule."""
    try:
        with open(GOD_PROMPT_FILE, "a") as f:
            f.write(f"\n- {new_rules}\n")
        log.info(f"[GodPrompt] Appended new global rule: {new_rules[:50]}")
        return True
    except:
        return False
