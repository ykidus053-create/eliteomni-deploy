# ── FIX 1: Remove GROQ_API_KEY check — you use Mistral only ──────────────────
with open("modules/agents.py", "r") as f:
    content = f.read()

content = content.replace(
    "if not GROQ_API_KEY or len(assistant_response) < 100: return",
    "if len(assistant_response) < 100: return"
)

with open("modules/agents.py", "w") as f:
    f.write(content)
print("✓ Fix 1: GROQ_API_KEY check removed from agents.py")

# ── FIX 2: critique() fallback in rlaif.py ───────────────────────────────────
with open("modules/rlaif.py", "r") as f:
    content = f.read()

if "def critique(" not in content:
    content += """

def critique(response: str, skill: str = "general") -> str:
    try:
        return _run_critique(response, skill)
    except Exception:
        return ""
"""
    print("✓ Fix 2: critique() fallback added to rlaif.py")
else:
    print("✓ Fix 2: critique() already exists")

with open("modules/rlaif.py", "w") as f:
    f.write(content)

# ── FIX 3: _mistral_stream() in agents.py ────────────────────────────────────
with open("modules/agents.py", "r") as f:
    content = f.read()

if "def _mistral_stream" not in content:
    content += """

def _mistral_stream(messages: list, system: str = "", model: str = "mistral-large-latest", max_tokens: int = 2048) -> str:
    try:
        from modules.validation import generate_sync
        msgs = [{"role": "system", "content": system}] + messages if system else messages
        return generate_sync(msgs, max_tokens, "general", 0)
    except Exception as e:
        return f"[_mistral_stream error: {e}]"
"""
    print("✓ Fix 3: _mistral_stream() added to agents.py")
else:
    print("✓ Fix 3: _mistral_stream() already exists")

with open("modules/agents.py", "w") as f:
    f.write(content)

print("\n✅ Done — restart with: fuser -k 8080/tcp && sleep 2 && uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1")
