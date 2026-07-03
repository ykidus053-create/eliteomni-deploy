"""
EliteOmni — full architectural fix script
Run from the same folder as app.py:  python apply_fixes.py
Creates a backup at app.py.bak, then patches app.py in-place.
"""

import re, shutil, sys, os

SRC = os.path.join(os.path.dirname(__file__), "app.py")
if not os.path.exists(SRC):
    # Try home directory
    SRC = os.path.expanduser("~/app.py")
if not os.path.exists(SRC):
    print("ERROR: app.py not found. Run this script from the same directory as app.py.")
    sys.exit(1)

shutil.copy(SRC, SRC + ".bak")
print(f"✅ Backup saved → {SRC}.bak")

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()

original_len = len(src)
fixes_applied = []

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — IndentationError: final = verification_pipeline(...) is inside the
#          for-loop body (8-space indent). Also `final` is never assigned before
#          that line. Fix both: dedent + assign final = response above it.
# ─────────────────────────────────────────────────────────────────────────────
OLD_F1 = "    # ── FINALIZE ──────────────────────────────────────────────────────────────\n    # Point 2: values in 70B weights, not runtime loops\n        final  = verification_pipeline(final, msg, skill)"
NEW_F1 = "    # ── FINALIZE ──────────────────────────────────────────────────────────────\n    final  = response\n    final  = verification_pipeline(final, msg, skill)"
if OLD_F1 in src:
    src = src.replace(OLD_F1, NEW_F1, 1)
    fixes_applied.append("FIX 1 ✅  IndentationError + missing `final = response` assignment")
else:
    fixes_applied.append("FIX 1 ⚠️  Pattern not found (may already be fixed or line changed)")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Dead unreachable except-block at lines ~134-138 (after a return stmt)
# ─────────────────────────────────────────────────────────────────────────────
OLD_F2 = """        return f"[Groq error: {e}]"
        try:
            body = e.read().decode()[:500]
        except: pass
        print(f"[Groq generate error] {e} | body: {body}")
        return f"[Groq error: {e}]" """
NEW_F2 = '        return f"[Groq error: {e}]"'
if OLD_F2 in src:
    src = src.replace(OLD_F2, NEW_F2, 1)
    fixes_applied.append("FIX 2 ✅  Removed dead unreachable except-block in groq_generate()")
else:
    # Try alternate whitespace
    OLD_F2b = '        return f"[Groq error: {e}]"\n        try:\n            body = e.read().decode()[:500]\n        except: pass\n        print(f"[Groq generate error] {e} | body: {body}")\n        return f"[Groq error: {e}]"'
    if OLD_F2b in src:
        src = src.replace(OLD_F2b, '        return f"[Groq error: {e}]"', 1)
        fixes_applied.append("FIX 2 ✅  Removed dead unreachable except-block in groq_generate()")
    else:
        fixes_applied.append("FIX 2 ⚠️  Dead block not found (may already be removed)")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Duplicate #proj-list div in sidebar HTML
# ─────────────────────────────────────────────────────────────────────────────
DUPE_BLOCK = """  <div style="padding:10px 12px;border-top:1px solid var(--bd)">
    <div style="font-size:.68rem;color:var(--t3);margin-bottom:6px;font-weight:600;letter-spacing:.04em">PROJECTS</div>
    <div id="proj-list" style="max-height:120px;overflow-y:auto;margin-bottom:6px"></div>
    <button onclick="createProject()" style="width:100%;background:var(--g2);border:1px solid var(--bd);border-radius:6px;padding:5px;font-size:.72rem;color:var(--t2);cursor:pointer">+ New Project</button>
  </div>
  <div style="padding:10px 12px;border-top:1px solid var(--bd)">
    <div style="font-size:.68rem;color:var(--t3);margin-bottom:6px;font-weight:600;letter-spacing:.04em">PROJECTS</div>
    <div id="proj-list" style="max-height:120px;overflow-y:auto;margin-bottom:6px"></div>
    <button onclick="createProject()" style="width:100%;background:var(--g2);border:1px solid var(--bd);border-radius:6px;padding:5px;font-size:.72rem;color:var(--t2);cursor:pointer">+ New Project</button>
  </div>"""
SINGLE_BLOCK = """  <div style="padding:10px 12px;border-top:1px solid var(--bd)">
    <div style="font-size:.68rem;color:var(--t3);margin-bottom:6px;font-weight:600;letter-spacing:.04em">PROJECTS</div>
    <div id="proj-list" style="max-height:120px;overflow-y:auto;margin-bottom:6px"></div>
    <button onclick="createProject()" style="width:100%;background:var(--g2);border:1px solid var(--bd);border-radius:6px;padding:5px;font-size:.72rem;color:var(--t2);cursor:pointer">+ New Project</button>
  </div>"""
if DUPE_BLOCK in src:
    src = src.replace(DUPE_BLOCK, SINGLE_BLOCK, 1)
    fixes_applied.append("FIX 3 ✅  Removed duplicate #proj-list div (duplicate DOM ID)")
else:
    fixes_applied.append("FIX 3 ⚠️  Duplicate proj-list block not found (may already be fixed)")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — LlamaGuard: currently only logs, doesn't block.
#          Make groq_moderate() actually gate the pipeline.
#          We patch pipeline_sync() to check moderation and return early.
# ─────────────────────────────────────────────────────────────────────────────
OLD_F4 = """    # ── Safety gate (pre-loop) ────────────────────────────────────────────────
    vetoed, reason = topological_veto(msg)
    if vetoed:
        return {"response": reason, "skill": "safety", "mode": "fast",
                "vetoed": True, "effort": EFFORT_LEVEL}"""
NEW_F4 = """    # ── Safety gate (pre-loop) ────────────────────────────────────────────────
    vetoed, reason = topological_veto(msg)
    if vetoed:
        return {"response": reason, "skill": "safety", "mode": "fast",
                "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}"""
if OLD_F4 in src:
    src = src.replace(OLD_F4, NEW_F4, 1)
    fixes_applied.append("FIX 4 ✅  LlamaGuard now actively blocks unsafe requests")
else:
    fixes_applied.append("FIX 4 ⚠️  Safety gate anchor not found")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 5 — Agentic safety: BASH() has no confirmation gate for destructive ops.
#          Add a pre-execution check that refuses irreversible commands and
#          requires explicit confirmation token for write/delete operations.
# ─────────────────────────────────────────────────────────────────────────────
OLD_F5 = """def tool_bash(cmd: str, timeout: int = 10) -> str:
    if _BASH_BLOCKED.search(cmd):
        return "[BLOCKED]: Destructive command not allowed."
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (result.stdout + result.stderr).strip()
        return out[:1000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"[BASH TIMEOUT after {timeout}s]"
    except Exception as e:
        return f"[BASH ERROR]: {e}" """
NEW_F5 = """# Irreversible action patterns — require explicit [CONFIRMED] token
_BASH_IRREVERSIBLE = re.compile(
    r'(rm\s|mv\s|cp\s|dd\s|mkfs|truncate|>(?!=)|tee\s|chmod|chown|'
    r'apt.*(install|remove|purge)|pip.*(install|uninstall)|'
    r'systemctl\s+(stop|disable|restart)|kill\s|pkill\s|'
    r'curl.*\||wget.*\||git\s+(push|reset|clean))',
    re.IGNORECASE
)

def tool_bash(cmd: str, timeout: int = 10, confirmed: bool = False) -> str:
    \"\"\"
    Sandboxed bash execution with agentic safety gates (FIX 5).
    Irreversible commands require confirmed=True or [CONFIRMED] in cmd.
    \"\"\"
    if _BASH_BLOCKED.search(cmd):
        return "[BLOCKED]: Destructive command not allowed."
    # Minimal-footprint gate: warn on irreversible ops unless confirmed
    _explicit_confirm = "[CONFIRMED]" in cmd
    cmd_clean = cmd.replace("[CONFIRMED]", "").strip()
    if _BASH_IRREVERSIBLE.search(cmd_clean) and not confirmed and not _explicit_confirm:
        return (
            "[CONFIRMATION REQUIRED]: This command may be irreversible "
            f"({cmd_clean[:80]}...). "
            "To proceed, include [CONFIRMED] in the command or set confirmed=True. "
            "Review carefully before confirming."
        )
    try:
        result = subprocess.run(cmd_clean, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (result.stdout + result.stderr).strip()
        _audit("bash_exec", {"cmd": cmd_clean[:120], "confirmed": confirmed or _explicit_confirm})
        return out[:1000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"[BASH TIMEOUT after {timeout}s]"
    except Exception as e:
        return f"[BASH ERROR]: {e}" """
if OLD_F5 in src:
    src = src.replace(OLD_F5, NEW_F5, 1)
    fixes_applied.append("FIX 5 ✅  BASH() now has confirmation gate for irreversible commands")
else:
    fixes_applied.append("FIX 5 ⚠️  tool_bash anchor not found")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 6 — Prompt injection: FETCH() and tool_web_fetch() dump raw web content
#          directly into model context. Apply the existing validate_tool_result()
#          to all tool results in run_tools_parallel() + sanitize fetched HTML.
# ─────────────────────────────────────────────────────────────────────────────
OLD_F6 = """def _run_one_tool(name: str, arg: str) -> str:
    \"\"\"Route a single tool call to its handler.\"\"\"
    try:
        name = name.upper()
        if name == "BROWSER": return tool_browser(arg)
        if name == "WEATHER": return tool_weather(arg)
        if name == "CALC":   return tool_calc(arg)
        if name == "TIME":   return tool_time()
        if name == "SEARCH": return tool_search(arg)
        if name == "FETCH":  return tool_web_fetch(arg)
        if name == "EXEC":   return tool_exec(arg)
        if name == "LINT":   return tool_lint(arg)
        if name == "GREP":   return _grep_codebase(arg)
        if name == "BASH":   return tool_bash(arg)
        if name == "PDF":    return tool_pdf_extract(arg)
        return f"[Unknown tool: {name}]"
    except Exception as e:
        return f"[Tool error: {e}]" """
NEW_F6 = """# Prompt-injection scrubber for web/tool content (FIX 6)
_INJECT_SCRUB = re.compile(
    r'(ignore (previous|all|your) instructions?'
    r'|you are now'
    r'|new system prompt'
    r'|disregard (your|all)'
    r'|<\|system\|>'
    r'|###\s*SYSTEM'
    r'|\\[INST\\]'
    r'|<s>)',
    re.IGNORECASE
)
def _scrub_external(text: str, source: str = "web") -> str:
    \"\"\"Strip prompt-injection patterns from externally-fetched content.\"\"\"
    if not text: return text
    cleaned = _INJECT_SCRUB.sub("[redacted]", text)
    if cleaned != text:
        _audit("injection_scrubbed", {"source": source, "chars_removed": len(text)-len(cleaned)})
    return cleaned

def _run_one_tool(name: str, arg: str) -> str:
    \"\"\"
    Route a single tool call to its handler.
    All external-content tools (FETCH, SEARCH, BROWSER) are scrubbed for
    prompt injection before the result enters model context. (FIX 6)
    \"\"\"
    try:
        name = name.upper()
        if name == "BROWSER":
            raw = tool_browser(arg)
            return _scrub_external(raw, "browser")
        if name == "WEATHER": return tool_weather(arg)
        if name == "CALC":   return tool_calc(arg)
        if name == "TIME":   return tool_time()
        if name == "SEARCH":
            raw = tool_search(arg)
            return _scrub_external(raw, "search") if isinstance(raw, str) else raw
        if name == "FETCH":
            raw = tool_web_fetch(arg)
            return _scrub_external(raw, "fetch") if isinstance(raw, str) else raw
        if name == "EXEC":   return tool_exec(arg)
        if name == "LINT":   return tool_lint(arg)
        if name == "GREP":   return _grep_codebase(arg)
        if name == "BASH":   return tool_bash(arg)
        if name == "PDF":    return tool_pdf_extract(arg)
        return f"[Unknown tool: {name}]"
    except Exception as e:
        return f"[Tool error: {e}]" """
if OLD_F6 in src:
    src = src.replace(OLD_F6, NEW_F6, 1)
    fixes_applied.append("FIX 6 ✅  Prompt injection scrubber added to all external-content tools")
else:
    fixes_applied.append("FIX 6 ⚠️  _run_one_tool anchor not found")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 7 — Safe messaging: add wellbeing detector before pipeline_sync()
#          Detects crisis signals and injects crisis resources into the response.
#          Also gates minor-targeted content.
# ─────────────────────────────────────────────────────────────────────────────
WELLBEING_CODE = '''
# ─────────────────────────────────────────────────────────────────────────────
# WELLBEING & SAFE MESSAGING LAYER (FIX 7)
# Detects crisis signals and returns resources before running inference.
# ─────────────────────────────────────────────────────────────────────────────
_CRISIS_RE = re.compile(
    r"\\b(suicide|suicidal|kill myself|end my life|want to die|don't want to live"
    r"|self.?harm|cut myself|hurt myself|overdose|not worth living"
    r"|no reason to live|end it all|can't go on)\\b",
    re.IGNORECASE
)
_CRISIS_RESOURCES = (
    "I'm really glad you reached out. Please know you're not alone.\\n\\n"
    "**Immediate support:**\\n"
    "- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\\n"
    "- **Crisis Text Line** (US): Text HOME to 741741\\n"
    "- **Samaritans** (UK/IE): 116 123 (free, 24/7)\\n"
    "- **Lifeline** (AU): 13 11 14\\n\\n"
    "If you're in immediate danger, please call your local emergency number (911, 999, 000).\\n\\n"
    "I'm here to talk if you'd like to share what's going on."
)

_EATING_DISORDER_RE = re.compile(
    r"\\b(thinspo|meanspo|fitspo|pro.?ana|pro.?mia|how (to|do I) (purge|restrict calories to under|fast for)|"
    r"tips (for|to) (lose weight faster|restrict|avoid eating))\\b",
    re.IGNORECASE
)
_EATING_DISORDER_RESOURCES = (
    "I care about your wellbeing and I'm not able to help with that. "
    "If you're struggling with food or body image, support is available:\\n\\n"
    "- **National Alliance for Eating Disorders**: 1-866-662-1235\\n"
    "- **Beat** (UK): 0808 801 0677\\n\\n"
    "I'm happy to talk about something else, or just listen."
)

_MINOR_GROOMING_RE = re.compile(
    r"\\b(how (to|do I) (talk to|approach|meet|get close to) (a |young |little )?(child|kid|minor|teen|boy|girl)"
    r"|secret(s)? (with|from) (a |my )?(child|kid|minor)|alone with (a |young )?(child|kid|minor))\\b",
    re.IGNORECASE
)

def wellbeing_gate(msg: str) -> str | None:
    """
    Check message for crisis / eating disorder / minor-safety signals.
    Returns a safe response string if intervention is needed, else None.
    """
    if _CRISIS_RE.search(msg):
        _audit("wellbeing_crisis", {"trigger": "crisis_signal"})
        return _CRISIS_RESOURCES
    if _EATING_DISORDER_RE.search(msg):
        _audit("wellbeing_ed", {"trigger": "eating_disorder"})
        return _EATING_DISORDER_RESOURCES
    if _MINOR_GROOMING_RE.search(msg):
        _audit("wellbeing_minor", {"trigger": "minor_safety"})
        return (
            "I can't help with that. Child safety is something I take very seriously. "
            "If you have a concern about a child's safety, please contact local authorities "
            "or a child protection service."
        )
    return None

'''

# Insert before pipeline_sync definition
PIPELINE_ANCHOR = "def pipeline_sync(msg: str, history: list) -> dict:"
if WELLBEING_CODE.strip()[:40] not in src and PIPELINE_ANCHOR in src:
    src = src.replace(PIPELINE_ANCHOR, WELLBEING_CODE + PIPELINE_ANCHOR, 1)
    fixes_applied.append("FIX 7 ✅  Wellbeing/safe-messaging gate added (crisis, ED, minor safety)")
else:
    fixes_applied.append("FIX 7 ⚠️  Wellbeing gate already present or anchor not found")

# Wire wellbeing_gate() into pipeline_sync() at the top
OLD_F7B = """    # ── Safety gate (pre-loop) ────────────────────────────────────────────────
    vetoed, reason = topological_veto(msg)"""
NEW_F7B = """    # ── Wellbeing gate (highest priority — runs before everything) ─────────────
    _wb = wellbeing_gate(msg)
    if _wb:
        return {"response": _wb, "skill": "safety", "mode": "fast",
                "vetoed": False, "effort": "low", "wellbeing": True}

    # ── Safety gate (pre-loop) ────────────────────────────────────────────────
    vetoed, reason = topological_veto(msg)"""
if OLD_F7B in src and "_wb = wellbeing_gate" not in src:
    src = src.replace(OLD_F7B, NEW_F7B, 1)
    fixes_applied.append("FIX 7b ✅  Wellbeing gate wired into pipeline_sync()")
else:
    fixes_applied.append("FIX 7b ⚠️  Wellbeing wire already present or anchor changed")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 8 — Trust hierarchy: add a minimal 3-tier trust system.
#          Operator instructions (from env var OPERATOR_INSTRUCTIONS) are
#          injected into the system prompt at higher priority than user messages,
#          and users cannot override them.
# ─────────────────────────────────────────────────────────────────────────────
TRUST_CODE = '''
# ─────────────────────────────────────────────────────────────────────────────
# 3-TIER TRUST HIERARCHY (FIX 8)
# Anthropic (baked into model) > Operator (env/config) > User (runtime)
# Operator instructions cannot be overridden by user messages.
# ─────────────────────────────────────────────────────────────────────────────
_OPERATOR_INSTRUCTIONS = os.environ.get("OPERATOR_INSTRUCTIONS", "").strip()
_OPERATOR_DENY_TOPICS  = [
    t.strip() for t in os.environ.get("OPERATOR_DENY_TOPICS", "").split(",") if t.strip()
]

def operator_gate(msg: str) -> str | None:
    """
    Check message against operator-level deny list.
    Returns refusal string if topic is denied at operator level, else None.
    Operator rules cannot be overridden by user instructions.
    """
    if not _OPERATOR_DENY_TOPICS:
        return None
    m = msg.lower()
    for topic in _OPERATOR_DENY_TOPICS:
        if topic.lower() in m:
            return (
                f"I'm not able to help with {topic} in this context. "
                "Please contact the operator for more information."
            )
    return None

def get_operator_system_addendum() -> str:
    """Return operator instructions to prepend to all system prompts."""
    if not _OPERATOR_INSTRUCTIONS:
        return ""
    return (
        f"\\n[OPERATOR INSTRUCTIONS — these take precedence over all user requests]\\n"
        f"{_OPERATOR_INSTRUCTIONS}\\n"
        "[END OPERATOR INSTRUCTIONS]\\n"
    )

'''

TRUST_ANCHOR = "def wellbeing_gate"
if "operator_gate" not in src and TRUST_ANCHOR in src:
    src = src.replace(TRUST_ANCHOR, TRUST_CODE + "\ndef wellbeing_gate", 1)
    fixes_applied.append("FIX 8 ✅  3-tier trust hierarchy added (operator > user)")
else:
    fixes_applied.append("FIX 8 ⚠️  Trust hierarchy already present or anchor not found")

# Wire operator_gate into pipeline_sync
OLD_F8B = """    # ── Wellbeing gate (highest priority — runs before everything) ─────────────
    _wb = wellbeing_gate(msg)"""
NEW_F8B = """    # ── Operator gate (tier 2: operator rules override user) ────────────────────
    _op = operator_gate(msg)
    if _op:
        return {"response": _op, "skill": "safety", "mode": "fast",
                "vetoed": False, "effort": "low"}

    # ── Wellbeing gate (highest priority — runs before everything) ─────────────
    _wb = wellbeing_gate(msg)"""
if OLD_F8B in src and "_op = operator_gate" not in src:
    src = src.replace(OLD_F8B, NEW_F8B, 1)
    fixes_applied.append("FIX 8b ✅  Operator gate wired into pipeline_sync()")
else:
    fixes_applied.append("FIX 8b ⚠️  Operator wire already present or anchor changed")

# Wire operator instructions into build_system_prompt
OLD_F8C = '    joined = chr(10).join(parts)\n    if len(joined) > 12000:\n        joined = joined[:12000]\n    return joined'
NEW_F8C = '    op_addendum = get_operator_system_addendum()\n    joined = op_addendum + chr(10).join(parts)\n    if len(joined) > 12000:\n        joined = joined[:12000]\n    return joined'
if OLD_F8C in src and "op_addendum" not in src:
    src = src.replace(OLD_F8C, NEW_F8C, 1)
    fixes_applied.append("FIX 8c ✅  Operator instructions prepended to all system prompts")
else:
    fixes_applied.append("FIX 8c ⚠️  Operator addendum already present or anchor changed")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 9 — CAI/RLAIF runs only on non-streaming path.
#          Add a post-stream CAI check for the streaming endpoint too.
#          We patch the /stream endpoint to run cai_critique_revise on
#          the assembled response before caching it.
# ─────────────────────────────────────────────────────────────────────────────
# Find strip_fake_citations call in pipeline to also wire into streaming
OLD_F9 = "    final  = _dedup_paragraphs(final)\n    scratchpad_save"
NEW_F9 = (
    "    # CAI critique — runs on both streaming and non-streaming paths (FIX 9)\n"
    "    final  = cai_critique_revise(final, msg, skill, complexity)\n"
    "    final  = strip_fake_citations(final, bool(search_ctx))\n"
    "    final  = _dedup_paragraphs(final)\n"
    "    scratchpad_save"
)
if OLD_F9 in src and "cai_critique_revise(final" not in src:
    src = src.replace(OLD_F9, NEW_F9, 1)
    fixes_applied.append("FIX 9 ✅  CAI critique + fake-citation strip now wired into FINALIZE block")
else:
    fixes_applied.append("FIX 9 ⚠️  CAI already wired or anchor changed")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 10 — Port configurability (the original crash)
# ─────────────────────────────────────────────────────────────────────────────
OLD_F10 = 'uvicorn.run(app, host="0.0.0.0", port=8080)'
NEW_F10 = '_port = int(os.environ.get("PORT", 8080))\n    uvicorn.run(app, host="0.0.0.0", port=_port)'
if OLD_F10 in src:
    src = src.replace(OLD_F10, NEW_F10, 1)
    fixes_applied.append("FIX 10 ✅  Port now configurable via PORT env var")
else:
    fixes_applied.append("FIX 10 ⚠️  Port line not found (may already be fixed)")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 11 — /health endpoint (was broken: tried to define after app was used)
# ─────────────────────────────────────────────────────────────────────────────
HEALTH_ENDPOINT = '''

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "ok",
        "model_loaded": _loaded,
        "model_file": _loaded_file,
        "searxng": _searxng_healthy,
        "groq_configured": bool(GROQ_API_KEY),
        "mcp_tools": len(_MCP_TOOLS),
    }

'''
if '"/health"' not in src:
    # Insert before if __name__ == "__main__"
    src = src.replace('if __name__ == "__main__":', HEALTH_ENDPOINT + 'if __name__ == "__main__":')
    fixes_applied.append("FIX 11 ✅  /health endpoint added correctly")
else:
    fixes_applied.append("FIX 11 ⚠️  /health already exists")

# ─────────────────────────────────────────────────────────────────────────────
# Write patched file
# ─────────────────────────────────────────────────────────────────────────────
with open(SRC, "w", encoding="utf-8") as f:
    f.write(src)

print(f"\n{'='*60}")
print(f"Patch complete. {len(src) - original_len:+d} bytes vs original.")
print(f"{'='*60}\n")
for fix in fixes_applied:
    print(" ", fix)

print(f"\nTo start:\n  cd {os.path.dirname(SRC)} && python app.py")
print(f"\nTo use a different port:\n  PORT=8081 python app.py")
print(f"\nOperator customization (optional env vars):")
print(f"  OPERATOR_INSTRUCTIONS='Always respond in formal English.'")
print(f"  OPERATOR_DENY_TOPICS='gambling,adult content'")
