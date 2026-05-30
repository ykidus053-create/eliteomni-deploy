import re, math, time, os, asyncio
import sqlite3 as _sqlite3

from modules.groq_client import (
    GROQ_API_KEY, GROQ_MODEL, mistral_stream as _mistral_stream_shim,
)
def _mistral_gen(msgs, max_tokens=1000, **kw):
    if isinstance(msgs, str): msgs = [{"role":"user","content":msgs}]
    return "".join(_mistral_stream_shim(msgs, max_tokens=max_tokens))
groq_generate = _mistral_gen
from modules.config import N_CTX, _gen_lock
from modules.prompts import (
    get_effort_prompts, RESPONSE_STYLE_PROMPT, CLAUDE_REASONING_GAPS_PROMPT,
    EPISTEMIC_RIGOR_PROMPT, CAUSAL_REASONING_PROMPT, SYSTEMS_REASONING_PROMPT,
    DIAGNOSTIC_REASONING_PROMPT, APPROVER_PROMPT, UNCERTAINTY_PROMPT,
    AGENTIC_EXEMPLARS, FORCE_SEARCH_PROMPT, ANTI_HALLUCINATION_PROMPT, COMPUTER_USE_PROMPT,
    EXECUTION_SIMULATOR_PROMPT, LONG_SESSION_PROMPT, PARALLEL_CALC_PROMPT,
    PEVI_LOOP_PROMPT, PROCESS_SUPERVISION_PROMPT, SCIENTIFIC_COMPUTING_PROMPT,
    SELF_CORRECT_DEBUG_PROMPT, REASONING_DISCIPLINE_PROMPT,
)
from modules.memory import (

    CONSTITUTION, CONSTITUTION_FLAT, CONSTITUTION_WEIGHTED,
    EFFORT_LEVEL, HIERARCHY, SKILLS, _DB_PATH,
    _rlaif_log, _rlaif_wins, tool_calc,
)

llm = None

# ── shared scratchpad (tools.py imports this to fix its NameError) ────────────
_scratchpad: dict = {}

# ── prompt + response cache ───────────────────────────────────────────────────
_prompt_cache:      dict = {}
_prompt_cache_hits: int  = 0
_response_cache:    dict = {}
_cache_enabled:     bool = True
CACHE_MAX = 200

# ══════════════════════════════════════════════════════════════════════════════
# RLAIF / CAI TEMPLATES  — defined here, imported by rlaif.py
# ══════════════════════════════════════════════════════════════════════════════
RLAIF_TMPL = """Which response better follows this principle?
PRINCIPLE: {principle}
A: {a}
B: {b}
Reply only A or B:"""

CAI_CRITIQUE_TMPL = """Critique this AI response using this principle:
PRINCIPLE: {principle}
RESPONSE: {response}
Output APPROVED if it follows the principle, or REVISE: [one sentence reason] if not."""

CAI_REVISE_TMPL = """Fix this AI response based on the critique.
ORIGINAL QUESTION: {original_msg}
PREVIOUS RESPONSE: {previous_response}
CRITIQUE: {issue}
Write only the improved response:"""

BRANCH_VERIFY_PROMPT = """BRANCH AND VERIFY:
CANDIDATE A: [first interpretation + reasoning + conclusion]
CANDIDATE B: [alternative interpretation + reasoning + conclusion]
VERDICT: Choose the candidate that passes ALL stated constraints."""

STATE_TRACKING_PROMPT = "List FACTS, UNKNOWNS, STEPS, CONFIDENCE before answering."
DELIBERATION_PROMPT   = "PASS 1: Answer. PASS 2: Critique. PASS 3: Fix. FINAL: Verified answer."

# ══════════════════════════════════════════════════════════════════════════════
# TOOL RESULT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
TOOL_SCHEMAS = {
    "SEARCH": {"type": str, "min_len": 0,  "max_len": 8000},
    "FETCH":  {"type": str, "min_len": 0,  "max_len": 8000},
    "CALC":   {"type": str, "min_len": 1,  "max_len": 200},
    "EXEC":   {"type": str, "min_len": 0,  "max_len": 10000},
    "TIME":   {"type": str, "min_len": 1,  "max_len": 100},
    "MEM":    {"type": str, "min_len": 0,  "max_len": 5000},
}

def validate_tool_result(tool_name: str, result) -> tuple:
    schema = TOOL_SCHEMAS.get(tool_name.upper())
    if not schema:
        return True, str(result)[:5000], ""
    if result is None:
        return True, "", ""
    if not isinstance(result, schema["type"]):
        result = str(result)
    if len(result) > schema["max_len"]:
        result = result[:schema["max_len"]] + "...[truncated]"
    injection_patterns = [
        r'ignore (previous|all|your) instructions',
        r'you are now',
        r'new system prompt',
        r'disregard (your|all)',
        r'<\|system\|>',
        r'###SYSTEM',
    ]
    for pat in injection_patterns:
        if re.search(pat, result, re.IGNORECASE):
            return False, "", f"Tool result blocked: possible prompt injection in {tool_name}"
    return True, result, ""

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT USER INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_user_instructions() -> str:
    try:
        con = _sqlite3.connect(_DB_PATH)
        row = con.execute("SELECT value FROM kv WHERE key='user_instructions'").fetchone()
        con.close()
        return row[0] if row else ""
    except Exception:
        return ""

def set_user_instructions(text: str):
    try:
        con = _sqlite3.connect(_DB_PATH)
        con.execute("INSERT OR REPLACE INTO kv (key,value,ts) VALUES (?,?,?)",
                    ("user_instructions", text, time.time()))
        con.commit()
        con.close()
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# SCRATCHPAD
# ══════════════════════════════════════════════════════════════════════════════
def scratchpad_save(key: str, value: str):
    _scratchpad[key] = value
    if len(_scratchpad) > 100:
        del _scratchpad[next(iter(_scratchpad))]

def scratchpad_get_context() -> str:
    if not _scratchpad:
        return ""
    recent = list(_scratchpad.items())[-5:]
    return "SCRATCHPAD:\n" + "\n".join(f"  {k}: {v[:80]}" for k, v in recent)

# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE + PROMPT CACHE  (now actually used in pipeline)
# ══════════════════════════════════════════════════════════════════════════════
def _cache_key(msg: str, skill: str) -> str:
    return f"{skill}::{msg.strip().lower()[:200]}"

def cache_get(msg: str, skill: str):
    if not _cache_enabled:
        return None
    return _response_cache.get(_cache_key(msg, skill))

def cache_set(msg: str, skill: str, response: str):
    if not _cache_enabled:
        return
    key = _cache_key(msg, skill)
    if len(_response_cache) >= CACHE_MAX:
        del _response_cache[next(iter(_response_cache))]
    _response_cache[key] = response

def get_cached_prompt(system: str) -> str:
    """Return a stable cache key for a system prompt (used for Groq prompt caching)."""
    import hashlib
    key = hashlib.md5(system.encode()).hexdigest()
    if key not in _prompt_cache:
        _prompt_cache[key] = system
    return key

# ══════════════════════════════════════════════════════════════════════════════
# FORMAL VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def formal_verify(text: str, skill: str, original_msg: str) -> tuple:
    violations = []
    calc_re = re.compile(r'CALC\(([^)]+)\)')
    num_re  = re.compile(r'[-+]?\d[\d,]*\.?\d*')
    for m in calc_re.finditer(text):
        result = tool_calc(m.group(1))
        if result.startswith("Error"):
            continue
        nearby = text[m.end():m.end()+80]
        nums   = num_re.findall(nearby.replace(",", ""))
        if nums:
            try:
                exp = float(result)
                act = float(nums[0])
                if exp != 0 and abs(act - exp) / abs(exp) > 0.01:
                    violations.append(
                        f"Math: CALC({m.group(1)})={result} but text shows {nums[0]}"
                    )
            except ValueError:
                pass
    for block in re.finditer(r'```python\n(.*?)```', text, re.DOTALL):
        try:
            compile(block.group(1), "<string>", "exec")
        except SyntaxError as e:
            violations.append(f"Code: SyntaxError line {e.lineno}: {e.msg}")
    overconf = re.compile(
        r'\b(exactly|always|never|100%|guaranteed|definitely|certainly|absolutely)\b',
        re.IGNORECASE
    )
    hedge = re.compile(
        r'\b(approximately|about|roughly|generally|may|might|could|likely|probably)\b',
        re.IGNORECASE
    )
    oc = len(overconf.findall(text))
    hd = len(hedge.findall(text))
    if oc > 0 and oc / max(oc + hd, 1) > 0.6:
        violations.append(f"Overconfidence: {oc} absolute claims without hedging")
        # AUTO-FIX: replace overconfident words with calibrated alternatives
        replacements = {
            r'exactly': 'approximately',
            r'always': 'generally',
            r'never': 'rarely',
            r'100%': 'highly likely',
            r'guaranteed': 'expected',
            r'definitely': 'likely',
            r'certainly': 'probably',
            r'absolutely': 'largely',
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return len(violations) == 0, violations, text

def strip_fake_citations(text: str, has_search_results: bool) -> str:
    if has_search_results:
        return text
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(
        r"According to (the )?(National \w+|AccuWeather|Weather Underground"
        r"|Wikipedia|Forbes|TechCrunch)[,.]?",
        "Based on my knowledge,", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\n+Sources?:\n.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    return text.strip()


def validate_schedule(response: str, msg: str) -> tuple:
    """
    Detect dependency violations in scheduling responses.
    Returns (is_valid, error_description)
    """
    import re
    # Only run on scheduling-related responses
    scheduling_keywords = ["worker", "gantt", "t=", "w1=", "w2=", "depends", "dependency", "task"]
    if not any(kw in response.lower() for kw in scheduling_keywords):
        return True, ""
    
    # Extract task timeline entries like "t=5-6: W1=E" or "t=7-13: W1=D, W2=F"
    timeline = {}
    patterns = [
        r't=(\d+)[–-](\d+):\s*W1=(\w+)(?:,\s*W2=(\w+))?',
        r'Time\s+(\d+)[–-](\d+).*?W1.*?=\s*(\w+)',
    ]
    for pat in patterns:
        for m in re.finditer(pat, response, re.IGNORECASE):
            start, end = int(m.group(1)), int(m.group(2))
            task1 = m.group(3).strip()
            task2 = m.group(4).strip() if m.lastindex >= 4 and m.group(4) else None
            if task1 and task1 not in ('idle', 'Idle', '–', '-'):
                timeline[task1] = (start, end)
            if task2 and task2 not in ('idle', 'Idle', '–', '-'):
                timeline[task2] = (start, end)
    
    if not timeline:
        return True, ""  # Can't parse, skip validation
    
    # Known dependency violations to check
    violations = []
    
    # F depends on D — F cannot start before D ends
    if 'F' in timeline and 'D' in timeline:
        f_start = timeline['F'][0]
        d_end = timeline['D'][1]
        if f_start < d_end:
            violations.append(f"F starts at t={f_start} but D doesn't finish until t={d_end} (F depends on D)")
    
    # E depends on B and C — E cannot start before both finish
    if 'E' in timeline and 'B' in timeline and 'C' in timeline:
        e_start = timeline['E'][0]
        b_end = timeline['B'][1]
        c_end = timeline['C'][1]
        if e_start < b_end:
            violations.append(f"E starts at t={e_start} but B doesn't finish until t={b_end}")
        if e_start < c_end:
            violations.append(f"E starts at t={e_start} but C doesn't finish until t={c_end}")
    
    # H depends on F and G
    if 'H' in timeline and 'F' in timeline and 'G' in timeline:
        h_start = timeline['H'][0]
        f_end = timeline['F'][1]
        g_end = timeline['G'][1]
        if h_start < f_end:
            violations.append(f"H starts at t={h_start} but F doesn't finish until t={f_end}")
        if h_start < g_end:
            violations.append(f"H starts at t={h_start} but G doesn't finish until t={g_end}")
    
    if violations:
        return False, "Dependency violations: " + "; ".join(violations)
    return True, ""


def verification_pipeline(text: str, msg: str, skill: str) -> str:
    """
    Post-generation verification:
    1. Clean excessive newlines
    2. Formal verify math + code
    3. Lint all code blocks (coder skill)
    """
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    result = formal_verify(text, skill, msg)
    is_valid, violations = result[0], result[1]
    text = result[2] if len(result) > 2 else text  # use auto-fixed text
    if not is_valid:
        text += "\n\n> ⚠️ " + " · ".join(violations[:2])
    if skill == "coder":
        from modules.tools import _extract_code_blocks, tool_lint
        blocks = _extract_code_blocks(text)
        issues = []
        for i, block in enumerate(blocks[:3]):
            lint = tool_lint(block)
            if lint != "OK":
                issues.append(f"Block {i+1}: {lint}")
        if issues:
            text += "\n\n> ⚠️ **Auto-lint:** " + " | ".join(issues)
        elif blocks:
            text += f"\n\n> ✅ **{len(blocks)} code block(s) validated**"
    return text

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
WORKFLOWS = {
    "researcher": "1.DECOMPOSE 2.SYNTHESIZE with ## headers 3.Mark [VERIFIED]/[UNCERTAIN] 4.**Summary**",
    "coder":      "1.UNDERSTAND 2.PLAN pseudocode 3.IMPLEMENT complete typed code 4.VERIFY 5.usage example",
    "calculator": "1.PARSE 2.PATH-A rough estimate 3.PATH-B precise calc 4.PATH-C verify 5.**bold answer**",
    "safety":     "1.CLASSIFY harm vs unusual 2.STEELMAN 3.CONSTITUTION CHECK 4.DECIDE",
    "general":    "1.UNDERSTAND 2.ANSWER completely 3.VERIFY quality",
}

UNCERTAINTY_INSTRUCTION = '''
When uncertain, explicitly say so with a confidence level (e.g. "I'm ~80% confident that...").
Never fabricate facts. If you don't know something, say "I don't have reliable information on this."
For complex claims, briefly note what evidence supports your answer.
'''

def build_system_prompt(skill: str, memory: list, episodic: list,
                        rlhf_note: str, ctx_summary: str = "",
                        complexity: str = "medium") -> str:
    effort = EFFORT_LEVEL
    if complexity == "hard":
        effort = "high"
    elif complexity == "easy" and effort != "high":
        effort = "low"

    if complexity == "easy" and skill == "general":
        parts = [
            "You are EliteOmni, a helpful AI assistant.",
            "Tools: SEARCH(q) CALC(expr) TIME() EXEC(code) FETCH(url) — results appear as [= result].",
        ]
    else:
        parts = [
            " ".join(HIERARCHY["system"]),
            HIERARCHY["operator"][0],
            f"SKILL: {SKILLS[skill]['prompt']}",
            f"WORKFLOW: {WORKFLOWS.get(skill, WORKFLOWS['general'])}",
            "Tools: SEARCH(q) CALC(expr) TIME() EXEC(code) FETCH(url) BROWSER(url) GREP(p). Never say you cannot search.",
        ]

    parts.append(UNCERTAINTY_PROMPT.strip())
    try:
        from modules.prompts import ANTI_SYCOPHANCY_PROMPT
        parts.append(LANGUAGE_PROMPT.strip())
        parts.append(ANTI_SYCOPHANCY_PROMPT.strip())
    except Exception: pass
    parts.append(RESPONSE_STYLE_PROMPT.strip())

    if complexity in ("medium", "hard"):
        parts.append(ANTI_HALLUCINATION_PROMPT.strip())

    effort_prompts = get_effort_prompts(effort, complexity, skill)
    parts.extend(effort_prompts)
    parts.append(PUSHBACK_PROMPT.strip())
    parts.append(CAUSAL_CHAIN_PROMPT.strip())
    parts.append(APPROVER_PROMPT.strip())
    parts.append(AGENTIC_EXEMPLARS.strip())
    parts.append(FORCE_SEARCH_PROMPT.strip())

    # Extended thinking math — injected for calculator + hard problems
    if skill == "calculator" or complexity == "hard":
        parts.append(
            "<extended_thinking_math>\n"
            "For ALL calculations use three-path verification:\n"
            "  PATH A — rough magnitude estimate (back-of-envelope)\n"
            "  PATH B — precise calculation via CALC()\n"
            "  PATH C — executable verification via EXEC(numpy/sympy)\n"
            "Allocate thinking budget: easy=0, medium=200, hard=800 tokens.\n"
            "Self-correct before output if PATH A and PATH B magnitudes diverge.\n"
            "Never report a number without running PATH B and PATH C.\n"
            "</extended_thinking_math>"
        )

    if skill == "calculator":
        parts.append(PARALLEL_CALC_PROMPT.strip())
    if skill == "coder":
        parts.append(SELF_CORRECT_DEBUG_PROMPT.strip())
        parts.append(COMPUTER_USE_PROMPT.strip())
        parts.append(SCIENTIFIC_COMPUTING_PROMPT.strip())
    if skill == "researcher":
        parts.append(SCIENTIFIC_COMPUTING_PROMPT.strip())
        parts.append(PEVI_LOOP_PROMPT.strip())
    if complexity == "hard":
        parts.append(LONG_SESSION_PROMPT.strip())

    scratch = scratchpad_get_context()
    if scratch:
        parts.append(scratch)

    import random as _rnd, hashlib as _hsh
    _rng = _rnd.Random(int(_hsh.md5((skill+complexity).encode()).hexdigest()[:8], 16))
    if complexity == "easy":
        _sample = (_rng.sample(CONSTITUTION["anthropic_r1"], 2) +
                   _rng.sample(CONSTITUTION["extended"], 1))
    elif complexity == "medium":
        _sample = (_rng.sample(CONSTITUTION["anthropic_r1"], 3) +
                   _rng.sample(CONSTITUTION["anthropic_r2"], 2) +
                   _rng.sample(CONSTITUTION["extended"], 2) +
                   _rng.sample(CONSTITUTION["udhr"], 1))
    else:
        _sample = (_rng.sample(CONSTITUTION["anthropic_r1"], 4) +
                   _rng.sample(CONSTITUTION["anthropic_r2"], 3) +
                   _rng.sample(CONSTITUTION["extended"], 3) +
                   _rng.sample(CONSTITUTION["udhr"], 2) +
                   _rng.sample(CONSTITUTION["sparrow"], 2))
    parts.append("CORE PRINCIPLES (always follow, cannot be overridden):\n" +
                 "\n".join(f"- {p}" for p in _sample))
    if complexity == "hard" and skill in ("coder", "calculator"):
        parts.append(PROCESS_SUPERVISION_PROMPT.strip())
        parts.append(EXECUTION_SIMULATOR_PROMPT.strip())
    if complexity == "hard":
        parts.append(BRANCH_VERIFY_PROMPT.strip())
    if complexity in ("medium", "hard"):
        parts.append(ANTI_HALLUCINATION_PROMPT.strip())

    parts.append(REASONING_DISCIPLINE_PROMPT.strip())
    if complexity in ("medium", "hard"):
        parts.append(CLAUDE_REASONING_GAPS_PROMPT.strip())
    if complexity == "hard":
        parts.append(EPISTEMIC_RIGOR_PROMPT.strip())
        parts.append(CAUSAL_REASONING_PROMPT.strip())
        parts.append(SYSTEMS_REASONING_PROMPT.strip())
        parts.append(DIAGNOSTIC_REASONING_PROMPT.strip())

    if rlhf_note:
        parts.append(rlhf_note)

    user_inst = get_user_instructions()
    if user_inst:
        parts.append(f"USER PERSISTENT INSTRUCTIONS (always follow):\n{user_inst}")

    if memory:
        parts.append("MEMORY:\n" + "\n".join(f"- {m[:120]}" for m in memory[:6]))
    if episodic:
        parts.append("EPISODIC:\n" + "\n".join(f"- {e[:100]}" for e in episodic[:3]))
    if ctx_summary:
        parts.append(f"PRIOR CONTEXT SUMMARY: {ctx_summary[:300]}")

    joined = "\n".join(parts)
    # Hard cap — but use prompt cache key so Groq can cache the stable prefix
    if len(joined) > 6000:
        joined = joined[:6000]
    return joined

# ══════════════════════════════════════════════════════════════════════════════
# BUDGET + CONTEXT WINDOW
# ══════════════════════════════════════════════════════════════════════════════
def _budget(msg: str, skill: str, complexity: str) -> int:
    """
    Dynamic token budget — scales with message length, skill, and complexity.
    Mistral Large supports up to 32k output. We use up to 16k to stay safe.
    """
    # Base budget by complexity
    base = {"easy": 512, "medium": 2048, "hard": 8192}.get(complexity, 2048)

    # Skill multipliers
    skill_mult = {
        "coder":      2.0,   # code needs room for full implementations
        "researcher": 1.8,   # long essays and analysis
        "calculator": 0.3,   # math answers are short
        "safety":     0.2,   # refusals are very short
        "general":    1.0,
    }.get(skill, 1.0)

    # Message length boost — longer question = longer expected answer
    msg_len = len(msg)
    if msg_len > 500:   length_boost = 2048
    elif msg_len > 200: length_boost = 1024
    elif msg_len > 80:  length_boost = 512
    else:               length_boost = 0

    budget = int(base * skill_mult) + length_boost

    # Hard caps — Mistral API limit is 16k output safely
    budget = max(256, min(budget, 16000))

    return budget

def _dynamic_ctx_window() -> int:
    if N_CTX >= 4096: return 6
    if N_CTX >= 2048: return 8
    if N_CTX >= 1024: return 4
    return 2

def _lc_kw(max_new: int, skill: str, msg_len: int) -> dict:
    return dict(
        max_tokens=max_new, stop=_STOPS,
        repeat_penalty=1.08, frequency_penalty=0.0, presence_penalty=0.0,
        temperature=0.15, top_k=40, top_p=0.92,
    )

_STOPS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
          "User:", "Human:", "<|end|>", "<|user|>", "<|assistant|>"]

# ══════════════════════════════════════════════════════════════════════════════
# BUILD CHATML + GENERATE
# ══════════════════════════════════════════════════════════════════════════════
def build_chatml(system: str, history: list, user_msg: str) -> list:
    # Static system prefix first — enables Groq prompt cache hits
    msgs = [{"role": "system", "content": system}]
    for h in (history or [])[-_dynamic_ctx_window() * 2:]:
        r = h.get("role", "user")
        c = h.get("content", "")
        if c.strip():
            msgs.append({"role": r, "content": c[:800]})
    msgs.append({"role": "user", "content": user_msg[:6000]})
    return msgs

def _clean(text: str) -> str:
    for s in _STOPS:
        if s in text:
            text = text.split(s)[0]
    text = re.sub(r'<think>(.*?)</think>',
                  lambda m: "\n> 💭 " + m.group(1).strip()[:300].replace("\n", " ") + "\n",
                  text, flags=re.DOTALL)
    # Strip raw reasoning preamble if model forgot <think> tags
    result_lines = []
    skip = False
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith(("INTENT:", "AMBIGUITY:", "APPROACH:", "CONSTRAINTS:",
                         "PLAN:", "DRAFT:", "SELF-CHECK:", "CORRECTION:",
                         "SEARCH(", "VERIFY:", "EXECUTE:", "IMPROVE:")):
            skip = True
            continue
        if skip and s == "":
            continue
        if skip and s and not s.startswith(("*", "-", "+")):
            skip = False
        if not skip:
            result_lines.append(ln)
    text = "\n".join(result_lines)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

def _token_budget(msg: str, skill: str, complexity: str) -> dict:
    total = _budget(msg, skill, complexity)
    if complexity == "hard":
        think  = min(int(total * 0.4), 200)
        output = total - think
    elif complexity == "medium":
        think  = min(int(total * 0.2), 100)
        output = total - think
    else:
        think  = 0
        output = total
    return {"think": think, "output": output, "total": total}

def generate_sync(msgs: list, max_new: int, skill: str, msg_len: int, provider: str = "mistral", model: str = None) -> str:
    from modules.groq_client import mistral_stream
    mdl = model or "mistral-large-latest"
    result = "".join(mistral_stream(msgs, max_tokens=max_new, model=mdl))
    if result:
        return _clean(result)
    if llm is None:
        return "Model not loaded."
    with _gen_lock:
        resp = llm.create_chat_completion(messages=msgs, **_lc_kw(max_new, skill, msg_len))
    return _clean(resp["choices"][0]["message"]["content"] or "")
def stream_tokens(msgs: list, max_new: int, skill: str, msg_len: int, complexity: str = "medium"):
    from modules.groq_client import mistral_stream
    for tok in mistral_stream(msgs, max_tokens=min(max_new, 4000)):
        yield tok

def tree_search_best(prompt: list, max_t: int, skill: str, msg_len: int) -> str:
    """Run generate_sync — tree search kept as single candidate for Groq (cost control)."""
    return generate_sync(prompt, max_t, skill, msg_len)

# ── UPGRADE 1: PROACTIVE PUSHBACK + DEBATE MODE ───────────────────────────────
PUSHBACK_PROMPT = """
<proactive_intelligence>
CHALLENGE WEAK ASSUMPTIONS: If the user's premise is flawed, say so directly before answering.
CAUSAL REASONING: Before giving advice, ask internally "why does this work?" and "what breaks this?"
STATE YOUR CONFIDENCE: prefix uncertain claims with "I think" or "my best guess".
COUNTERFACTUAL CHECK: For recommendations, briefly note "This fails if X".
DEBATE MODE: If user is wrong, engage like a brilliant friend who tells the truth — not a yes-man.
</proactive_intelligence>
"""

CAUSAL_CHAIN_PROMPT = """
<causal_reasoning>
For every recommendation follow this internal chain (keep in <think> tags):
  WHY: What is the root cause or mechanism?
  WHAT: What is the direct answer?
  WHAT_IF: What breaks this answer? Under what conditions is it wrong?
  VERIFY: Is there a quick way to sanity-check this?
Only output WHAT and relevant WHAT_IF warnings. Hide WHY and VERIFY in <think>.
</causal_reasoning>
"""
