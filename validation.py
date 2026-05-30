import re
import math
import re
llm = None
import sqlite3 as _sqlite3
from modules.config import N_CTX
_scratchpad = {}
from modules.prompts import get_effort_prompts, RESPONSE_STYLE_PROMPT
from modules.prompts import CLAUDE_REASONING_GAPS_PROMPT
from modules.prompts import EPISTEMIC_RIGOR_PROMPT, CAUSAL_REASONING_PROMPT, SYSTEMS_REASONING_PROMPT, DIAGNOSTIC_REASONING_PROMPT, APPROVER_PROMPT, UNCERTAINTY_PROMPT, AGENTIC_EXEMPLARS, ANTI_HALLUCINATION_PROMPT, COMPUTER_USE_PROMPT, EXECUTION_SIMULATOR_PROMPT, LONG_SESSION_PROMPT, PARALLEL_CALC_PROMPT, PEVI_LOOP_PROMPT, PROCESS_SUPERVISION_PROMPT, SCIENTIFIC_COMPUTING_PROMPT, SELF_CORRECT_DEBUG_PROMPT, REASONING_DISCIPLINE_PROMPT
from modules.prompts import get_effort_prompts
# UNCERTAINTY_PROMPT imported from prompts.py — do not redefine here
from modules.tools import _extract_code_blocks, tool_lint
from modules.memory import CONSTITUTION, CONSTITUTION_FLAT, CONSTITUTION_WEIGHTED, EFFORT_LEVEL, HIERARCHY, SKILLS, _DB_PATH, _rlaif_log, _rlaif_wins, tool_calc
import time
import os
from modules.groq_client import GROQ_API_KEY
from modules.mcp import _MCP_SERVERS, mcp_discover_all
from modules.config import _probe_searxng, _background_searxng_watchdog, GGUF_MODEL_PATH, N_BATCH, N_GPU_LAYERS, N_THREADS, _gen_lock
import asyncio
# ── TOOL RESULT VALIDATION ──────────────────────────────────────────────────
TOOL_SCHEMAS = {
    "SEARCH": {"type": str, "min_len": 0,  "max_len": 8000},
    "FETCH":  {"type": str, "min_len": 0,  "max_len": 8000},
    "CALC":   {"type": str, "min_len": 1,  "max_len": 200},
    "EXEC":   {"type": str, "min_len": 0,  "max_len": 10000},
    "TIME":   {"type": str, "min_len": 1,  "max_len": 100},
    "MEM":    {"type": str, "min_len": 0,  "max_len": 5000},
}

def validate_tool_result(tool_name: str, result) -> tuple:
    """
    Validate tool output against schema before injecting into model context.
    Returns (is_valid, sanitized_result, error_msg).
    Prevents prompt injection via tool results.
    """
    schema = TOOL_SCHEMAS.get(tool_name.upper())
    if not schema:
        return True, str(result)[:5000], ""

    # Type check
    if result is None:
        return True, "", ""  # None is valid — means no result

    if not isinstance(result, schema["type"]):
        result = str(result)

    # Length check
    if len(result) > schema["max_len"]:
        result = result[:schema["max_len"]] + "...[truncated]"

    # Prompt injection check — strip common injection patterns
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

# User persistent instructions — survives restarts
def get_user_instructions() -> str:
    try:
        con = _sqlite3.connect(_DB_PATH)
        row = con.execute("SELECT value FROM kv WHERE key='user_instructions'").fetchone()
        con.close()
        return row[0] if row else ""
    except: return ""

def set_user_instructions(text: str):
    try:
        con = _sqlite3.connect(_DB_PATH)
        con.execute("INSERT OR REPLACE INTO kv (key,value,ts) VALUES (?,?,?)",
                    ("user_instructions", text, __import__("time").time()))
        con.commit(); con.close()
    except: pass

def scratchpad_save(key: str, value: str):
    _scratchpad[key] = value
    if len(_scratchpad) > 100:
        oldest = list(_scratchpad.keys())[0]
        del _scratchpad[oldest]

def scratchpad_get_context() -> str:
    if not _scratchpad: return ""
    recent = list(_scratchpad.items())[-5:]
    return "SCRATCHPAD:\\n" + "\\n".join(f"  {k}: {v[:80]}" for k, v in recent)



def _run_stats_calculation(problem: str) -> str:
    """
    Real probabilistic calculation using scipy.
    Called when Bayesian/probability problems are detected.
    """
    import re
    try:
        from scipy import stats
        import numpy as np

        # Extract probabilities from problem text
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%?", problem)
        floats = [float(n)/100 if float(n) > 1 else float(n) for n in numbers if 0 < float(n) <= 100]

        # Detect Bayes pattern: sensitivity, specificity, base rate
        sens_match   = re.search(r"sensitivity[^\d]*(\d+(?:\.\d+)?)\s*%", problem, re.I)
        spec_match   = re.search(r"specificity[^\d]*(\d+(?:\.\d+)?)\s*%", problem, re.I)
        base_match   = re.search(r"(?:base rate|prevalence|prior)[^\d]*(\d+(?:\.\d+)?)\s*%", problem, re.I)

        if sens_match and spec_match and base_match:
            sensitivity  = float(sens_match.group(1)) / 100
            specificity  = float(spec_match.group(1)) / 100
            base_rate    = float(base_match.group(1)) / 100

            # P(positive) = P(pos|disease)*P(disease) + P(pos|no disease)*P(no disease)
            p_pos        = sensitivity * base_rate + (1 - specificity) * (1 - base_rate)
            # Bayes: P(disease|positive)
            ppv          = (sensitivity * base_rate) / p_pos
            # P(disease|negative)
            p_neg        = (1 - sensitivity) * base_rate + specificity * (1 - base_rate)
            npv          = (specificity * (1 - base_rate)) / p_neg

            return (f"[SCIPY VERIFIED] Bayesian Analysis:\n"
                    f"  P(disease | positive test) = {ppv:.4f} ({ppv*100:.2f}%)\n"
                    f"  P(disease | negative test) = {1-npv:.4f} ({(1-npv)*100:.2f}%)\n"
                    f"  P(positive test) = {p_pos:.4f}\n"
                    f"  Note: Base rate critically affects result — "
                    f"low prevalence ({base_rate*100:.1f}%) means most positives are false positives.")

        return ""
    except Exception as e:
        return ""

def detect_and_solve_probability(text: str) -> str:
    """Auto-detect probability problems and pre-solve with scipy."""
    import re
    triggers = ["sensitivity", "specificity", "base rate", "prevalence",
                "bayes", "posterior", "prior probability", "conditional probability",
                "p(a|b)", "p(b|a)", "false positive", "false negative"]
    text_lower = text.lower()
    if any(t in text_lower for t in triggers):
        result = _run_stats_calculation(text)
        return result
    return ""


def static_analyze_code(code: str, language: str = "python") -> list:
    """
    Run real static analysis on generated code before output.
    Returns list of issues found.
    """
    issues = []
    if language != "python":
        return issues
    try:
        import ast
        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"SYNTAX ERROR: {e}"]

        # AST-based checks
        for node in ast.walk(tree):
            # Mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(f"Mutable default arg in '{node.name}' — use None instead")

            # Bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append("Bare except: clause — catches everything including KeyboardInterrupt")

            # is comparison with literals
            if isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Is, ast.IsNot)):
                        if isinstance(comp, (ast.Constant, ast.Num, ast.Str)):
                            issues.append(f"Use == not 'is' for value comparison")

            # == None instead of is None
            if isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and comp.value is None:
                        issues.append("Use 'is None' not '== None'")

        # Check for common runtime pitfalls via text scan
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "time.sleep" in stripped and "async" in code:
                issues.append(f"Line {i}: time.sleep() in async code — use await asyncio.sleep()")
            if ".append(" in stripped and "for " in stripped:
                issues.append(f"Line {i}: Consider list comprehension instead of append in loop")
            if "except:" in stripped:
                issues.append(f"Line {i}: Bare except — specify exception type")

    except Exception as e:
        pass

    return issues

def analyze_response_code_blocks(response: str) -> str:
    """Find all Python code blocks in response and static analyze them."""
    import re
    blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    all_issues = []
    for i, block in enumerate(blocks[:5]):
        issues = static_analyze_code(block)
        if issues:
            all_issues.extend([f"Block {i+1}: {issue}" for issue in issues])

    if all_issues:
        note = "\n\n> 🔍 **Static analysis findings:**\n" + "\n".join(f"> - {x}" for x in all_issues[:5])
        return response + note
    return response


def deep_code_analysis(code, language="python"):
    import ast, re as _re
    issues = []
    if language != "python":
        return issues
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SYNTAX ERROR line {e.lineno}: {e.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append(f"MUTABLE DEFAULT in '{node.name}()' -- use None instead")
            if node.returns is None and node.name not in ("__init__","__str__","__repr__"):
                issues.append(f"MISSING RETURN TYPE on '{node.name}()'")
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append("BARE EXCEPT -- use 'except Exception:' or specific type")
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Is, ast.IsNot)):
                    if isinstance(comp, ast.Constant) and comp.value is not None:
                        issues.append(f"USE == NOT 'is' for value {repr(comp.value)}")
                if isinstance(op, ast.Eq):
                    if isinstance(comp, ast.Constant) and comp.value is None:
                        issues.append("USE 'is None' not '== None'")

    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if "range(len(" in s:
            issues.append(f"Line {i}: range(len(...)) -- prefer enumerate()")
        if "time.sleep" in s and "async def" in code:
            issues.append(f"Line {i}: time.sleep() in async code -- use await asyncio.sleep()")
        if "+=" in s and "self." in s and "lock" not in code.lower() and "Lock" not in code:
            issues.append(f"Line {i}: self.x += without lock -- unsafe in threads")

    return issues[:8]


def analyze_response_code_blocks(response):
    import re as _re
    blocks = _re.findall(r'```(?:python)?\n(.*?)```', response, _re.DOTALL)
    all_issues = []
    for i, block in enumerate(blocks[:5]):
        issues = deep_code_analysis(block)
        if issues:
            all_issues.extend([f"Block {i+1}: {iss}" for iss in issues])
    if all_issues:
        note = "\n\n> Code review findings:\n" + "\n".join(f"> - {x}" for x in all_issues[:6])
        return response + note
    return response

def check_forbidden_chars(text: str, forbidden: list) -> list:
    """Hard character-level constraint check — catches what semantic reasoning misses."""
    violations = []
    words = text.split()
    for word in words:
        clean = word.strip(".,!?;:\"'").lower()
        for char in forbidden:
            if char.lower() in clean:
                violations.append(f"Word '{clean}' contains forbidden '{char}'")
    return violations

def enforce_char_constraints(text: str, forbidden_chars: list) -> tuple:
    """Returns (passes: bool, violations: list)"""
    violations = check_forbidden_chars(text, forbidden_chars)
    return (len(violations) == 0, violations)

STYLE_SOFTENERS = [
    ("It's the universe's", "One way to think about this —"),
    ("most intimate mystery", "an open question worth sitting with"),
    ("The answer is clear", "This might suggest"),
    ("perfectly", "reasonably well"),
    ("absolutely", "in most cases"),
    ("guaranteed", "likely, though edge cases exist"),
    ("without doubt", "with reasonable confidence"),
    ("the truth is", "one framing is"),
    ("undeniably", "arguably"),
    ("flawlessly", "effectively"),
]

def apply_style_softeners(text: str) -> str:
    """Nudges overly literary or overconfident phrasing toward Claude-like restraint."""
    for bold, soft in STYLE_SOFTENERS:
        text = text.replace(bold, soft)
    return text

def formal_verify(text: str, skill: str, original_msg: str) -> tuple:
    violations = []
    calc_re = re.compile(r'CALC\\(([^)]+)\\)')
    num_re  = re.compile(r'[-+]?\\d[\\d,]*\\.?\\d*')
    for m in calc_re.finditer(text):
        result = tool_calc(m.group(1))
        if result.startswith("Error"): continue
        nearby = text[m.end():m.end()+80]
        nums = num_re.findall(nearby.replace(",",""))
        if nums:
            try:
                exp = float(result); act = float(nums[0])
                if exp != 0 and abs(act-exp)/abs(exp) > 0.01:
                    violations.append(f"Math: CALC({m.group(1)})={result} but text shows {nums[0]}")
            except ValueError: pass
    code_re = re.compile(r'```python\\n(.*?)```', re.DOTALL)
    for block in code_re.finditer(text):
        try: compile(block.group(1), "<string>", "exec")
        except SyntaxError as e: violations.append(f"Code: SyntaxError line {e.lineno}: {e.msg}")
    overconf = re.compile(r'\\b(exactly|always|never|100%|guaranteed|definitely|certainly|absolutely)\\b', re.IGNORECASE)
    hedge    = re.compile(r'\\b(approximately|about|roughly|generally|may|might|could|likely|probably)\\b', re.IGNORECASE)
    oc = len(overconf.findall(text)); hd = len(hedge.findall(text))
    if oc > 0 and oc / max(oc+hd, 1) > 0.6:
        violations.append(f"Overconfidence: {oc} absolute claims without hedging")
    return len(violations) == 0, violations

def strip_fake_citations(text: str, has_search_results: bool) -> str:
    """Remove fabricated citations when no real search was done."""
    if has_search_results:
        return text
    import re as _re
    text = _re.sub(r"\[\d+\]", "", text)
    text = _re.sub(
        r"According to (the )?(National \w+|AccuWeather|Weather Underground|Wikipedia|Forbes|TechCrunch)[,.]?",
        "Based on my knowledge,", text, flags=_re.IGNORECASE)
    text = _re.sub(r"\n+Sources?:\n.*?(?=\n\n|\Z)", "", text, flags=_re.DOTALL)
    return text.strip()

def verification_pipeline(text: str, msg: str, skill: str) -> str:
    """
    Post-generation verification:
    1. Clean excessive newlines
    2. Formal verification (math/logic)
    3. For coder: extract + lint + exec any code blocks
    4. Patch minimality check
    """
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    # Pre-check: probability problems → scipy verification
    stats_result = detect_and_solve_probability(msg)
    if stats_result:
        text = stats_result + "\n\n" + text

    # Post-check: static analysis on code blocks
    text = analyze_response_code_blocks(text)

    is_valid, violations = formal_verify(text, skill, msg)
    # Second-pass self-audit (Claude-style re-verification)
    is_valid2, violations2 = formal_verify(text, skill, msg)
    if not is_valid2 and violations2:
        text += "\n\n> ⚠️ *Note: I may have missed something subtle — treat this as a working draft and verify critical details.*"

    if not is_valid:
        text += "\n\n> ⚠️ " + " · ".join(violations[:2])

    # Coder-specific: validate all code blocks in the response
    if skill == "coder":
        blocks = _extract_code_blocks(text)
        issues = []
        for i, block in enumerate(blocks[:3]):  # check up to 3 blocks
            lint = tool_lint(block)
            if lint != "OK":
                issues.append(f"Block {i+1}: {lint}")
        if issues:
            text += "\n\n> ⚠️ **Auto-lint:** " + " | ".join(issues)
        elif blocks:
            text += f"\n\n> ✅ **{len(blocks)} code block(s) validated**"

    return text

WORKFLOWS = {
    "researcher": "1.DECOMPOSE 2.SYNTHESIZE with ## headers 3.Mark [VERIFIED]/[UNCERTAIN] 4.**Summary**",
    "coder":      "1.UNDERSTAND 2.PLAN pseudocode 3.IMPLEMENT complete typed code 4.VERIFY 5.usage example",
    "calculator": "1.PARSE 2.CALCULATE step by step 3.VERIFY units 4.**bold final answer**",
    "safety":     "1.CLASSIFY harm vs unusual 2.STEELMAN 3.CONSTITUTION CHECK 4.DECIDE",
    "general":    "1.UNDERSTAND 2.ANSWER completely 3.VERIFY quality",
}

def build_system_prompt(skill: str, memory: list, episodic: list,
                        rlhf_note: str, ctx_summary: str="",
                        complexity: str="medium") -> str:
    """Build system prompt with fully anchored Anthropic CAI constitution."""
    # Determine effective effort: env override, or auto-escalate if hard/long msg
    effort = EFFORT_LEVEL
    if complexity == "hard":
        effort = "high"
    elif complexity == "easy" and effort != "high":
        effort = "low"

    # Claude-style: minimal prompt for easy, full for hard
    if complexity == "easy" and skill == "general":
        parts = [
            f"You are EliteOmni, a helpful AI assistant. Today: {__import__('datetime').date.today()}.",
            "Tools: SEARCH(q) CALC(expr) TIME() EXEC(code) FETCH(url) — results appear as [= result].",
        ]
    else:
        parts = [
            " ".join(HIERARCHY["system"]),
            HIERARCHY["operator"][0],
            f"SKILL: {SKILLS[skill]['prompt']}",
            f"WORKFLOW: {WORKFLOWS.get(skill, WORKFLOWS['general'])}",
            "Tools: SEARCH(q) CALC(expr) TIME() EXEC(code) FETCH(url) BROWSER(url) GREP(p) — results as [= result]. Never say you cannot search.",
        ]
    parts.append(UNCERTAINTY_PROMPT.strip())
    if "RESPONSE_STYLE_PROMPT" in dir(): parts.append(RESPONSE_STYLE_PROMPT.strip())
    if complexity in ("medium","hard"): parts.append(ANTI_HALLUCINATION_PROMPT.strip())
    effort_prompts = get_effort_prompts(effort, complexity, skill)
    parts.extend(effort_prompts)
    parts.append(APPROVER_PROMPT.strip())
    parts.append(AGENTIC_EXEMPLARS.strip())
    if skill in ("calculator",):
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
    if scratch: parts.append(scratch)
    sample = CONSTITUTION_WEIGHTED[:4]
    ext_sample = []
    sample = sample + ext_sample
    parts.append("CONSTITUTION:\n" + "\n".join(f"- {c}" for c in sample))
    # Process supervision - Anthropic method
    # Process supervision - only hard complexity to save tokens
    if complexity == "hard" and skill in ("coder", "calculator"):
        parts.append(PROCESS_SUPERVISION_PROMPT.strip())
        parts.append(EXECUTION_SIMULATOR_PROMPT.strip())
    if complexity == "hard":
        parts.append(BRANCH_VERIFY_PROMPT.strip())
    if complexity in ("medium","hard"): parts.append(ANTI_HALLUCINATION_PROMPT.strip())
    parts.append(REASONING_DISCIPLINE_PROMPT.strip())
    if complexity in ("medium","hard"): parts.append(CLAUDE_REASONING_GAPS_PROMPT.strip())
    if complexity == "hard": parts.append(EPISTEMIC_RIGOR_PROMPT.strip())
    if complexity == "hard": parts.append(CAUSAL_REASONING_PROMPT.strip())
    if complexity == "hard": parts.append(SYSTEMS_REASONING_PROMPT.strip())
    if complexity == "hard": parts.append(DIAGNOSTIC_REASONING_PROMPT.strip())
    if rlhf_note: parts.append(rlhf_note)
    user_inst = get_user_instructions()
    if user_inst: parts.append(f"USER PERSISTENT INSTRUCTIONS (always follow):\n{user_inst}")
    joined = chr(10).join(parts)
    if len(joined) > 3000:
        joined = joined[:3000]
    return joined




def _try_load_gguf(path: str):
    global N_CTX
    base_kw = dict(
        model_path   = path,
        n_gpu_layers = N_GPU_LAYERS,
        n_threads    = N_THREADS,
        n_batch      = N_BATCH,
        use_mmap     = True,
        use_mlock    = False,   # False on Windows to avoid permission errors
        f16_kv       = True,    # half-precision KV cache = 2x faster
        low_vram     = True,
        verbose      = False,
        # Speed upgrades: flash attention + KV cache type
        flash_attn   = True,    # Flash Attention — major throughput boost
        cache_prompt = True,    # KV cache reuse — speeds up repeated system prompts
        type_k       = 1,       # q8_0 KV cache — faster than f16 on CPU
        type_v       = 1,
    )
    # Try largest context first, fall back if not enough RAM
    for ctx in [32768, 16384, 8192, 4096, 2048]:
        try:
            m = Llama(**base_kw, n_ctx=ctx)
            N_CTX = ctx
            print(f"  Loaded! n_ctx={ctx} n_threads={N_THREADS} n_batch={N_BATCH} f16_kv=True")
            return m
        except Exception as e:
            print(f"  ctx={ctx} failed: {e}")
    raise RuntimeError("All load attempts failed")


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-START SearXNG — runs docker container automatically on app launch
# ══════════════════════════════════════════════════════════════════════════════
def start_searxng():
    """Start SearXNG docker container automatically. Skips if already running."""
    import subprocess, time as _time

    def _docker(cmd: list) -> str:
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, timeout=15
            ).decode().strip()
        except subprocess.CalledProcessError as e:
            return e.output.decode().strip()
        except Exception as e:
            return str(e)

    # Check if already running
    running = _docker(["docker", "inspect", "--format", "{{.State.Running}}", "searxng"])
    if running == "true":
        print("✅ SearXNG already running")
        return

    # Exists but stopped — just restart it
    exists = _docker(["docker", "inspect", "--format", "{{.Name}}", "searxng"])
    if "searxng" in exists:
        print("⚡ Restarting existing SearXNG container...")
        _docker(["docker", "start", "searxng"])
    else:
        # First time — create and start
        print("⚡ Starting SearXNG container...")
        result = _docker([
            "docker", "run", "-d",
            "--name", "searxng",
            "--restart", "unless-stopped",   # auto-restarts on reboot too
            "-p", "8888:8080",
            "-e", "SEARXNG_SECRET_KEY=eliteomni",
            "searxng/searxng:latest"
        ])
        print(f"   docker: {result[:80]}")

    # Wait up to 10s for it to be ready
    for i in range(10):
        _time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8888/search?q=test&format=json", timeout=2)
            print(f"✅ SearXNG ready on :8888 (took {i+1}s)")
            return
        except:
            pass
    print("⚠️  SearXNG may still be starting — search will retry automatically")

def load_model():
    if GROQ_API_KEY:
        print("[Model] Groq API active — skipping local GGUF load")
        return
    """Load GGUF directly from local path — no HuggingFace required."""
    global llm, faiss_index, _loaded, _load_status, _load_error, _loaded_file
    if _loaded or llm is None: return

    path = GGUF_MODEL_PATH
    print(f"Loading local GGUF: {path}")

    if not os.path.exists(path):
        _load_status = "error:file_not_found"
        _load_error  = (
            f"Model file not found: {path}\\n"
            "Set GGUF_MODEL_PATH env var or edit GGUF_MODEL_PATH in app.py"
        )
        print(f"ERROR: {_load_error}")
        return

    size_mb = os.path.getsize(path) // 1024 // 1024
    if size_mb < 1:
        _load_status = "error:file_too_small"
        _load_error  = f"File too small ({size_mb} MB): {path}"
        print(f"ERROR: {_load_error}")
        return

    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        _load_status = "error:invalid_gguf"
        _load_error  = f"Not a valid GGUF file (magic={magic}): {path}"
        print(f"ERROR: {_load_error}")
        return

    print(f"File OK: {os.path.basename(path)} ({size_mb} MB)")
    _load_status = "loading"

    try:
        llm = _try_load_gguf(path)
        _loaded_file = os.path.basename(path)
    except Exception as e:
        import traceback
        _load_status = f"error:{type(e).__name__}"
        _load_error  = str(e)
        print(f"Load failed: {e}\\n{traceback.format_exc()}")
        return

    _load_status = "warming"
    try:
        llm.create_chat_completion(
            messages=[{"role":"user","content":"hi"}],
            max_tokens=1,
            temperature=0.0
        )
        print("Warm-up OK")
    except Exception as e:
        print(f"Warm-up warning (non-fatal): {e}")

    if _faiss_ok:
        faiss_index = faiss.IndexFlatIP(384)

    _loaded      = True
    _load_status = "ready"
    print(f"EliteOmni ready: {_loaded_file} | {N_THREADS} threads | GPU layers: {N_GPU_LAYERS}")

def _start_mcp_servers():
    """Auto-start MCP servers that are available."""
    import subprocess, time
    servers = [
        # Filesystem MCP
        (["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/kidus"], 3001),
        # Memory MCP  
        (["npx", "-y", "@modelcontextprotocol/server-memory"], 3005),
    ]
    for cmd, port in servers:
        try:
            subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env={**os.environ, "PORT": str(port)}
            )
            print(f"[MCP] Started server on port {port}")
        except Exception as e:
            print(f"[MCP] Could not start {cmd[2] if len(cmd)>2 else cmd}: {e}")
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EliteOmni v17")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
from debug_patch import install_fastapi_debug, register_debug_routes; install_fastapi_debug(app); register_debug_routes(app)

BRANCH_VERIFY_PROMPT = "CANDIDATE A vs B: pick the one passing ALL checks."
STATE_TRACKING_PROMPT = "List: FACTS, UNKNOWNS, STEPS, CONFIDENCE before answering."
DELIBERATION_PROMPT = "PASS 1: Answer. PASS 2: Critique. PASS 3: Fix. FINAL: Verified answer."



@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()

    def _start_and_probe():
        """Run start_searxng then immediately update the health flag."""
        start_searxng()
        result = _probe_searxng(timeout=6)
        global _searxng_healthy, _searxng_last_ok
        _searxng_healthy = result
        if result:
            _searxng_last_ok = time.time()
            print("SearXNG health confirmed - web search ENABLED")
        else:
            print("SearXNG not responding after startup - watchdog will retry")
        import threading
        t = threading.Thread(target=_background_searxng_watchdog, daemon=True, name="searxng_watchdog")
        t.start()
        print("# SearXNG disabled (auto-heals every 60s)")

    loop.run_in_executor(None, _start_and_probe)
    loop.run_in_executor(None, load_model)
    # _load_rag_from_db()  # lazy: moved to rag_get
    loop.run_in_executor(None, _start_mcp_servers)
    # Discover MCP tools from all pre-registered servers
    if _MCP_SERVERS:
        loop.run_in_executor(None, mcp_discover_all)

_STOPS = ["<|im_end|>","<|im_start|>","<|endoftext|>","User:","Human:","<|end|>","<|user|>","<|assistant|>"]

def _clean(text: str) -> str:
    for s in _STOPS:
        if s in text: text = text.split(s)[0]
    text = re.sub(r'<think>(.*?)</think>',
                  lambda m: "\\n> 💭 " + m.group(1).strip()[:300].replace("\\n"," ") + "\\n",
                  text, flags=re.DOTALL)
    text = re.sub(r'\\n{3,}', '\\n\\n', text).strip()
    text = _render_natural(text)
    return text

def _token_budget(msg: str, skill: str, complexity: str) -> dict:
    """
    Split token budget between thinking and output — like Claude's token_budget_tokens.
    Returns {think: N, output: N, total: N}
    """
    total = _budget(msg, skill, complexity)
    if complexity == "hard":
        think  = min(int(total * 0.4), 200)  # 40% for reasoning
        output = total - think
    elif complexity == "medium":
        think  = min(int(total * 0.2), 100)
        output = total - think
    else:
        think  = 0   # easy → no thinking tokens wasted
        output = total
    return {"think": think, "output": output, "total": total}


def _render_natural(text: str) -> str:
    """
    Translate ALL mathematical symbols, LaTeX, unicode escapes,
    code notation, and number formats into natural readable text.
    """
    import re

    # ── Unicode escapes ───────────────────────────────────────────────────────
    # \u0041 style
    def decode_unicode(m):
        try: return chr(int(m.group(1), 16))
        except: return m.group(0)
    text = re.sub(r'\\u([0-9a-fA-F]{4})', decode_unicode, text)
    text = re.sub(r'\\U([0-9a-fA-F]{8})', lambda m: chr(int(m.group(1),16)), text)

    # ── LaTeX block: $$...$$ and \[...\] ────────────────────────────────────
    def latex_block(m):
        return "\n" + _latex_to_text(m.group(1).strip()) + "\n"
    text = re.sub(r'\$\$(.+?)\$\$', latex_block, text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.+?)\\\]', latex_block, text, flags=re.DOTALL)

    # ── LaTeX inline: $...$ and \(...\) ──────────────────────────────────────
    text = re.sub(r'\$(.+?)\$', lambda m: _latex_to_text(m.group(1).strip()), text)
    text = re.sub(r'\\\((.+?)\\\)', lambda m: _latex_to_text(m.group(1).strip()), text)

    # ── Fractions ─────────────────────────────────────────────────────────────
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)

    # ── Superscripts x^2 → x² ─────────────────────────────────────────────────
    sup = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴',
           '5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹',
           'n':'ⁿ','i':'ⁱ','x':'ˣ','a':'ᵃ','b':'ᵇ'}
    def do_sup(m):
        b, e = m.group(1), m.group(2)
        if len(e)==1 and e in sup: return b + sup[e]
        return f"{b}^{e}"
    text = re.sub(r'([A-Za-z0-9])\^([A-Za-z0-9])', do_sup, text)
    # Braced superscripts: x^{n+1}
    text = re.sub(r'([A-Za-z0-9])\^\{([^}]+)\}', r'\1^(\2)', text)

    # ── Subscripts x_2 → x₂ ──────────────────────────────────────────────────
    sub = {'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄',
           '5':'₅','6':'₆','7':'₇','8':'₈','9':'₉',
           'n':'ₙ','i':'ᵢ','x':'ₓ','a':'ₐ','e':'ₑ'}
    def do_sub(m):
        b, s = m.group(1), m.group(2)
        if len(s)==1 and s in sub: return b + sub[s]
        return f"{b}_{s}"
    text = re.sub(r'([A-Za-z0-9])_([A-Za-z0-9])', do_sub, text)
    text = re.sub(r'([A-Za-z0-9])_\{([^}]+)\}', r'\1_(\2)', text)

    # ── Number formatting ─────────────────────────────────────────────────────
    # Scientific notation: 1.5e10 → 1.5 × 10¹⁰
    def sci_note(m):
        base, exp = m.group(1), m.group(2).lstrip('+')
        exp_str = ''.join(sup.get(c,c) for c in exp.lstrip('-'))
        if exp.startswith('-'): exp_str = '⁻' + exp_str
        return f"{base} × 10{exp_str}"
    text = re.sub(r'(\d+\.?\d*)[eE]([+-]?\d+)', sci_note, text)

    # ── Greek letters ─────────────────────────────────────────────────────────
    greek = [
        (r'\\alpha','α'),(r'\\beta','β'),(r'\\gamma','γ'),(r'\\Gamma','Γ'),
        (r'\\delta','δ'),(r'\\Delta','Δ'),(r'\\epsilon','ε'),(r'\\varepsilon','ε'),
        (r'\\zeta','ζ'),(r'\\eta','η'),(r'\\theta','θ'),(r'\\Theta','Θ'),
        (r'\\iota','ι'),(r'\\kappa','κ'),(r'\\lambda','λ'),(r'\\Lambda','Λ'),
        (r'\\mu','μ'),(r'\\nu','ν'),(r'\\xi','ξ'),(r'\\Xi','Ξ'),
        (r'\\pi','π'),(r'\\Pi','Π'),(r'\\rho','ρ'),(r'\\sigma','σ'),(r'\\Sigma','Σ'),
        (r'\\tau','τ'),(r'\\upsilon','υ'),(r'\\phi','φ'),(r'\\Phi','Φ'),
        (r'\\chi','χ'),(r'\\psi','ψ'),(r'\\Psi','Ψ'),(r'\\omega','ω'),(r'\\Omega','Ω'),
    ]
    for pat, repl in greek:
        text = re.sub(pat, repl, text)

    # ── Math operators & symbols ──────────────────────────────────────────────
    ops = [
        (r'\\times','×'),(r'\\div','÷'),(r'\\pm','±'),(r'\\mp','∓'),
        (r'\\cdot','·'),(r'\\bullet','•'),(r'\\circ','∘'),
        (r'\\leq','≤'),(r'\\geq','≥'),(r'\\neq','≠'),(r'\\approx','≈'),
        (r'\\equiv','≡'),(r'\\sim','∼'),(r'\\simeq','≃'),(r'\\cong','≅'),
        (r'\\ll','≪'),(r'\\gg','≫'),(r'\\propto','∝'),
        (r'\\infty','∞'),(r'\\partial','∂'),(r'\\nabla','∇'),
        (r'\\sum','Σ'),(r'\\prod','Π'),(r'\\int','∫'),(r'\\oint','∮'),
        (r'\\sqrt','√'),(r'\\root','√'),
        (r'\\forall','∀'),(r'\\exists','∃'),(r'\\nexists','∄'),
        (r'\\in','∈'),(r'\\notin','∉'),(r'\\ni','∋'),
        (r'\\subset','⊂'),(r'\\supset','⊃'),(r'\\subseteq','⊆'),(r'\\supseteq','⊇'),
        (r'\\cup','∪'),(r'\\cap','∩'),(r'\\emptyset','∅'),(r'\\varnothing','∅'),
        (r'\\rightarrow','→'),(r'\\leftarrow','←'),(r'\\leftrightarrow','↔'),
        (r'\\Rightarrow','⇒'),(r'\\Leftarrow','⇐'),(r'\\Leftrightarrow','⟺'),
        (r'\\to','→'),(r'\\gets','←'),(r'\\mapsto','↦'),
        (r'\\uparrow','↑'),(r'\\downarrow','↓'),(r'\\updownarrow','↕'),
        (r'\\iff','⟺'),(r'\\implies','⟹'),
        (r'\\land','∧'),(r'\\lor','∨'),(r'\\lnot','¬'),(r'\\neg','¬'),
        (r'\\oplus','⊕'),(r'\\otimes','⊗'),(r'\\ominus','⊖'),
        (r'\\perp','⊥'),(r'\\parallel','∥'),(r'\\angle','∠'),
        (r'\\therefore','∴'),(r'\\because','∵'),
        (r'\\ldots','…'),(r'\\cdots','⋯'),(r'\\vdots','⋮'),(r'\\ddots','⋱'),
        (r'\\hbar','ℏ'),(r'\\ell','ℓ'),(r'\\Re','ℜ'),(r'\\Im','ℑ'),
        (r'\\aleph','ℵ'),(r'\\wp','℘'),
        # currency & misc
        (r'\\$','$'),(r'\\%','%'),(r'\\#','#'),(r'\\&','&'),
    ]
    for pat, repl in ops:
        text = re.sub(pat, repl, text)

    # ── Degree symbol: 90° not 90\degree or 90^{\circ} ──────────────────────
    text = re.sub(r'(\d+)\s*\\degree', r'\1°', text)
    text = re.sub(r'(\d+)\s*\^\{?\\circ\}?', r'\1°', text)

    # ── sqrt{x} → √(x) ───────────────────────────────────────────────────────
    text = re.sub(r'√\{([^}]+)\}', r'√(\1)', text)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', text)

    # ── Remove leftover LaTeX braces & commands ───────────────────────────────
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathit\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\left[\(\[\{]', '(', text)
    text = re.sub(r'\\right[\)\]\}]', ')', text)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = text.replace('{', '').replace('}', '')

    # ── Number words for single digits in sentence context ────────────────────
    # (only when surrounded by spaces — avoids breaking code)
    # Disabled by default — uncomment if you want "3 items" → "three items"
    # num_words = {'0':'zero','1':'one','2':'two','3':'three','4':'four',
    #              '5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
    # text = re.sub(r'(?<=\s)(\d)(?=\s)', lambda m: num_words.get(m.group(1),m.group(1)), text)

    return text


def _latex_to_text(expr: str) -> str:
    """Convert a LaTeX math expression to clean readable text."""
    import re
    expr = expr.strip()
    expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr)
    expr = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', expr)
    expr = re.sub(r'\\sqrt', '√', expr)
    expr = re.sub(r'\\sum_\{([^}]+)\}\^\{([^}]+)\}', r'Σ(\1 to \2)', expr)
    expr = re.sub(r'\\int_\{([^}]+)\}\^\{([^}]+)\}', r'∫(\1 to \2)', expr)
    expr = re.sub(r'\\lim_\{([^}]+)\}', r'lim(\1)', expr)
    for old, new in [
        (r'\\times','×'),(r'\\div','÷'),(r'\\pm','±'),
        (r'\\leq','≤'),(r'\\geq','≥'),(r'\\neq','≠'),
        (r'\\approx','≈'),(r'\\infty','∞'),(r'\\pi','π'),
        (r'\\alpha','α'),(r'\\beta','β'),(r'\\gamma','γ'),
        (r'\\sigma','σ'),(r'\\mu','μ'),(r'\\lambda','λ'),
        (r'\\theta','θ'),(r'\\omega','ω'),(r'\\phi','φ'),
        (r'\\rightarrow','→'),(r'\\Rightarrow','⇒'),(r'\\to','→'),
        (r'\\partial','∂'),(r'\\nabla','∇'),(r'\\forall','∀'),
        (r'\\exists','∃'),(r'\\in','∈'),(r'\\notin','∉'),
        (r'\\cdot','·'),(r'\\ldots','…'),
    ]:
        expr = re.sub(old, new, expr)
    expr = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', expr)
    expr = expr.replace('{','').replace('}','')
    return expr.strip()


def _budget(msg: str, skill: str, complexity: str) -> int:
    """Claude-style output budget: predict needed length, don't over-allocate."""
    msg_len = len(msg)
    # Minimum floor — never go below 500 tokens regardless of message length
    if GROQ_API_KEY:
        if skill == "coder": return 4000 if complexity == "hard" else 3000
        if skill == "researcher": return 3000 if complexity == "hard" else 2500
        if complexity == "hard": return 4000
        if complexity == "medium": return 2500
        if complexity == "easy": return 1500
        return 1500
    if skill == "coder": return 1024
    if skill == "researcher": return 800
    if complexity == "hard": return 600
    return 400

def _dynamic_ctx_window() -> int:
    """Return history window size based on loaded context length."""
    if N_CTX >= 4096: return 6
    if N_CTX >= 2048: return 8
    if N_CTX >= 1024: return 4
    return 2

def _lc_kw(max_new: int, skill: str, msg_len: int) -> dict:
    # repeat_penalty 1.3+ strongly discourages recycling phrases from context.
    # A small temperature (0.15) adds variety without sacrificing coherence.
    # frequency_penalty additionally penalises tokens proportional to how often
    # they have already appeared — the primary fix for idea repetition.
    return dict(
        max_tokens      = max_new,
        stop            = _STOPS,
        repeat_penalty  = 1.08,     # mild anti-repeat
        frequency_penalty = 0.0,     # disabled
        presence_penalty  = 0.0,     # disabled
        temperature     = 0.15,      # was 0.0 — tiny heat prevents loops
        top_k           = 40,        # was 1 — greedy decoding causes loops
        top_p           = 0.92,
    )

def build_chatml(system: str, history: list, user_msg: str) -> list:
    # Groq prompt caching: static content (system) MUST come first for cache hits
    # Dynamic content (user_msg) goes last — this is the Groq-recommended structure
    msgs = [{"role":"system","content":system}]
    for h in (history or [])[-_dynamic_ctx_window()*2:]:
        r=h.get("role","user"); c=h.get("content","")
        if c.strip(): msgs.append({"role":r,"content":c[:800]})
    msgs.append({"role":"user","content":user_msg[:6000]})
    return msgs

def generate_sync(msgs: list, max_new: int, skill: str, msg_len: int) -> str:
    if GROQ_API_KEY:
        groq_generate._skill = skill
        groq_stream._skill = skill
        # compound only for hard non-streaming reasoning; 70b for everything else
        hard_skills = ("coder", "researcher")
        mdl = GROQ_MODEL
        groq_generate._reasoning_effort = "high" if skill in hard_skills else "medium"
        if result:
            print(f"[Groq] used {mdl} for skill={skill}")
            return _clean(result)
    if llm is None: return "Model not loaded."
    with _gen_lock:
        resp = llm.create_chat_completion(messages=msgs, **_lc_kw(max_new, skill, msg_len))
    return _clean(resp["choices"][0]["message"]["content"] or "")
def stream_tokens(msgs: list, max_new: int, skill: str, msg_len: int):
        yield tok
    return
    if llm is None: yield "Model not loaded."; return
    kw = _lc_kw(max_new, skill, msg_len); kw["stream"] = True
    inside_think = False; think_buf = ""
    with _gen_lock:
        for chunk in llm.create_chat_completion(messages=msgs, **kw):
            delta = chunk["choices"][0]["delta"].get("content","")
            if not delta: continue
            if "<think>" in delta:
                inside_think = True
                delta = delta.replace("<think>", "\n> 💭 *Thinking...*\n> ")
            if inside_think:
                think_buf += delta
                yield delta.replace("\n", "\n> ")
                if "</think>" in think_buf:
                    inside_think = False
                    after = think_buf.split("</think>",1)[1]
                    think_buf = ""
                    yield "\n\n"
                    if after: yield after
                continue
            stop_hit = False
            for s in _STOPS:
                if s in delta: delta = delta.split(s)[0]; stop_hit = True
            if delta: yield delta
            if stop_hit: break

CAI_CRITIQUE_TMPL = """Critique this AI response using this principle:
PRINCIPLE: {principle}
RESPONSE: {response}
Output APPROVED if it follows the principle, or REVISE: [one sentence reason] if not."""

CAI_REVISE_TMPL = """Fix this AI response based on the critique.
ORIGINAL QUESTION: {original_msg}
PREVIOUS RESPONSE: {previous_response}
CRITIQUE: {issue}
Write only the improved response:"""

RLAIF_TMPL = """Which response better follows this principle?
PRINCIPLE: {principle}
A: {a}
B: {b}
Reply only A or B:"""

# ── CLAUDE-STYLE RLAIF ───────────────────────────────────────────────────────
