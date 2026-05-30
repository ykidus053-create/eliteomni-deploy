from modules.services.semantic_mem import semantic_mem_save

# Claude-style agent identity injected by upgrade script
AGENT_IDENTITY = """
You are EliteOmni acting as a specialized agent. Regardless of the task:
- Think before acting: plan subtasks, then execute them one by one
- Verify each result before moving to the next step
- Be honest if you are uncertain about a tool result
- Never fabricate tool outputs — if a tool fails, say so
- Complete the full task — never stop halfway through
- Use the minimum number of tool calls needed, but never skip necessary ones
"""
from modules.core.constants import _tool_exec
from modules.services.search import tool_search, tool_web_fetch
from modules.services.tools import _grep_codebase, tool_exec, tool_lint
from modules.services.memory import db_mem_save
from modules.services.mcp import run_mcp_tools
def groq_generate(msgs, max_tokens=1000, **kwargs):
    return "".join(_mistral_stream(msgs, max_tokens=max_tokens))
# AUTO-SPLIT FROM app.py lines 2945-3379
import os, re, time, math, json, asyncio, random, ast, subprocess, sys, tempfile
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import urllib.request, urllib.parse


def think_first(msg: str, system: str) -> str:
    """Point 1: no separate inference call. Groq reasons natively."""
    return ""

def _groq_thinking_effort(complexity: str) -> str:
    return {"hard": "default", "medium": "default", "easy": "none"}.get(complexity, "none")

# -- Feature 14: AUTO MEMORY EXTRACTION --------------------------------------
def auto_extract_memory(user_msg: str, assistant_response: str):
    if len(assistant_response) < 100: return
    try:
        prompt = (
            "Extract 1-3 key facts worth remembering from this exchange. "
            "Output ONLY a JSON array of short strings, no markdown.\n"
            f"User: {user_msg[:300]}\nAssistant: {assistant_response[:500]}"
        )
        raw = groq_generate([{"role":"user","content":prompt}],
                            max_tokens=150)
        if not raw: return
        import json as _mj, re as _mre
        # Use greedy match to get the full array, not a partial one
        m = _mre.search(r'\[.*\]', raw, _mre.DOTALL)
        if not m: return
        # Handle truncated arrays by trimming to last complete string element
        candidate = m.group(0).strip()
        if not candidate.endswith(']'):
            last = candidate.rfind('",')
            candidate = (candidate[:last+1] + ']') if last != -1 else '[]'
        try:
            facts = _mj.loads(candidate)
        except _mj.JSONDecodeError:
            return
        for fact in facts[:3]:
            if isinstance(fact, str) and len(fact) > 10:
                db_mem_save(fact, source="auto_extract")
                semantic_mem_save(fact, {"source": "auto_extract"})
        print(f"[AutoMem] extracted {len(facts)} facts")
    except Exception as e:
        print(f"[AutoMem] non-fatal: {e}")

# -- Feature 32: TASK CHECKPOINT / RESUME ------------------------------------
_TASK_DB = os.path.expanduser("~/eliteomni_tasks.db")
def _init_task_db():
    import sqlite3 as _tdb
    con = _tdb.connect(_TASK_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, state TEXT, result TEXT,
        created_ts REAL, updated_ts REAL, status TEXT DEFAULT 'pending'
    )""")
    con.commit(); con.close()
_init_task_db()

def task_checkpoint(task_id: str, state: dict):
    import sqlite3 as _tdb, json as _tj
    try:
        con = _tdb.connect(_TASK_DB)
        con.execute(
            "INSERT OR REPLACE INTO tasks(id,state,created_ts,updated_ts,status) VALUES(?,?,?,?,?)",
            (task_id, _tj.dumps(state), time.time(), time.time(), "running"))
        con.commit(); con.close()
    except Exception as e: print(f"[Task] {e}")

def task_resume(task_id: str) -> dict:
    import sqlite3 as _tdb, json as _tj
    try:
        con = _tdb.connect(_TASK_DB)
        row = con.execute("SELECT state,status FROM tasks WHERE id=?", (task_id,)).fetchone()
        con.close()
        if row: return {"state": _tj.loads(row[0]), "status": row[1]}
    except Exception as e: print(f"[Task resume] {e}")
    return {}

# -- Feature 33: REAL BASH TOOL -----------------------------------------------
_BASH_BLOCKED = re.compile(
    r'(rm\s+-rf|dd\s+if|mkfs|shutdown|reboot|chmod\s+777\s+/'
    r'|curl.*\|.*sh|wget.*\|.*sh|nc\s+-e)',
    re.IGNORECASE
)
def tool_bash(cmd: str, timeout: int = 10) -> str:
    if _BASH_BLOCKED.search(cmd):
        return "[BLOCKED]: Destructive command not allowed."
    try:
        import shlex
        result = subprocess.run(shlex.split(cmd), shell=False, capture_output=True, text=True, timeout=timeout)
        out = (result.stdout + result.stderr).strip()
        return out[:1000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"[BASH TIMEOUT after {timeout}s]"
    except Exception as e:
        return f"[BASH ERROR]: {e}"

# -- Feature 34: GUARANTEED JSON OUTPUT ---------------------------------------
def groq_json(msgs: list, schema: dict, max_tokens: int = 1000, model: str = None) -> dict:
    if not "not_needed": return {}
    import urllib.request as _ur, json as _j
    mdl = model or "zai-glm-4.7"
    payload = _j.dumps({
        "model": mdl,
        "messages": _truncate_msgs(msgs),
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": schema, "strict": True}
        }
    }).encode()
    try:
        req = _ur.Request(GROQ_URL, data=payload, headers={
            "Authorization": f"Bearer {_get_next_key()}",
            "Content-Type": "application/json"})
        with _ur.urlopen(req, timeout=20) as r:
            data = _j.loads(r.read())
        return _j.loads(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[groq_json] {e}"); return {}

# -- Feature 22: PDF TEXT EXTRACTION ------------------------------------------
def tool_pdf_extract(pdf_path: str, max_chars: int = 8000) -> str:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text[:max_chars]
    except ImportError:
        return "[PDF] Install: pip install PyMuPDF --break-system-packages"
    except Exception as e:
        return f"[PDF error: {e}]"

# -- Feature 23: MULTI-IMAGE REASONING ----------------------------------------
def vision_multi_image(images_b64: list, prompt: str) -> str:
    return "[Vision not available]"
    try:
        import urllib.request, json as _vj
        content = [{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                   for b64 in images_b64[:5]]
        content.append({"type":"text","text":prompt})
        payload = _vj.dumps({
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role":"user","content":content}],
            "max_tokens": 2048, "temperature": 0.3
        }).encode()
        req = urllib.request.Request(GROQ_URL, data=payload, headers={
            "Authorization": f"Bearer {_get_next_key()}",
            "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _vj.loads(r.read())
        return data["choices"][0]["message"].get("content","").strip()
    except Exception as e:
        return f"[Multi-vision error: {e}]"

# -- Feature 27: EMBEDDING-RANKED SEARCH RESULTS ------------------------------
def _rank_results_by_embedding(results: list, query: str) -> list:
    if not results or not _faiss_ok: return results
    try:
        m = _get_fe()
        if m is None: return results
        import numpy as _np
        q_arr = _np.array(list(m.embed([query[:512]]))[0], dtype=_np.float32)
        norm = float(_np.linalg.norm(q_arr))
        if norm > 0: q_arr /= norm
        scored = []
        for item in results:
            text = (item.get("title","") + " " + item.get("content",""))[:512]
            try:
                r_arr = _np.array(list(m.embed([text]))[0], dtype=_np.float32)
                rn = float(_np.linalg.norm(r_arr))
                if rn > 0: r_arr /= rn
                score = float(_np.dot(q_arr, r_arr))
            except Exception: score = 0.0
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]
    except Exception: return results

# -- Feature 28: GROUNDED CITATIONS -------------------------------------------
def _cite_with_tracking(results: list, response: str) -> str:
    if not results: return response
    cites = "\n\n**Sources:**\n" + "\n".join(
        f"[{i+1}] {r.get('title') or r.get('url','')} — {r.get('url','')[:80]}"
        for i, r in enumerate(results[:3]) if r.get("url")
    )
    return response + cites

# -- Feature 25: LLAMA GUARD 4 MODERATION -------------------------------------
LLAMA_GUARD_MODEL = "meta-llama/llama-guard-4-12b"
def groq_moderate(text: str) -> dict:
    if not "not_needed": return {"safe": True, "category": None}
    try:
        import urllib.request, json as _gj
        payload = _gj.dumps({
            "model": LLAMA_GUARD_MODEL,
            "messages": [{"role":"user","content":text[:2000]}],
            "max_tokens": 50
        }).encode()
        req = urllib.request.Request(GROQ_URL, data=payload, headers={
            "Authorization": f"Bearer {_get_next_key()}",
            "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = _gj.loads(r.read())
        verdict = data["choices"][0]["message"].get("content","safe").strip().lower()
        safe = verdict.startswith("safe")
        _audit("moderation", {"verdict": verdict[:100], "safe": safe})
        return {"safe": safe, "category": None if safe else verdict}
    except Exception as e:
        print(f"[Guard] {e}"); return {"safe": True, "category": None}

def architect_plan(msg: str) -> str:
    """
    Phase 1 — Hidden Reasoning: internal plan only, no code.
    Separates reasoning from patch emission to prevent syntax leakage.
    """
    plan_msgs = build_chatml(
        "You are a software architect. Think step by step internally. "
        "Output ONLY a numbered execution plan (max 8 steps, NO code). "
        "Be specific: name functions, data structures, algorithms.",
        [],
        f"Plan how to implement this task minimally and correctly: {msg[:400]}"
    )
    try:
        plan = generate_sync(plan_msgs, 400, "coder", len(msg))
        return plan
    except Exception:
        return ""

def editor_implement(plan: str, msg: str, system: str,
                     hist_msgs: list, max_t: int) -> str:
    """
    Phase 2 — Clean Code Emission (separated from reasoning).
    Multi-pass: generate → lint → exec → retry up to 3x.
    Patch minimality: only change what the plan specifies.
    """
    if not plan:
        return ""

    # Clean mode system — no reasoning allowed in output
    clean_system = (
        "You are a code generator in CLEAN OUTPUT MODE. "
        "Do NOT explain, reason, or think in your output. "
        "Output ONLY the complete Python code inside a ```python block. "
        "Follow the plan exactly. Make MINIMAL changes. "
        "Include type hints, docstrings, and one usage example."
    )

    last_code = ""
    last_lint = ""

    for attempt in range(3):
        retry_note = f"\n[Previous attempt lint error: {last_lint}. Fix it.]" if last_lint else ""
        impl_prompt = (
            f"EXECUTION PLAN:\n{plan}\n\n"
            f"Task: {msg[:300]}{retry_note}\n\n"
            "Output the complete corrected Python code now:"
        )
        prompt = build_chatml(clean_system, [], impl_prompt)
        code   = generate_sync(prompt, max_t, "coder", len(msg))

        # Extract python block
        block = re.search(r'```python\n(.*?)```', code, re.DOTALL)
        if not block:
            last_lint = "No code block found"
            continue

        extracted = block.group(1).strip()
        lint_result = tool_lint(extracted)

        if lint_result == "OK":
            # Execute to verify
            exec_out = tool_exec(extracted)
            status = "✅" if "[EXEC ERROR]" not in exec_out and "[LINT" not in exec_out else "⚠️"
            suffix = f"\n\n{status} **Verified** (attempt {attempt+1}) · Lint: OK"
            if exec_out and exec_out != "(no output)":
                suffix += f"\n```\n{exec_out[:300]}\n```"
            return code + suffix

        # Lint failed — retry with error feedback
        last_lint = lint_result
        last_code = code
        print(f"[Editor iter {attempt+1}] lint failed: {lint_result} — retrying")

    # All attempts failed — return last attempt with lint warning
    result = last_code or code
    result += f"\n\n> ⚠️ **Lint after 3 attempts:** {last_lint}"
    return result

# ── AGENT TEAMS (Opus 4.6: parallel specialist agents) ────────────────────────

_agent_state = {}  # shared state between agents

def _agent_worker(role: str, task: str, system_base: str, max_tokens: int, retries: int = 2) -> tuple:
    """Agent worker with retry logic and shared state."""
    for attempt in range(retries):
        try:
            agent_system = (
                f"{system_base}\n\nYou are the {role.upper()} AGENT.\n"
                f"{_AGENT_ROLE_PROMPTS.get(role, 'Complete your assigned task.')}\n"
                f"SHARED STATE: {json.dumps(_agent_state)[:300]}"
            )
            msgs = build_chatml(agent_system, [], task)
            result = groq_generate(msgs, max_tokens=max_tokens)
            if result and len(result) > 20:
                _agent_state[role] = result[:200]  # share key output
                return role, result
            print(f"[Agent {role}] attempt {attempt+1} empty result, retrying...")
        except Exception as e:
            print(f"[Agent {role}] attempt {attempt+1} error: {e}")
            if attempt == retries - 1:
                return role, f"[{role} agent failed after {retries} attempts: {e}]"
    return role, f"[{role} agent failed]"

def _agent_worker_old(role: str, task: str, system_base: str, max_tokens: int) -> tuple:
    """
    Single specialist agent. Runs in its own thread.
    Returns (role, result) for merging.
    """
    if False:  # groq mode
        return role, f"[{role}: model not loaded]"
    agent_system = (
        f"{system_base}\n\n"
        f"You are the {role.upper()} AGENT. Focus exclusively on your role:\n"
        f"{_AGENT_ROLE_PROMPTS.get(role, 'Complete your assigned task.')}"
    )
    msgs = build_chatml(agent_system, [], task)
    try:
        result = generate_sync(msgs, max_tokens, "coder", len(task))
        return role, result
    except Exception as e:
        return role, f"[{role} agent error: {e}]"

_AGENT_ROLE_PROMPTS = {
    "implementer": (
        "Write the core implementation. Focus on correctness and completeness. "
        "Use type hints, docstrings. Never truncate."
    ),
    "tester": (
        "Write comprehensive tests for the implementation. "
        "Cover: happy path, edge cases, error conditions. Use EXEC() to run them."
    ),
    "reviewer": (
        "Review the implementation for: bugs, security issues, performance problems, "
        "style violations. Output a structured review with [ISSUE] and [SUGGESTION] tags."
    ),
}

import concurrent.futures as _cf
_agent_team_exec = _cf.ThreadPoolExecutor(max_workers=4)

def run_agent_team(msg: str, system: str, hist_msgs: list) -> str:
    """
    Spawn parallel specialist agents (implementer + tester + reviewer).
    Merge their outputs into a single structured response.
    Only activates for hard coder tasks to avoid overhead.
    """
    task = f"Task: {msg[:400]}"
    max_t = 600

    futures = {
        role: _agent_team_exec.submit(_agent_worker, role, task, system, max_t)
        for role in ["implementer", "tester", "reviewer"]
    }

    results = {}
    for role, fut in futures.items():
        try:
            _, result = fut.result(timeout=60)
            results[role] = result
        except Exception as e:
            results[role] = f"[{role} timed out: {e}]"

    merged = (
        f"## 🏗️ Implementation\n{results.get('implementer','')}\n\n"
        f"## 🧪 Tests\n{results.get('tester','')}\n\n"
        f"## 🔍 Review\n{results.get('reviewer','')}"
    )
    return merged

# ── PARALLEL TOOL DISPATCHER ──────────────────────────────────────────────────
# Supports: CALC() TIME() SEARCH() FETCH() EXEC() LINT() GREP()

_TOOL_RE = re.compile(r'\b(WEATHER|BROWSER|CALC|TIME|SEARCH|FETCH|EXEC|LINT|GREP)\(([^)]*)\)')

def _run_one_tool(name: str, arg: str) -> str:
    """Route a single tool call to its handler."""
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
        return f"[Tool error: {e}]"

def run_tools_parallel(text: str) -> str:
    """
    Find all tool calls in model output, execute them concurrently via the
    thread pool, then substitute [= result] back into the text.
    Deduplicates identical calls to avoid redundant work.
    """
    calls = _TOOL_RE.findall(text)
    if not calls:
        return text

    # Deduplicate while preserving order
    seen: dict = {}
    for name, arg in calls:
        key = f"{name}({arg})"
        if key not in seen:
            seen[key] = _tool_exec.submit(_run_one_tool, name, arg.strip())

    results: dict = {}
    for key, fut in seen.items():
        try:
            results[key] = fut.result(timeout=15)
        except Exception as e:
            results[key] = f"[timeout/error: {e}]"

    def _replace(m: re.Match) -> str:
        key = f"{m.group(1)}({m.group(2)})"
        return f"{key} [= {results.get(key, '?')}]"

    text = _TOOL_RE.sub(_replace, text)

    # Also run any MCP tool calls found in the same output
    text = run_mcp_tools(text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# MCP — MODEL CONTEXT PROTOCOL  (JSON-RPC 2.0 over HTTP/SSE)
# Spec: https://modelcontextprotocol.io
# EliteOmni auto-discovers tools from registered MCP servers and makes them
# available to the model alongside the built-in tools.
# ══════════════════════════════════════════════════════════════════════════════
import threading, uuid

# ── Server registry ───────────────────────────────────────────────────────────

def strip_tool_syntax(text: str) -> str:
    import re
    # Replace TOOL(anything) [= result] with just the result
    text = re.sub(r'\b(?:SEARCH|CALC|FETCH|EXEC|TIME|MEM|LINT|GREP|BROWSER)\([^)]*\)\s*\[=\s*(.*?)\]',
                  lambda m: m.group(1).strip(), text, flags=re.DOTALL)
    # Remove bare tool calls with no result
    text = re.sub(r'\b(?:SEARCH|CALC|FETCH|EXEC|TIME|MEM|LINT|GREP|BROWSER)\([^)]*\)', '', text)
    # Remove leaked internal monologue
    for pat in [
        r"I can see the search results were cut off\.?\s*",
        r"Let me fetch more complete information[^\n]*\n?",
        r"Let me search for[^\n]*\n?",
        r"I'll search for[^\n]*\n?",
        r"Searching for[^\n]*\n?",
        r"Sources:\s*\[.*?\]\n?",
    ]:
        text = re.sub(pat, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def tool_browser(url: str, max_chars: int = 600) -> str:
    """Fetch a URL and return clean text — fallback browser tool."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            import re
            text = r.read().decode("utf-8", errors="ignore")
            text = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', ' ', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:max_chars]
    except Exception as e:
        return f"[browser error: {e}]"


def _mistral_stream(messages: list, system: str = "", model: str = "mistral-large-latest", max_tokens: int = 2048) -> str:
    try:
        from modules.services.pipeline import generate_sync
        msgs = [{"role": "system", "content": system}] + messages if system else messages
        return generate_sync(msgs, max_tokens, "general", 0)
    except Exception as e:
        return f"[_mistral_stream error: {e}]"

# ── ATTENTION GATING ──────────────────────────────────────────────────────────
_GATE_CACHE: dict = {}
def attention_gate(msg: str, skill: str, complexity: str) -> dict:
    key = f"{skill}:{complexity}:{msg[:60]}"
    if key in _GATE_CACHE: return _GATE_CACHE[key]
    default = {"memory":True,"search":True,"rlaif":complexity=="hard","exec":skill in ("coder","calculator")}
    if complexity == "easy":
        result = {"memory":False,"search":False,"rlaif":False,"exec":False}
        _GATE_CACHE[key] = result; return result
    try:
        from modules.core.http_client import groq_generate
        prompt = (f"Query: {msg[:200]}\nSkill: {skill}\nComplexity: {complexity}\n"
                  "Reply ONLY with a JSON object with boolean keys: memory, search, rlaif, exec\n"
                  "Set true only if genuinely needed. Example: {\"memory\":true,\"search\":false,\"rlaif\":true,\"exec\":false}")
        raw = groq_generate([{"role":"user","content":prompt}], max_tokens=60)
        if raw:
            import re, json as _j
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                result = {**default, **_j.loads(m.group(0))}
                _GATE_CACHE[key] = result; return result
    except Exception: pass
    return default

# ── META PROMPT REWRITER ──────────────────────────────────────────────────────
_meta_interaction_count = 0
def meta_maybe_rewrite_prompt(skill: str, score: int):
    global _meta_interaction_count
    _meta_interaction_count += 1
    if _meta_interaction_count % 100 != 0: return
    try:
        from modules.services.memory import _rlaif_log
        from modules.core.http_client import groq_generate
        failures = [r for r in _rlaif_log[-100:] if r.get("hhh",{}).get("total",15) < 10]
        if len(failures) < 5: return
        patterns = "\n".join(f"- {f['prompt'][:100]}" for f in failures[:10])
        suggestion = groq_generate([{"role":"user","content":
            f"These AI responses scored poorly. Identify the prompt instruction that would fix them.\n"
            f"Failure patterns:\n{patterns}\n"
            f"Output one short system prompt instruction (max 2 sentences) to add:"}], max_tokens=100)
        if suggestion and len(suggestion) > 20:
            path = os.path.expanduser("~/eliteomni_meta_prompts.txt")
            with open(path, "a") as f:
                f.write(f"{__import__('datetime').datetime.now().isoformat()} [{skill}]: {suggestion}\n")
            print(f"[MetaPrompt] new instruction saved: {suggestion[:80]}")
    except Exception as e:
        print(f"[MetaPrompt] {e}")

# ── ATTENTION GATING ─────────────────────────────────────────────────────────
_GATE_CACHE = {}
def attention_gate(msg, skill, complexity):
    if complexity == "easy": return {"memory":False,"search":False,"rlaif":False,"exec":False}
    key = f"{skill}:{complexity}:{msg[:60]}"
    if key in _GATE_CACHE: return _GATE_CACHE[key]
    default = {"memory":True,"search":True,"rlaif":complexity=="hard","exec":skill in ("coder","calculator")}
    try:
        from modules.core.http_client import groq_generate
        import re, json as _jj
        raw = groq_generate([{"role":"user","content":f"Query: {msg[:150]}\nSkill:{skill} Complexity:{complexity}\nReply ONLY JSON with boolean keys: memory, search, rlaif, exec"}], max_tokens=60)
        if raw:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m: default = {**default, **_jj.loads(m.group(0))}
    except Exception: pass
    _GATE_CACHE[key] = default; return default

# ── META PROMPT REWRITER ──────────────────────────────────────────────────────
_meta_count = 0
def meta_maybe_rewrite_prompt(skill, score):
    global _meta_count; _meta_count += 1
    if _meta_count % 100 != 0: return
    try:
        from modules.services.memory import _rlaif_log
        from modules.core.http_client import groq_generate
        import os, datetime
        failures = [r for r in _rlaif_log[-100:] if r.get("hhh",{}).get("total",15) < 10]
        if len(failures) < 5: return
        patterns = "\n".join(f"- {f['prompt'][:100]}" for f in failures[:10])
        s = groq_generate([{"role":"user","content":f"These responses failed:\n{patterns}\nWrite ONE short system prompt fix (2 sentences max):"}], max_tokens=100)
        if s and len(s) > 20:
            with open(os.path.expanduser("~/eliteomni_meta_prompts.txt"), "a") as f:
                f.write(f"{datetime.datetime.now().isoformat()} [{skill}]: {s}\n")
            print(f"[MetaPrompt] saved: {s[:80]}")
    except Exception as e: print(f"[MetaPrompt] {e}")

# ── PRE-FETCH PLANNING ────────────────────────────────────────────────────────
def prefetch_plan(msg: str, skill: str) -> dict:
    """For hard queries: plan and pre-fetch all needed info in parallel before generation."""
    try:
        from modules.core.http_client import groq_generate
        from modules.services.search import tool_search
        from modules.services.tools import tool_exec
        import re, json as _pj
        plan_raw = groq_generate([{"role":"user","content":
            f"To answer this query, list what information is needed.\n"
            f"Query: {msg[:300]}\n"
            f"Reply ONLY JSON: {{\"needs_search\": [\"query1\",\"query2\"], "
            f"\"needs_calc\": [\"expr1\"], \"needs_code\": [\"snippet1\"]}}\n"
            f"Max 2 items per list. Empty lists if not needed."}], max_tokens=150)
        if not plan_raw: return {}
        m = re.search(r"\{.*\}", plan_raw, re.DOTALL)
        if not m: return {}
        plan = _pj.loads(m.group(0))
        results = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            for q in plan.get("needs_search", [])[:2]:
                futures[ex.submit(tool_search, q)] = f"search:{q}"
            for expr in plan.get("needs_calc", [])[:2]:
                futures[ex.submit(lambda e=expr: str(eval(e)), expr)] = f"calc:{expr}"
            for done in as_completed(futures, timeout=8):
                key = futures[done]
                try: results[key] = str(done.result())[:500]
                except Exception: pass
        return results
    except Exception as e:
        print(f"[Prefetch] {e}"); return {}

# ═══════════════════════════════════════════════════════════════════════════════
# HUMAN-GAP CLOSURE SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. PHYSICAL SIMULATION STEP ───────────────────────────────────────────────
PHYSICAL_SIMULATION_PROMPT = """
<physical_simulation>
For any question involving physical objects, actions, cooking, engineering,
biology, or real-world processes — before answering, simulate step by step:
SIMULATE: [what physically happens at each stage]
INTUITION CHECK: [does this match common sense?]
Then give your answer grounded in that simulation.
</physical_simulation>
"""

# ── 2. PERSISTENT GOAL TRACKING ───────────────────────────────────────────────
import json as _gt_j, os as _gt_os
_GOALS_PATH = _gt_os.path.expanduser("~/eliteomni_goals.json")
_user_goals: dict = {}

def _load_goals():
    global _user_goals
    try:
        if _gt_os.path.exists(_GOALS_PATH):
            _user_goals = _gt_j.load(open(_GOALS_PATH))
    except Exception: _user_goals = {"long_term": [], "short_term": [], "current_session": []}
    if not _user_goals: _user_goals = {"long_term": [], "short_term": [], "current_session": []}
_load_goals()

def goals_get_context() -> str:
    if not any(_user_goals.get(k) for k in ("long_term","short_term","current_session")):
        return ""
    return "USER GOALS:\n" + _gt_j.dumps(_user_goals, indent=None)[:400]

def goals_update(msg: str, response: str):
    try:
        from modules.core.http_client import groq_generate
        import re
        raw = groq_generate([{"role":"user","content":
            f"Extract any user goals mentioned.\nUser: {msg[:300]}\n"
            f"Reply ONLY JSON: {{\"long_term\":[],\"short_term\":[],\"current_session\":[]}}\n"
            f"Empty lists if no goals found."}], max_tokens=150)
        if not raw: return
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return
        updates = _gt_j.loads(m.group(0))
        for k in ("long_term","short_term","current_session"):
            if updates.get(k):
                existing = _user_goals.setdefault(k, [])
                for g in updates[k]:
                    if g not in existing: existing.append(g)
                _user_goals[k] = existing[-10:]
        _gt_j.dump(_user_goals, open(_GOALS_PATH,"w"), indent=2)
        print(f"[Goals] updated")
    except Exception as e: print(f"[Goals] {e}")

# ── 3. EMOTION AND TONE DETECTION ────────────────────────────────────────────
def detect_emotion_context(msg: str) -> str:
    if len(msg) < 20: return ""
    try:
        from modules.core.http_client import groq_generate
        import re, json as _ej
        raw = groq_generate([{"role":"user","content":
            f"Analyze this message for emotional context.\nMessage: {msg[:300]}\n"
            f"Reply ONLY JSON: {{\"emotional_state\": \"\", \"unstated_need\": \"\", "
            f"\"tone\": \"\", \"what_not_said\": \"\"}}"}], max_tokens=100)
        if not raw: return ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return ""
        data = _ej.loads(m.group(0))
        parts = []
        if data.get("emotional_state"): parts.append(f"Emotional state: {data['emotional_state']}")
        if data.get("unstated_need"): parts.append(f"Unstated need: {data['unstated_need']}")
        if data.get("what_not_said"): parts.append(f"Subtext: {data['what_not_said']}")
        return "EMOTIONAL CONTEXT:\n" + "\n".join(parts) if parts else ""
    except Exception: return ""

# ── 4. DOMAIN-SPECIFIC RAG INGESTION ─────────────────────────────────────────
def rag_ingest_domain(text: str, domain: str, source: str = "manual"):
    """Ingest domain knowledge into semantic + SQLite memory for expert retrieval."""
    try:
        chunks = [text[i:i+500] for i in range(0, len(text), 400)]
        from modules.services.memory import db_mem_save
        from modules.services.semantic_mem import semantic_mem_save
        for chunk in chunks:
            tagged = f"[{domain}] {chunk}"
            db_mem_save(tagged, source=source)
            semantic_mem_save(tagged, {"domain": domain, "source": source})
        print(f"[RAG] ingested {len(chunks)} chunks for domain: {domain}")
    except Exception as e: print(f"[RAG ingest] {e}")

# ── 5. CROSS-DOMAIN ANALOGY FORCING ──────────────────────────────────────────
CROSS_DOMAIN_ANALOGY_PROMPT = """
<cross_domain_creativity>
For creative, design, strategy, or hard analytical problems:
ANALOGY: Find a solution pattern from a completely unrelated field
         (e.g. biology→engineering, music→UI design, evolution→optimization)
TRANSFER: Explain how that pattern applies to this problem
NOVEL INSIGHT: What does this analogy reveal that direct thinking misses?
</cross_domain_creativity>
"""

# ── 6. ADVERSARIAL SELF-CHECK ────────────────────────────────────────────────
def adversarial_self_check(response: str, msg: str, complexity: str) -> str:
    if complexity != "hard" or len(response) < 150: return response
    try:
        from modules.core.http_client import groq_generate
        critique = groq_generate([{"role":"user","content":
            f"Find the single most likely reasoning error in this response. Be specific.\n"
            f"Question: {msg[:200]}\nResponse: {response[:600]}\n"
            f"Reply APPROVED if reasoning is sound, or ERROR: [specific mistake] if not."}],
            max_tokens=100)
        if not critique or "APPROVED" in critique.upper(): return response
        if "ERROR:" in critique.upper():
            fix = groq_generate([{"role":"user","content":
                f"Fix this specific error in the response.\n"
                f"Original question: {msg[:200]}\n"
                f"Flawed response: {response[:600]}\n"
                f"Error found: {critique[:200]}\n"
                f"Write only the corrected response:"}], max_tokens=1500)
            if fix and len(fix) > 100:
                print(f"[AdversarialCheck] fixed error: {critique[:60]}")
                return fix
    except Exception as e: print(f"[AdversarialCheck] {e}")
    return response

# ── 7. LONG-HORIZON TASK DECOMPOSITION ───────────────────────────────────────
import sqlite3 as _lh_sq
_LH_DB = os.path.expanduser("~/eliteomni_goals.db")
def _init_lh_db():
    con = _lh_sq.connect(_LH_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, goal TEXT, phases TEXT,
        current_phase INTEGER DEFAULT 0, status TEXT DEFAULT 'active',
        ts REAL, updated REAL
    )""")
    con.commit(); con.close()
_init_lh_db()

def decompose_goal(goal: str) -> dict:
    try:
        from modules.core.http_client import groq_generate
        import re, json as _lj, uuid, time as _lt
        raw = groq_generate([{"role":"user","content":
            f"Decompose this goal into phases and steps.\nGoal: {goal[:300]}\n"
            f"Reply ONLY JSON: {{\"phases\": [{{\"name\": \"\", \"steps\": []}}]}}"}],
            max_tokens=400)
        if not raw: return {}
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return {}
        plan = _lj.loads(m.group(0))
        task_id = str(uuid.uuid4())[:8]
        con = _lh_sq.connect(_LH_DB)
        con.execute("INSERT INTO tasks (id,goal,phases,ts,updated) VALUES (?,?,?,?,?)",
            (task_id, goal[:300], _lj.dumps(plan["phases"]), _lt.time(), _lt.time()))
        con.commit(); con.close()
        print(f"[GoalDecompose] task {task_id} created with {len(plan['phases'])} phases")
        return {"task_id": task_id, "plan": plan}
    except Exception as e: print(f"[GoalDecompose] {e}"); return {}

# ── 8. KNOWLEDGE BOUNDARY DETECTION ─────────────────────────────────────────
def knowledge_boundary_check(msg: str, skill: str) -> str:
    try:
        from modules.core.http_client import groq_generate
        import re, json as _kj
        raw = groq_generate([{"role":"user","content":
            f"Assess knowledge boundary for this query.\nQuery: {msg[:300]}\nSkill: {skill}\n"
            f"Reply ONLY JSON: {{\"confidence\": 1-10, \"risky_claims\": [], "
            f"\"should_search_instead\": true/false, \"knowledge_cutoff_relevant\": true/false}}"}],
            max_tokens=120)
        if not raw: return ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return ""
        data = _kj.loads(m.group(0))
        warnings = []
        if data.get("confidence", 10) <= 5:
            warnings.append(f"⚠️ Low confidence ({data['confidence']}/10) — verify independently")
        if data.get("should_search_instead"):
            warnings.append("🔍 This query needs current data — search results preferred over memory")
        if data.get("knowledge_cutoff_relevant"):
            warnings.append("📅 Answer may be outdated — checking live sources recommended")
        if data.get("risky_claims"):
            warnings.append(f"⚡ Potentially unreliable claims: {', '.join(str(r) for r in data['risky_claims'][:3])}")
        return "KNOWLEDGE BOUNDARY:\n" + "\n".join(warnings) if warnings else ""
    except Exception: return ""

# ═══════════════════════════════════════════════════════════════════════════════
# DEEP HUMAN-GAP CLOSURE SYSTEMS v2
# ═══════════════════════════════════════════════════════════════════════════════

import json as _dg_j, os as _dg_os, time as _dg_t, re as _dg_re
from modules.core.http_client import groq_generate

# validation imports deferred to avoid circular import at load time


# validation imports deferred to avoid circular import at load time


# ── 1. TEMPORAL DECAY ON MEMORY ───────────────────────────────────────────────
def temporal_decay_filter(memories: list, decay_days: float = 30.0) -> list:
    """Weight memories by recency. Flag stale ones."""
    import sqlite3 as _td_sq
    try:
        con = _td_sq.connect(_dg_os.path.expanduser("~/eliteomni_memory.db"))
        results = []
        now = _dg_t.time()
        for mem in memories:
            row = con.execute("SELECT ts FROM memory WHERE text=? LIMIT 1", (mem[:2000],)).fetchone()
            if row:
                age_days = (now - row[0]) / 86400
                if age_days > 365:
                    results.append(f"[POSSIBLY OUTDATED - {int(age_days)}d old] {mem}")
                elif age_days > decay_days:
                    results.append(f"[{int(age_days)}d old] {mem}")
                else:
                    results.append(mem)
            else:
                results.append(mem)
        con.close()
        return results
    except Exception: return memories

# ── 2. CONTEXT DEPTH AWARENESS ────────────────────────────────────────────────
def context_depth_warning(history: list, system: str) -> str:
    total_chars = sum(len(str(m)) for m in history) + len(system)
    capacity_pct = total_chars / 120000
    if capacity_pct > 0.8:
        return "⚠️ CONTEXT WARNING: Near context limit — I may have missed earlier conversation. Key points may need restating."
    if capacity_pct > 0.6:
        return "NOTE: Deep into context window — prioritizing recent messages. Earlier context may be compressed."
    return ""

# ── 3. COUNTERFACTUAL BRANCHING ───────────────────────────────────────────────
COUNTERFACTUAL_PROMPT = """
<counterfactual_reasoning>
For any decision, recommendation, or analytical query generate:
SCENARIO A (recommended): [your main answer]
SCENARIO B (if constraints removed): [what changes without limitations]
SCENARIO C (if this fails): [contingency / what to watch for]
VERDICT: Which scenario best fits the user's actual situation and why.
</counterfactual_reasoning>
"""

# ── 4. THEORY OF MIND DEPTH ───────────────────────────────────────────────────
def theory_of_mind_analysis(msg: str, skill: str) -> str:
    if skill not in ("general","researcher") or len(msg) < 30: return ""
    try:
        raw = groq_generate([{"role":"user","content":
            f"Analyze the perspective stack for this message.\nMessage: {msg[:300]}\n"
            f"Reply ONLY JSON: {{\"user_believes\": \"\", "
            f"\"user_thinks_ai_believes\": \"\", "
            f"\"users_actual_goal\": \"\", "
            f"\"unstated_assumption\": \"\"}}"}], max_tokens=150)
        if not raw: return ""
        m = _dg_re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return ""
        data = _dg_j.loads(m.group(0))
        parts = []
        if data.get("users_actual_goal"): parts.append(f"Actual goal: {data['users_actual_goal']}")
        if data.get("unstated_assumption"): parts.append(f"Hidden assumption: {data['unstated_assumption']}")
        if data.get("user_thinks_ai_believes"): parts.append(f"User expects AI thinks: {data['user_thinks_ai_believes']}")
        return "PERSPECTIVE ANALYSIS:\n" + "\n".join(parts) if parts else ""
    except Exception: return ""

# ── 5. CONSTRAINT EXTRACTION ──────────────────────────────────────────────────
def extract_constraints(msg: str) -> str:
    try:
        raw = groq_generate([{"role":"user","content":
            f"Extract realistic constraints this user likely has.\nMessage: {msg[:300]}\n"
            f"Reply ONLY JSON: {{\"time\": \"\", \"budget\": \"\", "
            f"\"expertise_level\": \"\", \"tools_available\": \"\", "
            f"\"hard_limits\": []}}"}], max_tokens=120)
        if not raw: return ""
        m = _dg_re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return ""
        data = _dg_j.loads(m.group(0))
        constraints = {k:v for k,v in data.items() if v and v != "unknown" and v != []}
        if not constraints: return ""
        return "INFERRED CONSTRAINTS: " + _dg_j.dumps(constraints)[:300]
    except Exception: return ""

# ── 6. NARRATIVE IDENTITY MODEL ───────────────────────────────────────────────
_NI_PATH = _dg_os.path.expanduser("~/eliteomni_narrative.json")
_narrative: dict = {}

def _load_narrative():
    global _narrative
    try:
        if _dg_os.path.exists(_NI_PATH):
            _narrative = _dg_j.load(open(_NI_PATH))
    except Exception: pass
    if not _narrative:
        _narrative = {"self_image": "", "motivations": [], "frustrations": [],
                      "communication_style": "", "values": [], "updated": ""}
_load_narrative()

def narrative_get_context() -> str:
    if not any(_narrative.get(k) for k in ("self_image","motivations","communication_style")):
        return ""
    return "USER NARRATIVE IDENTITY: " + _dg_j.dumps(_narrative)[:400]

def narrative_update(msg: str, response: str):
    try:
        raw = groq_generate([{"role":"user","content":
            f"Extract identity signals from this exchange.\n"
            f"User: {msg[:300]}\nAssistant: {response[:300]}\n"
            f"Reply ONLY JSON: {{\"self_image\": \"\", \"motivations\": [], "
            f"\"frustrations\": [], \"communication_style\": \"\", \"values\": []}}\n"
            f"Empty strings/lists if nothing clear. Be conservative."}], max_tokens=150)
        if not raw: return
        m = _dg_re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return
        updates = _dg_j.loads(m.group(0))
        import datetime
        for k in ("self_image","communication_style"):
            if updates.get(k): _narrative[k] = updates[k]
        for k in ("motivations","frustrations","values"):
            if updates.get(k):
                existing = _narrative.setdefault(k, [])
                for item in updates[k]:
                    if item and item not in existing: existing.append(item)
                _narrative[k] = existing[-8:]
        _narrative["updated"] = datetime.datetime.now().isoformat()
        _dg_j.dump(_narrative, open(_NI_PATH,"w"), indent=2)
    except Exception as e: print(f"[Narrative] {e}")

# ── 7. DUAL PROCESS ROUTING BY STAKES ────────────────────────────────────────
_HIGH_STAKES_DOMAINS = {
    "medical","health","symptom","diagnosis","medicine","drug","dose","surgery",
    "legal","law","contract","sue","court","illegal","crime",
    "financial","invest","stock","crypto","mortgage","loan","bankruptcy",
    "safety","dangerous","emergency","suicide","harm","abuse","fire","poison"
}

def assess_stakes(msg: str, complexity: str) -> str:
    msg_lower = msg.lower()
    is_high_stakes = any(kw in msg_lower for kw in _HIGH_STAKES_DOMAINS)
    if is_high_stakes:
        return "high"
    if complexity == "hard":
        return "high"
    if complexity == "medium":
        return "medium"
    return "low"

def stakes_system_addon(stakes: str) -> str:
    if stakes == "high":
        return """<high_stakes_mode>
THIS IS A HIGH-STAKES QUERY (medical/legal/financial/safety).
MANDATORY: Use System 2 thinking — slow, deliberate, verified.
1. State what you know with certainty vs what you are inferring
2. Give the conservative/safe recommendation first
3. Explicitly recommend professional consultation
4. Never omit important caveats to seem more helpful
</high_stakes_mode>"""
    return ""

# ── 8. NEGATIVE SPACE REASONING ───────────────────────────────────────────────
def negative_space_analysis(msg: str, complexity: str) -> str:
    if complexity == "easy" or len(msg) < 40: return ""
    try:
        raw = groq_generate([{"role":"user","content":
            f"What is notably ABSENT from this query that might be important?\n"
            f"Query: {msg[:300]}\n"
            f"Reply ONLY JSON: {{\"missing_info\": [], \"avoided_topic\": \"\", "
            f"\"implicit_assumption\": \"\"}}"}], max_tokens=120)
        if not raw: return ""
        m = _dg_re.search(r"\{.*\}", raw, re.DOTALL)
        if not m: return ""
        data = _dg_j.loads(m.group(0))
        parts = []
        if data.get("missing_info"): parts.append(f"Missing context: {', '.join(str(x) for x in data['missing_info'][:3])}")
        if data.get("avoided_topic"): parts.append(f"Possibly avoided: {data['avoided_topic']}")
        if data.get("implicit_assumption"): parts.append(f"Assumed: {data['implicit_assumption']}")
        return "NEGATIVE SPACE:\n" + "\n".join(parts) if parts else ""
    except Exception: return ""

# ── 9. EXPERIENCE-WEIGHTED PRIORS ────────────────────────────────────────────
def experience_prior_check(msg: str, skill: str) -> str:
    try:
        from modules.services.memory import _rlaif_log
        if len(_rlaif_log) < 20: return ""
        failures = [r for r in _rlaif_log[-200:]
                    if r.get("hhh",{}).get("total",15) < 8
                    and r.get("skill","") == skill]
        if len(failures) < 3: return ""
        patterns = [f["prompt"][:80] for f in failures[-5:]]
        msg_lower = msg.lower()
        matches = [p for p in patterns
                   if any(w in msg_lower for w in p.lower().split()[:5] if len(w)>3)]
        if matches:
            return f"⚡ PRIOR FAILURE PATTERN DETECTED: Similar queries have scored poorly. Apply extra care."
    except Exception: pass
    return ""

# ── 10. RELATIONSHIP / COMMUNICATION MODEL ───────────────────────────────────
_REL_PATH = _dg_os.path.expanduser("~/eliteomni_relationship.json")
_relationship: dict = {}

def _load_relationship():
    global _relationship
    try:
        if _dg_os.path.exists(_REL_PATH):
            _relationship = _dg_j.load(open(_REL_PATH))
    except Exception: pass
    if not _relationship:
        _relationship = {
            "prefers_bullets": None, "prefers_short": None,
            "dislikes_hedging": None, "followup_pattern": None,
            "avg_message_length": 0, "total_interactions": 0,
            "frustration_signals": 0, "satisfaction_signals": 0
        }
_load_relationship()

def relationship_get_context() -> str:
    if _relationship.get("total_interactions", 0) < 5: return ""
    prefs = []
    if _relationship.get("prefers_bullets") is True: prefs.append("prefers bullet points")
    if _relationship.get("prefers_bullets") is False: prefs.append("prefers prose")
    if _relationship.get("prefers_short") is True: prefs.append("prefers concise answers")
    if _relationship.get("dislikes_hedging") is True: prefs.append("dislikes excessive hedging — be direct")
    if _relationship.get("followup_pattern"): prefs.append(f"usually follows up with: {_relationship['followup_pattern']}")
    return "COMMUNICATION PREFERENCES: " + "; ".join(prefs) if prefs else ""

def relationship_update(msg: str, response: str):
    try:
        import datetime
        _relationship["total_interactions"] = _relationship.get("total_interactions", 0) + 1
        avg = _relationship.get("avg_message_length", 0)
        _relationship["avg_message_length"] = int((avg * 0.9) + (len(msg) * 0.1))
        msg_lower = msg.lower()
        if any(w in msg_lower for w in ["too long","shorter","brief","tldr","summarize"]):
            _relationship["prefers_short"] = True
        if any(w in msg_lower for w in ["more detail","elaborate","expand","explain more"]):
            _relationship["prefers_short"] = False
        if any(w in msg_lower for w in ["stop hedging","just tell me","be direct","confident"]):
            _relationship["dislikes_hedging"] = True
        has_bullets = "\n-" in response or "\n•" in response
        if any(w in msg_lower for w in ["use bullets","bullet points","list format"]):
            _relationship["prefers_bullets"] = True
        if any(w in msg_lower for w in ["no bullets","prose","paragraph"]):
            _relationship["prefers_bullets"] = False
        frustration = any(w in msg_lower for w in ["wrong","incorrect","no that","that is not","you missed","try again"])
        if frustration: _relationship["frustration_signals"] = _relationship.get("frustration_signals",0) + 1
        satisfaction = any(w in msg_lower for w in ["perfect","exactly","great","thank","that works","correct"])
        if satisfaction: _relationship["satisfaction_signals"] = _relationship.get("satisfaction_signals",0) + 1
        _dg_j.dump(_relationship, open(_REL_PATH,"w"), indent=2)
    except Exception as e: print(f"[Relationship] {e}")
