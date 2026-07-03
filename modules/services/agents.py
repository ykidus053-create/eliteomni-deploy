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
from modules.services.memory import db_mem_save, tool_calc, tool_weather, tool_time
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

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer"},
                    "state_before": {"type": "string", "description": "Exact runtime state before this step executes"},
                    "action": {"type": "string", "description": "What executes in this step — name real data structures, types, and concurrency primitives"},
                    "state_after": {"type": "string", "description": "Exact runtime state after this step"},
                    "failure_handling": {"type": "string", "description": "What can fail and how it is handled"},
                    "external_system": {"type": ["string", "null"], "description": "Name of any real external system this step depends on, or null"}
                },
                "required": ["step_number", "state_before", "action", "state_after", "failure_handling", "external_system"]
            }
        }
    },
    "required": ["steps"]
}

def architect_plan(msg: str) -> str:
    """
    Phase 1 — Hidden Reasoning: internal plan only, no code.
    Returns a structured JSON plan (via groq_json + json_schema) rendered
    back to a numbered-step string for editor_implement.
    Separates reasoning from patch emission to prevent syntax leakage.
    """
    plan_msgs = build_chatml(
        "You are a systems engineer, not an interviewer. "
        "Produce an execution plan as structured JSON (max 8 steps, NO code in any field). "
        "Name real data structures with types. Name real concurrency primitives. "
        "Never write a step that contains TODO, stub, or placeholder. "
        "If a step requires a real external system, name it explicitly; otherwise use null.",
        [],
        f"Plan how to implement this task as a real running system: {msg[:400]}"
    )
    try:
        result = groq_json(plan_msgs, _PLAN_SCHEMA, max_tokens=600)
        steps = result.get("steps", [])
        if not steps:
            return ""
        lines = []
        for s in steps:
            lines.append(
                f"{s.get('step_number', '?')}. {s.get('action', '')}\n"
                f"   - before: {s.get('state_before', '')}\n"
                f"   - after: {s.get('state_after', '')}\n"
                f"   - failure handling: {s.get('failure_handling', '')}"
                + (f"\n   - external system: {s['external_system']}" if s.get('external_system') else "")
            )
        return "\n".join(lines)
    except Exception as _e:
        print(f"[architect_plan] {_e}")
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
        "You are a production code generator. "
        "Output ONLY complete Python code. "
        "EVERY function FULLY implemented - no pass, TODO, stubs. "
        "No demos, no simplified versions. "
        "Include type hints and one-line docstrings. "
        "If code is long, omit usage example - NEVER omit implementation."
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


def _mistral_stream(messages: list, system: str = "", model: str = "magistral-medium-latest", max_tokens: int = 2048) -> str:
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
def detect_emotion_context(msg: str) -> dict:
    """Stub — returns neutral emotion context."""
    return {"emotion": "neutral", "intensity": 0.0, "requires_empathy": False}
