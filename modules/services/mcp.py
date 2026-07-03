# modules/services/mcp.py — stdio-based MCP subprocess manager
import os, json, uuid, threading, subprocess, time, re
from modules.core.constants import _tool_exec

MCP_HONESTY = """
When using MCP tools:
- Never fabricate tool results — if a tool fails, report the failure clearly
- Always show the raw tool result before interpreting it
- If a tool returns unexpected output, flag it rather than silently correcting it
- Prefer one accurate tool call over multiple speculative ones
- Never claim a tool succeeded if it returned an error
"""

MCP_TIMEOUT = 15

# ── Server definitions (stdio transport) ─────────────────────────────────────
_MCP_SERVER_CMDS = {
    "filesystem":        ["npx", "-y", "@modelcontextprotocol/server-filesystem", os.path.expanduser("~/eliteomni_app")],
    "memory":            ["npx", "-y", "@modelcontextprotocol/server-memory"],
    "sequential-think":  ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
    "sqlite":           ["npx", "-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/home/kidus/eliteomni_app/eliteomni.db"],
    "puppeteer":        ["npx", "-y", "@modelcontextprotocol/server-puppeteer"],
    "git":              ["/home/kidus/.local/bin/uvx", "mcp-server-git", "--repository", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))],
}

# ── Per-server subprocess state ───────────────────────────────────────────────
_procs:   dict = {}   # name -> subprocess.Popen
_locks:   dict = {n: threading.Lock() for n in _MCP_SERVER_CMDS}
_MCP_TOOLS: dict = {}
_TOOLS_LOCK = threading.Lock()

def _get_proc(name: str) -> subprocess.Popen:
    """Return running proc for server, starting it if needed."""
    proc = _procs.get(name)
    if proc and proc.poll() is None:
        return proc
    cmd = _MCP_SERVER_CMDS[name]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=0,
    )
    _procs[name] = proc
    print(f"[MCP] started {name} (pid {proc.pid})")
    return proc

def _rpc(name: str, method: str, params: dict) -> dict:
    """Send JSON-RPC to a stdio server, return result dict."""
    with _locks[name]:
        proc = _get_proc(name)
        msg = json.dumps({"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params})
        proc.stdin.write(msg + "\n")
        proc.stdin.flush()
        deadline = time.time() + MCP_TIMEOUT
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                resp = json.loads(line)
                if "error" in resp:
                    raise RuntimeError(resp["error"])
                return resp.get("result", {})
            except json.JSONDecodeError:
                continue
        raise TimeoutError(f"MCP {name} timed out")

# ── Discovery ─────────────────────────────────────────────────────────────────
def mcp_discover(name: str):
    try:
        result = _rpc(name, "tools/list", {})
        tools = result.get("tools", [])
        with _TOOLS_LOCK:
            for t in tools:
                tname = t.get("name", "")
                if tname:
                    _MCP_TOOLS[tname] = {
                        "server":      name,
                        "description": t.get("description", ""),
                        "schema":      t.get("inputSchema", {}),
                    }
        print(f"[MCP] {name}: {len(tools)} tools: {[t.get('name') for t in tools]}")
    except Exception as e:
        print(f"[MCP] discover {name} failed: {e}")

def mcp_discover_all():
    threads = [threading.Thread(target=mcp_discover, args=(n,), daemon=True) for n in _MCP_SERVER_CMDS]
    for t in threads: t.start()
    for t in threads: t.join()

# ── Tool call ─────────────────────────────────────────────────────────────────
def mcp_call(tool_name: str, arguments: dict) -> str:
    with _TOOLS_LOCK:
        entry = _MCP_TOOLS.get(tool_name)
    if not entry:
        return f"[MCP ERROR] Unknown tool: {tool_name}"
    try:
        result = _rpc(entry["server"], "tools/call", {"name": tool_name, "arguments": arguments})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else json.dumps(result)[:1000]
    except Exception as e:
        return f"[MCP ERROR] {tool_name}: {e}"

def mcp_call_tool(tool_name: str, arguments: dict) -> str:
    return mcp_call(tool_name, arguments)

# ── Prompt helpers ────────────────────────────────────────────────────────────
def mcp_tool_list_prompt() -> str:
    with _TOOLS_LOCK:
        tools = dict(_MCP_TOOLS)
    if not tools:
        return ""
    lines = ["MCP TOOLS (call as MCP.tool_name({...json args...})):"]
    for name, meta in tools.items():
        lines.append(f"  MCP.{name} — {meta['description'][:120]}")
    return "\n".join(lines)

def mcp_tools_prompt() -> str:
    with _TOOLS_LOCK:
        if not _MCP_TOOLS:
            return ""
        lines = ["You have access to these tools via MCP. To use one, output exactly:",
                 'MCP_CALL(tool_name, {"arg1": "value1", ...})',
                 "on its own line, then STOP and wait for the result.", ""]
        for name, t in _MCP_TOOLS.items():
            lines.append(f"- {name}: {t['description'][:150]}")
    return "\n".join(lines)

# ── Inline call dispatcher ────────────────────────────────────────────────────
_MCP_RE = re.compile(r'\bMCP\.([\w-]+)\(([^)]*)\)', re.DOTALL)

def _parse_mcp_args(raw: str) -> dict:
    raw = raw.strip()
    if not raw: return {}
    if raw.startswith("{"):
        try: return json.loads(raw)
        except: pass
    out = {}
    for part in raw.split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip().strip('"\'')
    return out

def run_mcp_tools(text: str) -> str:
    calls = _MCP_RE.findall(text)
    if not calls: return text
    seen = {}
    for tool_name, raw_args in calls:
        key = f"MCP.{tool_name}({raw_args})"
        if key not in seen:
            args = _parse_mcp_args(raw_args)
            seen[key] = _tool_exec.submit(mcp_call, tool_name, args)
    results = {}
    for key, fut in seen.items():
        try: results[key] = fut.result(timeout=MCP_TIMEOUT + 2)
        except Exception as e: results[key] = f"[MCP timeout/error: {e}]"
    def _replace(m):
        key = f"MCP.{m.group(1)}({m.group(2)})"
        return f"{key} [= {results.get(key, '?')}]"
    return _MCP_RE.sub(_replace, text)

# ── Status for health monitor ─────────────────────────────────────────────────
def mcp_status() -> dict:
    """Return {name: 'up'|'down'} for all configured servers."""
    status = {}
    for name in _MCP_SERVER_CMDS:
        proc = _procs.get(name)
        status[name] = "up" if (proc and proc.poll() is None) else "down"
    return status
