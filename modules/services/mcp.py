
# Claude-style MCP tool honesty rules
MCP_HONESTY = """
When using MCP tools:
- Never fabricate tool results — if a tool fails, report the failure clearly
- Always show the raw tool result before interpreting it
- If a tool returns unexpected output, flag it rather than silently correcting it
- Prefer one accurate tool call over multiple speculative ones
- Never claim a tool succeeded if it returned an error
"""
from modules.core.constants import _tool_exec
import uuid
import os
import threading
# AUTO-SPLIT FROM app.py lines 3379-3637
import os, re, time, math, json, asyncio, random, ast, subprocess, sys, tempfile
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import urllib.request, urllib.parse

# Add your MCP servers here or POST to /mcp/servers at runtime.
# Format: {"name": str, "url": str, "auth": optional bearer token}
_MCP_SERVERS: list = [
    # ── Filesystem — read/write local files and directories ───────────────────
    # Install: npx -y @modelcontextprotocol/server-filesystem /path/to/allow
    {"name": "filesystem",      "url": "http://localhost:3001", "auth": ""},
    {"name": "gdrive",          "url": "http://localhost:3030", "auth": os.environ.get("GOOGLE_TOKEN", "")},
    {"name": "gmail",           "url": "http://localhost:3031", "auth": os.environ.get("GOOGLE_TOKEN", "")},
    {"name": "gsheets",         "url": "http://localhost:3032", "auth": os.environ.get("GOOGLE_TOKEN", "")},
    {"name": "gdocs",           "url": "http://localhost:3033", "auth": os.environ.get("GOOGLE_TOKEN", "")},

    # ── GitHub — repos, issues, PRs, code search ──────────────────────────────
    # Install: npx -y @modelcontextprotocol/server-github
    # Set env: GITHUB_PERSONAL_ACCESS_TOKEN=ghp_yourtoken
    {"name": "github",          "url": "http://localhost:3002", "auth": os.environ.get("GITHUB_TOKEN", "")},

    # ── Brave Search — web + local search via Brave API ───────────────────────
    # Install: npx -y @modelcontextprotocol/server-brave-search
    # Set env: BRAVE_API_KEY=your_key  (free tier: brave.com/search/api)
    {"name": "brave-search",    "url": "http://localhost:3003", "auth": os.environ.get("BRAVE_API_KEY", "")},

    # ── Fetch — fetch any URL as clean text/markdown ──────────────────────────
    # Install: npx -y @modelcontextprotocol/server-fetch
    {"name": "fetch",           "url": "http://localhost:3004", "auth": ""},

    # ── Memory — persistent key-value memory across sessions ──────────────────
    # Install: npx -y @modelcontextprotocol/server-memory
    {"name": "memory",          "url": "http://localhost:3005", "auth": ""},

    # ── SQLite — query and manage local SQLite databases ──────────────────────
    # Install: npx -y @modelcontextprotocol/server-sqlite --db-path ./data.db
    {"name": "sqlite",          "url": "http://localhost:3006", "auth": ""},

    # ── PostgreSQL — read-only access to a Postgres database ──────────────────
    # Install: npx -y @modelcontextprotocol/server-postgres
    # Set env: DATABASE_URL=postgresql://user:pass@localhost/dbname
    {"name": "postgres",        "url": "http://localhost:3007", "auth": os.environ.get("DATABASE_URL", "")},

    # ── Slack — read channels, post messages ──────────────────────────────────
    # Install: npx -y @modelcontextprotocol/server-slack
    # Set env: SLACK_BOT_TOKEN=xoxb-...  SLACK_TEAM_ID=T...
    {"name": "slack",           "url": "http://localhost:3008", "auth": os.environ.get("SLACK_BOT_TOKEN", "")},

    # ── Google Drive — search and read Drive files ────────────────────────────
    # Install: npx -y @modelcontextprotocol/server-gdrive
    # Requires OAuth2 credentials: see MCP docs
    {"name": "gdrive",          "url": "http://localhost:3009", "auth": ""},

    # ── Google Maps — geocoding, directions, places ───────────────────────────
    # Install: npx -y @modelcontextprotocol/server-google-maps
    # Set env: GOOGLE_MAPS_API_KEY=your_key
    {"name": "google-maps",     "url": "http://localhost:3010", "auth": os.environ.get("GOOGLE_MAPS_API_KEY", "")},

    # ── Git — local git repo operations (log, diff, commit, etc.) ─────────────
    # Install: uvx mcp-server-git --repository /path/to/repo
    {"name": "git",             "url": "http://localhost:3011", "auth": ""},

    # ── Puppeteer — browser automation, screenshots, scraping ─────────────────
    # Install: npx -y @modelcontextprotocol/server-puppeteer
    {"name": "puppeteer",       "url": "http://localhost:3012", "auth": ""},

    # ── AWS KB Retrieval — query AWS Bedrock Knowledge Bases ──────────────────
    # Install: npx -y @modelcontextprotocol/server-aws-kb-retrieval
    # Set env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
    {"name": "aws-kb",          "url": "http://localhost:3013", "auth": os.environ.get("AWS_ACCESS_KEY_ID", "")},

    # ── EverArt — AI image generation ─────────────────────────────────────────
    # Install: npx -y @modelcontextprotocol/server-everart
    # Set env: EVERART_API_KEY=your_key
    {"name": "everart",         "url": "http://localhost:3014", "auth": os.environ.get("EVERART_API_KEY", "")},

    # ── Sequential Thinking — structured multi-step reasoning tool ────────────
    # Install: npx -y @modelcontextprotocol/server-sequential-thinking
    {"name": "sequential-think","url": "http://localhost:3015", "auth": ""},

    # ── Sentry — query Sentry issues and error traces ─────────────────────────
    # Install: npx -y @modelcontextprotocol/server-sentry
    # Set env: SENTRY_AUTH_TOKEN=your_token  SENTRY_ORG=your_org
    {"name": "sentry",          "url": "http://localhost:3016", "auth": os.environ.get("SENTRY_AUTH_TOKEN", "")},

    # ── Linear — issues, projects, team management ────────────────────────────
    # Install: npx -y @modelcontextprotocol/server-linear
    # Set env: LINEAR_API_KEY=your_key
    {"name": "linear",          "url": "http://localhost:3017", "auth": os.environ.get("LINEAR_API_KEY", "")},

    # ── Notion — read and write Notion pages/databases ────────────────────────
    # Install: npx -y @modelcontextprotocol/server-notion
    # Set env: NOTION_API_TOKEN=secret_...
    {"name": "notion",          "url": "http://localhost:3018", "auth": os.environ.get("NOTION_API_TOKEN", "")},

    # ── Stripe — payments, customers, subscriptions ───────────────────────────
    # Install: npx -y @modelcontextprotocol/server-stripe
    # Set env: STRIPE_SECRET_KEY=sk_...
    {"name": "stripe",          "url": "http://localhost:3019", "auth": os.environ.get("STRIPE_SECRET_KEY", "")},

    # ── Cloudflare — manage workers, KV, R2, D1 ──────────────────────────────
    # Install: npx -y @modelcontextprotocol/server-cloudflare
    # Set env: CLOUDFLARE_API_TOKEN=your_token
    {"name": "cloudflare",      "url": "http://localhost:3020", "auth": os.environ.get("CLOUDFLARE_API_TOKEN", "")},
]
# NOTE: servers that fail discovery are silently skipped — only running servers
# contribute tools. Add your auth tokens via environment variables (see above)
# or edit directly. Ports above are defaults; change if yours differ.
_MCP_TOOLS:   dict = {}          # name → {server, schema, description}
_MCP_LOCK     = threading.Lock()

MCP_TIMEOUT = 10   # seconds per tool call

# ── JSON-RPC helpers ──────────────────────────────────────────────────────────
def _mcp_rpc(url: str, method: str, params: dict, auth: str = "") -> dict:
    """
    Send a single JSON-RPC 2.0 request to an MCP server endpoint.
    Returns the result dict or raises on error.
    """
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id":      str(uuid.uuid4()),
        "method":  method,
        "params":  params,
    }).encode()
    req = urllib.request.Request(
        url,
        data    = payload,
        headers = {
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            **({"Authorization": f"Bearer {auth}"} if auth else {}),
        },
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=MCP_TIMEOUT) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError(f"MCP RPC error ({url}): {e}")
    if "error" in body:
        raise RuntimeError(f"MCP error: {body['error']}")
    return body.get("result", {})

# ── Tool discovery ────────────────────────────────────────────────────────────
def mcp_discover(server: dict) -> list:
    if not _mcp_server_up(server):
        return []  # skip dead server silently — degrade gracefully
    """
    Call tools/list on an MCP server and return its tool schemas.
    Stores discovered tools in _MCP_TOOLS for dispatcher use.
    """
    url  = server["url"].rstrip("/") + "/rpc"
    auth = server.get("auth", "")
    try:
        result = _mcp_rpc(url, "tools/list", {}, auth)
        tools  = result.get("tools", [])
        with _MCP_LOCK:
            for t in tools:
                name = t.get("name", "")
                if name:
                    _MCP_TOOLS[name] = {
                        "server":      server,
                        "description": t.get("description", ""),
                        "schema":      t.get("inputSchema", {}),
                    }
        print(f"  [MCP] {server['name']}: discovered {len(tools)} tools: "
              f"{[t.get('name') for t in tools]}")
        return tools
    except Exception as e:
        print(f"  [MCP] Discovery failed for {server['name']}: {e}")
        return []

def mcp_discover_all():
    """Discover tools from all registered MCP servers (runs at startup)."""
    from concurrent.futures import ThreadPoolExecutor as _TE
    with _TE(max_workers=8) as ex:
        list(ex.map(mcp_discover, _MCP_SERVERS))

# ── Tool call ─────────────────────────────────────────────────────────────────
def mcp_call_tool(tool_name: str, arguments: dict) -> str:
    """
    Call a discovered MCP tool by name with the given arguments.
    Returns the tool result as a string for context injection.
    """
    with _MCP_LOCK:
        meta = _MCP_TOOLS.get(tool_name)
    if not meta:
        return f"[MCP: unknown tool '{tool_name}']"
    url  = meta["server"]["url"].rstrip("/") + "/rpc"
    auth = meta["server"].get("auth", "")
    try:
        result = _mcp_rpc(url, "tools/call", {"name": tool_name, "arguments": arguments}, auth)
        # MCP returns content as list of {type, text} blocks
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(texts) or str(result)
        return str(content)
    except Exception as e:
        return f"[MCP tool error: {e}]"

# ── Model-facing dispatcher (MCP tool syntax: MCP.tool_name(json_args)) ───────
_MCP_RE = re.compile(r'\bMCP\.([\w-]+)\(([^)]*)\)', re.DOTALL)

def _parse_mcp_args(raw: str) -> dict:
    """Parse MCP tool arguments — accepts JSON object or key=value pairs."""
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            pass
    # key=value fallback
    out = {}
    for part in raw.split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip().strip('"\'\"')
    return out

def run_mcp_tools(text: str) -> str:
    """
    Find MCP.tool_name(args) calls in model output and execute them.
    Substitutes results inline as [= result].
    """
    calls = _MCP_RE.findall(text)
    if not calls:
        return text

    seen: dict = {}
    for tool_name, raw_args in calls:
        key = f"MCP.{tool_name}({raw_args})"
        if key not in seen:
            args = _parse_mcp_args(raw_args)
            seen[key] = _tool_exec.submit(mcp_call_tool, tool_name, args)

    results: dict = {}
    for key, fut in seen.items():
        try:
            results[key] = fut.result(timeout=MCP_TIMEOUT + 2)
        except Exception as e:
            results[key] = f"[MCP timeout/error: {e}]"

    def _replace(m: re.Match) -> str:
        key = f"MCP.{m.group(1)}({m.group(2)})"
        return f"{key} [= {results.get(key, '?')}]"

    return _MCP_RE.sub(_replace, text)

# ── MCP context builder ───────────────────────────────────────────────────────
def mcp_tool_list_prompt() -> str:
    """Return a formatted list of available MCP tools for the system prompt."""
    with _MCP_LOCK:
        tools = dict(_MCP_TOOLS)
    if not tools:
        return ""
    lines = ["MCP TOOLS (call as MCP.tool_name({...json args...})):"]
    for name, meta in tools.items():
        desc = meta["description"][:120]
        lines.append(f"  MCP.{name} — {desc}")
    return "\n".join(lines)

# ── REST API for runtime server management ────────────────────────────────────


# ── MCP HEALTH CACHE — skip dead servers without hammering them ───────────────
_mcp_health_cache: dict = {}   # url -> (is_up: bool, last_checked: float)
_MCP_HEALTH_TTL = 30.0         # re-check every 30s

def _mcp_server_up(server: dict) -> bool:
    """
    Check if an MCP server is reachable before trying to call it.
    Caches result for 30s to avoid hammering dead servers.
    Degrades gracefully — returns False instead of raising.
    """
    url = server.get("url", "")
    if not url:
        return False
    now = __import__("time").time()
    cached = _mcp_health_cache.get(url)
    if cached and (now - cached[1]) < _MCP_HEALTH_TTL:
        return cached[0]
    try:
        import urllib.request as _ur
        _ur.urlopen(f"{url}/health", timeout=2)
        _mcp_health_cache[url] = (True, now)
        return True
    except Exception:
        pass
    # Try root path as fallback
    try:
        import urllib.request as _ur
        _ur.urlopen(url, timeout=2)
        _mcp_health_cache[url] = (True, now)
        return True
    except Exception:
        _mcp_health_cache[url] = (False, now)
        return False
