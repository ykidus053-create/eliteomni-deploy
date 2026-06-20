import urllib.request as _urlmod
import json as _json
import time as _time
import sys as _sys
import os as _os
import threading as _threading
import traceback as _tb
import functools as _ft
import logging as _logging
import collections as _col
import socket as _socket
import io as _io

DEBUG_LOG_PATH  = _os.path.expanduser("~/eliteomni_debug.log")
SLOW_MS         = 3000
MAX_PAYLOAD_BYTES = 30_000
VERBOSE         = _os.environ.get("ELITE_DEBUG_VERBOSE", "1") == "1"
_lock         = _threading.Lock()
_req_counter  = 0

_GROQ_PRICING = {
    "llama-3.3-70b-versatile":  {"in": 0.59,  "out": 0.79},
    "llama-3.1-8b-instant":     {"in": 0.05,  "out": 0.08},
    "groq/compound":            {"in": 0.0,   "out": 0.0},
    "compound-beta":            {"in": 0.0,   "out": 0.0},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"in": 0.11, "out": 0.34},
    "meta-llama/llama-guard-4-12b": {"in": 0.20, "out": 0.20},
}

_stats = {
    "total_requests":   0,
    "groq_requests":    0,
    "groq_errors":      0,
    "total_input_tok":  0,
    "total_output_tok": 0,
    "total_cost_usd":   0.0,
    "latencies_ms":     _col.deque(maxlen=200),
    "errors_by_code":   _col.Counter(),
    "models_used":      _col.Counter(),
    "slow_calls":       0,
    "413_count":        0,
    "429_count":        0,
    "empty_responses":  0,
    "truncated":        0,
}

_logger = _logging.getLogger("EliteOmniDebug")
_logger.setLevel(_logging.DEBUG)
_fmt = _logging.Formatter("\033[90m%(asctime)s\033[0m [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_fmt_file = _logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
if not _logger.handlers:
    _fh = _logging.FileHandler(DEBUG_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(_fmt_file)
    _sh = _logging.StreamHandler(_sys.stdout)
    _sh.setFormatter(_fmt)
    _logger.addHandler(_fh)
    _logger.addHandler(_sh)

_C = {"reset":"\033[0m","red":"\033[91m","green":"\033[92m","yellow":"\033[93m","blue":"\033[94m","cyan":"\033[96m","grey":"\033[90m","bold":"\033[1m","magenta":"\033[95m"}
def _col_str(s, c): return f"{_C.get(c,'')}{s}{_C['reset']}"

def _log(level, section, msg, data=None):
    colour_map = {"debug":"grey","info":"cyan","warning":"yellow","error":"red","critical":"magenta"}
    c = colour_map.get(level, "reset")
    tag = _col_str(f"[{section}]", c)
    line = f"{tag} {msg}"
    if data:
        for k, v in data.items():
            line += f"\n    {_col_str(k, 'grey'):<28}: {str(v)[:300]}"
    getattr(_logger, level)(line)

def _classify_error(status_code, body, model=""):
    hints = {
        413: f"PAYLOAD TOO LARGE — model '{model}' token limit exceeded. Lower search_ctx cap or trim_msgs.",
        429: "RATE LIMITED — Groq rate limit hit. Auto-retry with backoff.",
        400: f"BAD REQUEST — check model name, message format. body={body[:150]}",
        401: "UNAUTHORIZED — check GROQ_API_KEY is set and valid.",
        404: f"NOT FOUND — model '{model}' may not exist on Groq.",
        500: "GROQ SERVER ERROR — transient, retry in a few seconds.",
        503: "GROQ UNAVAILABLE — try again shortly.",
    }
    return hints.get(status_code, f"HTTP {status_code}: {body[:150]}")

def _check_quality(reply, finish, model, req_id):
    issues = []
    if not reply or len(reply.strip()) < 5:
        issues.append("EMPTY response"); _stats["empty_responses"] += 1
    if finish == "length":
        issues.append("TRUNCATED — hit max_completion_tokens"); _stats["truncated"] += 1
    if finish == "content_filter":
        issues.append("CONTENT FILTERED by Groq")
    if issues:
        _log("warning", f"QUALITY #{req_id}", " | ".join(issues), {"model": model, "finish": finish, "len": len(reply)})

def _calc_cost(model, in_tok, out_tok):
    p = _GROQ_PRICING.get(model, {"in": 0.5, "out": 0.8})
    return (in_tok * p["in"] + out_tok * p["out"]) / 1_000_000

_real_urlopen = _urlmod.urlopen

_tls = _threading.local()

def _debug_urlopen(req, **kwargs):
    if getattr(_tls, "_in_call", False):
        return _real_urlopen(req, **kwargs)
    _tls._in_call = True
    try:
        return _debug_urlopen_inner(req, **kwargs)
    finally:
        _tls._in_call = False

def _debug_urlopen_inner(req, **kwargs):
    global _req_counter
    url = req.full_url if hasattr(req, "full_url") else str(req)
    is_groq    = "groq.com" in url
    is_searxng = "8888" in url or "8889" in url or "searxng" in url.lower()
    is_mcp     = any(f":{p}" in url for p in range(3001, 3035))
    with _lock:
        _req_counter += 1
        req_id = _req_counter
        _stats["total_requests"] += 1
    t0 = _time.time()
    model = ""
    if (is_groq) and hasattr(req, "data") and req.data:
        if len(req.data) > MAX_PAYLOAD_BYTES:
            _log("warning", f"PAYLOAD GUARD #{req_id}", _col_str(f"⚠️  {len(req.data):,} bytes — may hit 413!", "yellow"))
        try:
            body = _json.loads(req.data.decode())
            msgs = body.get("messages", [])
            model = body.get("model", "?")
            total_chars = sum(len(str(m.get("content", ""))) for m in msgs)
            _stats["groq_requests"] += 1
            _stats["models_used"][model] += 1
            msg_details = {}
            for i, m in enumerate(msgs):
                c = str(m.get("content", ""))
                msg_details[f"msg[{i}] {m.get('role','?')} ~{len(c)//4}tok"] = c[:180].replace("\n", " ")
            _log("debug", f"GROQ REQ #{req_id}", _col_str("="*64, "blue"), {
                "Model": _col_str(model, "cyan"), "Messages": len(msgs),
                "Chars": f"{total_chars:,}", "Bytes": f"{len(req.data):,}",
                "Est tokens": f"~{total_chars//4:,}",
                "Max tokens": body.get("max_completion_tokens") or body.get("max_tokens"),
                "Stream": body.get("stream", False), "Tools": len(body.get("tools", [])),
                **msg_details,
            })
        except Exception as e:
            _log("warning", f"GROQ REQ PARSE #{req_id}", str(e))
    elif is_searxng:
        _log("debug", f"SEARXNG REQ #{req_id}", url)
    elif is_mcp:
        try:
            b = _json.loads(req.data.decode()) if req.data else {}
            _log("debug", f"MCP REQ #{req_id}", url, {"method": b.get("method"), "params": str(b.get("params",""))[:150]})
        except:
            _log("debug", f"MCP REQ #{req_id}", url)
    try:
        resp = _real_urlopen(req, **kwargs)
        latency = round((_time.time() - t0) * 1000)
        _stats["latencies_ms"].append(latency)
        if latency > SLOW_MS:
            _stats["slow_calls"] += 1
            _log("warning", f"SLOW #{req_id}", _col_str(f"🐢 {latency}ms", "yellow"), {"URL": url, "Model": model})
        if is_groq:
            try:
                _req_body = _json.loads(req.data.decode()) if req.data else {}
                _is_stream = _req_body.get("stream", False)
            except:
                _is_stream = False
            if _is_stream:
                _log("debug", f"GROQ STREAM #{req_id}", _col_str(f"✓ {latency}ms streaming", "cyan"), {"Model": model})
                # Do NOT read response body for streams — it would consume the SSE data
            else:
                try:
                    raw = resp.read()
                    data = _json.loads(raw.decode())
                    choices = data.get("choices", [])
                    usage   = data.get("usage", {})
                    in_tok  = usage.get("prompt_tokens", 0)
                    out_tok = usage.get("completion_tokens", 0)
                    cost    = _calc_cost(model, in_tok, out_tok)
                    cached  = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    finish  = choices[0].get("finish_reason", "?") if choices else "?"
                    reply   = ""
                    if choices:
                        msg0 = choices[0].get("message", {})
                        reply = msg0.get("content") or msg0.get("reasoning") or ""
                    _stats["total_input_tok"]  += in_tok
                    _stats["total_output_tok"] += out_tok
                    _stats["total_cost_usd"]   += cost
                    _log("info", f"GROQ RESP #{req_id}", _col_str(f"✓ {latency}ms", "green"), {
                        "Model": model, "Latency": f"{latency}ms",
                        "In tok": f"{in_tok:,}", "Out tok": f"{out_tok:,}",
                        "Cached": f"{cached:,} ({round(cached/max(in_tok,1)*100)}%)",
                        "Cost": _col_str(f"${cost:.6f}", "yellow"),
                        "Session $": _col_str(f"${_stats['total_cost_usd']:.5f}", "yellow"),
                        "Finish": finish, "Reply": reply[:300].replace("\n", " "),
                    })
                    _check_quality(reply, finish, model, req_id)
                    resp = _urlmod.addinfourl(_io.BytesIO(raw), resp.headers, resp.url, resp.status)
                except Exception as e:
                    if "Expecting value" not in str(e):
                        _log("warning", f"GROQ RESP PARSE #{req_id}", str(e))
        elif is_searxng:
            _log("debug", f"SEARXNG RESP #{req_id}", _col_str(f"✓ {latency}ms", "green"))
        else:
            _log("debug", f"HTTP RESP #{req_id}", _col_str(f"✓ {latency}ms", "green") + f" {url}")
        return resp
    except Exception as err:
        latency = round((_time.time() - t0) * 1000)
        try: body_str = err.read().decode()[:1000]; status = getattr(err, "code", 0)
        except: body_str, status = str(err), 0
        _stats["groq_errors"] += 1
        _stats["errors_by_code"][status] += 1
        if status == 413: _stats["413_count"] += 1
        if status == 429: _stats["429_count"] += 1
        import traceback as _tb2; _log("error", f"STACK", _tb2.format_stack()[-8:])
        _log("error", f"HTTP ERR #{req_id}", _col_str(f"✗ {latency}ms", "red"), {
            "URL": url, "Error": str(err), "Body": body_str,
            "Hint": _col_str(_classify_error(status, body_str, model), "yellow"),
        })
        raise

if _urlmod.urlopen is not _debug_urlopen:
    _urlmod.urlopen = _debug_urlopen

# Trap: detect if anything re-patches urlopen after us
import sys as _sys
class _UrlopenGuard:
    def __init__(self):
        self._patched = _debug_urlopen
    def __set_name__(self, owner, name): pass

_orig_setattr = type(_urlmod).__setattr__ if hasattr(type(_urlmod), '__setattr__') else None

def _watch_urlopen():
    import time as _t
    while True:
        _t.sleep(1)
        current = _urlmod.urlopen
        if current is not _debug_urlopen:
            import traceback as _traceback
            print(f"[DEBUG PATCH] ⚠️  urlopen was replaced by: {current}")
            _traceback.print_stack()
            _urlmod.urlopen = _debug_urlopen

_threading.Thread(target=_watch_urlopen, daemon=True, name="urlopen_guard").start()

def install_fastapi_debug(app):
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    class DebugMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            t0 = _time.time()
            _log("info", "HTTP IN", f"{request.method} {request.url.path}", {
                "Client": request.client.host if request.client else "?",
                "Query": str(request.query_params)[:200],
                "Headers": dict(list(request.headers.items())[:6]),
            })
            try:
                response = await call_next(request)
                latency = round((_time.time() - t0) * 1000)
                c = "green" if response.status_code < 400 else "red"
                _log("info", "HTTP OUT", _col_str(f"{request.method} {request.url.path} → {response.status_code} {latency}ms", c))
                return response
            except Exception as e:
                _log("error", "HTTP CRASH", f"crashed after {round((_time.time()-t0)*1000)}ms", {"Exception": str(e), "Traceback": _tb.format_exc()[-800:]})
                raise
    app.add_middleware(DebugMiddleware)

def register_debug_routes(app):
    from fastapi.responses import JSONResponse, HTMLResponse
    @app.get("/debug/stats")
    async def debug_stats():
        lats = sorted(_stats["latencies_ms"]); n = len(lats)
        return JSONResponse({**{k: v for k, v in _stats.items() if k != "latencies_ms"},
            "p50_ms": lats[int(n*0.5)] if n else 0, "p95_ms": lats[int(n*0.95)] if n else 0,
            "p99_ms": lats[int(n*0.99)] if n else 0,
            "models_used": dict(_stats["models_used"]), "errors_by_code": dict(_stats["errors_by_code"])})
    @app.get("/debug/tail")
    async def debug_tail(lines: int = 150):
        try:
            with open(DEBUG_LOG_PATH, "r", encoding="utf-8") as f: all_lines = f.readlines()
            return "\n".join(l.rstrip() for l in all_lines[-lines:])
        except Exception as e: return f"Error: {e}"
    @app.post("/debug/reset_stats")
    async def debug_reset():
        for k in ("groq_requests","groq_errors","total_input_tok","total_output_tok","slow_calls","413_count","429_count","empty_responses","truncated","total_requests"):
            _stats[k] = 0
        _stats["total_cost_usd"] = 0.0; _stats["latencies_ms"].clear()
        _stats["errors_by_code"].clear(); _stats["models_used"].clear()
        return {"status": "reset"}
    @app.get("/debug/live", response_class=HTMLResponse)
    async def debug_live():
        return HTMLResponse("""<!DOCTYPE html><html><head><meta charset=UTF-8><title>EliteOmni Debug</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{background:#0a0c12;color:#e0e6ff;font-family:'DM Mono',monospace;font-size:13px;padding:20px}h1{color:#7aaeff;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:16px}
.card{background:#13162b;border:1px solid #2a2f50;border-radius:10px;padding:12px}
.card h3{color:#7aaeff;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.val{font-size:20px;font-weight:bold;color:#fff}.val.warn{color:#f0c040}.val.bad{color:#e05555}.val.good{color:#27c47a}
.log{background:#0d0f1e;border:1px solid #1e2240;border-radius:8px;padding:10px;height:420px;overflow-y:auto;font-size:11px;line-height:1.7;white-space:pre-wrap}
button{background:#1e2a5e;border:1px solid #3a4a8e;color:#7aaeff;padding:6px 14px;border-radius:6px;cursor:pointer;margin-right:6px;margin-bottom:12px}
button:hover{background:#2a3a7e}.ts{color:#555;font-size:10px}
</style></head><body>
<h1>⚡ EliteOmni Debug Dashboard</h1>
<button onclick="load()">🔄 Refresh</button>
<button onclick="tailLog()">📜 Tail Log</button>
<button onclick="fetch('/debug/reset_stats',{method:'POST'}).then(()=>load())">🗑 Reset Stats</button>
<div class="grid" id="cards"></div>
<div class="log" id="log">Click Refresh or Tail Log...</div>
<script>
async function load(){
  const d=await(await fetch('/debug/stats')).json();
  const cards=[
    {l:'Groq Requests',v:d.groq_requests,c:''},
    {l:'Errors',v:d.groq_errors,c:d.groq_errors>0?'bad':'good'},
    {l:'413s (Too Large)',v:d['413_count'],c:d['413_count']>0?'bad':'good'},
    {l:'429s (Rate Limit)',v:d['429_count'],c:d['429_count']>0?'warn':'good'},
    {l:'Input Tokens',v:(d.total_input_tok||0).toLocaleString(),c:''},
    {l:'Output Tokens',v:(d.total_output_tok||0).toLocaleString(),c:''},
    {l:'Session Cost',v:'$'+(d.total_cost_usd||0).toFixed(5),c:'warn'},
    {l:'Empty Responses',v:d.empty_responses,c:d.empty_responses>0?'warn':'good'},
    {l:'Truncated',v:d.truncated,c:d.truncated>0?'warn':'good'},
    {l:'Slow Calls',v:d.slow_calls,c:d.slow_calls>0?'warn':'good'},
    {l:'p50 Latency',v:(d.p50_ms||0)+'ms',c:''},
    {l:'p95 Latency',v:(d.p95_ms||0)+'ms',c:d.p95_ms>5000?'warn':''},
    {l:'p99 Latency',v:(d.p99_ms||0)+'ms',c:d.p99_ms>10000?'bad':''},
    {l:'Total Requests',v:d.total_requests,c:''},
  ];
  document.getElementById('cards').innerHTML=cards.map(c=>`<div class="card"><h3>${c.l}</h3><div class="val ${c.c}">${c.v}</div></div>`).join('');
  document.getElementById('log').textContent='Models: '+JSON.stringify(d.models_used)+'\nErrors by code: '+JSON.stringify(d.errors_by_code);
}
async function tailLog(){
  const t=await(await fetch('/debug/tail?lines=150')).text();
  const el=document.getElementById('log');el.textContent=t;el.scrollTop=el.scrollHeight;
}
load();setInterval(load,8000);
</script></body></html>""")

_orig_excepthook = _sys.excepthook
def _debug_excepthook(t, v, tb):
    _log("critical", "UNHANDLED EXCEPTION", _col_str(str(v), "red"), {"Type": t.__name__, "Traceback": "".join(_tb.format_tb(tb))[-1000:]})
    _orig_excepthook(t, v, tb)
_sys.excepthook = _debug_excepthook

_orig_thread_hook = _threading.excepthook
def _debug_thread_hook(args):
    _log("critical", "THREAD EXCEPTION", _col_str(str(args.exc_value), "red"), {"Thread": args.thread.name if args.thread else "?", "Type": args.exc_type.__name__, "Traceback": "".join(_tb.format_tb(args.exc_traceback))[-800:]})
    _orig_thread_hook(args)
_threading.excepthook = _debug_thread_hook

def debug_timer(label=None):
    def decorator(fn):
        name = label or fn.__qualname__
        @_ft.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = _time.time()
            try:
                result = fn(*args, **kwargs)
                ms = round((_time.time()-t0)*1000)
                if ms > SLOW_MS: _log("warning", "SLOW FN", f"{name} took {ms}ms")
                return result
            except Exception as e:
                _log("error", f"FN ERR {name}", str(e), {"tb": _tb.format_exc()[-400:]}); raise
        import asyncio as _a
        @_ft.wraps(fn)
        async def awrapper(*args, **kwargs):
            t0 = _time.time()
            try:
                result = await fn(*args, **kwargs)
                ms = round((_time.time()-t0)*1000)
                if ms > SLOW_MS: _log("warning", "SLOW ASYNC FN", f"{name} took {ms}ms")
                return result
            except Exception as e:
                _log("error", f"ASYNC FN ERR {name}", str(e), {"tb": _tb.format_exc()[-400:]}); raise
        return awrapper if _a.iscoroutinefunction(fn) else wrapper
    return decorator

def _stats_reporter():
    try:
        import psutil, os; proc = psutil.Process(os.getpid()); has_psutil = True
    except ImportError:
        has_psutil = False; _log("warning", "SYSTEM STATS", "psutil not installed")
    while True:
        _time.sleep(60)
        try:
            lats = sorted(_stats["latencies_ms"]); n = len(lats)
            report = {
                "Groq requests": _stats["groq_requests"],
                "Errors": _col_str(str(_stats["groq_errors"]), "red") if _stats["groq_errors"] else "0",
                "413s": _col_str(str(_stats["413_count"]), "red") if _stats["413_count"] else "0",
                "429s": _col_str(str(_stats["429_count"]), "yellow") if _stats["429_count"] else "0",
                "Input tokens": f"{_stats['total_input_tok']:,}", "Output tokens": f"{_stats['total_output_tok']:,}",
                "Session cost": _col_str(f"${_stats['total_cost_usd']:.5f}", "yellow"),
                "Empty/Truncated": f"{_stats['empty_responses']}/{_stats['truncated']}",
                "Slow calls": _stats["slow_calls"],
                "p50/p95/p99": f"{lats[int(n*.5)] if n else 0}ms / {lats[int(n*.95)] if n else 0}ms / {lats[int(n*.99)] if n else 0}ms",
                "Models": dict(_stats["models_used"]),
            }
            if has_psutil:
                mem = proc.memory_info()
                report["RAM MB"] = round(mem.rss/1024/1024, 1)
                report["CPU %"] = proc.cpu_percent(interval=1)
                report["Threads"] = _threading.active_count()
            _log("info", "STATS REPORT", _col_str("="*50, "cyan"), report)
        except Exception as e: _log("error", "STATS REPORTER", str(e))

_threading.Thread(target=_stats_reporter, daemon=True, name="debug_stats").start()

def _searxng_watcher():
    while True:
        _time.sleep(30)
        try:
            r = _real_urlopen(_os.environ.get("SEARXNG_URL", "http://localhost:8888") + "/healthz", timeout=3)
            _log("info", "SEARXNG HEALTH", _col_str(f"✓ status={r.status}", "green"))
        except Exception as e: _log("warning", "SEARXNG HEALTH", _col_str(f"✗ {e}", "yellow"))
_threading.Thread(target=_searxng_watcher, daemon=True, name="debug_searxng").start()

_MCP_PORTS = {3001:"filesystem",3002:"github",3003:"brave-search",3004:"fetch",3005:"memory",3006:"sqlite",3007:"postgres",3008:"slack",3009:"gdrive",3010:"google-maps",3011:"git",3012:"puppeteer",3030:"gdrive2",3031:"gmail",3032:"gsheets",3033:"gdocs"}
def _mcp_watcher():
    while True:
        _time.sleep(60)
        try:
            from modules.services.mcp import mcp_status
            status = mcp_status()
            up   = [n for n,s in status.items() if s == "up"]
            down = [n for n,s in status.items() if s == "down"]
        except Exception as e:
            up, down = [], [f"error:{e}"]
        _log("info", "MCP STATUS", _col_str(f"{len(up)} up", "green") + f" / {len(down)} down", {"UP": ", ".join(up) or "none", "DOWN": ", ".join(down[:8]) or "none"})
_threading.Thread(target=_mcp_watcher, daemon=True, name="debug_mcp").start()

class SlowCallDetector:
    def __init__(self):
        self._active = {}; self._lock = _threading.Lock()
        _threading.Thread(target=self._monitor, daemon=True, name="debug_slow").start()
    def start(self, name):
        with self._lock: self._active[name] = _time.time()
    def done(self, name):
        with self._lock: self._active.pop(name, None)
    def _monitor(self):
        while True:
            _time.sleep(5); now = _time.time()
            with self._lock:
                for name, t0 in list(self._active.items()):
                    ms = round((now-t0)*1000)
                    if ms > SLOW_MS: _log("warning", "SLOW CALL", _col_str(f"🐢 {name} running {ms}ms", "yellow"))

slow_detector = SlowCallDetector()

_log("info", "DEBUG PATCH", f"✓ EliteOmni debug v2 — log → {DEBUG_LOG_PATH}", {
    "Groq HTTP":       "monkey-patched + token cost tracking",
    "Payload guard":   f"warns if >{MAX_PAYLOAD_BYTES:,} bytes",
    "Error hints":     "413/429/400/401/404/500 with fix hints",
    "Quality checker": "empty/truncated/filtered detection",
    "Stats reporter":  "60s — tokens, cost, latency p50/p95/p99",
    "Live dashboard":  "localhost:8000/debug/live",
    "Stats API":       "localhost:8000/debug/stats",
    "Log tail":        "localhost:8000/debug/tail",
})
