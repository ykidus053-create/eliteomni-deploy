import urllib.request, urllib.parse
import random
import os, re, time, json, urllib.request, urllib.parse
import pathlib as _pl
import threading as _threading
import hashlib

# ── ENV ───────────────────────────────────────────────────────────────────────
_env_path = _pl.Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── KEYS & CONSTANTS ──────────────────────────────────────────────────────────
MISTRAL_API_KEY     = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_URL         = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL       = "mistral-large-latest"
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL          = "llama-3.3-70b-versatile"
GROQ_URL            = "https://api.groq.com/openai/v1/chat/completions"
GROQ_CRITIC_MODEL   = "llama-3.3-70b-versatile"

# ── CODESTRAL ROUTING ─────────────────────────────────────────────────────────
CODESTRAL_MODEL = "codestral-latest"
CODESTRAL_URL   = "https://api.mistral.ai/v1/chat/completions"

CODING_SKILLS = {"coder", "code", "coding", "swe", "calculator", "debug", "refactor", "engineer"}

def get_model_for_skill(skill: str, model: str = None) -> str:
    """Return Codestral for coding tasks, Mistral-large for everything else."""
    if model:
        return model
    if skill and skill.lower() in CODING_SKILLS:
        print(f"[Router] skill={skill} → codestral-latest")
        return CODESTRAL_MODEL
    return model or MISTRAL_MODEL

FEEDBACK_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_store.json")

if not MISTRAL_API_KEY:
    print("[Mistral] WARNING: MISTRAL_API_KEY not set")
else:
    print("[Mistral] API key loaded")

def _get_next_key() -> str:
    return MISTRAL_API_KEY

# ── AUDIT ─────────────────────────────────────────────────────────────────────
_AUDIT_LOG_PATH = os.path.expanduser("~/eliteomni_audit.jsonl")
def _audit(event: str, data: dict):
    try:
        import datetime as _dt
        record = {"ts": _dt.datetime.utcnow().isoformat(), "event": event, **data}
        with open(_AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

# ── MESSAGE TRIMMER ───────────────────────────────────────────────────────────
def _trim_msgs(msgs: list, max_chars: int = 6000) -> list:
    system      = [m for m in msgs if m.get("role") == "system"]
    others      = [m for m in msgs if m.get("role") != "system"]
    sys_trimmed = [{**m, "content": m.get("content","")[:4000]} for m in system]
    oth_trimmed = [{**m, "content": m.get("content","")[:2000]} for m in others]
    budget = max_chars - sum(len(m.get("content","")) for m in sys_trimmed)
    kept = []
    for m in reversed(oth_trimmed):
        chars = len(m.get("content",""))
        if budget - chars < 200:
            break
        kept.insert(0, m)
        budget -= chars
    if not kept and oth_trimmed:
        kept = [oth_trimmed[-1]]
    return sys_trimmed + kept


# ── PERSISTENT HTTP SESSION (TCP connection reuse — 20-30ms TTFT saving) ─────
# urllib doesn't support keep-alive easily; use requests with a Session.
# The session holds an open TCP connection to api.mistral.ai across calls.
try:
    import requests as _requests
    _session = _requests.Session()
    _session.headers.update({
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type":  "application/json",
        "Connection":    "keep-alive",
    })
    _USE_SESSION = True
    print("[Mistral] HTTP session ready (keep-alive + prompt caching enabled)")
except ImportError:
    _USE_SESSION = False
    print("[Mistral] requests not available — falling back to urllib")

# ── GLOBAL RATE LIMITER ───────────────────────────────────────────────────────
_mistral_semaphore = _threading.Semaphore(2)
_rate_lock         = _threading.Lock()
_last_call_time    = 0.0

def _rate_wait():
    global _last_call_time
    with _rate_lock:
        elapsed = time.time() - _last_call_time
        gap = 15.0  # 4 RPM safe margin for Mistral
        if elapsed < gap:
            time.sleep(gap - elapsed)
        _last_call_time = time.time()

# ── MISTRAL STREAM ────────────────────────────────────────────────────────────
def mistral_stream(msgs: list, max_tokens: int = 2000, model: str = None, skill: str = None):
    if not MISTRAL_API_KEY:
        yield "[MISTRAL_API_KEY not set]"; return

    mdl      = get_model_for_skill(skill, model)
    trimmed  = _trim_msgs(msgs, max_chars=6000)

    # Route to correct endpoint based on model name
    if mdl.startswith("mistral-") or mdl.startswith("codestral-"):
        _url = "https://api.mistral.ai/v1/chat/completions"
        _key = os.environ.get("MISTRAL_API_KEY", MISTRAL_API_KEY)
        if _USE_SESSION:
            _session.headers.update({"Authorization": f"Bearer {_key}"})
    else:
        _url = MISTRAL_URL  # fireworks
        _key = MISTRAL_API_KEY

    payload = {
        "model":      mdl,
        "messages":   trimmed,
        "max_tokens": min(max_tokens, 16000),
        "temperature": 0.6,
        "stream":     True,
    }

    # ── helper: open one SSE stream and yield tokens ─────────────────────────
    def _do_stream():
        """Returns a generator of tokens from one HTTP attempt. Raises on 429/error."""
        if _USE_SESSION:
            resp = _session.post(_url, json=payload, stream=True, timeout=120)
            if resp.status_code == 429:
                raise _requests.exceptions.HTTPError(response=resp)
            resp.raise_for_status()
            for line in resp.iter_lines(chunk_size=None, decode_unicode=True):
                if not line: continue
                line = line.strip()
                if not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                try:
                    tok = json.loads(d)["choices"][0]["delta"].get("content", "")
                    if tok: yield tok
                except Exception: continue
        else:
            body = json.dumps(payload).encode()
            req  = urllib.request.Request(
                _url, data=body,
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                for line in r:
                    line = line.strip()
                    if not line.startswith("data:"): continue
                    d = line[5:].strip()
                    if d == "[DONE]": break
                    try:
                        tok = json.loads(d)["choices"][0]["delta"].get("content", "")
                        if tok: yield tok
                    except Exception: continue

    _rate_wait()
    for _attempt in range(5):
        try:
            # Stream tokens immediately — no buffering
            for tok in _do_stream():
                yield tok
            return  # clean exit

        except Exception as e:
            status = None
            if _USE_SESSION and hasattr(e, "response") and e.response is not None:
                status = e.response.status_code
            elif isinstance(e, urllib.error.HTTPError):
                status = e.code

            if status == 429:
                if _attempt == 4:
                    print("[Mistral 429] giving up after 5 attempts")
                    yield "[Rate limit reached — try again in a moment]"; return

                retry_after = None
                try:
                    if _USE_SESSION and hasattr(e, "response"):
                        retry_after = e.response.headers.get("Retry-After") or \
                                      e.response.headers.get("x-ratelimit-reset-requests")
                    elif isinstance(e, urllib.error.HTTPError):
                        retry_after = e.headers.get("Retry-After")
                except Exception:
                    pass
                try:
                    wait = float(retry_after) if retry_after else None
                except (ValueError, TypeError):
                    wait = None
                if wait is None:
                    wait = min(1.0 * (2 ** _attempt), 60.0) + random.uniform(0, 2)

                print(f"[Mistral 429] Retry-After={retry_after or 'none'} → waiting {wait:.1f}s (attempt {_attempt+1}/5)")
                # Yield a subtle waiting indicator so the stream stays alive
                yield f"\n_(rate limited, retrying in {wait:.0f}s…)_\n"
                time.sleep(wait)
                _rate_wait()
                continue

            # Non-429: don't retry
            if status and 400 <= status < 500:
                print(f"[Mistral stream error] HTTP {status}: {e}")
                yield f"[Stream error: HTTP {status}]"; return
            print(f"[Mistral stream error] {e}")
            yield f"[Stream error: {e}]"; return

# ── MISTRAL BLOCKING GENERATE ─────────────────────────────────────────────────
def mistral_generate(msgs: list, max_tokens: int = 2000, model: str = None, skill: str = None) -> str:
    return "".join(mistral_stream(msgs, max_tokens=max_tokens, model=model, skill=skill))

# ── LEGACY ALIASES ────────────────────────────────────────────────────────────
def groq_generate(msgs, max_tokens=2000, model=None, skill=None) -> str:
    return mistral_generate(msgs, max_tokens=max_tokens, skill=skill)

def groq_stream(msgs, max_tokens=2000, model=None, skill=None):
    yield from mistral_stream(msgs, max_tokens=max_tokens, skill=skill)

def nvidia_generate(msgs, max_tokens=2000, model=None) -> str:
    return mistral_generate(msgs, max_tokens=max_tokens)

def nvidia_stream(msgs, max_tokens=2000, model=None):
    yield from mistral_stream(msgs, max_tokens=max_tokens)

# ── VISION ────────────────────────────────────────────────────────────────────
def vision_describe(image_b64: str, prompt: str = "Describe this image in detail.") -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        {"type": "text", "text": prompt}
    ]}]
    try:
        _r = _session.post(
            "https://api.mistral.ai/v1/chat/completions",
            json={"model": "mistral-large-latest", "messages": msgs, "max_tokens": 1000},
            timeout=30,
        ) if _USE_SESSION else None
        if _r:
            return _r.json()["choices"][0]["message"]["content"]
        import requests as _req
        _r2 = _req.post("https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={"model": "mistral-large-latest", "messages": msgs, "max_tokens": 1000}, timeout=30)
        return _r2.json()["choices"][0]["message"]["content"]
    except Exception as _e:
        return f"Vision error: {_e}"

# ── MODEL ROUTER ──────────────────────────────────────────────────────────────
def route_model_v3(skill: str, complexity: str) -> tuple:
    try:
        from modules.reliability import route_model_v3 as _r
        return _r(skill, complexity)
    except Exception:
        return ("mistral", MISTRAL_MODEL)
_truncate_msgs = _trim_msgs
