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
MISTRAL_MODEL       = "mistral-medium-3.5"  # reasoning default
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL          = "llama-3.3-70b-versatile"
GROQ_URL            = "https://api.groq.com/openai/v1/chat/completions"
GROQ_CRITIC_MODEL   = "llama-3.3-70b-versatile"

# ── CODESTRAL ROUTING ─────────────────────────────────────────────────────────
CODESTRAL_MODEL = "mistral-medium-3.5"
CODESTRAL_URL   = "https://api.mistral.ai/v1/chat/completions"

CODING_SKILLS = {"coder", "code", "coding", "swe", "calculator", "debug", "refactor", "engineer"}

def get_model_for_skill(skill: str, model: str = None) -> str:
    """Return Codestral for coding tasks, Mistral-large for everything else."""
    if model:
        return model
    if skill and skill.lower() in CODING_SKILLS:
        print(f"[Router] skill={skill} → mistral-medium-3.5 (code)")
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
def _trim_msgs(msgs: list, max_chars: int = 180000) -> list:
    system      = [m for m in msgs if m.get("role") == "system"]
    others      = [m for m in msgs if m.get("role") != "system"]
    sys_trimmed = [{**m, "content": m.get("content","")[:40000]} for m in system]
    oth_trimmed = [{**m, "content": m.get("content","")[:30000]} for m in others]
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
    # TCP optimizations — reduce TTFT and improve throughput
    adapter = _requests.adapters.HTTPAdapter(
        pool_connections=4,
        pool_maxsize=16,
        max_retries=0,
        pool_block=False,
    )
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)
    _session.headers.update({
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type":  "application/json",
        "Connection":    "keep-alive",
    })
    _USE_SESSION = True
    import socket as _socket
    _orig_gai = _socket.getaddrinfo
    def _ipv4_only(h,p,f=0,t=0,pr=0,fl=0): return _orig_gai(h,p,_socket.AF_INET,t,pr,fl)
    _socket.getaddrinfo = _ipv4_only
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
        gap = 12.0  # free tier ~5 RPM — 12s gap stays safe
        if elapsed < gap:
            time.sleep(gap - elapsed)
        _last_call_time = time.time()

def _rate_on_success():
    global _last_call_time
    with _rate_lock:
        _last_call_time = time.time()

# ── MISTRAL STREAM ────────────────────────────────────────────────────────────
def mistral_stream(msgs: list, max_tokens: int = 2000, model: str = None, skill: str = None, tools: list = None):
    if not MISTRAL_API_KEY:
        yield "[MISTRAL_API_KEY not set]"; return

    mdl      = get_model_for_skill(skill, model)
    trimmed  = _trim_msgs(msgs, max_chars=180000)

    # Route to correct endpoint based on model name
    if mdl.startswith("mistral-") or mdl.startswith("codestral-") or mdl.startswith("devstral-") or mdl.startswith("magistral-"):
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
        "temperature": 0.1 if (skill and skill.lower() in CODING_SKILLS) else 0.3,
        "stream":     True,
    }
    if mdl.startswith("mistral-medium") or mdl.startswith("mistral-large"):
        payload["reasoning_effort"] = "high" if (skill and skill.lower() in CODING_SKILLS) else "none"
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    print(f"[mistral_stream] tools sent: {len(tools) if tools else 0}")

    # ── helper: open one SSE stream and yield tokens ─────────────────────────
    # Pre-compile for speed — avoid re-instantiation per token
    _loads = json.loads
    _data_prefix = "data:"
    _data_len = 5

    _tool_call_acc = {}
    def _do_stream():
        """Returns a generator of tokens from one HTTP attempt. Raises on 429/error."""
        if _USE_SESSION:
            resp = _session.post(_url, json=payload, stream=True, timeout=(10, 180))
            if resp.status_code == 429:
                raise _requests.exceptions.HTTPError(response=resp)
            resp.raise_for_status()
            for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
                if not line or line[0] != "d": continue  # fast-path skip non-data lines
                if line[:5] != _data_prefix: continue
                d = line[_data_len:].strip()
                if d == "[DONE]": break
                try:
                    choice = _loads(d)["choices"][0]
                    delta = choice["delta"]
                    tcs = delta.get("tool_calls")
                    if tcs:
                        for tc in tcs:
                            idx = tc.get("index", 0)
                            fn  = tc.get("function", {})
                            entry = _tool_call_acc.setdefault(idx, {"name": "", "arguments": ""})
                            if fn.get("name"):
                                entry["name"] += fn["name"]
                            if fn.get("arguments"):
                                entry["arguments"] += fn["arguments"]
                    if choice.get("finish_reason") == "tool_calls" and _tool_call_acc:
                        import json as _j3
                        for tc in _tool_call_acc.values():
                            yield "\x00TOOLCALL\x00" + _j3.dumps(tc)
                        _tool_call_acc.clear()
                        continue
                    content = delta.get("content", "")
                    if not content: continue
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                tok = block.get("text", "")
                                if tok: yield tok
                    else:
                        yield content
                except Exception as _e:
                    print(f"[mistral stream parse err] {_e} | line={d[:300]}")
                    continue
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
            _rate_on_success()
            return  # clean exit

        except Exception as e:
            status = None
            if _USE_SESSION and hasattr(e, "response") and e.response is not None:
                status = e.response.status_code
            elif isinstance(e, urllib.error.HTTPError):
                status = e.code

            # Retry on timeout errors
            is_timeout = "timed out" in str(e).lower() or "timeout" in str(e).lower() or "read timeout" in str(e).lower()
            if is_timeout:
                wait = 2 ** _attempt
                print(f"[mistral_stream] timeout on attempt {_attempt+1}, retrying in {wait}s...")
                time.sleep(wait)
                continue

            if status == 429:
                if _attempt == 4:
                    _rate_on_429()
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

                _rate_on_429(retry_after or 0)
                print(f"[Mistral 429] Retry-After={retry_after or 'none'} → waiting {wait:.1f}s (attempt {_attempt+1}/5)")
                # Yield a subtle waiting indicator so the stream stays alive
                yield f"\n_(rate limited, retrying in {wait:.0f}s…)_\n"
                time.sleep(wait)
                _rate_wait()
                continue

            # Non-429: don't retry
            if status and 400 <= status < 500:
                _body = None
                try:
                    _resp = getattr(e, "response", None)
                    if _resp is not None:
                        _body = _resp.text[:1000]
                except Exception:
                    pass
                print(f"[Mistral stream error] HTTP {status}: {e} | BODY: {_body}")
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

# ── VISION — DeepMind/Hassabis-style pipeline ─────────────────────────────────
# Implements: spatial decomposition, chain-of-thought visual reasoning,
# world-model grounding, cross-modal attention simulation, self-consistency

def _vision_call(msgs: list, max_tokens: int = 800, model: str = "mistral-small-latest") -> str:
    """Raw vision API call — mistral-medium-latest."""
    import requests as _req
    try:
        _r = _session.post(
            "https://api.mistral.ai/v1/chat/completions",
            json={"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.3},
            timeout=30,
        ) if _USE_SESSION else None
        if _r and _r.status_code == 200:
            return _r.json()["choices"][0]["message"]["content"].strip()
        _r2 = _req.post("https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.3},
            timeout=30)
        return _r2.json()["choices"][0]["message"]["content"].strip()
    except Exception as _e:
        return f"[vision_call error: {_e}]"

def ocr_document(file_b64: str, filename: str = "document.pdf") -> str:
    """
    Extract text from a PDF/document using Mistral's dedicated OCR model.
    Returns markdown-formatted extracted text, or an error string.
    """
    import requests as _req
    try:
        is_pdf = filename.lower().endswith(".pdf")
        doc_payload = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{file_b64}"
        } if is_pdf else {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{file_b64}"
        }
        resp = _req.post(
            "https://api.mistral.ai/v1/ocr",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={"model": "mistral-ocr-latest", "document": doc_payload},
            timeout=45,
        )
        if resp.status_code != 200:
            return f"[OCR error: HTTP {resp.status_code}: {resp.text[:300]}]"
        data = resp.json()
        pages = data.get("pages", [])
        text = "\n\n".join(p.get("markdown", "") for p in pages)
        return text.strip() or "[OCR returned no text]"
    except Exception as _e:
        return f"[OCR error: {type(_e).__name__}: {_e}]"

def _img_content(image_b64: str) -> dict:
    """Build image content block — auto-detect JPEG vs PNG."""
    mime = "image/png" if image_b64.startswith("iVBOR") else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}

def vision_describe(image_b64: str, prompt: str = "") -> str:
    import concurrent.futures
    img = _img_content(image_b64)

    def _stage1():
        msgs = [{"role": "user", "content": [img, {"type": "text", "text":
            """Analyze this image using spatial decomposition. Answer each section briefly:
FOREGROUND: What objects/subjects are in the foreground?
BACKGROUND: What is in the background or environment?
LAYOUT: How are elements spatially arranged (left/right/center, near/far)?
TEXT: Any visible text, labels, numbers, or symbols?
COLORS: Dominant colors and lighting conditions?"""
        }]}]
        return _vision_call(msgs, max_tokens=400)

    def _stage2():
        msgs = [{"role": "user", "content": [img, {"type": "text", "text":
            """Apply world-model reasoning to this image:
MATERIALS: What are objects made of? (metal, wood, fabric, glass, etc.)
PHYSICS: What physical state/action is happening? (static, moving, falling, pouring, etc.)
SCALE: Approximate size relationships between objects?
CONTEXT: What setting or environment is this? (indoors, outdoors, digital, document, etc.)
PURPOSE: What is the likely function or purpose of what you see?"""
        }]}]
        return _vision_call(msgs, max_tokens=400)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_stage1)
        f2 = ex.submit(_stage2)
        spatial = f1.result()
        world   = f2.result()

    user_question = prompt or "What is shown in this image?"
    stage3_msgs = [{"role": "user", "content": [img, {"type": "text", "text":
        f"""Using the visual analysis below, answer this specific question using cross-modal reasoning.

QUESTION: {user_question}

Visual context:
- Spatial: {spatial[:250]}
- World model: {world[:250]}

Answer the question directly. Use only what is visually present."""
    }]}]
    cross_modal = _vision_call(stage3_msgs, max_tokens=500)

    stage4_msgs = [{"role": "user", "content": [img, {"type": "text", "text":
        f"""You are a vision verifier. A model produced this answer about the image:

ANSWER: {cross_modal}

Look at the image directly and verify this answer.
- Correct any hallucinations or unsupported claims
- Output only the final verified, concise description (2-4 sentences max)
- Do not add lists, bullet points, or explanations"""
    }]}]
    final = _vision_call(stage4_msgs, max_tokens=300)

    return final if not final.startswith("[vision_call error") else cross_modal

# ── MODEL ROUTER ──────────────────────────────────────────────────────────────
def route_model_v3(skill: str, complexity: str) -> tuple:
    try:
        from modules.reliability import route_model_v3 as _r
        return _r(skill, complexity)
    except Exception:
        return ("mistral", MISTRAL_MODEL)
_truncate_msgs = _trim_msgs

# ── VOXTRAL AUDIO PIPELINE (Hassabis-style multimodal grounding) ──────────────
import time as _time

def _rate_on_429(wait: float = 5.0):
    _time.sleep(max(float(wait), 1.0))
import requests as _req, base64 as _b64, mimetypes as _mime

def _audio_content(audio_b64: str, filename: str = "audio.wav") -> dict:
    mime = _mime.guess_type(filename)[0] or "audio/wav"
    return {"type": "input_audio", "input_audio": {"data": audio_b64, "format": mime.split("/")[-1]}}

def voxtral_transcribe(audio_b64: str, filename: str = "audio.webm") -> str:
    """Transcribe audio via Mistral /v1/audio/transcriptions endpoint."""
    import requests as _req, base64 as _b64
    try:
        audio_bytes = _b64.b64decode(audio_b64)
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "webm"
        mime = {"webm": "audio/webm", "wav": "audio/wav", "mp3": "audio/mpeg",
                "ogg": "audio/ogg", "m4a": "audio/mp4"}.get(ext, "audio/webm")
        resp = _req.post(
            "https://api.mistral.ai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            files={"file": (filename, audio_bytes, mime)},
            data={"model": "voxtral-mini-latest"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
        return f"[voxtral error {resp.status_code}: {resp.text[:200]}]"
    except Exception as e:
        return f"[voxtral_transcribe error: {e}]" 

def voxtral_ground(transcript: str, audio_b64: str, filename: str = "audio.wav") -> dict:
    """Stage 2 — World-model grounding: prosody, intent, affect, context."""
    msgs = [{"role": "user", "content": [
        _audio_content(audio_b64, filename),
        {"type": "text", "text": f"""You are a multimodal audio analyst. Given this transcript:
TRANSCRIPT: {transcript}

Analyze the audio signal itself (not just words):
TONE: Speaker's emotional tone? (neutral, urgent, confused, confident, etc.)
PACE: Speaking pace? (slow/normal/fast) — any pauses or emphasis?
INTENT: What does the speaker actually want or mean?
CONTEXT: What kind of audio is this? (conversation, command, question, narration)
LANGUAGE: Primary language and any code-switching?

Be concise. One line per field."""}
    ]}]
    raw = _vision_call(msgs, max_tokens=300, model="voxtral-small-latest")
    result = {"transcript": transcript, "grounding": raw}
    return result

def voxtral_understand(audio_b64: str, filename: str = "audio.wav", prompt: str = "") -> dict:
    """
    Hassabis-style audio pipeline:
    Stage 1 — Perceptual transcription (what was said)
    Stage 2 — World-model grounding (how + why it was said)
    Stage 3 — Cross-modal synthesis (unified understanding for downstream use)
    """
    # Stage 1
    transcript = voxtral_transcribe(audio_b64, filename)
    if transcript.startswith("[vision_call error"):
        return {"transcript": "", "grounding": "", "answer": transcript}

    # Stage 2
    grounded = voxtral_ground(transcript, audio_b64, filename)

    # Stage 3 — synthesize into actionable understanding
    question = prompt or "What is the speaker saying and what do they want?"
    msgs = [{"role": "user", "content": [
        _audio_content(audio_b64, filename),
        {"type": "text", "text": f"""Synthesize a final grounded understanding of this audio.

TRANSCRIPT: {transcript}
GROUNDING: {grounded['grounding'][:400]}
QUESTION: {question}

Output a single concise answer (2-3 sentences). Ground your answer in both the words AND the audio signal. No bullet points."""}
    ]}]
    answer = _vision_call(msgs, max_tokens=400, model="voxtral-small-latest")
    return {"transcript": transcript, "grounding": grounded["grounding"], "answer": answer}

def voxtral_file(path: str, prompt: str = "") -> dict:
    """Convenience: load audio file from disk and run the full pipeline."""
    with open(path, "rb") as f:
        data = _b64.b64encode(f.read()).decode()
    return voxtral_understand(data, filename=path.split("/")[-1], prompt=prompt)

# ── MISTRAL OCR ───────────────────────────────────────────────────────────────
def mistral_ocr(file_b64: str, filename: str = "document.pdf") -> str:
    """Extract text from a PDF or image using mistral-ocr-latest."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"
    mime = "application/pdf" if ext == "pdf" else f"image/{ext}"
    try:
        resp = _session.post(
            "https://api.mistral.ai/v1/ocr",
            json={
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": f"data:{mime};base64,{file_b64}"
                },
                "include_image_base64": False
            },
            timeout=60
        )
        resp.raise_for_status()
        pages = resp.json().get("pages", [])
        text = "\n\n".join(p.get("markdown", "") for p in pages).strip()
        print(f"[OCR] {filename} → {len(text)} chars across {len(pages)} pages")
        return text or "[OCR returned no text]"
    except Exception as e:
        print(f"[OCR error] {e}")
        return f"[OCR failed: {e}]"


def mistral_stream_traced(msgs: list, max_tokens: int = 2000, model: str = None,
                           skill: str = None, tools: list = None, label: str = "default"):
    """
    Thin tracing wrapper around mistral_stream. Logs prompt/response/latency
    to eliteomni.db via modules.langchain_tracing, without changing any
    existing generation behavior. Drop-in replacement: same signature + yields.
    """
    import time as _time_tr
    try:
        from modules.langchain_tracing import trace_call
    except Exception:
        trace_call = None

    _t0 = _time_tr.time()
    _prompt_preview = ""
    try:
        _last_user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        _prompt_preview = str(_last_user)[:500]
    except Exception:
        pass

    _chunks = []
    _error = None
    try:
        for tok in mistral_stream(msgs, max_tokens=max_tokens, model=model, skill=skill, tools=tools):
            _chunks.append(tok if isinstance(tok, str) else "")
            yield tok
    except Exception as _e:
        _error = str(_e)[:500]
        raise
    finally:
        if trace_call:
            _latency_ms = (_time_tr.time() - _t0) * 1000
            _response_text = "".join(_chunks)
            try:
                trace_call(
                    label=label, skill=skill or "", complexity="", model=model or "",
                    prompt_text=_prompt_preview, response_text=_response_text,
                    latency_ms=_latency_ms, error=_error
                )
            except Exception as _te:
                print(f"[LangChainTrace] wrapper log failed: {_te}")


def mistral_stream_traced(msgs: list, max_tokens: int = 2000, model: str = None,
                           skill: str = None, tools: list = None, label: str = "default"):
    """
    Thin tracing wrapper around mistral_stream. Logs prompt/response/latency
    to eliteomni.db via modules.langchain_tracing, without changing any
    existing generation behavior. Drop-in replacement: same signature + yields.
    """
    import time as _time_tr
    try:
        from modules.langchain_tracing import trace_call
    except Exception:
        trace_call = None

    _t0 = _time_tr.time()
    _prompt_preview = ""
    try:
        _last_user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        _prompt_preview = str(_last_user)[:500]
    except Exception:
        pass

    _chunks = []
    _error = None
    try:
        for tok in mistral_stream(msgs, max_tokens=max_tokens, model=model, skill=skill, tools=tools):
            _chunks.append(tok if isinstance(tok, str) else "")
            yield tok
    except Exception as _e:
        _error = str(_e)[:500]
        raise
    finally:
        if trace_call:
            _latency_ms = (_time_tr.time() - _t0) * 1000
            _response_text = "".join(_chunks)
            try:
                trace_call(
                    label=label, skill=skill or "", complexity="", model=model or "",
                    prompt_text=_prompt_preview, response_text=_response_text,
                    latency_ms=_latency_ms, error=_error
                )
            except Exception as _te:
                print(f"[LangChainTrace] wrapper log failed: {_te}")
