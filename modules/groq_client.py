
# ── NVIDIA NIM ────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
NVIDIA_URL     = "https://api.fireworks.ai/inference/v1/chat/completions"
NVIDIA_MODEL   = "accounts/fireworks/models/deepseek-v4-pro"

def nvidia_generate(msgs: list, max_tokens: int = 2000, model: str = None) -> str:
    if not NVIDIA_API_KEY:
        return "[NVIDIA_API_KEY not set]"
    mdl = model or NVIDIA_MODEL
    payload = json.dumps({"model": mdl, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.15}).encode()
    req = urllib.request.Request(NVIDIA_URL, data=payload,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt
                print(f"[NVIDIA 429] retry {attempt+1}/4 in {wait}s")
                time.sleep(wait)
            else:
                return f"[NVIDIA error: {e}]"
    return "[NVIDIA: gave up after 4 retries]"

def nvidia_stream(msgs: list, max_tokens: int = 4000, model: str = None):
    if not NVIDIA_API_KEY:
        yield "[NVIDIA_API_KEY not set]"; return
    mdl = model or NVIDIA_MODEL
    payload = json.dumps({"model": mdl, "messages": msgs, "max_tokens": max_tokens,
                          "temperature": 0.15, "stream": True}).encode()
    req = urllib.request.Request(NVIDIA_URL, data=payload,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                        if delta: yield delta
                    except Exception:
                        pass
    except Exception as e:
        yield f"[NVIDIA stream error: {e}]"
