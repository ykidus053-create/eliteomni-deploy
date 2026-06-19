from __future__ import annotations
import os, json, time, hashlib, threading, urllib.request
from typing import Iterator
from modules.model_router import select_model, record_outcome, get_token_budget, trim_system

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_URL     = "https://api.mistral.ai/v1/chat/completions"

_DEDUP_CACHE: dict = {}
_DEDUP_LOCK  = threading.Lock()
DEDUP_TTL    = 120

def _cache_key(msgs: list, max_tokens: int, model: str) -> str:
    raw = json.dumps(msgs, sort_keys=True) + str(max_tokens) + model
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

def _cache_get(key: str):
    with _DEDUP_LOCK:
        e = _DEDUP_CACHE.get(key)
        if e and time.time() - e["ts"] < DEDUP_TTL:
            return e["val"]
        if e:
            del _DEDUP_CACHE[key]
    return None

def _cache_set(key: str, val: str):
    with _DEDUP_LOCK:
        if len(_DEDUP_CACHE) > 200:
            oldest = min(_DEDUP_CACHE, key=lambda k: _DEDUP_CACHE[k]["ts"])
            del _DEDUP_CACHE[oldest]
        _DEDUP_CACHE[key] = {"val": val, "ts": time.time()}

def cache_stats() -> dict:
    with _DEDUP_LOCK:
        return {"entries": len(_DEDUP_CACHE), "ttl": DEDUP_TTL}

def _trim_msgs(msgs: list, max_chars: int = 8000) -> list:
    system = [m for m in msgs if m.get("role") == "system"]
    others = [m for m in msgs if m.get("role") != "system"]
    budget = max_chars - sum(len(m.get("content","")) for m in system)
    kept   = []
    for m in reversed(others):
        c = len(m.get("content",""))
        if budget - c < 200:
            break
        kept.insert(0, m)
        budget -= c
    if not kept and others:
        kept = [others[-1]]
    return system + kept

def _prep_msgs(msgs: list, complexity: str) -> list:
    out = []
    for m in _trim_msgs(msgs):
        if m.get("role") == "system":
            out.append({**m, "content": trim_system(m["content"], complexity)})
        else:
            out.append(m)
    return out

def _call(msgs, max_tokens, model, stream, timeout=90):
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set — add it to .env")
    payload = json.dumps({
        "model": model, "messages": msgs,
        "max_tokens": max_tokens, "temperature": 0.15, "stream": stream,
    }).encode()
    req = urllib.request.Request(MISTRAL_URL, data=payload, headers={
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    })
    return urllib.request.urlopen(req, timeout=timeout)

def generate(msgs: list, complexity: str = "medium",
             max_tokens: int = 0, model: str = None) -> str:
    mdl      = model or select_model(complexity)
    max_tok  = max_tokens or get_token_budget(complexity)
    trimmed  = _prep_msgs(msgs, complexity)
    key      = _cache_key(trimmed, max_tok, mdl)
    cached   = _cache_get(key)
    if cached:
        print(f"[LLM] Cache HIT {mdl}")
        return cached
    last_err = None
    for attempt in range(3):
        cur = mdl if attempt == 0 else select_model(complexity)
        try:
            t0   = time.time()
            resp = _call(trimmed, max_tok, cur, False, 30 + attempt * 30)
            body = json.loads(resp.read())
            res  = (body["choices"][0]["message"].get("content") or "").strip()
            print(f"[LLM] {cur} complexity={complexity} latency={round((time.time()-t0)*1000)}ms")
            record_outcome(cur, True)
            _cache_set(key, res)
            return res
        except Exception as e:
            s = str(e)
            record_outcome(cur, False)
            last_err = s
            if "429" in s:
                time.sleep(min(5*(attempt+1), 30))
            elif "401" in s:
                return "[Error: Invalid MISTRAL_API_KEY — check .env]"
            else:
                time.sleep(2**attempt)
    return f"[LLM Error: {last_err}]"

def stream(msgs: list, complexity: str = "medium",
           max_tokens: int = 0, model: str = None) -> Iterator[str]:
    mdl     = model or select_model(complexity)
    max_tok = max_tokens or get_token_budget(complexity)
    trimmed = _prep_msgs(msgs, complexity)
    last_err = None
    for attempt in range(3):
        cur = mdl if attempt == 0 else select_model(complexity)
        try:
            t0    = time.time()
            first = True
            resp  = _call(trimmed, max_tok, cur, True, 30 + attempt * 30)
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue
                try:
                    tok = json.loads(line[6:])["choices"][0].get("delta",{}).get("content","")
                    if tok:
                        if first:
                            print(f"[LLM] {cur} TTFT={round((time.time()-t0)*1000)}ms complexity={complexity}")
                            first = False
                        yield tok
                except Exception:
                    continue
            record_outcome(cur, True)
            return
        except Exception as e:
            s = str(e)
            record_outcome(cur, False)
            last_err = s
            if "429" in s:
                time.sleep(min(8*(attempt+1), 40))
                continue
            elif "401" in s:
                yield "[Error: Invalid MISTRAL_API_KEY — check .env]"
                return
            else:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                yield f"[Stream error: {last_err}]"
                return
    yield f"[Stream failed: {last_err}]"
