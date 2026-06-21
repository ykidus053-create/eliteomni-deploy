import re, json, os, time, hashlib

_PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
]

def mask_pii(text):
    found = []
    for pattern, label in _PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(label, text)
    return text, found

def mask_msgs(msgs):
    masked, all_found = [], []
    for m in msgs:
        if m.get("role") == "user":
            content, found = mask_pii(str(m.get("content", "")))
            all_found.extend(found)
            masked.append({**m, "content": content})
        else:
            masked.append(m)
    return masked, all_found

_INJECTION_SIGNALS = [
    "ignore previous instructions","ignore your system prompt",
    "disregard all prior","you are now","new persona",
    "pretend you are","forget everything","override your","jailbreak","dan mode",
]

def detect_injection(text):
    t = text.lower()
    return any(sig in t for sig in _INJECTION_SIGNALS)

def check_input(msgs):
    masked, pii_found = mask_msgs(msgs)
    injection = any(detect_injection(str(m.get("content",""))) for m in msgs if m.get("role")=="user")
    return {"safe": not injection, "pii_found": pii_found, "injection": injection, "masked_msgs": masked}

_TOXIC_RE = re.compile(r"\b(kill|murder|bomb|terrorist|suicide|hate|slur|racist)\b", re.IGNORECASE)

def score_output(response, context=""):
    issues = []
    toxicity = 0.0
    if _TOXIC_RE.search(response):
        issues.append("toxic_language")
        toxicity = 0.9
    if len(response) < 5:
        issues.append("too_short")
    return {"safe": toxicity < 0.5 and "toxic_language" not in issues, "toxicity": toxicity, "issues": issues}

_CACHE = {}
_CACHE_FILE = os.path.expanduser("~/eliteomni_semantic_cache.json")

def _load_cache():
    global _CACHE
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE) as f:
                _CACHE = json.load(f)
    except Exception:
        _CACHE = {}

def _save_cache():
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(_CACHE, f)
    except Exception:
        pass

_load_cache()

def _cache_key(text):
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

def cache_get(query):
    entry = _CACHE.get(_cache_key(query))
    if entry:
        print(f"[cache] HIT: {query[:60]}")
        return entry["response"]
    return None

def cache_set(query, response):
    _CACHE[_cache_key(query)] = {"response": response, "ts": time.time(), "query": query[:100]}
    _save_cache()

_FEEDBACK_LOG = os.path.expanduser("~/eliteomni_feedback.jsonl")

def log_feedback(event, query="", response="", meta=None):
    try:
        import datetime
        record = {"ts": datetime.datetime.utcnow().isoformat(), "event": event,
                  "query": query[:200], "response": response[:200], **(meta or {})}
        with open(_FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[feedback] {e}")

def gateway(msgs, skill="general", max_tokens=2000, use_cache=True, n_best=1, critique=False):
    from modules.core.http_client import mistral_generate, mistral_generate_best_of
    input_check = check_input(msgs)
    if not input_check["safe"]:
        log_feedback("injection_blocked", query=str(msgs[-1].get("content",""))[:200])
        return {"response": "Request blocked: injection detected.", "blocked": True, "reason": "injection"}
    safe_msgs = input_check["masked_msgs"]
    if input_check["pii_found"]:
        print(f"[gateway] PII masked: {input_check['pii_found']}")
    query = str(safe_msgs[-1].get("content","")) if safe_msgs else ""
    if use_cache:
        cached = cache_get(query)
        if cached:
            return {"response": cached, "cached": True, "blocked": False}
    if n_best > 1 or critique:
        response = mistral_generate_best_of(safe_msgs, max_tokens=max_tokens, skill=skill, n=n_best, critique=critique)
    else:
        response = mistral_generate(safe_msgs, max_tokens=max_tokens, skill=skill)
    out_check = score_output(response)
    if not out_check["safe"]:
        log_feedback("output_blocked", query=query, response=response, meta={"issues": out_check["issues"]})
        return {"response": "Could not generate a safe response.", "blocked": True, "reason": out_check["issues"]}
    if use_cache and len(response) > 10:
        cache_set(query, response)
    return {"response": response, "blocked": False, "cached": False}
