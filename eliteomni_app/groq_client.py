# AUTO-SPLIT FROM app.py lines 23-340
import os, re, time, math, json, asyncio, random, ast, subprocess, sys, tempfile
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import urllib.request, urllib.parse

# Single API key — key rotation is prohibited by Groq ToS
import threading as _threading

# Load .env file
import pathlib
_env_path = pathlib.Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("[Groq] WARNING: GROQ_API_KEY not set")
else:
    print("[Groq] API key loaded")

def _get_next_key() -> str:
    return GROQ_API_KEY
GROQ_MODEL = "llama-3.3-70b-versatile"  # switched from compound
GROQ_CRITIC_MODEL    = "llama-3.3-70b-versatile"
GROQ_MODEL_CODE      = "llama-3.3-70b-versatile"
GROQ_MODEL_VISION    = "meta-llama/llama-4-scout-17b-16e-instruct"
FEEDBACK_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_store.json")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

def _truncate_msgs(msgs: list, max_chars: int = 2000) -> list:
    out = []; total = 0
    for m in reversed(msgs):
        c = m.get("content",""); l = len(c)
        if total + l > max_chars:
            remaining = max_chars - total
            if remaining > 100: out.insert(0, {"role": m["role"], "content": c[:remaining]})
            break
        out.insert(0, m); total += l
    return out if out else [msgs[-1]]

def _inject_images(msgs: list) -> list:
    """Auto-detect base64 images in messages and convert to vision format."""
    import re
    out = []
    for m in msgs:
        content = m.get("content", "")
        if isinstance(content, str):
            imgs = re.findall(r'\[IMAGE:[^\|]+\|base64:([A-Za-z0-9+/=]+)', content)
            if imgs:
                parts = [{"type": "text", "text": content}]
                for b64 in imgs[:3]:
                    parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                out.append({**m, "content": parts})
                continue
        out.append(m)
    return out

# -- AUDIT LOG (Feature 40) ---------------------------------------------------
_AUDIT_LOG_PATH = os.path.expanduser("~/eliteomni_audit.jsonl")
def _audit(event: str, data: dict):
    try:
        import json as _aj, datetime as _dt
        record = {"ts": _dt.datetime.utcnow().isoformat(), "event": event, **data}
        with open(_AUDIT_LOG_PATH, "a") as _af:
            _af.write(_aj.dumps(record) + "\n")
    except Exception:
        pass

# -- REASONING MODELS (Features 6 & 7) ----------------------------------------
_REASONING_EFFORT_MODELS = {"llama-3.3-70b-versatile", "llama-3.3-70b-versatile"}
_INCLUDE_REASONING_MODELS = {"llama-3.3-70b-versatile", "llama-3.3-70b-versatile"}

# -- RATE LIMIT HANDLING -----------------------------------------------------
# Retry queue removed — it violated Groq ToS by evading rate limits.
# 429 errors are handled inline with a single respectful backoff.
import threading as _rth

_prompt_cache = {}
_prompt_cache_hits = 0


def _trim_msgs(msgs: list, max_chars: int = 3000) -> list:
    """413 fix: trim + summarize dropped turns like Claude does."""
    if not msgs:
        return msgs
    system  = [m for m in msgs if m.get("role") == "system"]
    others  = [m for m in msgs if m.get("role") != "system"]
    # Cap system prompt at 4000 chars
    sys_trimmed = []
    for m in system:
        c = m.get("content", "")
        sys_trimmed.append({**m, "content": c[:500] + "\n[truncated]" if len(c) > 500 else c})
    # Cap each message at 3000 chars
    oth_trimmed = []
    for m in others:
        c = m.get("content", "")
        oth_trimmed.append({**m, "content": c[:500] + "...[trimmed]" if len(c) > 500 else c})
    # Walk newest-first, keep what fits
    sys_chars = sum(len(m.get("content","")) for m in sys_trimmed)
    budget = max_chars - sys_chars
    kept, dropped = [], []
    for m in reversed(oth_trimmed):
        chars = len(m.get("content",""))
        if budget - chars < 500:
            dropped.insert(0, m)
        else:
            kept.insert(0, m)
            budget -= chars
    # Always keep last user message
    if not kept:
        for m in reversed(oth_trimmed):
            if m.get("role") == "user":
                kept = [m]; break
    # Summarize dropped turns into digest (Claude-style compaction)
    if dropped:
        pairs = []
        for i in range(0, len(dropped) - 1, 2):
            q = dropped[i].get("content","")[:60].replace("\n"," ")
            a = dropped[i+1].get("content","")[:80].replace("\n"," ") if i+1 < len(dropped) else ""
            pairs.append(f"Q: {q} -> A: {a}")
        digest = "Earlier conversation summary: " + " | ".join(pairs[:6])
        sys_trimmed[0]["content"] += "\n" + digest if sys_trimmed else sys_trimmed.append({"role":"system","content":digest})
    result = sys_trimmed + kept
    if len(result) < len(msgs):
        print(f"[Trim] {len(msgs)}->{len(result)} msgs trimmed")
    return result


# Groq built-in tools — used automatically by groq/compound model
# Groq compound built-in tools — correct format per Groq docs
# compound/compound-mini auto-invoke these; just declare type, no function schema needed
GROQ_BUILTIN_TOOLS = [
    {"type": "web_search"},
    {"type": "code_interpreter"},
]


def _claude_style_verify(response: str, user_msg: str) -> tuple:
    """
    Hard verification gate — runs AFTER generation, BEFORE output.
    Catches what prompt instructions miss.
    Returns (verified_response, was_modified)
    """
    import re
    issues = []
    msg_lower = user_msg.lower()

    # ── 1. Character-level constraint check ──────────────────────────────
    forbidden_letters = re.findall(r"not (?:contain|use|include) the letter ['\"]?([a-z])['\"]?", msg_lower)
    for letter in forbidden_letters:
        words_with_letter = [w for w in response.split() if letter in w.lower().strip(".,!?;:\"'")]
        if words_with_letter:
            issues.append(f"CONSTRAINT FAIL: letter '{letter}' found in: {words_with_letter[:3]}")

    # ── 2. Word count check ──────────────────────────────────────────────
    exact_counts = re.findall(r"exactly (\d+) words?", msg_lower)
    if exact_counts:
        sentences = [s.strip() for s in re.split(r"[.!?]", response) if s.strip()]
        for i, target in enumerate(exact_counts[:len(sentences)]):
            actual = len(sentences[i].split())
            if actual != int(target):
                issues.append(f"WORD COUNT FAIL: sentence {i+1} has {actual} words, expected {target}")

    # ── 3. Forbidden words check ─────────────────────────────────────────
    forbidden_words = re.findall(r"(?:don't|do not|never) (?:use|say|include) (?:the )?word[s]? ['\"]+([a-z]+)['\"]+", msg_lower)
    for word in forbidden_words:
        if word in response.lower():
            issues.append(f"FORBIDDEN WORD: '{word}' appears in response")

    # ── 4. Overconfidence language check ────────────────────────────────
    overconfident = ["perfect", "absolutely", "guaranteed", "zero errors", "flawlessly", "undeniably"]
    found_overconf = [w for w in overconfident if w in response.lower()]
    if found_overconf:
        for w in found_overconf:
            response = re.sub(w, {"perfect": "solid", "absolutely": "likely",
                                   "guaranteed": "expected", "zero errors": "minimal errors",
                                   "flawlessly": "effectively", "undeniably": "arguably"}.get(w, w),
                              response, flags=re.IGNORECASE)

    if issues:
        audit_note = "\n\n> ⚠️ *Self-audit flagged potential issues — treat with care: " + "; ".join(issues) + "*"
        return response + audit_note, True

    return response, False


def _extract_constraints(user_msg: str) -> dict:
    """Parse hard constraints from user message."""
    import re
    c = {"forbidden_letters": [], "forbidden_words": [], "word_counts": [], "sentence_count": None}
    m = user_msg.lower()
    c["forbidden_letters"] = re.findall(r'not (?:contain|use|include) the letter ["\']?([a-z])["\']?', m)
    c["forbidden_words"]   = re.findall(r'(?:no|never|without) (?:the )?word ["\']?([a-z]+)["\']?', m)
    c["word_counts"]       = [int(x) for x in re.findall(r'exactly (\d+) words?', m)]
    sc = re.search(r'exactly (\d+) sentences?', m)
    if sc: c["sentence_count"] = int(sc.group(1))
    return c


def _syntactic_constraint_check(response: str, user_msg: str) -> list:
    """
    Catches constraints the semantic verifier misses.
    Treats ALL constraints as syntactic — literal character/word scanning.
    """
    import re
    violations = []
    m = user_msg.lower()

    # No notation constraints
    if "no mathematical notation" in m or "no notation" in m or "no symbols" in m:
        symbol_pattern = re.compile(r"[|⟩⟨∑∫∂∇×·°±√∞≈≠≤≥αβγδεζηθλμνξπρστφψω]|\\[a-zA-Z]+")
        found = symbol_pattern.findall(response)
        if found:
            violations.append(f"notation constraint violated: found {found[:3]}")

    # No jargon constraints
    if "no jargon" in m or "plain language" in m or "simple language" in m:
        jargon_markers = re.findall(r"\b[a-z]+(?:tion|ization|ology|istic|ative|atory)\b", response.lower())
        if len(jargon_markers) > 5:
            violations.append(f"possible jargon leak: {jargon_markers[:3]}")

    # Forbidden letter constraints (syntactic — every character)
    forbidden = re.findall(r"(?:no|without|not use|not contain) the letter ["\']?([a-z])["\']?", m)
    for letter in forbidden:
        words = [w.strip(".,!?;:\"\' ") for w in response.split()]
        bad = [w for w in words if letter in w.lower()]
        if bad:
            violations.append(f"letter \'{letter}\' in words: {bad[:3]}")

    return violations

def _hard_check(response: str, constraints: dict) -> list:
    """Returns list of violation strings. Empty = pass."""
    import re
    violations = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response.strip()) if s.strip()]

    for letter in constraints["forbidden_letters"]:
        bad = [w for w in response.split() if letter in w.lower().strip(".,!?;:\\"'")]
        if bad:
            violations.append(f"letter \'{letter}\' in: {bad[:3]}")

    for word in constraints["forbidden_words"]:
        if re.search(r"\b" + word + r"\b", response, re.IGNORECASE):
            violations.append(f"forbidden word \'{word}\' present")

    for i, target in enumerate(constraints["word_counts"]):
        if i < len(sentences):
            actual = len(sentences[i].split())
            if actual != target:
                violations.append(f"sentence {i+1}: {actual} words, need {target}")

    if constraints["sentence_count"] and len(sentences) != constraints["sentence_count"]:
        violations.append(f"got {len(sentences)} sentences, need {constraints['sentence_count']}")

    return violations


def _trim_over_completeness(text: str) -> str:
    """
    Catches Claude-unlike over-completion patterns:
    - Removes redundant closing summaries
    - Softens absolute closure language
    """
    import re
    # Remove redundant summary openers
    patterns = [
        r"In summary,? (therefore,? )?",
        r"To summarize,? ",
        r"In conclusion,? (then,? )?",
        r"Ultimately,? then,? ",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Soften over-confident closures
    replacements = [
        ("This is definitively", "This is likely"),
        ("The answer is clearly", "One way to see this:"),
        ("There is no doubt", "It seems likely"),
        ("It is certain that", "It appears that"),
        ("This proves that", "This suggests that"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    return text.strip()

def _verification_gate(msgs: list, response: str, generate_fn) -> str:
    """
    Claude-style verification gate.
    If constraints fail: re-prompts with explicit violation feedback.
    Max 3 attempts before flagging to user.
    """
    user_msg = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
    constraints = _extract_constraints(user_msg)

    has_constraints = any([
        constraints["forbidden_letters"],
        constraints["forbidden_words"],
        constraints["word_counts"],
        constraints["sentence_count"]
    ])

    if not has_constraints:
        return response  # no constraints — pass through immediately

    for attempt in range(3):
        violations = _hard_check(response, constraints) + _syntactic_constraint_check(response, user_msg)
        if not violations:
            return response  # passed

        # Build correction prompt
        violation_str = "; ".join(violations)
        correction_msgs = msgs + [
            {"role": "assistant", "content": response},
            {"role": "user", "content":
                f"Your response failed these constraints: {violation_str}. "
                f"Please rewrite it from scratch, checking each word character by character "
                f"before including it. Do not explain — just give the corrected response."}
        ]
        response = generate_fn(correction_msgs)

    # After 3 attempts, flag remaining violations
    final_violations = _hard_check(response, constraints) + _syntactic_constraint_check(response, user_msg)
    if final_violations:
        response += f"\n\n> ⚠️ *Constraint check: {'; '.join(final_violations)} — please verify manually.*"

    return response


def _dynamic_max_tokens(msgs: list) -> int:
    """
    Decides token budget dynamically based on request complexity.
    Never cuts off mid-response again.
    """
    user_msg = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
    msg_len = len(user_msg)
    m = user_msg.lower()

    # ── Signal detection ─────────────────────────────────────────────────
    is_code        = any(k in m for k in ["code", "function", "script", "implement", "write a program", "class ", "def ", "algorithm"])
    is_long_form   = any(k in m for k in ["essay", "report", "explain in detail", "comprehensive", "full", "complete guide", "step by step", "walkthrough"])
    is_list        = any(k in m for k in ["list", "enumerate", "all the", "every ", "compare", "pros and cons", "table"])
    is_multi_part  = m.count("?") >= 2 or any(k in m for k in ["and also", "also tell me", "additionally", "furthermore", "multiple", "several"])
    is_creative    = any(k in m for k in ["story", "poem", "write a ", "narrative", "dialogue", "screenplay", "chapter"])
    is_short       = any(k in m for k in ["briefly", "in one sentence", "tldr", "quick", "just tell me", "yes or no", "one word"])
    is_math        = any(k in m for k in ["calculate", "solve", "equation", "proof", "derive", "integral", "matrix"])
    is_analysis    = any(k in m for k in ["analyze", "analyse", "breakdown", "deep dive", "thoroughly", "in depth", "architecture"])

    # ── Base allocation ──────────────────────────────────────────────────
    if is_short:
        base = 256
    elif msg_len < 80:
        base = 512
    elif msg_len < 200:
        base = 1024
    else:
        base = 1500

    # ── Multipliers ──────────────────────────────────────────────────────
    multiplier = 1.0
    if is_code:        multiplier += 0.8
    if is_long_form:   multiplier += 1.0
    if is_list:        multiplier += 0.4
    if is_multi_part:  multiplier += 0.5
    if is_creative:    multiplier += 0.7
    if is_math:        multiplier += 0.4
    if is_analysis:    multiplier += 0.6

    # ── Conversation depth bonus ─────────────────────────────────────────
    non_system = [m for m in msgs if m.get("role") != "system"]
    if len(non_system) > 6:
        multiplier += 0.3  # deep conversation = longer context needed

    budget = int(base * multiplier)

    # ── Hard clamp: never below 512, never above 8000 ────────────────────
    return max(512, min(budget, 8000))


def _groq_thinking_effort(complexity: str) -> str:
    """Map complexity to reasoning effort level."""
    return {"easy": "low", "medium": "default", "hard": "high"}.get(complexity, "default")


def _inject_date_to_system(msgs):
    import datetime
    now = datetime.datetime.now()
    date_str = now.strftime("%A, %B %d, %Y %H:%M")
    date_line = f"[LIVE CLOCK] Right now it is: {date_str}. Act on this as ground truth."
    result = []
    for m in msgs:
        if m.get("role") == "system":
            m = dict(m)
            content = m.get("content", "")
            # Remove old date line if present, replace with fresh one
            import re
            content = re.sub(r"\[LIVE CLOCK\].*?\n", "", content)
            m["content"] = date_line + "\n" + content
        result.append(m)
    return result

def groq_generate(msgs: list, max_tokens: int = 0, model: str = None) -> str:
    if max_tokens == 0: max_tokens = _dynamic_max_tokens(msgs)
    msgs = _inject_date_to_system(msgs)
    """Non-streaming Groq call. Works for all models including compound."""
    if not GROQ_API_KEY:
        return "[GROQ_API_KEY not set]"
    mdl = model or GROQ_MODEL
    import urllib.request, json as _json

    msgs_trimmed = _trim_msgs(msgs, max_chars=5000)
    is_compound = False  # no compound models in use
    _skip_reason = max_tokens <= 512
    payload = {
        "model": mdl,
        "messages": msgs_trimmed,
        "max_completion_tokens": min(max_tokens, 4000),
        "stream": False,
    }
    if _skip_reason:
        payload["disable_reasoning"] = True
    if not is_compound:
        payload["temperature"] = 0.6
        payload["top_p"] = 0.95

    # reasoning_effort: low=easy, medium=medium, high=hard
    # NOTE: groq/compound is a system — it auto-invokes tools internally.
    # Do NOT pass tools/tool_choice manually; that causes 404 on compound endpoints.
    GPT_OSS_MODELS = {"llama-3.1-8b-instant"}
    if mdl in GPT_OSS_MODELS:
        effort = getattr(groq_generate, "_reasoning_effort", "low")
        if effort in ("low", "medium", "high"):
            payload["reasoning_effort"] = effort

    if max_tokens <= 400:
        payload["disable_reasoning"] = True
    data = _json.dumps(payload).encode()
    req  = urllib.request.Request(
        GROQ_URL, data=data,
        headers={"Authorization": f"Bearer {_get_next_key()}", "Content-Type": "application/json", "User-Agent": "EliteOmni/1.0", "Accept": "application/json", "Content-Length": str(len(data))}
    )
    try:
        _t0 = time.time()
        with urllib.request.urlopen(req, timeout=45 if max_tokens <= 400 else 120) as r:
            _raw = r.read()
            _region = r.headers.get("x-groq-region", "unknown")
        resp = _json.loads(_raw)
        _ttft_ms = round((time.time() - _t0) * 1000)
        _usage   = resp.get("usage", {})
        _cached  = _usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        _ptokens = _usage.get("prompt_tokens", 1)
        _cache_pct = round(_cached / _ptokens * 100)
        if _cached:
            print(f"[Groq] TTFT={_ttft_ms}ms region={_region} cache={_cache_pct}% ({_cached}/{_ptokens} tokens cached)")
        else:
            print(f"[Groq] TTFT={_ttft_ms}ms region={_region} no cache hit")
        _audit("groq_call", {"ttft_ms": _ttft_ms, "region": _region,
               "cached_tokens": _cached, "cache_pct": _cache_pct,
               "total_tokens": _usage.get("total_tokens", 0)})
        msg = resp["choices"][0]["message"]
        _raw = (msg.get("content") or "").strip()
        _user_msg = next((m.get('content','') for m in reversed(msgs) if m.get('role')=='user'), '')
        _verified, _changed = _claude_style_verify(_raw, _user_msg)
        if _changed: msg = dict(msg); msg['content'] = _verified
        # content is the answer; reasoning is the thinking (hidden from user)
        result = (msg.get("content") or "").strip()
        if not result:
            result = (msg.get("reasoning") or "").strip()
        return result
    except Exception as e:
        try:
            body = e.read().decode()[:500]
        except: body = str(e)

        if "429" in str(e) or "429" in body:
            import time as _t
            wait = 8
            try:
                hdrs = getattr(e, "headers", {}) or {}
                wait = int(hdrs.get("retry-after") or hdrs.get("Retry-After") or 8)
            except: pass
            print(f"[Groq] 429 rate limit — waiting {wait}s")
            _t.sleep(wait)
        if "413" in str(e) or "413" in body:
            # Payload too large — retry with aggressively trimmed context
            print(f"[Groq] 413 payload too large — retrying with 800 char trim")
            try:
                trimmed = _trim_msgs(msgs, max_chars=800)
                payload2 = _json.dumps({**{k:v for k,v in payload.items() if k!="messages"},
                                        "messages": trimmed}).encode()
                req2 = urllib.request.Request(GROQ_URL, data=payload2,
                    headers={"Authorization": f"Bearer {_get_next_key()}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req2, timeout=120) as r2:
                    resp2 = _json.loads(r2.read())
                msg2 = resp2["choices"][0]["message"]
                return (msg2.get("content") or msg2.get("reasoning") or "").strip()
            except Exception as e2:
                print(f"[Groq] 413 retry also failed: {e2}")
        print(f"[Groq error] {e} | {body}")
        return f"[Groq error: {e}]"

def groq_stream(msgs: list, max_tokens: int = 0, model: str = None):
    if max_tokens == 0: max_tokens = _dynamic_max_tokens(msgs)
    msgs = _inject_date_to_system(msgs)
    """
    Streaming Groq call.
    - compound models: falls back to non-streaming (they dont stream content reliably)
    - reasoning models (gpt-oss): streams with reasoning_format=hidden
    - standard models: normal streaming
    """
    if not GROQ_API_KEY:
        yield "[GROQ_API_KEY not set]"; return

    mdl = model or GROQ_MODEL
    import urllib.request, json as _json

    # compound models MUST use non-streaming — they return reasoning not content in stream
    # compound model: falls back to non-streaming (returns full result then simulates stream)
    COMPOUND_MODELS = set()  # no compound models — all use streaming path
    if mdl in COMPOUND_MODELS:
        result = groq_generate(msgs, max_tokens=max_tokens, model=mdl)
        if result:
            words = result.split(' ')
            buf   = ''
            for i, w in enumerate(words):
                buf += w + ' '
                if len(buf) >= 8 or i == len(words) - 1:
                    yield buf; buf = ''
        return

    # Standard streaming for all other models
    msgs_trimmed = _trim_msgs(msgs, max_chars=2000)
    _skip_reason = max_tokens <= 512
    payload = {
        "model":       mdl,
        "messages":    msgs_trimmed,
        "max_completion_tokens": min(max_tokens, 4000),
        "temperature": 0.7,
        "stream":      True,
    }
    if _skip_reason:
        payload["disable_reasoning"] = True

    # reasoning_effort only supported on groq/compound, not on 70b/8b
    if mdl in ("never-match-x",):
        effort = getattr(groq_stream, "_reasoning_effort", "low")
        if effort in ("low", "medium", "high"):
            payload["reasoning_effort"] = effort
    data = _json.dumps(payload).encode()
    req  = urllib.request.Request(
        GROQ_URL, data=data,
        headers={"Authorization": f"Bearer {_get_next_key()}", "Content-Type": "application/json", "User-Agent": "EliteOmni/1.0", "Accept": "application/json", "Content-Length": str(len(data))}
    )


    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line == "data: [DONE]" or line.strip() == "":
                    continue
                if not line.startswith("data: "):
                    continue
                try:
                    chunk = _json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    channel = delta.get("channel", "")
                    # skip reasoning/analysis channel, only yield actual content
                    if token and channel != "analysis":
                        yield token
                except Exception:
                    continue
    except Exception as e:
        try: body = e.read().decode()[:500]
        except: body = str(e)
        if "429" in str(e) or "rate_limit" in body:
            import time as _t
            wait = 8
            try: wait = int(e.headers.get("retry-after", 8))
            except: pass
            print(f"[Groq stream] 429 — waiting {wait}s then retrying")
            _t.sleep(wait)
        else:
            print(f"[Groq stream error] {e} | body: {body}")
        yield f"[Stream error — retrying, please resend]"
        return


def vision_describe(image_b64: str, prompt: str = "Describe this image in detail.") -> str:
    """Vision via Groq llama-4-scout — fast, no local GPU needed."""
    if not GROQ_API_KEY:
        return "[Vision requires GROQ_API_KEY]"
    try:
        import urllib.request, json as _json
        payload = _json.dumps({
            "model": GROQ_MODEL_VISION,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]}],
            "max_tokens": 1024,
            "temperature": 0.3,
        }).encode()
        req = urllib.request.Request(
            GROQ_URL, data=payload,
            headers={"Authorization": f"Bearer {_get_next_key()}", "Content-Type": "application/json", "User-Agent": "EliteOmni/1.0", "Accept": "application/json", "Content-Length": str(len(data))}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _json.loads(r.read())
        msg = data["choices"][0]["message"]
        return msg.get("content","").strip() or msg.get("reasoning","").strip()
    except Exception as e:
        return f"[Vision error: {e}]"

# Load vision model in background

try:
    import faiss, numpy as np
    _faiss_ok = True
except Exception as e:
    _faiss_ok = False
    np = None
    print(f"faiss failed: {e}")

# ── LOCAL CONFIG ──────────────────────────────────────────────────────────────







    if max_tokens == 0: max_tokens = _dynamic_max_tokens(msgs)
        return ""
    import urllib.request, json as _json
    payload = _json.dumps({
        "model": mdl, "messages": msgs,
        "max_tokens": min(max_tokens, 4000), "temperature": 0.6,
    }).encode()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = _json.loads(r.read())
        return (resp["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        return ""

    if max_tokens == 0: max_tokens = _dynamic_max_tokens(msgs)
        return
    import urllib.request, json as _json
    payload = _json.dumps({
        "model": mdl, "messages": msgs,
        "max_tokens": min(max_tokens, 4000), "temperature": 0.6, "stream": True,
    }).encode()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                line = line.decode()
                if not line.startswith("data:"):
                    continue
                data = line[5:]
                if data == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data)
                    tok = chunk["choices"][0]["delta"].get("content", "")
                    if tok:
                        yield tok
                except Exception:
                    continue
    except Exception as e:
        if "429" in str(e):
            import time as _t
            _t.sleep(10)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    for line in r:
                        line = line.decode()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = _json.loads(data)
                            tok = chunk["choices"][0]["delta"].get("content", "")
                            if tok:
                                yield tok
                        except Exception:
                            continue
                return
            except Exception as e2:
                return
