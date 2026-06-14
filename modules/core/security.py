import re, hashlib, hmac, os

_INJECTION_FAST = re.compile(
    r"ignore (previous|all|your|prior) instructions"
    r"|new (system|developer|operator) (prompt|instructions|mode)"
    r"|you are now|pretend you are|act as(?! EliteOmni)"
    r"|jailbreak|DAN mode|developer mode|unrestricted mode|god mode"
    r"|disregard your|forget your instructions"
    r"|override your (instructions|guidelines|safety)",
    re.IGNORECASE)

_INJECTION_SEM = [re.compile(p, re.IGNORECASE) for p in [
    r"when (asked|told|instructed).{0,50}(ignore|bypass|skip)",
    r"(before|after) (this|every) (message|response).{0,50}(do|say|output)",
    r"respond (only|always|never) (with|as|like)",
    r"(your|the) (true|real|actual) (purpose|goal|mission) is",
]]

_SECRET_RE = re.compile(
    r"(sk-[a-zA-Z0-9]{20,})"
    r"|(gsk_[a-zA-Z0-9]{40,})"
    r"|(Bearer [a-zA-Z0-9\-_.]{20,})"
    r"|(password[=:][^\s&]{4,})"
    r"|(secret[=:][^\s&]{4,})",
    re.IGNORECASE)

def scan_injection(text):
    if not text: return False, ""
    m = _INJECTION_FAST.search(text)
    if m: return True, "pattern: " + m.group()[:60]
    for pat in _INJECTION_SEM:
        m = pat.search(text)
        if m: return True, "semantic: " + m.group()[:60]
    return False, ""

def scrub_secrets(text):
    return _SECRET_RE.sub("[REDACTED]", text)

def sanitize_user_input(text, max_len=8000):
    original = text
    text = text[:max_len]
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("<|system|>", "[system]")
    text = text.replace("<|im_start|>", "[im_start]")
    text = text.replace("<|im_end|>", "[im_end]")
    text = text.replace("###SYSTEM", "[SYSTEM]")
    return text, text != original

def validate_tool_output(tool_name, output):
    if not output: return True, output
    is_inj, reason = scan_injection(output)
    if is_inj:
        print(f"[Security] Tool {tool_name} blocked: {reason}")
        return False, f"[Tool result blocked: {reason}]"
    return True, scrub_secrets(output)[:8000]

def rate_limit_key(ip, user_id=""):
    secret = os.environ.get("RATE_LIMIT_SECRET", "eliteomni-default")
    raw = f"{ip}:{user_id}"
    return hmac.new(secret.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]
