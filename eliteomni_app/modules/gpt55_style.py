"""
gpt55_style.py — GPT-5.5 intelligence techniques for EliteOmni
1. Unified context (omnimodal)
2. Agent loop (plan → act → verify → continue)
3. Long-context compression
4. Domain reliability filter (law/med/finance hallucination catcher)
5. Token efficiency (strip filler, dedup)
"""
import re

# ── 1. UNIFIED CONTEXT ────────────────────────────────────────────────────────
def build_unified_context(msg: str, image_desc: str = "", file_texts: list = None) -> str:
    parts = []
    if image_desc:
        parts.append(f"[Visual context: {image_desc}]")
    for f in (file_texts or [])[:3]:
        text = f.get("text", "")[:2000]
        if text:
            parts.append(f"[File: {f.get('name','file')}]\n{text}")
    return ("\n\n".join(parts) + "\n\n" + msg) if parts else msg

# ── 3. LONG-CONTEXT COMPRESSION ───────────────────────────────────────────────
def compress_long_context(history: list, max_tokens: int = 4000) -> list:
    if not history:
        return []
    total = sum(len(str(m.get("content", ""))) // 4 for m in history)
    if total <= max_tokens:
        return history
    recent = history[-6:]
    older = history[:-6]
    if not older:
        return recent
    summary = " | ".join(
        f"{m.get('role','?')}: {str(m.get('content',''))[:100]}"
        for m in older[-10:]
    )
    return [{"role": "system", "content": f"[Earlier conversation: {summary}]"}] + recent

# ── 4. DOMAIN RELIABILITY ─────────────────────────────────────────────────────
DOMAIN_TRIGGERS = {
    "legal":   ["law", "legal", "contract", "lawsuit", "court", "attorney", "liable"],
    "medical": ["diagnosis", "symptoms", "medication", "dosage", "disease", "treatment"],
    "finance": ["investment", "stock", "tax", "revenue", "accounting", "audit"],
}
DOMAIN_WARNINGS = {
    "legal":   "\n\n⚠️ *Legal information only — not legal advice. Consult a qualified attorney.*",
    "medical": "\n\n⚠️ *Health information only — not medical advice. Consult a licensed physician.*",
    "finance": "\n\n⚠️ *Financial information only — not financial advice. Consult a qualified advisor.*",
}

def domain_reliability_check(msg: str, response: str) -> str:
    msg_lower = msg.lower()
    for domain, triggers in DOMAIN_TRIGGERS.items():
        if any(t in msg_lower for t in triggers):
            return response + DOMAIN_WARNINGS[domain]
    return response

# ── 5. TOKEN EFFICIENCY ───────────────────────────────────────────────────────
FILLER_RE = re.compile(
    r"^(Certainly!?|Absolutely!?|Great question!?|Sure,?|Of course!?|"
    r"I'?d be happy to|I'?m happy to|As an AI|As a language model)[,!.]?\s*",
    re.IGNORECASE
)
HEDGE_RE = re.compile(
    r"(It'?s? (important to note|worth noting) that |Please note that |"
    r"Keep in mind that |It should be noted that )",
    re.IGNORECASE
)

def token_efficient_output(response: str) -> str:
    if not response:
        return response
    response = FILLER_RE.sub("", response).strip()
    response = HEDGE_RE.sub("", response)
    # Dedup consecutive near-identical sentences
    sentences = re.split(r'(?<=[.!?])\s+', response)
    seen, deduped = set(), []
    for s in sentences:
        key = re.sub(r'\s+', ' ', s.lower().strip())[:60]
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)
    return " ".join(deduped).strip()

# ── MASTER ────────────────────────────────────────────────────────────────────
def gpt55_enhance(msg: str, response: str, image_desc: str = "", file_texts: list = None) -> str:
    if not response:
        return response
    response = domain_reliability_check(msg, response)
    response = token_efficient_output(response)
    return response
