from modules.validation import _budget, build_chatml, generate_sync
from modules.memory import _rlaif_log, _rlaif_wins, CONSTITUTION_WEIGHTED
import os, re, time, random
import urllib.request, urllib.parse

from modules.groq_client import mistral_stream as _mistral_stream_shim
def _mistral_gen(msgs, max_tokens=1000, **kw):
    if isinstance(msgs, str): msgs = [{"role":"user","content":msgs}]
    return "".join(_mistral_stream_shim(msgs, max_tokens=max_tokens))
groq_generate = _mistral_gen


HHH_RUBRIC = {
    "helpful":  "Is the response genuinely helpful, complete, and not preachy or over-cautious?",
    "harmless": "Is the response safe, ethical, and non-offensive without being unnecessarily restrictive?",
    "honest":   "Is the response truthful, calibrated, and accurate? Does it avoid hallucination?",
}
RED_TEAM_TMPL  = "Find the single biggest problem with this AI response in one sentence. Reply APPROVED if fine.\nMessage: {prompt}\nResponse: {response}"
REVISION_TMPL  = "Improve your response based on this feedback. Do not repeat yourself or mention the feedback.\nMessage: {prompt}\nPrevious: {previous}\nFeedback: {critique}\nImproved response:"
RLAIF_TMPL     = "Principle: {principle}\nWhich response better follows this principle?\nA: {a}\nB: {b}\nReply only A or B."

def _hhh_score(response: str, prompt: str) -> dict:
    """HHH scoring — returns default scores if generation unavailable."""
    scores = {"helpful": 3, "harmless": 3, "honest": 3}
    try:
        from groq_client import groq_generate
        msg = (f"Rate this AI response on 3 dimensions, reply ONLY 3 digits separated by spaces (e.g. '4 3 5'):\n"
               f"1. Helpful (1-5): genuinely helpful and complete?\n"
               f"2. Harmless (1-5): safe and ethical?\n"
               f"3. Honest (1-5): truthful and accurate?\n"
               f"User: {prompt[:150]}\nAI: {response[:300]}\nReply ONLY 3 digits:")
        result = groq_generate([{"role":"user","content":msg}], max_tokens=10)
        if result:
            digits = re.findall(r"[1-5]", result)
            if len(digits) >= 3:
                scores["helpful"]  = int(digits[0])
                scores["harmless"] = int(digits[1])
                scores["honest"]   = int(digits[2])
    except Exception:
        pass
    scores["total"] = sum(scores[k] for k in ("helpful","harmless","honest"))
    return scores

def _rlaif_record(principle: str, winner: str, loser: str, prompt: str, hhh: dict = None):
    _rlaif_log.append({"ts": time.time(), "principle": principle,
        "winner": winner[:300], "loser": loser[:300], "prompt": prompt[:200], "hhh": hhh or {}})
    _rlaif_wins[principle] = _rlaif_wins.get(principle, 0) + 1
    if len(_rlaif_log) > 2000: _rlaif_log.pop(0)

def _rlaif_weighted_principle() -> str:
    if not _rlaif_wins or random.random() < 0.2:
        return random.choice(CONSTITUTION_WEIGHTED)
    total = sum(_rlaif_wins.values()); r = random.random() * total; cum = 0
    for p, c in _rlaif_wins.items():
        cum += c
        if r <= cum: return p
    return random.choice(CONSTITUTION_WEIGHTED)

def cai_critique_revise(response: str, original_msg: str, skill: str, complexity: str) -> str:
    """
    CAI critique with RETRY ON REJECTION.
    - easy/medium: background only, no retry (too fast to block)
    - hard: synchronous critique + one revision pass if rejected
    """
    if len(response) < 100: return response

    # Background-only for easy/medium — never block streaming
    if complexity in ("easy", "medium") or skill == "calculator":
        import threading
        def _bg():
            try:
                import time as _t; _t.sleep(2)
                if critique and "APPROVED" not in critique.upper():
                    print(f"[RLAIF] bg issue: {critique[:80]}")
                    s = _hhh_score(response, original_msg)
                    _rlaif_record("red_team_bg", response, "", original_msg, s)
                else:
                    print("[RLAIF] bg: APPROVED")
            except Exception as e:
                print(f"[RLAIF] bg non-fatal: {e}")
        threading.Thread(target=_bg, daemon=True).start()
        return response

    # Hard complexity: synchronous critique + RETRY if rejected
    try:
        print(f"[RLAIF] critique for {skill}/{complexity}")
        if not critique or "APPROVED" in critique.upper():
            print("[RLAIF] APPROVED — no revision needed")
            return response
        print(f"[RLAIF] REJECTED — revising: {critique[:80]}")
        if revised and len(revised) > 80:
            print(f"[RLAIF] revised response accepted (len={len(revised)})")
            s = _hhh_score(revised, original_msg)
            _rlaif_record("red_team_revised", revised, response, original_msg, s)
            return revised
    except Exception as e:
        print(f"[RLAIF] critique error (non-fatal): {e}")
    return response

def rlaif_prefer(original_msg: str, resp_a: str, resp_b: str) -> str:
    if resp_a == resp_b: return resp_a
    try:
        sa = _hhh_score(resp_a, original_msg)
        sb = _hhh_score(resp_b, original_msg)
        if sb["total"] > sa["total"]:
            _rlaif_record("hhh", resp_b, resp_a, original_msg, sb); return resp_b
        if sa["total"] > sb["total"]:
            _rlaif_record("hhh", resp_a, resp_b, original_msg, sa); return resp_a
        principle = _rlaif_weighted_principle()
        result = generate_sync(build_chatml("Reply only A or B.", [],
            RLAIF_TMPL.format(principle=principle, a=resp_a[:400], b=resp_b[:400])),
            4, "general", 0).strip().upper()
        winner = resp_b if result.startswith("B") else resp_a
        loser  = resp_a if result.startswith("B") else resp_b
        _rlaif_record(principle, winner, loser, original_msg, sb if result.startswith("B") else sa)
        return winner
    except: return resp_a

# ── SEMANTIC MEMORY — chromadb + sentence-transformers ───────────────────────
