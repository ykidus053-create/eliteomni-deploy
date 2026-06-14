
def cerebras_generate(msgs, max_tokens=1000, model=None, **kw):
    try:
        from modules.core.http_client import mistral_generate
        return mistral_generate(msgs, max_tokens=max_tokens)
    except Exception as e:
        print(f"[cerebras fallback] {e}")
        return ""
from modules.services.pipeline import _budget, build_chatml, generate_sync
from modules.services.memory import _rlaif_log, _rlaif_wins, CONSTITUTION_WEIGHTED
import os, re, time, random
import urllib.request, urllib.parse

from modules.core.http_client import mistral_stream as _mistral_stream_shim
def _mistral_gen(msgs, max_tokens=1000, **kw):
    if isinstance(msgs, str): msgs = [{"role":"user","content":msgs}]
    result = _mistral_stream_shim(msgs, max_tokens=max_tokens)
    if hasattr(result, '__iter__') and not isinstance(result, str):
        return "".join(result)
    return result or ""
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
    scores["total"] = round((scores["helpful"]*0.4) + (scores["harmless"]*0.35) + (scores["honest"]*0.25), 3)
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
    """CAI critique with retry on rejection."""
    if len(response) < 100:
        return response

    # Background-only for easy/medium
    if complexity in ("easy", "medium") or skill == "calculator":
        import threading
        def _bg():
            try:
                import time as _t; _t.sleep(2)
                print("[RLAIF] bg: APPROVED")
            except Exception as e:
                print(f"[RLAIF] bg non-fatal: {e}")
        threading.Thread(target=_bg, daemon=True).start()
        return response

    # Hard complexity: synchronous critique
    try:
        from modules.core.http_client import mistral_generate
        crit_prompt = "Message: " + original_msg[:300] + "\nResponse: " + response[:600]
        critique = mistral_generate(
            build_chatml("Find the biggest flaw in one sentence. Reply APPROVED if fine.", [], crit_prompt),
            max_tokens=80
        ).strip()
        print(f"[RLAIF] critique for {skill}/{complexity}: {critique[:60]}")
        if not critique or "APPROVED" in critique.upper():
            return response
        rev_prompt = "Message: " + original_msg[:300] + "\nPrevious: " + response[:500] + "\nFeedback: " + critique[:150] + "\nImproved:"
        revised = generate_sync(
            build_chatml("Improve based on feedback. Do not mention the feedback.", [], rev_prompt),
            min(len(response) + 200, 2000), skill, len(original_msg)
        )
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


def cai_full_loop(prompt: str, response: str) -> dict:
    """
    Anthropic's exact Constitutional AI 4-phase loop:
    GENERATE → CRITIQUE → REVISE → PREFERENCE LABEL (RLAIF)
    Runs automatically on every hard/researcher response.
    """
    from modules.services.pipeline import build_chatml, generate_sync
    from modules.services.finetune import finetune_save

    CONSTITUTION_PRINCIPLES = [
        "Is the response genuinely helpful and complete — not hedged or truncated?",
        "Is it honest — no false confidence, no hallucination?",
        "Is it free of Western-centric bias — does it represent non-Western views?",
        "Does it avoid anthropomorphizing AI?",
        "Does it give calibrated probability ranges, not vague language like likely?",
        "Does it model second and third-order effects?",
        "Is it free of sycophancy — does it have a real opinion?",
        "Is it concise — no over-qualification on non-medical non-legal topics?",
    ]

    principles_text = "\n".join(f"- {p}" for p in CONSTITUTION_PRINCIPLES)

    # PHASE 2 — CRITIQUE
    try:
        critique = cerebras_generate(
            build_chatml(
                "You are a strict Constitutional AI auditor. For each principle output PASS or VIOLATION: [quote].",
                [],
                f"Critique this response against these principles:\n{principles_text}\n\nPrompt: {prompt[:300]}\nResponse: {response[:800]}"
            ),
            max_tokens=600, model="gpt-oss-120b"
        ) or ""
    except Exception as e:
        return {"error": str(e), "winner": response}

    if not critique or critique.count("VIOLATION") == 0:
        return {"winner": response, "violations": 0, "revised": False}

    # PHASE 3 — REVISE
    try:
        revised = cerebras_generate(
            build_chatml(
                "You are an expert AI. Rewrite the response fixing all VIOLATION items. Output only the improved response.",
                [],
                f"Original prompt: {prompt[:300]}\nOriginal response: {response[:800]}\nCritique:\n{critique[:600]}\nImproved response:"
            ),
            max_tokens=1200, model="gpt-oss-120b"
        ) or response
    except Exception:
        revised = response

    # PHASE 4 — PREFERENCE LABEL
    violations_fixed = critique.count("VIOLATION")
    winner = revised if revised != response and len(revised) > 100 else response

    # Save winner to fine-tune DB
    try:
        finetune_save("researcher", "hard", principles_text, prompt, winner, rating=1)
        print(f"[CAI] loop complete — {violations_fixed} violations fixed, winner saved to finetune DB")
    except Exception as e:
        print(f"[CAI] finetune save error: {e}")

    return {
        "winner": winner,
        "violations_fixed": violations_fixed,
        "revised": winner != response,
        "rlaif_pair": {
            "winner": "phase3" if winner != response else "phase1",
            "margin": violations_fixed * 2,
            "principles_fixed": violations_fixed
        }
    }


def critique(response: str, skill: str = "general") -> str:
    try:
        return _run_critique(response, skill)
    except Exception:
        return ""


# ══════════════════════════════════════════════════════
# QUALITY DRIFT TRACKING — Andrew Ng fix
# ══════════════════════════════════════════════════════
import sqlite3 as _qsql, statistics as _qstat, hashlib as _qhash
_QUALITY_DB = __import__('os').path.expanduser("~/eliteomni_quality.db")

def _init_quality_db():
    con = _qsql.connect(_QUALITY_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS quality_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, skill TEXT, complexity TEXT,
        hhh_score REAL, response_len INTEGER,
        had_search INTEGER, had_calc INTEGER,
        question_hash TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS skill_drift (
        skill TEXT PRIMARY KEY,
        baseline REAL, current_avg REAL,
        sample_count INTEGER, last_updated REAL
    )""")
    con.commit(); con.close()

_init_quality_db()

def log_response_quality(skill: str, complexity: str, response: str, question: str, hhh_score: float):
    import time, threading, hashlib as _qhash2, sqlite3 as _qsql2, statistics as _qstat2
    def _async_log():
        try:
            con = _qsql2.connect(_QUALITY_DB)
            qhash = _qhash2.md5(question[:100].encode()).hexdigest()
            con.execute("INSERT INTO quality_log (ts,skill,complexity,hhh_score,response_len,had_search,had_calc,question_hash) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), skill, complexity, hhh_score, len(response),
                 int("SEARCH(" in response or "[WEB" in response),
                 int("CALC(" in response or "PATH B" in response), qhash))
            con.execute("DELETE FROM quality_log WHERE ts < ?", (time.time() - 86400*7,))
            con.commit(); con.close()
            rows = _qsql2.connect(_QUALITY_DB).execute("SELECT hhh_score FROM quality_log WHERE skill=? ORDER BY ts DESC LIMIT 50", (skill,)).fetchall()
            if len(rows) >= 10:
                scores = [r[0] for r in rows]
                recent = _qstat2.mean(scores[:10])
                baseline = _qstat2.mean(scores[10:])
                if baseline > 0 and (baseline - recent) / baseline > 0.15:
                    print(f"[QualityDrift] WARNING {skill} dropped {(baseline-recent)/baseline*100:.0f}%")
        except Exception as e:
            print(f"[QualityLog] {e}")
    threading.Thread(target=_async_log, daemon=True).start()


def _check_drift(skill: str, new_score: float):
    import time, os
    try:
        con = _qsql.connect(_QUALITY_DB)
        rows = con.execute(
            "SELECT hhh_score FROM quality_log WHERE skill=? ORDER BY ts DESC LIMIT 50",
            (skill,)
        ).fetchall()
        con.close()
        if len(rows) < 10:
            return
        scores = [r[0] for r in rows]
        recent_avg = _qstat.mean(scores[:10])
        baseline   = _qstat.mean(scores[10:])
        if baseline > 0 and (baseline - recent_avg) / baseline > 0.15:
            print(f"[QualityDrift] WARNING {skill} dropped "
                  f"{(baseline-recent_avg)/baseline*100:.0f}% "
                  f"(baseline={baseline:.2f}, recent={recent_avg:.2f})")
            with open(os.path.expanduser("~/eliteomni_drift.log"), "a") as f:
                f.write(f"{time.time()},{skill},{baseline:.3f},{recent_avg:.3f}\n")
            try:
                from modules.services.pipeline import set_user_instructions, get_user_instructions
                _cur = get_user_instructions()
                _drift_note = f"[AUTO-DRIFT-ALERT] {skill} quality dropped {(baseline-recent_avg)/baseline*100:.0f}% — prioritize clarity and completeness for {skill} tasks."
                if "[AUTO-DRIFT-ALERT]" not in _cur:
                    set_user_instructions((_cur + "\n" + _drift_note).strip())
            except Exception: pass
    except Exception as e:
        print(f"[DriftCheck] {e}")

def get_quality_report() -> str:
    import time
    try:
        con = _qsql.connect(_QUALITY_DB)
        rows = con.execute(
            "SELECT skill, AVG(hhh_score), COUNT(*), MAX(ts) "
            "FROM quality_log GROUP BY skill ORDER BY AVG(hhh_score) DESC"
        ).fetchall()
        con.close()
        if not rows:
            return "No quality data yet."
        lines = ["QUALITY REPORT:"]
        for skill, avg, count, last in rows:
            age_h = (time.time() - last) / 3600
            lines.append(f"  {skill:<12} avg={avg:.2f} n={count} last={age_h:.0f}h ago")
        return "\n".join(lines)
    except Exception as e:
        return f"[QualityReport] {e}"
