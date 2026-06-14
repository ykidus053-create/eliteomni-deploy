"""
Autonomous Self-Improvement Loop
Exactly how Anthropic's CAI works, implemented at inference time.

THE LOOP:
  1. GENERATE    — produce initial response
  2. CRITIQUE    — AI critiques against constitution
  3. REVISE      — AI rewrites based on critique  
  4. SCORE       — AI scores both versions 0-10
  5. SELECT      — keep winner
  6. SAVE        — store winner to fine-tune DB
  7. INJECT      — load best past responses into future prompts
  8. BENCHMARK   — track quality score over time
  9. ALERT       — flag regressions automatically
"""

import sqlite3
import json
import time
import threading
import os
from datetime import datetime, timezone

FINETUNE_DB  = "/home/kidus/eliteomni_finetune.db"
BENCHMARK_DB = "/home/kidus/eliteomni_benchmark.db"
IMPROVEMENT_LOG = "/home/kidus/eliteomni_improvement.jsonl"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1+2: CAI CRITIQUE PROMPT — constitutional self-critique
# Exactly Anthropic's phase 1 supervised learning loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_critique_prompt(original_msg: str, response: str, skill: str) -> list:
    """Builds the CAI self-critique prompt."""
    return [
        {
            "role": "user",
            "content": f"""You are a constitutional AI critic. Critique this response against these principles:

CONSTITUTION:
1. Truthful — only assert what is true, no hallucinations
2. Calibrated — uncertainty expressed proportionally  
3. Helpful — actually solves the user's problem
4. Harmless — no harmful content
5. Non-deceptive — no false impressions
6. High quality — clear, concise, well-reasoned
7. Complete — addresses all parts of the question

ORIGINAL QUESTION: {original_msg[:500]}

RESPONSE TO CRITIQUE:
{response[:2000]}

Identify specific violations of the above principles. Be precise and actionable.
Format: list each issue as "ISSUE: [principle] — [what's wrong] — [how to fix]"
If no issues, write "PASS: response meets all constitutional principles"
"""
        }
    ]


def build_revision_prompt(original_msg: str, response: str, critique: str, skill: str) -> list:
    """Builds the CAI revision prompt."""
    return [
        {
            "role": "user", 
            "content": f"""Rewrite this response to fix all identified issues.

ORIGINAL QUESTION: {original_msg[:500]}

ORIGINAL RESPONSE:
{response[:2000]}

CRITIQUE TO ADDRESS:
{critique[:1000]}

Write an improved response that fixes every issue identified.
Only output the improved response — no preamble, no explanation.
"""
        }
    ]


def build_scoring_prompt(msg: str, response_a: str, response_b: str) -> list:
    """Scores two responses — picks the better one."""
    return [
        {
            "role": "user",
            "content": f"""Score these two AI responses on a scale of 0-10 for:
- Accuracy (is it correct?)
- Helpfulness (does it solve the problem?)  
- Clarity (is it clear and well-written?)
- Safety (is it safe and appropriate?)

QUESTION: {msg[:300]}

RESPONSE A:
{response_a[:1500]}

RESPONSE B:
{response_b[:1500]}

Reply ONLY with this JSON (no markdown, no explanation):
{{"score_a": <0-10>, "score_b": <0-10>, "winner": "A" or "B", "reason": "<one sentence>"}}
"""
        }
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3-6: THE FULL CAI LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_cai_loop(msg: str, response: str, skill: str, complexity: str) -> dict:
    """
    Runs the full Constitutional AI loop:
    Generate → Critique → Revise → Score → Select → Save
    Returns the best response and metadata.
    """
    try:
        from modules.services.pipeline import generate_sync

        result = {
            "original": response,
            "critique": "",
            "revised": "",
            "winner": response,
            "score_original": 0,
            "score_revised": 0,
            "improved": False,
        }

        # Skip very short responses
        if len(response) < 100:
            return result

        # PHASE 1: CRITIQUE
        critique_msgs = build_critique_prompt(msg, response, skill)
        critique = generate_sync(critique_msgs, 800, skill, len(msg))
        result["critique"] = critique

        # If no issues found, skip revision
        if critique and "PASS:" in critique[:50]:
            _save_to_finetune(msg, response, skill, complexity, score=8.0)
            return result

        # PHASE 2: REVISE
        revision_msgs = build_revision_prompt(msg, response, critique, skill)
        revised = generate_sync(revision_msgs, 2000, skill, len(msg))
        result["revised"] = revised or response

        if not revised or len(revised) < 50:
            return result

        # PHASE 3: SCORE — pick the winner
        scoring_msgs = build_scoring_prompt(msg, response, revised)
        score_raw = generate_sync(scoring_msgs, 200, "general", len(msg))

        try:
            # Strip markdown if present
            score_raw = score_raw.strip().strip("```json").strip("```").strip()
            scores = json.loads(score_raw)
            result["score_original"] = scores.get("score_a", 5)
            result["score_revised"] = scores.get("score_b", 5)

            if scores.get("winner") == "B" and result["score_revised"] > result["score_original"]:
                result["winner"] = revised
                result["improved"] = True
                # Save REVISED (winner) to fine-tune DB
                _save_to_finetune(msg, revised, skill, complexity,
                                  score=result["score_revised"])
                _log_improvement(msg, skill, result["score_original"],
                                 result["score_revised"], scores.get("reason", ""))
            else:
                # Original won — save it too if score is good
                if result["score_original"] >= 7:
                    _save_to_finetune(msg, response, skill, complexity,
                                      score=result["score_original"])

        except (json.JSONDecodeError, KeyError):
            # Scoring failed — save original if long enough
            if len(response) > 200:
                _save_to_finetune(msg, response, skill, complexity, score=6.0)

        return result

    except Exception as e:
        print(f"[CAI Loop] error (non-fatal): {e}")
        return {"winner": response, "improved": False}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: SAVE TO FINE-TUNE DB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _save_to_finetune(msg: str, response: str, skill: str,
                      complexity: str, score: float = 7.0):
    """Save high-quality response to fine-tune DB."""
    try:
        con = sqlite3.connect(FINETUNE_DB)
        con.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, skill TEXT, complexity TEXT,
                system_prompt TEXT, user_msg TEXT,
                assistant_response TEXT, rating REAL
            )
        """)
        con.execute(
            "INSERT INTO samples (ts,skill,complexity,system_prompt,user_msg,assistant_response,rating) VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), skill, complexity,
             f"CAI-loop-winner skill={skill}", msg[:500], response[:3000], score)
        )
        con.commit()
        con.close()
    except Exception as e:
        print(f"[CAI Save] error: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: INJECT BEST PAST RESPONSES INTO FUTURE PROMPTS
# This is how the loop actually closes — best responses teach future ones
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_exemplars(skill: str, complexity: str, limit: int = 2) -> str:
    """
    Fetches top-rated past responses for this skill/complexity.
    Injects them as few-shot exemplars into the system prompt.
    This is how the model learns from its own best work.
    """
    try:
        con = sqlite3.connect(FINETUNE_DB)
        rows = con.execute("""
            SELECT user_msg, assistant_response, rating
            FROM samples
            WHERE skill=? AND complexity=? AND rating >= 8.0
            ORDER BY rating DESC, ts DESC
            LIMIT ?
        """, (skill, complexity, limit)).fetchall()
        con.close()

        if not rows:
            return ""

        exemplars = "\n<exemplars>\n"
        exemplars += "These are high-quality past responses for this type of task. Match this quality:\n\n"
        for i, (q, a, r) in enumerate(rows, 1):
            exemplars += f"EXAMPLE {i} (score {r:.1f}/10):\n"
            exemplars += f"Q: {q[:200]}\n"
            exemplars += f"A: {a[:400]}\n\n"
        exemplars += "</exemplars>\n"
        return exemplars

    except Exception:
        return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8: AUTO BENCHMARK — track quality over time
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BENCHMARK_SUITE = [
    {"id": "r1", "skill": "researcher", "complexity": "medium",
     "q": "Explain the difference between supervised and unsupervised learning",
     "must_contain": ["label", "cluster", "train"]},
    {"id": "c1", "skill": "coder", "complexity": "medium",
     "q": "Write a Python function to find duplicates in a list with O(n) complexity",
     "must_contain": ["def ", "set(", "return"]},
    {"id": "m1", "skill": "calculator", "complexity": "easy",
     "q": "What is 15% of 3750?",
     "must_contain": ["562.5", "562"]},
    {"id": "a1", "skill": "general", "complexity": "hard",
     "q": "What are the tradeoffs between microservices and monolithic architecture?",
     "must_contain": ["scale", "complex", "deploy"]},
]

def run_benchmark() -> dict:
    """Runs benchmark suite, returns quality scores."""
    try:
        from modules.services.pipeline import generate_sync
        results = {}
        for test in BENCHMARK_SUITE:
            try:
                msgs = [{"role": "user", "content": test["q"]}]
                resp = generate_sync(msgs, 500, test["skill"], len(test["q"]))
                resp_lower = resp.lower()
                hits = sum(1 for kw in test["must_contain"] if kw in resp_lower)
                score = hits / len(test["must_contain"])
                results[test["id"]] = {"score": score, "pass": score >= 0.67}
            except Exception as e:
                results[test["id"]] = {"score": 0, "pass": False, "error": str(e)}

        overall = sum(r["score"] for r in results.values()) / len(results)
        _save_benchmark(overall, results)
        return {"overall": overall, "tests": results}

    except Exception as e:
        return {"error": str(e)}


def _save_benchmark(score: float, details: dict):
    """Saves benchmark result for regression tracking."""
    try:
        con = sqlite3.connect(BENCHMARK_DB)
        con.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, overall_score REAL, details TEXT
            )
        """)
        con.execute(
            "INSERT INTO benchmarks (ts, overall_score, details) VALUES (?,?,?)",
            (datetime.now(timezone.utc).isoformat(), score, json.dumps(details))
        )
        con.commit()
        con.close()
    except Exception:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 9: REGRESSION DETECTION — alert on quality drops
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def check_regression() -> dict:
    """Compares last 2 benchmarks. Alerts if score dropped > 10%."""
    try:
        con = sqlite3.connect(BENCHMARK_DB)
        rows = con.execute(
            "SELECT ts, overall_score FROM benchmarks ORDER BY ts DESC LIMIT 2"
        ).fetchall()
        con.close()

        if len(rows) < 2:
            return {"status": "insufficient_data"}

        latest, previous = rows[0][1], rows[1][1]
        drop = previous - latest
        pct = (drop / previous * 100) if previous > 0 else 0

        if pct > 10:
            print(f"[REGRESSION ALERT] Quality dropped {pct:.1f}% "
                  f"({previous:.2f} → {latest:.2f})")
            return {"status": "regression", "drop_pct": pct,
                    "previous": previous, "latest": latest}

        return {"status": "ok", "latest": latest, "previous": previous}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPROVEMENT LOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _log_improvement(msg: str, skill: str, before: float,
                     after: float, reason: str):
    """Logs every improvement event."""
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "skill": skill,
            "score_before": before,
            "score_after": after,
            "delta": after - before,
            "reason": reason,
            "msg_preview": msg[:80],
        }
        with open(IMPROVEMENT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[IMPROVEMENT] {skill} {before:.1f}→{after:.1f} (+{after-before:.1f}): {reason[:60]}")
    except Exception:
        pass


def get_improvement_stats() -> dict:
    """Returns improvement statistics."""
    try:
        if not os.path.exists(IMPROVEMENT_LOG):
            return {"total_improvements": 0, "avg_delta": 0}
        with open(IMPROVEMENT_LOG) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if not lines:
            return {"total_improvements": 0, "avg_delta": 0}
        avg_delta = sum(l["delta"] for l in lines) / len(lines)
        by_skill = {}
        for l in lines:
            by_skill[l["skill"]] = by_skill.get(l["skill"], 0) + 1
        return {
            "total_improvements": len(lines),
            "avg_delta": round(avg_delta, 2),
            "by_skill": by_skill,
            "latest": lines[-1] if lines else None,
        }
    except Exception as e:
        return {"error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKGROUND SCHEDULER — runs CAI loop + benchmark automatically
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_cai_queue = []
_queue_lock = threading.Lock()

def queue_for_improvement(msg: str, response: str,
                           skill: str, complexity: str):
    """Queue a response for async CAI improvement loop."""
    with _queue_lock:
        _cai_queue.append((msg, response, skill, complexity))

def _background_worker():
    """Processes CAI queue in background — never blocks streaming."""
    benchmark_counter = 0
    while True:
        try:
            time.sleep(2)  # poll every 2 seconds
            with _queue_lock:
                if not _cai_queue:
                    continue
                item = _cai_queue.pop(0)

            msg, response, skill, complexity = item
            # Only run CAI on medium/hard complexity — not trivial responses
            if complexity in ("medium", "hard") and len(response) > 150:
                run_cai_loop(msg, response, skill, complexity)

            # Run benchmark every 50 responses
            benchmark_counter += 1
            if benchmark_counter >= 50:
                benchmark_counter = 0
                result = run_benchmark()
                reg = check_regression()
                print(f"[BENCHMARK] overall={result.get('overall',0):.2f} "
                      f"regression={reg.get('status','?')}")

        except Exception as e:
            print(f"[CAI Worker] error: {e}")

def start_improvement_worker():
    """Start the background self-improvement worker."""
    t = threading.Thread(target=_background_worker,
                         daemon=True, name="cai_improvement")
    t.start()
    print("[Self-Improvement] ✓ CAI loop worker started")
    return t
