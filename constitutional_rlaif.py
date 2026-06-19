"""
Constitutional AI + RLAIF — Anthropic-style implementation.
Two phases:
1. SL phase: critique → revise → save to finetune db
2. RLAIF phase: pairwise comparison using constitution principles → preference data
"""

import json, random, sqlite3, os
from typing import Optional

# ── Claude's constitution adapted for coding agent ──────────────────────────
CONSTITUTION = [
    "Does the response implement all claimed features with real code, not stubs or comments?",
    "Does the response handle errors with specific exceptions, not bare except or pass?",
    "Is the response truthful about what is and isn't implemented?",
    "Does the response avoid placeholder text like 'In real implementation' or 'TODO'?",
    "Does the response follow secure coding practices, avoiding obvious vulnerabilities?",
    "Is the code complete and runnable without requiring additional implementation?",
    "Does the response directly answer the task without unnecessary padding?",
    "Does the code handle edge cases like None, empty inputs, and concurrent access?",
]

DB = os.path.expanduser("~/eliteomni_rlaif.db")

def _init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS preferences (
        id INTEGER PRIMARY KEY,
        prompt TEXT,
        chosen TEXT,
        rejected TEXT,
        principle TEXT,
        created_at REAL DEFAULT (strftime('%s','now'))
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS revisions (
        id INTEGER PRIMARY KEY,
        original TEXT,
        critique TEXT,
        revised TEXT,
        principle TEXT,
        created_at REAL DEFAULT (strftime('%s','now'))
    )""")
    conn.commit()
    conn.close()

_init_db()

def critique_and_revise(prompt: str, response: str, generate_fn) -> tuple[str, str, str]:
    """
    SL phase: critique response against a random principle, then revise.
    Returns (principle, critique, revised_response)
    """
    principle = random.choice(CONSTITUTION)

    critique_prompt = [{"role": "user", "content": f"""Critique this AI response based on this principle:
PRINCIPLE: {principle}

TASK: {prompt[:300]}
RESPONSE: {response[:1000]}

Write a short critique (2-3 sentences) of how well the response follows the principle.
Be specific about what is missing or wrong. Critique:"""}]

    critique = generate_fn(critique_prompt, max_tokens=200) or ""

    revise_prompt = [{"role": "user", "content": f"""Revise this response to better follow this principle:
PRINCIPLE: {principle}
CRITIQUE: {critique}
ORIGINAL TASK: {prompt[:300]}
ORIGINAL RESPONSE: {response[:1000]}

Write an improved response that addresses the critique. Improved response:"""}]

    revised = generate_fn(revise_prompt, max_tokens=2000) or response

    # Save to DB
    try:
        conn = sqlite3.connect(DB)
        conn.execute("INSERT INTO revisions (original, critique, revised, principle) VALUES (?,?,?,?)",
                     (response[:2000], critique, revised[:2000], principle))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[CAI] revision save error: {e}")

    return principle, critique, revised

def pairwise_preference(prompt: str, response_a: str, response_b: str, generate_fn) -> tuple[str, str]:
    """
    RLAIF phase: compare two responses using a random constitutional principle.
    Returns (chosen, rejected) based on AI judgment.
    """
    principle = random.choice(CONSTITUTION)

    compare_prompt = [{"role": "user", "content": f"""Compare these two AI responses based on this principle:
PRINCIPLE: {principle}
TASK: {prompt[:300]}

RESPONSE A:
{response_a[:800]}

RESPONSE B:
{response_b[:800]}

Which response better follows the principle? Reply with ONLY "A" or "B". Answer:"""}]

    result = generate_fn(compare_prompt, max_tokens=5) or "A"
    winner = "A" if "A" in result.upper() else "B"
    chosen = response_a if winner == "A" else response_b
    rejected = response_b if winner == "A" else response_a

    # Save preference pair
    try:
        conn = sqlite3.connect(DB)
        conn.execute("INSERT INTO preferences (prompt, chosen, rejected, principle) VALUES (?,?,?,?)",
                     (prompt[:500], chosen[:2000], rejected[:2000], principle))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[CAI] preference save error: {e}")

    print(f"[CAI] pairwise: winner={winner} principle='{principle[:50]}'")
    return chosen, rejected

def export_preference_dataset(path: str = None) -> str:
    """Export preference pairs as JSONL for DPO fine-tuning."""
    path = path or os.path.expanduser("~/eliteomni_preferences.jsonl")
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT prompt, chosen, rejected FROM preferences").fetchall()
    conn.close()
    with open(path, "w") as f:
        for prompt, chosen, rejected in rows:
            f.write(json.dumps({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }) + "\n")
    print(f"[CAI] exported {len(rows)} preference pairs to {path}")

    # Auto-trigger Mistral fine-tune if enough data
    if len(rows) >= 100:
        try:
            import threading
            from mistral_finetune import run_finetune_pipeline
            print(f"[CAI] {len(rows)} pairs — triggering Mistral DPO fine-tune in background")
            threading.Thread(target=run_finetune_pipeline, args=(path,), daemon=True).start()
        except Exception as e:
            print(f"[CAI] finetune trigger error: {e}")

    return path

def get_stats() -> dict:
    conn = sqlite3.connect(DB)
    revisions = conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
    preferences = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
    conn.close()
    return {"revisions": revisions, "preferences": preferences}
