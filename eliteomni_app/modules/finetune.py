
# Claude-style SFT quality filter
# Only train on examples that meet HHH standards
SFT_QUALITY_FILTER = """
Only use this example for fine-tuning if ALL of these are true:
1. The response is genuinely helpful and complete — not hedged or truncated
2. The response is honest — no false confidence, no hallucination
3. The response is direct — no sycophantic opener, no filler
4. The response matches the user tone (casual/technical)
5. For code: it is complete, typed, and runnable
6. For math: CALC() was used and the answer is correct
7. The response does NOT start with: Certainly, Absolutely, Great, Sure, Of course
"""

SFT_REJECT_PATTERNS = [
    "Certainly!", "Absolutely!", "Great question!", "Sure!", "Of course!",
    "I cannot", "I'm unable to", "As an AI language model",
    "I don't have the ability", "I apologize, but I cannot",
]
from modules.config import RATE_LIMIT
from modules.config import RATE_LIMIT
from modules.memory import _rate_lim
# AUTO-SPLIT FROM app.py lines 2878-2945
import os, re, time, math, json, asyncio, random, ast, subprocess, sys, tempfile
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import urllib.request, urllib.parse

import sqlite3 as _sqlite3

FINETUNE_DB = os.path.expanduser("~/eliteomni_finetune.db")

def _init_finetune_db():
    con = _sqlite3.connect(FINETUNE_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, skill TEXT, complexity TEXT,
        system_prompt TEXT, user_msg TEXT,
        assistant_response TEXT, rating INTEGER DEFAULT 0
    )""")
    con.commit(); con.close()

_init_finetune_db()

def finetune_save(skill: str, complexity: str, system: str, user: str, response: str, rating: int = 0):
    """Save every conversation as a fine-tune training sample."""
    try:
        con = _sqlite3.connect(FINETUNE_DB)
        con.execute(
            "INSERT INTO samples (ts,skill,complexity,system_prompt,user_msg,assistant_response,rating) VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), skill, complexity, system[:800], user[:600], response[:1200], rating)
        )
        con.commit(); con.close()
    except Exception as e:
        print(f"[finetune_save] {e}")

def finetune_export_jsonl(min_rating: int = 0, path: str = None) -> str:
    """Export training data as JSONL ready for Unsloth/HuggingFace fine-tuning."""
    path = path or os.path.expanduser("~/eliteomni_finetune.jsonl")
    try:
        con = _sqlite3.connect(FINETUNE_DB)
        rows = con.execute(
            "SELECT system_prompt,user_msg,assistant_response FROM samples WHERE rating>=? ORDER BY ts DESC LIMIT 5000",
            (min_rating,)
        ).fetchall()
        con.close()
        with open(path, "w") as f:
            for system, user, assistant in rows:
                record = {
                    "conversations": [
                        {"role": "system",    "content": system},
                        {"role": "user",      "content": user},
                        {"role": "assistant", "content": assistant}
                    ]
                }
                f.write(json.dumps(record) + "\n")
        return f"Exported {len(rows)} samples to {path}"
    except Exception as e:
        return f"Export error: {e}"

# Groq rate limit handler with auto-retry
def _groq_rate_wait(retry: int = 3, wait: float = 5.0):
    """Auto-retry on Groq 429 rate limit."""
    import time
    for i in range(retry):
        time.sleep(wait * (i + 1))
        print(f"[RateLimit] retry {i+1}/{retry} after {wait*(i+1)}s")

def check_rate(ip: str) -> bool:
    now = time.time()
    _rate_lim[ip] = [t for t in _rate_lim[ip] if now-t<60]
    if len(_rate_lim[ip]) >= RATE_LIMIT: return False
    _rate_lim[ip].append(now); return True

# ── ARCHITECT / EDITOR SPLIT (Claude Code: safer code generation) ─────────────
