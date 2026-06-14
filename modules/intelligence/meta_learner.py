"""
Meta-Learning and Reasoning Corpus
- Stores high-quality reasoning PATTERNS (not just answers)
- Retrieves patterns by structural similarity for few-shot reasoning
- Tracks which reasoning strategies work for which problem types
- Implements a lightweight data flywheel
"""
import json, sqlite3, time, re, hashlib
from pathlib import Path
from typing import List, Optional

DB = Path.home() / "eliteomni_reasoning_corpus.db"

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS patterns (
        id TEXT PRIMARY KEY,
        problem_type TEXT,
        skill TEXT,
        complexity TEXT,
        reasoning_chain TEXT,
        final_answer TEXT,
        step_count INTEGER,
        prm_score REAL,
        user_rating INTEGER DEFAULT 0,
        use_count INTEGER DEFAULT 0,
        created REAL,
        last_used REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS strategy_performance (
        strategy TEXT,
        skill TEXT,
        problem_type TEXT,
        success_count INTEGER DEFAULT 0,
        fail_count INTEGER DEFAULT 0,
        avg_prm_score REAL DEFAULT 0.5
    )""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_patterns_skill ON patterns(skill, prm_score)""")
    c.commit()
    return c

def _problem_type(msg: str) -> str:
    m = msg.lower()
    if any(w in m for w in ["calculate", "compute", "what is", "how much", "percent"]):
        return "calculation"
    if any(w in m for w in ["write", "implement", "create", "build", "code"]):
        return "implementation"
    if any(w in m for w in ["explain", "what is", "how does", "why"]):
        return "explanation"
    if any(w in m for w in ["compare", "difference", "versus", "vs"]):
        return "comparison"
    if any(w in m for w in ["debug", "fix", "error", "bug", "broken"]):
        return "debugging"
    return "general"

def _extract_chain(response: str) -> str:
    """Extract just the reasoning chain, not the final answer prose."""
    lines = response.split('\n')
    chain_lines = []
    for line in lines:
        if re.match(r'^\s*(\d+\.|Step|→|First|Then|Therefore)', line):
            chain_lines.append(line.strip())
    return '\n'.join(chain_lines[:12]) if chain_lines else response[:600]

def store_pattern(msg: str, response: str, skill: str, complexity: str,
                  prm_score: float = 0.8, user_rating: int = 0):
    """Store a reasoning pattern if quality threshold met."""
    if prm_score < 0.65 and user_rating < 1:
        return
    chain = _extract_chain(response)
    if len(chain) < 50:
        return
    ptype = _problem_type(msg)
    pid = hashlib.md5(f"{skill}{ptype}{chain[:100]}".encode()).hexdigest()[:16]
    try:
        c = _conn()
        c.execute("""INSERT OR REPLACE INTO patterns
            (id, problem_type, skill, complexity, reasoning_chain, final_answer,
             step_count, prm_score, user_rating, use_count, created, last_used)
            VALUES (?,?,?,?,?,?,?,?,?,0,?,?)""",
            (pid, ptype, skill, complexity, chain, response[:800],
             len(chain.split('\n')), prm_score, user_rating,
             time.time(), time.time()))
        c.commit()
        c.close()
    except Exception as e:
        print(f"[MetaLearner] store error: {e}")

def retrieve_patterns(msg: str, skill: str, k: int = 2) -> List[str]:
    """Retrieve top-k reasoning patterns for this problem type."""
    ptype = _problem_type(msg)
    try:
        c = _conn()
        rows = c.execute("""
            SELECT reasoning_chain, prm_score FROM patterns
            WHERE skill=? AND problem_type=? AND prm_score >= 0.7
            ORDER BY (prm_score * 0.6 + user_rating * 0.4) DESC, last_used DESC
            LIMIT ?""", (skill, ptype, k)).fetchall()
        # Update use count
        c.execute("""UPDATE patterns SET use_count = use_count + 1, last_used = ?
                     WHERE skill=? AND problem_type=?""",
                  (time.time(), skill, ptype))
        c.commit()
        c.close()
        return [row[0] for row in rows]
    except Exception:
        return []

def build_reasoning_exemplars(msg: str, skill: str) -> str:
    """Build exemplar injection from reasoning corpus."""
    patterns = retrieve_patterns(msg, skill, k=2)
    if not patterns:
        return ""
    parts = ["\n<reasoning_exemplars>"]
    parts.append("Reference these high-quality reasoning patterns for this problem type:")
    for i, p in enumerate(patterns, 1):
        parts.append(f"\nEXEMPLAR {i}:\n{p}")
    parts.append("</reasoning_exemplars>")
    return "\n".join(parts)

def record_strategy_outcome(strategy: str, skill: str, problem_type: str,
                             success: bool, prm_score: float):
    """Track which strategies work for which problem types."""
    try:
        c = _conn()
        row = c.execute("""SELECT success_count, fail_count, avg_prm_score
                            FROM strategy_performance
                            WHERE strategy=? AND skill=? AND problem_type=?""",
                        (strategy, skill, problem_type)).fetchone()
        if row:
            sc, fc, avg = row
            sc = sc + (1 if success else 0)
            fc = fc + (0 if success else 1)
            new_avg = (avg * (sc + fc - 1) + prm_score) / (sc + fc)
            c.execute("""UPDATE strategy_performance SET success_count=?, fail_count=?,
                         avg_prm_score=? WHERE strategy=? AND skill=? AND problem_type=?""",
                      (sc, fc, new_avg, strategy, skill, problem_type))
        else:
            c.execute("""INSERT INTO strategy_performance VALUES (?,?,?,?,?,?)""",
                      (strategy, skill, problem_type,
                       1 if success else 0, 0 if success else 1, prm_score))
        c.commit()
        c.close()
    except Exception as e:
        print(f"[MetaLearner] strategy record error: {e}")

def get_best_strategy(skill: str, problem_type: str) -> Optional[str]:
    """Return the empirically best strategy for this skill+problem combination."""
    try:
        c = _conn()
        row = c.execute("""
            SELECT strategy FROM strategy_performance
            WHERE skill=? AND problem_type=?
            AND (success_count + fail_count) >= 3
            ORDER BY avg_prm_score DESC LIMIT 1""",
            (skill, problem_type)).fetchone()
        c.close()
        return row[0] if row else None
    except Exception:
        return None
