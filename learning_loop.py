"""
Online Learning Loop — the system actually improves after each interaction.
Implements: Preference learning, style adaptation, error pattern detection,
automatic prompt evolution, and behavioral drift detection.
"""
import sqlite3, json, time, re, os, threading, random
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

_DB = os.path.expanduser("~/eliteomni_learning.db")
_lock = threading.Lock()

def _init():
    con = sqlite3.connect(_DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        skill TEXT,
        complexity TEXT,
        msg_hash TEXT,
        msg_preview TEXT,
        response_preview TEXT,
        response_len INTEGER,
        latency_ms INTEGER,
        user_rating INTEGER DEFAULT 0,
        implicit_signal TEXT DEFAULT 'none'
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS error_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT,
        example TEXT,
        frequency INTEGER DEFAULT 1,
        last_seen REAL,
        fix_applied TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS style_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        preference_type TEXT,
        value TEXT,
        confidence REAL DEFAULT 0.5,
        evidence_count INTEGER DEFAULT 1,
        last_updated REAL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS prompt_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        component TEXT,
        version INTEGER,
        content TEXT,
        win_rate REAL DEFAULT 0.5,
        n_trials INTEGER DEFAULT 0,
        active INTEGER DEFAULT 0,
        created_ts REAL
    )""")
    con.commit(); con.close()

_init()

# ── Interaction Logging ───────────────────────────────────────────────────────

def log_interaction(skill: str, complexity: str, msg: str, response: str,
                    latency_ms: int, user_rating: int = 0):
    """Log every interaction for pattern learning."""
    import hashlib
    msg_hash = hashlib.md5(msg[:100].encode()).hexdigest()[:8]
    threading.Thread(
        target=_async_log,
        args=(skill, complexity, msg_hash, msg[:80], response[:100], len(response), latency_ms, user_rating),
        daemon=True
    ).start()

def _async_log(skill, complexity, msg_hash, msg_preview, response_preview,
               response_len, latency_ms, user_rating):
    try:
        with _lock:
            con = sqlite3.connect(_DB)
            con.execute(
                "INSERT INTO interactions (ts,skill,complexity,msg_hash,msg_preview,response_preview,response_len,latency_ms,user_rating) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), skill, complexity, msg_hash, msg_preview, response_preview, response_len, latency_ms, user_rating)
            )
            # Prune old records
            con.execute("DELETE FROM interactions WHERE id NOT IN (SELECT id FROM interactions ORDER BY ts DESC LIMIT 50000)")
            con.commit(); con.close()
        # Analyze for patterns every 50 interactions
        count = _get_interaction_count()
        if count % 50 == 0:
            _analyze_error_patterns()
            _update_style_preferences()
    except Exception as e:
        print(f"[LearningLoop] log error: {e}")

def _get_interaction_count() -> int:
    try:
        with _lock:
            con = sqlite3.connect(_DB)
            count = con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            con.close()
        return count
    except Exception:
        return 0

# ── Error Pattern Detection ───────────────────────────────────────────────────

def _analyze_error_patterns():
    """
    Detect recurring error patterns in responses.
    Looks for: hallucination signals, truncation patterns, refusal over-triggering.
    """
    try:
        with _lock:
            con = sqlite3.connect(_DB)
            recent = con.execute(
                "SELECT response_preview, skill, user_rating FROM interactions WHERE ts > ? ORDER BY ts DESC LIMIT 500",
                (time.time() - 86400 * 7,)
            ).fetchall()
            con.close()

        error_signals = {
            'over_refusal': r'(?:cannot|unable|I\'m sorry|I apologize|as an AI)',
            'truncation': r'\.{3}(?:\s*\[truncated\])?$',
            'hallucination_signal': r'(?:according to|as of \d{4}|recent studies show)',
            'filler_start': r'^(?:Certainly|Absolutely|Great question|Sure)',
            'excessive_caveats': r'(?:note that|keep in mind|it\'s worth noting|disclaimer)',
        }

        pattern_counts = defaultdict(list)
        for response, skill, rating in recent:
            for pattern_name, regex in error_signals.items():
                if re.search(regex, response, re.IGNORECASE):
                    pattern_counts[pattern_name].append({'skill': skill, 'rating': rating})

        with _lock:
            con = sqlite3.connect(_DB)
            for pattern_type, occurrences in pattern_counts.items():
                if len(occurrences) >= 5:  # Only log frequent patterns
                    neg_ratings = sum(1 for o in occurrences if o['rating'] < 0)
                    example = f"Occurred {len(occurrences)}x, {neg_ratings} negative ratings"
                    existing = con.execute(
                        "SELECT id, frequency FROM error_patterns WHERE pattern_type=?",
                        (pattern_type,)
                    ).fetchone()
                    if existing:
                        con.execute(
                            "UPDATE error_patterns SET frequency=?, last_seen=? WHERE id=?",
                            (existing[1] + len(occurrences), time.time(), existing[0])
                        )
                    else:
                        con.execute(
                            "INSERT INTO error_patterns (pattern_type, example, frequency, last_seen) VALUES (?,?,?,?)",
                            (pattern_type, example, len(occurrences), time.time())
                        )
            con.commit(); con.close()
    except Exception as e:
        print(f"[LearningLoop] error pattern analysis: {e}")

# ── Style Preference Learning ─────────────────────────────────────────────────

def _update_style_preferences():
    """
    Infer user style preferences from interaction patterns.
    Detects: preferred response length, markdown usage, detail level.
    """
    try:
        with _lock:
            con = sqlite3.connect(_DB)
            recent = con.execute(
                "SELECT response_len, user_rating, skill FROM interactions WHERE user_rating != 0 AND ts > ? LIMIT 200",
                (time.time() - 86400 * 30,)
            ).fetchall()
            con.close()

        if len(recent) < 10:
            return

        # Analyze preferred length
        pos_lengths = [r for r, rating, _ in recent if rating > 0]
        neg_lengths = [r for r, rating, _ in recent if rating < 0]

        if pos_lengths:
            avg_pos = sum(pos_lengths) / len(pos_lengths)
            preferred_length = "short" if avg_pos < 300 else "long" if avg_pos > 1000 else "medium"
            _upsert_preference("response_length", preferred_length, 0.6)

        # Skill-specific preferences
        skill_ratings = defaultdict(list)
        for _, rating, skill in recent:
            skill_ratings[skill].append(rating)

        for skill, ratings in skill_ratings.items():
            if len(ratings) >= 5:
                avg = sum(ratings) / len(ratings)
                if avg > 0.5:
                    _upsert_preference(f"skill_{skill}_quality", "high", min(0.5 + avg * 0.1, 0.9))
                elif avg < -0.5:
                    _upsert_preference(f"skill_{skill}_quality", "needs_improvement", 0.7)
    except Exception as e:
        print(f"[StylePref] {e}")

def _upsert_preference(pref_type: str, value: str, confidence: float):
    try:
        with _lock:
            con = sqlite3.connect(_DB)
            existing = con.execute(
                "SELECT id, evidence_count FROM style_preferences WHERE preference_type=?",
                (pref_type,)
            ).fetchone()
            if existing:
                new_count = existing[1] + 1
                # Bayesian confidence update
                new_conf = (confidence + existing[1] * confidence) / (new_count + 1)
                con.execute(
                    "UPDATE style_preferences SET value=?, confidence=?, evidence_count=?, last_updated=? WHERE id=?",
                    (value, min(new_conf, 0.95), new_count, time.time(), existing[0])
                )
            else:
                con.execute(
                    "INSERT INTO style_preferences (preference_type, value, confidence, last_updated) VALUES (?,?,?,?)",
                    (pref_type, value, confidence, time.time())
                )
            con.commit(); con.close()
    except Exception as e:
        print(f"[Preference upsert] {e}")

# ── System Prompt Evolution ───────────────────────────────────────────────────

def get_learned_system_addendum() -> str:
    """
    Build system prompt additions from learned patterns.
    Directly improves behavior based on what worked and failed.
    """
    parts = []
    try:
        with _lock:
            con = sqlite3.connect(_DB)
            # High-frequency error patterns → explicit avoidance instructions
            errors = con.execute(
                "SELECT pattern_type, frequency FROM error_patterns WHERE frequency > 10 ORDER BY frequency DESC LIMIT 5"
            ).fetchall()
            prefs = con.execute(
                "SELECT preference_type, value, confidence FROM style_preferences WHERE confidence > 0.6 ORDER BY confidence DESC LIMIT 5"
            ).fetchall()
            con.close()

        if errors:
            error_fixes = {
                'over_refusal': 'Be helpful. Do not refuse unless genuinely harmful.',
                'filler_start': 'Never start with "Certainly", "Absolutely", or "Great question".',
                'excessive_caveats': 'State information directly. Minimize caveats.',
                'truncation': 'Always complete your response. Never truncate mid-sentence.',
                'hallucination_signal': 'Only cite real sources. If uncertain, say so.',
            }
            active_fixes = [error_fixes[e] for e, _ in errors if e in error_fixes]
            if active_fixes:
                parts.append("[LEARNED BEHAVIOR CORRECTIONS]\n" + "\n".join(f"- {f}" for f in active_fixes))

        if prefs:
            pref_instructions = []
            for pref_type, value, confidence in prefs:
                if pref_type == 'response_length':
                    if value == 'short':
                        pref_instructions.append("User prefers concise responses. Be direct.")
                    elif value == 'long':
                        pref_instructions.append("User prefers detailed responses. Be thorough.")
            if pref_instructions:
                parts.append("[LEARNED USER PREFERENCES]\n" + "\n".join(f"- {p}" for p in pref_instructions))

    except Exception as e:
        print(f"[LearnedAddendum] {e}")

    return "\n\n".join(parts) if parts else ""

# ── Behavioral Drift Detection ────────────────────────────────────────────────

def check_drift() -> Optional[str]:
    """
    Detect if system behavior is drifting negatively.
    Returns warning string if drift detected, None if OK.
    """
    try:
        with _lock:
            con = sqlite3.connect(_DB)
            # Compare last 7 days vs prior 7 days ratings
            now = time.time()
            recent = con.execute(
                "SELECT AVG(user_rating) FROM interactions WHERE ts > ? AND user_rating != 0",
                (now - 86400 * 7,)
            ).fetchone()[0] or 0
            prior = con.execute(
                "SELECT AVG(user_rating) FROM interactions WHERE ts BETWEEN ? AND ? AND user_rating != 0",
                (now - 86400 * 14, now - 86400 * 7)
            ).fetchone()[0] or 0
            con.close()

        if prior != 0 and (recent - prior) < -0.2:
            return f"[DRIFT ALERT] Average rating dropped from {prior:.2f} to {recent:.2f} this week"
        return None
    except Exception:
        return None

def get_learning_stats() -> dict:
    """Return current learning statistics."""
    try:
        with _lock:
            con = sqlite3.connect(_DB)
            total = con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            rated = con.execute("SELECT COUNT(*) FROM interactions WHERE user_rating != 0").fetchone()[0]
            avg_rating = con.execute("SELECT AVG(user_rating) FROM interactions WHERE user_rating != 0").fetchone()[0] or 0
            error_count = con.execute("SELECT COUNT(*) FROM error_patterns").fetchone()[0]
            pref_count = con.execute("SELECT COUNT(*) FROM style_preferences").fetchone()[0]
            con.close()
        return {
            "total_interactions": total,
            "rated_interactions": rated,
            "avg_rating": round(avg_rating, 3),
            "error_patterns_detected": error_count,
            "style_preferences_learned": pref_count,
        }
    except Exception:
        return {}
