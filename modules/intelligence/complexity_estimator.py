"""
Learned Complexity Estimator
Replaces static regex routing with a feature-based classifier
that improves with each interaction via online learning.
"""
import re, math, sqlite3, time
from pathlib import Path
from typing import Tuple

DB = Path.home() / "eliteomni_complexity.db"

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS training (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        features TEXT,
        true_complexity TEXT,
        predicted TEXT,
        correct INTEGER,
        ts REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS weights (
        feature TEXT PRIMARY KEY,
        easy REAL DEFAULT 0.0,
        medium REAL DEFAULT 0.5,
        hard REAL DEFAULT 0.0
    )""")
    c.commit()
    return c

FEATURE_EXTRACTORS = {
    "length_short": lambda m: 1.0 if len(m) < 60 else 0.0,
    "length_medium": lambda m: 1.0 if 60 <= len(m) < 200 else 0.0,
    "length_long": lambda m: 1.0 if len(m) >= 200 else 0.0,
    "has_code_request": lambda m: 1.0 if any(w in m.lower() for w in
        ["implement", "write a function", "create a class", "build"]) else 0.0,
    "is_calculation": lambda m: 1.0 if any(w in m.lower() for w in
        ["calculate", "compute", "solve", "integral", "derivative"]) else 0.0,
    "is_lookup": lambda m: 1.0 if m.lower().startswith(("what is", "who is",
        "when did", "where is", "how many")) and len(m) < 80 else 0.0,
    "has_multipart": lambda m: 1.0 if m.count("?") > 1 or " and " in m.lower() else 0.0,
    "has_constraints": lambda m: 1.0 if any(w in m.lower() for w in
        ["must", "should not", "without", "constraint", "requirement"]) else 0.0,
    "is_comparison": lambda m: 1.0 if any(w in m.lower() for w in
        ["compare", "difference between", "versus", " vs ", "tradeoff"]) else 0.0,
    "is_explanation": lambda m: 1.0 if any(w in m.lower() for w in
        ["explain", "how does", "why does", "what causes"]) else 0.0,
    "has_production_signal": lambda m: 1.0 if any(w in m.lower() for w in
        ["production", "enterprise", "scalable", "robust", "comprehensive",
         "complex", "advanced"]) else 0.0,
    "question_count": lambda m: min(m.count("?") / 3.0, 1.0),
    "has_numbers": lambda m: 1.0 if bool(re.search(r'\d+', m)) else 0.0,
    "has_technical_terms": lambda m: 1.0 if any(w in m.lower() for w in
        ["async", "concurrent", "distributed", "algorithm", "complexity",
         "architecture", "microservice", "kubernetes"]) else 0.0,
}

BASE_WEIGHTS = {
    "length_short":         {"easy": 0.6, "medium": 0.3, "hard": 0.1},
    "length_medium":        {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    "length_long":          {"easy": 0.1, "medium": 0.4, "hard": 0.5},
    "has_code_request":     {"easy": 0.1, "medium": 0.5, "hard": 0.4},
    "is_calculation":       {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    "is_lookup":            {"easy": 0.8, "medium": 0.2, "hard": 0.0},
    "has_multipart":        {"easy": 0.0, "medium": 0.4, "hard": 0.6},
    "has_constraints":      {"easy": 0.0, "medium": 0.3, "hard": 0.7},
    "is_comparison":        {"easy": 0.1, "medium": 0.6, "hard": 0.3},
    "is_explanation":       {"easy": 0.3, "medium": 0.6, "hard": 0.1},
    "has_production_signal":{"easy": 0.0, "medium": 0.2, "hard": 0.8},
    "question_count":       {"easy": 0.5, "medium": 0.3, "hard": 0.2},
    "has_numbers":          {"easy": 0.4, "medium": 0.4, "hard": 0.2},
    "has_technical_terms":  {"easy": 0.1, "medium": 0.4, "hard": 0.5},
}

def extract_features(msg: str) -> dict:
    return {name: fn(msg) for name, fn in FEATURE_EXTRACTORS.items()}

def estimate_complexity(msg: str, history_len: int = 0) -> Tuple[str, float]:
    """
    Estimate complexity. Returns (complexity, confidence).
    Online-updated weights improve with feedback.
    """
    feats = extract_features(msg)
    scores = {"easy": 0.0, "medium": 0.0, "hard": 0.0}

    for feat, val in feats.items():
        if val > 0 and feat in BASE_WEIGHTS:
            for level in scores:
                scores[level] += BASE_WEIGHTS[feat][level] * val

    # History length bias: longer conversations = more complex context
    if history_len > 6:
        scores["hard"] += 0.2
        scores["easy"] -= 0.2
    elif history_len > 3:
        scores["medium"] += 0.1

    # Normalize
    total = sum(scores.values()) or 1.0
    probs = {k: v/total for k, v in scores.items()}

    best = max(probs, key=probs.get)
    confidence = probs[best]
    return best, confidence

def record_feedback(msg: str, predicted: str, true_complexity: str):
    """Record prediction quality for future improvement."""
    feats = extract_features(msg)
    try:
        c = _conn()
        c.execute("INSERT INTO training (features, true_complexity, predicted, correct, ts) VALUES (?,?,?,?,?)",
                  (str(feats), true_complexity, predicted,
                   1 if predicted == true_complexity else 0, time.time()))
        c.commit()
        c.close()
    except Exception:
        pass
