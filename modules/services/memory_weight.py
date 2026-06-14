
"""
Fei-Fei Li: Not all memories are equal. ImageNet succeeded because
we weighted and curated data obsessively. This module scores memories
by importance so the retrieval system surfaces what actually matters.
"""
import re
import math
from datetime import datetime, timedelta

# Importance signals
HIGH_IMPORTANCE = [
    "decided", "confirmed", "agreed", "will use", "going with",
    "important", "critical", "must", "need to", "deadline",
    "allergic", "medical", "diagnosis", "emergency",
    "password", "key", "secret", "private",
    "hired", "fired", "promoted", "married", "divorced",
]

LOW_IMPORTANCE = [
    "maybe", "perhaps", "thinking about", "not sure",
    "just wondering", "random", "nevermind", "forget it",
    "lol", "haha", "thanks", "ok", "sure", "yes", "no",
]

def score_memory_importance(text: str, role: str = "user",
                             recency_hours: float = 0) -> float:
    """
    Score a memory 0.0-1.0 for importance.
    High score = retrieve often. Low score = let decay.
    """
    t = text.lower()
    score = 0.3  # baseline

    # Role weight: user statements > AI responses
    if role == "user":
        score += 0.1
    
    # Length heuristic: longer = more substantive
    word_count = len(text.split())
    score += min(word_count / 200, 0.15)

    # High importance signals
    hi_count = sum(1 for w in HIGH_IMPORTANCE if w in t)
    score += min(hi_count * 0.08, 0.25)

    # Low importance signals
    lo_count = sum(1 for w in LOW_IMPORTANCE if w in t)
    score -= min(lo_count * 0.05, 0.15)

    # Specific facts boost (numbers, names, dates)
    facts = len(re.findall(r"\b\d+\b|\b[A-Z][a-z]+\b", text))
    score += min(facts * 0.01, 0.10)

    # Recency decay: memories fade over time (Ebbinghaus curve)
    if recency_hours > 0:
        decay = math.exp(-recency_hours / (24 * 7))  # half-life ~1 week
        score *= (0.5 + 0.5 * decay)

    return round(max(0.0, min(1.0, score)), 3)


def structure_extraction(text: str) -> dict:
    """
    Fei-Fei Li: Extract structure from flat text.
    Identify what type of content this is before storing.
    """
    t = text.lower().strip()

    # Detect content type
    if re.search(r"```|def |class |import |function |{|}|<[a-z]+>", text):
        content_type = "code"
    elif re.search(r"\b\d{4}-\d{2}-\d{2}|\bjanuary|\bfebruary|\bmarch|\bapril|"
                   r"\bmay|\bjune|\bjuly|\baugust|\bseptember|\boctober|"
                   r"\bnovember|\bdecember\b", t):
        content_type = "temporal"
    elif re.search(r"\b(i am|i'm|my name|i work|i live|i have|i've been)\b", t):
        content_type = "personal_fact"
    elif re.search(r"\b(decided|will|going to|plan to|confirmed)\b", t):
        content_type = "decision"
    elif re.search(r"\?$|^(what|how|why|when|where|who)\b", t):
        content_type = "question"
    elif re.search(r"\b(error|bug|fix|broken|issue|problem|crash)\b", t):
        content_type = "problem"
    else:
        content_type = "general"

    # Extract key entities
    entities = list(set(re.findall(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*\b", text)))[:6]

    # Extract key numbers
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:%|ms|GB|MB|K|M|B)?\b", text)[:5]

    return {
        "type":     content_type,
        "entities": entities,
        "numbers":  numbers,
        "importance": score_memory_importance(text),
    }


def weighted_memory_retrieve(memories: list, query: str,
                              top_k: int = 6) -> list:
    """
    Retrieve memories ranked by relevance * importance.
    Not just recency — what matters most for this query.
    """
    if not memories:
        return []

    query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    scored = []

    for mem in memories:
        text = mem.get("content", mem) if isinstance(mem, dict) else str(mem)
        importance = mem.get("importance", 0.5) if isinstance(mem, dict) else 0.5

        # Relevance: word overlap with query
        mem_words = set(re.findall(r"\b\w{3,}\b", text.lower()))
        overlap = len(query_words & mem_words) / max(len(query_words), 1)

        # Final score: relevance * importance
        final = overlap * 0.6 + importance * 0.4
        scored.append((final, text))

    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:top_k]]
