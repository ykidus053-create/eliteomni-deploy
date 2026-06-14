
import json, os
from datetime import datetime

SUMMARY_PATH = os.path.expanduser("~/.eliteomni_chat_log.json")

def _load_log() -> list:
    try:
        return json.load(open(SUMMARY_PATH))
    except Exception:
        return []

def chat_log_save(user_msg: str, ai_response: str, skill: str):
    """Append a summarized exchange to the persistent chat log."""
    log = _load_log()
    log.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "skill": skill,
        "user": user_msg[:300],
        "ai": ai_response[:400],
    })
    log = log[-200:]  # keep last 200 exchanges
    json.dump(log, open(SUMMARY_PATH, "w"), indent=2)

def chat_log_search(query: str, top_k: int = 5) -> list:
    """Simple keyword search over past chat log."""
    log = _load_log()
    query_words = set(query.lower().split())
    scored = []
    for entry in log:
        text = (entry.get("user","") + " " + entry.get("ai","")).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, entry))
    scored.sort(reverse=True)
    return [e for _, e in scored[:top_k]]

def chat_log_get_context(query: str) -> str:
    """Return relevant past exchanges as a system prompt injection."""
    hits = chat_log_search(query, top_k=4)
    if not hits:
        return ""
    lines = []
    for h in hits:
        lines.append(f"  [{h['date']}] ({h['skill']}) You asked: {h['user'][:120]}")
        lines.append(f"    I answered: {h['ai'][:150]}")
    return "[RELEVANT PAST CONVERSATIONS]\n" + "\n".join(lines) + "\n[/RELEVANT PAST CONVERSATIONS]"
