
import json, os, re
from datetime import datetime

PROFILE_PATH = os.path.expanduser("~/.eliteomni_profile.json")

def _load_profile() -> dict:
    try:
        return json.load(open(PROFILE_PATH))
    except Exception:
        return {"name": None, "facts": [], "emotional_events": [], "preferences": {}, "last_seen": None}

def _save_profile(p: dict):
    p["last_seen"] = datetime.now().isoformat()
    json.dump(p, open(PROFILE_PATH, "w"), indent=2)

def profile_extract_and_save(msg: str, response: str):
    """Auto-extract identity facts from conversation and persist them."""
    p = _load_profile()

    # Name detection
    name_match = re.search(r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)", msg, re.IGNORECASE)
    if name_match and not p["name"]:
        p["name"] = name_match.group(1)

    # Fact extraction triggers
    fact_triggers = [
        r"i(?:'m| am) a ([^.!?]{3,60})",
        r"i work (?:as |at |for )([^.!?]{3,60})",
        r"i(?:'m| am) (\d+ years? old)",
        r"i live in ([^.!?]{3,40})",
        r"i(?:'m| am) allergic to ([^.!?]{3,40})",
        r"i(?:'m| am) learning ([^.!?]{3,40})",
        r"i(?:'ve| have) been ([^.!?]{3,60})",
        r"my (?:kid|son|daughter|wife|husband|partner) ([^.!?]{3,60})",
    ]
    for pattern in fact_triggers:
        m = re.search(pattern, msg, re.IGNORECASE)
        if m:
            fact = m.group(0).strip()
            if fact not in p["facts"]:
                p["facts"].append(fact)
                p["facts"] = p["facts"][-40:]  # keep last 40 facts

    # Emotional event detection
    emotional_triggers = ["struggling", "venting", "breakup", "crisis", "anxious",
                          "depressed", "stressed", "overwhelmed", "grieving", "fired",
                          "divorced", "sick", "diagnosed", "lost my", "failed"]
    if any(t in msg.lower() for t in emotional_triggers):
        event = {"date": datetime.now().strftime("%Y-%m-%d"), "summary": msg[:200]}
        p["emotional_events"].append(event)
        p["emotional_events"] = p["emotional_events"][-10:]

    _save_profile(p)

def profile_get_context() -> str:
    """Return a system prompt injection summarizing what we know about the user."""
    p = _load_profile()
    parts = []

    if p.get("name"):
        parts.append(f"The user's name is {p['name']}.")

    if p.get("last_seen"):
        parts.append(f"They were last seen: {p['last_seen'][:10]}.")

    if p.get("facts"):
        parts.append("Known facts about the user:\n" +
                     "\n".join(f"  - {f}" for f in p["facts"][-10:]))

    if p.get("emotional_events"):
        recent = p["emotional_events"][-3:]
        parts.append("Recent emotional context:\n" +
                     "\n".join(f"  - [{e['date']}] {e['summary'][:100]}" for e in recent))

    if not parts:
        return ""

    return "[USER MEMORY]\n" + "\n".join(parts) + "\n[/USER MEMORY]"
