import sqlite3, json, time, re, os, threading
from dataclasses import dataclass, field
from typing import Dict, Any

DB = os.path.expanduser("~/eliteomni_world.db")
_lock = threading.Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS world_state (
        key TEXT PRIMARY KEY, value TEXT NOT NULL,
        confidence REAL DEFAULT 0.7, source TEXT DEFAULT 'inferred', updated_ts REAL)''')
    con.commit()
    con.close()

_init()

def set_state(key, value, confidence=0.7, source="inferred"):
    try:
        with _lock:
            con = sqlite3.connect(DB)
            con.execute("INSERT OR REPLACE INTO world_state VALUES (?,?,?,?,?)",
                        (key, json.dumps(value), confidence, source, time.time()))
            con.commit()
            con.close()
    except Exception: pass

def get_state(key=None):
    try:
        with _lock:
            con = sqlite3.connect(DB)
            if key:
                row = con.execute("SELECT value FROM world_state WHERE key=?", (key,)).fetchone()
                con.close()
                return json.loads(row[0]) if row else None
            rows = con.execute("SELECT key, value, confidence FROM world_state ORDER BY updated_ts DESC LIMIT 20").fetchall()
            con.close()
            return {r[0]: {"value": json.loads(r[1]), "confidence": r[2]} for r in rows}
    except Exception:
        return {} if key is None else None

def update_from_conversation(user_msg, response, session_id="default"):
    """Upgraded: Extracts OS, Language, and Project context dynamically."""
    m_lower = user_msg.lower()
    
    name_m = re.search(r"my name is (\w+)", user_msg, re.IGNORECASE)
    if name_m: set_state("user_name", name_m.group(1), confidence=0.9, source="user_statement")
    
    role_m = re.search(r"I(?:'m| am) (?:a |an )?(\w+)", user_msg, re.IGNORECASE)
    if role_m: set_state("user_type", role_m.group(1), confidence=0.8, source="user_statement")
        
    # Upgraded: Environment detection
    os_m = re.search(r"(?:on|using|running)\s+(ubuntu|windows|mac|linux|arch|debian)", m_lower)
    if os_m: set_state("user_os", os_m.group(1), confidence=0.9, source="user_statement")
        
    lang_m = re.search(r"(?:coding|writing|working)\s+in\s+(python|javascript|typescript|c\+\+|rust|go|java)", m_lower)
    if lang_m: set_state("user_language", lang_m.group(1), confidence=0.9, source="user_statement")

def get_world_model_context(session_id="", user_msg=""):
    state = get_state()
    if not state: return ""
    relevant = [k + ": " + str(v["value"]) for k, v in list(state.items())[:6] if v.get("confidence", 0) >= 0.6]
    if not relevant: return ""
    return "[WORLD STATE]\n" + "\n".join(relevant) + "\n[/WORLD STATE]"

@dataclass
class UserModel:
    preferences: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    def to_context_str(self): return get_world_model_context(self.session_id)

@dataclass
class ConversationState:
    session_id: str = ""
    turn_count: int = 0
    topic: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    def update(self, key, value): self.context[key] = value
    def to_dict(self): return {"session_id": self.session_id, "turn_count": self.turn_count, "topic": self.topic}
