"""
Structured World Model — persistent graph of entities, relations, and beliefs.
Updated each turn. Read by planner, reasoner, and response generator.
This replaces flat context stuffing with a queryable structured representation.
"""
import json, time, sqlite3, threading
from typing import Optional
from pathlib import Path

DB = Path.home() / "eliteomni_world_model.db"
_lock = threading.Lock()

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY, type TEXT, attributes TEXT, updated REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS relations (
        subj TEXT, pred TEXT, obj TEXT, confidence REAL, updated REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS beliefs (
        key TEXT PRIMARY KEY, value TEXT, source TEXT,
        confidence REAL, updated REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS user_model (
        session TEXT, key TEXT, value TEXT, updated REAL,
        PRIMARY KEY(session, key))""")
    c.commit()
    return c

class WorldModel:
    def __init__(self, session_id: str = "default"):
        self.session = session_id
        self._db = _conn()

    def assert_entity(self, eid: str, etype: str, attrs: dict):
        with _lock:
            self._db.execute(
                "INSERT OR REPLACE INTO entities VALUES (?,?,?,?)",
                (eid, etype, json.dumps(attrs), time.time()))
            self._db.commit()

    def assert_relation(self, subj: str, pred: str, obj: str, confidence: float = 1.0):
        with _lock:
            self._db.execute(
                "INSERT OR REPLACE INTO relations VALUES (?,?,?,?,?)",
                (subj, pred, obj, confidence, time.time()))
            self._db.commit()

    def assert_belief(self, key: str, value: str, source: str = "inference",
                      confidence: float = 0.9):
        with _lock:
            self._db.execute(
                "INSERT OR REPLACE INTO beliefs VALUES (?,?,?,?,?)",
                (key, value, source, confidence, time.time()))
            self._db.commit()

    def get_belief(self, key: str) -> Optional[str]:
        row = self._db.execute(
            "SELECT value FROM beliefs WHERE key=? ORDER BY confidence DESC LIMIT 1",
            (key,)).fetchone()
        return row[0] if row else None

    def update_user_model(self, key: str, value: str):
        with _lock:
            self._db.execute(
                "INSERT OR REPLACE INTO user_model VALUES (?,?,?,?)",
                (self.session, key, value, time.time()))
            self._db.commit()

    def get_user_model(self) -> dict:
        rows = self._db.execute(
            "SELECT key, value FROM user_model WHERE session=?",
            (self.session,)).fetchall()
        return dict(rows)

    def get_context_snapshot(self) -> str:
        """Compact structured summary for injection into system prompt."""
        user = self.get_user_model()
        beliefs = self._db.execute(
            "SELECT key, value, confidence FROM beliefs ORDER BY confidence DESC LIMIT 8"
        ).fetchall()
        relations = self._db.execute(
            "SELECT subj, pred, obj FROM relations ORDER BY updated DESC LIMIT 6"
        ).fetchall()
        parts = []
        if user:
            parts.append("USER_MODEL: " + "; ".join(f"{k}={v}" for k,v in user.items()))
        if beliefs:
            parts.append("BELIEFS: " + "; ".join(
                f"{k}={v}(conf={c:.1f})" for k,v,c in beliefs))
        if relations:
            parts.append("RELATIONS: " + "; ".join(
                f"{s}-{p}->{o}" for s,p,o in relations))
        return "\n".join(parts) if parts else ""

    def extract_and_update(self, user_msg: str, response: str):
        """Auto-extract facts from conversation turn and update model."""
        import re
        # Extract expertise signals
        expertise_signals = {
            "expert": ["I work as", "I'm a", "as a developer", "as an engineer",
                       "my codebase", "our system", "production"],
            "intermediate": ["I'm learning", "I'm trying to understand", "how do I"],
            "beginner": ["what is", "explain", "I don't understand", "what does"]
        }
        for level, signals in expertise_signals.items():
            if any(s.lower() in user_msg.lower() for s in signals):
                self.update_user_model("expertise", level)
                break
        # Extract topic focus
        topics = re.findall(r'\b(Python|JavaScript|TypeScript|Rust|Go|SQL|FastAPI|'
                            r'React|Docker|Kubernetes|ML|AI|LLM|database|API|'
                            r'microservices|security|performance)\b', user_msg, re.I)
        if topics:
            self.update_user_model("current_topic", topics[0].lower())
        # Length preference inference
        if len(response) > 2000:
            self.assert_belief("user_prefers_detail", "true", "response_length", 0.6)
        # Domain focus
        if any(w in user_msg.lower() for w in ["code", "function", "class", "bug"]):
            self.assert_belief("session_domain", "software_engineering", "message_pattern", 0.85)

_world_models: dict = {}

def get_world_model(session_id: str = "default") -> WorldModel:
    if session_id not in _world_models:
        _world_models[session_id] = WorldModel(session_id)
    return _world_models[session_id]
