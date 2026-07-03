"""
Pure-Python Knowledge Graph — Extracts and stores (Subject, Predicate, Object) triples.
Upgraded: Created the missing file! Gives the AI true relational memory.
"""
import sqlite3, re, time, os, threading
from collections import defaultdict

DB = os.path.expanduser("~/eliteomni_graph.db")
_lock = threading.Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS triples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT, predicate TEXT, object TEXT,
        ts REAL DEFAULT CURRENT_TIMESTAMP
    )''')
    con.execute("CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_object ON triples(object)")
    con.commit(); con.close()
_init()

# Heuristic patterns for triple extraction
PATTERNS = [
    (r"(\b[A-Z][a-z]+\b)\s+(?:is|are)\s+(?:a|an)\s+(.+?)(?:\.|$)", "is_a"),
    (r"(\b[A-Z][a-z]+\b)\s+(?:works at|employed at)\s+(.+?)(?:\.|$)", "works_at"),
    (r"(\b[A-Z][a-z]+\b)\s+(?:created|built|wrote|developed)\s+(.+?)(?:\.|$)", "created"),
    (r"(\b[A-Z][a-z]+\b)\s+(?:owns|founded)\s+(.+?)(?:\.|$)", "owns"),
    (r"(\b[A-Z][a-z]+\b)\s+(?:uses|prefers|likes)\s+(.+?)(?:\.|$)", "uses"),
]

def extract_and_store(text: str):
    """Extracts triples from text and stores them."""
    extracted = []
    for pattern, predicate in PATTERNS:
        matches = re.findall(pattern, text)
        for subj, obj in matches:
            subj, obj = subj.strip(), obj.strip()
            if len(subj) > 1 and len(obj) > 1:
                extracted.append((subj, predicate, obj))
    
    if not extracted: return
    try:
        with _lock:
            con = sqlite3.connect(DB)
            for subj, pred, obj in extracted:
                con.execute("INSERT INTO triples (subject, predicate, object) VALUES (?,?,?)", (subj, pred, obj))
            con.commit(); con.close()
    except Exception:
        pass

def query_graph(entity: str) -> list:
    """Retrieves all relationships for a given entity."""
    try:
        with _lock:
            con = sqlite3.connect(DB)
            rows = con.execute(
                "SELECT predicate, object FROM triples WHERE subject LIKE ? ORDER BY ts DESC LIMIT 10",
                (f"%{entity}%",)
            ).fetchall()
            con.close()
        return [{"predicate": r[0], "object": r[1]} for r in rows]
    except Exception:
        return []

def get_graph_context(entities: list) -> str:
    """Builds context string for prompt injection."""
    if not entities: return ""
    parts = []
    for ent in entities[:3]:
        rels = query_graph(ent)
        if rels:
            parts.append(f"Known facts about {ent}: " + "; ".join(f"{r['predicate']} {r['object']}" for r in rels[:3]))
    return "[KNOWLEDGE GRAPH]\n" + "\n".join(parts) + "\n[/KNOWLEDGE GRAPH]" if parts else ""
