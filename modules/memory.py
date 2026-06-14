# modules/memory.py — SQLite persistent memory
import sqlite3, time, os

DB_PATH = os.environ.get("MEMORY_DB", "/home/kidus/eliteomni_memory.db")

def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS memory
                   (id INTEGER PRIMARY KEY, text TEXT, ts REAL, skill TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS episodic
                   (id INTEGER PRIMARY KEY, summary TEXT, ts REAL)""")
    con.commit()
    return con

def mem_store(text: str, skill: str = "general"):
    with _conn() as con:
        con.execute("INSERT INTO memory (text, ts, skill) VALUES (?,?,?)",
                    (text[:2000], time.time(), skill))

def mem_get(limit: int = 10, skill: str = None):
    with _conn() as con:
        if skill:
            rows = con.execute(
                "SELECT text FROM memory WHERE skill=? ORDER BY ts DESC LIMIT ?",
                (skill, limit)).fetchall()
        else:
            rows = con.execute(
                "SELECT text FROM memory ORDER BY ts DESC LIMIT ?",
                (limit,)).fetchall()
    return [r[0] for r in rows]

def episodic_get(limit: int = 5):
    with _conn() as con:
        rows = con.execute(
            "SELECT summary FROM episodic ORDER BY ts DESC LIMIT ?",
            (limit,)).fetchall()
    return [r[0] for r in rows]

def episodic_store(summary: str):
    with _conn() as con:
        con.execute("INSERT INTO episodic (summary, ts) VALUES (?,?)",
                    (summary[:2000], time.time()))
