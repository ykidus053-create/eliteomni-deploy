"""
Cross-Chat Persistent Memory
Stores facts, preferences, corrections, and skills that persist across ALL sessions.
Three memory types:
  - Semantic (embedding-indexed facts)
  - Episodic (conversation summaries)  
  - Procedural (learned skills/preferences)

Auto-extracts on every conversation end.
Auto-injects relevant memories at conversation start.
"""
import re, time, json, sqlite3, threading, hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path

DB = Path.home() / "eliteomni_persistent_memory.db"
_lock = threading.Lock()

def _conn():
    c = sqlite3.connect(str(DB))
    c.executescript("""
    CREATE TABLE IF NOT EXISTS semantic_memory (
        id TEXT PRIMARY KEY, content TEXT, category TEXT,
        importance REAL, access_count INTEGER DEFAULT 0,
        created REAL, last_accessed REAL, tags TEXT);
    CREATE TABLE IF NOT EXISTS episodic_memory (
        id TEXT PRIMARY KEY, session_id TEXT,
        summary TEXT, key_facts TEXT,
        skill TEXT, outcome_quality REAL,
        created REAL, turn_count INTEGER);
    CREATE TABLE IF NOT EXISTS procedural_memory (
        id TEXT PRIMARY KEY, skill TEXT, pattern TEXT,
        strategy TEXT, success_rate REAL,
        sample_count INTEGER, last_updated REAL);
    CREATE TABLE IF NOT EXISTS memory_index (
        word TEXT, memory_id TEXT, memory_type TEXT,
        weight REAL,
        PRIMARY KEY(word, memory_id));
    """)
    c.commit()
    return c

EXTRACTION_PATTERNS = {
    "preference": [
        r"(?:I prefer|I like|I want|always use|please use|use only)[:\s]+(.{5,80})",
        r"(?:my preference is|I always)[:\s]+(.{5,80})",
    ],
    "fact": [
        r"(?:I am|I'm|my name is|I work at|I work on)[:\s]+(.{5,80})",
        r"(?:I use|I'm using|my (?:stack|language|framework) is)[:\s]+(.{5,60})",
    ],
    "correction": [
        r"(?:actually|no,|wrong|that's not)[,\s]+(?:the (?:correct|right|actual) (?:answer|rule|fact) is[:\s]+)?(.{10,100})",
    ],
    "constraint": [
        r"(?:never|always|must|cannot|only)[:\s]+(?:in my (?:project|code|system)[,\s]+)?(.{10,80})",
    ],
    "skill_preference": [
        r"(?:I prefer to use|in my projects? I use|my (?:preferred|default))[:\s]+(.{5,60})",
    ],
}

def extract_memories_from_conversation(
        history: List[Dict], skill: str = "general",
        outcome_quality: float = 0.7) -> List[Dict]:
    """Extract persistable memories from a completed conversation."""
    memories = []
    full_text = "\n".join(
        f"{m.get('role','?')}: {str(m.get('content',''))[:300]}"
        for m in history[-20:]
    )

    for category, patterns in EXTRACTION_PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, full_text, re.IGNORECASE):
                content = m.group(1).strip()
                if len(content) < 5 or len(content) > 200:
                    continue
                mem_id = hashlib.md5(
                    (category + content.lower()).encode()
                ).hexdigest()[:12]
                memories.append({
                    "id": mem_id,
                    "content": content,
                    "category": category,
                    "importance": _score_importance(category, content, outcome_quality),
                    "tags": _extract_tags(content),
                })

    return _deduplicate_memories(memories)

def _score_importance(category: str, content: str, outcome_quality: float) -> float:
    base = {"correction": 0.9, "constraint": 0.85, "preference": 0.8,
            "fact": 0.75, "skill_preference": 0.7}.get(category, 0.6)
    return min(1.0, base * outcome_quality + 0.1)

def _extract_tags(content: str) -> str:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all",
                 "can", "has", "have", "was", "with", "this", "that"}
    tags = [w for w in words if w not in stopwords][:6]
    return ",".join(tags)

def _deduplicate_memories(memories: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for m in memories:
        key = m["id"]
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result

def save_memories(memories: List[Dict], session_id: str = "default"):
    if not memories:
        return
    c = _conn()
    saved = 0
    for mem in memories:
        try:
            existing = c.execute(
                "SELECT importance, access_count FROM semantic_memory WHERE id=?",
                (mem["id"],)).fetchone()
            if existing:
                new_importance = max(existing[0], mem["importance"])
                c.execute("""UPDATE semantic_memory 
                    SET importance=?, access_count=access_count+1, last_accessed=?
                    WHERE id=?""",
                    (new_importance, time.time(), mem["id"]))
            else:
                c.execute("""INSERT INTO semantic_memory
                    (id, content, category, importance, created, last_accessed, tags)
                    VALUES (?,?,?,?,?,?,?)""",
                    (mem["id"], mem["content"], mem["category"],
                     mem["importance"], time.time(), time.time(), mem.get("tags","")))
                for word in mem.get("tags", "").split(","):
                    if word.strip():
                        c.execute("""INSERT OR REPLACE INTO memory_index
                            VALUES (?,?,?,?)""",
                            (word.strip(), mem["id"], "semantic", mem["importance"]))
                saved += 1
        except Exception as e:
            print(f"[PersistMem] save error: {e}")
    c.commit()
    if saved:
        print(f"[PersistMem] saved {saved} new memories")

def save_episode(session_id: str, history: List[Dict],
                 skill: str, quality: float):
    """Save a conversation episode summary."""
    if not history or len(history) < 2:
        return
    user_msgs = [m for m in history if m.get("role") == "user"]
    topics = " | ".join(
        str(m.get("content",""))[:50] for m in user_msgs[:3]
    )
    key_facts = extract_memories_from_conversation(history, skill, quality)
    facts_json = json.dumps([{"c": f["content"], "cat": f["category"]}
                              for f in key_facts[:5]])
    ep_id = hashlib.md5(f"{session_id}{time.time()}".encode()).hexdigest()[:12]
    c = _conn()
    try:
        c.execute("""INSERT OR REPLACE INTO episodic_memory
            (id, session_id, summary, key_facts, skill, outcome_quality, created, turn_count)
            VALUES (?,?,?,?,?,?,?,?)""",
            (ep_id, session_id, topics[:200], facts_json,
             skill, quality, time.time(), len(history)))
        c.commit()
    except Exception as e:
        print(f"[PersistMem] episode save: {e}")

def retrieve_relevant_memories(query: str, skill: str = "general",
                                top_k: int = 5) -> List[Dict]:
    """Retrieve memories relevant to the current query."""
    c = _conn()
    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all",
                 "can", "has", "have", "was", "with", "this", "that", "what",
                 "how", "why", "when", "where", "which", "who"}
    query_words -= stopwords

    if not query_words:
        rows = c.execute("""SELECT content, category, importance FROM semantic_memory
            ORDER BY last_accessed DESC, importance DESC LIMIT ?""",
            (top_k,)).fetchall()
        return [{"content": r[0], "category": r[1], "importance": r[2]}
                for r in rows]

    placeholders = ",".join("?" * len(query_words))
    word_list = list(query_words)[:10]
    placeholders = ",".join("?" * len(word_list))

    rows = c.execute(f"""
        SELECT sm.content, sm.category, sm.importance,
               COUNT(mi.word) as word_hits,
               SUM(mi.weight) as relevance_score
        FROM semantic_memory sm
        JOIN memory_index mi ON sm.id = mi.memory_id
        WHERE mi.word IN ({placeholders})
        GROUP BY sm.id
        ORDER BY relevance_score DESC, sm.importance DESC
        LIMIT ?""",
        word_list + [top_k]).fetchall()

    if not rows:
        rows = c.execute("""SELECT content, category, importance, 0, importance
            FROM semantic_memory
            ORDER BY importance DESC, last_accessed DESC LIMIT ?""",
            (top_k,)).fetchall()

    c.execute("""UPDATE semantic_memory SET access_count=access_count+1,
        last_accessed=? WHERE content IN ({})""".format(
        ",".join("?" * len(rows))),
        [time.time()] + [r[0] for r in rows])
    c.commit()

    return [{"content": r[0], "category": r[1], "importance": r[2],
             "relevance": r[4]} for r in rows]

def build_memory_context(query: str, skill: str = "general") -> str:
    """Build context string from persistent memories for injection."""
    memories = retrieve_relevant_memories(query, skill, top_k=6)
    if not memories:
        return ""
    by_category = {}
    for m in memories:
        cat = m["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(m["content"])

    parts = ["<persistent_memory>"]
    priority_order = ["correction", "constraint", "preference",
                      "skill_preference", "fact"]
    for cat in priority_order:
        if cat in by_category:
            parts.append(f"[{cat.upper()}S]")
            for content in by_category[cat][:2]:
                parts.append(f"  - {content[:100]}")
    parts.append("</persistent_memory>")
    return "\n".join(parts)

def get_memory_stats() -> Dict:
    c = _conn()
    try:
        total = c.execute("SELECT COUNT(*) FROM semantic_memory").fetchone()[0]
        by_cat = c.execute("""SELECT category, COUNT(*) FROM semantic_memory
            GROUP BY category ORDER BY COUNT(*) DESC""").fetchall()
        episodes = c.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        return {"total_memories": total, "by_category": dict(by_cat),
                "episodes": episodes}
    except Exception:
        return {}

_auto_save_queue = []
_queue_lock = threading.Lock()

def queue_session_save(session_id: str, history: List[Dict],
                       skill: str, quality: float):
    """Non-blocking: queue a session for background memory extraction."""
    with _queue_lock:
        _auto_save_queue.append((session_id, history[:], skill, quality))

def _background_memory_worker():
    while True:
        time.sleep(3)
        with _queue_lock:
            if not _auto_save_queue:
                continue
            item = _auto_save_queue.pop(0)
        session_id, history, skill, quality = item
        try:
            memories = extract_memories_from_conversation(history, skill, quality)
            save_memories(memories, session_id)
            save_episode(session_id, history, skill, quality)
        except Exception as e:
            print(f"[PersistMem worker] {e}")

def start_memory_worker():
    t = threading.Thread(target=_background_memory_worker,
                         daemon=True, name="persistent_memory_worker")
    t.start()
    print("[PersistMem] background worker started")
    return t
