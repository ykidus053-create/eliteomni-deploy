import sqlite3, json, time, hashlib, re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class MemoryType(Enum):
    FACT = "fact"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    GOAL = "goal"
    WORLD = "world"

class MemoryTier(Enum):
    WORKING = "working"
    SHORT = "short"
    LONG = "long"

@dataclass
class MemoryEntry:
    id: str
    content: str
    memory_type: MemoryType
    tier: MemoryTier
    confidence: float
    source: str
    provenance: List[str]
    contradicts: List[str]
    created_ts: float
    accessed_ts: float
    access_count: int
    embedding: Optional[List[float]] = None

class UnifiedMemory:
    def __init__(self, db_path: str, embedder=None):
        self.db_path = db_path
        self.embedder = embedder
        self._init_schema()

    def _init_schema(self):
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, memory_type TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'working', confidence REAL NOT NULL DEFAULT 0.7,
                source TEXT NOT NULL DEFAULT 'inferred', provenance TEXT NOT NULL DEFAULT '[]',
                contradicts TEXT NOT NULL DEFAULT '[]', created_ts REAL NOT NULL,
                accessed_ts REAL NOT NULL, access_count INTEGER NOT NULL DEFAULT 0, embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_type_tier ON memories(memory_type, tier);
            CREATE INDEX IF NOT EXISTS idx_confidence ON memories(confidence DESC);
        """)
        con.commit()
        con.close()

    def save(self, content: str, memory_type: MemoryType, confidence: float = 0.7, source: str = "inferred") -> str:
        memory_id = hashlib.sha256(f"{memory_type.value}:{content[:100]}".encode()).hexdigest()[:16]
        contradictions = self._detect_contradictions(content, memory_type)
        
        entry = MemoryEntry(
            id=memory_id, content=content, memory_type=memory_type, tier=MemoryTier.WORKING,
            confidence=confidence, source=source, provenance=[], contradicts=[c.id for c in contradictions],
            created_ts=time.time(), accessed_ts=time.time(), access_count=0
        )
        
        con = sqlite3.connect(self.db_path)
        con.execute("""INSERT OR REPLACE INTO memories
            (id, content, memory_type, tier, confidence, source, provenance, contradicts, created_ts, accessed_ts, access_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (entry.id, entry.content, entry.memory_type.value, entry.tier.value, entry.confidence, entry.source,
             json.dumps(entry.provenance), json.dumps(entry.contradicts), entry.created_ts, entry.accessed_ts, entry.access_count))
        con.commit()
        con.close()
        return memory_id

    def retrieve(self, query: str, k: int = 6, min_confidence: float = 0.5) -> List[MemoryEntry]:
        con = sqlite3.connect(self.db_path)
        working = con.execute("SELECT * FROM memories WHERE tier='working' AND confidence >= ? ORDER BY accessed_ts DESC LIMIT ?", (min_confidence, k // 2)).fetchall()
        keywords = query.lower().split()[:5]
        keyword_filter = " OR ".join(f"LOWER(content) LIKE '%{kw}%'" for kw in keywords)
        other = con.execute(f"SELECT * FROM memories WHERE tier != 'working' AND confidence >= ? AND ({keyword_filter}) ORDER BY confidence DESC, accessed_ts DESC LIMIT ?", (min_confidence, k)).fetchall() if keyword_filter else []
        con.close()
        
        results, seen_ids = [], set()
        for row in list(working) + list(other):
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                results.append(self._row_to_entry(row))
        self._update_access(seen_ids)
        return results[:k]

    def build_context_string(self, query: str = "") -> str:
        memories = self.retrieve(query, k=8)
        if not memories: return ""
        sections = {MemoryType.FACT: [], MemoryType.GOAL: [], MemoryType.EPISODIC: []}
        for m in memories:
            if m.memory_type in sections: sections[m.memory_type].append(m.content)
        parts = ["[MEMORY CONTEXT]"]
        if sections[MemoryType.FACT]: parts.append("Facts: " + " | ".join(sections[MemoryType.FACT][:4]))
        if sections[MemoryType.GOAL]: parts.append("Active goals: " + " | ".join(sections[MemoryType.GOAL][:2]))
        if sections[MemoryType.EPISODIC]: parts.append("Recent context: " + sections[MemoryType.EPISODIC][0][:200])
        parts.append("[/MEMORY CONTEXT]")
        return "\n".join(parts)

    def _detect_contradictions(self, content: str, memory_type: MemoryType) -> List[MemoryEntry]:
        """Upgraded: Implemented contradiction detection using keyword overlap and negation."""
        if memory_type != MemoryType.FACT: return []
        con = sqlite3.connect(self.db_path)
        rows = con.execute("SELECT id, content FROM memories WHERE memory_type='fact'").fetchall()
        con.close()
        
        conflicts = []
        content_lower = content.lower()
        content_words = set(re.findall(r'\b\w{4,}\b', content_lower))
        negation_words = ["not", "isn't", "wasn't", "never", "don't", "doesn't", "instead"]
        has_negation = any(neg in content_lower for neg in negation_words)
        
        for r_id, r_content in rows:
            r_lower = r_content.lower()
            r_words = set(re.findall(r'\b\w{4,}\b', r_lower))
            r_has_negation = any(neg in r_lower for neg in negation_words)
            
            # If high word overlap but differing negation status, it's a conflict
            overlap = len(content_words & r_words)
            if overlap >= 2 and has_negation != r_has_negation:
                conflicts.append(MemoryEntry(
                    id=r_id, content=r_content, memory_type=memory_type, tier=MemoryTier.WORKING,
                    confidence=0.7, source="inferred", provenance=[], contradicts=[], 
                    created_ts=0, accessed_ts=0, access_count=0
                ))
        return conflicts

    def _row_to_entry(self, row) -> MemoryEntry:
        return MemoryEntry(
            id=row[0], content=row[1], memory_type=MemoryType(row[2]), tier=MemoryTier(row[3]),
            confidence=row[4], source=row[5], provenance=json.loads(row[6]), contradicts=json.loads(row[7]),
            created_ts=row[8], accessed_ts=row[9], access_count=row[10]
        )

    def _update_access(self, ids: set):
        if not ids: return
        con = sqlite3.connect(self.db_path)
        placeholders = ",".join("?" * len(ids))
        con.execute(f"UPDATE memories SET accessed_ts=?, access_count=access_count+1 WHERE id IN ({placeholders})", [time.time()] + list(ids))
        con.commit()
        con.close()

    def promote_tiers(self):
        con = sqlite3.connect(self.db_path)
        con.execute("UPDATE memories SET tier='short' WHERE tier='working' AND access_count >= 3 AND created_ts < ?", (time.time() - 3600,))
        con.execute("UPDATE memories SET tier='long' WHERE tier='short' AND access_count >= 10 AND created_ts < ?", (time.time() - 86400,))
        con.commit()
        con.close()
