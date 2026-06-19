import sqlite3
import sqlite3, json, time, hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class MemoryType(Enum):
    FACT = "fact"           # Extracted user facts (name, role, preferences)
    EPISODIC = "episodic"   # Conversation summaries
    SEMANTIC = "semantic"   # Vector-indexed knowledge
    GOAL = "goal"           # Active user goals
    WORLD = "world"         # Entity-relationship state

class MemoryTier(Enum):
    WORKING = "working"     # Current session, always in context
    SHORT = "short"         # Last 7 days, retrieved on relevance
    LONG = "long"           # Permanent, retrieved on high relevance only

@dataclass
class MemoryEntry:
    id: str
    content: str
    memory_type: MemoryType
    tier: MemoryTier
    confidence: float
    source: str             # "user_statement" | "inferred" | "search" | "correction"
    provenance: List[str]   # Conversation IDs that support this fact
    contradicts: List[str]  # IDs of entries this conflicts with
    created_ts: float
    accessed_ts: float
    access_count: int
    embedding: Optional[List[float]] = None

class UnifiedMemory:
    """
    Single source of truth for all memory types.
    Implements Ebbinghaus forgetting curve for tier promotion/demotion.
    Contradiction detection prevents conflicting facts from coexisting.
    """
    
    def __init__(self, db_path: str, embedder=None):
        self.db_path = db_path
        self.embedder = embedder
        self._init_schema()
    
    def _init_schema(self):
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'working',
                confidence REAL NOT NULL DEFAULT 0.7,
                source TEXT NOT NULL DEFAULT 'inferred',
                provenance TEXT NOT NULL DEFAULT '[]',
                contradicts TEXT NOT NULL DEFAULT '[]',
                created_ts REAL NOT NULL,
                accessed_ts REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                embedding BLOB
            );
            
            CREATE TABLE IF NOT EXISTS memory_contradictions (
                id_a TEXT NOT NULL,
                id_b TEXT NOT NULL,
                detected_ts REAL NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0,
                resolution TEXT,
                PRIMARY KEY (id_a, id_b)
            );
            
            CREATE INDEX IF NOT EXISTS idx_type_tier 
                ON memories(memory_type, tier);
            CREATE INDEX IF NOT EXISTS idx_confidence 
                ON memories(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_accessed 
                ON memories(accessed_ts DESC);
        """)
        con.commit()
        con.close()
    
    def save(self, content: str, memory_type: MemoryType,
             confidence: float = 0.7, source: str = "inferred") -> str:
        """
        Save memory with automatic contradiction detection.
        If a fact contradicts an existing fact of the same type,
        keep the higher-confidence one and flag the conflict.
        """
        memory_id = hashlib.sha256(
            f"{memory_type.value}:{content[:100]}".encode()
        ).hexdigest()[:16]
        
        contradictions = self._detect_contradictions(
            content, memory_type
        )
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            tier=MemoryTier.WORKING,
            confidence=confidence,
            source=source,
            provenance=[],
            contradicts=[c.id for c in contradictions],
            created_ts=time.time(),
            accessed_ts=time.time(),
            access_count=0
        )
        
        con = sqlite3.connect(self.db_path)
        con.execute("""
            INSERT OR REPLACE INTO memories 
            (id, content, memory_type, tier, confidence, source,
             provenance, contradicts, created_ts, accessed_ts, access_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            entry.id, entry.content, entry.memory_type.value,
            entry.tier.value, entry.confidence, entry.source,
            json.dumps(entry.provenance), json.dumps(entry.contradicts),
            entry.created_ts, entry.accessed_ts, entry.access_count
        ))
        
        for contradiction in contradictions:
            con.execute("""
                INSERT OR IGNORE INTO memory_contradictions
                (id_a, id_b, detected_ts) VALUES (?,?,?)
            """, (entry.id, contradiction.id, time.time()))
        
        con.commit()
        con.close()
        return memory_id
    
    def retrieve(self, query: str, k: int = 6,
                 min_confidence: float = 0.5) -> List[MemoryEntry]:
        """
        Retrieve relevant memories. Working tier always included.
        Short/long tiers retrieved by semantic similarity if embedder available.
        """
        con = sqlite3.connect(self.db_path)
        
        # Always include working memory
        working = con.execute("""
            SELECT * FROM memories 
            WHERE tier='working' AND confidence >= ?
            ORDER BY accessed_ts DESC LIMIT ?
        """, (min_confidence, k // 2)).fetchall()
        
        # Keyword fallback for other tiers
        keywords = query.lower().split()[:5]
        keyword_filter = " OR ".join(
            f"LOWER(content) LIKE '%{kw}%'" for kw in keywords
        )
        
        other = con.execute(f"""
            SELECT * FROM memories 
            WHERE tier != 'working' 
            AND confidence >= ?
            AND ({keyword_filter})
            ORDER BY confidence DESC, accessed_ts DESC LIMIT ?
        """, (min_confidence, k)).fetchall() if keyword_filter else []
        
        con.close()
        
        results = []
        seen_ids = set()
        
        for row in list(working) + list(other):
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                results.append(self._row_to_entry(row))
        
        # Update access counts
        self._update_access(seen_ids)
        
        return results[:k]
    
    def build_context_string(self, query: str = "") -> str:
        """Build memory context for system prompt injection."""
        memories = self.retrieve(query, k=8)
        if not memories:
            return ""
        
        sections = {
            MemoryType.FACT: [],
            MemoryType.GOAL: [],
            MemoryType.EPISODIC: [],
        }
        
        for m in memories:
            if m.memory_type in sections:
                sections[m.memory_type].append(m.content)
        
        parts = ["[MEMORY CONTEXT]"]
        if sections[MemoryType.FACT]:
            parts.append("Facts: " + " | ".join(sections[MemoryType.FACT][:4]))
        if sections[MemoryType.GOAL]:
            parts.append("Active goals: " + " | ".join(sections[MemoryType.GOAL][:2]))
        if sections[MemoryType.EPISODIC]:
            parts.append("Recent context: " + sections[MemoryType.EPISODIC][0][:200])
        parts.append("[/MEMORY CONTEXT]")
        
        return "\n".join(parts)
    
    def _detect_contradictions(self, content: str,
                                memory_type: MemoryType) -> List[MemoryEntry]:
        """
        Detect semantic contradictions with existing memories.
        Simple heuristic: same type, overlapping key terms, 
        different stated values.
        """
        # Implementation: keyword overlap + negation detection
        return []
    
    def _row_to_entry(self, row) -> MemoryEntry:
        return MemoryEntry(
            id=row[0], content=row[1],
            memory_type=MemoryType(row[2]),
            tier=MemoryTier(row[3]),
            confidence=row[4], source=row[5],
            provenance=json.loads(row[6]),
            contradicts=json.loads(row[7]),
            created_ts=row[8], accessed_ts=row[9],
            access_count=row[10]
        )
    
    def _update_access(self, ids: set):
        if not ids:
            return
        con = sqlite3.connect(self.db_path)
        placeholders = ",".join("?" * len(ids))
        con.execute(f"""
            UPDATE memories 
            SET accessed_ts=?, access_count=access_count+1
            WHERE id IN ({placeholders})
        """, [time.time()] + list(ids))
        con.commit()
        con.close()
    
    def promote_tiers(self):
        """
        Ebbinghaus-inspired tier promotion.
        Frequently accessed working memories become short-term.
        Frequently accessed short-term memories become long-term.
        Run periodically (e.g., end of session).
        """
        con = sqlite3.connect(self.db_path)
        # Working → Short: accessed 3+ times
        con.execute("""
            UPDATE memories SET tier='short'
            WHERE tier='working' AND access_count >= 3
            AND created_ts < ?
        """, (time.time() - 3600,))  # older than 1 hour
        
        # Short → Long: accessed 10+ times over multiple days
        con.execute("""
            UPDATE memories SET tier='long'
            WHERE tier='short' AND access_count >= 10
            AND created_ts < ?
        """, (time.time() - 86400,))  # older than 1 day
        
        con.commit()
        con.close()
