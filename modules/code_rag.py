"""
code_rag.py
Not pattern matching — first-principles engineering knowledge.
Teaches the model HOW to think about any problem correctly.
"""

# ── FIRST PRINCIPLES (apply to every coding task) ────────────────────────────

PRINCIPLES = """
ENGINEERING FIRST PRINCIPLES — apply these to EVERY coding task:

═══════════════════════════════════════════════════════
1. DATA STRUCTURE SELECTION (most code is wrong here)
═══════════════════════════════════════════════════════
Ask before writing a single line:
  - What is the dominant operation? (lookup / insert / delete / iterate / range-query)
  - What is the required complexity for that operation?
  - Which data structure gives that complexity?

Lookup by key          → dict (O(1))
Ordered + range query  → SortedDict/SortedList (O(log n)) — NOT a heap
Priority + cancel      → SortedDict keyed by priority — NOT a heap (heap can't cancel)
FIFO within same key   → dict[key, deque] — NOT dict[key, list]
Graph traversal        → adjacency dict + visited set — NOT adjacency matrix
Prefix search          → Trie — NOT list + startswith
Membership test        → set — NOT list (O(1) vs O(n))
Sliding window max     → deque (monotonic) — NOT sorted list rebuilt each step

HEAP LIMITATION (most LLMs get this wrong):
  heapq CANNOT efficiently cancel/modify elements — O(n) remove + O(n) heapify
  Use tombstone pattern OR SortedList for cancellable priority queues
  Rule: if your problem needs cancel/modify + priority → use SortedList, not heap

═══════════════════════════════════════════════════════
2. CONCURRENCY RULES (deadlocks kill production systems)
═══════════════════════════════════════════════════════
  - NEVER call a method that acquires a lock from inside a locked section
  - NEVER mutate a collection while iterating it — copy keys first
  - NEVER use += on shared state without a lock
  - ALWAYS acquire locks in the same order everywhere (lock ordering)
  - PREFER asyncio.Queue over threading.Queue in async code
  - Deadlock pattern to avoid:
      def method_a(self):
          with self.lock:        # acquires lock
              self.method_b()    # method_b also acquires self.lock → DEADLOCK
      Fix: extract shared logic to _method_b_unlocked(), call that instead

═══════════════════════════════════════════════════════
3. STATE CONSISTENCY (subtle bugs hide here)
═══════════════════════════════════════════════════════
  - If you maintain TWO structures for the same data (e.g. dict + sorted list),
    EVERY mutation must update BOTH atomically — draw a sync table
  - Sorted collections: mutating a field used as sort key corrupts the sort order
    Fix: remove → mutate → re-insert, never mutate in place
  - Lazy deletion (tombstone): mark as deleted in O(1), clean up on access
    Use when: removal from middle of heap/sorted structure is needed
  - Invariant check: after every operation, what must be true?
    State it explicitly: "order_map and buy_orders contain exactly the same active orders"

═══════════════════════════════════════════════════════
4. API / INTERFACE DESIGN
═══════════════════════════════════════════════════════
  - Methods that modify state should return what changed (trades, affected IDs)
  - Never return None when you can return [] or {} — callers can't iterate None
  - Raise specific exceptions with context: ValueError("order B1 not found in book")
  - Validate inputs at public API boundary, trust internally
  - Idempotency: can this be called twice safely? Document if not.

═══════════════════════════════════════════════════════
5. ERROR HANDLING
═══════════════════════════════════════════════════════
  - Catch the SPECIFIC exception that can occur, not Exception
  - Every except must either: re-raise with context, log + return typed fallback,
    or handle completely. Never silently swallow.
  - Resource cleanup: use context managers (with), not try/finally manually
  - Partial failure: if step 2 of 3 fails, undo step 1 (rollback pattern)

═══════════════════════════════════════════════════════
6. PERFORMANCE REASONING
═══════════════════════════════════════════════════════
  - State the complexity of every public method before implementing
  - If O(n) appears in a hot path — it will be a bottleneck at scale
  - Memory: every object has a cost — use __slots__ for high-frequency objects
  - I/O: never do blocking I/O in async context (requests in async def = blocks loop)
  - Batch: N individual DB queries in a loop → 1 batch query

═══════════════════════════════════════════════════════
7. CORRECTNESS BEFORE CLEVERNESS
═══════════════════════════════════════════════════════
  - Trace through your algorithm with a concrete example BEFORE coding
  - Draw the state after each operation: what does the data structure look like?
  - Edge cases to always check:
      empty input, single element, duplicate keys, zero/negative values,
      concurrent modification, partial fill, exact match on boundary
  - If the domain has a canonical algorithm (Dijkstra, RAFT, price-time priority)
    use it exactly — do not invent a "simpler version" that is subtly wrong

═══════════════════════════════════════════════════════
8. DOMAIN PROTOCOL (look this up before implementing)
═══════════════════════════════════════════════════════
  Before implementing any domain-specific system, answer:
  - What is the STANDARD algorithm practitioners use? (name it)
  - What invariants must ALWAYS hold in this domain?
  - What are the domain-specific failure modes that don't exist in generic CS?
  - Would a senior engineer at a company running this in production accept this?

  Examples of domain invariants:
  - Order book: bid prices must always be < ask prices (no crossed book)
  - Database: WAL must be fsynced before acknowledging write
  - Distributed: never assume message delivery order without explicit sequencing
  - Compiler: every identifier must be in scope before use
"""

# ── Domain-specific additions keyed by topic ─────────────────────────────────
DOMAIN_KNOWLEDGE = {
    "order book|matching engine|exchange|bid|ask|trade": """
ORDER BOOK DOMAIN INVARIANTS:
- Structure: SortedDict[price → deque[Order]] for each side
- Bids sorted descending, asks sorted ascending
- Trade price = RESTING order's price (not aggressor's)
- Cancel = O(1) via order_id dict + remove from deque
- Iceberg: hidden_qty tracked separately; replenish with NEW timestamp
- Stop buy triggers when last_price >= trigger (price rises)
- Stop sell triggers when last_price <= trigger (price falls)
- Deadlock risk: never call place_buy/sell while holding book lock
- Crossed book (bid >= ask) must never exist after any operation
""",
    "distributed|consensus|raft|replication|leader": """
DISTRIBUTED SYSTEMS INVARIANTS:
- Leader election: term numbers must be monotonically increasing
- Log replication: never apply before majority acknowledgment
- Split brain: fencing token invalidates old leader writes
- Exactly-once: idempotency key + dedup store, not just retry
- Network partition: prefer CP (refuse writes) over AP (accept divergence) for financial data
""",
    "database|sql|transaction|acid": """
DATABASE INVARIANTS:
- WAL must fsync before acknowledging commit
- Read-your-writes: sticky session or synchronous replication read
- N+1: always batch queries, never query in loop
- Index: B-tree for range, hash for equality, never index low-cardinality columns
- Connection: always pool, never open per request
""",
    "cache|lru|eviction|ttl": """
CACHE INVARIANTS:
- LRU: OrderedDict move_to_end on every get — not just on insert
- TTL: store insertion time, check on access — don't rely on background sweep
- Cache stampede: use lock per key, not global lock
- Never cache mutable objects without deep copy
""",
    "graph|shortest path|bfs|dfs|dijkstra": """
GRAPH ALGORITHM INVARIANTS:
- BFS: use deque not list for O(1) popleft
- Dijkstra: skip stale heap entries (dist[u] < d check)
- DFS: explicit stack for deep graphs (recursion = stack overflow at n>1000)
- Visited: use set not list (O(1) vs O(n) lookup)
- Negative weights: Bellman-Ford not Dijkstra
""",
}

import re

def get_reference_context(msg: str) -> str:
    """Return first-principles + any relevant domain knowledge."""
    if not msg or not msg.strip():
        return ""

    msg_lower = msg.lower()
    domain_additions = []
    for pattern, knowledge in DOMAIN_KNOWLEDGE.items():
        if any(kw in msg_lower for kw in pattern.split("|")):
            domain_additions.append(knowledge)

    result = PRINCIPLES
    if domain_additions:
        result += "\n\nDOMAIN-SPECIFIC INVARIANTS FOR THIS TASK:\n" + "\n".join(domain_additions)

    return result
