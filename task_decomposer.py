"""
Fixes: Concurrency correctness, Distributed behavior, Replication logic.
Instead of asking model to write everything at once, decomposes into
atomic requirements and solves each one with a focused prompt.
"""

DECOMPOSITION_RULES = {
    "distributed": [
        "Implement node discovery and membership only. No other logic.",
        "Implement leader election only using Raft term voting. No other logic.",
        "Implement WAL write and fsync only. No other logic.",
        "Implement replication: leader sends entry to followers, collect ACKs. No other logic.",
        "Implement majority quorum commit only. No other logic.",
        "Implement crash recovery: replay WAL on startup only. No other logic.",
        "Wire all components: write path client→leader→WAL→replicate→quorum→ack.",
    ],
    "concurrency": [
        "Implement the shared state with a threading.Lock only. No other logic.",
        "Implement one atomic read operation using the lock only.",
        "Implement one atomic write operation using the lock only.",
        "Implement deadlock prevention: always acquire locks in fixed order only.",
        "Write a stress test: 100 threads doing concurrent reads and writes, assert no corruption.",
    ],
    "caching": [
        "Implement the cache store as a dict with threading.Lock only.",
        "Implement get with expiry check only.",
        "Implement set with TTL only.",
        "Implement eviction when cache exceeds max size only.",
        "Write a test: concurrent gets and sets with expiry, assert correctness.",
    ],
}

def detect_domain(msg: str) -> str:
    m = msg.lower()
    if any(w in m for w in ["distributed", "raft", "replication", "consensus", "kvstore", "kv store"]):
        return "distributed"
    if any(w in m for w in ["concurrent", "thread", "lock", "race", "atomic"]):
        return "concurrency"
    if any(w in m for w in ["cache", "caching", "ttl", "expir"]):
        return "caching"
    return ""

def decompose_and_solve(msg: str, generate_fn, skill: str) -> str:
    if skill != "coder":
        return ""

    domain = detect_domain(msg)
    if not domain:
        return ""

    steps = DECOMPOSITION_RULES[domain]
    print(f"[decomposer] domain={domain} steps={len(steps)}")

    parts = []
    context = ""
    for i, step in enumerate(steps):
        prompt = f"""You are implementing one specific part of a larger system.
Previous parts already implemented:
{context[-1500:] if context else 'None yet.'}

YOUR ONLY TASK: {step}
Original request context: {msg[:200]}

Write ONLY the code for this specific task. Real implementation, no stubs, no pass."""

        result = generate_fn(prompt)
        if result:
            parts.append(f"# Step {i+1}: {step}\n{result}")
            context += f"\n{result}"
            print(f"[decomposer] step {i+1}/{len(steps)} done ({len(result)} chars)")

    if not parts:
        return ""

    # Final wiring step
    wire_prompt = f"""Given these implemented components:

{context[-3000:]}

Wire them together into a single working class/module for: {msg[:300]}
No stubs. All components must be connected and functional."""

    final = generate_fn(wire_prompt)
    return final or "\n\n".join(parts)
