"""
reasoning_core.py
-----------------
Closes the architectural gaps identified in the intelligence audit:

1. EPISTEMIC STATE TRACKER  — explicit belief/uncertainty representation
2. ACTIVE BELIEF REVISION   — hypotheses update when tool results arrive
3. GOAL DECOMPOSITION       — dependency-aware task breakdown
4. WORKING MEMORY           — scratchpad that persists across tool calls
5. ADAPTIVE ROUTING         — self-model scores change actual behavior
6. CONTEXT COMPRESSOR       — relevance-scored injection, not additive appending
7. REASONING TRACE          — inspectable chain before final answer
8. VERIFIER LOOP            — separate verify pass before streaming
"""

import re
import time
import json
import sqlite3
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

DB = Path.home() / "eliteomni_reasoning_core.db"

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS belief_states (
        id TEXT PRIMARY KEY, session TEXT, claim TEXT,
        confidence REAL, source TEXT, contradicted INTEGER DEFAULT 0,
        created REAL, updated REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS reasoning_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT, step_type TEXT, content TEXT,
        confidence REAL, ts REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS working_memory (
        key TEXT PRIMARY KEY, session TEXT, value TEXT,
        importance REAL, created REAL, accessed REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS goal_graph (
        id TEXT PRIMARY KEY, session TEXT, goal TEXT,
        parent_id TEXT, status TEXT, depends_on TEXT,
        result TEXT, created REAL)""")
    c.commit()
    return c


# ── 1. EPISTEMIC STATE TRACKER ────────────────────────────────────────────────

@dataclass
class Belief:
    id: str
    claim: str
    confidence: float        # 0.0 = unknown, 1.0 = certain
    source: str              # "prior", "search", "calc", "user", "inferred"
    contradicted: bool = False

    def to_prompt_line(self) -> str:
        conf_label = (
            "certain" if self.confidence > 0.85 else
            "likely" if self.confidence > 0.65 else
            "uncertain" if self.confidence > 0.40 else
            "speculative"
        )
        flag = " [CONTRADICTED]" if self.contradicted else ""
        return f"  [{conf_label}] {self.claim} (src={self.source}){flag}"


class EpistemicStateTracker:
    """
    Maintains explicit belief state for the current reasoning chain.
    Unlike hypothesis_engine which ranks hypotheses, this tracks
    individual factual claims and their confidence.
    """
    def __init__(self, session_id: str):
        self.session = session_id
        self.beliefs: Dict[str, Belief] = {}
        self._load()

    def assert_belief(self, claim: str, confidence: float, source: str) -> str:
        bid = hashlib.md5(f"{self.session}{claim}".encode()).hexdigest()[:10]
        self.beliefs[bid] = Belief(bid, claim[:200], confidence, source)
        self._save(bid)
        return bid

    def contradict(self, claim_fragment: str, new_evidence: str, new_confidence: float):
        """Call this when a tool result contradicts a prior belief."""
        updated = []
        for bid, b in self.beliefs.items():
            if claim_fragment.lower() in b.claim.lower() and not b.contradicted:
                b.contradicted = True
                b.confidence = min(b.confidence, 1.0 - new_confidence)
                updated.append(b.claim[:60])
        if updated:
            self.assert_belief(
                f"REVISION: {new_evidence[:150]}",
                new_confidence,
                "contradiction"
            )
        return updated

    def get_uncertain_claims(self) -> List[Belief]:
        return [b for b in self.beliefs.values()
                if 0.2 < b.confidence < 0.65 and not b.contradicted]

    def to_injection(self) -> str:
        if not self.beliefs:
            return ""
        active = [b for b in self.beliefs.values() if not b.contradicted][-8:]
        contradicted = [b for b in self.beliefs.values() if b.contradicted][-3:]
        lines = ["<epistemic_state>", "CURRENT BELIEFS:"]
        for b in active:
            lines.append(b.to_prompt_line())
        if contradicted:
            lines.append("CONTRADICTED (revise these):")
            for b in contradicted:
                lines.append(b.to_prompt_line())
        uncertain = self.get_uncertain_claims()
        if uncertain:
            lines.append(f"VERIFY THESE ({len(uncertain)} uncertain claims):")
            for b in uncertain[:3]:
                lines.append(f"  ? {b.claim[:80]}")
        lines.append("</epistemic_state>")
        return "\n".join(lines)

    def _save(self, bid: str):
        try:
            b = self.beliefs[bid]
            c = _conn()
            c.execute("""INSERT OR REPLACE INTO belief_states
                VALUES (?,?,?,?,?,?,?,?)""",
                (bid, self.session, b.claim, b.confidence,
                 b.source, 1 if b.contradicted else 0,
                 time.time(), time.time()))
            c.commit(); c.close()
        except Exception as _e: print(f"[reasoning_core] suppressed: {_e}")

    def _load(self):
        try:
            c = _conn()
            rows = c.execute(
                "SELECT id,claim,confidence,source,contradicted FROM belief_states "
                "WHERE session=? ORDER BY created DESC LIMIT 20",
                (self.session,)).fetchall()
            c.close()
            for bid, claim, conf, src, contra in rows:
                self.beliefs[bid] = Belief(bid, claim, conf, src, bool(contra))
        except Exception as _e: print(f"[reasoning_core] suppressed: {_e}")


# ── 2. WORKING MEMORY ─────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Scratchpad that persists WITHIN a reasoning chain across tool calls.
    Unlike the global memory DB, this is ephemeral per session
    but survives the turn boundary — enabling multi-step tool use.
    Key architectural gap it fills: tool result A informs tool call B.
    """
    def __init__(self, session_id: str, max_items: int = 20):
        self.session = session_id
        self.max_items = max_items
        self._mem: Dict[str, Dict] = {}
        self._load()

    def write(self, key: str, value: Any, importance: float = 0.5):
        self._mem[key] = {
            "value": str(value)[:500],
            "importance": importance,
            "created": time.time(),
            "accessed": time.time()
        }
        if len(self._mem) > self.max_items:
            # Evict least important + oldest
            sorted_keys = sorted(
                self._mem.keys(),
                key=lambda k: self._mem[k]["importance"] * 0.6 +
                              (time.time() - self._mem[k]["created"]) / 86400 * -0.4
            )
            del self._mem[sorted_keys[0]]
        self._persist(key)

    def read(self, key: str) -> Optional[str]:
        if key in self._mem:
            self._mem[key]["accessed"] = time.time()
            return self._mem[key]["value"]
        return None

    def read_relevant(self, query: str, top_k: int = 4) -> List[Tuple[str, str]]:
        q = query.lower()
        scored = []
        for k, v in self._mem.items():
            score = v["importance"]
            if q in k.lower() or q in v["value"].lower():
                score += 0.3
            recency = max(0, 1.0 - (time.time() - v["accessed"]) / 3600)
            score += recency * 0.2
            scored.append((score, k, v["value"]))
        scored.sort(reverse=True)
        return [(k, v) for _, k, v in scored[:top_k]]

    def to_injection(self, query: str = "") -> str:
        relevant = self.read_relevant(query, top_k=5)
        if not relevant:
            return ""
        lines = ["<working_memory>"]
        for k, v in relevant:
            lines.append(f"  {k}: {v[:120]}")
        lines.append("Use these facts from earlier in this reasoning chain.")
        lines.append("</working_memory>")
        return "\n".join(lines)

    def _persist(self, key: str):
        try:
            item = self._mem[key]
            c = _conn()
            c.execute("""INSERT OR REPLACE INTO working_memory
                VALUES (?,?,?,?,?,?)""",
                (key, self.session, item["value"], item["importance"],
                 item["created"], item["accessed"]))
            c.commit(); c.close()
        except Exception as _e: print(f"[reasoning_core] suppressed: {_e}")

    def _load(self):
        try:
            c = _conn()
            rows = c.execute(
                "SELECT key,value,importance,created,accessed FROM working_memory "
                "WHERE session=? ORDER BY accessed DESC LIMIT 20",
                (self.session,)).fetchall()
            c.close()
            for key, val, imp, cr, ac in rows:
                self._mem[key] = {"value": val, "importance": imp,
                                  "created": cr, "accessed": ac}
        except Exception as _e: print(f"[reasoning_core] suppressed: {_e}")


# ── 3. GOAL DECOMPOSITION WITH DEPENDENCY GRAPH ───────────────────────────────

@dataclass
class Goal:
    id: str
    description: str
    parent_id: Optional[str]
    depends_on: List[str]     # list of goal IDs that must complete first
    status: str               # pending, active, done, failed, blocked
    result: str = ""

class GoalGraph:
    """
    Dependency-aware task decomposition.
    Unlike linear planners, this allows parallel execution of
    independent subgoals and blocks dependent ones until ready.
    """
    def __init__(self, session_id: str):
        self.session = session_id
        self.goals: Dict[str, Goal] = {}

    def add_goal(self, description: str, parent_id: Optional[str] = None,
                 depends_on: List[str] = None) -> str:
        gid = hashlib.md5(f"{self.session}{description}{time.time()}".encode()).hexdigest()[:8]
        self.goals[gid] = Goal(
            id=gid,
            description=description,
            parent_id=parent_id,
            depends_on=depends_on or [],
            status="pending"
        )
        return gid

    def get_ready_goals(self) -> List[Goal]:
        """Returns goals whose dependencies are all done."""
        done_ids = {g.id for g in self.goals.values() if g.status == "done"}
        return [
            g for g in self.goals.values()
            if g.status == "pending" and all(d in done_ids for d in g.depends_on)
        ]

    def complete_goal(self, gid: str, result: str):
        if gid in self.goals:
            self.goals[gid].status = "done"
            self.goals[gid].result = result[:300]

    def fail_goal(self, gid: str, reason: str):
        if gid in self.goals:
            self.goals[gid].status = "failed"
            self.goals[gid].result = reason
            # Block dependents
            for g in self.goals.values():
                if gid in g.depends_on:
                    g.status = "blocked"

    def to_injection(self) -> str:
        if not self.goals:
            return ""
        ready = self.get_ready_goals()
        active = [g for g in self.goals.values() if g.status == "active"]
        done = [g for g in self.goals.values() if g.status == "done"]
        blocked = [g for g in self.goals.values() if g.status == "blocked"]
        lines = ["<goal_graph>"]
        if done:
            lines.append(f"COMPLETED ({len(done)}): " +
                        " | ".join(g.description[:40] for g in done[-3:]))
        if active:
            lines.append("IN PROGRESS: " +
                        " | ".join(g.description[:50] for g in active))
        if ready:
            lines.append("READY TO EXECUTE: " +
                        " | ".join(f"[{g.id}] {g.description[:50]}" for g in ready[:3]))
        if blocked:
            lines.append(f"BLOCKED ({len(blocked)}): dependency failed")
        lines.append("Execute READY goals next. Check dependencies before acting.")
        lines.append("</goal_graph>")
        return "\n".join(lines)

    def decompose(self, main_task: str, skill: str) -> List[Goal]:
        """Auto-decompose a task into a dependency graph."""
        self.goals.clear()
        root = self.add_goal(main_task)
        self.goals[root].status = "active"

        if skill == "researcher":
            g1 = self.add_goal(f"Search for primary sources: {main_task[:60]}", root)
            g2 = self.add_goal(f"Search for recent developments: {main_task[:60]}", root)
            g3 = self.add_goal("Cross-validate sources for consistency", root, [g1, g2])
            g4 = self.add_goal("Synthesize findings into answer", root, [g3])
        elif skill == "coder":
            g1 = self.add_goal(f"Understand requirements: {main_task[:60]}", root)
            g2 = self.add_goal("Design data structures and interfaces", root, [g1])
            g3 = self.add_goal("Implement core logic", root, [g2])
            g4 = self.add_goal("Add error handling and edge cases", root, [g3])
            g5 = self.add_goal("Verify correctness with test cases", root, [g4])
        elif skill == "calculator":
            g1 = self.add_goal("Parse problem and identify formula", root)
            g2 = self.add_goal("Calculate via direct path", root, [g1])
            g3 = self.add_goal("Verify via alternative method", root, [g1])
            g4 = self.add_goal("Cross-check both paths agree", root, [g2, g3])
        else:
            g1 = self.add_goal(f"Gather information: {main_task[:60]}", root)
            g2 = self.add_goal("Analyze and reason about findings", root, [g1])
            g3 = self.add_goal("Formulate and verify answer", root, [g2])

        return list(self.goals.values())


# ── 4. REASONING TRACE ────────────────────────────────────────────────────────

class ReasoningTrace:
    """
    Inspectable reasoning chain that can be critiqued mid-generation.
    Frontier labs call this 'process supervision' — we score each step
    not just the final answer.
    """
    def __init__(self, session_id: str):
        self.session = session_id
        self.steps: List[Dict] = []

    def add_step(self, step_type: str, content: str, confidence: float = 0.7):
        """
        step_type: "observation", "hypothesis", "tool_call",
                   "tool_result", "inference", "verification", "conclusion"
        """
        step = {
            "type": step_type,
            "content": content[:300],
            "confidence": confidence,
            "ts": time.time()
        }
        self.steps.append(step)
        try:
            c = _conn()
            c.execute("INSERT INTO reasoning_traces VALUES (NULL,?,?,?,?,?)",
                (self.session, step_type, content[:300], confidence, time.time()))
            c.commit(); c.close()
        except Exception as _e: print(f"[reasoning_core] suppressed: {_e}")

    def get_weak_steps(self) -> List[Dict]:
        """Returns steps with low confidence — candidates for verification."""
        return [s for s in self.steps if s["confidence"] < 0.5]

    def to_verifier_prompt(self, original_question: str) -> str:
        if not self.steps:
            return ""
        chain = "\n".join(
            f"  [{s['type'].upper()}] (conf={s['confidence']:.0%}) {s['content']}"
            for s in self.steps[-8:]
        )
        weak = self.get_weak_steps()
        prompt = f"""REASONING TRACE VERIFICATION
Original question: {original_question[:200]}

Steps taken:
{chain}
"""
        if weak:
            prompt += f"\nWEAK STEPS TO VERIFY ({len(weak)}):\n"
            for s in weak[:3]:
                prompt += f"  - [{s['type']}] {s['content'][:100]}\n"
            prompt += "\nVerify weak steps before finalizing answer."
        return prompt

    def summary(self) -> str:
        if not self.steps:
            return ""
        counts = {}
        for s in self.steps:
            counts[s["type"]] = counts.get(s["type"], 0) + 1
        avg_conf = sum(s["confidence"] for s in self.steps) / len(self.steps)
        return (f"Trace: {len(self.steps)} steps, "
                f"avg confidence {avg_conf:.0%}, "
                f"types: {counts}")


# ── 5. ADAPTIVE ROUTER ────────────────────────────────────────────────────────

class AdaptiveRouter:
    """
    Reads self_model capability scores and CHANGES ACTUAL BEHAVIOR.
    This closes the gap where self-model tracked scores but nothing read them.
    """

    def get_strategy(self, skill: str, domain: str, complexity: str) -> Dict:
        """
        Returns a strategy dict that changes: verification_passes,
        use_multi_agent, require_search, confidence_threshold.
        """
        try:
            from modules.intelligence.self_model import get_self_model
            sm = get_self_model()
            cap = sm.get_capability(skill, domain)
            weaknesses = sm.get_known_weaknesses(skill)
            cal_error = sm.get_calibration_error()
        except Exception:
            cap = 0.75
            weaknesses = []
            cal_error = 0.0

        strategy = {
            "verification_passes": 1,
            "use_multi_agent": False,
            "require_search": False,
            "confidence_threshold": 0.7,
            "max_tokens_multiplier": 1.0,
            "add_critic_pass": False,
            "known_weaknesses": weaknesses,
        }

        # Low capability -> more verification
        if cap < 0.5:
            strategy["verification_passes"] = 3
            strategy["add_critic_pass"] = True
            strategy["confidence_threshold"] = 0.8
            strategy["max_tokens_multiplier"] = 1.3

        elif cap < 0.65:
            strategy["verification_passes"] = 2
            strategy["add_critic_pass"] = True

        # Overconfidence detected -> require search for factual claims
        if cal_error > 0.25:
            strategy["require_search"] = True
            strategy["confidence_threshold"] = 0.85

        # Hard + researcher -> always multi-agent
        if complexity == "hard" and skill == "researcher":
            strategy["use_multi_agent"] = True

        return strategy

    def to_injection(self, skill: str, domain: str, complexity: str) -> str:
        strat = self.get_strategy(skill, domain, complexity)
        lines = ["<adaptive_strategy>"]
        if strat["verification_passes"] > 1:
            lines.append(f"VERIFY {strat['verification_passes']} times before answering "
                        f"(low historical accuracy in {skill}/{domain})")
        if strat["require_search"]:
            lines.append("SEARCH REQUIRED: overconfidence detected — verify factual claims")
        if strat["add_critic_pass"]:
            lines.append("CRITIC PASS: write answer, then critique it, then revise")
        if strat["known_weaknesses"]:
            lines.append("WATCH FOR: " + "; ".join(strat["known_weaknesses"][:2]))
        if len(lines) == 1:
            return ""  # No special strategy needed
        lines.append("</adaptive_strategy>")
        return "\n".join(lines)


# ── 6. CONTEXT COMPRESSOR ─────────────────────────────────────────────────────

class ContextCompressor:
    """
    Replaces additive context injection with relevance-scored compression.
    The current system appends every module's injection without checking
    relevance or total size. This compresses to the most relevant tokens.
    """

    def compress(self, injections: List[Tuple[str, str, float]],
                 query: str, max_chars: int = 2000) -> str:
        """
        injections: list of (name, content, base_relevance_score)
        Returns compressed context fitting within max_chars.
        """
        if not injections:
            return ""

        scored = []
        q = query.lower()
        for name, content, base_score in injections:
            if not content or not content.strip():
                continue
            # Boost relevance if content mentions query keywords
            query_words = [w for w in q.split() if len(w) > 3]
            keyword_hits = sum(1 for w in query_words if w in content.lower())
            relevance = base_score + keyword_hits * 0.1
            scored.append((relevance, name, content))

        scored.sort(key=lambda x: x[0], reverse=True)

        result_parts = []
        chars_used = 0
        for relevance, name, content in scored:
            if chars_used >= max_chars:
                break
            # Allocate proportional to relevance
            alloc = min(len(content), int(max_chars * relevance / 2))
            alloc = max(alloc, 100)  # minimum useful chunk
            chunk = content[:alloc].strip()
            if chunk:
                result_parts.append(chunk)
                chars_used += len(chunk)

        return "\n\n".join(result_parts)

    def score_injection(self, content: str, query: str,
                        injection_type: str) -> float:
        """Score how relevant an injection is to this specific query."""
        base_scores = {
            "hypothesis": 0.8,
            "working_memory": 0.9,
            "epistemic_state": 0.85,
            "goal_graph": 0.7,
            "adaptive_strategy": 0.75,
            "causal": 0.5,
            "reflection": 0.6,
            "tool_recommendations": 0.8,
            "self_awareness": 0.65,
        }
        base = base_scores.get(injection_type, 0.5)
        q_lower = query.lower()
        content_lower = content.lower()
        shared_words = sum(
            1 for w in q_lower.split()
            if len(w) > 3 and w in content_lower
        )
        return min(1.0, base + shared_words * 0.05)


# ── 7. VERIFIER LOOP ──────────────────────────────────────────────────────────

def build_pre_generation_verifier(
        msg: str, skill: str, complexity: str,
        trace: Optional[ReasoningTrace] = None,
        epistemic: Optional[EpistemicStateTracker] = None) -> str:
    """
    Builds a verification prompt that runs BEFORE streaming.
    This is the 'separate verifier from generator' pattern.
    """
    parts = ["\n<pre_generation_verification>"]

    # Skill-specific verification requirements
    if skill == "coder":
        parts.extend([
            "BEFORE writing code, verify:",
            "  1. Do I understand the full requirements? (list them)",
            "  2. What are the 3 most likely failure modes?",
            "  3. What is the minimal correct implementation?",
            "  4. What test case would catch a wrong answer?",
        ])
    elif skill == "calculator":
        parts.extend([
            "BEFORE calculating, verify:",
            "  1. What formula applies? (state it explicitly)",
            "  2. What are the units? (write them out)",
            "  3. What magnitude should the answer be? (estimate first)",
            "  4. Run CALC() and compare to estimate",
        ])
    elif skill == "researcher":
        parts.extend([
            "BEFORE answering, verify:",
            "  1. Is this time-sensitive? (search if yes)",
            "  2. What are the 2 strongest opposing viewpoints?",
            "  3. What would change my answer if I learned it?",
            "  4. Am I distinguishing fact from inference?",
        ])

    # Epistemic check
    if epistemic:
        uncertain = epistemic.get_uncertain_claims()
        if uncertain:
            parts.append(f"VERIFY BEFORE ASSERTING ({len(uncertain)} uncertain claims):")
            for b in uncertain[:3]:
                parts.append(f"  ? {b.claim[:80]}")

    # Reasoning trace weak steps
    if trace:
        weak = trace.get_weak_steps()
        if weak:
            parts.append(f"WEAK REASONING STEPS TO REVISIT ({len(weak)}):")
            for s in weak[:2]:
                parts.append(f"  - {s['content'][:80]}")

    parts.append("</pre_generation_verification>")
    return "\n".join(parts)


# ── MASTER BUILDER ────────────────────────────────────────────────────────────

_trackers: Dict[str, EpistemicStateTracker] = {}
_memories: Dict[str, WorkingMemory] = {}
_routers: Dict[str, AdaptiveRouter] = {}
_graphs: Dict[str, GoalGraph] = {}
_traces: Dict[str, ReasoningTrace] = {}
_compressor = ContextCompressor()
_router = AdaptiveRouter()

def get_epistemic_tracker(session_id: str) -> EpistemicStateTracker:
    if session_id not in _trackers:
        _trackers[session_id] = EpistemicStateTracker(session_id)
    return _trackers[session_id]

def get_working_memory(session_id: str) -> WorkingMemory:
    if session_id not in _memories:
        _memories[session_id] = WorkingMemory(session_id)
    return _memories[session_id]

def get_goal_graph(session_id: str) -> GoalGraph:
    if session_id not in _graphs:
        _graphs[session_id] = GoalGraph(session_id)
    return _graphs[session_id]

def get_reasoning_trace(session_id: str) -> ReasoningTrace:
    if session_id not in _traces:
        _traces[session_id] = ReasoningTrace(session_id)
    return _traces[session_id]


def build_reasoning_core_context(
        msg: str, skill: str, complexity: str,
        session_id: str = "default") -> Tuple[str, Dict]:
    """
    Master builder for reasoning core context.
    Returns (injection_string, metadata).
    Uses ContextCompressor to stay within token budget.
    """
    meta = {}
    raw_injections = []

    # Domain detection
    domain = "general"
    for d in ["medical", "legal", "financial", "math", "code", "science", "history"]:
        if d in msg.lower():
            domain = d
            break

    # 1. Adaptive strategy (reads self-model, changes behavior)
    try:
        strat_inj = _router.to_injection(skill, domain, complexity)
        if strat_inj:
            raw_injections.append(("adaptive_strategy", strat_inj,
                _compressor.score_injection(strat_inj, msg, "adaptive_strategy")))
    except Exception as e:
        print(f"[ReasoningCore] adaptive_router: {e}")

    # 2. Working memory (cross-tool-call persistence)
    try:
        wm = get_working_memory(session_id)
        wm_inj = wm.to_injection(msg)
        if wm_inj:
            raw_injections.append(("working_memory", wm_inj,
                _compressor.score_injection(wm_inj, msg, "working_memory")))
    except Exception as e:
        print(f"[ReasoningCore] working_memory: {e}")

    # 3. Epistemic state (explicit belief tracking)
    try:
        est = get_epistemic_tracker(session_id)
        ep_inj = est.to_injection()
        if ep_inj:
            raw_injections.append(("epistemic_state", ep_inj,
                _compressor.score_injection(ep_inj, msg, "epistemic_state")))
            meta["uncertain_claims"] = len(est.get_uncertain_claims())
    except Exception as e:
        print(f"[ReasoningCore] epistemic_tracker: {e}")

    # 4. Goal graph (dependency-aware decomposition for hard tasks)
    if complexity == "hard":
        try:
            gg = get_goal_graph(session_id)
            gg.decompose(msg, skill)
            gg_inj = gg.to_injection()
            if gg_inj:
                raw_injections.append(("goal_graph", gg_inj,
                    _compressor.score_injection(gg_inj, msg, "goal_graph")))
                meta["goals"] = len(gg.goals)
        except Exception as e:
            print(f"[ReasoningCore] goal_graph: {e}")

    # 5. Pre-generation verifier
    try:
        trace = get_reasoning_trace(session_id)
        est_ref = get_epistemic_tracker(session_id)
        verifier = build_pre_generation_verifier(msg, skill, complexity, trace, est_ref)
        if verifier:
            raw_injections.append(("verifier", verifier,
                _compressor.score_injection(verifier, msg, "reflection")))
    except Exception as e:
        print(f"[ReasoningCore] verifier: {e}")

    # Compress all injections to fit token budget
    budget = {"easy": 800, "medium": 1600, "hard": 2800}.get(complexity, 1600)
    compressed = _compressor.compress(raw_injections, msg, max_chars=budget)

    meta["reasoning_core_chars"] = len(compressed)
    meta["injection_sources"] = [name for name, _, _ in raw_injections]

    return compressed, meta


def post_turn_update(msg: str, response: str, skill: str,
                     session_id: str = "default",
                     tool_results: Optional[Dict] = None):
    """
    Called after each turn to update epistemic state from tool results.
    This is the ACTIVE BELIEF REVISION loop that was missing.
    """
    def _async():
        try:
            est = get_epistemic_tracker(session_id)
            wm = get_working_memory(session_id)
            trace = get_reasoning_trace(session_id)

            # Extract claims from response and add to belief state
            sentences = re.split(r'(?<=[.!?])\s+', response)
            for sent in sentences[:10]:
                if len(sent) < 20:
                    continue
                # High confidence markers
                if re.search(r'\b(is|are|was|were|will be|equals|=)\b', sent, re.I):
                    conf = 0.75
                elif re.search(r'\b(likely|probably|possibly|may|might)\b', sent, re.I):
                    conf = 0.45
                elif re.search(r'\b(definitely|certainly|always|never|exactly)\b', sent, re.I):
                    conf = 0.9
                else:
                    conf = 0.6
                est.assert_belief(sent.strip(), conf, source=skill)

            # Update working memory with key findings
            if tool_results:
                for tool, result in tool_results.items():
                    if result and len(result) > 10:
                        wm.write(f"{tool}_result", result[:300], importance=0.8)
                        trace.add_step("tool_result", f"{tool}: {result[:150]}", confidence=0.8)

                        # Active belief revision: check if tool result contradicts beliefs
                        for bid, belief in list(est.beliefs.items()):
                            if belief.contradicted:
                                continue
                            # Simple contradiction check: negation in same topic
                            if (len(set(belief.claim.lower().split()) &
                                    set(result.lower().split())) > 2):
                                if re.search(r"\b(not|no|never|false|incorrect|wrong)\b",
                                           result, re.I):
                                    est.contradict(belief.claim[:40], result[:100], 0.75)

            # Log key conclusion to trace
            if response and len(response) > 50:
                last_para = response.strip().split("\n\n")[-1]
                trace.add_step("conclusion", last_para[:200], confidence=0.7)

        except Exception as e:
            print(f"[ReasoningCore] post_turn_update: {e}")

    threading.Thread(target=_async, daemon=True).start()
