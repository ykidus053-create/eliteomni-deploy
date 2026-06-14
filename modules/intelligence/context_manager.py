import re, time, json, threading
from typing import List, Dict, Optional

ZONE_PRIORITIES = {
    "identity":     (1, 500),
    "constraints":  (2, 300),
    "tools":        (3, 150),
    "memory":       (4, 400),
    "exemplars":    (5, 600),
    "history":      (6, 800),
    "scratchpad":   (7, 300),
    "user_inst":    (8, 300),
}

class ContextZone:
    def __init__(self, name, content, priority, budget):
        self.name = name
        self.content = content.strip() if content else ""
        self.priority = priority
        self.budget = budget

    def render(self):
        if not self.content:
            return ""
        if len(self.content) <= self.budget:
            return self.content
        cut = self.content[:self.budget]
        nl = cut.rfind("\n")
        if nl > self.budget * 0.75:
            cut = cut[:nl]
        return cut + "…[trimmed]"

class DriftPreventer:
    def __init__(self, session_id="default"):
        self.session_id = session_id
        self.zones: Dict[str, ContextZone] = {}
        self.turn = 0

    def set(self, name, content):
        p, b = ZONE_PRIORITIES.get(name, (9, 200))
        self.zones[name] = ContextZone(name, content, p, b)

    def build(self, complexity="medium"):
        self.turn += 1
        budgets = {"easy": 1800, "medium": 3000, "hard": 5000}
        budget = budgets.get(complexity, 3000)
        ordered = sorted(self.zones.values(), key=lambda z: z.priority)
        parts, used = [], 0
        for z in ordered:
            rendered = z.render()
            if not rendered:
                continue
            if used + len(rendered) > budget:
                if z.priority <= 3:
                    parts.append(rendered)
                    used += len(rendered)
                else:
                    break
            else:
                parts.append(rendered)
                used += len(rendered)
        assembled = "\n\n".join(parts)
        if self.turn >= 3 and "identity" in self.zones:
            anchor = self.zones["identity"].content[:220]
            assembled += f"\n\n[REANCHOR t={self.turn}] {anchor}"
        if "constraints" in self.zones and self.turn >= 5:
            con = self.zones["constraints"].content[:150]
            assembled += f"\n[CONSTRAINTS REMINDER] {con}"
        return assembled

_registry: Dict[str, DriftPreventer] = {}
_lock = threading.Lock()

def get_dp(session_id="default") -> DriftPreventer:
    with _lock:
        if session_id not in _registry:
            _registry[session_id] = DriftPreventer(session_id)
        return _registry[session_id]

def score_relevance(chunk: str, query: str) -> float:
    q = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    c = set(re.sub(r"[^\w\s]", "", chunk.lower()).split())
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c) * 3.0

def filter_memories(memories: List[str], query: str, char_budget=400) -> str:
    if not memories:
        return ""
    scored = sorted([(score_relevance(m, query), m) for m in memories], reverse=True)
    out, used = [], 0
    for sc, mem in scored:
        if sc < 0.04:
            break
        if used + len(mem) > char_budget:
            break
        out.append(mem[:160])
        used += len(mem)
    if not out:
        return ""
    return "[RELEVANT MEMORY]\n" + "\n".join(f"- {m}" for m in out[:6])

def compress_history(history: List[Dict], keep_recent=6) -> List[Dict]:
    if len(history) <= keep_recent:
        return history
    system = [m for m in history if m.get("role") == "system"]
    conv = [m for m in history if m.get("role") != "system"]
    old, recent = conv[:-keep_recent], conv[-keep_recent:]
    if not old:
        return history
    n = len(old)
    summary = {"role": "system", "content": f"[{n} earlier turns: " + " | ".join(
        f"{m.get('role','?')}:{str(m.get('content',''))[:80]}" for m in old[-4:]
    ) + "]"}
    return system + [summary] + recent

def build_drift_free_prompt(base_prompt, skill, complexity, memories, query, history, user_inst="", session_id="default") -> str:
    dp = get_dp(session_id)
    skill_constraints = {
        "coder":      "Write complete typed code. Verify with EXEC(). State complexity.",
        "researcher": "Use SEARCH(). Cite sources. Mark [VERIFIED]/[UNCERTAIN].",
        "calculator": "Use CALC(). Show all steps. State units. Bold final answer.",
        "general":    "Answer directly. Be complete. Do not pad."
    }
    dp.set("identity", base_prompt[:450])
    dp.set("constraints", skill_constraints.get(skill, skill_constraints["general"]))
    dp.set("memory", filter_memories(memories, query))
    if user_inst:
        dp.set("user_inst", f"USER INSTRUCTIONS (always follow): {user_inst}")
    return dp.build(complexity)
