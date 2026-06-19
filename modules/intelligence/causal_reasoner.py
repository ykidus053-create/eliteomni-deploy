import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class CausalNode:
    id: str
    description: str
    node_type: str = "variable"

@dataclass
class CausalEdge:
    cause: str
    effect: str
    strength: float = 0.7
    mechanism: str = ""
    is_confounded: bool = False

class CausalGraph:
    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []

    def add_node(self, nid: str, desc: str, ntype: str = "variable"):
        self.nodes[nid] = CausalNode(nid, desc, ntype)

    def add_edge(self, cause: str, effect: str, strength: float = 0.7,
                 mechanism: str = "", confounded: bool = False):
        self.edges.append(CausalEdge(cause, effect, strength, mechanism, confounded))

    def get_causes(self, effect_id: str) -> List[CausalEdge]:
        return [e for e in self.edges if e.effect == effect_id]

    def get_effects(self, cause_id: str) -> List[CausalEdge]:
        return [e for e in self.edges if e.cause == cause_id]

    def counterfactual(self, intervention_node: str, new_value: str) -> str:
        effects = self.get_effects(intervention_node)
        if not effects:
            return f"Intervening on {intervention_node} has no downstream effects in the current model."
        lines = [f"If we intervene on {intervention_node} (set to: {new_value}):"]
        for edge in effects:
            mech = edge.mechanism or "unknown"
            conf_note = " [WARNING: confounding present]" if edge.is_confounded else ""
            lines.append(f"  -> {edge.effect} would change (mechanism: {mech}, strength: {edge.strength:.0%}){conf_note}")
        return "\n".join(lines)

    def to_context(self) -> str:
        if not self.nodes:
            return ""
        lines = ["<causal_model>"]
        for edge in self.edges[:8]:
            conf = " [confounded]" if edge.is_confounded else ""
            lines.append(f"  {edge.cause} -> {edge.effect} (strength={edge.strength:.0%}){conf}")
        lines.append("</causal_model>")
        return "\n".join(lines)

def extract_causal_structure(msg: str) -> Optional[CausalGraph]:
    causal_signals = ["cause","effect","because","leads to","results in",
                      "due to","impact of","consequence","why does","what happens if"]
    if not any(s in msg.lower() for s in causal_signals):
        return None
    g = CausalGraph()
    patterns = [
        (r"(\w+(?:\s+\w+)?)\s+causes?\s+(\w+(?:\s+\w+)?)", 0.8),
        (r"(\w+(?:\s+\w+)?)\s+leads?\s+to\s+(\w+(?:\s+\w+)?)", 0.75),
        (r"(\w+(?:\s+\w+)?)\s+results?\s+in\s+(\w+(?:\s+\w+)?)", 0.75),
        (r"impact\s+of\s+(\w+(?:\s+\w+)?)\s+on\s+(\w+(?:\s+\w+)?)", 0.7),
    ]
    for pat, strength in patterns:
        for m in re.finditer(pat, msg, re.I):
            cause, effect = m.group(1).strip().lower(), m.group(2).strip().lower()
            if len(cause) > 2 and len(effect) > 2:
                g.add_node(cause, cause)
                g.add_node(effect, effect)
                g.add_edge(cause, effect, strength)
    return g if g.edges else None

def build_causal_injection(msg: str) -> str:
    g = extract_causal_structure(msg)
    if not g:
        return ""
    ctx = g.to_context()
    if not ctx:
        return ""
    return ("\n" + ctx +
            "\nWhen reasoning about causation: (1) distinguish correlation from causation, "
            "(2) identify confounders, (3) consider counterfactuals explicitly.\n")
