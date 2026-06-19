import re, time, json, sqlite3, hashlib
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

DB = Path.home() / "eliteomni_hypotheses.db"

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS hypotheses (
        id TEXT PRIMARY KEY, session TEXT, question TEXT,
        hypothesis TEXT, prior REAL, likelihood REAL,
        posterior REAL, evidence TEXT, status TEXT,
        created REAL, updated REAL)""")
    c.commit()
    return c

@dataclass
class Hypothesis:
    id: str
    text: str
    prior: float = 0.5
    likelihood: float = 0.5
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)

    @property
    def posterior(self):
        p = self.prior
        for _ in self.evidence_for:
            p = (p * 0.8) / (p * 0.8 + (1-p) * 0.3)
        for _ in self.evidence_against:
            p = (p * 0.2) / (p * 0.2 + (1-p) * 0.7)
        return round(max(0.01, min(0.99, p)), 3)

    def update(self, evidence: str, supports: bool):
        if supports:
            self.evidence_for.append(evidence[:100])
        else:
            self.evidence_against.append(evidence[:100])

    def to_dict(self):
        return {"id": self.id, "text": self.text, "prior": self.prior,
                "posterior": self.posterior,
                "evidence_for": len(self.evidence_for),
                "evidence_against": len(self.evidence_against)}

class HypothesisEngine:
    def __init__(self, session_id="default"):
        self.session = session_id
        self.active: dict = {}

    def generate_hypotheses(self, question: str, skill: str) -> List[Hypothesis]:
        q = question.lower()
        hyps = []
        hid = lambda t: hashlib.md5(t.encode()).hexdigest()[:8]
        if skill == "researcher" or any(w in q for w in ["why","cause","reason","explain why"]):
            hyps = [
                Hypothesis(hid(question+"h1"), f"The primary cause is a direct mechanism: {question[:60]}", prior=0.4),
                Hypothesis(hid(question+"h2"), "Multiple interacting factors explain this", prior=0.35),
                Hypothesis(hid(question+"h3"), "This is a surface symptom of a deeper systemic issue", prior=0.25),
            ]
        elif skill == "coder" or any(w in q for w in ["bug","error","fail","broken","not working"]):
            hyps = [
                Hypothesis(hid(question+"h1"), "Logic error in the core algorithm", prior=0.35),
                Hypothesis(hid(question+"h2"), "State mutation or side effect bug", prior=0.25),
                Hypothesis(hid(question+"h3"), "External dependency or environment issue", prior=0.2),
                Hypothesis(hid(question+"h4"), "Off-by-one or boundary condition failure", prior=0.2),
            ]
        elif skill == "calculator" or any(w in q for w in ["calculate","compute","what is","how much"]):
            hyps = [
                Hypothesis(hid(question+"h1"), "Direct arithmetic path is correct", prior=0.5),
                Hypothesis(hid(question+"h2"), "A unit conversion or scaling factor is required", prior=0.3),
                Hypothesis(hid(question+"h3"), "The problem requires a multi-step intermediate result", prior=0.2),
            ]
        else:
            hyps = [
                Hypothesis(hid(question+"h1"), "The most common interpretation is correct", prior=0.5),
                Hypothesis(hid(question+"h2"), "An alternative framing better explains the situation", prior=0.3),
                Hypothesis(hid(question+"h3"), "The question contains a hidden assumption that needs challenging", prior=0.2),
            ]
        for h in hyps:
            self.active[h.id] = h
        self._save(question, hyps)
        return hyps

    def rank(self) -> List[Hypothesis]:
        return sorted(self.active.values(), key=lambda h: h.posterior, reverse=True)

    def best(self) -> Optional[Hypothesis]:
        ranked = self.rank()
        return ranked[0] if ranked else None

    def update_from_evidence(self, evidence: str, supports_id: str = None):
        for hid, h in self.active.items():
            if supports_id and hid == supports_id:
                h.update(evidence, supports=True)
            elif supports_id:
                h.update(evidence, supports=False)

    def to_injection(self) -> str:
        if not self.active:
            return ""
        ranked = self.rank()
        lines = ["<hypothesis_space>"]
        for h in ranked[:4]:
            d = h.to_dict()
            lines.append(f"  H[{d['id']}] p={d['posterior']:.2f}: {d['text']}")
            if h.evidence_for:
                lines.append(f"    supporting evidence: {len(h.evidence_for)} items")
            if h.evidence_against:
                lines.append(f"    contradicting evidence: {len(h.evidence_against)} items")
        lines.append("Reason toward the highest-posterior hypothesis. Explicitly update beliefs as evidence emerges.")
        lines.append("</hypothesis_space>")
        return "\n".join(lines)

    def _save(self, question: str, hyps: List[Hypothesis]):
        try:
            c = _conn()
            for h in hyps:
                c.execute("INSERT OR REPLACE INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (h.id, self.session, question[:200], h.text, h.prior,
                     h.likelihood, h.posterior, json.dumps([]), "active", time.time(), time.time()))
            c.commit(); c.close()
        except: pass

_engines: dict = {}

def get_hypothesis_engine(session_id="default") -> HypothesisEngine:
    if session_id not in _engines:
        _engines[session_id] = HypothesisEngine(session_id)
    return _engines[session_id]
