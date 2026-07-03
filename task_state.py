"""Structured task state for long-horizon agentic tasks.
Prevents the model from forgetting early instructions by maintaining
a persistent state object injected into every prompt."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

log = logging.getLogger(__name__)

@dataclass
class TaskState:
    objectives: List[str] = field(default_factory=list)
    completed_objectives: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    last_action: str = ""
    confidence: float = 1.0
    step_count: int = 0

    def add_objective(self, obj: str) -> None:
        if obj and obj not in self.objectives:
            self.objectives.append(obj)
            log.debug("[TaskState] Added objective: %s", obj[:60])

    def complete_objective(self, obj: str) -> None:
        if obj in self.objectives:
            self.objectives.remove(obj)
            self.completed_objectives.append(obj)
            log.debug("[TaskState] Completed objective: %s", obj[:60])

    def record_decision(self, decision: str) -> None:
        if decision:
            self.decisions.append(f"[step {self.step_count}] {decision}")

    def add_open_question(self, q: str) -> None:
        if q and q not in self.open_questions:
            self.open_questions.append(q)

    def resolve_open_question(self, q: str) -> None:
        if q in self.open_questions:
            self.open_questions.remove(q)

    def add_fact(self, fact: str) -> None:
        if fact:
            self.facts.append(fact)

    def record_step(self, action: str, success: bool, confidence: float = 1.0) -> None:
        self.step_count += 1
        self.last_action = action
        self.confidence = (0.7 * self.confidence) + (0.3 * confidence)
        if not success:
            self.failed_steps.append(f"[step {self.step_count}] {action}")

    def to_prompt_section(self) -> str:
        if not any([self.objectives, self.decisions, self.open_questions, self.facts]):
            return ""

        lines = ["[CURRENT TASK STATE]"]
        if self.objectives:
            lines.append("Active objectives:")
            for o in self.objectives[:10]:
                lines.append(f"  - [ ] {o}")
        if self.completed_objectives:
            lines.append(f"Completed: {len(self.completed_objectives)} objective(s)")
        if self.decisions:
            lines.append("Recent decisions:")
            for d in self.decisions[-5:]:
                lines.append(f"  - {d}")
        if self.open_questions:
            lines.append("Open questions (blockers):")
            for q in self.open_questions[:5]:
                lines.append(f"  - ? {q}")
        if self.facts:
            lines.append("Grounded facts:")
            for f in self.facts[-10:]:
                lines.append(f"  - {f}")
        if self.failed_steps:
            lines.append(f"Failed steps: {len(self.failed_steps)} (check for compounding errors)")
        lines.append(f"Overall confidence: {self.confidence:.2f} | Step: {self.step_count}")
        lines.append(f"Last action: {self.last_action}")
        lines.append("[/CURRENT TASK STATE]")
        lines.append("")
        lines.append("RULE: Before your next action, re-read the state above. Never drop an active objective. "
                     "If confidence < 0.5, re-ground before proceeding.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives": self.objectives,
            "completed_objectives": self.completed_objectives,
            "decisions": self.decisions,
            "open_questions": self.open_questions,
            "facts": self.facts,
            "failed_steps": self.failed_steps,
            "last_action": self.last_action,
            "confidence": self.confidence,
            "step_count": self.step_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def should_escalate(self) -> bool:
        if len(self.failed_steps) >= 3:
            return True
        if self.confidence < 0.4:
            return True
        return False

_global_state: Optional[TaskState] = None

def get_global_state() -> TaskState:
    global _global_state
    if _global_state is None:
        _global_state = TaskState()
    return _global_state

def reset_global_state() -> TaskState:
    global _global_state
    _global_state = TaskState()
    return _global_state
