"""
Goal-Directed Planning Module
HTN-lite: decomposes goals into subtasks with dependency tracking.
Replaces the string WORKFLOW descriptions with executable plans.
"""
import json, time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    id: str
    description: str
    skill: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    created: float = field(default_factory=time.time)
    priority: int = 5  # 1=highest

@dataclass
class Plan:
    goal: str
    tasks: List[Task] = field(default_factory=list)
    complexity: str = "medium"
    created: float = field(default_factory=time.time)

    def next_executable(self) -> Optional[Task]:
        done_ids = {t.id for t in self.tasks if t.status == TaskStatus.DONE}
        for task in sorted(self.tasks, key=lambda t: t.priority):
            if task.status == TaskStatus.PENDING:
                if all(dep in done_ids for dep in task.dependencies):
                    return task
        return None

    def is_complete(self) -> bool:
        return all(t.status in (TaskStatus.DONE, TaskStatus.FAILED) for t in self.tasks)

    def to_context(self) -> str:
        lines = [f"PLAN: {self.goal}"]
        for t in self.tasks:
            deps = f" [needs: {','.join(t.dependencies)}]" if t.dependencies else ""
            lines.append(f"  [{t.status.value.upper()}] {t.id}: {t.description}{deps}")
        return "\n".join(lines)

DECOMPOSITION_PATTERNS: Dict[str, List[dict]] = {
    "research": [
        {"id": "search", "desc": "Search for current information", "skill": "researcher", "priority": 1},
        {"id": "analyze", "desc": "Analyze and synthesize findings", "skill": "researcher",
         "deps": ["search"], "priority": 2},
        {"id": "verify", "desc": "Verify key claims", "skill": "researcher", "deps": ["analyze"], "priority": 3},
        {"id": "respond", "desc": "Compose final response", "skill": "general",
         "deps": ["analyze", "verify"], "priority": 4},
    ],
    "code": [
        {"id": "understand", "desc": "Understand requirements and constraints", "skill": "coder", "priority": 1},
        {"id": "design", "desc": "Design solution architecture", "skill": "coder",
         "deps": ["understand"], "priority": 2},
        {"id": "implement", "desc": "Implement the solution", "skill": "coder",
         "deps": ["design"], "priority": 3},
        {"id": "verify", "desc": "Verify correctness and edge cases", "skill": "coder",
         "deps": ["implement"], "priority": 4},
    ],
    "math": [
        {"id": "parse", "desc": "Parse and formalize the problem", "skill": "calculator", "priority": 1},
        {"id": "estimate", "desc": "Back-of-envelope magnitude estimate", "skill": "calculator",
         "deps": ["parse"], "priority": 2},
        {"id": "calculate", "desc": "Precise calculation with CALC()", "skill": "calculator",
         "deps": ["parse"], "priority": 2},
        {"id": "verify", "desc": "Independent verification via EXEC()", "skill": "calculator",
         "deps": ["calculate"], "priority": 3},
        {"id": "respond", "desc": "Present verified answer with work shown", "skill": "calculator",
         "deps": ["calculate", "verify"], "priority": 4},
    ],
    "general": [
        {"id": "understand", "desc": "Fully understand the question and intent", "skill": "general", "priority": 1},
        {"id": "respond", "desc": "Compose direct, complete answer", "skill": "general",
         "deps": ["understand"], "priority": 2},
    ],
}

def decompose(goal: str, skill: str, complexity: str) -> Plan:
    """Decompose a goal into a dependency-tracked task plan."""
    if complexity == "easy":
        pattern = "general"
    elif skill in ("researcher",):
        pattern = "research"
    elif skill in ("coder",):
        pattern = "code"
    elif skill in ("calculator",):
        pattern = "math"
    else:
        pattern = "general"

    templates = DECOMPOSITION_PATTERNS[pattern]
    tasks = []
    for t in templates:
        tasks.append(Task(
            id=t["id"],
            description=t["desc"],
            skill=t.get("skill", skill),
            dependencies=t.get("deps", []),
            priority=t.get("priority", 5),
        ))
    return Plan(goal=goal, tasks=tasks, complexity=complexity)

def plan_to_system_injection(plan: Plan) -> str:
    if not plan or plan.complexity == "easy":
        return ""
    return f"\n<active_plan>\n{plan.to_context()}\n</active_plan>\n"
