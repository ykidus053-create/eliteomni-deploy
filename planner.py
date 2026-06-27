"""
Hierarchical Planning System — replaces the stub architect_plan.
Upgraded: True parallel execution for independent DAG nodes.
"""
import json, re, time, os, sqlite3, threading, uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

_DB = os.path.expanduser("~/eliteomni_plans.db")
_lock = threading.Lock()

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PlanStep:
    id: str
    description: str
    tool: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

@dataclass
class Plan:
    id: str
    goal: str
    steps: List[PlanStep]
    status: str = "active"
    created_ts: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

def _init_db():
    con = sqlite3.connect(_DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS plans (
        id TEXT PRIMARY KEY, goal TEXT, steps TEXT,
        status TEXT DEFAULT 'active', created_ts REAL, context TEXT DEFAULT '{}'
    )""")
    con.commit(); con.close()

_init_db()

def create_plan(msg: str, generate_fn, model: str) -> Plan:
    plan_prompt = [{
        "role": "system",
        "content": (
            "You are a strategic planner. Given a task, create a detailed execution plan.\n"
            "Output ONLY valid JSON:\n"
            '{"goal": "...", "steps": [\n'
            '  {"id": "s1", "description": "...", "tool": null|"SEARCH"|"CALC"|"EXEC"|"FETCH", "depends_on": []},\n'
            '  {"id": "s2", "description": "...", "tool": null, "depends_on": ["s1"]}\n'
            "]}\n"
            "Rules:\n- Maximum 8 steps\n- Each step must be atomic\n- depends_on must reference prior step IDs"
        )
    }, {"role": "user", "content": f"Create a plan for: {msg[:600]}"}]

    try:
        raw = generate_fn(plan_prompt, max_tokens=2500, model=model)
        raw = re.sub(r'```json|```', '', raw).strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match: raise ValueError("No JSON found")
        data = json.loads(match.group())
        steps = [PlanStep(
            id=s.get('id', f's{i}'), description=s.get('description', ''),
            tool=s.get('tool'), depends_on=s.get('depends_on', [])
        ) for i, s in enumerate(data.get('steps', []))]
        plan = Plan(id=str(uuid.uuid4())[:8], goal=data.get('goal', msg[:200]), steps=steps)
        _save_plan(plan)
        return plan
    except Exception as e:
        print(f"[Planner] plan creation error: {e}")
        return Plan(id=str(uuid.uuid4())[:8], goal=msg[:200], steps=[PlanStep(id='s1', description=msg[:200])])

def _save_plan(plan: Plan):
    with _lock:
        con = sqlite3.connect(_DB)
        con.execute(
            "INSERT OR REPLACE INTO plans (id, goal, steps, status, created_ts, context) VALUES (?,?,?,?,?,?)",
            (plan.id, plan.goal, json.dumps([asdict(s) for s in plan.steps]),
             plan.status, plan.created_ts, json.dumps(plan.context))
        )
        con.commit(); con.close()

def _execute_step(step: PlanStep, plan: Plan, results: Dict, generate_fn, tool_fn, model: str):
    """Worker function for parallel thread execution."""
    step.status = StepStatus.IN_PROGRESS
    step.start_ts = time.time()
    try:
        if step.tool:
            result = tool_fn(step.tool, step.description)
        else:
            context_str = "\n".join(f"Step {sid}: {res[:200]}" for sid, res in results.items())
            step_prompt = [{
                "role": "system",
                "content": f"You are executing step {step.id} of a plan.\nGoal: {plan.goal}\nPrior results:\n{context_str}"
            }, {"role": "user", "content": step.description}]
            result = generate_fn(step_prompt, max_tokens=2000, model=model)
        
        step.result = result[:500] if result else "(no output)"
        step.status = StepStatus.COMPLETED
    except Exception as e:
        step.error = str(e)
        step.status = StepStatus.FAILED
        # Attempt repair directly in the thread
        try:
            repair_prompt = [{"role": "system", "content": "A plan step failed. Provide a direct answer or workaround."},
                             {"role": "user", "content": f"Step failed: {step.description}\nError: {e}\nProvide result directly:"}]
            fallback = generate_fn(repair_prompt, max_tokens=1500, model=model)
            step.result = f"[REPAIRED] {fallback[:400]}"
            step.status = StepStatus.COMPLETED
        except:
            pass
    step.end_ts = time.time()
    return step.id, step.result

def execute_plan(plan: Plan, generate_fn, tool_fn, model: str) -> str:
    """Upgraded: Executes independent steps in parallel using ThreadPoolExecutor."""
    results = {}
    max_iterations = len(plan.steps) * 2

    for iteration in range(max_iterations):
        executable = []
        for step in plan.steps:
            if step.status != StepStatus.PENDING: continue
            deps_done = all(any(s.id == dep and s.status == StepStatus.COMPLETED for s in plan.steps) for dep in step.depends_on) if step.depends_on else True
            if deps_done: executable.append(step)

        if not executable: break

        # Upgraded: Run independent steps in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_execute_step, step, plan, results, generate_fn, tool_fn, model): step
                for step in executable
            }
            for future in as_completed(futures):
                step_id, step_result = future.result()
                results[step_id] = step_result
                # Update the actual step object in the plan
                for s in plan.steps:
                    if s.id == step_id:
                        s.result = step_result
                        if s.status == StepStatus.COMPLETED:
                            s.end_ts = time.time()
                        break

        if not [s for s in plan.steps if s.status == StepStatus.PENDING]: break

    plan.status = "completed"
    _save_plan(plan)

    completed = [s for s in plan.steps if s.status == StepStatus.COMPLETED]
    if not completed: return "Plan execution produced no results."
    
    synthesis_parts = [f"**Goal:** {plan.goal}\n"]
    for step in completed:
        if step.result: synthesis_parts.append(f"**{step.description}:**\n{step.result}")
    return "\n\n".join(synthesis_parts)

def plan_format_display(plan: Plan) -> str:
    lines = [f"**Plan: {plan.goal}**\n"]
    status_icons = {StepStatus.PENDING: "⬜", StepStatus.IN_PROGRESS: "🔄", StepStatus.COMPLETED: "✅", StepStatus.FAILED: "❌", StepStatus.SKIPPED: "⏭️"}
    for i, step in enumerate(plan.steps, 1):
        icon = status_icons.get(step.status, "⬜")
        dep_str = f" (after {', '.join(step.depends_on)})" if step.depends_on else ""
        lines.append(f"{icon} **Step {i}:** {step.description}{dep_str}")
    return "\n".join(lines)
