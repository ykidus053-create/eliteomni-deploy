import threading, time, uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: str
    msg_type: str = "task"
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created: float = field(default_factory=time.time)

@dataclass
class AgentTask:
    id: str
    description: str
    skill: str
    context: str = ""
    result: str = ""
    status: str = "pending"
    created: float = field(default_factory=time.time)
    completed: float = 0.0

class AgentOrchestrator:
    def __init__(self, generate_fn: Callable):
        self.generate = generate_fn
        self.tasks: Dict[str, AgentTask] = {}
        self.results: Dict[str, str] = {}
        self.messages: List[AgentMessage] = []

    def spawn_agent(self, task: AgentTask) -> str:
        system = (
            f"You are a specialized {task.skill} agent.\n"
            f"Your ONLY job: {task.description}\n"
            f"Context: {task.context[:500]}\n"
            "Be concise. Return ONLY the result of your specific task.\n"
            "Do NOT attempt to answer the full question — only your assigned subtask."
        )
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": task.description}]
        try:
            task.status = "running"
            result = self.generate(msgs, max_new=800, skill=task.skill, msg_len=len(task.description))
            task.result = result
            task.status = "done"
            task.completed = time.time()
            return result
        except Exception as e:
            task.status = "failed"
            task.result = f"Agent failed: {e}"
            return task.result

    def run_parallel(self, tasks: List[AgentTask], timeout: float = 30.0) -> Dict[str, str]:
        results = {}
        threads = []
        def _run(t):
            results[t.id] = self.spawn_agent(t)
        for task in tasks:
            self.tasks[task.id] = task
            thread = threading.Thread(target=_run, args=(task,), daemon=True)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join(timeout=timeout / max(len(threads), 1))
        return results

    def synthesize(self, original_goal: str, subtask_results: Dict[str, str],
                   tasks: List[AgentTask]) -> str:
        if not subtask_results:
            return "No agent results to synthesize."
        parts = [f"SYNTHESIS TASK: {original_goal}\n", "AGENT RESULTS:"]
        for task in tasks:
            result = subtask_results.get(task.id, "[no result]")
            parts.append(f"[{task.skill.upper()} AGENT - {task.description[:60]}]:")
            parts.append(result[:400])
            parts.append("")
        parts.append("Synthesize the above agent results into a unified, coherent response.")
        parts.append("Resolve any contradictions. Identify the most reliable findings.")
        parts.append("Produce a final answer that integrates all perspectives.")
        synthesis_msgs = [
            {"role": "system", "content": "You are a synthesis agent. Integrate multiple specialist reports."},
            {"role": "user", "content": "\n".join(parts)}
        ]
        return self.generate(synthesis_msgs, max_new=2000, skill="general", msg_len=500)

def should_use_multi_agent(msg: str, skill: str, complexity: str) -> bool:
    if complexity != "hard":
        return False
    multi_signals = ["compare","analyze and","research and","find and explain",
                     "what are all","comprehensive","thorough","detailed analysis",
                     "multiple","various","across"]
    return any(s in msg.lower() for s in multi_signals)

def decompose_for_agents(msg: str, skill: str) -> List[AgentTask]:
    if skill == "researcher":
        return [
            AgentTask("search_agent", f"Search for: {msg[:100]}", "researcher", msg),
            AgentTask("analysis_agent", f"Analyze the key factors in: {msg[:100]}", "researcher", msg),
            AgentTask("critique_agent", f"Identify weaknesses and counterarguments for: {msg[:100]}", "general", msg),
        ]
    elif skill == "coder":
        return [
            AgentTask("design_agent", f"Design the architecture for: {msg[:100]}", "coder", msg),
            AgentTask("impl_agent", f"Implement the core logic for: {msg[:100]}", "coder", msg),
            AgentTask("review_agent", f"Security and correctness review for: {msg[:100]}", "coder", msg),
        ]
    else:
        return [
            AgentTask("main_agent", msg[:200], skill, msg),
            AgentTask("critic_agent", f"Critique and verify: {msg[:100]}", "general", msg),
        ]
