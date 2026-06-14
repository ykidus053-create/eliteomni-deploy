"""
Progressive Loading: loads context in stages rather than all at once.
Stage 0 (always): identity + constraints + skill (500 chars)
Stage 1 (medium+): tools + memory + recent history (1500 chars)
Stage 2 (hard): exemplars + episodic + world model (2500 chars)
Stage 3 (hard+long): full context with compaction applied

Sub-Agent spawner: decomposes tasks and runs sub-agents in parallel.
"""
import re, time, threading, json
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

@dataclass
class ContextStage:
    stage: int
    content: str
    token_estimate: int
    load_time_ms: float = 0.0

@dataclass
class SubAgentTask:
    id: str
    description: str
    skill: str
    context: str
    result: str = ""
    status: str = "pending"
    confidence: float = 0.0
    error: str = ""

class ProgressiveContextLoader:
    """
    Loads context in stages. Each stage adds more depth.
    Allows streaming to start faster (stage 0 only) while
    deeper context loads in background.
    """
    def __init__(self):
        self.stages: List[ContextStage] = []
        self._lock = threading.Lock()

    def load_stage_0(self, identity: str, constraints: str, skill_prompt: str) -> str:
        t0 = time.time()
        parts = []
        if identity:
            parts.append(identity[:300])
        if constraints:
            parts.append(constraints[:200])
        if skill_prompt:
            parts.append(skill_prompt[:200])
        content = "\n".join(parts)
        stage = ContextStage(0, content, len(content)//4,
                             (time.time()-t0)*1000)
        with self._lock:
            self.stages = [stage]
        return content

    def load_stage_1(self, tools: str, memory_ctx: str,
                     recent_history: List[Dict]) -> str:
        t0 = time.time()
        parts = []
        if tools:
            parts.append(tools[:200])
        if memory_ctx:
            parts.append(memory_ctx[:600])
        if recent_history:
            hist_text = "\n".join(
                f"{m.get('role','?')}: {str(m.get('content',''))[:120]}"
                for m in recent_history[-4:]
            )
            parts.append(f"[RECENT]\n{hist_text}")
        content = "\n".join(parts)
        stage = ContextStage(1, content, len(content)//4,
                             (time.time()-t0)*1000)
        with self._lock:
            self.stages.append(stage)
        return content

    def load_stage_2(self, exemplars: str, episodic: str,
                     world_model: str = "") -> str:
        t0 = time.time()
        parts = []
        if exemplars:
            parts.append(exemplars[:600])
        if episodic:
            parts.append(episodic[:400])
        if world_model:
            parts.append(world_model[:300])
        content = "\n".join(parts)
        stage = ContextStage(2, content, len(content)//4,
                             (time.time()-t0)*1000)
        with self._lock:
            self.stages.append(stage)
        return content

    def get_full_context(self, complexity: str = "medium") -> str:
        with self._lock:
            if complexity == "easy":
                return "\n\n".join(s.content for s in self.stages[:1])
            if complexity == "medium":
                return "\n\n".join(s.content for s in self.stages[:2])
            return "\n\n".join(s.content for s in self.stages)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "stages_loaded": len(self.stages),
                "total_tokens": sum(s.token_estimate for s in self.stages),
                "load_times_ms": [round(s.load_time_ms, 1) for s in self.stages],
            }

DECOMPOSITION_PROMPT = """Decompose this task into 2-4 independent sub-tasks.
Each sub-task should be completable without the others.
Output JSON array: [{"id":"t1","desc":"...","skill":"general|coder|researcher|calculator"}]
Task: {task}
Output only the JSON array, nothing else."""

class SubAgentSpawner:
    """
    Decomposes complex tasks and runs sub-agents in parallel.
    Each sub-agent gets its own context and skill routing.
    Results are synthesized by the orchestrator.
    """
    def __init__(self, generate_fn: Callable):
        self.generate = generate_fn
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent")

    def should_decompose(self, msg: str, complexity: str) -> bool:
        if complexity != "hard":
            return False
        decompose_signals = [
            r"\b(and then|also|additionally|furthermore|moreover)\b",
            r"\b(first|second|third|finally|lastly)\b",
            r"\b(compare|contrast|analyze and|research and)\b",
            r"\?.*\?",
        ]
        signal_count = sum(1 for p in decompose_signals
                          if re.search(p, msg, re.IGNORECASE))
        return signal_count >= 2 or len(msg) > 500

    def decompose(self, msg: str) -> List[SubAgentTask]:
        try:
            prompt = DECOMPOSITION_PROMPT.format(task=msg[:400])
            raw = self.generate([{"role": "user", "content": prompt}],
                                 300, "general", len(msg))
            raw = raw.strip()
            raw = re.sub(r'^```(?:json)?', '', raw).strip()
            raw = re.sub(r'```$', '', raw).strip()
            tasks_data = json.loads(raw)
            tasks = []
            for i, t in enumerate(tasks_data[:4]):
                tasks.append(SubAgentTask(
                    id=t.get("id", f"t{i+1}"),
                    description=t.get("desc", "")[:200],
                    skill=t.get("skill", "general"),
                    context=msg[:300],
                ))
            return tasks
        except Exception as e:
            print(f"[SubAgent] decompose error: {e}")
            return []

    def run_subtask(self, task: SubAgentTask) -> SubAgentTask:
        prompt = f"""Complete this specific sub-task. Be concise and precise.
Sub-task: {task.description}
Full context: {task.context[:400]}
Output only the result for this specific sub-task."""
        try:
            result = self.generate(
                [{"role": "user", "content": prompt}],
                800, task.skill, len(task.description))
            task.result = result or ""
            task.status = "completed"
            task.confidence = 0.8 if result and len(result) > 50 else 0.4
        except Exception as e:
            task.status = "failed"
            task.error = str(e)[:100]
            task.confidence = 0.0
        return task

    def run_parallel(self, tasks: List[SubAgentTask],
                     timeout: float = 25.0) -> List[SubAgentTask]:
        if not tasks:
            return []
        futures = {
            self.executor.submit(self.run_subtask, task): task
            for task in tasks
        }
        completed = []
        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    completed.append(future.result())
                except Exception as e:
                    task = futures[future]
                    task.status = "failed"
                    task.error = str(e)[:100]
                    completed.append(task)
        except TimeoutError:
            print("[SubAgent] timeout — using partial results")
            for task in tasks:
                if task.status == "pending":
                    task.status = "timeout"
                    completed.append(task)
        return completed

    def synthesize(self, original_msg: str,
                   completed_tasks: List[SubAgentTask]) -> str:
        if not completed_tasks:
            return ""
        good = [t for t in completed_tasks if t.status == "completed"
                and t.confidence >= 0.4]
        if not good:
            return ""
        parts = [f"[Sub-task {t.id}: {t.description[:60]}]\n{t.result[:400]}"
                 for t in good]
        synthesis_prompt = f"""Synthesize these sub-task results into a coherent answer.
Original question: {original_msg[:300]}

Sub-task results:
{chr(10).join(parts)}

Write a unified, coherent answer. Do not list sub-tasks — integrate the results."""
        try:
            return self.generate(
                [{"role": "user", "content": synthesis_prompt}],
                1500, "general", len(original_msg))
        except Exception as e:
            print(f"[SubAgent] synthesis error: {e}")
            return "\n\n".join(t.result for t in good if t.result)

_loaders: Dict[str, ProgressiveContextLoader] = {}
_spawner: Optional[SubAgentSpawner] = None

def get_loader(session_id: str = "default") -> ProgressiveContextLoader:
    if session_id not in _loaders:
        _loaders[session_id] = ProgressiveContextLoader()
    return _loaders[session_id]

def get_spawner(generate_fn: Callable) -> SubAgentSpawner:
    global _spawner
    if _spawner is None:
        _spawner = SubAgentSpawner(generate_fn)
    return _spawner
