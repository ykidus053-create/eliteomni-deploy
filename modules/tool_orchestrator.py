"""
Tool Orchestration System
- Reliability tracking per tool
- Parallel execution
- Fallback chains
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class ToolOrchestrator:
    def __init__(self):
        self.reliability: dict[str, list[bool]] = {}
        self._lock = threading.Lock()

    def record(self, tool: str, success: bool):
        with self._lock:
            if tool not in self.reliability:
                self.reliability[tool] = []
            self.reliability[tool].append(success)
            if len(self.reliability[tool]) > 50:
                self.reliability[tool] = self.reliability[tool][-50:]

    def score(self, tool: str) -> float:
        results = self.reliability.get(tool, [])
        if not results:
            return 1.0
        return sum(results) / len(results)

    def run_with_fallback(self, fns: list, *args, **kwargs):
        """Try each fn in order, return first success."""
        for fn in fns:
            try:
                result = fn(*args, **kwargs)
                self.record(fn.__name__, True)
                if result:
                    return result
            except Exception:
                self.record(fn.__name__, False)
        return None

    def run_parallel(self, tasks: list[tuple]) -> list:
        """Run (fn, args) pairs in parallel, return results."""
        results = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(fn, *args): i for i, (fn, args) in enumerate(tasks)}
            ordered = [None] * len(tasks)
            for f in as_completed(futures):
                idx = futures[f]
                try:
                    ordered[idx] = f.result()
                except Exception:
                    ordered[idx] = None
        return ordered


_orchestrator = ToolOrchestrator()

def get_orchestrator() -> ToolOrchestrator:
    return _orchestrator
