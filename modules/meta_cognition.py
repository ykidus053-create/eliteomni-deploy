"""
Meta-Cognition Engine (MCE)
- Confidence tracking per reasoning step
- Loop/dead-end detection
- Alternative hypothesis suggestion
"""
import time
from collections import deque

class ReasoningMonitor:
    def __init__(self):
        self.steps = deque(maxlen=20)
        self.confidence_history = []
        self.start_time = time.time()

    def add_step(self, step: str, confidence: float = 1.0):
        self.steps.append({"step": step, "conf": confidence, "t": time.time()})
        self.confidence_history.append(confidence)

    def detect_loop(self) -> bool:
        recent = [s["step"][:60] for s in self.steps]
        seen = set()
        for s in recent:
            if s in seen:
                return True
            seen.add(s)
        return False

    def avg_confidence(self) -> float:
        if not self.confidence_history:
            return 1.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def should_escalate(self) -> bool:
        return self.avg_confidence() < 0.6 or self.detect_loop()

    def summary(self) -> str:
        return (
            f"Steps: {len(self.steps)} | "
            f"Avg confidence: {self.avg_confidence():.0%} | "
            f"Loop detected: {self.detect_loop()} | "
            f"Elapsed: {time.time()-self.start_time:.1f}s"
        )


_monitor = ReasoningMonitor()

def get_monitor() -> ReasoningMonitor:
    return _monitor

def reset_monitor():
    global _monitor
    _monitor = ReasoningMonitor()
