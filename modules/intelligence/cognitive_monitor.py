"""
Cognitive Monitor — Meta-cognitive oversight of the AI's own reasoning.
Detects: reasoning loops, confidence miscalibration, scope creep,
context rot, and goal drift.
"""
import re, time
from collections import deque
from typing import List, Optional

class CognitiveState:
    def __init__(self, session_id: str):
        self.session = session_id
        self.turn = 0
        self.response_lengths: deque = deque(maxlen=10)
        self.topics: deque = deque(maxlen=8)
        self.correction_count = 0
        self.loop_signatures: deque = deque(maxlen=6)
        self.last_goal: str = ""
        self.goal_drift_score = 0.0
        self.overconfidence_events = 0
        self.start_time = time.time()

    def update(self, user_msg: str, response: str):
        self.turn += 1
        self.response_lengths.append(len(response))

        # Topic tracking
        topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', user_msg)
        self.topics.extend(topics[:3])

        # Loop detection: repeated phrase signatures
        sig = response[:60].lower().strip()
        if sig in self.loop_signatures:
            self.correction_count += 1
        self.loop_signatures.append(sig)

        # Overconfidence detection
        absolutes = len(re.findall(
            r'\b(definitely|certainly|absolutely|guaranteed|always|never|100%)\b',
            response, re.I))
        hedges = len(re.findall(
            r'\b(approximately|likely|probably|might|may|could|I think|suggests)\b',
            response, re.I))
        if absolutes > 0 and hedges == 0:
            self.overconfidence_events += 1

    def is_looping(self) -> bool:
        if len(self.loop_signatures) < 4:
            return False
        unique = len(set(list(self.loop_signatures)[-4:]))
        return unique < 3

    def has_context_rot(self) -> bool:
        if len(self.response_lengths) < 4:
            return False
        recent = list(self.response_lengths)[-4:]
        # Declining quality signal: very short responses after normal ones
        avg_early = sum(list(self.response_lengths)[:-4]) / max(len(self.response_lengths)-4, 1)
        avg_recent = sum(recent) / 4
        return avg_early > 500 and avg_recent < avg_early * 0.3

    def is_overconfident(self) -> bool:
        return self.overconfidence_events > self.turn * 0.4 and self.turn >= 3

    def get_intervention(self) -> Optional[str]:
        """Return intervention prompt if cognitive issue detected."""
        if self.is_looping():
            return ("\n[COGNITIVE MONITOR] Loop detected. Stop and reframe entirely. "
                    "Approach this from a completely different angle.")
        if self.has_context_rot():
            return ("\n[COGNITIVE MONITOR] Context degradation detected. "
                    "Restate the core goal and key established facts before continuing.")
        if self.is_overconfident():
            return ("\n[COGNITIVE MONITOR] Overconfidence pattern detected. "
                    "Increase epistemic humility. Add appropriate hedges and caveats.")
        return None

    def get_state_summary(self) -> str:
        return (f"Turn {self.turn} | "
                f"Loops: {self.correction_count} | "
                f"Overconfidence events: {self.overconfidence_events} | "
                f"Context rot: {self.has_context_rot()}")

_states: dict = {}

def get_cognitive_state(session_id: str) -> CognitiveState:
    if session_id not in _states:
        _states[session_id] = CognitiveState(session_id)
    return _states[session_id]
