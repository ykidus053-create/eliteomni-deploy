"""
Adaptive Contextual Memory
- User style model
- Cross-session pattern recognition
- Relevance-scored retrieval
"""
import json
import os
import time

MEMORY_PATH = "/home/kidus/eliteomni_adaptive_memory.json"

class AdaptiveMemory:
    def __init__(self):
        self.interactions: list[dict] = []
        self.preferences: dict[str, str] = {}
        self._load()

    def record(self, user_msg: str, assistant_msg: str, skill: str, rating: int = 0):
        self.interactions.append({
            "ts": time.time(),
            "user": user_msg[:300],
            "assistant": assistant_msg[:500],
            "skill": skill,
            "rating": rating
        })
        # Keep last 200
        if len(self.interactions) > 200:
            self.interactions = self.interactions[-200:]
        self._save()

    def set_preference(self, key: str, value: str):
        self.preferences[key] = value
        self._save()

    def get_relevant(self, query: str, top_k: int = 3) -> list[str]:
        q = query.lower()
        scored = []
        for item in self.interactions:
            score = 0
            if q in item["user"].lower():
                score += 2
            if item["rating"] > 0:
                score += item["rating"]
            recency = 1 / (1 + (time.time() - item["ts"]) / 86400)
            scored.append((score + recency, item["assistant"]))
        scored.sort(reverse=True)
        return [s[1] for s in scored[:top_k] if s[0] > 0]

    def proactive_hint(self, query: str) -> str:
        relevant = self.get_relevant(query, top_k=1)
        if relevant:
            return f"[Memory: Similar question answered before — {relevant[0][:120]}...]"
        return ""

    def _save(self):
        try:
            with open(MEMORY_PATH, "w") as f:
                json.dump({"interactions": self.interactions, "preferences": self.preferences}, f)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(MEMORY_PATH):
                with open(MEMORY_PATH) as f:
                    data = json.load(f)
                self.interactions = data.get("interactions", [])
                self.preferences = data.get("preferences", {})
        except Exception:
            pass


_adaptive = AdaptiveMemory()

def adaptive_record(user_msg: str, assistant_msg: str, skill: str, rating: int = 0):
    _adaptive.record(user_msg, assistant_msg, skill, rating)

def adaptive_hint(query: str) -> str:
    return _adaptive.proactive_hint(query)

def adaptive_set_pref(key: str, value: str):
    _adaptive.set_preference(key, value)
