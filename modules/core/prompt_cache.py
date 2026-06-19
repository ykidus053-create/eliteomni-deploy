"""
System prompt cache with 5-minute TTL.
Key = hash(skill + complexity + user_id).
Eliminates rebuilding 3000-char system prompt on every request.
Expected saving: 40-80ms per request.
"""
import hashlib, time, threading

class PromptCache:
    def __init__(self, ttl: int = 300, max_size: int = 500):
        self._lock  = threading.Lock()
        self._store = {}          # key → {prompt, ts}
        self._ttl   = ttl
        self._max   = max_size

    def _make_key(self, skill: str, complexity: str,
                  user_id: str = "default") -> str:
        raw = f"{skill}:{complexity}:{user_id}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def get(self, skill: str, complexity: str,
            user_id: str = "default") -> str | None:
        key = self._make_key(skill, complexity, user_id)
        with self._lock:
            entry = self._store.get(key)
            if entry and time.time() - entry["ts"] < self._ttl:
                return entry["prompt"]
            return None

    def set(self, skill: str, complexity: str, prompt: str,
            user_id: str = "default"):
        key = self._make_key(skill, complexity, user_id)
        with self._lock:
            if len(self._store) >= self._max:
                # evict oldest
                oldest = min(self._store, key=lambda k: self._store[k]["ts"])
                del self._store[oldest]
            self._store[key] = {"prompt": prompt, "ts": time.time()}

    def invalidate(self, user_id: str = "default"):
        """Call when user updates persistent instructions."""
        with self._lock:
            dead = [k for k in self._store
                    if user_id in k or user_id == "default"]
            for k in dead:
                del self._store[k]

    @property
    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._store), "ttl": self._ttl,
                    "max": self._max}

prompt_cache = PromptCache()
