"""
Rate limiter with TTL eviction — prevents unbounded memory growth at scale.
Replaces: _rate_lim: dict = defaultdict(list)
"""
import threading, time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int = 60, window: int = 60,
                 evict_interval: int = 300):
        self._lock    = threading.Lock()
        self._store   = defaultdict(list)   # {ip: [timestamps]}
        self._max     = max_requests
        self._window  = window
        # Background eviction — removes stale keys every evict_interval seconds
        threading.Thread(target=self._evict_loop, daemon=True,
                         name="ratelim_evict",
                         args=(evict_interval,)).start()

    def check(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            self._store[ip] = [t for t in self._store[ip] if now - t < self._window]
            if len(self._store[ip]) >= self._max:
                return False
            self._store[ip].append(now)
            return True

    def _evict_loop(self, interval: int):
        while True:
            time.sleep(interval)
            cutoff = time.time() - self._window
            with self._lock:
                dead = [ip for ip, ts in self._store.items()
                        if not ts or ts[-1] < cutoff]
                for ip in dead:
                    del self._store[ip]
            if dead:
                print(f"[RateLimiter] evicted {len(dead)} stale keys")

rate_limiter = RateLimiter()
