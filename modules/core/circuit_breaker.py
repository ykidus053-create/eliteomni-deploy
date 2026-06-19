"""
Production circuit breaker for Groq, Mistral, SearXNG, ChromaDB.
Prevents cascade failures. Auto-heals after recovery_timeout.
"""
import threading, time

class CircuitBreaker:
    CLOSED   = "closed"    # normal operation
    OPEN     = "open"      # failing — reject fast
    HALF_OPEN= "half_open" # testing recovery

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0):
        self.name      = name
        self._threshold= failure_threshold
        self._timeout  = recovery_timeout
        self._failures = 0
        self._opened_at= 0.0
        self._state    = self.CLOSED
        self._lock     = threading.Lock()
        self._calls    = 0
        self._rejected = 0

    @property
    def state(self) -> str:
        with self._lock: return self._state

    def call(self, fn, *args, **kwargs):
        with self._lock:
            self._calls += 1
            if self._state == self.OPEN:
                if time.time() - self._opened_at >= self._timeout:
                    self._state = self.HALF_OPEN
                    print(f"[CB:{self.name}] HALF-OPEN — testing")
                else:
                    self._rejected += 1
                    raise RuntimeError(
                        f"[CircuitBreaker:{self.name}] OPEN — "
                        f"retry in {self._timeout-(time.time()-self._opened_at):.0f}s")

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self._failures = 0
            self._state    = self.CLOSED

    def _on_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._state    = self.OPEN
                self._opened_at= time.time()
                print(f"[CB:{self.name}] OPEN after {self._failures} failures")

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "name":     self.name,
                "state":    self._state,
                "failures": self._failures,
                "calls":    self._calls,
                "rejected": self._rejected,
            }

# Module-level breakers — one per external dependency
breaker_groq    = CircuitBreaker("groq",    failure_threshold=5, recovery_timeout=30)
breaker_mistral = CircuitBreaker("mistral", failure_threshold=5, recovery_timeout=30)
breaker_searxng = CircuitBreaker("searxng", failure_threshold=3, recovery_timeout=20)
breaker_chroma  = CircuitBreaker("chroma",  failure_threshold=3, recovery_timeout=60)
