import time
import threading

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int, recovery_timeout: float):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._probe_in_progress = False
        self._lock = threading.Lock()

    def call(self, func, expected_exceptions=None):
        return self.inspect_and_call(func, expected_exceptions)

    def inspect_and_call(self, func, expected_exceptions=None):
        if expected_exceptions is None:
            expected_exceptions = (Exception,)

        with self._lock:
            now = time.time()
            
            # State evaluation: Open to Half-Open probe window
            if self.state == "open":
                if now - self.last_failure_time >= self.recovery_timeout:
                    if not self._probe_in_progress:
                        self._probe_in_progress = True
                    else:
                        return None  # Fast-fail concurrent threads during active probe
                else:
                    return None  # Fast-fail while cooldown is active

        try:
            result = func()
            with self._lock:
                if self._probe_in_progress:
                    self.state = "closed"
                    self.failure_count = 0
                    self._probe_in_progress = False
            return result
        except Exception as e:
            with self._lock:
                is_tripping = any(isinstance(e, ex) for ex in expected_exceptions)
                
                if self._probe_in_progress:
                    self._probe_in_progress = False
                    self.state = "open"
                    self.last_failure_time = time.time()
                    raise e
                
                if self.state == "closed" and is_tripping:
                    self.failure_count += 1
                    if self.failure_count >= self.failure_threshold:
                        self.state = "open"
                        self.last_failure_time = time.time()
                raise e

# Instantiate global service tracking breakers expected by the tools engine
arxiv_cb = CircuitBreaker("arxiv-service", failure_threshold=3, recovery_timeout=5.0)
wolfram_cb = CircuitBreaker("wolfram-service", failure_threshold=3, recovery_timeout=5.0)
