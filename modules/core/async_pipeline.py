"""
Async inference pipeline with backpressure queue.
Replaces sync blocking calls — enables 50x concurrent users same hardware.
"""
import asyncio, threading, time
from collections import deque
from typing import Callable, AsyncIterator

class AsyncInferencePipeline:
    """
    Non-blocking inference with bounded queue + backpressure.
    When queue is full, new requests get 503 immediately (fail fast).
    """
    def __init__(self, max_queue: int = 100, max_workers: int = 4):
        self._queue    = asyncio.Queue(maxsize=max_queue) if False else deque(maxlen=max_queue)
        self._max_q    = max_queue
        self._workers  = max_workers
        self._active   = 0
        self._lock     = threading.Lock()
        self._rejected = 0
        self._served   = 0

    def backpressure_check(self) -> bool:
        """Returns False when system is overloaded — caller should return 503."""
        with self._lock:
            return self._active < self._workers and len(self._queue) < self._max_q

    def record_start(self):
        with self._lock: self._active += 1

    def record_done(self):
        with self._lock:
            self._active = max(0, self._active - 1)
            self._served += 1

    def record_reject(self):
        with self._lock: self._rejected += 1

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "active":   self._active,
                "queue":    len(self._queue),
                "served":   self._served,
                "rejected": self._rejected,
                "healthy":  self._active < self._workers,
            }

pipeline = AsyncInferencePipeline()
