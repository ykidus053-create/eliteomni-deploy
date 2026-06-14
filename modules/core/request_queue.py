"""
Async inference queue with backpressure.
Hassabis: without this, a single slow LLM call blocks all concurrent users.
At 100M users this is the difference between 50 RPS and 5000 RPS.
"""
import asyncio
from typing import Callable, Any

_QUEUE_MAX  = 100
_queue: asyncio.Queue | None = None

def get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue(maxsize=_QUEUE_MAX)
    return _queue

async def enqueue_inference(
    fn: Callable,
    *args,
    timeout: float = 30.0,
    **kwargs
) -> Any:
    """
    Enqueue an inference call. Returns 503 dict if queue full.
    Applies 30s hard timeout to prevent indefinite hangs.

    Args:
        fn: Async or sync callable to execute.
        timeout: Hard timeout in seconds.

    Returns:
        Result of fn(*args, **kwargs) or error dict.

    Raises:
        asyncio.TimeoutError: If fn exceeds timeout.
    """
    q = get_queue()
    if q.full():
        return {
            "error": "server_busy",
            "message": "Too many concurrent requests. Retry in a moment.",
            "retry_after": 2
        }
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    await q.put((fn, args, kwargs, future))
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "error": "timeout",
            "message": "Request timed out after 30s.",
            "partial": True
        }

async def queue_worker():
    """Background worker that drains the inference queue."""
    q = get_queue()
    while True:
        fn, args, kwargs, future = await q.get()
        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
            if not future.done():
                future.set_result(result)
        except Exception as e:
            if not future.done():
                future.set_exception(e)
        finally:
            q.task_done()
