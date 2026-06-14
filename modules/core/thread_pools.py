"""
Module-level thread pool singletons.
Replaces: ThreadPoolExecutor created per-request (causes thread leak).
Import these instead of creating new executors inside functions.
"""
from concurrent.futures import ThreadPoolExecutor
import os

_cores = os.cpu_count() or 4

# General tool execution (SEARCH, CALC, EXEC, MCP calls)
tool_pool    = ThreadPoolExecutor(max_workers=_cores,
                                  thread_name_prefix="eo_tool")

# Pipeline / agent team parallel inference
pipeline_pool = ThreadPoolExecutor(max_workers=4,
                                   thread_name_prefix="eo_pipeline")

# Agent team specialist workers
agent_pool   = ThreadPoolExecutor(max_workers=6,
                                  thread_name_prefix="eo_agent")

# Background tasks (memory extraction, RLAIF, audit flush)
bg_pool      = ThreadPoolExecutor(max_workers=2,
                                  thread_name_prefix="eo_bg")
