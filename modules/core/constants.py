# AUTO-SPLIT FROM app.py lines 340-449
import os, re, time, math, json, asyncio, random, ast, subprocess, sys, tempfile
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import urllib.request
try:
    from debug_patch import _real_urlopen
except ImportError:
    import urllib.request
    _real_urlopen = urllib.request.urlopen

# FIXED: Updated default path to include the 'eliteomni' subfolder to match user's actual path
GGUF_MODEL_PATH = os.environ.get(
    "GGUF_MODEL_PATH",
    "/mnt/c/Users/kidus yared/Downloads/eliteomni/"
)

# SearXNG base URL — matches the docker-compose setup below
# Override with env var SEARXNG_URL if needed
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")

# ── SEARXNG HEALTH TRACKING ───────────────────────────────────────────────────
_searxng_healthy   = False   # flips to True once a probe succeeds
_searxng_last_ok   = 0.0    # epoch of last successful probe
_searxng_fail_count = 0     # consecutive failures
_searxng_lock      = Lock()

def _probe_searxng(timeout: float = 15.0) -> bool:
    """Return True if SearXNG answers a health-check ping."""
    try:
        import requests as _req
        r = _req.get(
            f"{SEARXNG_URL}/search",
            params={"q": "test", "format": "json", "engines": "google"},
            timeout=10
        )
        return r.status_code == 200
    except Exception as _e:
        print(f"[SearXNG] probe exception: {type(_e).__name__}: {_e}")
        return False

def _ensure_searxng() -> bool:
    """
    Check SearXNG health; attempt one Docker restart if it's down.
    Returns True when SearXNG is confirmed up.
    Thread-safe.
    """
    global _searxng_healthy, _searxng_last_ok, _searxng_fail_count
    with _searxng_lock:
        # Re-use a recent OK result (cache for 30 s)
        if _searxng_healthy and (time.time() - _searxng_last_ok) < 30:
            return True

        if _probe_searxng():
            _searxng_healthy   = True
            _searxng_last_ok   = time.time()
            _searxng_fail_count = 0
            return True

        _searxng_healthy = False
        _searxng_fail_count += 1
        print(f"[SearXNG] probe failed (streak={_searxng_fail_count}), attempting restart…")

        # Try to restart via Docker (works in WSL + Docker Desktop)
        for cmd in (
            ["docker", "restart", "searxng"],
            ["docker", "compose", "restart", "searxng"],
            ["docker-compose", "restart", "searxng"],
        ):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=20
                )
                if result.returncode == 0:
                    print(f"[SearXNG] restart issued via: {' '.join(cmd)}")
                    time.sleep(4)   # give it a moment to come up
                    if _probe_searxng(timeout=6):
                        _searxng_healthy   = True
                        _searxng_last_ok   = time.time()
                        _searxng_fail_count = 0
                        print("[SearXNG] back online after restart ✓")
                        return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        print("[SearXNG] could not restart — will answer from knowledge cache")
        return False

def _background_searxng_watchdog():
    """Daemon thread: probes SearXNG every 60 s and auto-heals if needed."""
    while True:
        try:
            _ensure_searxng()
        except Exception as e:
            print(f"[SearXNG watchdog] unexpected error: {e}")
        time.sleep(60)

N_THREADS    = os.cpu_count() or 4          # use ALL cores
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))   # set >0 if you have a GPU
N_CTX        = 131072
N_BATCH      = 1024                          # doubled batch = faster prefill
MAX_MEM      = 500
RATE_LIMIT   = 60
CTX_WINDOW   = 4  # Point 9

llm               = None
_loaded           = True   # groq mode — no local model needed
_load_status      = "ready"
_load_error       = ""
_gen_lock         = Lock()
_tool_exec        = ThreadPoolExecutor(max_workers=4, thread_name_prefix="eo_tool")
mem_store: list      = []
episodic_store: list = []
faiss_index          = None

# ── PERSISTENT MEMORY (SQLite) ───────────────────────────────────────────────

# FAISS availability (shared across modules)
try:
    import faiss as _faiss
    import numpy as np
    _faiss_ok = True
except ImportError:
    _faiss = None
    np = None
    _faiss_ok = False
faiss_index = None
faiss_texts = []


# Claude-style generation defaults injected by upgrade script
# These mirror Claude's calibrated output behaviour:
# - Low temperature keeps responses focused and coherent
# - Higher repeat penalty prevents the loops/filler Claude avoids
# - No artificial length limits — let the budget system handle it
CLAUDE_STYLE_DEFAULTS = {
    "temperature": 0.15,       # focused but not robotic
    "repeat_penalty": 1.08,    # mild anti-repetition
    "top_p": 0.92,             # nucleus sampling like Claude
    "top_k": 40,               # avoid degenerate tokens
}

# ── CLAUDE-STYLE INFRASTRUCTURE TIER ROUTING ─────────────────────────────────
# Mirrors Anthropic's actual chip strategy:
#   TIER 1 (TPU/Trainium equivalent) → fast, cheap, high-volume inference
#   TIER 2 (TPU Pod equivalent)      → medium complexity, balanced
#   TIER 3 (SuperPod equivalent)     → hard/research, max capability
# PUE target: 1.1 (Google TPU DC standard) — minimize redundant calls
INFRA_TIERS = {
    "tier1": {"models": ["cerebras/zai-glm-4.7"], "complexity": ["easy"], "max_tokens": 512, "label": "fast-inference"},
    "tier2": {"models": ["cerebras/zai-glm-4.7"], "complexity": ["medium"], "max_tokens": 2048, "label": "balanced"},
    "tier3": {"models": ["cerebras/zai-glm-4.7"], "complexity": ["hard", "research"], "max_tokens": 8192, "label": "frontier"},
}

CODING_SKILLS = {"coder", "code", "coding", "swe", "calculator", "debug", "refactor", "engineer"}

def get_infra_tier(complexity: str, skill: str = "") -> dict:
    """Return the optimal model tier for a given complexity.
    Coding skills → devstral-latest. Everything else → mistral-medium-3.5.
    """
    is_code = skill and skill.lower() in CODING_SKILLS
    is_research = skill and skill.lower() in ("researcher", "research")
    if is_code or is_research:
        return {"models": ["cerebras/zai-glm-4.7"], "complexity": [complexity], "max_tokens": 8192, "label": "code-inference"}
    if complexity == "easy":   return INFRA_TIERS["tier1"]
    if complexity == "hard":   return INFRA_TIERS["tier3"]
    return INFRA_TIERS["tier2"]
