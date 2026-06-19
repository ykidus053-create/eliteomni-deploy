from __future__ import annotations
import os, time, urllib.request

def check_all() -> dict:
    results = {}
    overall = True

    mk = os.environ.get("MISTRAL_API_KEY","")
    results["mistral_api_key"] = {
        "ok": bool(mk),
        "detail": "set" if mk else "MISSING — add MISTRAL_API_KEY to .env"}
    if not mk:
        overall = False

    gk = os.environ.get("GROQ_API_KEY","")
    results["groq_api_key"] = {
        "ok": bool(gk),
        "detail": "set" if gk else "not set (optional — needed for vision)"}

    try:
        from modules.memory_fast import stats
        results["memory_db"] = {"ok":True, **stats()}
    except Exception as e:
        results["memory_db"] = {"ok":False,"error":str(e)}
        overall = False

    try:
        from modules.model_router import CircuitState
        cb = CircuitState.stats()
        open_cb = [m for m,s in cb.items() if s.get("open")]
        results["circuit_breakers"] = {"ok":not open_cb,"open":open_cb,"stats":cb}
    except Exception as e:
        results["circuit_breakers"] = {"ok":True,"error":str(e)}

    try:
        su = os.environ.get("SEARXNG_URL","http://localhost:8888")
        r  = _real_urlopen(f"{su}/healthz", timeout=3)
        results["searxng"] = {"ok":r.status==200,"url":su}
    except Exception as e:
        results["searxng"] = {"ok":False,"detail":str(e)[:80]}

    try:
        from modules.llm_client import cache_stats
        results["dedup_cache"] = {"ok":True, **cache_stats()}
    except Exception as e:
        results["dedup_cache"] = {"ok":True,"error":str(e)}

    results["overall"]   = overall
    results["pipeline"]  = "v2"
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return results
