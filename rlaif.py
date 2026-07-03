import importlib, threading as _th, logging
log = logging.getLogger(__name__)
_mod_results: dict = {}
_mod_errors:  dict = {}

def _load(name):
    try:
        _mod_results[name] = importlib.import_module(name)
    except Exception as e:
        _mod_errors[name] = e
        log.warning("[rlaif] failed to load module %s: %s", name, e)

_threads = [_th.Thread(target=_load, args=(m,), daemon=True)
            for m in ("book_gaps_impl", "aie_book_impl", "final_gaps", "book8_gaps")]
for t in _threads: t.start()
for t in _threads: t.join()

def _inject(mod_name: str, symbols: list, *, msg: str, skill: str, complexity: str, history: list, image_b64: str = "") -> dict:
    """
    Bug Fix: Previously generated _rc_injection but discarded it.
    Now properly returns the injection payload dictionary so it can be utilized upstream.
    """
    mod = _mod_results.get(mod_name)
    if mod is None:
        log.error("[%s] not loaded: %s", mod_name, _mod_errors.get(mod_name))
        return {"success": False, "injection": ""}
    
    _session_id = "default"
    _rc_injection = ""
    _rc_meta = {}
    
    try:
        from modules.intelligence.reasoning_core import build_reasoning_core_context
        import hashlib as _hlib
        _session_id = _hlib.md5(str(history[:1]).encode()).hexdigest()[:12] if history else "default"
        _rc_injection, _rc_meta = build_reasoning_core_context(msg, skill, complexity, _session_id)
        log.debug("[ReasoningCore] %d chars | %s", len(_rc_injection), _rc_meta.get("injection_sources", []))
    except Exception as e:
        log.debug("[ReasoningCore] skipped: %s", e)
        
    return {
        "success": True,
        "module": mod_name,
        "session_id": _session_id,
        "injection": _rc_injection,
        "meta": _rc_meta
    }

def critique(response: str, skill: str = "general") -> str:
    try:
        from modules.intelligence.critique import _run_critique
        return _run_critique(response, skill)
    except Exception as e:
        log.debug("[critique] skipped: %s", e)
        return ""
