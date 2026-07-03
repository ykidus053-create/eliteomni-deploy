# AUTO-SPLIT FROM app.py lines 3379-3637
# ── Parallel module loader — imports all heavy modules concurrently ──────────
import importlib, threading as _th
_mod_results = {}
_mod_errors  = {}

def _load(name):
    try:
        _mod_results[name] = importlib.import_module(name)
    except Exception as e:
        _mod_errors[name] = e

_threads = [_th.Thread(target=_load, args=(m,), daemon=True)
            for m in ("book_gaps_impl","aie_book_impl","final_gaps","book8_gaps")]
for t in _threads: t.start()
for t in _threads: t.join()

# Inject symbols into global namespace exactly as before
def _inject(mod_name, symbols):
    mod = _mod_results.get(mod_name)
    if mod is None:
        print(f"[{mod_name}] ❌ {_mod_errors.get(mod_name)}")
        return False
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
    # groq_moderate was previously log-only; now blocks and redirects
    _mod = groq_moderate(msg)
    if not _mod.get("safe", True):
        _cat = _mod.get("category") or "policy violation"
        return {"response": (
            f"I can't help with that request ({_cat}). "
            "If you're in distress, please reach out to a trusted person or a crisis line."
        ), "skill": "safety", "mode": "fast", "vetoed": True, "effort": EFFORT_LEVEL}

    # ── LlamaGuard active moderation gate (FIX 4) ────────────────────────────
