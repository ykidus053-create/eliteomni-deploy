
# ── TTFT FIX: parallel pre-fetch with hard timeout ──────────────────────────
import concurrent.futures as _cfu, time as _tfix

_TTFT_POOL = _cfu.ThreadPoolExecutor(max_workers=8, thread_name_prefix="ttft")

def _build_stream_context_fast(msg: str, hist: list) -> dict:
    from modules.core.constants import get_infra_tier
    skill      = classify_skill(msg)
    complexity = route_complexity(msg)
    _tier      = get_infra_tier(complexity)
    effort     = "high" if complexity=="hard" else ("low" if complexity=="easy" else "medium")

    cached = cache_get(msg, skill)
    if cached and complexity == "easy":
        return {"cached": cached, "skill": skill, "complexity": complexity,
                "mode": "cached", "effort": effort, "msgs": [], "max_t": 0}

    clean_msg, search_ctx = extract_search_context(msg)

    # ALL I/O in parallel — hard 2s wall clock budget
    futs = {
        "hist":    _TTFT_POOL.submit(lambda: compress_history(_strip_thinking_from_history(compress_history(hist)[0] if _count_tokens(hist)>6000 else hist))),
        "mem":     _TTFT_POOL.submit(lambda: mem_get(msg, k=3)),
        "sem":     _TTFT_POOL.submit(lambda: semantic_mem_get(msg, k=4)),
        "sqlite":  _TTFT_POOL.submit(lambda: db_mem_get(msg, k=3) if True else []),
        "episodic":_TTFT_POOL.submit(lambda: mem_get_episodic(msg)),
        "rag":     _TTFT_POOL.submit(lambda: rag_get(msg, k=3)),
        "rlhf":    _TTFT_POOL.submit(lambda: get_rlhf_note(skill)),
    }

    # Search runs in parallel too — NOT blocking generation
    search_fut = None
    if not search_ctx and complexity != "easy" and not msg.startswith("[VISION_CONTEXT:"):
        search_fut = _TTFT_POOL.submit(lambda: tool_search(msg[:300]))

    _deadline = 2.0
    def _get(k, default):
        try: return futs[k].result(timeout=_deadline)
        except Exception: return default

    recent, ctx_sum  = _get("hist", ([], ""))
    _mem_working     = _get("mem", [])
    sem_memory       = _get("sem", [])
    _mem_sqlite      = _get("sqlite", [])
    _mem_episodic    = _get("episodic", [])
    rag_hits         = _get("rag", [])
    rlhf_note        = _get("rlhf", "")

    # Collect search result if it finished — don't wait if still running
    if search_fut:
        try:
            r = search_fut.result(timeout=0.05)
            if r and "error" not in r.lower():
                search_ctx = r[:2000]
        except Exception:
            pass  # didn't finish in time — skip, don't block

    _seen_m, memory = set(), []
    for _src in [_mem_working, _mem_sqlite, sem_memory[:3], _mem_episodic[:2]]:
        for _m in (_src or []):
            _txt = _m if isinstance(_m, str) else str(_m)
            _k = _txt[:80].lower()
            if _k not in _seen_m:
                _seen_m.add(_k); memory.append(_txt)
    memory = memory[:8]

    episodic = list(_mem_episodic or [])
    if ctx_sum:
        mem_save_episodic(ctx_sum)
        episodic = [ctx_sum] + episodic

    rag_ctx = ("\n[KNOWLEDGE BASE]\n" +
               "\n".join("- " + r.get("text","")[:200] for r in rag_hits) +
               "\n[END KNOWLEDGE BASE]") if rag_hits else ""

    # Cache key that doesn't thrash
    _sys_key = f"{skill}:{complexity}"
    if _sys_key not in _sys_prompt_cache:
        _sys_prompt_cache[_sys_key] = build_system_prompt(skill, memory, episodic, rlhf_note, ctx_sum or "", complexity)
    system = _sys_prompt_cache[_sys_key]

    if rag_ctx: system += rag_ctx
    if search_ctx:
        import datetime
        system += (f"\n\n[WEB TODAY={datetime.date.today()}]\n{search_ctx[:2000]}\n[/WEB]")

    mcp_p = mcp_tool_list_prompt()
    if mcp_p: system += f"\n\n{mcp_p}"

    hist_msgs = []
    for h in (recent or [])[-8:]:
        r2 = h.get("role","user"); c2 = h.get("content","").strip()
        if c2 and len(c2) > 2:
            hist_msgs.append({"role": r2, "content": c2})

    max_t = get_optimal_max_tokens(msg, skill, complexity) if _CTXBUDGET else 16000
    mode  = "extended_think" if effort=="high" else ("think" if effort=="medium" else "fast")
    msgs  = build_chatml(system, hist_msgs, clean_msg)

    return {
        "cached": None, "skill": skill, "complexity": complexity,
        "mode": mode, "effort": effort, "msgs": msgs,
        "max_t": max_t, "system": system, "msg": msg, "model": _tier["models"][0],
    }
