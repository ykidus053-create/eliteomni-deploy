# AUTO-SPLIT FROM app.py lines 2744-2837
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
    _session_id = "default"
    try:
        from modules.intelligence.reasoning_core import build_reasoning_core_context, post_turn_update
        import hashlib as _hlib
        _session_id = _hlib.md5(str(history[:1]).encode()).hexdigest()[:12] if history else "default"
        _rc_injection, _rc_meta = build_reasoning_core_context(msg, skill, complexity, _session_id)
        print(f"[ReasoningCore] {len(_rc_injection)} chars | {_rc_meta.get('injection_sources', [])}")
    except Exception as _rce:
        print(f"[ReasoningCore] skipped: {_rce}")

    # ── Build prompt — PARALLEL I/O ─────────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor
    # ── Vision: describe image and prepend to message ──────────────────────
    if image_b64:
        try:
            from modules.core.http_client import vision_describe
            vision_result = vision_describe(image_b64, msg or "Describe this image in detail.")
            msg = f"[Image analysis: {vision_result}]\n\nUser question: {msg}" if msg else vision_result
        except Exception as _ve:
            msg = f"[Vision error: {_ve}]\n\n{msg}"

    clean_msg, search_ctx = extract_search_context(msg)
    # Always enrich msg with search_ctx so it reaches the model regardless of path
    if search_ctx and search_ctx.strip():
        msg = search_ctx + "\n\nUser question: " + msg
    if _needs_fresh_search(msg) and not search_ctx:
        print(f'[KnowledgeCutoff] Stale topic — triggering search')
        try:
            from search import tool_search_multi as _msearch
            search_ctx = _msearch(msg, max_results=5)
            complexity = "medium"
            print(f'[KnowledgeCutoff] got {len(search_ctx)} chars')
            if search_ctx and len(search_ctx) > 50:
                import datetime as _dt
                _today_str = str(_dt.date.today())
                print(f'[DEBUG] injecting search into msg, old msg={repr(msg[:50])}')
                # Inject as assistant pre-turn — model treats this as its own prior output
                _search_prime = ("I have just searched the web and found these real-time results for today "
                                 + _today_str + ":\n\n" + search_ctx[:3000])
                msg = ("LIVE SEARCH RESULTS fetched " + _today_str + ":\n"
                       + search_ctx[:4000]
                       + "\n\nAnswer this using ONLY the results above: " + msg)
        except Exception as _se:
            print(f'[KnowledgeCutoff] failed: {_se}')
            search_ctx = ""
    history = clean_history(history or [])
    if _count_tokens(history) > 1500:
        history = compress_history(history)[0]

    with ThreadPoolExecutor(max_workers=4) as _ex:
        _f_hist    = _ex.submit(lambda: compress_history(_strip_thinking_from_history(history)))
        _f_mem     = _ex.submit(lambda: mem_get(msg, k=3))
        _f_episodic= _ex.submit(lambda: mem_get_episodic(msg))
        _f_rlhf    = _ex.submit(lambda: get_rlhf_note(skill))

    recent, ctx_sum = _f_hist.result()
    _mem_working    = _f_mem.result()
    _mem_episodic   = _f_episodic.result()
    rlhf_note       = _f_rlhf.result()
    _mem_ctx        = build_memory_context(msg)
    system          = build_system_prompt(skill, _mem_working, _mem_episodic, rlhf_note, ctx_sum or "", complexity, msg=msg, search_ctx=search_ctx or "")
    if _mem_ctx: system += _mem_ctx
    if _rc_injection: system += "\n\n" + _rc_injection
    system          = trim_system_prompt(system, complexity, skill=skill) if not search_ctx else system
    hist_msgs       = trim_history_for_ttft(
                         [{"role": h.get("role","user"), "content": str(h.get("content",""))[:800]} for h in (recent or [])[-8:]],
                         complexity)

    # ── Fast path for easy queries ────────────────────────────────────────────
    if complexity == "easy" and skill not in ("coder", "calculator", "researcher"):
        # Build user-turn injected messages — no system role, search results in user msg
        fast_msgs = []
        fast_msgs.append({"role": "user", "content": "<instructions>\n" + system + "\n</instructions>\nFollow exactly. Use search results as ground truth."})
        fast_msgs.append({"role": "assistant", "content": "Confirmed. Following all instructions. Search results = ground truth."})
        fast_msgs.extend(hist_msgs[-4:])
        # Use msg (enriched with search results) not clean_msg
        fast_msgs.append({"role": "user", "content": msg[:6000]})
        yield {"_meta": True, "skill": skill, "mode": "fast", "vetoed": False, "complexity": complexity}
        chunks = []
        print(f"[DEBUG] msg_preview={repr(msg[:200])}")
        print(f"[DEBUG] fast_msgs[0]={repr(str(fast_msgs[0])[:200])}")
        print(f"[DEBUG] fast_msgs[-1]={repr(str(fast_msgs[-1])[:200])}")
        for tok in mistral_stream(fast_msgs, max_tokens=32000, model="mistral-code-agent-latest" if skill == "coder" else "magistral-medium-latest"):
            yield tok
            chunks.append(tok)
        fast_response = "".join(chunks)
        if fast_response:
            cache_set(msg, skill, fast_response)
            mem_save(f"Q:{msg[:80]} A:{fast_response[:160]}")
        return

    # ── Full agentic path — stream from generate ──────────────────────────────
    prompt   = build_chatml(system, hist_msgs, msg)


def critique(response: str, skill: str = "general") -> str:
    try:
        return _run_critique(response, skill)
    except Exception:
        return ""
