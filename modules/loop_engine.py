"""
EliteOmni Loop Engine — based on patterns used by Claude, GPT-5, and Gemini.

PLAN      → decompose task (Gemini Deep Research style)
SEARCH    → iterative multi-hop Tavily (Gemini Deep Research)
REASON    → chain-of-thought scratchpad (GPT-5 deliberative reasoning)
ACT       → tool calls in ReAct loop (GPT-5 / Claude Code)
CRITIQUE  → constitutional self-check (Claude CAI)
REWRITE   → fix if critique fails (Reflexion / GPT-5 self-correction)
ANSWER    → final grounded synthesis
"""
import re, time, datetime
from typing import Callable

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
LOOP_TIMEOUT = 25  # hard ceiling across all loops (seconds)

# ── 1. PLAN (Gemini Deep Research style) ──────────────────────────────────────
def plan_subtasks(msg: str, generate_fn: Callable, system: str) -> list:
    """Break complex query into subtasks. Skip for simple queries."""
    if len(msg.split()) < 12:
        return [msg]
    prompt = (
        f"Break this into 2-4 concrete subtasks a research agent should solve:\n{msg}\n\n"
        f"Rules: numbered list only, one per line, be specific, no preamble."
    )
    raw = generate_fn([{"role": "system", "content": system},
                       {"role": "user", "content": prompt}]) or ""
    tasks = [re.sub(r'^\d+[\.\)]\s*', '', l).strip()
             for l in raw.split('\n') if re.match(r'^\d+', l.strip())]
    return tasks if tasks else [msg]


# ── 2. MULTI-HOP SEARCH (Gemini Deep Research style) ──────────────────────────
def multi_hop_search(queries: list, max_hops: int = 3) -> str:
    """
    Iterative search across multiple queries — like Gemini Deep Research.
    Each hop can refine based on what the previous found.
    """
    from modules.services.search import tavily_search, _tavily_cache
    all_chunks = []
    seen = set()

    for i, q in enumerate(queries[:max_hops]):
        # Check cache first
        ck = q.strip().lower()[:120]
        if ck in _tavily_cache:
            result = _tavily_cache[ck]
            print(f"[MultiHop] hop={i+1} cache hit")
        else:
            print(f"[MultiHop] hop={i+1} query={repr(q[:60])}")
            result = tavily_search(q, max_results=3)

        if result and result not in seen:
            seen.add(result[:100])
            all_chunks.append(f"[Search {i+1}: {q[:40]}]\n{result[:1500]}")

    return "\n\n---\n\n".join(all_chunks)


# ── 3. REACT (GPT-5 / Claude Code style) ──────────────────────────────────────
def react_loop(
    msg: str,
    system: str,
    generate_fn: Callable,
    tools: dict,
    search_ctx: str = "",
    max_iters: int = 3
) -> str:
    """
    Reason → Act → Observe → Repeat.
    Model explicitly calls tools and builds on observations.
    """
    scratchpad = []
    observation = search_ctx[:800] if search_ctx else ""
    t0 = time.time()

    for i in range(max_iters):
        if time.time() - t0 > LOOP_TIMEOUT:
            print(f"[ReAct] timeout at iter {i+1}")
            break

        pad = "\n".join(scratchpad[-3:]) if scratchpad else "None"
        react_prompt = (
            f"Scratchpad:\n{pad}\n\n"
            f"Last observation:\n{observation[:600]}\n\n"
            f"Task: {msg}\n\n"
            f"Tools available: {list(tools.keys())}\n\n"
            f"Respond with ONE of:\n"
            f"THOUGHT: <reasoning>\nACTION: TOOL_NAME(<query>)\n\n"
            f"OR if done:\n"
            f"THOUGHT: <reasoning>\nFINAL: <complete answer>"
        )
        resp = generate_fn([
            {"role": "system", "content": system},
            {"role": "user", "content": react_prompt}
        ]) or ""

        scratchpad.append(f"[Iter {i+1}] {resp[:150]}")

        final = re.search(r'FINAL:\s*(.+)', resp, re.DOTALL)
        if final:
            print(f"[ReAct] FINAL at iter {i+1}")
            return final.group(1).strip()

        action = re.search(r'ACTION:\s*(\w+)\((.+?)\)', resp, re.DOTALL)
        if action:
            tool_name = action.group(1).upper()
            tool_query = action.group(2).strip().strip('"\'')
            fn = tools.get(tool_name)
            if fn:
                print(f"[ReAct] tool={tool_name} query={repr(tool_query[:50])}")
                try:
                    observation = str(fn(tool_query) or "No result")[:1000]
                except Exception as e:
                    observation = f"Tool error: {e}"
            else:
                observation = f"Unknown tool: {tool_name}"
        else:
            # No action tag — treat as final answer
            return resp

    return "\n".join(scratchpad)


# ── 4. CONSTITUTIONAL CRITIQUE (Claude CAI style) ──────────────────────────────
def constitutional_critique(
    msg: str,
    response: str,
    system: str,
    generate_fn: Callable
) -> tuple:
    """
    Claude's Constitutional AI pattern:
    - Check factuality, helpfulness, harmlessness
    - Return (critique, should_rewrite, score)
    """
    critique_prompt = (
        f"You are a strict quality reviewer. Evaluate this response:\n\n"
        f"Question: {msg}\n\n"
        f"Response:\n{response[:1500]}\n\n"
        f"Score 1-10 on each:\n"
        f"FACTUAL: (is it accurate and grounded?)\n"
        f"COMPLETE: (does it fully answer the question?)\n"
        f"GROUNDED: (uses evidence, not assumptions?)\n\n"
        f"Then write:\n"
        f"SCORES: factual=X complete=X grounded=X\n"
        f"ISSUES: <specific problems or 'none'>\n"
        f"VERDICT: PASS or REWRITE"
    )
    critique = generate_fn([
        {"role": "system", "content": "You are a strict but fair quality reviewer."},
        {"role": "user", "content": critique_prompt}
    ]) or ""

    score_match = re.search(
        r'SCORES:.*?factual=(\d+).*?complete=(\d+).*?grounded=(\d+)',
        critique, re.DOTALL | re.IGNORECASE
    )
    avg_score = 0.7
    if score_match:
        scores = [int(score_match.group(j)) for j in range(1, 4)]
        avg_score = sum(scores) / 30.0

    should_rewrite = "VERDICT: REWRITE" in critique and avg_score < 0.7
    print(f"[CAI] score={avg_score:.2f} rewrite={should_rewrite}")
    return critique, should_rewrite, avg_score


# ── 5. REFLEXION REWRITE (GPT-5 self-correction style) ────────────────────────
def reflexion_rewrite(
    msg: str,
    response: str,
    critique: str,
    search_ctx: str,
    system: str,
    generate_fn: Callable
) -> str:
    """Rewrite response based on critique. Used when CAI score is low."""
    issues_match = re.search(r'ISSUES:\s*(.+?)(?=VERDICT|$)', critique, re.DOTALL)
    issues = issues_match.group(1).strip() if issues_match else "Improve accuracy and completeness"

    rewrite_prompt = (
        f"Your previous response had issues:\n{issues}\n\n"
        f"Question: {msg}\n\n"
        f"Search evidence available:\n{search_ctx[:2000]}\n\n"
        f"Previous response:\n{response[:1000]}\n\n"
        f"Write an improved response that fixes the issues. "
        f"Ground every claim in the search evidence. Be direct and complete."
    )
    rewritten = generate_fn([
        {"role": "system", "content": system},
        {"role": "user", "content": rewrite_prompt}
    ]) or ""
    return rewritten if len(rewritten) > 80 else response


# ── MASTER ORCHESTRATOR ────────────────────────────────────────────────────────
def run_loops(
    msg: str,
    system: str,
    generate_fn: Callable,
    skill: str = "general",
    complexity: str = "medium",
    search_ctx: str = "",
    initial_response: str = ""
) -> str:
    """
    Full loop pipeline — gates each loop by skill/complexity.

    easy      → CAI critique only (fast, no search overhead)
    medium    → multi-hop search + CAI critique + reflexion if needed
    hard      → plan + multi-hop search + ReAct + CAI + reflexion
    researcher→ plan + multi-hop search + ReAct + CAI + reflexion
    """
    from modules.services.search import tool_search_multi, tool_web_fetch

    t0 = time.time()

    tools = {
        "SEARCH": tool_search_multi,
        "FETCH":  lambda url: tool_web_fetch(url, 3000),
    }
    try:
        from modules.services.tools import tool_calc
        tools["CALC"] = tool_calc
    except Exception:
        pass

    result = initial_response

    # ── PLAN (hard + researcher only) ─────────────────────────────────────────
    subtasks = [msg]
    if complexity == "hard" or skill == "researcher":
        if len(msg.split()) > 10:
            print("[LoopEngine] planning subtasks")
            subtasks = plan_subtasks(msg, generate_fn, system)

    # ── MULTI-HOP SEARCH (medium+ or researcher) ───────────────────────────────
    if not search_ctx or len(search_ctx) < 200:
        if complexity in ("medium", "hard") or skill == "researcher":
            print("[LoopEngine] multi-hop search")
            search_ctx = multi_hop_search(subtasks[:3], max_hops=3)

    if time.time() - t0 > LOOP_TIMEOUT:
        print("[LoopEngine] timeout after search")
        return result

    # ── REACT (hard + researcher only) ────────────────────────────────────────
    if complexity == "hard" or skill == "researcher":
        print("[LoopEngine] ReAct loop")
        result = react_loop(msg, system, generate_fn, tools,
                            search_ctx=search_ctx, max_iters=3)

    if time.time() - t0 > LOOP_TIMEOUT:
        print("[LoopEngine] timeout after ReAct")
        return result or initial_response

    # ── CONSTITUTIONAL CRITIQUE + REFLEXION (medium+) ─────────────────────────
    if complexity in ("medium", "hard") or skill == "researcher":
        target = result or initial_response
        if target and len(target) > 80:
            print("[LoopEngine] constitutional critique")
            critique, should_rewrite, score = constitutional_critique(
                msg, target, system, generate_fn
            )
            if should_rewrite and time.time() - t0 < LOOP_TIMEOUT - 5:
                print("[LoopEngine] reflexion rewrite")
                result = reflexion_rewrite(
                    msg, target, critique, search_ctx, system, generate_fn
                )

    final = result or initial_response
    print(f"[LoopEngine] done in {time.time()-t0:.1f}s len={len(final)}")
    return final
