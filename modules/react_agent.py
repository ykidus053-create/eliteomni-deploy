"""
ReAct Agent (Huyen Ch.6) — Reason + Act pattern with reflection and error correction.
Decouples planning from execution — validates plans before running actions.
"""
import json
from modules.core.http_client import mistral_generate

MAX_STEPS = 6
REFLECT_EVERY = 2  # reflect after every N steps

REACT_SYSTEM = """You are a ReAct agent. For each step output EXACTLY this JSON:
{
  "thought": "your reasoning about what to do next",
  "action": "search | calculate | answer | reflect",
  "action_input": "input for the action",
  "done": false
}
Set done=true and action=answer when you have the final answer.
Output ONLY valid JSON. No markdown, no explanation."""

def _parse_step(raw: str) -> dict:
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        return json.loads(raw)
    except Exception:
        return {"thought": raw, "action": "answer", "action_input": raw, "done": True}

def _execute_action(action: str, action_input: str, tools: dict) -> str:
    """Execute action safely — validate before running (Huyen Ch.6)."""
    action = action.lower().strip()
    if action not in tools and action not in ("answer", "reflect"):
        return f"[error] unknown action '{action}'. Available: {list(tools.keys())}"
    if action in ("answer", "reflect"):
        return action_input
    try:
        return str(tools[action](action_input))
    except Exception as e:
        return f"[error] {action} failed: {e}"

def _reflect(steps: list, msgs: list) -> str:
    """Self-reflection pass — check for errors and correct course (Huyen Ch.6)."""
    history = "\n".join(
        f"Step {i+1}: thought={s.get('thought','')} action={s.get('action','')} result={s.get('result','')[:100]}"
        for i, s in enumerate(steps)
    )
    reflect_msgs = msgs + [{"role": "user", "content": (
        f"Review your reasoning so far:\n{history}\n\n"
        "Are you on the right track? Any errors? "
        "Output a corrected next step as JSON with thought/action/action_input/done."
    )}]
    return mistral_generate(reflect_msgs, max_tokens=300)

def react_run(query: str, tools: dict = None, verbose: bool = True) -> str:
    """
    Run ReAct loop.
    tools: dict of {name: callable} — e.g. {"search": hybrid_search}
    Returns final answer string.
    """
    if tools is None:
        tools = {}

    # Default search tool
    if "search" not in tools:
        try:
            from modules.search import hybrid_search
            tools["search"] = lambda q: hybrid_search(q)
        except Exception:
            tools["search"] = lambda q: "[search unavailable]"

    msgs = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user",   "content": f"Task: {query}"}
    ]

    steps = []
    for step_num in range(MAX_STEPS):

        # Reflection pass every N steps
        if step_num > 0 and step_num % REFLECT_EVERY == 0:
            raw = _reflect(steps, msgs)
            if verbose:
                print(f"[ReAct] REFLECT step {step_num}: {raw[:100]}")
        else:
            raw = mistral_generate(msgs, max_tokens=400)

        parsed = _parse_step(raw)
        if verbose:
            print(f"[ReAct] step {step_num+1}: {parsed.get('action')} — {parsed.get('thought','')[:80]}")

        if parsed.get("done") or parsed.get("action") == "answer":
            return parsed.get("action_input", raw)

        # Execute
        result = _execute_action(parsed["action"], parsed["action_input"], tools)
        parsed["result"] = result

        if verbose:
            print(f"[ReAct] result: {str(result)[:120]}")

        steps.append(parsed)

        # Feed result back into conversation
        msgs += [
            {"role": "assistant", "content": raw},
            {"role": "user",      "content": f"Observation: {str(result)[:1000]}"}
        ]

    # Max steps hit — return best effort
    return steps[-1].get("result", "[ReAct max steps reached]") if steps else "[no result]"
