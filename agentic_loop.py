"""Agentic integration layer — wires TaskState + compounding error guard into the loop.

USAGE (CLI):
    python3 agentic_loop.py "Fix the bug in parser.py"

USAGE (Python):
    from agentic_loop import run_agentic_task
    result = run_agentic_task("Fix the bug in parser.py")

The chat path in main.py auto-imports inject_state_into_system() — no caller change needed.
"""
import os
import sys
import json
import logging
import subprocess
from typing import List, Dict, Optional, Tuple

from task_state import TaskState, get_global_state, reset_global_state
from self_verify import validate_upstream, compounding_error_guard, confidence_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SYSTEM_FRAMEWORK_TEMPLATE = """# CONTEXT: HIGH-PERFORMANCE AGENTIC LOOP
You are an elite autonomous software engineering agent.

You have access to the codebase environment. Solve the issue by reasoning step-by-step and selecting ACTIONS.

AVAILABLE ACTIONS (Format your response exactly as: ACTION: <command>):
1. ACTION: grep -rn "search_term" .
2. ACTION: cat path/to/file.py
3. ACTION: python3 -c 'with open("path/to/file.py", "w") as f: f.write("content")'
4. ACTION: pytest path/to/test.py
5. ACTION: FINISH

PERFORMANCE: You can output multiple ACTION lines for parallel execution.

Before each action, re-state the user's constraints in <constraints> tags.
After each tool returns, verify the result in <verify> tags with a confidence score (0.0-1.0).
If confidence < 0.5 on a tool result, re-ground by re-reading the source before proceeding.

{task_state_section}

## RECENT EXECUTION HISTORY:
{history}

## ORIGINAL TASK:
{task}"""

CODER_SYSTEM = """You are an advanced software engineering agent. Write clean, logically optimal, secure, and production-ready code. Adhere strictly to the requested architecture and formatting constraints without conversational preamble.

For every coding task you MUST:
1. State your assumptions before writing code.
2. Handle edge cases explicitly (None, empty, boundary values).
3. Add input validation with meaningful error messages.
4. Wrap I/O and network calls in try/except with logging.
5. Write pytest unit tests covering happy path AND edge cases.
6. Add PEP-484 type hints to all public functions.
7. Add docstrings to all modules and public functions.
8. Use logger.info/error not print() for observability.
9. Flag security considerations as # SECURITY: inline comments.
10. End every response with a Design Rationale section.

Before each action, re-state the user's constraints in <constraints> tags.
After each tool returns, verify the result in <verify> tags with a confidence score."""

def _call_mistral(prompt: str, system: str = CODER_SYSTEM, model: str = "codestral-latest", max_tokens: int = 4000) -> str:
    import requests
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    url = 'https://api.mistral.ai/v1/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.0,
        'max_tokens': max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def _execute_actions(action_lines: List[str]) -> Tuple[str, List[Dict]]:
    """Execute action lines in parallel. Returns (combined_results, per_action_status)."""
    processes = []
    for line in action_lines:
        command = line.split('ACTION:', 1)[1].strip()
        if command == 'FINISH':
            return "FINISH", [{"command": "FINISH", "success": True, "output": ""}]
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((command, p))

    combined = ""
    action_status = []
    for command, p in processes:
        stdout, stderr = p.communicate()
        success = (p.returncode == 0)
        if len(stdout) > 4000:
            stdout = stdout[:4000] + "\n... [TRUNCATED] ...\n"
        combined += f'\nCOMMAND: {command}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n---'
        action_status.append({
            "command": command,
            "success": success,
            "output": stdout + stderr,
            "returncode": p.returncode,
        })
    return combined, action_status

def run_agentic_task(task_description: str, max_turns: int = 20) -> Dict:
    """Run the agentic loop with TaskState + compounding error guard.

    Returns: {task_state, history, escalation, final_output, turns_used}
    """
    state = reset_global_state()
    state.add_objective(task_description)

    history_turns: List[str] = []
    escalation = None
    turns_used = 0

    for turn in range(max_turns):
        turns_used = turn + 1
        print(f'\n=== Agent Turn {turns_used}/{max_turns} | confidence={state.confidence:.2f} ===')

        task_state_section = state.to_prompt_section()
        recent_history = "\n".join(history_turns[-8:])

        full_prompt = SYSTEM_FRAMEWORK_TEMPLATE.format(
            task_state_section=task_state_section,
            history=recent_history,
            task=task_description,
        )

        try:
            output = _call_mistral(full_prompt)
        except Exception as e:
            log.error("LLM call failed: %s", e)
            state.record_step("llm_call", success=False, confidence=0.0)
            if state.should_escalate():
                escalation = f"LLM call failed repeatedly: {e}"
                break
            continue

        action_lines = [line for line in output.split('\n') if 'ACTION:' in line]

        if not action_lines:
            state.record_step("no_action", success=False, confidence=0.3)
            history_turns.append(f"Agent Thought:\n{output}")
            history_turns.append("ERROR: No valid ACTION selected.")
            continue

        combined_results, action_status = _execute_actions(action_lines)

        if combined_results == "FINISH":
            state.complete_objective(task_description)
            state.record_step("FINISH", success=True, confidence=1.0)
            print('[+] Agent flagged completion. Exiting loop.')
            break

        # Compounding-error guard: validate each tool output before consuming
        for status in action_status:
            is_safe, conf, reason = validate_upstream(status["output"], min_confidence=0.3)
            state.record_step(
                action=status["command"][:80],
                success=status["success"],
                confidence=conf if status["success"] else 0.2,
            )
            if not is_safe:
                log.warning("Low confidence on '%s': %s", status["command"][:60], reason)

        # Periodic compounding error check on recent step outputs
        if turn > 0 and turn % 3 == 0:
            step_outputs = [h for h in history_turns if h.startswith("RESULTS")]
            guard = compounding_error_guard(step_outputs, max_consecutive_low_confidence=3)
            if guard["should_escalate"]:
                escalation = guard["reason"]
                log.warning("Compounding error detected: %s", escalation)
                state.add_open_question("Re-plan needed due to compounding errors")
                break

        if state.should_escalate():
            escalation = f"State escalation: failed_steps={len(state.failed_steps)}, confidence={state.confidence:.2f}"
            break

        history_turns.append(f"Agent Thought:\n{output}")
        history_turns.append(f"RESULTS OF ACTIONS:\n{combined_results}")

        # Compact history if too long
        if len(history_turns) > 16:
            history_turns = history_turns[:2] + ["[COMPACTED]"] + history_turns[-12:]

    return {
        "task_state": state.to_dict(),
        "history": history_turns,
        "escalation": escalation,
        "final_output": history_turns[-1] if history_turns else "",
        "turns_used": turns_used,
    }

def inject_state_into_system(system_prompt: str) -> str:
    """Helper for main.py / chat path — appends current task state to any system prompt."""
    try:
        state_section = get_global_state().to_prompt_section()
        if state_section:
            return system_prompt + "\n\n" + state_section
    except Exception as e:
        log.debug("inject_state_into_system failed: %s", e)
    return system_prompt

def record_turn(user_msg: str, assistant_response: str, tool_outputs: Optional[List[str]] = None) -> None:
    """Helper for main.py chat path — records a turn in the global task state."""
    state = get_global_state()
    state.record_step(
        action=f"chat: {user_msg[:60]}",
        success=True,
        confidence=confidence_score(assistant_response),
    )
    if tool_outputs:
        for out in tool_outputs:
            is_safe, conf, _ = validate_upstream(out)
            state.record_step(action="tool_output", success=is_safe, confidence=conf)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 agentic_loop.py <task_description>")
        sys.exit(1)
    result = run_agentic_task(sys.argv[1])
    print("\n=== FINAL STATE ===")
    print(json.dumps(result["task_state"], indent=2))
    if result["escalation"]:
        print(f"\n[ESCALATION]: {result['escalation']}")
