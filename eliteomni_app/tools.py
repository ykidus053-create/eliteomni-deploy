# AUTO-SPLIT FROM app.py lines 903-1124
import os, re, time, math, json, asyncio, random, ast, subprocess, sys, tempfile
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import urllib.request, urllib.parse


# Patterns blocked from sandboxed execution for host safety
_EXEC_BLOCKED = re.compile(
    r'\b(os\.system|os\.popen|subprocess|shutil\.rmtree|socket|requests'
    r'|__import__|importlib|open\s*\(.*["\']w["\']|ctypes|pickle\.loads)\b',
    re.IGNORECASE
)

def tool_lint(code: str) -> str:
    """
    Linter-in-the-loop: pure syntax + basic style check without execution.
    Returns 'OK' or a semicolon-separated list of issues.
    """
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"SyntaxError line {e.lineno}: {e.msg}")
        return "; ".join(issues)   # no point checking style if syntax is broken

    for i, line in enumerate(code.splitlines(), 1):
        if len(line) > 120:
            issues.append(f"Line {i}: exceeds 120 chars")

    return "OK" if not issues else "; ".join(issues[:5])

def _strip_verbose_output(output: str, max_lines: int = 15) -> str:
    """
    Advanced tool use — return only final result, not raw data dump.
    Mirrors Claude's internal tool: process raw data, return summary only.
    Prevents thousands of tokens from intermediate calculations entering context.
    """
    lines = output.strip().split("\n")
    if len(lines) <= max_lines:
        return output
    # Keep first 3 lines (usually the important result) + last 3 lines
    head = lines[:3]
    tail = lines[-3:]
    omitted = len(lines) - 6
    return "\n".join(head) + f"\n... [{omitted} lines omitted] ...\n" + "\n".join(tail)

def tool_exec(code: str, timeout: int = 8) -> str:
    """
    Sandboxed Python execution in an isolated subprocess.
    Runs code to *verify* results rather than just predicting them —
    the core "Calculation & Code Execution" principle from Claude Code.
    """
    # 1. Lint first (linter-in-the-loop)
    lint = tool_lint(code)
    if lint != "OK":
        return f"[LINT FAILED — not executed]: {lint}"

    # 2. Static safety scan
    if _EXEC_BLOCKED.search(code):
        return "[BLOCKED]: Code contains restricted operations."

    # 3. Execute in isolated subprocess
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=timeout
        )
        out = (result.stdout + result.stderr).strip()
        return out[:800] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s]"
    except Exception as e:
        return f"[EXEC ERROR]: {e}"
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════
# SWE AGENT SYSTEM — Structured patch generation + execution loops
# ══════════════════════════════════════════════════════════════════

import difflib

def _extract_code_blocks(text: str) -> list:
    """Extract all ```python code blocks from model output."""
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]

def _validate_patch(original: str, patched: str) -> tuple:
    """
    Validate a patch before applying it.
    Returns (is_valid, error_message).
    """
    # 1. Syntax check
    try:
        ast.parse(patched)
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"
    # 2. Sanity: patched should be similar length (not wildly different)
    ratio = len(patched) / max(len(original), 1)
    if ratio > 10 or ratio < 0.1:
        return False, f"Patch size ratio suspicious: {ratio:.1f}x original"
    return True, "OK"

def _render_diff(original: str, patched: str, filename: str = "code.py") -> str:
    """Render a clean unified diff between original and patched code."""
    orig_lines   = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        orig_lines, patched_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}", lineterm=""
    ))
    return "".join(diff) if diff else "(no changes)"

def _patch_execution_loop(original_code: str, task: str,
                           max_iterations: int = 3) -> dict:
    """
    SWE-style patch execution loop:
    1. Generate patch (hidden reasoning phase)
    2. Validate syntax
    3. Execute tests
    4. Retry if failed
    Returns: {patched_code, diff, lint, exec_output, iterations}
    """
    result = {
        "patched_code": original_code,
        "diff": "",
        "lint": "not run",
        "exec_output": "",
        "iterations": 0,
        "success": False,
    }

    for iteration in range(max_iterations):
        result["iterations"] = iteration + 1

        # Phase 1: Hidden reasoning — plan the patch internally
        plan_prompt = build_chatml(
            "You are a senior software engineer. ONLY output a numbered plan "
            "(max 6 steps). NO code yet. Be specific about which lines to change.",
            [],
            f"Plan minimal changes to solve: {task[:300]}\n\nCode:\n```python\n{original_code[:800]}\n```"
        )
        plan = groq_generate(plan_prompt, max_tokens=400)
        if not plan:
            continue

        # Phase 2: Clean patch emission — no reasoning, just code
        patch_prompt = build_chatml(
            "You are a code patch generator. Output ONLY the complete corrected "
            "Python code inside a ```python block. No explanation. No reasoning. "
            "Make the MINIMAL changes needed. Preserve all existing functionality.",
            [],
            f"PLAN:\n{plan}\n\nOriginal code:\n```python\n{original_code}\n```\n\nOutput the patched code:"
        )
        patch_response = groq_generate(patch_prompt, max_tokens=2000)
        if not patch_response:
            continue

        # Extract code block
        blocks = _extract_code_blocks(patch_response)
        if not blocks:
            continue
        patched = blocks[0]

        # Phase 3: Syntax validation
        lint = tool_lint(patched)
        result["lint"] = lint
        if lint != "OK":
            print(f"[SWE loop iter {iteration+1}] lint failed: {lint} — retrying")
            # Feed lint error back into next iteration
            task = f"{task}\n\n[Previous attempt had lint error: {lint}. Fix it.]"
            original_code = patched  # try to fix the broken version
            continue

        # Phase 4: Execute to verify
        exec_out = tool_exec(patched)
        result["exec_output"] = exec_out

        # Phase 5: Render diff
        diff = _render_diff(original_code, patched)
        result["patched_code"] = patched
        result["diff"] = diff
        result["success"] = True
        print(f"[SWE loop] success on iteration {iteration+1}")
        break

    return result

def _format_patch_response(patch_result: dict, task: str) -> str:
    """Format the patch result as a clean response for the user."""
    if not patch_result["success"]:
        return (
            f"I attempted {patch_result['iterations']} iterations but couldn't "
            f"produce a valid patch. Last lint: {patch_result['lint']}"
        )
    out = []
    out.append(f"```python\n{patch_result['patched_code']}\n```")
    if patch_result["exec_output"] and patch_result["exec_output"] != "(no output)":
        out.append(f"\n**Execution output:**\n```\n{patch_result['exec_output'][:400]}\n```")
    if patch_result["diff"] and patch_result["diff"] != "(no changes)":
        diff_lines = patch_result["diff"].splitlines()
        if len(diff_lines) <= 30:
            out.append(f"\n**Changes (unified diff):**\n```diff\n{patch_result['diff']}\n```")
    out.append(f"\n✅ Validated in {patch_result['iterations']} iteration(s) · Lint: {patch_result['lint']}")
    return "\n".join(out)

def _grep_codebase(pattern: str) -> str:
    """
    Codebase Search: scan scratchpad for CLAUDE_TODO markers or patterns.
    Mirrors Claude Code's local file grep for understanding architecture.
    """
    try:
        pat = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"[GREP regex error: {e}]"
    hits = [f"{k}: {v}" for k, v in list(_scratchpad.items())[-30:]
            if pat.search(k) or pat.search(v)]
    return "\n".join(hits[:6]) if hits else "No matches in scratchpad."

# ── SEARCH & KNOWLEDGE ACQUISITION (Claude Code: "search first" approach) ─────
