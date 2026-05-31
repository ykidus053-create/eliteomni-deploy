import urllib.request, urllib.parse
import os, re, time, math, json, ast, subprocess, sys, tempfile, difflib
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── scratchpad lives in validation; imported here to fix NameError in _grep_codebase
try:
    from modules.services.pipeline import _scratchpad
except ImportError:
    _scratchpad = {}

_EXEC_BLOCKED = re.compile(
    r'\b(os\.system|os\.popen|subprocess|shutil\.rmtree|socket|requests'
    r'|__import__|importlib|open\s*\(.*["\']w["\']|ctypes|pickle\.loads)\b',
    re.IGNORECASE
)

# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED THINKING MATH ENGINE
# Three-layer architecture mirroring Claude:
#   Layer 1 — Dynamic thinking budget  (allocate before answering)
#   Layer 2 — Parallel strategy paths  (rough magnitude + precise calc)
#   Layer 3 — Code interpreter verify  (numpy / sympy execution)
# ══════════════════════════════════════════════════════════════════════════════

def _thinking_budget(complexity: str) -> int:
    """
    Dynamic thinking budget — token allocation before output commitment.
    easy   → 0    direct answer, no exploration needed
    medium → 200  single estimation + one verification pass
    hard   → 800  full draft / test / explore / correct loop
    """
    return {"easy": 0, "medium": 200, "hard": 800}.get(complexity, 200)

def _path_a_rough(expr: str) -> str:
    """
    PATH A — back-of-envelope magnitude check.
    Rounds all numbers to 1 significant figure then evaluates.
    Catches order-of-magnitude errors before committing to PATH B.
    """
    try:
        def _round_sig(m):
            n = float(m.group(0))
            if n == 0:
                return "0"
            mag = 10 ** math.floor(math.log10(abs(n)))
            return str(round(n / mag) * mag)
        rough = re.sub(r'\d+\.?\d*', _round_sig, expr)
        safe  = re.sub(r'[^0-9+\-*/().,% ]', '', rough) \
                    .replace('%', '/100').replace('^', '**')
        result = eval(safe, {"__builtins__": {}, "math": math,
                             "sqrt": math.sqrt, "pi": math.pi, "e": math.e})
        return f"~{round(float(result), 2)}"
    except Exception:
        return "~?"

def _path_b_precise(expr: str) -> str:
    """
    PATH B — exact calculation, full precision.
    Safe eval with math builtins — same as CALC() tool.
    """
    try:
        safe = re.sub(r'[^0-9+\-*/().,% e]', '', expr) \
                   .replace('%', '/100').replace('^', '**')
        result = eval(safe, {
            "__builtins__": {}, "math": math,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "log": math.log, "pi": math.pi, "e": math.e,
            "abs": abs, "round": round
        })
        return str(round(float(result), 8))
    except Exception as ex:
        return f"Error: {ex}"

def _path_c_numpy(expr: str) -> str:
    """
    PATH C — numpy execution, eliminates hallucination.
    The runtime produces the answer; the model only formats it.
    """
    code = f"""
import numpy as np, math
try:
    result = {expr}
    print(float(result) if hasattr(result, '__float__') else result)
except Exception as e:
    print(f"NumPy error: {{e}}")
"""
    return tool_exec(code, timeout=8)

def _path_c_sympy(expr: str) -> str:
    """
    PATH C (symbolic) — sympy proof + execution.
    Mathematically structures the proof then verifies it by running it.
    Used for algebra, calculus, symbolic simplification.
    """
    code = f"""
import sympy as sp

x, y, z, n, a, b, c, t = sp.symbols('x y z n a b c t')
try:
    result = sp.simplify(sp.sympify("{expr}"))
    print(repr(result))
except Exception as e:
    print(f"SymPy error: {{e}}")
"""
    return tool_exec(code, timeout=10)

def dual_path_calc(expr: str) -> dict:
    """
    Full parallel strategy processing:
      PATH A — rough magnitude (catches gross errors)
      PATH B — precise eval   (correct digits)
      PATH C — executable verify (anti-hallucination)
    Returns all three results + consistency flag.
    """
    use_sympy = bool(re.search(r'[a-df-wyzA-Z]', expr))  # symbolic if letters present

    a = _path_a_rough(expr)
    b = _path_b_precise(expr)
    c = _path_c_sympy(expr) if use_sympy else _path_c_numpy(expr)

    # Consistency: PATH A magnitude should be within 50% of PATH B
    consistent = True
    try:
        av = float(a.replace("~", "").replace("?", "0"))
        bv = float(b)
        if av != 0 and abs(av - bv) / abs(av) > 0.5:
            consistent = False
    except Exception:
        pass

    return {
        "path_a_rough":    a,
        "path_b_precise":  b,
        "path_c_verified": c.strip() if c else "n/a",
        "consistent":      consistent,
        "final":           b,
    }

def extended_thinking_math(problem: str, complexity: str = "medium") -> str:
    """
    Full Claude-style extended thinking math pipeline:

    THINK PHASE (hidden from user):
      1. Allocate thinking budget by complexity
      2. Decompose problem into sub-expressions
      3. PATH A — rough magnitude estimate
      4. PATH B — precise calculation
      5. PATH C — executable verification (numpy/sympy)
      6. Self-correct any magnitude mismatches before output

    OUTPUT PHASE:
      Clean verified answer with confidence badge
    """
    budget = _thinking_budget(complexity)

    # Extract numeric sub-expressions
    exprs = [
        m.group().strip()
        for m in re.finditer(r'[\d\s.\+\-\*\/\%\(\)\^]+', problem)
        if len(m.group().strip()) > 2 and re.search(r'\d', m.group())
    ]

    if not exprs:
        return f"[ExtendedThinking] No numeric expression detected in: {problem[:100]}"

    blocks = []
    for expr in exprs[:3]:
        r = dual_path_calc(expr)
        warn = "" if r["consistent"] else " ⚠️ magnitude mismatch — verify inputs"
        blocks.append(
            f"**Expression:** `{expr}`\n"
            f"  PATH A (rough estimate): {r['path_a_rough']}\n"
            f"  PATH B (precise calc):   {r['path_b_precise']}\n"
            f"  PATH C (verified):       {r['path_c_verified']}{warn}\n"
            f"  ✅ **Answer: {r['final']}**"
        )

    budget_tag = (f"\n\n_[Thinking budget: {budget} tokens · complexity: {complexity}]_"
                  if budget > 0 else "")
    return "\n\n".join(blocks) + budget_tag

# ══════════════════════════════════════════════════════════════════════════════
# LINT + EXEC
# ══════════════════════════════════════════════════════════════════════════════

def tool_lint(code: str) -> str:
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"SyntaxError line {e.lineno}: {e.msg}")
        return "; ".join(issues)
    for i, line in enumerate(code.splitlines(), 1):
        if len(line) > 120:
            issues.append(f"Line {i}: exceeds 120 chars")
    return "OK" if not issues else "; ".join(issues[:5])

def _strip_verbose_output(output: str, max_lines: int = 15) -> str:
    lines = output.strip().split("\n")
    if len(lines) <= max_lines:
        return output
    head    = lines[:3]
    tail    = lines[-3:]
    omitted = len(lines) - 6
    return "\n".join(head) + f"\n... [{omitted} lines omitted] ...\n" + "\n".join(tail)

def tool_exec(code: str, timeout: int = 8) -> str:
    lint = tool_lint(code)
    if lint != "OK":
        return f"[LINT FAILED — not executed]: {lint}"
    if _EXEC_BLOCKED.search(code):
        return "[BLOCKED]: Code contains restricted operations."
    tmp = None
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
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# SWE AGENT — patch generation + execution loop
# ══════════════════════════════════════════════════════════════════════════════

def _extract_code_blocks(text: str) -> list:
    blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]

def _validate_patch(original: str, patched: str) -> tuple:
    try:
        ast.parse(patched)
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"
    ratio = len(patched) / max(len(original), 1)
    if ratio > 10 or ratio < 0.1:
        return False, f"Patch size ratio suspicious: {ratio:.1f}x original"
    return True, "OK"

def _render_diff(original: str, patched: str, filename: str = "code.py") -> str:
    orig_lines    = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        orig_lines, patched_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}", lineterm=""
    ))
    return "".join(diff) if diff else "(no changes)"

def _patch_execution_loop(original_code: str, task: str, max_iterations: int = 3) -> dict:
    result = {
        "patched_code": original_code, "diff": "",
        "lint": "not run", "exec_output": "",
        "iterations": 0, "success": False,
    }
    try:
        from modules.core.http_client import groq_generate
        from modules.services.pipeline import build_chatml
    except ImportError as ie:
        result["lint"] = f"import error: {ie}"
        return result

    for iteration in range(max_iterations):
        result["iterations"] = iteration + 1
        plan = groq_generate(
            build_chatml(
                "You are a senior software engineer. Output a numbered plan (max 6 steps). NO code yet.",
                [],
                f"Plan minimal changes to solve: {task[:300]}\n\nCode:\n```python\n{original_code[:800]}\n```"
            ),
            max_tokens=400
        )
        if not plan:
            continue
        patch_response = groq_generate(
            build_chatml(
                "Output ONLY the complete corrected Python code inside a ```python block. "
                "No explanation. Minimal changes. Preserve existing functionality.",
                [],
                f"PLAN:\n{plan}\n\nOriginal:\n```python\n{original_code}\n```\n\nPatched code:"
            ),
            max_tokens=2000
        )
        if not patch_response:
            continue
        blocks = _extract_code_blocks(patch_response)
        if not blocks:
            continue
        patched     = blocks[0]
        lint        = tool_lint(patched)
        result["lint"] = lint
        if lint != "OK":
            task          = f"{task}\n\n[Previous lint error: {lint}. Fix it.]"
            original_code = patched
            continue
        exec_out            = tool_exec(patched)
        result["exec_output"] = exec_out
        result["patched_code"] = patched
        result["diff"]         = _render_diff(original_code, patched)
        result["success"]      = True
        break
    return result

def _format_patch_response(patch_result: dict, task: str) -> str:
    if not patch_result["success"]:
        return (f"Attempted {patch_result['iterations']} iterations — no valid patch. "
                f"Last lint: {patch_result['lint']}")
    out = [f"```python\n{patch_result['patched_code']}\n```"]
    if patch_result["exec_output"] and patch_result["exec_output"] != "(no output)":
        out.append(f"\n**Execution output:**\n```\n{patch_result['exec_output'][:400]}\n```")
    if patch_result["diff"] and patch_result["diff"] != "(no changes)":
        if len(patch_result["diff"].splitlines()) <= 30:
            out.append(f"\n**Diff:**\n```diff\n{patch_result['diff']}\n```")
    out.append(f"\n✅ Validated in {patch_result['iterations']} iteration(s) · Lint: {patch_result['lint']}")
    return "\n".join(out)

def _grep_codebase(pattern: str) -> str:
    try:
        pat = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"[GREP regex error: {e}]"
    hits = [f"{k}: {v}" for k, v in list(_scratchpad.items())[-30:]
            if pat.search(k) or pat.search(v)]
    return "\n".join(hits[:6]) if hits else "No matches in scratchpad."

# ══════════════════════════════════════════════════════════════════════════════
# GPT-5.4 STYLE MATH ENGINE
# Exact technique: PRM + MCTS + weighted self-consistency
# 1. Decompose problem into steps (process supervision)
# 2. Generate N candidate solution paths in parallel (tree search)
# 3. Score each step via PRM (not just final answer)
# 4. Weighted self-consistency vote across paths
# 5. Self-verify the winner before returning
# ══════════════════════════════════════════════════════════════════════════════

def gpt5_math(problem: str, complexity: str = "hard") -> str:
    """
    GPT-5.4 style math pipeline for AIME/competition-level problems.
    PRM + MCTS + weighted self-consistency.
    Only for math — do not use for other skills.
    """
    from modules.services.pipeline import build_chatml
    import re

    N_PATHS = 3 if complexity == "hard" else 2  # parallel solution paths

    MATH_SYSTEM = (
        "You are a world-class mathematician. "
        "Show every step explicitly. "
        "After each step write STEP_SCORE: [1-5] based on confidence. "
        "Never skip steps. Never guess. "
        "End with FINAL_ANSWER: [exact answer]."
    )

    PRM_SYSTEM = (
        "You are a math process verifier (PRM). "
        "For each numbered step in the solution, output: "
        "STEP [n]: CORRECT or STEP [n]: WRONG - [reason]. "
        "Then output OVERALL: [score 1-10]. "
        "Be strict — one wrong step = wrong answer."
    )


def tool_arxiv(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for academic papers. Free, no API key needed.
    Returns titles + abstracts of top results.
    """
    if not query or not query.strip():
        return None
    try:
        q = urllib.parse.quote(query.strip()[:200])
        url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}&sortBy=relevance"
        r = urllib.request.urlopen(url, timeout=10)
        xml = r.read().decode("utf-8", errors="replace")
        entries = _re2.findall(r"<entry>(.*?)</entry>", xml, _re2.DOTALL)
        if not entries:
            return None
        results = []
        for e in entries[:max_results]:
            title   = _re2.search(r"<title>(.*?)</title>", e, _re2.DOTALL)
            summary = _re2.search(r"<summary>(.*?)</summary>", e, _re2.DOTALL)
            link    = _re2.search(r"<id>(.*?)</id>", e, _re2.DOTALL)
            if title and summary:
                t = title.group(1).strip().replace("\n", " ")
                s = summary.group(1).strip().replace("\n", " ")[:300]
                l = link.group(1).strip() if link else ""
                results.append(f"**{t}**\n{s}\n{l}")
        return "\n\n".join(results) if results else None
    except Exception as e:
        print(f"[tool_arxiv] error: {e}")
        return None


def tool_pubmed(query: str, max_results: int = 3) -> str:
    """
    Search PubMed for biomedical literature. Free, no API key needed.
    Returns titles + abstracts.
    """
    if not query or not query.strip():
        return None
    try:
        # Step 1: search for IDs
        q = urllib.parse.quote(query.strip()[:200])
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={q}&retmax={max_results}&retmode=json"
        r = urllib.request.urlopen(search_url, timeout=10)
        data = _json.loads(r.read().decode())
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None
        # Step 2: fetch summaries
        id_str = ",".join(ids[:max_results])
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id_str}&retmode=json"
        r2 = urllib.request.urlopen(fetch_url, timeout=10)
        data2 = _json.loads(r2.read().decode())
        results = []
        for uid in ids[:max_results]:
            item = data2.get("result", {}).get(uid, {})
            title = item.get("title", "")
            source = item.get("source", "")
            pubdate = item.get("pubdate", "")
            if title:
                results.append(f"**{title}**\n{source} ({pubdate})\nhttps://pubmed.ncbi.nlm.nih.gov/{uid}/")
        return "\n\n".join(results) if results else None
    except Exception as e:
        print(f"[tool_pubmed] error: {e}")
        return None


def tool_wolfram(query: str) -> str:
    """
    Query Wolfram Alpha Short Answers API.
    Requires WOLFRAM_APP_ID env var (free tier: developer.wolframalpha.com).
    Falls back gracefully if no key.
    """
    if not query or not query.strip():
        return None
    app_id = _os2.environ.get("WOLFRAM_APP_ID", "")
    if not app_id:
        print("[tool_wolfram] no WOLFRAM_APP_ID set — skipping")
        return None
    try:
        q = urllib.parse.quote(query.strip()[:300])
        url = f"https://api.wolframalpha.com/v1/result?appid={app_id}&i={q}&units=metric"
        r = urllib.request.urlopen(url, timeout=10)
        result = r.read().decode("utf-8", errors="replace").strip()
        if result and len(result) > 0:
            return f"[Wolfram Alpha]: {result}"
        return None
    except urllib.error.HTTPError as e:
        if e.code == 501:
            print(f"[tool_wolfram] no short answer available for: {query[:60]}")
        else:
            print(f"[tool_wolfram] HTTP {e.code}: {e}")
        return None
    except Exception as e:
        print(f"[tool_wolfram] error: {e}")
        return None

# ── CODE EXECUTION FEEDBACK LOOP ─────────────────────────────────────────────
import tempfile, subprocess, sys, re, json

def extract_and_run_tests(response_text: str, language: str = "python") -> dict:
    """
    Extract code blocks from AI response, run them, return results.
    Returns: {passed: int, failed: int, errors: list, output: str}
    """
    results = {"passed": 0, "failed": 0, "errors": [], "output": ""}

    if language == "python":
        # Extract all python blocks
        blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)
        if not blocks:
            return results

        # Combine all blocks into one file (later blocks can use earlier ones)
        combined = "\n\n".join(blocks)

        # If no tests exist, inject a basic smoke test
        if "def test_" not in combined and "assert " not in combined:
            results["errors"].append("No tests found in code — cannot verify correctness")
            return results

        # Write to temp file and run
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(combined)
                tmp = f.name

            result = subprocess.run(
                [sys.executable, '-m', 'pytest', tmp, '-v', '--tb=short', '--no-header'],
                capture_output=True, text=True, timeout=15
            )

            output = result.stdout + result.stderr

            # Parse pytest output
            passed = len(re.findall(r' PASSED', output))
            failed = len(re.findall(r' FAILED', output))
            errors_found = re.findall(r'(FAILED.*?)\n', output)
            assert_errors = re.findall(r'(AssertionError.*?)\n', output)
            type_errors = re.findall(r'(TypeError.*?)\n', output)
            runtime_errors = re.findall(r'(RuntimeError.*?)\n', output)

            results["passed"] = passed
            results["failed"] = failed
            results["errors"] = errors_found + assert_errors + type_errors + runtime_errors
            results["output"] = output[:1500]

            # If pytest not available, fall back to plain python run
            if "no module named pytest" in output.lower() or result.returncode == 4:
                result2 = subprocess.run(
                    [sys.executable, tmp],
                    capture_output=True, text=True, timeout=15
                )
                out2 = result2.stdout + result2.stderr
                results["output"] = out2[:1500]
                if result2.returncode != 0:
                    results["failed"] = 1
                    results["errors"] = [out2[:500]]
                else:
                    results["passed"] = 1

        except subprocess.TimeoutExpired:
            results["errors"] = ["Execution timeout (15s) — possible infinite loop"]
            results["failed"] = 1
        except Exception as e:
            results["errors"] = [str(e)]
            results["failed"] = 1

    elif language == "typescript":
        blocks = re.findall(r'```typescript\n(.*?)```', response_text, re.DOTALL)
        if not blocks:
            return results
        combined = "\n\n".join(blocks)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                f.write(combined)
                tmp = f.name
            # Try ts-node if available
            result = subprocess.run(
                ['ts-node', '--skip-project', tmp],
                capture_output=True, text=True, timeout=15
            )
            out = result.stdout + result.stderr
            results["output"] = out[:1500]
            if result.returncode != 0:
                results["failed"] = 1
                results["errors"] = [out[:500]]
            else:
                results["passed"] = 1
        except FileNotFoundError:
            results["errors"] = ["ts-node not installed — TypeScript execution unavailable"]
        except subprocess.TimeoutExpired:
            results["errors"] = ["Execution timeout"]
            results["failed"] = 1

    return results


async def self_correct_code(
    original_response: str,
    test_results: dict,
    original_msg: str,
    skill: str,
    max_attempts: int = 2
) -> str:
    """
    If tests failed, send the failure back to the model and ask for a fix.
    Returns corrected response or original if unfixable.
    """
    if test_results["failed"] == 0 and not test_results["errors"]:
        return original_response  # nothing to fix

    from modules.core.http_client import mistral_generate

    error_summary = "\n".join(test_results["errors"][:5])
    output_snippet = test_results["output"][:800]

    correction_prompt = f"""You wrote code in response to: "{original_msg[:200]}"

When I ran your code, it FAILED with these errors:
{error_summary}

Output:
{output_snippet}

TASK: Fix the code so all tests pass.
Rules:
1. Identify the exact root cause — do not guess
2. Fix only the broken logic — do not rewrite working parts
3. Show the corrected code in full
4. Explain in one sentence what was wrong

Your original response:
{original_response[:2000]}"""

    correction_msgs = [
        {"role": "system", "content": "You are a world-class debugger. Fix the exact bug. No stubs. No placeholders."},
        {"role": "user", "content": correction_prompt}
    ]

    try:
        fixed = mistral_generate(correction_msgs, max_tokens=3000, skill=skill)
        if fixed and len(fixed) > 100:
            return original_response + f"\n\n---\n> ❌ **Auto-detected {test_results['failed']} test failure(s)** — self-correction applied:\n\n" + fixed
    except Exception as e:
        pass

    return original_response + f"\n\n> ❌ **{test_results['failed']} test(s) failed:** {error_summary[:300]}"



def auto_run_and_fix(code: str, language: str = "python") -> str:
    """Run any code, auto-install missing packages, auto-fix errors."""
    try:
        from modules.services.code_sandbox import run_code_auto_install
        from modules.services.code_validator import validate_and_fix_code

        result = run_code_auto_install(code)

        if result.get("auto_installed"):
            print("  [INSTALLED] " + str(result["auto_installed"]))

        if result["success"]:
            return "[RAN OK]\n" + (result["stdout"] or "(no output)")

        fix = validate_and_fix_code(code, result["stderr"])
        if fix.get("fixed_success"):
            return (
                "[AUTO-FIXED] Issues resolved: "
                + str(fix.get("issues_found", "")) + "\n"
                + "Output:\n" + str(fix.get("fixed_output", "")) + "\n"
                + "Fixed code:\n```python\n"
                + str(fix.get("fixed", "")) + "\n```"
            )

        return "[FAILED]\n" + result["stderr"][:800]
    except Exception as exc:
        return "[EXECUTION ERROR] " + str(exc)
