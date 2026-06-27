"""
Deliberative Reasoning Engine — 100x Upgrade.
Implements: Tree of Thoughts (ToT), LLM Logic Judge, and Python Code Execution for Math.
"""
import re, time, random, threading, asyncio, subprocess, tempfile, os
from concurrent.futures import ThreadPoolExecutor, as_completed

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reason")

def execute_math_code(code: str) -> str:
    """Executes python math code safely and returns the exact output."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run(["python", fname], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip() or r.stderr.strip()
        return f"EXECUTION ERROR: {r.stderr.strip()[:200]}"
    except Exception as e:
        return f"EXECUTION ERROR: {str(e)}"
    finally:
        if os.path.exists(fname): os.unlink(fname)

def extract_and_run_math(response: str) -> str:
    """Finds [PYTHON CALC START]...[END] blocks, runs them, and injects the result."""
    pattern = r'\[PYTHON CALC START\](.*?)\[PYTHON CALC END\]'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if not matches:
        return response, False  # False means no math code was found
        
    final_response = response
    for code in matches:
        result = execute_math_code(code.strip())
        # Replace the code block with the executed result
        final_response = final_response.replace(
            f"[PYTHON CALC START]{code}[PYTHON CALC END]",
            f"[CALCULATED RESULT: {result}]"
        )
    return final_response, True

def generate_hypotheses(msg: str, system: str, history: list, generate_fn, model: str, n: int = 3) -> list:
    def _gen_one(approach_hint: str) -> str:
        prompt = [{"role": "system", "content": system + f"\n\nApproach hint: {approach_hint}"}] + history[-6:] + [{"role": "user", "content": msg}]
        try: return generate_fn(prompt, max_tokens=1200, model=model)
        except: return ""

    approaches = [
        "Direct and concise. Lead with the answer.",
        "Step by step reasoning. Show your work explicitly.",
        "Consider edge cases and alternative interpretations first.",
    ][:n]

    futures = [_executor.submit(_gen_one, approach) for approach in approaches]
    results = []
    for fut in as_completed(futures, timeout=30):
        try:
            r = fut.result(timeout=5)
            if r and len(r) > 50: results.append(r)
        except: pass
    return results

def llm_logic_judge(msg: str, candidates: list, generate_fn, model: str) -> str:
    """Upgraded: LLM as a Judge to pick the most logically sound candidate."""
    if not candidates: return ""
    if len(candidates) == 1: return candidates[0]

    prompt = [{"role": "system", "content": "You are an impartial Logic Judge. Evaluate the candidates for logical consistency, absence of fallacies, and correctness. Reply ONLY with the best candidate verbatim."}]
    
    candidate_text = ""
    for i, c in enumerate(candidates[:3]):
        candidate_text += f"\n\n--- CANDIDATE {i+1} ---\n{c[:1000]}\n"
        
    prompt.append({"role": "user", "content": f"Question: {msg[:300]}\n{candidate_text}\n\nWhich candidate is logically flawless? Reply with the exact text of the winner:"})
    
    try:
        winner = generate_fn(prompt, max_tokens=1000, model=model)
        # Basic check to ensure it didn't hallucinate a completely new response
        if len(winner) > 50 and any(c[:100] in winner for c in candidates):
            return winner
    except: pass
    return max(candidates, key=len)

def deliberate(msg: str, system: str, history: list, generate_fn, model: str, complexity: str = "medium", skill: str = "general") -> str:
    t0 = time.time()
    
    # Easy path: single generation
    if complexity == "easy":
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        resp = generate_fn(prompt, max_tokens=2500, model=model)
        # If it's a calculator task, enforce math execution
        if skill == "calculator":
            resp, _ = extract_and_run_math(resp)
        return resp

    # Hard/Medium path: Tree of Thoughts + Math Execution
    candidates = generate_hypotheses(msg, system, history, generate_fn, model, n=3 if complexity == "hard" else 2)
    if not candidates:
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(prompt, max_tokens=1200, model=model)

    best = llm_logic_judge(msg, candidates, generate_fn, model)
    
    # If calculator, execute the math in the winning response
    if skill == "calculator":
        best, had_math = extract_and_run_math(best)
        if not had_math:
            # If AI didn't write math code, force it to retry with code
            retry_prompt = [{"role": "system", "content": "You MUST output [PYTHON CALC START] print(answer) [PYTHON CALC END] to calculate this."}, {"role": "user", "content": msg}]
            retry_resp = generate_fn(retry_prompt, max_tokens=500, model=model)
            best, _ = extract_and_run_math(retry_resp)

    print(f"[Deliberate] ToT done, t={int((time.time()-t0)*1000)}ms")
    return best
