"""
Deliberative Reasoning Engine — Frontier Tier (o1-style).
Implements: Hidden Scratchpad, Self-Correcting Math Loop, and Devil's Advocate Logic Verification.
"""
import re, time, random, threading, subprocess, tempfile, os, resource
from concurrent.futures import ThreadPoolExecutor, as_completed

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reason")

def _set_limits():
    # 3 seconds CPU time limit for math execution
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024))

def execute_math_code(code: str) -> tuple[bool, str]:
    """Executes python math code safely and returns (success, output)."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run(["python", fname], capture_output=True, text=True, timeout=5, preexec_fn=_set_limits)
        if r.returncode == 0:
            return True, r.stdout.strip() or r.stderr.strip()
        return False, r.stderr.strip()[:300]
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(fname): os.unlink(fname)

def self_correcting_math(msg: str, generate_fn, model: str, max_retries: int = 3) -> str:
    """Forces the AI to write math code, executes it, and feeds errors back for self-correction."""
    prompt = [
        {"role": "system", "content": "You are a mathematical computation engine. You MUST output a python code block formatted exactly as [PYTHON CALC START] print(answer) [PYTHON CALC END] to calculate this. Do not guess numbers."},
        {"role": "user", "content": msg}
    ]
    
    last_error = ""
    for attempt in range(max_retries):
        if last_error:
            prompt.append({"role": "assistant", "content": f"[PYTHON CALC START]\n{last_code}\n[PYTHON CALC END]"})
            prompt.append({"role": "user", "content": f"Execution failed with error: {last_error}\nFix the code and output the corrected [PYTHON CALC START]...[PYTHON CALC END] block."})
        
        resp = generate_fn(prompt, max_tokens=500, model=model)
        match = re.search(r'\[PYTHON CALC START\](.*?)\[PYTHON CALC END\]', resp, re.DOTALL)
        
        if not match:
            return resp # Fallback if AI refuses to use tags
            
        code = match.group(1).strip()
        last_code = code
        success, output = execute_math_code(code)
        
        if success:
            return f"[CALCULATED RESULT: {output}]"
        else:
            last_error = output
            
    return f"[MATH EXECUTION FAILED AFTER {max_retries} ATTEMPTS. Last error: {last_error}]"

def extract_and_run_math(response: str) -> str:
    """Finds [PYTHON CALC START]...[END] blocks in general responses, runs them, and injects the result."""
    pattern = r'\[PYTHON CALC START\](.*?)\[PYTHON CALC END\]'
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches: return response, False
        
    final_response = response
    for code in matches:
        success, result = execute_math_code(code.strip())
        final_response = final_response.replace(
            f"[PYTHON CALC START]{code}[PYTHON CALC END]",
            f"[CALCULATED RESULT: {result}]"
        )
    return final_response, True

def generate_hypotheses(msg: str, system: str, history: list, generate_fn, model: str, n: int = 3) -> list:
    def _gen_one(approach_hint: str) -> str:
        prompt = [{"role": "system", "content": system + f"\n\nApproach hint: {approach_hint}"}] + history[-6:] + [{"role": "user", "content": msg}]
        try: return generate_fn(prompt, max_tokens=1500, model=model)
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

def devils_advocate(winner: str, msg: str, generate_fn, model: str) -> str:
    """Upgraded: Tries to break the AI's logic. If it finds a hole, forces a rewrite."""
    prompt = [
        {"role": "system", "content": "You are a Ruthless Devil's Advocate. Find the ONE logical flaw, unsupported claim, or math error in the candidate. If it is flawless, reply EXACTLY: FLAWLESS. If flawed, reply: FLAW: [explain]"},
        {"role": "user", "content": f"Question: {msg[:300]}\nCandidate Answer: {winner[:1000]}"}
    ]
    try:
        critique = generate_fn(prompt, max_tokens=200, model=model)
        if "FLAW:" in critique.upper():
            # Force a rewrite addressing the flaw
            rewrite_prompt = [
                {"role": "system", "content": "Fix the flaw in your reasoning and output the corrected, flawless answer."},
                {"role": "user", "content": f"Original Answer: {winner[:800]}\n\nFlaw Identified: {critique}\n\nProvide the corrected answer:"}
            ]
            fixed = generate_fn(rewrite_prompt, max_tokens=1000, model=model)
            return fixed
    except: pass
    return winner

def deliberate(msg: str, system: str, history: list, generate_fn, model: str, complexity: str = "medium", skill: str = "general") -> str:
    t0 = time.time()
    
    # Easy path: single generation
    if complexity == "easy":
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        resp = generate_fn(prompt, max_tokens=2500, model=model)
        if skill == "calculator":
            resp, _ = extract_and_run_math(resp)
        return resp

    # Hard/Medium path: Tree of Thoughts + Devil's Advocate + Math Execution
    candidates = generate_hypotheses(msg, system, history, generate_fn, model, n=3 if complexity == "hard" else 2)
    if not candidates:
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(prompt, max_tokens=1200, model=model)

    # Pick the longest/most detailed candidate as the initial winner
    winner = max(candidates, key=len)
    
    # Run Devil's Advocate to break and rewrite logic
    winner = devils_advocate(winner, msg, generate_fn, model)
    
    # If calculator, use the self-correcting math loop
    if skill == "calculator":
        math_result = self_correcting_math(msg, generate_fn, model)
        winner = f"{math_result}\n\n{winner}"
    else:
        # Still run any embedded math in the general/researcher response
        winner, _ = extract_and_run_math(winner)

    print(f"[Deliberate] Frontier ToT + Devil's Advocate done, t={int((time.time()-t0)*1000)}ms")
    return winner
