"""
Deliberative Reasoning Engine — Absolute Frontier Problem Solving.
Implements: PDDL-style Problem Decomposition, Iterative Self-Correction, and Knowledge Retrieval.
"""
import re, time, random, threading, subprocess, tempfile, os, resource, json
from concurrent.futures import ThreadPoolExecutor, as_completed

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reason")

def _set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (150 * 1024 * 1024, 150 * 1024 * 1024))

def execute_math_code(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("import math\nimport sympy\n" + code)
        fname = f.name
    try:
        r = subprocess.run(["python", fname], capture_output=True, text=True, timeout=5, preexec_fn=_set_limits)
        if r.returncode == 0: return True, r.stdout.strip() or r.stderr.strip()
        return False, r.stderr.strip()[:300]
    except Exception as e: return False, str(e)
    finally:
        if os.path.exists(fname): os.unlink(fname)

def self_correcting_math(msg: str, generate_fn, model: str, max_retries: int = 3) -> str:
    prompt = [
        {"role": "system", "content": "You are a mathematical computation engine. You MUST output a python code block formatted exactly as [PYTHON CALC START] print(answer) [PYTHON CALC END] to calculate this. Do not guess numbers."},
        {"role": "user", "content": msg}
    ]
    last_error, last_code = "", ""
    for attempt in range(max_retries):
        if last_error:
            prompt.append({"role": "assistant", "content": f"[PYTHON CALC START]\n{last_code}\n[PYTHON CALC END]"})
            prompt.append({"role": "user", "content": f"Execution failed with error: {last_error}\nFix the code and output the corrected [PYTHON CALC START]...[PYTHON CALC END] block."})
        resp = generate_fn(prompt, max_tokens=500, model=model)
        match = re.search(r'\[PYTHON CALC START\](.*?)\[PYTHON CALC END\]', resp, re.DOTALL)
        if not match: return resp
        code = match.group(1).strip()
        last_code = code
        success, output = execute_math_code(code)
        if success: return f"[CALCULATED RESULT: {output}]"
        last_error = output
    return f"[MATH EXECUTION FAILED AFTER {max_retries} ATTEMPTS. Last error: {last_error}]"

def extract_and_run_math(response: str) -> str:
    final_response = response
    has_math = False
    calc_matches = re.findall(r'CALC\((.*?)\)', response, re.DOTALL)
    for expr in calc_matches:
        has_math = True
        success, result = execute_math_code(f"print({expr})")
        if success: final_response = final_response.replace(f"CALC({expr})", f"**{result}**")
        else: final_response = final_response.replace(f"CALC({expr})", f"[Calc Error: {result}]")

    pattern = r'\[PYTHON CALC START\](.*?)\[PYTHON CALC END\]'
    matches = re.findall(pattern, final_response, re.DOTALL)
    for code in matches:
        has_math = True
        success, result = execute_math_code(code.strip())
        if success: final_response = final_response.replace(f"[PYTHON CALC START]{code}[PYTHON CALC END]", f"**{result}**")
        else: final_response = final_response.replace(f"[PYTHON CALC START]{code}[PYTHON CALC END]", f"[Calc Error: {result}]")
    return final_response, has_math

def deep_problem_decomposition(msg: str, system: str, history: list, generate_fn, model: str) -> str:
    """Upgraded: Forces the AI to formally decompose the problem (PDDL) before solving."""
    decomp_prompt = [
        {"role": "system", "content": system + "\nYou are a Strategic Problem Solver. Before solving, you MUST decompose the problem into a formal plan. Output your plan inside <decomposition> tags.\n<decomposition>\n1. INITIAL STATE: What is given?\n2. GOAL STATE: What must be true to succeed?\n3. OPERATORS: What are the distinct logical/algorithmic steps to get from Initial to Goal?\n4. EDGE CASES: What can go wrong?\n</decomposition>\nAfter the decomposition, provide the full solution."},
    ] + history[-6:] + [{"role": "user", "content": msg}]
    
    return generate_fn(decomp_prompt, max_tokens=3000, model=model)

def iterative_self_correction(msg: str, system: str, history: list, generate_fn, model: str) -> str:
    """Upgraded: o1-style iterative thought process. The AI critiques its own logic and backtracks."""
    
    # 1. Initial attempt with decomposition
    initial_solution = deep_problem_decomposition(msg, system, history, generate_fn, model)
    
    # 2. Critique the logic
    critique_prompt = [
        {"role": "system", "content": "You are a Ruthless Logic Critic. Does the provided solution logically satisfy the Initial State and Goal State? Are there any missing operators or logical leaps? If it is flawless, reply EXACTLY: FLAWLESS. If flawed, reply: FLAW: [explain]"},
        {"role": "user", "content": f"Question: {msg}\n\nSolution:\n{initial_solution[:2000]}"}
    ]
    critique = generate_fn(critique_prompt, max_tokens=300, model=model)
    
    if "FLAWLESS" in critique.upper():
        return initial_solution
        
    # 3. Backtrack and rewrite
    print(f"[Reasoning] Logic flaw detected: {critique[:100]}... Backtracking and rewriting.")
    rewrite_prompt = [
        {"role": "system", "content": system + "\nYour previous solution had a logical flaw. You MUST backtrack, fix the flawed operator, and provide the corrected, flawless solution."},
        {"role": "user", "content": f"Question: {msg}\n\nPrevious Solution:\n{initial_solution[:1500]}\n\nFlaw Identified: {critique}\n\nProvide the corrected solution:"}
    ]
    return generate_fn(rewrite_prompt, max_tokens=2000, model=model)

def generate_hypotheses(msg: str, system: str, history: list, generate_fn, model: str, n: int = 3) -> list:
    def _gen_one(approach_hint: str) -> str:
        prompt = [{"role": "system", "content": system + f"\n\nApproach hint: {approach_hint}"}] + history[-6:] + [{"role": "user", "content": msg}]
        try: return generate_fn(prompt, max_tokens=1500, model=model)
        except: return ""
    approaches = ["Direct and concise. Lead with the answer.", "Step by step reasoning. Show your work explicitly.", "Consider edge cases and alternative interpretations first."][:n]
    futures = [_executor.submit(_gen_one, approach) for approach in approaches]
    results = []
    for fut in as_completed(futures, timeout=30):
        try:
            r = fut.result(timeout=5)
            if r and len(r) > 50: results.append(r)
        except: pass
    return results

def devils_advocate(winner: str, msg: str, generate_fn, model: str) -> str:
    prompt = [
        {"role": "system", "content": "You are a Ruthless Devil's Advocate. Find the ONE logical flaw, unsupported claim, or math error in the candidate. If it is flawless, reply EXACTLY: FLAWLESS. If flawed, reply: FLAW: [explain]"},
        {"role": "user", "content": f"Question: {msg[:300]}\nCandidate Answer: {winner[:1000]}"}
    ]
    try:
        critique = generate_fn(prompt, max_tokens=200, model=model)
        if "FLAW:" in critique.upper():
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

    # Calculator / Math path: Self-Correcting Math Loop
    if skill == "calculator":
        math_result = self_correcting_math(msg, generate_fn, model)
        winner = f"{math_result}\n\n"
    else:
        # Hard/Medium path: Iterative Self-Correction (Problem Solving)
        candidates = generate_hypotheses(msg, system, history, generate_fn, model, n=3 if complexity == "hard" else 2)
        if not candidates:
            prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
            winner = generate_fn(prompt, max_tokens=1200, model=model)
        else:
            winner = max(candidates, key=len)
        
        # Run Iterative Self-Correction (Decomposition + Critique + Backtrack)
        winner = iterative_self_correction(msg, system, history, generate_fn, model)
        
        # Run Devil's Advocate to break and rewrite logic
        winner = devils_advocate(winner, msg, generate_fn, model)

    # Still run any embedded math in the general/researcher response
    winner, _ = extract_and_run_math(winner)

    print(f"[Deliberate] Frontier Problem Solving (Decomposition + Self-Correction) done, t={int((time.time()-t0)*1000)}ms")
    return winner
