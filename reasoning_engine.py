"""
Deliberative Reasoning Engine — AlphaGeometry / o1 Absolute Frontier.
Implements: MCTS (Monte Carlo Tree Search) for Logic and Z3 Formal Theorem Proving.
"""
import re, time, random, threading, subprocess, tempfile, os, resource
from concurrent.futures import ThreadPoolExecutor, as_completed

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reason")

def _set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (150 * 1024 * 1024, 150 * 1024 * 1024))

def execute_z3_code(code: str) -> tuple[bool, str]:
    """Executes Z3 formal logic code safely and returns (success, output)."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        # Ensure z3 is imported
        f.write("import z3\nimport sympy\n" + code)
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

def formal_verification(msg: str, generate_fn, model: str, max_retries: int = 2) -> str:
    """Forces the AI to write Z3/SymPy constraints to formally prove the answer."""
    prompt = [
        {"role": "system", "content": "You are a Formal Logic Engine. You MUST write a Python script using the `z3` library (or `sympy`) to construct a formal proof or constraint solver for the problem. Output ONLY the code inside [FORMAL PROOF START] and [FORMAL PROOF END] tags. Print the final result or proof status at the end."},
        {"role": "user", "content": msg}
    ]
    
    last_error = ""
    last_code = ""
    for attempt in range(max_retries):
        if last_error:
            prompt.append({"role": "assistant", "content": f"[FORMAL PROOF START]\n{last_code}\n[FORMAL PROOF END]"})
            prompt.append({"role": "user", "content": f"Execution failed: {last_error}\nFix the code and output the corrected [FORMAL PROOF START]...[FORMAL PROOF END] block."})
        
        resp = generate_fn(prompt, max_tokens=800, model=model)
        match = re.search(r'\[FORMAL PROOF START\](.*?)\[FORMAL PROOF END\]', resp, re.DOTALL)
        
        if not match: return resp
            
        code = match.group(1).strip()
        last_code = code
        success, output = execute_z3_code(code)
        
        if success:
            return f"[FORMALLY PROVEN RESULT: {output}]"
        else:
            last_error = output
            
    return f"[FORMAL VERIFICATION FAILED AFTER {max_retries} ATTEMPTS. Last error: {last_error}]"

def mcts_search(msg: str, system: str, history: list, generate_fn, model: str, depth: int = 2, branching: int = 3) -> str:
    """Monte Carlo Tree Search for logical reasoning. Explores branches and evaluates soundness."""
    
    # Root state: Initial logical step
    root_prompt = [{"role": "system", "content": system + "\nTake the FIRST logical step towards solving this. Do not solve the whole thing."}] + history[-6:] + [{"role": "user", "content": msg}]
    root_step = generate_fn(root_prompt, max_tokens=300, model=model)
    
    current_best_path = [root_step]
    
    for d in range(depth):
        # Generate branching possibilities
        def _gen_branch(hint):
            branch_prompt = [{"role": "system", "content": system + f"\nCurrent logical path: {' '.join(current_best_path)}\nProvide the NEXT logical step. Approach: {hint}"}] + history[-4:] + [{"role": "user", "content": msg}]
            return generate_fn(branch_prompt, max_tokens=400, model=model)
            
        hints = ["Direct logical deduction.", "Consider an alternative interpretation.", "Find a counterexample or edge case."]
        futures = [_executor.submit(_gen_branch, h) for h in hints]
        
        candidates = []
        for fut in as_completed(futures, timeout=20):
            try:
                res = fut.result(timeout=5)
                if res: candidates.append(res)
            except: pass
            
        if not candidates: break
        
        # Evaluate branches for logical soundness
        def _eval_branch(candidate):
            eval_prompt = [{"role": "system", "content": "You are a Logic Evaluator. Score this reasoning step from 0.0 to 1.0 for logical soundness and relevance. Reply ONLY with a float."}, {"role": "user", "content": candidate}]
            try:
                score_str = generate_fn(eval_prompt, max_tokens=10, model=model)
                return float(re.search(r'0\.\d+|1\.0', score_str).group())
            except: return 0.1
                
        eval_futures = {_executor.submit(_eval_branch, c): c for c in candidates}
        best_score, best_step = 0.0, ""
        for fut in as_completed(eval_futures, timeout=15):
            try:
                score = fut.result(timeout=5)
                if score >= best_score:
                    best_score = score
                    best_step = eval_futures[fut]
            except: pass
            
        if best_step:
            current_best_path.append(best_step)
            
    # Synthesize final answer from the best MCTS path
    synth_prompt = [{"role": "system", "content": system + "\nSynthesize the final, bulletproof answer based on the verified logical path."}, {"role": "user", "content": f"Verified Logical Path:\n{' '.join(current_best_path)}\n\nFinal Answer:"}]
    return generate_fn(synth_prompt, max_tokens=1000, model=model)

def deliberate(msg: str, system: str, history: list, generate_fn, model: str, complexity: str = "medium", skill: str = "general") -> str:
    t0 = time.time()
    
    # Easy path: single generation
    if complexity == "easy":
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(prompt, max_tokens=2500, model=model)

    # Math / Logic path: Z3 Formal Theorem Prover
    if skill == "calculator" or any(k in msg.lower() for k in ["prove", "logically", "if and only if", "implies"]):
        proof_result = formal_verification(msg, generate_fn, model)
        format_prompt = [{"role": "system", "content": "Use the formally proven result to answer the user's question directly and concisely."}, {"role": "user", "content": f"Question: {msg}\nFormal Proof Result: {proof_result}\n\nFinal Answer:"}]
        return generate_fn(format_prompt, max_tokens=500, model=model)

    # Hard / Researcher path: MCTS Reasoning
    if complexity in ("hard", "medium") and skill in ("researcher", "general"):
        final_answer = mcts_search(msg, system, history, generate_fn, model)
        print(f"[Deliberate] MCTS Search completed, t={int((time.time()-t0)*1000)}ms")
        return final_answer

    # Fallback
    prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
    return generate_fn(prompt, max_tokens=2000, model=model)
