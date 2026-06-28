import re, time, random, threading, subprocess, tempfile, os, resource, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="reason")

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

def execution_guided_reasoning(msg: str, system: str, history: list, generate_fn, model: str) -> str:
    """Upgraded: Mental Simulation. The AI traces the algorithm before writing it."""
    trace_prompt = [
        {"role": "system", "content": system + "\nYou are an Algorithmic Reasoning Engine. Before writing code, you MUST output a mental execution trace inside <trace> tags.\n<trace>\n1. Define the exact inputs and edge cases.\n2. Step through the algorithm manually with a concrete example.\n3. Identify any logic flaws or state mutations.\n</trace>\nAfter the trace, write the complete solution."},
    ] + history[-6:] + [{"role": "user", "content": msg}]
    
    return generate_fn(trace_prompt, max_tokens=3000, model=model)

def prover_skeptic_judge(msg: str, system: str, history: list, generate_fn, model: str) -> str:
    prover_prompt = [{"role": "system", "content": system + "\nYou are the Prover. Provide a detailed, step-by-step solution. Use <trace> to plan your steps first."}] + history[-6:] + [{"role": "user", "content": msg}]
    proposal = generate_fn(prover_prompt, max_tokens=1500, model=model)
    
    skeptic_prompt = [
        {"role": "system", "content": "You are a Ruthless Skeptic. Find the exact logical flaw, missing edge case, or fallacy in the Prover's solution. If it is flawless, reply EXACTLY: FLAWLESS."},
        {"role": "user", "content": f"Question: {msg}\nProver's Solution:\n{proposal}"}
    ]
    critique = generate_fn(skeptic_prompt, max_tokens=300, model=model)
    
    if "FLAWLESS" in critique.upper(): return proposal
        
    print(f"[Reasoning] Prover-Skeptic-Judge: Flaw found. Judge synthesizing...")
    judge_prompt = [
        {"role": "system", "content": system + "\nYou are the Judge. Synthesize a final, bulletproof solution that addresses the Skeptic's critique. Ensure no logical fallacies remain."},
        {"role": "user", "content": f"Question: {msg}\n\nProver's Solution:\n{proposal}\n\nSkeptic's Critique:\n{critique}\n\nProvide the final, corrected solution:"}
    ]
    return generate_fn(judge_prompt, max_tokens=1500, model=model)

def deliberate(msg: str, system: str, history: list, generate_fn, model: str, complexity: str = "medium", skill: str = "general") -> str:
    t0 = time.time()
    
    if complexity == "easy":
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        resp = generate_fn(prompt, max_tokens=2500, model=model)
        if skill == "calculator":
            resp, _ = extract_and_run_math(resp)
        return resp

    if skill == "calculator":
        math_result = self_correcting_math(msg, generate_fn, model)
        winner = f"{math_result}\n\n"
    else:
        # Hard/Medium path: Execution-Guided Reasoning + Debate
        winner = execution_guided_reasoning(msg, system, history, generate_fn, model)
        winner = prover_skeptic_judge(msg, system, history, generate_fn, model)

    winner, _ = extract_and_run_math(winner)
    print(f"[Deliberate] Execution-Guided Reasoning + Debate done, t={int((time.time()-t0)*1000)}ms")
    return winner
