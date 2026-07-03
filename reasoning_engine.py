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

def execution_augmented_planning(msg: str, system: str, history: list, generate_fn, model: str) -> str:
    """Upgraded: o1-style Execution-Augmented Planning. The AI can run code during its hidden thinking phase."""
    planning_prompt = [
        {"role": "system", "content": system + "\nYou are an Algorithmic Reasoning Engine. You MUST plan your approach in a <step_back> block.\nIf you need to calculate something during planning, output <exec>print(2+2)</exec>. The system will execute it and provide the result in <result> tags. Use this to verify your math before writing the final code."},
    ] + history[-6:] + [{"role": "user", "content": msg}]
    
    # We might need multiple passes if the AI uses <exec> tags
    current_context = planning_prompt
    final_thought = ""
    
    for _ in range(3): # Max 3 execution rounds during planning
        thought = generate_fn(current_context, max_tokens=2000, model=model)
        final_thought += thought
        
        exec_matches = re.findall(r'<exec>(.*?)</exec>', thought, re.DOTALL)
        if not exec_matches:
            break # No more code to execute, planning is done
            
        # Execute all code blocks found
        exec_results = ""
        for code in exec_matches:
            success, output = execute_math_code(code.strip())
            exec_results += f"<result>{output}</result>\n"
            
        # Feed results back to AI to continue planning
        current_context.append({"role": "assistant", "content": thought})
        current_context.append({"role": "user", "content": f"Execution results:\n{exec_results}\nContinue your planning and output the final solution."})
        
    return final_thought

def prover_skeptic_judge(msg: str, system: str, history: list, generate_fn, model: str) -> str:
    prover_prompt = [{"role": "system", "content": system + "\nYou are the Prover. Provide a detailed, step-by-step solution. Use <step_back> and <exec> to plan and calculate."}] + history[-6:] + [{"role": "user", "content": msg}]
    proposal = execution_augmented_planning(msg, system, history, generate_fn, model)
    
    skeptic_prompt = [
        {"role": "system", "content": "You are a Ruthless Skeptic. Find the exact logical flaw or math error. If flawless, reply EXACTLY: FLAWLESS."},
        {"role": "user", "content": f"Question: {msg}\nProver's Solution:\n{proposal}"}
    ]
    critique = generate_fn(skeptic_prompt, max_tokens=300, model=model)
    
    if "FLAWLESS" in critique.upper(): return proposal
        
    judge_prompt = [
        {"role": "system", "content": system + "\nYou are the Judge. Synthesize a final, bulletproof solution addressing the critique."},
        {"role": "user", "content": f"Question: {msg}\nProver's Solution:\n{proposal}\nCritique:\n{critique}\nFinal Corrected Solution:"}
    ]
    return generate_fn(judge_prompt, max_tokens=1500, model=model)

def deliberate(msg: str, system: str, history: list, generate_fn, model: str, complexity: str = "medium", skill: str = "general") -> str:
    t0 = time.time()
    
    if complexity == "easy":
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        resp = generate_fn(prompt, max_tokens=2500, model=model)
        resp, _ = extract_and_run_math(resp)
        return resp

    if skill == "calculator":
        math_prompt = [
            {"role": "system", "content": "You are a mathematical computation engine. You MUST use <exec>print(answer)</exec> to calculate. The system will execute it. Base your final answer on the result."},
            {"role": "user", "content": msg}
        ]
        winner = execution_augmented_planning(msg, "You are a math engine.", [], generate_fn, model)
    else:
        winner = prover_skeptic_judge(msg, system, history, generate_fn, model)

    winner, _ = extract_and_run_math(winner)
    print(f"[Deliberate] Execution-Augmented Reasoning done, t={int((time.time()-t0)*1000)}ms")
    return winner
