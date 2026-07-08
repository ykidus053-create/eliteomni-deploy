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

def execution_augmented_planning_stream(msg, system, history, generate_fn, model):
    planning_prompt = [
        {"role": "system", "content": system + "\nYou are an Algorithmic Reasoning Engine. You MUST plan your approach in a <step_back> block.\nIf you need to calculate something during planning, output <exec>print(2+2)</exec>. The system will execute it and provide the result in <result> tags. Use this to verify your math before writing the final code."},
    ] + history[-6:] + [{"role": "user", "content": msg}]
    current_context = planning_prompt
    final_thought = ""
    for _round in range(3):
        yield ("progress", "Thinking through the approach..." if _round == 0 else "Refining the plan based on calculations...")
        thought = generate_fn(current_context, max_tokens=2000, model=model)
        final_thought += thought
        exec_matches = re.findall(r'<exec>(.*?)</exec>', thought, re.DOTALL)
        if not exec_matches:
            break
        yield ("progress", "Running calculations to verify...")
        exec_results = ""
        for code in exec_matches:
            success, output = execute_math_code(code.strip())
            exec_results += f"<result>{output}</result>\n"
        current_context.append({"role": "assistant", "content": thought})
        current_context.append({"role": "user", "content": f"Execution results:\n{exec_results}\nContinue your planning and output the final solution."})
    yield ("done", final_thought)

def execution_augmented_planning(msg, system, history, generate_fn, model):
    result = ""
    for kind, payload in execution_augmented_planning_stream(msg, system, history, generate_fn, model):
        if kind == "done":
            result = payload
    return result

def prover_skeptic_judge_stream(msg, system, history, generate_fn, model):
    yield ("progress", "Working through the problem...")
    proposal = ""
    for kind, payload in execution_augmented_planning_stream(msg, system, history, generate_fn, model):
        if kind == "progress":
            yield ("progress", payload)
        else:
            proposal = payload
    yield ("progress", "Double-checking my reasoning for errors...")
    skeptic_prompt = [
        {"role": "system", "content": "You are a Ruthless Skeptic. Find the exact logical flaw or math error. If flawless, reply EXACTLY: FLAWLESS."},
        {"role": "user", "content": f"Question: {msg}\nProver's Solution:\n{proposal}"}
    ]
    critique = generate_fn(skeptic_prompt, max_tokens=300, model=model)
    if "FLAWLESS" in critique.upper():
        yield ("done", proposal)
        return
    yield ("progress", "Correcting and finalizing the solution...")
    judge_prompt = [
        {"role": "system", "content": system + "\nYou are the Judge. Synthesize a final, bulletproof solution addressing the critique."},
        {"role": "user", "content": f"Question: {msg}\nProver's Solution:\n{proposal}\nCritique:\n{critique}\nFinal Corrected Solution:"}
    ]
    final = generate_fn(judge_prompt, max_tokens=1500, model=model)
    yield ("done", final)

def prover_skeptic_judge(msg, system, history, generate_fn, model):
    result = ""
    for kind, payload in prover_skeptic_judge_stream(msg, system, history, generate_fn, model):
        if kind == "done":
            result = payload
    return result

def deliberate_stream(msg, system, history, generate_fn, model, complexity="medium", skill="general"):
    t0 = time.time()
    if complexity == "easy":
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        resp = generate_fn(prompt, max_tokens=2500, model=model)
        resp, _ = extract_and_run_math(resp)
        yield ("done", resp)
        return
    if skill == "calculator":
        winner = ""
        for kind, payload in execution_augmented_planning_stream(msg, "You are a math engine.", [], generate_fn, model):
            if kind == "progress":
                yield ("progress", payload)
            else:
                winner = payload
    else:
        winner = ""
        for kind, payload in prover_skeptic_judge_stream(msg, system, history, generate_fn, model):
            if kind == "progress":
                yield ("progress", payload)
            else:
                winner = payload
    winner, _ = extract_and_run_math(winner)
    print(f"[Deliberate] Execution-Augmented Reasoning done, t={int((time.time()-t0)*1000)}ms")
    yield ("done", winner)

def deliberate(msg, system, history, generate_fn, model, complexity="medium", skill="general"):
    result = ""
    for kind, payload in deliberate_stream(msg, system, history, generate_fn, model, complexity, skill):
        if kind == "done":
            result = payload
    return result
