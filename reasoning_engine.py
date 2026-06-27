"""
Deliberative Reasoning Engine — AlphaProof Tier.
Implements: Chain of Code (SymPy simulation) and Socratic Multi-Agent Debate.
"""
import re, time, random, threading, subprocess, tempfile, os, resource
from concurrent.futures import ThreadPoolExecutor, as_completed

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reason")

def _set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (150 * 1024 * 1024, 150 * 1024 * 1024))

def execute_logic_code(code: str) -> tuple[bool, str]:
    """Executes python logic/math code safely and returns (success, output)."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("import sympy\nimport math\n" + code)
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

def chain_of_code(msg: str, generate_fn, model: str, max_retries: int = 2) -> str:
    """Forces the AI to write a SymPy python script to solve the problem, executes it, and returns the result."""
    prompt = [
        {"role": "system", "content": "You are a formal logic and math engine. You MUST write a Python script using the `sympy` library to simulate and solve the problem. Output ONLY the code inside [PYTHON LOGIC START] and [PYTHON LOGIC END] tags. Print the final answer at the end."},
        {"role": "user", "content": msg}
    ]
    
    last_error = ""
    last_code = ""
    for attempt in range(max_retries):
        if last_error:
            prompt.append({"role": "assistant", "content": f"[PYTHON LOGIC START]\n{last_code}\n[PYTHON LOGIC END]"})
            prompt.append({"role": "user", "content": f"Execution failed: {last_error}\nFix the code and output the corrected [PYTHON LOGIC START]...[PYTHON LOGIC END] block."})
        
        resp = generate_fn(prompt, max_tokens=800, model=model)
        match = re.search(r'\[PYTHON LOGIC START\](.*?)\[PYTHON LOGIC END\]', resp, re.DOTALL)
        
        if not match: return resp
            
        code = match.group(1).strip()
        last_code = code
        success, output = execute_logic_code(code)
        
        if success:
            return f"[SIMULATION RESULT: {output}]"
        else:
            last_error = output
            
    return f"[LOGIC EXECUTION FAILED AFTER {max_retries} ATTEMPTS. Last error: {last_error}]"

def socratic_debate(msg: str, system: str, history: list, generate_fn, model: str) -> str:
    """Proposer, Skeptic, and Moderator debate the answer for bulletproof logic."""
    
    # 1. Proposer generates initial answer
    proposer_prompt = [{"role": "system", "content": system + "\nYou are the Proposer. Provide a detailed, step-by-step answer."}] + history[-6:] + [{"role": "user", "content": msg}]
    proposal = generate_fn(proposer_prompt, max_tokens=1000, model=model)
    
    # 2. Skeptic attacks the proposal
    skeptic_prompt = [
        {"role": "system", "content": "You are a ruthless Skeptic. Find the exact logical flaw, missing edge case, or fallacy in the Proposer's answer. If it is flawless, reply EXACTLY: FLAWLESS."},
        {"role": "user", "content": f"Question: {msg}\nProposer's Answer: {proposal}"}
    ]
    critique = generate_fn(skeptic_prompt, max_tokens=300, model=model)
    
    if "FLAWLESS" in critique.upper():
        return proposal
        
    # 3. Moderator synthesizes the final truth
    moderator_prompt = [
        {"role": "system", "content": system + "\nYou are the Moderator. Synthesize a final, bulletproof answer that addresses the Skeptic's critique. Ensure no logical fallacies remain."},
        {"role": "user", "content": f"Question: {msg}\nProposer's Answer: {proposal}\nSkeptic's Critique: {critique}\n\nProvide the final, corrected answer:"}
    ]
    final_answer = generate_fn(moderator_prompt, max_tokens=1000, model=model)
    return final_answer

def deliberate(msg: str, system: str, history: list, generate_fn, model: str, complexity: str = "medium", skill: str = "general") -> str:
    t0 = time.time()
    
    # Easy path: single generation
    if complexity == "easy":
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(prompt, max_tokens=2500, model=model)

    # Calculator / Math path: Chain of Code (SymPy)
    if skill == "calculator":
        sim_result = chain_of_code(msg, generate_fn, model)
        # Format the final response
        format_prompt = [
            {"role": "system", "content": "Use the simulation result to answer the user's question directly and concisely."},
            {"role": "user", "content": f"Question: {msg}\nSimulation Result: {sim_result}\n\nFinal Answer:"}
        ]
        return generate_fn(format_prompt, max_tokens=500, model=model)

    # Hard / Researcher path: Socratic Debate
    if complexity in ("hard", "medium") and skill in ("researcher", "general"):
        final_answer = socratic_debate(msg, system, history, generate_fn, model)
        print(f"[Deliberate] Socratic Debate completed, t={int((time.time()-t0)*1000)}ms")
        return final_answer

    # Fallback for medium/coder
    prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
    resp = generate_fn(prompt, max_tokens=2000, model=model)
    return resp
