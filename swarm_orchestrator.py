import re, time, json, threading
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="swarm")

def _worker_agent(task: str, skill: str, generate_fn) -> dict:
    """A specialized worker agent that writes and tests one specific module."""
    try:
        from system_prompts import build_adaptive_prompt
        system = build_adaptive_prompt(skill, task)
        
        # Worker writes the code
        prompt = [
            {"role": "system", "content": system + "\nYou are a specialized Worker Agent. Write the COMPLETE, production-ready code for your assigned task. Output inside [PYTHON IMPL START]...[PYTHON IMPL END] tags."},
            {"role": "user", "content": task}
        ]
        code = generate_fn(prompt, max_tokens=4000)
        
        # Extract code
        match = re.search(r'\[PYTHON IMPL START\](.*?)\[PYTHON IMPL END\]', code, re.DOTALL)
        if match:
            return {"task": task, "code": match.group(1).strip(), "success": True}
        return {"task": task, "code": code, "success": False}
    except Exception as e:
        return {"task": task, "code": "", "success": False, "error": str(e)}

def run_swarm(msg: str, generate_fn) -> str:
    """Upgraded: Swarm Intelligence. Manager breaks task, workers execute in parallel."""
    t0 = time.time()
    
    # 1. Manager Agent decomposes the massive task into sub-tasks
    manager_prompt = [
        {"role": "system", "content": "You are a Manager Agent. Break the user's complex request into 2-4 distinct, independent coding sub-tasks. Output ONLY a JSON list of strings. Example: ['Write the FastAPI routes', 'Write the SQLite database schema']"},
        {"role": "user", "content": msg}
    ]
    
    try:
        raw = generate_fn(manager_prompt, max_tokens=300)
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return "" # Fallback if manager fails
        sub_tasks = json.loads(match.group())
    except:
        return ""
        
    if not sub_tasks:
        return ""
        
    print(f"[Swarm] Manager decomposed task into {len(sub_tasks)} sub-tasks. Spawning workers...")
    
    # 2. Spawn Worker Agents in parallel
    futures = [_executor.submit(_worker_agent, task, "coder", generate_fn) for task in sub_tasks]
    results = []
    for fut in futures:
        try:
            res = fut.result(timeout=60)
            if res["success"]:
                print(f"[Swarm] Worker completed: {res['task'][:50]}...")
                results.append(res)
        except:
            pass
            
    if not results:
        return ""
        
    # 3. Manager Agent synthesizes the parallel outputs into a final cohesive system
    print(f"[Swarm] Merging {len(results)} worker outputs...")
    synth_input = "Original Request: " + msg + "\n\nHere are the independently developed modules:\n"
    for i, res in enumerate(results):
        synth_input += f"\n--- MODULE {i+1}: {res['task']} ---\n```python\n{res['code'][:2000]}\n```\n"
        
    synth_prompt = [
        {"role": "system", "content": "You are the Manager Agent. Integrate the independently developed modules into a single, cohesive, production-ready system. Resolve any import conflicts. Output the final merged code inside [PYTHON IMPL START]...[PYTHON IMPL END] tags."},
        {"role": "user", "content": synth_input}
    ]
    
    final_code = generate_fn(synth_prompt, max_tokens=8000)
    match = re.search(r'\[PYTHON IMPL START\](.*?)\[PYTHON IMPL END\]', final_code, re.DOTALL)
    
    print(f"[Swarm] Synthesis complete, t={int((time.time()-t0)*1000)}ms")
    return match.group(1).strip() if match else final_code
