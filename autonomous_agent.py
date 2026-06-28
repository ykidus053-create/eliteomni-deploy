import re, time, json, subprocess, os, tempfile
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="autonomous")

def _execute_tool(action_str: str) -> str:
    """Parses and executes an autonomous action."""
    try:
        match = re.search(r'<action:\s*(\w+)\("(.+?)"\)>', action_str)
        if not match: return "[Tool Error: Invalid format. Use <action: tool(\"arg\")>]"
        
        tool, arg = match.groups()
        
        if tool == "web_search":
            from modules.services.search import tool_search
            return tool_search(arg)[:2000]
            
        elif tool == "read_file":
            if os.path.exists(arg):
                with open(arg, 'r', encoding='utf-8', errors='ignore') as f: return f.read()[:3000]
            return f"[Error: File '{arg}' not found]"
            
        elif tool == "write_file":
            parts = arg.split(":::", 1)
            if len(parts) == 2:
                with open(parts[0], 'w') as f: f.write(parts[1])
                return f"[Successfully wrote to {parts[0]}]"
            return "[Error: Invalid write_file format]"
            
        elif tool == "run_python":
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(arg); fname = f.name
            r = subprocess.run(["python", fname], capture_output=True, text=True, timeout=15)
            os.unlink(fname)
            return (r.stdout + r.stderr)[:2000]
            
        elif tool == "run_bash":
            blocked = ['rm -rf', 'sudo', 'shutdown']
            if any(b in arg for b in blocked): return "[Error: Command blocked]"
            r = subprocess.run(arg, shell=True, capture_output=True, text=True, timeout=15)
            return (r.stdout + r.stderr)[:2000]
            
        return f"[Error: Unknown tool '{tool}']"
    except Exception as e:
        return f"[Tool Error: {str(e)}]"

def run_autonomous_task(msg: str, system: str, history: list, generate_fn, model: str, max_steps: int = 6) -> str:
    """Upgraded: Fully Autonomous Plan-and-Solve Loop with parallel tool execution."""
    t0 = time.time()
    
    # 1. Generate the Plan
    plan_prompt = [
        {"role": "system", "content": system + "\nYou are an Autonomous Agent. Break the user's request into a step-by-step plan. Output your plan inside <plan> tags. For each step, specify if you need to use a tool."},
        {"role": "user", "content": msg}
    ]
    plan_resp = generate_fn(plan_prompt, max_tokens=1000, model=model)
    
    observations = []
    current_thought = plan_resp
    
    for step in range(max_steps):
        if "<final_answer:" in current_thought:
            final = current_thought.split("<final_answer:")[1].rsplit(">", 1)[0].strip()
            print(f"[Autonomous] Task completed in {step} steps, t={int((time.time()-t0)*1000)}ms")
            return final
            
        actions = re.findall(r'<action:\s*\w+\(".+?"\)>', current_thought)
        
        if actions:
            # Execute all actions found in this step in parallel
            futures = {_executor.submit(_execute_tool, act): act for act in actions}
            tool_results = []
            for fut in futures:
                act = futures[fut]
                try:
                    res = fut.result(timeout=15)
                    tool_results.append(f"{act}\n[OBSERVATION]: {res}")
                    print(f"[Autonomous] Executed: {act[:50]}... -> {res[:50]}...")
                except Exception as e:
                    tool_results.append(f"{act}\n[OBSERVATION]: [Tool Error: {e}]")
                    
            observations.extend(tool_results)
            
            react_prompt = [
                {"role": "system", "content": system + "\nYou executed actions and observed the results. What is your next step? Output <action: tool(\"arg\")> or <final_answer: response>."},
                {"role": "user", "content": f"{msg}\n\nHistory:\n" + "\n".join(observations[-6:])}
            ]
            current_thought = generate_fn(react_prompt, max_tokens=2000, model=model)
        else:
            # No actions found, ask for final answer or next action
            prompt = [
                {"role": "system", "content": system + "\nIf you have enough information, output <final_answer: your response>. Otherwise, output your next <action: tool(\"arg\")>."},
                {"role": "user", "content": f"{msg}\n\nHistory:\n" + "\n".join(observations[-6:])}
            ]
            current_thought = generate_fn(prompt, max_tokens=2000, model=model)

    # Max steps reached, force final synthesis
    synth_prompt = [
        {"role": "system", "content": system + "\nSynthesize your final answer based on the observations gathered."},
        {"role": "user", "content": f"{msg}\n\nObservations:\n" + "\n".join(observations)}
    ]
    return generate_fn(synth_prompt, max_tokens=2000, model=model)
