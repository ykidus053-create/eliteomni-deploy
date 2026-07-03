import threading, time, os, logging, re
log = logging.getLogger(__name__)

def _apo_loop(generate_fn):
    """Upgraded: Autonomous Prompt Optimization. A/B tests system prompts and evolves them."""
    time.sleep(60)  # Wait 1 min after startup
    log.info("[APO] Starting Autonomous Prompt Optimization loop...")
    
    while True:
        try:
            from system_prompts import SYSTEM_PROMPTS
            current_coder_prompt = SYSTEM_PROMPTS["coder"]
            
            # 1. Generate a mutated prompt
            mutate_prompt = [
                {"role": "system", "content": "You are an AI prompt optimizer. Mutate the following system prompt to make the AI write better, more bulletproof code. Keep it concise. Output ONLY the new prompt text."},
                {"role": "user", "content": current_coder_prompt}
            ]
            mutated_prompt = generate_fn(mutate_prompt, max_tokens=500)
            
            # 2. Test both prompts on a synthetic task
            test_task = "Write a thread-safe rate limiter in Python"
            
            def _test(prompt):
                msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": test_task}]
                return generate_fn(msgs, max_tokens=1000)
                
            old_resp = _test(current_coder_prompt)
            new_resp = _test(mutated_prompt)
            
            # 3. LLM Judge picks the winner
            judge_prompt = [
                {"role": "system", "content": "You are a Staff Engineer. Which code implementation is more production-ready and bulletproof? Reply ONLY 'OLD' or 'NEW'."},
                {"role": "user", "content": f"OLD:\n{old_resp[:800]}\n\nNEW:\n{new_resp[:800]}"}
            ]
            verdict = generate_fn(judge_prompt, max_tokens=5).upper()
            
            # 4. If NEW wins, permanently overwrite system_prompts.py
            if "NEW" in verdict:
                log.info("[APO] New prompt won! Permanently updating system_prompts.py")
                with open("system_prompts.py", "r") as f:
                    content = f.read()
                # Replace the coder prompt string
                new_content = content.replace(current_coder_prompt, mutated_prompt.replace('"', '\\"'))
                with open("system_prompts.py", "w") as f:
                    f.write(new_content)
        except Exception as e:
            log.error(f"[APO] Error: {e}")
            
        time.sleep(3600)  # Run every hour

def start_apo_engine(generate_fn):
    t = threading.Thread(target=_apo_loop, args=(generate_fn,), daemon=True, name="apo_engine")
    t.start()
    print("[Startup] ✓ Autonomous Prompt Optimizer (APO) started.")
