import os
import re
import sys
import json
import math
import traceback
from io import StringIO

def run_adaptive_thinking(prompt_msgs, generate_fn, active_model, task_payload):
    """FEATURE 1: INTERLEAVED ADAPTIVE THINKING"""
    complexity_signals = ["algorithm", "optimize", "refactor", "bug", "class", "async", "database", "matrix"]
    matches = sum(1 for signal in complexity_signals if signal in task_payload.lower())
    
    # Dynamically adjust processing allocation based on logic complexity
    adaptive_tokens = 3000 if matches >= 3 else 1200
    print(f"[Engine] Complexity signals found: {matches}. Allocation budget set to {adaptive_tokens} tokens.")

    thinking_prompt = [
        {"role": "system", "content": "You are a senior engineer. You must open your output with an extensive internal chain of thought wrapped strictly within <think>...</think> tags. Evaluate multiple design angles, check constraints, and plan step-by-step before resolving the request."}
    ] + prompt_msgs
    
    raw_thought = generate_fn(thinking_prompt, max_tokens=adaptive_tokens, model=active_model)
    if not raw_thought.strip().startswith("<think>"):
        raw_thought = f"<think>\n[Autonomous Readjustment Matrix]\n{raw_thought.strip()}\n</think>"
    return raw_thought, matches

def verify_and_repair_output(prompt_msgs, raw_thought, generate_fn, active_model):
    """FEATURE 2: AUTONOMOUS OUTPUT VERIFICATION"""
    formulation_prompt = prompt_msgs + [{"role": "assistant", "content": raw_thought}]
    final_draft = generate_fn(formulation_prompt, max_tokens=2500, model=active_model)
    
    # Extract code blocks to verify compilation and execution safety
    python_blocks = re.findall(r"```python\n(.*?)```", final_draft, re.DOTALL)
    for idx, block in enumerate(python_blocks, 1):
        if any(bad in block for bad in ["import socket", "os.system", "subprocess"]):
            print(f"  [!] Constraint Violation: Prohibited execution pattern blocked.")
            final_draft = final_draft.replace(block, "# [REDACTED: System verification engine caught safety deviation.]")
            continue
            
        old_stdout = sys.stdout
        redirected_out = StringIO()
        sys.stdout = redirected_out
        try:
            local_env = {}
            exec(block, {}, local_env)
            sys.stdout = old_stdout
            print(f"  [✓] Verification Pass {idx}: Code executed successfully with zero runtime errors.")
        except Exception as e:
            sys.stdout = old_stdout
            error_trace = "".join(traceback.format_exception_only(type(e), e)).strip()
            print(f"  [!] Verification Fail {idx}: {error_trace}. Deploying self-repair loop...")
            
            repair_prompt = formulation_prompt + [
                {"role": "assistant", "content": final_draft},
                {"role": "user", "content": f"[AUTONOMOUS FAULT] Code failed verification: {error_trace}. Regenerate the script with fixed scopes."}
            ]
            final_draft = generate_fn(repair_prompt, max_tokens=2500, model=active_model)
            
    return final_draft

def process_vision_grid(vision_data):
    """FEATURE 3: HIGH-RESOLUTION VISION PROCESSING"""
    if not vision_data:
        return
    img_width, img_height = vision_data.get("resolution", (1000, 1000))
    megapixels = (img_width * img_height) / 1000000.0
    if megapixels > 3.75:
        scale_factor = math.sqrt(3.75 / megapixels)
        img_width, img_height = int(img_width * scale_factor), int(img_height * scale_factor)
        print(f"[Vision] Resolution scale clamped to exactly 3.75 MP limit: {img_width}x{img_height}")
    return img_width, img_height

def manage_persistent_memory(user_id, write_data=None):
    """FEATURE 4: FILE SYSTEM-BASED MEMORY"""
    memory_file = f"memory_store_{user_id}.json"
    if write_data:
        try:
            with open(memory_file, "w") as mf:
                json.dump(write_data, mf, indent=4)
        except Exception: pass
        return write_data
        
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r") as mf:
                return json.load(mf)
        except Exception: pass
    return {}
