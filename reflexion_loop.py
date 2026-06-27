import re
import subprocess
import tempfile
import os

def run_code(code: str) -> tuple[bool, str]:
    """Upgraded: Runs code in a real subprocess and captures exact tracebacks."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run(["python", fname], capture_output=True, text=True, timeout=30)
        return r.returncode == 0, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return False, "Execution timed out after 30 seconds."
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(fname):
            os.unlink(fname)

def has_stub(code: str) -> bool:
    """True if code has pass/... with no real logic."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'#.*(implement|support|handle|cache|replicate|persist|commit|sync)', line, re.I):
            next_real = [l.strip() for l in lines[i+1:i+5] if l.strip()]
            if not next_real or all(l in ('pass', '...', 'return None', 'return {}', 'return []') for l in next_real):
                return True
    return False

def reflexion_verify(code: str, generate_fn, model: str = "", max_rounds: int = 5) -> str:
    """Upgraded: Execution-based reflexion. Runs code, feeds stderr back to LLM."""
    memory = []
    
    for round_num in range(1, max_rounds + 1):
        stubs = has_stub(code)
        ok, output = run_code(code)
        
        if not stubs and ok:
            print(f"[Reflexion] Code passed execution on round {round_num}")
            break
            
        failures = []
        if stubs: failures.append("Code contains unimplemented stubs (pass or ...).")
        if not ok: failures.append(f"Runtime Error:\n{output[:500]}")
        
        if not failures: break

        reflection = f"""
[REFLEXION ROUND {round_num}]
Execution failed. Errors detected:
{chr(10).join(failures)}

RULE: Fix the runtime error or implement the missing logic. Output the COMPLETE, corrected code.
"""
        print(reflection)
        memory = (memory + [reflection])[-3:]
        
        prompt = "\n".join(memory) + f"\n\nOriginal Code:\n{code}\n\nCorrected Code:"
        # Upgraded: Use standard generate_fn signature
        msgs = [{"role": "user", "content": prompt}]
        new_code = generate_fn(msgs, max_tokens=2000) or code
        
        # Extract code block if present
        match = re.search(r'```python\n(.*?)```', new_code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = new_code

    return code

def get_max_rounds(skill: str = "coder") -> int:
    return {"researcher": 7, "coder": 5, "general": 3, "calculator": 2}.get(skill, 3)
