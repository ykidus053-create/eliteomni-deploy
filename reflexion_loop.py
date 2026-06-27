import re
import subprocess
import tempfile
import os

# Upgraded: Expanded list of forbidden prototype phrases
PROTOTYPE_PHRASES = [
    "for simplicity", "for educational purposes", "basic version", "simplified",
    "example implementation", "skeleton", "stub", "placeholder", "demo",
    "in real implementation", "extend as needed", "similarly for others",
    "production implementation", "actual implementation", "full implementation"
]

def run_code(code: str) -> tuple[bool, str]:
    """Runs code in a real subprocess and captures exact tracebacks."""
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
    """Upgraded: Detects stubs AND forbidden prototype phrases."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        # Check for pass/... after a comment
        if re.search(r'#.*(implement|support|handle|cache|replicate|persist|commit|sync)', line, re.I):
            next_real = [l.strip() for l in lines[i+1:i+5] if l.strip()]
            if not next_real or all(l in ('pass', '...', 'return None', 'return {}', 'return []') for l in next_real):
                return True
                
    # Check for forbidden phrases
    code_lower = code.lower()
    for phrase in PROTOTYPE_PHRASES:
        if phrase in code_lower:
            return True
            
    return False

def reflexion_verify(code: str, generate_fn, model: str = "", max_rounds: int = 5) -> str:
    """Execution-based reflexion. Runs code, feeds stderr back to LLM."""
    memory = []
    
    for round_num in range(1, max_rounds + 1):
        stubs = has_stub(code)
        ok, output = run_code(code)
        
        if not stubs and ok:
            print(f"[Reflexion] Code passed execution on round {round_num}")
            break
            
        failures = []
        if stubs:
            failures.append("CRITICAL FAILURE: Code contains unimplemented stubs (pass/...) OR forbidden prototype phrases (e.g., 'for simplicity', 'educational purposes').")
            failures.append("You MUST write the EXACT, COMPLETE production code. No placeholders.")
        if not ok:
            failures.append(f"Runtime Error:\n{output[:500]}")
        
        if not failures: break

        reflection = f"""
[REFLEXION ROUND {round_num} - PRODUCTION ENFORCEMENT]
Execution failed. Errors detected:
{chr(10).join(failures)}

RULE: You are a Production Engineer. Fix the runtime error OR implement the missing logic completely.
DO NOT output a prototype. Output the FULL, RUNNABLE, production-grade code.
"""
        print(reflection)
        memory = (memory + [reflection])[-3:]
        
        prompt = "\n".join(memory) + f"\n\nOriginal Code:\n{code}\n\nCorrected Production Code:"
        msgs = [{"role": "user", "content": prompt}]
        new_code = generate_fn(msgs, max_tokens=4000) or code
        
        match = re.search(r'```python\n(.*?)```', new_code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = new_code

    return code

def get_max_rounds(skill: str = "coder") -> int:
    return {"researcher": 7, "coder": 5, "general": 3, "calculator": 2}.get(skill, 3)
