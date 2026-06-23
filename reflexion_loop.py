import re
import subprocess
import tempfile
import os

def run_code(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run(["python", fname], capture_output=True, text=True, timeout=30)
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(fname)

def extract_claims(code: str) -> list[str]:
    """Extract full comment lines that claim behavior."""
    return re.findall(r'#\s*([^\n]{10,})', code)

def has_stub(code: str) -> bool:
    """True if code has pass/... with no real logic after a comment claim."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'#.*(implement|support|handle|cache|replicate|persist|commit|sync|elect|recover|expir)', line, re.I):
            next_real = [l.strip() for l in lines[i+1:i+5] if l.strip()]
            if not next_real or all(l in ('pass', '...', 'return None', 'return {}', 'return []') for l in next_real):
                return True
    return False

def reflexion_verify(code: str, generate_fn, model: str = "", max_rounds: int = 5) -> str:
    memory = []

    for round_num in range(1, max_rounds + 1):
        claims = extract_claims(code)
        stubs = has_stub(code)
        ok, output = run_code(code)

        if not stubs and ok:
            break

        failures = []
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if re.search(r'#.*(implement|support|handle|cache|replicate|persist|commit|sync|elect|recover|expir)', line, re.I):
                next_real = [l.strip() for l in lines[i+1:i+5] if l.strip()]
                if not next_real or all(l in ('pass', '...', 'return None') for l in next_real):
                    failures.append(f"Comment '{line.strip()}' has no implementation — only stub follows")

        if not failures:
            break

        reflection = f"""
[REFLEXION ROUND {round_num}]
Unimplemented stubs found:
{chr(10).join(failures[:5])}

Runtime output: {output[:200] if not ok else 'runs but logic missing'}

RULE: Every comment claiming behavior MUST be followed by real code, not pass or ...
Rewrite with actual implementations:
"""
        print(reflection)
        memory = (memory + [reflection])[-3:]
        prompt = "\n".join(memory) + f"\n\nCode to fix:\n{code}\n\nRewritten code with real implementations:"
        code = generate_fn(prompt, model) or code

    return code

def get_max_rounds(skill: str = "coder") -> int:
    """Return tool round budget by skill type."""
    return {"researcher": 7, "coder": 5, "general": 3, "calculator": 2}.get(skill, 3)
