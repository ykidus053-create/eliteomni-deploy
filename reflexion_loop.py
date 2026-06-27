import re
import subprocess
import tempfile
import os
import ast

PROTOTYPE_PHRASES = [
    "for simplicity", "for educational purposes", "basic version", "simplified",
    "example implementation", "skeleton", "stub", "placeholder", "demo",
    "in real implementation", "extend as needed", "similarly for others",
    "production implementation", "actual implementation", "full implementation"
]

def has_stub(code: str) -> bool:
    """Upgraded: Uses AST to detect empty function bodies (pass, ..., return None)."""
    # 1. AST check for empty function bodies
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_stub = True
                for stmt in node.body:
                    # Ignore docstrings
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        continue
                    # Ignore 'pass'
                    if isinstance(stmt, ast.Pass):
                        continue
                    # Ignore '...' (Ellipsis)
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                        continue
                    # Ignore 'raise NotImplementedError'
                    if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == 'NotImplementedError':
                        continue
                    # Ignore 'return None' or 'return'
                    if isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)):
                        continue
                    
                    # If it has any other statement, it's a real implementation
                    is_stub = False
                    break
                if is_stub:
                    print(f"[Reflexion] AST detected empty function body: {node.name}")
                    return True
    except SyntaxError:
        pass # Let the execution gate catch syntax errors

    # 2. Regex check for prototype phrases
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'#.*(implement|support|handle|cache|replicate|persist|commit|sync)', line, re.I):
            next_real = [l.strip() for l in lines[i+1:i+5] if l.strip()]
            if not next_real or all(l in ('pass', '...', 'return None', 'return {}', 'return []') for l in next_real):
                return True
    code_lower = code.lower()
    for phrase in PROTOTYPE_PHRASES:
        if phrase in code_lower: return True
    return False

def extract_code_blocks(text: str) -> dict:
    """Extracts implementation and test blocks from LLM output."""
    tests_match = re.search(r'\[PYTHON TESTS START\](.*?)\[PYTHON TESTS END\]', text, re.DOTALL)
    impl_match = re.search(r'\[PYTHON IMPL START\](.*?)\[PYTHON IMPL END\]', text, re.DOTALL)
    
    if tests_match and impl_match:
        return {"implementation": impl_match.group(1).strip(), "tests": tests_match.group(1).strip()}

    matches = re.findall(r'```(?:python|py)?\n(.*?)```', text, re.DOTALL)
    impl_code = ""
    test_code = ""
    for match in matches:
        if "import pytest" in match or "def test_" in match:
            test_code = match.strip()
        else:
            impl_code = match.strip()
    if not test_code and len(matches) == 1:
        impl_code = matches[0].strip()
    return {"implementation": impl_code, "tests": test_code}

def run_pytest(impl_code: str, test_code: str) -> tuple[bool, str]:
    if not test_code:
        return run_code(impl_code)
    with tempfile.TemporaryDirectory() as tmpdir:
        impl_path = os.path.join(tmpdir, "module.py")
        test_path = os.path.join(tmpdir, "test_module.py")
        with open(impl_path, "w") as f: f.write(impl_code)
        with open(test_path, "w") as f: f.write(test_code)
        try:
            r = subprocess.run(["python", "-m", "pytest", test_path, "-v", "--tb=long"], capture_output=True, text=True, timeout=30, cwd=tmpdir)
            return r.returncode == 0, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return False, "Pytest execution timed out after 30 seconds."
        except Exception as e:
            return False, str(e)

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
        if os.path.exists(fname): os.unlink(fname)

def reflexion_verify(raw_output: str, generate_fn, model: str = "", max_rounds: int = 5) -> str:
    """Upgraded: AST-level stub detection forces the AI to write real implementations."""
    blocks = extract_code_blocks(raw_output)
    impl_code = blocks["implementation"]
    test_code = blocks["tests"]
    memory = []
    
    for round_num in range(1, max_rounds + 1):
        stubs = has_stub(impl_code)
        ok, output = run_pytest(impl_code, test_code)
        
        if not stubs and ok:
            print(f"[Reflexion] Pytests passed (100%) and AST verified no stubs on round {round_num}")
            break
            
        failures = []
        if stubs:
            failures.append("CRITICAL FAILURE: AST analysis detected empty function bodies (pass, ..., return None) OR forbidden prototype phrases.")
        if not ok:
            failures.append(f"Test/Execution Failures:\n{output[:800]}")
        if not test_code:
            failures.append("CRITICAL FAILURE: You did not provide any pytest unit tests.")
            
        if not failures: break

        reflection = f"[REFLEXION ROUND {round_num} - STRICT TDD ENFORCEMENT]\nExecution failed. Errors detected:\n{chr(10).join(failures)}\n\nRULE: You MUST fix the runtime error OR implement the missing logic completely.\nYou MUST output BOTH the corrected implementation AND the tests in separate python blocks."
        print(reflection)
        memory = (memory + [reflection])[-3:]
        
        prompt = "\n".join(memory) + f"\n\nFailed Implementation:\n{impl_code}\n\nFailed Tests:\n{test_code}\n\nCorrected Code:"
        msgs = [{"role": "user", "content": prompt}]
        new_output = generate_fn(msgs, max_tokens=4000) or raw_output
        
        new_blocks = extract_code_blocks(new_output)
        if new_blocks["implementation"]: impl_code = new_blocks["implementation"]
        if new_blocks["tests"]: test_code = new_blocks["tests"]

    return impl_code

def get_max_rounds(skill: str = "coder") -> int:
    return {"researcher": 7, "coder": 5, "general": 3, "calculator": 2}.get(skill, 3)
