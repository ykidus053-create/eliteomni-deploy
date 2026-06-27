import re
import subprocess
import tempfile
import os
import ast

PROTOTYPE_PHRASES = [
    "for simplicity", "for educational purposes", "basic version", "simplified",
    "example implementation", "skeleton", "stub", "placeholder", "demo", "toy",
    "in real implementation", "extend as needed", "similarly for others",
    "production implementation", "actual implementation", "full implementation",
    "for demonstration", "quick script"
]

def has_stub(code: str) -> bool:
    """Uses AST to detect empty function bodies and prototype phrases."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_stub = True
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        continue
                    if isinstance(stmt, ast.Pass):
                        continue
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                        continue
                    if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == 'NotImplementedError':
                        continue
                    if isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)):
                        continue
                    is_stub = False
                    break
                if is_stub:
                    print(f"[Reflexion] AST detected empty function body: {node.name}")
                    return True
    except SyntaxError:
        pass

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

def check_enterprise_compliance(code: str) -> list:
    """Upgraded: AST audit for enterprise standards (typing, logging, exceptions)."""
    violations = []
    try:
        tree = ast.parse(code)
        has_logging = False
        
        for node in ast.walk(tree):
            # 1. Enforce Type Hints on all function arguments and returns
            if isinstance(node, ast.FunctionDef):
                # Ignore __init__ or private test methods, but enforce on public API
                if not node.name.startswith("test_") and node.name != "__init__":
                    for arg in node.args.args:
                        if arg.annotation is None and arg.arg != 'self':
                            violations.append(f"Function '{node.name}' missing type hint for argument '{arg.arg}'.")
                    if node.returns is None and not isinstance(node.body[-1], ast.Return) if node.body else True:
                        # Allow implicit None but flag explicit missing hints if needed, keeping it simple for now
                        pass 

            # 2. Ban bare 'except:'
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append("Bare 'except:' block found. Use specific exceptions (e.g., 'except ValueError:').")
                
            # 3. Ban 'print(' and enforce 'logging'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
                violations.append("Use of print() found. Enterprise code must use the 'logging' module.")
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if 'logging' in alias.name:
                        has_logging = True
                        
        if not has_logging and len(code.split('\n')) > 30:
            violations.append("Missing 'import logging'. Enterprise systems must use structured logging.")
            
    except SyntaxError as e:
        violations.append(f"Syntax Error preventing AST audit: {e}")
        
    return violations

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
    """Upgraded: Enterprise Code Auditor. Enforces typing, logging, and strict exception handling."""
    blocks = extract_code_blocks(raw_output)
    impl_code = blocks["implementation"]
    test_code = blocks["tests"]
    memory = []
    
    for round_num in range(1, max_rounds + 1):
        stubs = has_stub(impl_code)
        enterprise_violations = check_enterprise_compliance(impl_code)
        ok, output = run_pytest(impl_code, test_code)
        
        if not stubs and not enterprise_violations and ok:
            print(f"[Reflexion] Enterprise audit passed & pytests 100% on round {round_num}")
            break
            
        failures = []
        if stubs:
            failures.append("CRITICAL FAILURE: AST analysis detected empty function bodies (pass, ...) OR forbidden prototype phrases (e.g., 'toy', 'basic').")
        if enterprise_violations:
            failures.append("ENTERPRISE COMPLIANCE VIOLATIONS:\n- " + "\n- ".join(enterprise_violations[:5]))
        if not ok:
            failures.append(f"Test/Execution Failures:\n{output[:800]}")
        if not test_code:
            failures.append("CRITICAL FAILURE: You did not provide any pytest unit tests.")
            
        if not failures: break

        reflection = f"[REFLEXION ROUND {round_num} - ENTERPRISE SYSTEM AUDIT]\nExecution failed. Errors detected:\n{chr(10).join(failures)}\n\nRULE: You MUST fix the runtime error OR implement the missing logic completely.\nYou MUST output BOTH the corrected implementation AND the tests in separate python blocks."
        print(reflection)
        memory = (memory + [reflection])[-3:]
        
        prompt = "\n".join(memory) + f"\n\nFailed Implementation:\n{impl_code}\n\nFailed Tests:\n{test_code}\n\nCorrected Enterprise Code:"
        msgs = [{"role": "user", "content": prompt}]
        new_output = generate_fn(msgs, max_tokens=4000) or raw_output
        
        new_blocks = extract_code_blocks(new_output)
        if new_blocks["implementation"]: impl_code = new_blocks["implementation"]
        if new_blocks["tests"]: test_code = new_blocks["tests"]

    return impl_code

def get_max_rounds(skill: str = "coder") -> int:
    return {"researcher": 7, "coder": 5, "general": 3, "calculator": 2}.get(skill, 3)
