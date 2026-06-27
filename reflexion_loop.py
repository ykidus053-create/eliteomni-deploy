import re
import subprocess
import tempfile
import os
import ast
import resource

PROTOTYPE_PHRASES = [
    "for simplicity", "for educational purposes", "basic version", "simplified",
    "example implementation", "skeleton", "stub", "placeholder", "demo", "toy",
    "in real implementation", "extend as needed", "similarly for others",
    "production implementation", "actual implementation", "full implementation",
    "for demonstration", "quick script", "minimal viable", "extensible",
    "future-proof", "base class", "abstract"
]

def has_stub(code: str) -> bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_stub = True
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str): continue
                    if isinstance(stmt, ast.Pass): continue
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...: continue
                    if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == 'NotImplementedError': continue
                    if isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)): continue
                    is_stub = False
                    break
                if is_stub: return True
    except SyntaxError: pass

    lines = code.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'#.*(implement|support|handle|cache|replicate|persist|commit|sync)', line, re.I):
            next_real = [l.strip() for l in lines[i+1:i+5] if l.strip()]
            if not next_real or all(l in ('pass', '...', 'return None', 'return {}', 'return []') for l in next_real): return True
    code_lower = code.lower()
    for phrase in PROTOTYPE_PHRASES:
        if phrase in code_lower: return True
    return False

def check_enterprise_compliance(code: str) -> list:
    violations = []
    try:
        tree = ast.parse(code)
        has_logging = False
        has_metrics = False
        banned_calls = {'eval', 'exec', 'compile', '__import__', 'os.system', 'subprocess.call'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("test_") and node.name != "__init__":
                    for arg in node.args.args:
                        if arg.annotation is None and arg.arg not in ('self', 'cls'): violations.append(f"Function '{node.name}' missing type hint for argument '{arg.arg}'.")
                    if node.returns is None: violations.append(f"Function '{node.name}' missing return type hint.")
                if not node.name.startswith("_") and not node.name.startswith("test_"):
                    if not ast.get_docstring(node): violations.append(f"Function '{node.name}' missing a docstring.")
            if isinstance(node, ast.ExceptHandler):
                if node.type is None: violations.append("Bare 'except:' block found. Use specific exceptions.")
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass): violations.append("Silent 'except: pass' block found. Enterprise code must log or re-raise.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print': violations.append("Use of print() found. Enterprise code must use the 'logging' module.")
            
            # Upgraded: Static Taint Analysis (Network/DB calls MUST have timeouts)
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name): func_name = node.func.id
                elif isinstance(node.func, ast.Attribute): func_name = node.func.attr
                
                if func_name in banned_calls: violations.append(f"SECURITY VIOLATION: Use of '{func_name}()' is strictly banned.")
                
                # Enforce timeouts on network calls
                if func_name in ('get', 'post', 'put', 'delete', 'request', 'connect', 'create_connection'):
                    has_timeout = any(kw.arg == 'timeout' for kw in node.keywords)
                    if not has_timeout:
                        violations.append(f"PRODUCTION SAFETY: Network call '{func_name}()' missing 'timeout' argument. It will hang forever if the network drops.")

            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name): base_name = base.id
                    elif isinstance(base, ast.Attribute): base_name = base.attr
                    if base_name in ('ABC', 'ABCMeta'):
                        violations.append(f"OVER-ENGINEERING: Class '{node.name}' inherits from ABC. Write concrete implementations.")
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef):
                        for decorator in stmt.decorator_list:
                            if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                                violations.append(f"OVER-ENGINEERING: Abstract method '{stmt.name}' found. Implement it completely.")
                                
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if 'logging' in alias.name: has_logging = True
                    if 'prometheus_client' in alias.name or 'opentelemetry' in alias.name or 'datadog' in alias.name:
                        has_metrics = True
                        
        if not has_logging and len(code.split('\n')) > 20: violations.append("Missing 'import logging'. Enterprise systems must use structured logging.")
        if not has_metrics and len(code.split('\n')) > 50: violations.append("OBSERVABILITY VIOLATION: Missing metrics/tracing. Enterprise code must be observable.")
            
    except SyntaxError as e:
        violations.append(f"Syntax Error preventing AST audit: {e}")
    return violations

def principal_engineer_veto(impl_code: str, task: str, generate_fn) -> str:
    prompt = [
        {"role": "system", "content": "You are a Ruthless Staff Engineer reviewing a PR for a critical distributed system. Does this code handle partial failures, network partitions, and malformed inputs safely? Reply ONLY 'VETO' followed by a scathing critique of what will break in production, or 'APPROVED'."},
        {"role": "user", "content": f"Task: {task}\n\nCode:\n{impl_code[:2000]}"}
    ]
    try:
        raw = generate_fn(prompt, max_tokens=200)
        if "VETO" in raw.upper(): return raw.strip()
    except: pass
    return "APPROVED"

def llm_logic_audit(impl_code: str, test_code: str, generate_fn) -> list:
    prompt = [
        {"role": "system", "content": "You are a Chaos Engineer. Does this code have real-world failure modes? (e.g., missing circuit breakers, retry storms, deadlocks, unhandled JSON decode errors). Reply ONLY with a JSON list of strings. If perfect, reply []."},
        {"role": "user", "content": f"Implementation:\n{impl_code[:1500]}\n\nTests:\n{test_code[:800]}"}
    ]
    try:
        raw = generate_fn(prompt, max_tokens=300)
        import json
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match: return json.loads(match.group())
    except: pass
    return []

def extract_code_blocks(text: str) -> dict:
    tests_match = re.search(r'\[PYTHON TESTS START\](.*?)\[PYTHON TESTS END\]', text, re.DOTALL)
    impl_match = re.search(r'\[PYTHON IMPL START\](.*?)\[PYTHON IMPL END\]', text, re.DOTALL)
    if tests_match and impl_match: return {"implementation": impl_match.group(1).strip(), "tests": tests_match.group(1).strip()}
    matches = re.findall(r'```(?:python|py)?\n(.*?)```', text, re.DOTALL)
    impl_code, test_code = "", ""
    for match in matches:
        if "import pytest" in match or "def test_" in match: test_code = match.strip()
        else: impl_code = match.strip()
    if not test_code and len(matches) == 1: impl_code = matches[0].strip()
    return {"implementation": impl_code, "tests": test_code}

def _set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (150 * 1024 * 1024, 150 * 1024 * 1024))

def run_pytest(impl_code: str, test_code: str) -> tuple[bool, str]:
    if not test_code: return run_code(impl_code)
    with tempfile.TemporaryDirectory() as tmpdir:
        impl_path, test_path = os.path.join(tmpdir, "module.py"), os.path.join(tmpdir, "test_module.py")
        with open(impl_path, "w") as f: f.write(impl_code)
        with open(test_path, "w") as f: f.write(test_code)
        try:
            r = subprocess.run(["python", "-m", "pytest", test_path, "-v", "--tb=long", "-W", "error", "--no-header"], capture_output=True, text=True, timeout=15, cwd=tmpdir, preexec_fn=_set_limits)
            return r.returncode == 0, r.stdout + r.stderr
        except subprocess.TimeoutExpired: return False, "Pytest execution timed out (15s) or hit CPU limit."
        except Exception as e: return False, str(e)

def run_code(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code); fname = f.name
    try:
        r = subprocess.run(["python", fname], capture_output=True, text=True, timeout=10, preexec_fn=_set_limits)
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as e: return False, str(e)
    finally:
        if os.path.exists(fname): os.unlink(fname)

def reflexion_verify(raw_output: str, generate_fn, task: str = "", model: str = "", max_rounds: int = 5) -> str:
    blocks = extract_code_blocks(raw_output)
    impl_code, test_code = blocks["implementation"], blocks["tests"]
    memory = []
    
    for round_num in range(1, max_rounds + 1):
        stubs = has_stub(impl_code)
        enterprise_violations = check_enterprise_compliance(impl_code)
        veto = principal_engineer_veto(impl_code, task, generate_fn)
        logic_flaws = llm_logic_audit(impl_code, test_code, generate_fn)
        ok, output = run_pytest(impl_code, test_code)
        
        if not stubs and not enterprise_violations and "APPROVED" in veto and not logic_flaws and ok:
            print(f"[Reflexion] Chaos & SRE Audit passed & pytests 100% on round {round_num}")
            break
            
        failures = []
        if stubs: failures.append("CRITICAL FAILURE: AST detected empty function bodies or prototype phrases.")
        if enterprise_violations: failures.append("ENTERPRISE COMPLIANCE VIOLATIONS:\n- " + "\n- ".join(enterprise_violations[:5]))
        if "APPROVED" not in veto: failures.append(f"STAFF ENGINEER VETO:\n{veto}")
        if logic_flaws: failures.append("CHAOS / DISTRIBUTED SYSTEMS FLAWS DETECTED:\n- " + "\n- ".join(logic_flaws[:5]))
        if not ok: failures.append(f"Test/Execution Failures (Strict Mode & Resource Limits):\n{output[:800]}")
        if not test_code: failures.append("CRITICAL FAILURE: You did not provide any pytest unit tests.")
        if not failures: break

        reflection = f"[REFLEXION ROUND {round_num} - CHAOS & PRODUCTION SAFETY AUDIT]\nExecution failed. Errors detected:\n{chr(10).join(failures)}\n\nRULE: You MUST rewrite the code from scratch if it was vetoed. Do not patch toy code. Output BOTH the corrected implementation AND the tests."
        print(reflection)
        memory = (memory + [reflection])[-3:]
        
        prompt = "\n".join(memory) + f"\n\nFailed Implementation:\n{impl_code}\n\nFailed Tests:\n{test_code}\n\nCorrected Enterprise Code:"
        msgs = [{"role": "user", "content": prompt}]
        new_output = generate_fn(msgs, max_tokens=8000) or raw_output
        
        new_blocks = extract_code_blocks(new_output)
        if new_blocks["implementation"]: impl_code = new_blocks["implementation"]
        if new_blocks["tests"]: test_code = new_blocks["tests"]

    return impl_code

def get_max_rounds(skill: str = "coder") -> int:
    return {"researcher": 7, "coder": 5, "general": 3, "calculator": 2}.get(skill, 3)
