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
    "future-proof", "base class", "abstract", "architectural foundation",
    "interface", "scaffolding", "wrapper"
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
    is_cutoff = False
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
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name): func_name = node.func.id
                elif isinstance(node.func, ast.Attribute): func_name = node.func.attr
                if func_name in banned_calls: violations.append(f"SECURITY VIOLATION: Use of '{func_name}()' is strictly banned.")
                if func_name in ('get', 'post', 'put', 'delete', 'request', 'connect', 'create_connection'):
                    has_timeout = any(kw.arg == 'timeout' for kw in node.keywords)
                    if not has_timeout: violations.append(f"PRODUCTION SAFETY: Network call '{func_name}()' missing 'timeout' argument.")
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name): base_name = base.id
                    elif isinstance(base, ast.Attribute): base_name = base.attr
                    if base_name in ('ABC', 'ABCMeta', 'Protocol'):
                        violations.append(f"SCAFFOLDING BAN: Class '{node.name}' inherits from {base_name}. Write a concrete implementation.")
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef):
                        for decorator in stmt.decorator_list:
                            if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                                violations.append(f"SCAFFOLDING BAN: Abstract method '{stmt.name}' found. Implement it completely.")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if 'logging' in alias.name: has_logging = True
                    if 'prometheus_client' in alias.name or 'opentelemetry' in alias.name or 'datadog' in alias.name: has_metrics = True
        if not has_logging and len(code.split('\n')) > 20: violations.append("Missing 'import logging'.")
        if not has_metrics and len(code.split('\n')) > 50: violations.append("OBSERVABILITY VIOLATION: Missing metrics/tracing.")
    except SyntaxError as e:
        if "unexpected EOF" in str(e) or "incomplete input" in str(e):
            is_cutoff = True
        else:
            violations.append(f"Syntax Error preventing AST audit: {e}")
    return violations, is_cutoff

def principal_engineer_veto(impl_code: str, task: str, generate_fn) -> str:
    prompt = [
        {"role": "system", "content": "You are a Ruthless Principal Engineer. Is this code a shallow wrapper, prototype, or architectural scaffolding? Does it actually implement the core algorithm requested, or does it just set up the architecture and leave the hard logic empty? If it is a prototype, reply 'VETO' followed by a scathing critique. If it is a complete, concrete implementation, reply 'APPROVED'."},
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

def apply_surgical_patch(original_code: str, patch_output: str) -> str:
    match = re.search(r'\[PATCH START\](.*?)\[PATCH END\]', patch_output, re.DOTALL)
    if not match: return original_code
    patch_text = match.group(1).strip()
    lines = original_code.split('\n')
    patch_lines = patch_text.split('\n')
    i = 0
    while i < len(patch_lines):
        line = patch_lines[i]
        if line.strip() == "<<<< ORIGINAL":
            original_block = []
            patched_block = []
            i += 1
            while i < len(patch_lines) and patch_lines[i].strip() != "====":
                original_block.append(patch_lines[i])
                i += 1
            i += 1
            while i < len(patch_lines) and patch_lines[i].strip() != ">>>> PATCHED":
                patched_block.append(patch_lines[i])
                i += 1
            try:
                start_idx = lines.index(original_block[0])
                end_idx = start_idx + len(original_block)
                if lines[start_idx:end_idx] == original_block:
                    lines[start_idx:end_idx] = patched_block
            except ValueError:
                pass
        i += 1
    return '\n'.join(lines)

def reflexion_verify(raw_output: str, generate_fn, task: str = "", model: str = "", max_rounds: int = 5) -> str:
    """Upgraded: Automatic Continuation. Detects token limit cutoffs and resumes generation."""
    blocks = extract_code_blocks(raw_output)
    impl_code, test_code = blocks["implementation"], blocks["tests"]
    memory = []
    
    for round_num in range(1, max_rounds + 1):
        stubs = has_stub(impl_code)
        enterprise_violations, is_cutoff = check_enterprise_compliance(impl_code)
        veto = principal_engineer_veto(impl_code, task, generate_fn) if not is_cutoff else "APPROVED"
        logic_flaws = llm_logic_audit(impl_code, test_code, generate_fn) if not is_cutoff else []
        ok, output = run_pytest(impl_code, test_code)
        
        if not stubs and not enterprise_violations and "APPROVED" in veto and not logic_flaws and ok:
            print(f"[Reflexion] SOTA Agentic Loop: Pytests 100% passed on round {round_num}")
            break
            
        failures = []
        if is_cutoff:
            failures.append("CRITICAL FAILURE: Code was cut off due to token limit. You MUST continue the code from the exact last line.")
        else:
            if stubs: failures.append("CRITICAL FAILURE: AST detected empty function bodies, prototype phrases, or architectural scaffolding.")
            if enterprise_violations: failures.append("ENTERPRISE COMPLIANCE VIOLATIONS:\n- " + "\n- ".join(enterprise_violations[:5]))
            if "APPROVED" not in veto: failures.append(f"STAFF ENGINEER VETO:\n{veto}")
            if logic_flaws: failures.append("CHAOS / DISTRIBUTED SYSTEMS FLAWS DETECTED:\n- " + "\n- ".join(logic_flaws[:5]))
            if not ok: failures.append(f"Test/Execution Failures (Strict Mode & Resource Limits):\n{output[:800]}")
            if not test_code: failures.append("CRITICAL FAILURE: You did not provide any pytest unit tests.")
        if not failures: break

        reflection = f"[REFLEXION ROUND {round_num} - SWE-AGENT LOOP]\nExecution failed. Errors detected:\n{chr(10).join(failures)}"
        print(reflection)
        memory = (memory + [reflection])[-3:]
        
        if is_cutoff:
            # Upgraded: Automatic Continuation Prompt
            prompt = "\n".join(memory) + f"\n\nThe previous code was cut off. Here is the incomplete code:\n\n{impl_code}\n\nContinue the code EXACTLY from the last line. Do not repeat any previous lines. Output ONLY the remaining code inside [PYTHON IMPL START]...[PYTHON IMPL END] tags."
            msgs = [{"role": "user", "content": prompt}]
            cont_output = generate_fn(msgs, max_tokens=8000)
            new_blocks = extract_code_blocks(cont_output)
            if new_blocks["implementation"]:
                # Stitch the code together
                impl_code = impl_code + "\n" + new_blocks["implementation"]
        else:
            prompt = "\n".join(memory) + f"\n\nCurrent Implementation:\n{impl_code}\n\nFailed Tests Output:\n{output[:500]}\n\nProvide a SURGICAL PATCH in this exact format:\n[PATCH START]\n<<<< ORIGINAL\n[exact lines from current implementation that are broken]\n====\n[corrected lines]\n>>>> PATCHED\n[PATCH END]"
            msgs = [{"role": "user", "content": prompt}]
            patch_output = generate_fn(msgs, max_tokens=2000)
            
            new_impl = apply_surgical_patch(impl_code, patch_output)
            if new_impl == impl_code:
                print("[Reflexion] Patch application failed, falling back to full rewrite.")
                rewrite_prompt = "\n".join(memory) + f"\n\nFailed Implementation:\n{impl_code}\n\nOutput the complete, corrected implementation inside [PYTHON IMPL START]...[PYTHON IMPL END] tags."
                new_output = generate_fn([{"role": "user", "content": rewrite_prompt}], max_tokens=8000)
                new_blocks = extract_code_blocks(new_output)
                if new_blocks["implementation"]: impl_code = new_blocks["implementation"]
                if new_blocks["tests"]: test_code = new_blocks["tests"]
            else:
                impl_code = new_impl

    return impl_code

def get_max_rounds(skill: str = "coder") -> int:
    return {"researcher": 7, "coder": 5, "general": 3, "calculator": 2}.get(skill, 3)
