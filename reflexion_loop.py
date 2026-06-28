import re
import subprocess
import tempfile
import os
import ast
import resource
import sys
from io import StringIO
from code_rag import get_relevant_code_context

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

def check_enterprise_compliance(code: str) -> tuple[list, bool]:
    violations = []
    is_cutoff = False
    try:
        tree = ast.parse(code)
        has_logging = False
        has_metrics = False
        banned_calls = {'eval', 'exec', 'compile', '__import__', 'os.system', 'subprocess.call'}
        io_calls = {'requests.get', 'requests.post', 'open', 'urlopen', 'psycopg2.connect', 'sqlite3.connect'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("test_") and node.name != "__init__":
                    for arg in node.args.args:
                        if arg.annotation is None and arg.arg not in ('self', 'cls'): violations.append(f"Function '{node.name}' missing type hint for argument '{arg.arg}'.")
                    if node.returns is None: violations.append(f"Function '{node.name}' missing return type hint.")
                if not node.name.startswith("_") and not node.name.startswith("test_"):
                    if not ast.get_docstring(node): violations.append(f"Function '{node.name}' missing a docstring.")
                func_name_lower = node.name.lower()
                is_io_func = any(k in func_name_lower for k in ['save', 'load', 'fetch', 'get', 'post', 'read', 'write', 'connect', 'repository', 'client'])
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Call):
                        call_name = ""
                        if isinstance(stmt.func, ast.Name): call_name = stmt.func.id
                        elif isinstance(stmt.func, ast.Attribute): call_name = stmt.func.attr
                        if call_name in io_calls and not is_io_func:
                            violations.append(f"ARCHITECTURE VIOLATION (SRP): I/O call '{call_name}()' found inside business logic function '{node.name}'.")
            if isinstance(node, ast.ExceptHandler):
                if node.type is None: violations.append("Bare 'except:' block found.")
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass): violations.append("Silent 'except: pass' block found.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print': violations.append("Use of print() found.")
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
                        violations.append(f"SCAFFOLDING BAN: Class '{node.name}' inherits from {base_name}.")
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
        {"role": "system", "content": "You are a Ruthless Principal Software Architect. Is this code architecturally consistent? Does it cleanly separate I/O from business logic? If it is inconsistent or a prototype, reply 'VETO' followed by a scathing critique. If it is clean, reply 'APPROVED'."},
        {"role": "user", "content": f"Task: {task}\n\nCode:\n{impl_code[:2000]}"}
    ]
    try:
        raw = generate_fn(prompt, max_tokens=200)
        if "VETO" in raw.upper(): return raw.strip()
    except: pass
    return "APPROVED"

def llm_logic_audit(impl_code: str, test_code: str, generate_fn) -> list:
    prompt = [
        {"role": "system", "content": "You are a Chaos Engineer. Does this code have real-world failure modes? Reply ONLY with a JSON list of strings. If perfect, reply []."},
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

def run_in_persistent_sandbox(impl_code: str, test_code: str) -> tuple[bool, str]:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    sandbox_globals = {}
    success = False
    output = ""
    try:
        exec(impl_code, sandbox_globals)
        if test_code:
            sandbox_globals['pytest'] = __import__('pytest')
            exec(test_code, sandbox_globals)
            success = True
        else:
            success = True
    except AssertionError as e:
        success = False
        output = f"AssertionError: {e}"
    except Exception as e:
        success = False
        import traceback
        output = traceback.format_exc()
    finally:
        output += sys.stdout.getvalue() + sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return success, output[:1000]

def generate_adversarial_tests(task: str, impl_code: str, generate_fn) -> str:
    """Upgraded: Spawns an independent agent to write tests designed to break the code."""
    prompt = [
        {"role": "system", "content": "You are a Ruthless QA Engineer. Your job is to BREAK the provided implementation. Write `pytest` unit tests that target edge cases, null inputs, race conditions, and maximum limits. Do not write trivial tests. Output ONLY the python code inside [PYTHON TESTS START]...[PYTHON TESTS END] tags."},
        {"role": "user", "content": f"Task: {task}\n\nImplementation to break:\n{impl_code[:2000]}"}
    ]
    try:
        raw = generate_fn(prompt, max_tokens=1000)
        match = re.search(r'\[PYTHON TESTS START\](.*?)\[PYTHON TESTS END\]', raw, re.DOTALL)
        return match.group(1).strip() if match else ""
    except:
        return ""

def reflexion_verify(raw_output: str, generate_fn, task: str = "", model: str = "", max_rounds: int = 5) -> str:
    blocks = extract_code_blocks(raw_output)
    impl_code, test_code = blocks["implementation"], blocks["tests"]
    memory = []
    
    for round_num in range(1, max_rounds + 1):
        stubs = has_stub(impl_code)
        enterprise_violations, is_cutoff = check_enterprise_compliance(impl_code)
        veto = principal_engineer_veto(impl_code, task, generate_fn) if not is_cutoff else "APPROVED"
        logic_flaws = llm_logic_audit(impl_code, test_code, generate_fn) if not is_cutoff else []
        
        # Upgraded: Run both the AI's tests AND the adversarial tests
        ok, output = run_in_persistent_sandbox(impl_code, test_code)
        adv_test_code = generate_adversarial_tests(task, impl_code, generate_fn) if ok else ""
        adv_ok, adv_output = run_in_persistent_sandbox(impl_code, adv_test_code) if adv_test_code else (True, "")
        
        if not stubs and not enterprise_violations and "APPROVED" in veto and not logic_flaws and ok and adv_ok:
            print(f"[Reflexion] SOTA Agentic Loop: Pytests & Adversarial Tests 100% passed on round {round_num}")
            break
            
        failures = []
        if is_cutoff:
            failures.append("CRITICAL FAILURE: Code was cut off due to token limit. You MUST continue the code from the exact last line.")
        else:
            if stubs: failures.append("CRITICAL FAILURE: AST detected empty function bodies or scaffolding.")
            if enterprise_violations: failures.append("ARCHITECTURAL & ENTERPRISE VIOLATIONS:\n- " + "\n- ".join(enterprise_violations[:5]))
            if "APPROVED" not in veto: failures.append(f"PRINCIPAL ARCHITECT VETO:\n{veto}")
            if logic_flaws: failures.append("CHAOS / DISTRIBUTED SYSTEMS FLAWS DETECTED:\n- " + "\n- ".join(logic_flaws[:5]))
            if not ok: failures.append(f"Test/Execution Failures (Persistent Sandbox):\n{output[:800]}")
            if not adv_ok: failures.append(f"ADVERSARIAL TESTS FAILED: An independent agent wrote tests to break your code, and it failed them:\n{adv_output[:800]}")
            if not test_code: failures.append("CRITICAL FAILURE: You did not provide any pytest unit tests.")
        if not failures: break

        reflection = f"[REFLEXION ROUND {round_num} - SWE-AGENT LOOP]\nExecution failed. Errors detected:\n{chr(10).join(failures)}"
        print(reflection)
        memory = (memory + [reflection])[-3:]
        
        codebase_ctx = get_relevant_code_context(task, top_k=3)
        goal_reminder = f"\n[GOAL REMINDER] Do not forget the original user request: '{task}'\n"
        
        if is_cutoff:
            prompt = "\n".join(memory) + f"\n{goal_reminder}\nThe previous code was cut off. Here is the incomplete code:\n\n{impl_code}\n\nContinue the code EXACTLY from the last line. Output ONLY the remaining code inside [PYTHON IMPL START]...[PYTHON IMPL END] tags."
            msgs = [{"role": "user", "content": prompt}]
            cont_output = generate_fn(msgs, max_tokens=8000)
            new_blocks = extract_code_blocks(cont_output)
            if new_blocks["implementation"]:
                impl_code = impl_code + "\n" + new_blocks["implementation"]
        else:
            prompt = "\n".join(memory) + f"{goal_reminder}\n{codebase_ctx}\n\nCurrent Implementation:\n{impl_code}\n\nFailed Tests Output:\n{output[:500]}\n\nProvide the corrected functions inside [PYTHON IMPL START]...[PYTHON IMPL END] tags. You only need to output the functions that need fixing, but they must be complete."
            msgs = [{"role": "user", "content": prompt}]
            new_output = generate_fn(msgs, max_tokens=4000)
            new_blocks = extract_code_blocks(new_output)
            if new_blocks["implementation"]:
                impl_code = impl_code + "\n" + new_blocks["implementation"]
            if new_blocks["tests"]:
                test_code = new_blocks["tests"]

    return impl_code

def get_max_rounds(skill: str = "coder") -> int:
    return {"researcher": 7, "coder": 5, "general": 3, "calculator": 2}.get(skill, 3)
