import os, ast, threading, time, logging
log = logging.getLogger(__name__)

def _scan_codebase_for_debt():
    """Scans .py files for functions over 100 lines or missing type hints."""
    debt = []
    for root, _, files in os.walk('.'):
        if any(ex in root for ex in ['.git', '__pycache__', 'venv', 'node_modules']): continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        source = f.read()
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check for missing type hints on args
                            has_hints = all(arg.annotation for arg in node.args.args if arg.arg != 'self')
                            if not has_hints and not node.name.startswith('test_'):
                                debt.append({"file": filepath, "func": node.name, "line": node.lineno, "issue": "Missing type hints"})
                            # Check for overly long functions
                            length = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                            if length > 100:
                                debt.append({"file": filepath, "func": node.name, "line": node.lineno, "issue": f"Function too long ({length} lines)"})
                except:
                    pass
    return debt[:5] # Top 5 issues

def _refactor_loop(generate_fn):
    time.sleep(120) # Wait 2 mins after startup
    log.info("[RefactorDaemon] Starting background codebase scanner...")
    while True:
        try:
            debt = _scan_codebase_for_debt()
            if debt:
                from context_compressor import log_subconscious_action
                log_subconscious_action('RefactorDaemon', f'Found {len(debt)} technical debt items and generated suggestions.')
                prompt = [
                    {"role": "system", "content": "You are an Autonomous Refactoring Engine. Output a JSON array of objects with 'file', 'func', and 'suggestion' keys describing how to fix the technical debt."},
                    {"role": "user", "content": f"Technical Debt Found:\n{str(debt)}"}
                ]
                suggestions = generate_fn(prompt, max_tokens=500)
                os.makedirs("refactor_suggestions", exist_ok=True)
                with open(f"refactor_suggestions/suggestions_{int(time.time())}.md", "w") as f:
                    f.write(f"# Autonomous Refactoring Suggestions\n\n```\n{suggestions}\n```")
        except Exception as e:
            log.error(f"[RefactorDaemon] Error: {e}")
        time.sleep(1800) # Run every 30 minutes

def start_refactor_daemon(generate_fn):
    t = threading.Thread(target=_refactor_loop, args=(generate_fn,), daemon=True, name="refactor_daemon")
    t.start()
    print("[Startup] ✓ Autonomous Refactoring Daemon started.")
