import ast, re, json

def predict_error_lines(code: str, rlef_context: str) -> list:
    """Upgraded: Uses RLEF history to predict which lines are most likely to fail."""
    suspicious_lines = []
    lines = code.split('\n')
    
    # Heuristics for suspicious code
    for i, line in enumerate(lines):
        # Bare excepts are highly suspicious
        if re.search(r'except\s*:', line): suspicious_lines.append(i+1)
        # Mutable defaults
        if re.search(r'def\s+\w+\(.*=\s*[\[\{]', line): suspicious_lines.append(i+1)
        # Missing timeouts on network calls
        if re.search(r'requests\.(get|post|put|delete)\(', line) and 'timeout' not in line: suspicious_lines.append(i+1)
        # TODOs or Pass
        if 'TODO' in line or 'pass' in line: suspicious_lines.append(i+1)
        
    # If RLEF context mentions specific errors (e.g., KeyError), look for dict access without .get()
    if "KeyError" in rlef_context:
        for i, line in enumerate(lines):
            if re.search(r'\w+\[\"[^\"]+\"\]', line) and '.get(' not in line:
                suspicious_lines.append(i+1)
                
    return sorted(list(set(suspicious_lines)))

def apply_ast_mutation(code: str, mutation_directive: str) -> str:
    """Upgraded: Parses AST mutation directives and applies them surgically."""
    try:
        # Directive format: {"line": 45, "action": "wrap_try_except", "exception": "KeyError"}
        directive = json.loads(mutation_directive)
        target_line = directive.get("line")
        action = directive.get("action")
        
        tree = ast.parse(code)
        lines = code.split('\n')
        
        # Find the node at the target line
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == target_line:
                if action == "wrap_try_except":
                    # Wrap the statement in a try/except block
                    indent = " " * (node.col_offset)
                    exc = directive.get("exception", "Exception")
                    new_lines = [
                        f"{indent}try:",
                        f"{indent}    {lines[target_line-1].strip()}",
                        f"{indent}except {exc} as e:",
                        f"{indent}    logging.error(f'Handled error: {e}')"
                    ]
                    lines[target_line-1] = "\n".join(new_lines)
                    return "\n".join(lines)
                    
                elif action == "replace":
                    lines[target_line-1] = directive.get("new_code", lines[target_line-1])
                    return "\n".join(lines)
                    
    except Exception:
        pass
        
    return code
