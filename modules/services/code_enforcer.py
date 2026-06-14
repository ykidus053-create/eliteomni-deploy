"""
code_enforcer.py — Deep production-code enforcement layer.
Intercepts coder responses, validates code blocks, forces rewrite if stubs detected.
Works at AST level — not just string matching.
"""
import re, ast, textwrap
from typing import Tuple

# ── Stub detection via AST ────────────────────────────────────────────────────
def _ast_has_stubs(code: str) -> list[str]:
    """Parse code and find stub patterns at AST level."""
    issues = []
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return []  # Can't parse = not Python, skip

    for node in ast.walk(tree):
        # Functions with only Pass or just a docstring
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            non_doc = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if not non_doc:
                issues.append(f"empty function: {node.name}()")
            elif len(non_doc) == 1 and isinstance(non_doc[0], ast.Pass):
                issues.append(f"stub function (pass only): {node.name}()")
            elif len(non_doc) == 1 and isinstance(non_doc[0], ast.Raise):
                # raise NotImplementedError
                exc = non_doc[0]
                if isinstance(exc.exc, ast.Call) and hasattr(exc.exc.func, 'id'):
                    if exc.exc.func.id == 'NotImplementedError':
                        issues.append(f"stub function (NotImplementedError): {node.name}()")

        # Return None stubs
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is None:
            pass  # None returns are valid in many cases

    return issues

# ── String-level pseudocode signals ──────────────────────────────────────────
_REGEX_SIGNALS = [
    (re.compile(r'#\s*(TODO|FIXME|HACK|XXX|IMPLEMENT|ADD LOGIC|YOUR CODE)', re.I), "TODO/FIXME comment"),
    (re.compile(r'#\s*In (a |the )?real (impl|implementation|system|world)', re.I), "'In real impl' comment"),
    (re.compile(r'#\s*For (a |the )?(real|production|actual)', re.I), "production disclaimer comment"),
    (re.compile(r'(fake|mock|stub|dummy|placeholder)_(db|client|server|api|data|conn|connection)', re.I), "fake/mock variable"),
    (re.compile(r'["\'](your_api_key|your_password|your_db_url|your_token|YOUR_KEY)["\']', re.I), "hardcoded placeholder credential"),
    (re.compile(r'pass\s*#', re.I), "pass with comment (stub)"),
    (re.compile(r'\.\.\.\s*#', re.I), "ellipsis stub"),
    (re.compile(r'print\(f?"(Forwarding|Simulating|Would send|Placeholder)', re.I), "simulation print statement"),
    (re.compile(r'#\s*(simplified|this is a demo|example only|not production)', re.I), "non-production disclaimer"),
]

def detect_pseudocode(response: str) -> list[str]:
    """Return list of violations found in response. Empty = clean."""
    violations = []

    # Extract code blocks
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
    all_code = '\n'.join(code_blocks) if code_blocks else response

    # String-level checks
    for pattern, label in _REGEX_SIGNALS:
        if pattern.search(all_code):
            violations.append(label)

    # AST-level checks on each block
    for block in code_blocks:
        ast_issues = _ast_has_stubs(block)
        violations.extend(ast_issues)

    return list(set(violations))

# ── Rewrite prompt builder ────────────────────────────────────────────────────
def build_rewrite_prompt(original_request: str, violations: list[str]) -> str:
    return (
        f"CRITICAL REWRITE REQUIRED.\n\n"
        f"Your previous response had these production code violations:\n"
        + "\n".join(f"  ❌ {v}" for v in violations) +
        f"\n\nOriginal request: {original_request}\n\n"
        "RULES FOR THIS REWRITE (ZERO EXCEPTIONS):\n"
        "1. Every function must have a COMPLETE real implementation\n"
        "2. No pass, no ..., no raise NotImplementedError()\n"
        "3. No fake_*, mock_*, stub_* variables or functions\n"
        "4. No '# In real impl' or '# TODO' comments\n"
        "5. Use real pip-installable libraries (httpx, sqlalchemy, redis, grpc, etc.)\n"
        "6. Every network call must use real HTTP/gRPC/socket code\n"
        "7. Config from os.environ — never hardcoded strings\n"
        "8. Code must run with: pip install -r requirements.txt && python main.py\n"
        "9. Include a requirements.txt block at the end\n\n"
        "OUTPUT FORMAT: Only code. No prose explanations. No 'considerations' section.\n"
        "Start your response with ```python immediately.\n"
    )

# ── Main enforcement entry point ──────────────────────────────────────────────
def enforce_production_code(response: str, request: str) -> Tuple[bool, list[str], str]:
    """
    Check response for pseudocode violations.
    Returns (is_clean, violations, rewrite_prompt_if_needed)
    """
    violations = detect_pseudocode(response)
    if not violations:
        return True, [], ""
    rewrite = build_rewrite_prompt(request, violations)
    return False, violations, rewrite
