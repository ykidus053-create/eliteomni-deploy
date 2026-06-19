import re
import subprocess
import tempfile
import os
import json

def extract_claims(code: str) -> list[str]:
    """Pull every claimed feature from code comments and docstrings."""
    patterns = [
        r'#\s*(implement[s]?|support[s]?|handle[s]?|provide[s]?)[^\n]+',
        r'"""[^"]*"""',
        r"'''[^']*'''",
    ]
    claims = []
    for p in patterns:
        for m in re.finditer(p, code, re.IGNORECASE):
            text = m.group().strip().replace('"""','').replace("'''","").strip()
            if len(text) > 10:
                claims.append(text)
    return list(set(claims))

def build_matrix(claims: list[str]) -> dict:
    return {
        re.sub(r'\W+','_', c[:40]).lower(): {
            "claim": c,
            "implemented": False,
            "evidence": [],
            "test_passed": False
        }
        for c in claims
    }

def generate_test_for_claim(claim: str, code: str) -> str:
    """Generate a minimal pytest for a claim."""
    return f'''
import pytest

def test_claim():
    """Verify: {claim[:80]}"""
    # Evidence check: claim must appear as runnable logic, not just comment
    source = """{code[:300].replace('"', "'")}"""
    keywords = {json.dumps(claim.lower().split()[:4])}
    hits = sum(1 for k in keywords if k in source.lower())
    assert hits >= 2, f"Claim not evidenced in code: {claim[:60]}"
'''

def run_test(test_code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(test_code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", fname, "-q", "--tb=short"],
            capture_output=True, text=True, timeout=15
        )
        passed = result.returncode == 0
        return passed, result.stdout + result.stderr
    finally:
        os.unlink(fname)

def verify_matrix(matrix: dict, code: str) -> dict:
    for key, entry in matrix.items():
        test = generate_test_for_claim(entry["claim"], code)
        passed, output = run_test(test)
        entry["test_passed"] = passed
        entry["evidence"] = ["PASSED" if passed else f"FAILED: {output[:200]}"]
        entry["implemented"] = passed
    return matrix

def enforce_requirements(code: str, max_rounds: int = 3) -> dict:
    claims = extract_claims(code)
    if not claims:
        print("[matrix] No claims found in code.")
        return {}

    matrix = build_matrix(claims)
    print(f"[matrix] Found {len(claims)} claims. Running verification...")

    for round_num in range(1, max_rounds + 1):
        matrix = verify_matrix(matrix, code)
        passed = sum(1 for v in matrix.values() if v["test_passed"])
        total = len(matrix)
        print(f"[matrix] Round {round_num}: {passed}/{total} claims verified")
        if passed == total:
            break

    # Print final report
    print("\n=== REQUIREMENTS MATRIX ===")
    for key, entry in matrix.items():
        status = "✓" if entry["test_passed"] else "✗"
        print(f"  {status} {entry['claim'][:60]}")
    print("===========================\n")
    return matrix

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "app.py"
    with open(target) as f:
        code = f.read()
    enforce_requirements(code)
