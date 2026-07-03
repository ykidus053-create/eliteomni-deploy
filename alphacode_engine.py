
import re, hashlib, subprocess, tempfile, os, sys
from collections import defaultdict

# ─── AlphaCode Engine ────────────────────────────────────────────────────────
# Exact DeepMind technique:
# 1. Generate N diverse code samples (large-scale sampling)
# 2. Filter by execution against test cases (~99% eliminated)
# 3. Cluster remaining by behavior (output signature)
# 4. Return best candidate from each cluster (top 10)

def _extract_code(text):
    """Pull code block from LLM response."""
    m = re.search(r"```(?:python|py)?\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: lines that look like code
    lines = [l for l in text.splitlines() if l.startswith(("def ","class ","import ","for ","while ","if ","    "))]
    return "\n".join(lines) if lines else text.strip()

def _run_code(code, test_input="", timeout=5):
    """Execute code safely, return (stdout, stderr, passed)."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            fname = f.name
        result = subprocess.run(
            [sys.executable, fname],
            input=test_input, capture_output=True, text=True, timeout=timeout
        )
        os.unlink(fname)
        return result.stdout.strip(), result.stderr.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", False
    except Exception as e:
        return "", str(e), False

def _behavior_signature(code, test_cases):
    """
    Run code against all test cases, hash the outputs.
    Two programs with identical signatures behave the same -> same cluster.
    """
    outputs = []
    for inp, _ in test_cases:
        stdout, _, passed = _run_code(code, inp)
        outputs.append(stdout[:100] if passed else "FAIL")
    sig = hashlib.md5("|".join(outputs).encode()).hexdigest()
    return sig, outputs

def _extract_test_cases(problem_text):
    """
    Parse example input/output pairs from problem description.
    Looks for patterns like:
        Input: ...
        Output: ...
    """
    test_cases = []
    blocks = re.findall(
        r"(?:Input|input)[:\s]+([^\n]+(?:\n(?!Output|output)[^\n]+)*)\s*"
        r"(?:Output|output)[:\s]+([^\n]+(?:\n(?!Input|input)[^\n]+)*)",
        problem_text
    )
    for inp, out in blocks:
        test_cases.append((inp.strip(), out.strip()))
    return test_cases[:5]  # max 5 test cases

def _score_candidate(code, test_cases):
    """Score 0-1 based on how many test cases pass."""
    if not test_cases:
        return 0.5  # no tests = neutral
    passed = 0
    for inp, expected in test_cases:
        stdout, _, ok = _run_code(code, inp)
        if ok and stdout.strip() == expected.strip():
            passed += 1
    return passed / len(test_cases)

def alphacode_select(samples, problem_text="", n_best=3):
    """
    AlphaCode pipeline:
    samples      : list of raw LLM response strings
    problem_text : original problem (for test case extraction)
    n_best       : max candidates to return

    Returns list of (code, score, cluster_id) sorted best-first.
    """
    test_cases = _extract_test_cases(problem_text)
    candidates = []

    # Step 1: Extract code from each sample
    for raw in samples:
        code = _extract_code(raw)
        if len(code) < 10:
            continue
        candidates.append(code)

    if not candidates:
        return []

    # Step 2: Filter — run against test cases, keep passing ones
    if test_cases:
        scored = []
        for code in candidates:
            score = _score_candidate(code, test_cases)
            if score > 0:  # filter: must pass at least 1 test
                scored.append((code, score))
        # If nothing passes, keep all (fallback)
        if not scored:
            scored = [(c, 0.0) for c in candidates]
    else:
        scored = [(c, 0.5) for c in candidates]

    # Step 3: Cluster by behavior signature
    clusters = defaultdict(list)
    for code, score in scored:
        if test_cases:
            sig, _ = _behavior_signature(code, test_cases)
        else:
            # Cluster by structural similarity (line count + keywords)
            key = str(len(code)//50) + str(sorted(re.findall(r"\b(def|for|while|if|return|class)\b", code)))
            sig = hashlib.md5(key.encode()).hexdigest()
        clusters[sig].append((code, score))

    # Step 4: Pick best from each cluster, rank clusters by best score
    best_per_cluster = []
    for sig, members in clusters.items():
        best = max(members, key=lambda x: x[1])
        best_per_cluster.append((best[0], best[1], sig[:8]))

    best_per_cluster.sort(key=lambda x: x[1], reverse=True)
    return best_per_cluster[:n_best]

def alphacode_best(samples, problem_text=""):
    """Return single best code solution."""
    results = alphacode_select(samples, problem_text, n_best=1)
    if results:
        return results[0][0]
    codes = [_extract_code(s) for s in samples if s]
    return codes[0] if codes else ""

def alphacode_report(results):
    """Human-readable summary of selection."""
    if not results:
        return ""
    lines = [f"AlphaCode selected {len(results)} candidate(s) from cluster analysis:"]
    for i, (code, score, cid) in enumerate(results):
        lines.append(f"  Candidate {i+1} | cluster={cid} | test_pass={score:.0%} | {len(code.splitlines())} lines")
    return "\n".join(lines)
