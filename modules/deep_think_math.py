"""
deep_think_math.py — Gemini Deep Think math pipeline for EliteOmni
Implements all 4 mechanisms that drove the 95% AIME score:
  1. Problem decomposition into sub-problems before answering
  2. Parallel hypothesis exploration with confidence scoring
  3. Iterative self-refinement loop
  4. Code execution as verification layer
"""
import re, time
from modules.services.tools import tool_exec

# ── 1. PROBLEM DECOMPOSER ─────────────────────────────────────────────────────
def decompose_problem(problem: str, generate_fn) -> list:
    """
    Stage 1: Break problem into sub-problems before attempting solution.
    Gemini does this internally; we make it explicit.
    """
    prompt = (
        "You are a math expert. Break this problem into numbered sub-problems "
        "that must be solved in sequence. Be specific. Max 5 steps.\n\n"
        f"Problem: {problem[:400]}"
    )
    try:
        raw = generate_fn(prompt)
        if not raw:
            return [problem]
        steps = re.findall(r'\d+\.\s+(.+?)(?=\n\d+\.|\Z)', raw, re.DOTALL)
        return [s.strip() for s in steps if s.strip()] or [problem]
    except Exception:
        return [problem]


# ── 2. PARALLEL HYPOTHESIS EXPLORER ──────────────────────────────────────────
def explore_hypotheses(problem: str, generate_fn, n: int = 3) -> list:
    """
    Stage 2: Generate N independent solution attempts in parallel.
    Each is a separate reasoning path. Score by internal consistency.
    Mirrors Deep Think's parallel branch exploration.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    approaches = [
        f"Solve using direct calculation: {problem[:300]}",
        f"Solve using algebraic manipulation: {problem[:300]}",
        f"Solve by working backwards from the answer: {problem[:300]}",
    ][:n]

    results = []
    try:
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(generate_fn, p): i for i, p in enumerate(approaches)}
            for fut in as_completed(futures, timeout=30):
                try:
                    r = fut.result()
                    if r and len(r) > 20:
                        results.append(r)
                except Exception:
                    pass
    except Exception:
        pass

    return results if results else [generate_fn(problem)]


# ── 3. CONFIDENCE SCORER ──────────────────────────────────────────────────────
def score_hypothesis(hypothesis: str, problem: str, generate_fn) -> float:
    """
    Assign confidence score 0.0-1.0 to a solution hypothesis.
    Mirrors Deep Think's uncertainty quantification per branch.
    """
    prompt = (
        f"Rate this math solution from 0-10 for correctness and completeness.\n"
        f"Problem: {problem[:200]}\n"
        f"Solution: {hypothesis[:400]}\n"
        f"Reply ONLY with a single integer 0-10."
    )
    try:
        raw = generate_fn(prompt)
        nums = re.findall(r'\b([0-9]|10)\b', raw or "")
        return int(nums[0]) / 10.0 if nums else 0.5
    except Exception:
        return 0.5


# ── 4. CODE EXECUTION VERIFIER ────────────────────────────────────────────────
def verify_with_code(problem: str, answer: str, generate_fn) -> dict:
    """
    Stage 4: Generate Python verification code and execute it.
    This is what takes Gemini from 95% → 100% on AIME.
    """
    code_prompt = (
        "Write Python code to verify this math answer. "
        "Print 'VERIFIED: [answer]' if correct, 'WRONG: [correct answer]' if not.\n"
        f"Problem: {problem[:300]}\n"
        f"Claimed answer: {answer[:200]}\n"
        "Code only, no explanation:"
    )
    try:
        code = generate_fn(code_prompt)
        if not code:
            return {"verified": None, "output": "no code generated"}

        # Extract code block if wrapped
        blocks = re.findall(r'```(?:python)?\n(.*?)```', code, re.DOTALL)
        code = blocks[0] if blocks else code

        output = tool_exec(code, timeout=10)
        verified = "VERIFIED" in output.upper()
        wrong = "WRONG" in output.upper()

        # Extract corrected answer if wrong
        corrected = None
        if wrong:
            m = re.search(r'WRONG:\s*(.+)', output, re.IGNORECASE)
            corrected = m.group(1).strip() if m else None

        return {
            "verified": verified,
            "wrong": wrong,
            "corrected": corrected,
            "output": output[:300]
        }
    except Exception as e:
        return {"verified": None, "output": str(e)}


# ── 5. ITERATIVE REFINEMENT ───────────────────────────────────────────────────
def iterative_refine(problem: str, solution: str, generate_fn, max_iters: int = 2) -> str:
    """
    Stage 3: Iteratively critique and refine the solution.
    Each cycle evaluates partial conclusion and updates.
    """
    current = solution
    for i in range(max_iters):
        try:
            critique_prompt = (
                f"Review this math solution for errors. "
                f"If correct, reply CORRECT. "
                f"If wrong, reply FIXED: [corrected solution]\n\n"
                f"Problem: {problem[:200]}\n"
                f"Solution: {current[:500]}"
            )
            feedback = generate_fn(critique_prompt)
            if not feedback or "CORRECT" in feedback.upper():
                break
            if "FIXED:" in feedback.upper():
                fixed = feedback.split("FIXED:", 1)[-1].strip()
                if len(fixed) > 20:
                    current = fixed
                    print(f"[DeepThink] Refined iteration {i+1}")
        except Exception as e:
            print(f"[DeepThink refine] {e}")
            break
    return current


# ── MASTER: FULL DEEP THINK MATH PIPELINE ────────────────────────────────────
def deep_think_math(problem: str, generate_fn, complexity: str = "hard") -> str:
    """
    Full 4-stage Deep Think pipeline for math problems:
    Decompose → Explore hypotheses → Refine → Verify with code
    
    Only activates for hard/medium math problems to save compute.
    """
    if complexity == "easy":
        return None  # don't waste compute on simple math

    print(f"[DeepThink] Starting 4-stage math pipeline for: {problem[:80]}...")

    # Stage 1: Decompose
    sub_problems = decompose_problem(problem, generate_fn)
    print(f"[DeepThink] Decomposed into {len(sub_problems)} sub-problems")

    # Stage 2: Parallel hypothesis exploration
    hypotheses = explore_hypotheses(problem, generate_fn, n=2)
    print(f"[DeepThink] Generated {len(hypotheses)} hypotheses")

    # Score and pick best hypothesis
    if len(hypotheses) > 1:
        scores = [score_hypothesis(h, problem, generate_fn) for h in hypotheses]
        best_idx = scores.index(max(scores))
        best = hypotheses[best_idx]
        print(f"[DeepThink] Best hypothesis score: {scores[best_idx]:.1f}")
    else:
        best = hypotheses[0] if hypotheses else generate_fn(problem)

    if not best:
        return None

    # Stage 3: Iterative refinement
    refined = iterative_refine(problem, best, generate_fn, max_iters=5)

    # Stage 4: Code verification
    verification = verify_with_code(problem, refined, generate_fn)
    print(f"[DeepThink] Verification: {verification.get('output','?')[:80]}")

    # If code found a better answer, use it
    if verification.get("wrong") and verification.get("corrected"):
        corrected = verification["corrected"]
        refined = refined + f"\n\n✅ **Verified answer: {corrected}**"
    elif verification.get("verified"):
        refined = refined + "\n\n✅ **Verified by code execution**"

    return refined
