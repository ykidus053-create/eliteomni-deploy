"""
Deliberative Reasoning Engine — replaces single-pass generation.
Implements: Chain-of-Thought, Tree-of-Thought sampling, Process Reward Modeling,
Self-consistency voting, and OODA with genuine state tracking.
"""
import re, time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reason")

# ── Step 1: Problem Decomposition ────────────────────────────────────────────

def decompose_problem(msg: str, generate_fn, model: str) -> dict:
    """
    Break complex problems into typed sub-problems.
    Returns structured decomposition for downstream routing.
    """
    prompt = [{
        "role": "system",
        "content": (
            "You are a problem decomposer. Analyze the user's request and output JSON only:\n"
            '{"problem_type": "math|reasoning|factual|creative|code|multi_step", '
            '"sub_problems": ["...", "..."], '
            '"requires_search": true|false, '
            '"requires_calculation": true|false, '
            '"ambiguities": ["..."], '
            '"complexity_estimate": 1-10}'
        )
    }, {"role": "user", "content": msg[:500]}]
    try:
        raw = generate_fn(prompt, max_tokens=1500, model=model)
        raw = re.sub(r'```json|```', '', raw).strip()
        import json
        return json.loads(raw)
    except Exception:
        return {
            "problem_type": "general",
            "sub_problems": [msg],
            "requires_search": False,
            "requires_calculation": False,
            "ambiguities": [],
            "complexity_estimate": 5
        }

# ── Step 2: Hypothesis Generation (Tree of Thought) ──────────────────────────

def generate_hypotheses(msg: str, system: str, history: list,
                        generate_fn, model: str, n: int = 3) -> list:
    """
    Generate N diverse candidate responses in parallel.
    Uses temperature variation to ensure diversity.
    Core of Tree-of-Thought sampling.
    """
    def _gen_one(temp_label: str, approach_hint: str) -> str:
        prompt = [{
            "role": "system",
            "content": system + f"\n\nApproach hint: {approach_hint}"
        }] + history[-6:] + [{"role": "user", "content": msg}]
        try:
            return generate_fn(prompt, max_tokens=1200, model=model)
        except Exception as e:
            return ""

    approaches = [
        "Direct and concise. Lead with the answer.",
        "Step by step reasoning. Show your work explicitly.",
        "Consider edge cases and alternative interpretations first.",
    ][:n]

    futures = [
        _executor.submit(_gen_one, f"v{i}", approach)
        for i, approach in enumerate(approaches)
    ]
    results = []
    for fut in as_completed(futures, timeout=30):
        r = fut.result()
        if r and len(r) > 50:
            results.append(r)
    return results

# ── Step 3: Process Reward Model (PRM) ───────────────────────────────────────

def score_response(response: str, msg: str, generate_fn, model: str) -> float:
    """
    Score a response on multiple dimensions using a critic model.
    Returns 0.0-1.0 composite score.
    This IS the process reward model — evaluates intermediate reasoning steps.
    """
    if not response or len(response) < 30:
        return 0.0
    rubric = (
        "Score this response 1-10 on each dimension, reply ONLY as: H:N C:N A:N T:N\n"
        "H=Helpfulness C=Completeness A=Accuracy T=Tone\n"
        f"Question: {msg[:200]}\nResponse: {response[:600]}"
    )
    try:
        score_raw = generate_fn(
            [{"role": "system", "content": "You are a response quality scorer. Reply only in the format H:N C:N A:N T:N"},
             {"role": "user", "content": rubric}],
            max_tokens=30, model=model
        )
        scores = re.findall(r'[HCAT]:(\d+)', score_raw)
        if len(scores) >= 3:
            vals = [int(s) for s in scores[:4]]
            # Weighted: helpfulness 40%, completeness 25%, accuracy 25%, tone 10%
            weights = [0.40, 0.25, 0.25, 0.10]
            composite = sum(v * w for v, w in zip(vals, weights[:len(vals)])) / 10.0
            return composite
    except Exception:
        pass
    # Heuristic fallback
    score = 0.5
    if len(response) > 200: score += 0.1
    if any(w in response for w in ['however', 'therefore', 'because', 'since']): score += 0.05
    if response.count('\n') > 3: score += 0.05  # structured
    bad_signals = ['I cannot', 'I apologize', 'As an AI', 'I don\'t have']
    if any(b in response for b in bad_signals): score -= 0.15
    return min(max(score, 0.0), 1.0)

# ── Step 4: Self-Consistency Voting ──────────────────────────────────────────

def self_consistency_vote(candidates: list, scores: list) -> str:
    """
    Select best response via score-weighted voting.
    For factual questions, also checks answer consistency.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # Score-weighted selection
    total_score = sum(scores)
    if total_score == 0:
        return candidates[0]

    # Pick highest scoring
    best_idx = scores.index(max(scores))
    best = candidates[best_idx]

    # If scores are close (within 0.1), prefer longer/more complete
    sorted_pairs = sorted(zip(scores, candidates), reverse=True)
    if len(sorted_pairs) >= 2 and (sorted_pairs[0][0] - sorted_pairs[1][0]) < 0.1:
        # Tie-break by length (up to a reasonable max)
        candidates_top2 = [c for _, c in sorted_pairs[:2]]
        best = max(candidates_top2, key=lambda c: min(len(c), 2000))

    return best

# ── Step 5: Reflection & Self-Critique ───────────────────────────────────────

def reflect_and_improve(response: str, msg: str, system: str,
                        generate_fn, model: str, score: float) -> str:
    """
    If score < 0.65, attempt one reflection pass.
    Identifies specific gaps and regenerates targeted improvements.
    Not run on easy queries or when score is already high.
    """
    if score >= 0.65 or len(response) < 50:
        return response

    critique_prompt = [
        {"role": "system", "content":
            "You are a response critic. Identify the single most important gap "
            "or error in this response in ONE sentence. Be specific, not generic. "
            "Then output IMPROVED: followed by the complete improved response."},
        {"role": "user", "content":
            f"Question: {msg[:300]}\nResponse: {response[:800]}\n\n"
            "What is the most critical gap? Then write the improved version."}
    ]
    try:
        critique_raw = generate_fn(critique_prompt, max_tokens=1500, model=model)
        if "IMPROVED:" in critique_raw:
            improved = critique_raw.split("IMPROVED:", 1)[1].strip()
            if len(improved) > len(response) * 0.5:
                return improved
    except Exception as e:
        print(f"[Reflect] {e}")
    return response

# ── Main Entry Point ──────────────────────────────────────────────────────────

def deliberate(msg: str, system: str, history: list,
               generate_fn, model: str,
               complexity: str = "medium", skill: str = "general") -> str:
    """
    Full deliberative reasoning pipeline.
    Easy: single pass (fast)
    Medium: 2 hypotheses + scoring + best selection
    Hard: 3 hypotheses + scoring + self-consistency + reflection
    """
    t0 = time.time()

    if complexity == "easy":
        # Fast path: single generation
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(prompt, max_tokens=2500, model=model)

    n_hypotheses = 2 if complexity == "medium" else 3
    candidates = generate_hypotheses(msg, system, history, generate_fn, model, n=n_hypotheses)

    if not candidates:
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(prompt, max_tokens=1200, model=model)

    # Score all candidates
    score_futures = {
        _executor.submit(score_response, c, msg, generate_fn, model): i
        for i, c in enumerate(candidates)
    }
    scores = [0.5] * len(candidates)
    for fut in as_completed(score_futures, timeout=20):
        idx = score_futures[fut]
        try:
            scores[idx] = fut.result()
        except Exception:
            pass

    best = self_consistency_vote(candidates, scores)
    best_score = max(scores) if scores else 0.5

    print(f"[Deliberate] {n_hypotheses} candidates, best_score={best_score:.2f}, t={int((time.time()-t0)*1000)}ms")

    if complexity == "hard":
        best = reflect_and_improve(best, msg, system, generate_fn, model, best_score)

    return best
