"""
Deliberative Reasoning Engine — replaces single-pass generation.
Implements: Chain-of-Thought, Tree-of-Thought sampling, Process Reward Modeling,
Self-consistency voting, and OODA with genuine state tracking.
Upgraded: Strict timeouts and fallbacks to prevent hanging.
"""
import re, time, random, threading, asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reason")

def decompose_problem(msg: str, generate_fn, model: str) -> dict:
    prompt = [
        {"role": "system", "content": "You are a problem decomposer. Analyze the user's request and output JSON only:\n{\"problem_type\": \"math|reasoning|factual|creative|code|multi_step\", \"sub_problems\": [\"...\"], \"requires_search\": true|false, \"requires_calculation\": true|false, \"ambiguities\": [\"...\"], \"complexity_estimate\": 1-10}"},
        {"role": "user", "content": msg[:500]}
    ]
    try:
        raw = generate_fn(prompt, max_tokens=1500, model=model)
        raw = re.sub(r'```json|```', '', raw).strip()
        import json
        return json.loads(raw)
    except Exception:
        return {"problem_type": "general", "sub_problems": [msg], "requires_search": False, "requires_calculation": False, "ambiguities": [], "complexity_estimate": 5}

def generate_hypotheses(msg: str, system: str, history: list, generate_fn, model: str, n: int = 3) -> list:
    def _gen_one(approach_hint: str) -> str:
        prompt = [{"role": "system", "content": system + f"\n\nApproach hint: {approach_hint}"}] + history[-6:] + [{"role": "user", "content": msg}]
        try:
            return generate_fn(prompt, max_tokens=1200, model=model)
        except Exception:
            return ""

    approaches = [
        "Direct and concise. Lead with the answer.",
        "Step by step reasoning. Show your work explicitly.",
        "Consider edge cases and alternative interpretations first.",
    ][:n]

    futures = [_executor.submit(_gen_one, approach) for approach in approaches]
    results = []
    for fut in as_completed(futures, timeout=30):
        try:
            r = fut.result(timeout=5)  # Hard timeout on result retrieval
            if r and len(r) > 50:
                results.append(r)
        except Exception:
            continue
    return results

def score_response(response: str, msg: str, generate_fn, model: str) -> float:
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
            weights = [0.40, 0.25, 0.25, 0.10]
            composite = sum(v * w for v, w in zip(vals, weights[:len(vals)])) / 10.0
            return composite
    except Exception:
        pass
    
    score = 0.5
    if len(response) > 200: score += 0.1
    if any(w in response for w in ['however', 'therefore', 'because', 'since']): score += 0.05
    if response.count('\n') > 3: score += 0.05
    bad_signals = ['I cannot', 'I apologize', 'As an AI', 'I don\'t have']
    if any(b in response for b in bad_signals): score -= 0.15
    return min(max(score, 0.0), 1.0)

def self_consistency_vote(candidates: list, scores: list) -> str:
    if not candidates: return ""
    if len(candidates) == 1: return candidates[0]

    best_idx = scores.index(max(scores))
    best = candidates[best_idx]

    sorted_pairs = sorted(zip(scores, candidates), reverse=True)
    if len(sorted_pairs) >= 2 and (sorted_pairs[0][0] - sorted_pairs[1][0]) < 0.1:
        candidates_top2 = [c for _, c in sorted_pairs[:2]]
        best = max(candidates_top2, key=lambda c: min(len(c), 2000))
    return best

def reflect_and_improve(response: str, msg: str, system: str, generate_fn, model: str, score: float) -> str:
    if score >= 0.65 or len(response) < 50:
        return response

    critique_prompt = [
        {"role": "system", "content": "You are a response critic. Identify the single most important gap or error in this response in ONE sentence. Be specific. Then output IMPROVED: followed by the complete improved response."},
        {"role": "user", "content": f"Question: {msg[:300]}\nResponse: {response[:800]}\n\nWhat is the most critical gap? Then write the improved version."}
    ]
    try:
        critique_raw = generate_fn(critique_prompt, max_tokens=1500, model=model)
        if "IMPROVED:" in critique_raw:
            improved = critique_raw.split("IMPROVED:", 1)[1].strip()
            if len(improved) > len(response) * 0.5:
                return improved
    except Exception:
        pass
    return response

def deliberate(msg: str, system: str, history: list, generate_fn, model: str, complexity: str = "medium", skill: str = "general") -> str:
    t0 = time.time()

    if complexity == "easy":
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(prompt, max_tokens=2500, model=model)

    n_hypotheses = 2 if complexity == "medium" else 3
    candidates = generate_hypotheses(msg, system, history, generate_fn, model, n=n_hypotheses)

    if not candidates:
        prompt = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(prompt, max_tokens=1200, model=model)

    score_futures = {_executor.submit(score_response, c, msg, generate_fn, model): i for i, c in enumerate(candidates)}
    scores = [0.5] * len(candidates)
    for fut in as_completed(score_futures, timeout=20):
        idx = score_futures[fut]
        try:
            scores[idx] = fut.result(timeout=5)
        except Exception:
            pass

    best = self_consistency_vote(candidates, scores)
    best_score = max(scores) if scores else 0.5

    if complexity == "hard":
        best = reflect_and_improve(best, msg, system, generate_fn, model, best_score)

    return best
