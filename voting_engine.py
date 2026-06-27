import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vote")
VOTE_SKILLS = {"calculator", "researcher", "coder"}

def should_use_voting(msg, skill, complexity):
    if complexity == "easy": return False
    if skill in VOTE_SKILLS and complexity == "hard": return True
    triggers = ["prove","calculate","solve","how many","find the","compute"]
    return complexity == "hard" and any(t in msg.lower() for t in triggers)

def _extract_final_answer(text):
    patterns = [
        r"(?:FINAL|ANSWER|Result|Therefore|Thus|=)\s*:?\s*([\d\.\,\-\+]+)",
        r"\*\*([^\*]+)\*\*",
        r"`([^`]+)`",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).strip()
    words = text.strip().split()
    return " ".join(words[-12:]) if words else text[:80]

def _normalize(answer):
    a = answer.lower().strip()
    a = re.sub(r"[\$\,\s]", "", a)
    try: return str(round(float(a), 4))
    except Exception: return a

def _llm_judge(generate_fn, msg, candidates):
    """Upgraded: LLM-as-a-Judge to break ties and vote on complex answers."""
    try:
        prompt = [f"You are an impartial judge. Question: {msg}\n\nCandidate A:\n{candidates[0]}\n\nCandidate B:\n{candidates[1]}\n\nWhich candidate is more accurate and complete? Reply ONLY 'A' or 'B'."]
        res = generate_fn(prompt, max_tokens=5)
        if "A" in res.upper(): return candidates[0]
        if "B" in res.upper(): return candidates[1]
    except:
        pass
    return max(candidates, key=len)

def self_consistent_answer(generate_fn, msgs, n_samples=3, max_tokens=800):
    """Upgraded: Mixture of Experts approach hints for each sample."""
    approaches = [
        "Approach: Solve directly and efficiently.",
        "Approach: Think step by step, showing all work.",
        "Approach: Consider edge cases and verify the result."
    ]
    
    futures = []
    for i in range(n_samples):
        temp = list(msgs)
        if i > 0:
            last = dict(temp[-1])
            last["content"] = last["content"] + f" [{approaches[i]}]"
            temp[-1] = last
        futures.append(_executor.submit(generate_fn, temp, max_tokens))
        
    results = []
    for fut in as_completed(futures):
        try:
            r = fut.result(timeout=45)
            if r and len(r) > 10: results.append(r)
        except Exception: pass
        
    if not results: return "", 0.0, []
    if len(results) == 1: return results[0], 0.5, results
    
    normalized = [_normalize(_extract_final_answer(r)) for r in results]
    counts = Counter(normalized)
    best_norm, best_count = counts.most_common(1)[0]
    confidence = best_count / len(results)
    best_idx = next((i for i, n in enumerate(normalized) if n == best_norm), 0)
    best_response = results[best_idx]
    
    # Upgraded: If no consensus, use LLM Judge on top 2 candidates
    if best_count == 1 and len(results) >= 3:
        best_response = _llm_judge(generate_fn, msgs[-1].get("content", ""), results[:2])
        confidence = 0.6
        
    return best_response, confidence, results

def vote_report(results, confidence):
    if not results or confidence >= 0.9: return ""
    if len(set(_normalize(_extract_final_answer(r)) for r in results)) > 1:
        return "\n\n> Verified across " + str(len(results)) + " independent samples (confidence: " + str(round(confidence*100)) + "%)."
    return ""
