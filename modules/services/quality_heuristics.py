
# REAL QUALITY HEURISTICS — Andrew Ng suggestion
# Called instead of hardcoded 9.0

def compute_response_quality(response: str, question: str, skill: str) -> float:
    """Heuristic quality score 0-10. No LLM call needed."""
    score = 5.0
    # Length relative to question complexity
    q_len = len(question); r_len = len(response)
    if skill == "calculator" and r_len < 20: score -= 3
    if skill == "coder" and "```" not in response: score -= 2
    if skill == "researcher" and r_len < 200: score -= 2
    # Hallucination signals
    import re
    fake_cites = len(re.findall(r"According to \w+,", response))
    score -= min(fake_cites * 0.5, 2)
    # Good signals
    if "CALC(" in response or "PATH B" in response: score += 0.5
    if "[VERIFIED]" in response: score += 0.5
    if "I don't know" in response or "I'm not certain" in response: score += 0.3
    # Sycophancy penalty
    bad_starts = ["Certainly!", "Absolutely!", "Great question", "Sure!", "Of course!"]
    if any(response.startswith(b) for b in bad_starts): score -= 1
    return max(0.0, min(10.0, score))
