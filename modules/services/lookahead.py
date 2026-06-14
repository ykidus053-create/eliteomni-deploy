
"""
Hassabis: Before committing to a response, simulate N candidates
and score them. Return the best one. This is the core of AlphaGo
applied to text generation — search beats pattern matching.
"""
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def score_response(response: str, question: str) -> float:
    """
    Evaluate a candidate response on 5 dimensions.
    No LLM call needed — fast heuristics inspired by constitutional AI.
    """
    if not response or len(response.strip()) < 20:
        return 0.0

    score = 0.0
    r = response.lower()
    q = question.lower()

    # 1. Relevance — does it address the question?
    question_words = set(re.findall(r"\b\w{4,}\b", q))
    response_words = set(re.findall(r"\b\w{4,}\b", r))
    overlap = len(question_words & response_words) / max(len(question_words), 1)
    score += overlap * 0.25

    # 2. Completeness — length relative to question complexity
    q_complexity = len(question) / 100
    ideal_length = min(max(q_complexity * 200, 100), 2000)
    length_score = 1.0 - abs(len(response) - ideal_length) / max(ideal_length, 1)
    score += max(length_score, 0) * 0.20

    # 3. Confidence calibration — penalize overconfidence
    overconf = len(re.findall(
        r"\b(exactly|always|never|100%|guaranteed|certainly|absolutely)\b",
        r, re.IGNORECASE
    ))
    score += max(0.15 - overconf * 0.05, 0)

    # 4. Structure — headers, lists, code blocks signal organized thinking
    has_structure = bool(
        re.search(r"^#{1,3} |^\d+\.|^[-*] |```", response, re.MULTILINE)
    )
    score += 0.20 if has_structure else 0.05

    # 5. Honesty markers — "I think", "likely", "uncertain" = calibrated
    honest_markers = len(re.findall(
        r"\b(likely|probably|i think|i believe|uncertain|might|could be|not sure)\b",
        r, re.IGNORECASE
    ))
    score += min(honest_markers * 0.05, 0.20)

    return round(min(score, 1.0), 4)


def generate_candidates(msgs: list, generate_fn, n: int = 3,
                        max_tokens: int = 1500) -> list:
    """
    Generate N candidate responses in parallel using slight temperature variation.
    Returns list of (response, score) tuples sorted best-first.
    """
    candidates = []
    lock = threading.Lock()

    def _gen(temp_offset: float):
        try:
            # Vary temperature slightly for diversity
            result = generate_fn(msgs, max_tokens=max_tokens)
            if result and len(result.strip()) > 30:
                with lock:
                    candidates.append(result)
        except Exception as e:
            print(f"[Lookahead] candidate failed: {e}")

    # Generate in parallel
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(_gen, i * 0.1) for i in range(n)]
        for f in as_completed(futures):
            pass

    if not candidates:
        return []

    question = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            question = m.get("content", "")
            break

    scored = [(r, score_response(r, question)) for r in candidates]
    scored.sort(key=lambda x: -x[1])
    return scored


def best_of_n(msgs: list, generate_fn, n: int = 3,
              complexity: str = "medium", max_tokens: int = 1500) -> str:
    """
    Main entry point.
    Easy queries: just generate once (fast path).
    Medium/hard: generate N, score all, return best.
    """
    if complexity == "easy" or n <= 1:
        return generate_fn(msgs, max_tokens=max_tokens) or ""

    scored = generate_candidates(msgs, generate_fn, n=n, max_tokens=max_tokens)

    if not scored:
        return generate_fn(msgs, max_tokens=max_tokens) or ""

    best, best_score = scored[0]
    print(f"[Lookahead] {len(scored)} candidates scored. "
          f"Best: {best_score:.3f} | "
          f"Worst: {scored[-1][1]:.3f}")
    return best
