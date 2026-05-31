# PROCESS REWARD MODEL — Hassabis suggestion
# Scores each reasoning step independently before committing to output
import re as _prm_re

def prm_score_steps(response: str, question: str, groq_fn) -> dict:
    """
    Score each reasoning step 1-5 via an independent critic call.
    Returns {steps: [...], min_score: N, approved: bool}
    Low min_score triggers a regeneration request.
    """
    steps = _prm_re.split(r'\n(?=\d+\.\s|STEP\s*\d|##\s)', response)
    steps = [s.strip() for s in steps if len(s.strip()) > 40][:6]
    if not steps:
        return {"steps": [], "min_score": 5, "approved": True}
    scored = []
    for i, step in enumerate(steps):
        try:
            raw = groq_fn([{"role": "user", "content":
                f"Rate reasoning quality 1-5.\nQuestion: {question[:150]}\nStep: {step[:300]}\nReply ONLY a digit 1-5:"}],
                max_tokens=3)
            d = _prm_re.search(r'[1-5]', raw or "3")
            scored.append(int(d.group()) if d else 3)
        except Exception:
            scored.append(3)
    min_s = min(scored) if scored else 3
    return {
        "steps": list(zip(steps, scored)),
        "min_score": min_s,
        "approved": min_s >= 3,
        "needs_regen": min_s <= 2,
    }
