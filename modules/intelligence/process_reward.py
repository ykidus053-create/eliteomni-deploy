"""
Process Reward Model (PRM)
Scores intermediate reasoning steps, not just final answers.
This is what separates o1-style reasoning from vanilla generation.
Detects reasoning errors BEFORE they propagate to the final answer.
"""
import re
from typing import List, Tuple

STEP_PATTERNS = [
    r'(?:Step \d+|First|Then|Next|Finally|Therefore|Thus|So|Now)[,:]?\s+',
    r'(?:\d+\.\s+)',
    r'(?:→|⟹|∴)\s+',
]

def extract_reasoning_steps(text: str) -> List[str]:
    """Extract individual reasoning steps from a chain-of-thought response."""
    steps = []
    for pat in STEP_PATTERNS:
        parts = re.split(pat, text)
        if len(parts) > 2:
            steps = [p.strip() for p in parts if p.strip() and len(p.strip()) > 20]
            break
    if not steps:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        steps = [s for s in sentences if len(s) > 30]
    return steps[:15]

def score_step(step: str, prev_steps: List[str], problem: str) -> Tuple[float, str]:
    """
    Heuristic PRM scoring for a reasoning step.
    Returns (score 0-1, issue description).
    In production this would call a trained reward model.
    """
    issues = []
    score = 1.0

    # Check for unsupported absolute claims
    absolutes = re.findall(
        r'\b(always|never|guaranteed|impossible|certainly|definitely|100%)\b',
        step, re.I)
    if absolutes and not any(h in step.lower() for h in
                              ["approximately", "likely", "probably", "I think"]):
        score -= 0.2 * len(absolutes)
        issues.append(f"unsupported absolute: {absolutes[0]}")

    # Check for circular reasoning (repeating prior step verbatim)
    for prev in prev_steps[-3:]:
        overlap = len(set(step.lower().split()) & set(prev.lower().split()))
        total = max(len(step.split()), 1)
        if overlap / total > 0.7:
            score -= 0.35
            issues.append("circular repetition detected")
            break

    # Check for units in math steps
    if any(c.isdigit() for c in step):
        has_unit = bool(re.search(
            r'(km|m|kg|g|s|ms|GB|MB|KB|%|dollars|\$|hours|days|years|tokens|bits)',
            step, re.I))
        if not has_unit and "=" in step:
            score -= 0.1
            issues.append("numeric result without units")

    # Check for unjustified leaps
    conjunctions = len(re.findall(r'\b(because|since|therefore|due to|given that)\b',
                                   step, re.I))
    if len(step.split()) > 40 and conjunctions == 0:
        score -= 0.15
        issues.append("long claim without justification")

    score = max(0.0, min(1.0, score))
    issue_str = "; ".join(issues) if issues else "ok"
    return score, issue_str

def evaluate_reasoning_chain(response: str, problem: str) -> dict:
    """
    Evaluate the full reasoning chain of a response.
    Returns structured quality assessment.
    """
    steps = extract_reasoning_steps(response)
    if not steps:
        return {"steps": 0, "avg_score": 1.0, "min_score": 1.0,
                "issues": [], "recommendation": "no_chain_detected"}

    scores = []
    issues = []
    prev = []
    for step in steps:
        score, issue = score_step(step, prev, problem)
        scores.append(score)
        if issue != "ok":
            issues.append({"step": step[:80], "issue": issue, "score": score})
        prev.append(step)

    avg = sum(scores) / len(scores) if scores else 1.0
    min_s = min(scores) if scores else 1.0

    recommendation = "accept"
    if min_s < 0.4:
        recommendation = "revise_step"
    elif avg < 0.65:
        recommendation = "revise_chain"
    elif len([i for i in issues if i["score"] < 0.6]) > 2:
        recommendation = "flag_for_review"

    return {
        "steps": len(steps),
        "avg_score": round(avg, 3),
        "min_score": round(min_s, 3),
        "issues": issues[:3],
        "recommendation": recommendation,
    }

def prm_annotation(response: str, problem: str) -> str:
    """Append PRM quality note to response if issues found."""
    result = evaluate_reasoning_chain(response, problem)
    if result["recommendation"] in ("accept", "no_chain_detected"):
        return response
    if result["recommendation"] == "revise_step" and result["issues"]:
        worst = result["issues"][0]
        note = (f"\n\n> ⚠️ **Reasoning note:** Step confidence {result['min_score']:.0%} "
                f"— {worst['issue']}. Verify this step independently.")
        return response + note
    if result["recommendation"] == "revise_chain":
        note = (f"\n\n> 📊 **Chain quality:** {result['avg_score']:.0%} avg step confidence "
                f"across {result['steps']} steps. Key issues: "
                + "; ".join(i['issue'] for i in result['issues'][:2]))
        return response + note
    return response
