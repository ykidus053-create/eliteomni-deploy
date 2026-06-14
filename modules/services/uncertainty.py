
"""
Hassabis: A system that doesn't know what it doesn't know
will hallucinate confidently. This module detects and signals
uncertainty before it reaches the user.
"""
import re

# Phrases that signal the model is guessing
UNCERTAINTY_SIGNALS = [
    r"\b(i think|i believe|probably|likely|might|could be|perhaps|possibly)\b",
    r"\b(not sure|uncertain|unclear|depends|varies|approximate)\b",
    r"\b(around|roughly|approximately|about|nearly|almost)\b",
]

# Phrases that signal overconfidence (red flags)
OVERCONFIDENCE_SIGNALS = [
    r"\b(definitely|certainly|absolutely|always|never|exactly|guaranteed)\b",
    r"\b(100%|without a doubt|undoubtedly|unquestionably)\b",
]

# Topics where LLMs commonly hallucinate
HIGH_RISK_TOPICS = [
    "statistics", "numbers", "dates", "names", "citations",
    "research", "study", "paper", "published", "source",
    "price", "cost", "revenue", "population", "percentage",
]

def assess_uncertainty(response: str, question: str) -> dict:
    """
    Returns uncertainty assessment:
    {
      level: "low" | "medium" | "high",
      score: 0.0-1.0,
      flags: [...],
      disclaimer: str | None
    }
    """
    r = response.lower()
    q = question.lower()
    flags = []
    uncertainty_score = 0.0

    # Check for overconfidence on risky topics
    is_risky_topic = any(t in q for t in HIGH_RISK_TOPICS)
    if is_risky_topic:
        uncertainty_score += 0.3
        flags.append("high-risk topic (statistics/facts/dates)")

    # Count overconfidence markers
    overconf_count = sum(
        len(re.findall(sig, r, re.IGNORECASE))
        for sig in OVERCONFIDENCE_SIGNALS
    )
    if overconf_count > 0:
        uncertainty_score += min(overconf_count * 0.15, 0.4)
        flags.append(f"overconfidence markers detected ({overconf_count})")

    # Check for specific numbers without sources
    numbers_in_response = re.findall(r"\b\d+\.?\d*%?\b", response)
    if len(numbers_in_response) > 3 and not any(
        w in r for w in ["source", "according", "reference", "cited"]
    ):
        uncertainty_score += 0.2
        flags.append("multiple specific numbers without sources")

    # Calibrate: uncertainty markers REDUCE the score
    calib_count = sum(
        len(re.findall(sig, r, re.IGNORECASE))
        for sig in UNCERTAINTY_SIGNALS
    )
    uncertainty_score -= min(calib_count * 0.05, 0.2)
    uncertainty_score = max(0.0, min(1.0, uncertainty_score))

    if uncertainty_score >= 0.5:
        level = "high"
        disclaimer = (
            "⚠️ *This response contains claims I'm not fully certain about. "
            "Please verify specific numbers, dates, or facts independently.*"
        )
    elif uncertainty_score >= 0.25:
        level = "medium"
        disclaimer = (
            "*Note: Some details here may benefit from independent verification.*"
        )
    else:
        level = "low"
        disclaimer = None

    return {
        "level": level,
        "score": round(uncertainty_score, 3),
        "flags": flags,
        "disclaimer": disclaimer,
    }


def append_uncertainty_disclaimer(response: str, question: str) -> str:
    """Append uncertainty disclaimer to response if warranted."""
    assessment = assess_uncertainty(response, question)
    if assessment["disclaimer"]:
        return response + "\n\n" + assessment["disclaimer"]
    return response
