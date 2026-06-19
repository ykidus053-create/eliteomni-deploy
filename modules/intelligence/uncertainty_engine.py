"""
Uncertainty Quantification Engine
Moves beyond hedging words to structured uncertainty representation.
Tracks: epistemic uncertainty (lack of knowledge),
        aleatoric uncertainty (inherent randomness),
        model uncertainty (calibration gaps).
"""
import re
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class UncertaintyProfile:
    epistemic: float   # 0-1: how uncertain due to missing knowledge
    aleatoric: float   # 0-1: inherent randomness in the domain
    temporal: float    # 0-1: uncertainty due to information staleness
    domain_risk: float # 0-1: risk of being wrong in this domain

    def overall(self) -> float:
        return (self.epistemic * 0.4 + self.aleatoric * 0.2 +
                self.temporal * 0.2 + self.domain_risk * 0.2)

    def to_hedge_level(self) -> str:
        o = self.overall()
        if o > 0.7: return "high"
        if o > 0.4: return "medium"
        if o > 0.2: return "low"
        return "minimal"

DOMAIN_RISKS = {
    "medical": 0.9, "legal": 0.85, "financial": 0.8,
    "scientific": 0.5, "technical": 0.3, "historical": 0.2,
    "mathematical": 0.1, "coding": 0.2, "general": 0.3,
}

TEMPORAL_SIGNALS = [
    "current", "latest", "recent", "now", "today", "2024", "2025", "2026",
    "this year", "last year", "trending", "new",
]

EPISTEMIC_SIGNALS = [
    "why", "how does", "what causes", "explain", "reason",
    "mechanism", "theory", "hypothesis",
]

def assess_uncertainty(msg: str, skill: str) -> UncertaintyProfile:
    m = msg.lower()

    # Epistemic: questions about causal mechanisms or explanations
    epistemic = 0.3
    if any(s in m for s in EPISTEMIC_SIGNALS):
        epistemic = 0.5
    if "prove" in m or "certain" in m:
        epistemic = 0.6

    # Temporal: questions about current state of the world
    temporal = 0.2
    if any(s in m for s in TEMPORAL_SIGNALS):
        temporal = 0.7

    # Domain risk
    domain = "general"
    for d in DOMAIN_RISKS:
        if d in m or d in skill:
            domain = d
            break
    domain_risk = DOMAIN_RISKS.get(domain, 0.3)

    # Aleatoric: probabilistic or forecasting questions
    aleatoric = 0.1
    if any(s in m for s in ["predict", "forecast", "will", "probability", "chance", "risk"]):
        aleatoric = 0.6

    return UncertaintyProfile(
        epistemic=epistemic,
        aleatoric=aleatoric,
        temporal=temporal,
        domain_risk=domain_risk,
    )

HEDGE_TEMPLATES = {
    "high": [
        "I'm not certain, but my understanding is that",
        "This is a domain with significant uncertainty —",
        "Based on my training data (which may be outdated):",
        "I'd recommend verifying this with a primary source, but",
    ],
    "medium": [
        "Generally speaking,",
        "Based on my knowledge,",
        "This is likely accurate, though worth verifying:",
        "My understanding is that",
    ],
    "low": ["", "", "In most cases,", ""],  # mostly don't hedge
    "minimal": [""] * 4,
}

def get_calibration_prefix(profile: UncertaintyProfile) -> str:
    """Return an appropriate epistemic prefix for the response."""
    level = profile.to_hedge_level()
    if level == "minimal":
        return ""
    templates = HEDGE_TEMPLATES[level]
    import random
    return random.choice(templates)

def build_uncertainty_injection(msg: str, skill: str) -> str:
    """Build uncertainty-aware instruction for this specific query."""
    profile = assess_uncertainty(msg, skill)
    level = profile.to_hedge_level()

    if level == "minimal":
        return ""

    parts = [f"\n<uncertainty_profile level='{level}'>"]
    if profile.epistemic > 0.5:
        parts.append("HIGH EPISTEMIC UNCERTAINTY: Express genuine uncertainty about causal claims.")
    if profile.temporal > 0.5:
        parts.append("HIGH TEMPORAL UNCERTAINTY: Flag that information may be outdated; recommend verification.")
    if profile.domain_risk > 0.6:
        parts.append(f"HIGH DOMAIN RISK ({profile.domain_risk:.0%}): Add professional consultation recommendation.")
    if profile.aleatoric > 0.4:
        parts.append("INHERENT UNPREDICTABILITY: Express probabilistic language, not point estimates.")
    parts.append("</uncertainty_profile>")
    return "\n".join(parts)
