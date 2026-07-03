import sqlite3
from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum
import time

class ModelTier(Enum):
    CODING = "devstral-latest"         # Tailored for programming tasks
    GENERAL = "magistral-medium-latest"    # Used for everything else

@dataclass
class RoutingDecision:
    model: ModelTier
    use_tools: bool
    use_voting: bool
    max_output_tokens: int
    estimated_cost_usd: float
    rationale: str

class IntelligentRouter:
    """
    ROI-driven routing optimized for Codestral and Mistral Large.
    - Coding tasks -> codestral-latest
    - All other tasks -> mistral-large-latest
    """

    VOTING_INDICATORS = [
        "calculate", "what is the answer to", "solve",
        "how many", "what percentage", "probability of"
    ]

    def route(self,
              user_message: str,
              skill: str,
              complexity: str,
              conversation_length: int = 0) -> RoutingDecision:

        msg_lower = user_message.lower()

        # Enforce model routing rule
        if skill == "coder":
            model = ModelTier.CODING
        else:
            model = ModelTier.GENERAL

        # Determine if voting is cost-justified
        is_math_query = (
            skill == "calculator" or
            any(ind in msg_lower for ind in self.VOTING_INDICATORS)
        )
        
        use_voting = (
            is_math_query and
            complexity in ("medium", "hard") and
            model == ModelTier.GENERAL
        )

        # Output token budget mapping
        token_budgets = {
            ("easy", "general"):     512,
            ("medium", "general"):  1024,
            ("hard", "general"):    2048,
            ("easy", "coder"):      1024,
            ("medium", "coder"):    2048,
            ("hard", "coder"):      4096,
            ("easy", "researcher"):  512,
            ("medium", "researcher"): 1500,
            ("hard", "researcher"):  3000,
            ("easy", "calculator"):  256,
            ("medium", "calculator"): 512,
            ("hard", "calculator"):  1024,
        }
        max_tokens = token_budgets.get((complexity, skill), 1024)

        # Cost estimate math
        est_input_tokens = 800 + (conversation_length * 100)
        vote_multiplier = 5 if use_voting else 1

        cost_per_million = {
            ModelTier.CODING: (0.20, 0.60),   
            ModelTier.GENERAL: (2.00, 6.00),  
        }
        input_cost, output_cost = cost_per_million[model]

        estimated_cost = (
            (est_input_tokens / 1_000_000) * input_cost +
            (max_tokens / 1_000_000) * output_cost
        ) * vote_multiplier

        rationale = (
            f"{model.value} | "
            f"voting={'yes' if use_voting else 'no'} | "
            f"~${estimated_cost:.4f}"
        )

        return RoutingDecision(
            model=model,
            use_tools=complexity != "easy",
            use_voting=use_voting,
            max_output_tokens=max_tokens,
            estimated_cost_usd=estimated_cost,
            rationale=rationale
        )
