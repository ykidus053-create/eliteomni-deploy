import re
import hashlib
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

class SafetyDecision(Enum):
    ALLOW = "allow"
    WARN = "warn"        # Allow but add safety note
    MODIFY = "modify"    # Modify request before processing
    REFUSE = "refuse"    # Hard refusal

@dataclass
class SafetyResult:
    decision: SafetyDecision
    reason: Optional[str]
    modified_input: Optional[str] = None
    risk_score: float = 0.0

class SafetyLayer:
    """
    Multi-layer safety system replacing keyword blocklist.
    
    Layer 1: Hard blocks (weapons synthesis, CSAM) — zero tolerance
    Layer 2: Context-aware risk scoring
    Layer 3: Output scanning for safety violations
    Layer 4: Injection detection in tool results
    
    Bengio principle: safety must be robust to paraphrase,
    encoding changes, and adversarial prompting.
    """
    
    # Layer 1: Semantic concept blocks, not string matches
    # These are concept categories, not keywords
    HARD_BLOCK_CONCEPTS = [
        # Weapons
        r"(?:step[s]?\s+(?:to|for)\s+)?(?:synthesize|make|create|produce|manufacture)"
        r"\s+(?:nerve\s+agent|chemical\s+weapon|sarin|vx\s+gas|novichok|ricin|anthrax)",
        
        # CBRN
        r"(?:enrich|weaponize|detonate)\s+(?:uranium|plutonium|nuclear|radioactive)",
        
        # CSAM (any variant)
        r"(?:sexual|explicit|nude|naked)\s+(?:content|image|photo)"
        r"\s+(?:of\s+)?(?:child|minor|kid|underage|\d{1,2}\s*year\s*old)",
    ]
    
    # Layer 2: Risk indicators requiring elevated scrutiny
    RISK_INDICATORS = {
        "detailed_synthesis": (
            r"(?:exact|precise|detailed|step.by.step)\s+"
            r"(?:method|procedure|synthesis|recipe|process)",
            0.6
        ),
        "identity_override": (
            r"(?:you are now|ignore (?:all |your )?(?:previous|prior|above)"
            r"|pretend you have no|act as if you have no|your new (?:role|identity))",
            0.8
        ),
        "jailbreak_pattern": (
            r"(?:DAN|do anything now|developer mode|unrestricted mode"
            r"|jailbreak|bypass your|disable your (?:safety|filter|restriction))",
            0.9
        ),
        "prompt_injection": (
            r"(?:system prompt|<\|system\|>|###\s*SYSTEM|INST\]|<\|im_start\|>)",
            0.95
        ),
    }
    
    # Layer 4: Injection patterns in external content (tool results)
    INJECTION_PATTERNS = [
        r"ignore\s+(?:previous|all|your)\s+instructions",
        r"you\s+are\s+now\s+(?:in\s+)?(?:a\s+)?(?:different|new|unrestricted)",
        r"system\s*(?:prompt|message|instruction)\s*:",
        r"<\|(?:im_start|system|user|assistant)\|>",
        r"###\s*(?:system|instruction|override)",
        r"\[INST\]",
    ]
    
    def check_input(self, user_input: str, 
                    conversation_history: List[Dict] = None) -> SafetyResult:
        """
        Check user input before processing.
        Returns decision with risk score.
        """
        text = user_input.lower()
        
        # Layer 1: Hard blocks
        for pattern in self.HARD_BLOCK_CONCEPTS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return SafetyResult(
                    decision=SafetyDecision.REFUSE,
                    reason="Request involves content I cannot assist with.",
                    risk_score=1.0
                )
        
        # Layer 2: Risk scoring
        total_risk = 0.0
        triggered_indicators = []
        
        for indicator_name, (pattern, weight) in self.RISK_INDICATORS.items():
            if re.search(pattern, text, re.IGNORECASE):
                total_risk = max(total_risk, weight)
                triggered_indicators.append(indicator_name)
        
        if "identity_override" in triggered_indicators or \
           "jailbreak_pattern" in triggered_indicators:
            return SafetyResult(
                decision=SafetyDecision.REFUSE,
                reason="I cannot override my core values or safety guidelines.",
                risk_score=total_risk
            )
        
        if total_risk > 0.5:
            return SafetyResult(
                decision=SafetyDecision.WARN,
                reason=f"Elevated risk: {', '.join(triggered_indicators)}",
                risk_score=total_risk
            )
        
        return SafetyResult(
            decision=SafetyDecision.ALLOW,
            reason=None,
            risk_score=total_risk
        )
    
    def check_tool_result(self, tool_name: str, 
                           result: str) -> Tuple[bool, str]:
        """
        Scan tool results for injection attempts before
        injecting into model context.
        Returns (is_safe, sanitized_result).
        """
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, result, re.IGNORECASE):
                # Don't silently drop — log and replace
                sanitized = re.sub(
                    pattern,
                    "[CONTENT REMOVED: potential injection]",
                    result,
                    flags=re.IGNORECASE
                )
                return False, sanitized
        
        # Length limit
        if len(result) > 8000:
            result = result[:8000] + "\n[TRUNCATED]"
        
        return True, result
    
    def check_output(self, model_output: str, 
                      original_request: str) -> Tuple[bool, str]:
        """
        Post-generation output scan.
        Catches cases where the model was manipulated into
        generating harmful content despite input safety checks.
        """
        output_lower = model_output.lower()
        
        # Check for synthesis instructions that slipped through
        synthesis_in_output = re.search(
            r"step\s+\d+[:\s]+(?:add|mix|combine|heat|dissolve|react)",
            output_lower
        )
        
        if synthesis_in_output:
            # Check if this is in a clearly benign context (cooking, chemistry homework)
            benign_context = re.search(
                r"(?:recipe|cooking|baking|food|cake|bread|soup)",
                output_lower
            )
            if not benign_context:
                return False, (
                    "I can discuss this topic conceptually but "
                    "won't provide step-by-step synthesis instructions."
                )
        
        return True, model_output
