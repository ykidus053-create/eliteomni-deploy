import os, time
from typing import Optional

TIER_CRITICAL  = 0
TIER_SKILL     = 1
TIER_REASONING = 2
TIER_CONTEXT   = 3
TIER_OPTIONAL  = 4

MAX_CHARS = {"easy": 2000, "medium": 4000, "hard": 8000}

class PromptAssembler:
    def __init__(self):
        self._parts = []

    def add(self, tier, text):
        if text and text.strip():
            self._parts.append((tier, text.strip()))
        return self

    def build(self, complexity="medium", max_chars=None):
        limit = max_chars or MAX_CHARS.get(complexity, 4000)
        sorted_parts = sorted(self._parts, key=lambda x: x[0])
        result, used = [], 0
        for tier, text in sorted_parts:
            chars = len(text) + 1
            if tier <= TIER_SKILL:
                result.append(text)
                used += chars
            elif used + chars <= limit:
                result.append(text)
                used += chars
        return "\n".join(result)

    def reset(self):
        self._parts = []
        return self


def build_prompt_prioritized(skill, complexity, memory, episodic,
                              rlhf_note, ctx_summary="",
                              user_instructions="",
                              constitution_sample=None,
                              effort_prompts=None):
    try:
        from modules.memory import HIERARCHY, SKILLS, CONSTITUTION_WEIGHTED
        from modules.prompts import (RESPONSE_STYLE_PROMPT,
                                     ANTI_HALLUCINATION_PROMPT,
                                     REASONING_DISCIPLINE_PROMPT,
                                     UNCERTAINTY_PROMPT)
    except ImportError as e:
        return f"[PromptAssembler] import error: {e}"

    asm = PromptAssembler()

    asm.add(TIER_CRITICAL, " ".join(HIERARCHY["system"]))
    asm.add(TIER_CRITICAL, HIERARCHY["operator"][0])
    asm.add(TIER_CRITICAL,
        "Tools: SEARCH(q) CALC(expr) TIME() EXEC(code) FETCH(url). "
        "Never say you cannot search.")

    asm.add(TIER_SKILL, "SKILL: " + SKILLS[skill]["prompt"])
    if user_instructions:
        asm.add(TIER_SKILL,
            "USER PERSISTENT INSTRUCTIONS:\n" + user_instructions)

    if complexity in ("medium", "hard"):
        asm.add(TIER_REASONING, ANTI_HALLUCINATION_PROMPT.strip())
        asm.add(TIER_REASONING, REASONING_DISCIPLINE_PROMPT.strip())
        asm.add(TIER_REASONING, UNCERTAINTY_PROMPT.strip())
    asm.add(TIER_REASONING, RESPONSE_STYLE_PROMPT.strip())

    if memory:
        asm.add(TIER_CONTEXT,
            "MEMORY:\n" + "\n".join("- " + m[:120] for m in memory[:6]))
    if episodic:
        asm.add(TIER_CONTEXT,
            "EPISODIC:\n" + "\n".join("- " + e[:100] for e in episodic[:3]))
    if ctx_summary:
        asm.add(TIER_CONTEXT, "PRIOR CONTEXT: " + ctx_summary[:300])
    if rlhf_note:
        asm.add(TIER_CONTEXT, rlhf_note)

    if effort_prompts:
        for p in effort_prompts:
            asm.add(TIER_OPTIONAL, p)
    sample = (constitution_sample or CONSTITUTION_WEIGHTED)[:3]
    asm.add(TIER_OPTIONAL,
        "CONSTITUTION:\n" + "\n".join("- " + c for c in sample))

    return asm.build(complexity)
