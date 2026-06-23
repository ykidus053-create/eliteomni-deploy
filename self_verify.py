def self_verify(answer: str, original_prompt: str, generate_fn, skill: str = "general", complexity: str = "medium") -> str:
    """Run a critique pass on hard/researcher answers before returning."""
    if skill not in ("researcher", "coder") and complexity != "hard":
        return answer

    critique_prompt = f"""You previously answered this question:
<question>{original_prompt}</question>

<answer>{answer}</answer>

Critique this answer:
1. Are there factual errors or unsupported claims?
2. Is anything missing that the question required?
3. Is the structure clear and complete?

If the answer is correct and complete, reply: VERIFIED
If it needs fixes, reply: REVISED\n<corrected answer here>"""

    result = generate_fn(critique_prompt) or ""
    if result.strip().startswith("REVISED"):
        return result.split("REVISED", 1)[-1].strip()
    return answer
