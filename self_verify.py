import re

def self_verify(answer: str, original_prompt: str, generate_fn, skill: str = "general", complexity: str = "medium") -> str:
    """Upgraded: Added structural code checks before LLM critique."""
    if skill not in ("researcher", "coder") and complexity != "hard":
        return answer

    # Upgraded: Structural check for code blocks
    if skill == "coder":
        if "```" not in answer and ("def " in answer or "class " in answer or "import " in answer):
            return "REVISED\nI need to provide the code in a proper markdown block.\n```python\n" + answer + "\n```"
            
    critique_prompt = f"""You previously answered this question:
<question>{original_prompt}</question>

<answer>{answer}</answer>

Critique this answer:
1. Are there factual errors or unsupported claims?
2. Is anything missing that the question required?
3. Is the structure clear and complete?

If the answer is correct and complete, reply EXACTLY: VERIFIED
If it needs fixes, reply EXACTLY: REVISED\n<corrected answer here>"""

    try:
        result = generate_fn(critique_prompt) or ""
        if result.strip().startswith("REVISED"):
            revised_content = result.split("REVISED", 1)[-1].strip()
            if len(revised_content) > 20:
                return revised_content
        elif "VERIFIED" in result:
            return answer
    except Exception:
        pass
        
    return answer
