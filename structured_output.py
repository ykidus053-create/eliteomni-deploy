import re

TEMPLATES = {
    "research_report": """Respond ONLY in this structure:
# Summary
# Findings (bulleted, with evidence)
# Counterarguments
# Confidence: High/Medium/Low
# Sources""",

    "code_review": """Respond ONLY in this structure:
# Critical Issues (bugs, security, correctness)
# Style Issues (naming, formatting, docs)
# Performance Notes
# Verdict: Approve / Request Changes""",

    "comparison": """Respond ONLY in this structure:
# Criteria Used
# Side-by-Side Table (markdown)
# Winner per Category
# Overall Recommendation""",
}

def detect_template(prompt: str) -> str | None:
    p = prompt.lower()
    if any(w in p for w in ["research", "analyze", "history of", "explain in detail"]):
        return TEMPLATES["research_report"]
    if any(w in p for w in ["review this code", "code review", "check my code"]):
        return TEMPLATES["code_review"]
    if any(w in p for w in ["compare", "vs ", "versus", "pros and cons", "difference between"]):
        return TEMPLATES["comparison"]
    return None

def inject_template(system_prompt: str, user_prompt: str) -> str:
    template = detect_template(user_prompt)
    if template:
        return system_prompt + "\n\n" + template
    return system_prompt
