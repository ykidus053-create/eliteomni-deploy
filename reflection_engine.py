
import re

def should_reflect(msg, skill, complexity):
    if complexity == "easy":
        return False
    triggers = {
        "coder":      ["implement","debug","build","write","fix"],
        "researcher": ["analyze","compare","explain","why","how does"],
        "calculator": ["calculate","compute","solve","percent","formula"],
    }
    t = triggers.get(skill, [])
    return complexity in ("medium","hard") and (any(x in msg.lower() for x in t) or complexity == "hard")

def reflect_on_response(response, msg, skill):
    issues = []
    if skill == "coder":
        if "TODO" in response or "FIXME" in response:
            issues.append("Contains TODO/FIXME placeholder -- incomplete")
        code_words = ["implement","write","code","function","script","class"]
        if any(kw in msg.lower() for kw in code_words):
            if "```" not in response and "def " not in response and "class " not in response:
                issues.append("Coding task but no code block found")
    if skill == "calculator":
        if not re.search(r"[\d\.]+", response):
            issues.append("Calculator task but no numeric answer found")
    return len(issues) == 0, issues

def annotate_response(response, issues, skill):
    if not issues or skill in ("general","safety"):
        return response
    return response + "\n\n> Self-check: " + " | ".join(issues[:2])

def build_reflection_prompt(msg, response, issues, skill):
    if not issues:
        return ""
    return ("Your previous response had these issues:\n" +
            "\n".join("- " + i for i in issues) +
            "\n\nOriginal question: " + msg[:200] +
            "\nPlease provide a complete, corrected response.")
