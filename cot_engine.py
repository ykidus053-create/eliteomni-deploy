
import re

DECOMPOSE_PROMPT = (
    "Before answering, privately work through:\n"
    "RESTATE: What is actually being asked?\n"
    "CONSTRAINTS: What must the answer satisfy?\n"
    "APPROACH: What method will you use and why?\n"
    "RISKS: Where could you go wrong?\n"
    "ANSWER: Now give your complete response.\n"
    "Do NOT show the RESTATE/CONSTRAINTS/APPROACH/RISKS lines in output."
)
MATH_COT_PROMPT = (
    "PARSE: Extract all numbers and operations explicitly.\n"
    "PLAN: Write each arithmetic step before doing it.\n"
    "EXECUTE: Work each step showing intermediate value.\n"
    "VERIFY: Confirm with a second method.\n"
    "FINAL: State the answer in **bold**."
)
CODE_COT_PROMPT = (
    "UNDERSTAND: Restate what the code must do in one sentence.\n"
    "EDGE_CASES: List at least 3 inputs that could cause failures.\n"
    "DESIGN: Write pseudocode before writing real code.\n"
    "IMPLEMENT: Write the complete, runnable implementation.\n"
    "VERIFY: Mentally trace through with one concrete test input."
)
RESEARCH_COT_PROMPT = (
    "SCOPE: Define what is and is not covered.\n"
    "DECOMPOSE: Break into 3-5 sub-questions.\n"
    "SYNTHESIZE: Answer each sub-question with evidence.\n"
    "INTEGRATE: Combine into a coherent answer.\n"
    "CONFIDENCE: Rate each major claim HIGH / MEDIUM / LOW."
)

def get_cot_prompt(skill, complexity, msg):
    m = msg.lower()
    is_math = skill == "calculator" or any(op in m for op in [" + ", " - ", " * ", " / ", "sqrt", "percent"])
    if is_math:
        return MATH_COT_PROMPT
    if skill == "coder":
        return CODE_COT_PROMPT
    if skill == "researcher" or complexity == "hard":
        return RESEARCH_COT_PROMPT
    if complexity == "easy" and len(msg) < 80:
        return ""
    return DECOMPOSE_PROMPT

def inject_cot(system_prompt, skill, complexity, msg):
    cot = get_cot_prompt(skill, complexity, msg)
    if not cot:
        return system_prompt
    return system_prompt + "\n\n" + cot

def strip_reasoning_artifacts(text):
    for label in ["RESTATE:","CONSTRAINTS:","APPROACH:","RISKS:",
                  "PARSE:","PLAN:","EXECUTE:","VERIFY:",
                  "SCOPE:","DECOMPOSE:","SYNTHESIZE:","INTEGRATE:",
                  "CONFIDENCE:","UNDERSTAND:","EDGE_CASES:","DESIGN:"]:
        text = re.sub(re.escape(label) + r"[^\n]*\n?", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def cot_complexity_gate(msg, response, skill):
    if len(response) < 50:
        return False
    signals = ["calculate","prove","debug","analyze","compare","why does","how does","design","implement","explain"]
    return any(s in msg.lower() for s in signals)
