
import re

TOOL_SIGNALS = {
    "SEARCH": ["latest","current","news","today","2025","2026","who is",
               "what happened","recent","price","stock","weather","live","breaking"],
    "CALC":   ["calculate","compute","how much is","percent","%",
               "multiply","divide","square root","sqrt","formula","solve for"],
    "TIME":   ["what time","current time","what date","today is","right now"],
    "FETCH":  ["from this url","at this link","fetch","scrape","at http"],
}

TOOL_EXCLUSIONS = {
    "SEARCH": ["def ","print(","function","class ","import "],
    "CALC":   ["explain","concept of","history of","what is the meaning"],
}

def plan_tools(msg, skill, complexity):
    plan = []
    m = msg.lower()
    for tool, signals in TOOL_SIGNALS.items():
        excl = TOOL_EXCLUSIONS.get(tool, [])
        if any(s in m for s in signals) and not any(e in m for e in excl):
            triggered_by = next((s for s in signals if s in m), signals[0])
            urgency = "required" if triggered_by in signals[:3] else "optional"
            plan.append({"tool": tool, "urgency": urgency, "triggered_by": triggered_by})
    if skill == "calculator" and not any(p["tool"] == "CALC" for p in plan):
        if len(re.findall(r"\d+", msg)) >= 2:
            plan.append({"tool": "CALC", "urgency": "required", "triggered_by": "numeric_values"})
    return plan

def tool_plan_to_system(msg, skill, complexity):
    plan = plan_tools(msg, skill, complexity)
    if not plan:
        return ""
    required = [p["tool"] for p in plan if p["urgency"] == "required"]
    optional = [p["tool"] for p in plan if p["urgency"] == "optional"]
    parts = []
    if required:
        parts.append("REQUIRED TOOLS: " + ", ".join(required) + " -- you MUST call these before answering.")
    if optional:
        parts.append("SUGGESTED TOOLS: " + ", ".join(optional) + " -- call if it improves accuracy.")
    return "[TOOL PLAN]\n" + "\n".join(parts) + "\n[/TOOL PLAN]" if parts else ""

def requires_tool(msg, tool):
    return any(p["tool"] == tool for p in plan_tools(msg, "general", "medium"))
