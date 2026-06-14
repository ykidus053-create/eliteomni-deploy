import re, time, json, sqlite3
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

DB = Path.home() / "eliteomni_tool_policy.db"

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS tool_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tool TEXT, trigger_pattern TEXT, skill TEXT,
        success INTEGER, latency_ms REAL, result_quality REAL, ts REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS tool_policy (
        state_signature TEXT PRIMARY KEY,
        recommended_tool TEXT, confidence REAL,
        success_rate REAL, sample_count INTEGER, updated REAL)""")
    c.commit()
    return c

@dataclass
class ToolDecision:
    tool: str
    args: str
    confidence: float
    reason: str
    priority: int = 5

TOOL_TRIGGERS = [
    (r"\b(latest|current|recent|today|news|2024|2025|2026|trending|now)\b", "SEARCH", 0.92, ["researcher","general"]),
    (r"\b(weather|temperature|forecast|rain|humidity)\b", "WEATHER", 0.95, ["general"]),
    (r"\b(calculate|compute|evaluate)\b", "CALC", 0.90, ["calculator","general"]),
    (r"https?://\S+", "FETCH", 0.95, ["researcher","general"]),
    (r"\b(execute|run|test|output of|result of)\s+(this|the)\s+(code|script|program)", "EXEC", 0.85, ["coder"]),
    (r"\b(what time|current time|what date|today's date)\b", "TIME", 0.99, ["general"]),
    (r"\b(search for|find information|look up|research)\b", "SEARCH", 0.80, ["researcher","general"]),
    (r"\b(grep|find in|search in|locate in)\s+\w+\.(py|js|ts|go|rs)", "GREP", 0.85, ["coder"]),
]

def decide_tools(msg: str, skill: str, complexity: str) -> List[ToolDecision]:
    decisions = []
    seen_tools = set()
    m = msg.lower()
    try:
        c = _conn()
        policy_rows = c.execute(
            "SELECT recommended_tool, confidence FROM tool_policy WHERE success_rate > 0.6 ORDER BY confidence DESC").fetchall()
        c.close()
        learned_boosts = {row[0]: row[1] for row in policy_rows}
    except:
        learned_boosts = {}

    for pattern, tool, base_conf, target_skills in TOOL_TRIGGERS:
        if re.search(pattern, m, re.I):
            if tool not in seen_tools:
                conf = base_conf
                if tool in learned_boosts:
                    conf = min(0.99, conf * 0.7 + learned_boosts[tool] * 0.3)
                if skill in target_skills:
                    conf = min(0.99, conf + 0.05)
                decisions.append(ToolDecision(
                    tool=tool, args=_extract_args(msg, tool),
                    confidence=conf, reason=f"pattern match: {pattern[:40]}"))
                seen_tools.add(tool)

    if complexity == "hard" and "SEARCH" not in seen_tools:
        if any(w in m for w in ["best","compare","recommend","which","should I"]):
            decisions.append(ToolDecision("SEARCH", msg[:80], 0.7,
                "hard complexity comparison — search for current data", priority=8))

    return sorted(decisions, key=lambda d: d.confidence, reverse=True)[:3]

def _extract_args(msg: str, tool: str) -> str:
    if tool == "CALC":
        exprs = re.findall(r"[\d\.\+\-\*\/\^\(\)\s]+", msg)
        return max(exprs, key=len, default=msg[:40]).strip()
    if tool == "FETCH":
        urls = re.findall(r"https?://\S+", msg)
        return urls[0] if urls else ""
    if tool == "TIME":
        return ""
    return msg[:100]

def record_outcome(tool: str, trigger: str, skill: str,
                   success: bool, latency_ms: float, quality: float):
    try:
        c = _conn()
        c.execute("INSERT INTO tool_outcomes VALUES (NULL,?,?,?,?,?,?,?)",
            (tool, trigger[:80], skill, 1 if success else 0, latency_ms, quality, time.time()))
        sig = f"{tool}::{skill}"
        row = c.execute("SELECT success_rate, sample_count FROM tool_policy WHERE state_signature=?",
                        (sig,)).fetchone()
        if row:
            sr, n = row
            new_sr = (sr * n + (1.0 if success else 0.0)) / (n + 1)
            c.execute("UPDATE tool_policy SET success_rate=?, sample_count=?, confidence=?, updated=? WHERE state_signature=?",
                      (new_sr, n+1, new_sr * quality, time.time(), sig))
        else:
            c.execute("INSERT INTO tool_policy VALUES (?,?,?,?,?,?)",
                      (sig, tool, quality, 1.0 if success else 0.0, 1, time.time()))
        c.commit(); c.close()
    except Exception as e:
        print(f"[ToolPolicy] {e}")

def build_tool_injection(msg: str, skill: str, complexity: str) -> str:
    decisions = decide_tools(msg, skill, complexity)
    if not decisions:
        return ""
    lines = ["<tool_recommendations>"]
    for d in decisions:
        lines.append(f"  USE {d.tool}({d.args[:60]}) — confidence {d.confidence:.0%}: {d.reason}")
    lines.append("Execute these tools proactively. Do not describe what you would do — do it.")
    lines.append("</tool_recommendations>")
    return "\n".join(lines)
