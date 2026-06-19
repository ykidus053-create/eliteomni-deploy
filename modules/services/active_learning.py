
"""
Andrew Ng: The training loop IS the product.
Every conversation should make the system measurably better.
This module reads accumulated feedback and distills it into
actionable system prompt improvements — closing the loop.
"""
import json
import os
import re
from collections import Counter

FEEDBACK_INSIGHTS_PATH = os.path.expanduser("~/.eliteomni_insights.json")

def analyze_failure_patterns(feedback_data: list) -> dict:
    """
    Read all negative feedback and extract patterns.
    What types of questions does the system consistently fail?
    """
    failures = [f for f in feedback_data if f.get("rating", 5) <= 2]
    if not failures:
        return {"patterns": [], "top_failure_type": None}

    failure_texts = [f.get("question", "") + " " + f.get("response", "")
                     for f in failures]

    # Count failure categories
    categories = Counter()
    for text in failure_texts:
        t = text.lower()
        if any(w in t for w in ["code", "function", "bug", "error", "python"]):
            categories["coding"] += 1
        if any(w in t for w in ["calculate", "math", "number", "%", "+"]):
            categories["math"] += 1
        if any(w in t for w in ["explain", "what is", "how does", "why"]):
            categories["explanation"] += 1
        if any(w in t for w in ["write", "draft", "create", "generate"]):
            categories["generation"] += 1
        if any(w in t for w in ["research", "find", "search", "latest"]):
            categories["research"] += 1

    top = categories.most_common(3)
    return {
        "total_failures": len(failures),
        "patterns": top,
        "top_failure_type": top[0][0] if top else None,
        "failure_rate": len(failures) / max(len(feedback_data), 1),
    }


def extract_improvement_signals(sft_demos: list) -> list:
    """
    From SFT demos (human-preferred responses), extract
    what made them better — so we can inject that into prompts.
    """
    insights = []
    for demo in sft_demos[:20]:
        good = demo.get("good_response", "")
        bad  = demo.get("bad_response", "")
        if not good or not bad:
            continue

        # What does the good response have that the bad one doesn't?
        good_has_code   = "```" in good and "```" not in bad
        good_has_steps  = bool(re.search(r"\d+\.", good)) and not bool(re.search(r"\d+\.", bad))
        good_is_shorter = len(good) < len(bad) * 0.7
        good_is_longer  = len(good) > len(bad) * 1.5
        good_hedges     = any(w in good.lower() for w in ["likely", "probably", "i think"])
        bad_overconf    = any(w in bad.lower() for w in ["definitely", "certainly", "always"])

        if good_has_code:
            insights.append("Include code examples when explaining technical concepts")
        if good_has_steps:
            insights.append("Break complex answers into numbered steps")
        if good_is_shorter:
            insights.append("Be more concise — brevity is preferred")
        if good_is_longer:
            insights.append("Provide more detail and depth")
        if good_hedges and bad_overconf:
            insights.append("Express appropriate uncertainty instead of overconfidence")

    # Deduplicate
    return list(dict.fromkeys(insights))


def generate_system_prompt_patch(insights: list,
                                 failure_analysis: dict) -> str:
    """
    Turn learned insights into a system prompt patch.
    This is the actual learning — updating behavior based on data.
    """
    patch_lines = []

    if failure_analysis.get("top_failure_type") == "coding":
        patch_lines.append(
            "When answering coding questions: always include runnable code, "
            "explain the key lines, and mention edge cases."
        )
    if failure_analysis.get("top_failure_type") == "math":
        patch_lines.append(
            "When doing calculations: show your work step by step, "
            "double-check the arithmetic, express uncertainty if needed."
        )
    if failure_analysis.get("failure_rate", 0) > 0.3:
        patch_lines.append(
            "Recent feedback suggests responses have been unclear. "
            "Prioritize clarity over completeness."
        )

    for insight in insights[:5]:
        patch_lines.append(insight)

    if not patch_lines:
        return ""

    return (
        "[LEARNED IMPROVEMENTS — from user feedback]\n"
        + "\n".join(f"- {p}" for p in patch_lines)
    )


def run_learning_cycle(db_path: str = None) -> str:
    """
    Full learning cycle:
    1. Load feedback from DB
    2. Analyze failure patterns
    3. Extract improvement signals
    4. Generate system prompt patch
    5. Save insights
    Returns the system prompt patch string.
    """
    feedback_data = []
    sft_demos = []

    # Load from EliteOmni DB
    try:
        import sqlite3
        db = db_path or os.path.expanduser("~/eliteomni_memory.db")
        conn = sqlite3.connect(db)
        cur = conn.cursor()

        # Load ratings
        try:
            cur.execute("SELECT skill, rating, question, response FROM feedback ORDER BY id DESC LIMIT 100")
            for row in cur.fetchall():
                feedback_data.append({
                    "skill": row[0], "rating": row[1],
                    "question": row[2] or "", "response": row[3] or ""
                })
        except Exception:
            pass

        # Load SFT demos
        try:
            cur.execute("SELECT question, good_response, bad_response FROM sft_demos LIMIT 50")
            for row in cur.fetchall():
                sft_demos.append({
                    "question": row[0], "good_response": row[1] or "",
                    "bad_response": row[2] or ""
                })
        except Exception:
            pass

        conn.close()
    except Exception as e:
        print(f"[ActiveLearn] DB load failed: {e}")

    if not feedback_data and not sft_demos:
        return ""

    failure_analysis = analyze_failure_patterns(feedback_data)
    insights = extract_improvement_signals(sft_demos)
    patch = generate_system_prompt_patch(insights, failure_analysis)

    # Save insights
    try:
        json.dump({
            "failure_analysis": failure_analysis,
            "insights": insights,
            "patch": patch,
        }, open(FEEDBACK_INSIGHTS_PATH, "w"), indent=2)
    except Exception:
        pass

    if failure_analysis["patterns"]:
        print(f"[ActiveLearn] Top failure: {failure_analysis['top_failure_type']} "
              f"({failure_analysis['failure_rate']:.0%} failure rate)")
    if insights:
        print(f"[ActiveLearn] {len(insights)} improvements extracted from demos")

    return patch
