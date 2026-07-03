"""
Hallucination Guard — implements all 6 Huyen AI Engineering recommendations.
"""
import re

GROUNDING_PROMPT = """
## FACTUAL GROUNDING RULES
When search results are provided:
- ONLY state facts that appear in the search results
- Cite every factual claim with [1], [2] etc matching the source number
- If search results don't contain the answer say "I couldn't find current information on this"
- NEVER invent statistics, dates, names, or URLs

When no search results are provided:
- Prefix uncertain facts with "Based on my training data (which may be outdated)..."
- For anything time-sensitive always call SEARCH() first
- If you're less than 90% confident in a specific fact, say so explicitly
"""

COT_PROMPT = """
## REASONING BEFORE ANSWERING
1. Identify what type of information is needed
2. Check if search results cover it — if not, call SEARCH()
3. Reason from evidence to answer
4. State confidence at end: [Confidence: High/Medium/Low]
"""

FACTUAL_TRIGGERS = [
    "who is", "who are", "what is the current", "what is the latest",
    "how much does", "how many", "when did", "when was", "when is",
    "price of", "cost of", "stock", "weather", "news", "recently",
    "today", "this week", "this year", "2025", "2026",
    "ceo of", "president of", "founded", "headquartered",
]

def needs_tool_first(msg):
    m = msg.lower()
    return any(t in m for t in FACTUAL_TRIGGERS)

def self_consistency_check(msg, generate_fn, n=2):
    try:
        answers = []
        for _ in range(n):
            ans = generate_fn()
            if ans:
                answers.append(ans.strip())
        if len(answers) < 2:
            return answers[0] if answers else ""
        words_a = set(re.findall(r'\w+', answers[0].lower()))
        words_b = set(re.findall(r'\w+', answers[1].lower()))
        if not words_a or not words_b:
            return answers[0]
        overlap = len(words_a & words_b) / len(words_a | words_b)
        if overlap < 0.6:
            print("[HallucinationGuard] self-consistency divergence: " + str(round(overlap, 2)))
            return answers[0] + "\n\n*Note: I got different answers on this question — please verify with an authoritative source.*"
        return answers[0]
    except Exception as e:
        print("[HallucinationGuard] self-consistency error: " + str(e))
        return ""

def focus_search_results(results, query, top_k=3):
    if not results:
        return []
    query_words = set(re.findall(r'\w+', query.lower()))
    def _score(r):
        text = (r.get("title","") + " " + r.get("content", r.get("snippet",""))).lower()
        text_words = set(re.findall(r'\w+', text))
        return len(query_words & text_words) / max(len(query_words), 1)
    return sorted(results, key=_score, reverse=True)[:top_k]

def format_grounded_context(results):
    if not results:
        return ""
    lines = ["## Search Results (cite as [1], [2], etc.)"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Source")
        url = r.get("url", "")
        snippet = r.get("content", r.get("snippet", ""))[:400]
        lines.append("[" + str(i) + "] **" + title + "**\n" + snippet + "\nURL: " + url)
    return "\n\n".join(lines)

def build_hallucination_guard_prompt(msg, search_results, skill, complexity):
    parts = [GROUNDING_PROMPT]
    if complexity == "hard" or skill in ("researcher", "general"):
        parts.append(COT_PROMPT)
    if needs_tool_first(msg):
        parts.append("\nThis query needs current info — call SEARCH() before answering if you haven't already.")
    return "\n".join(parts)
