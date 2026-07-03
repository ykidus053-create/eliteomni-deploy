import re, os
path = "app.py"
if not os.path.exists(path):
    print("❌ Error: app.py not found! Make sure you are in the eliteomni folder.")
    exit()

with open(path, "r") as f: content = f.read()

# 1. Fix Greedy Mode (_lc_kw)
lc_kw_new = """def _lc_kw(max_new: int, skill: str, msg_len: int) -> dict:
    return dict(
        max_tokens     = max_new,
        stop           = _STOPS,
        repeat_penalty = 1.1,
        temperature    = 0.0,
        top_k          = 1,
        top_p          = 1.0
    )"""
content = re.sub(r"def _lc_kw\(.*?\n    return kw", lc_kw_new, content, flags=re.DOTALL)

# 2. Double Budget (_budget)
budget_new = """def _budget(msg: str, skill: str, complexity: str) -> int:
    if skill == "coder": return 1024
    if skill == "researcher": return 800
    if complexity == "hard": return 600
    return 400"""
content = re.sub(r"def _budget\(.*?\n    return \d+", budget_new, content, flags=re.DOTALL)

# 3. Update System Prompt
prompt_new = """SYSTEM_PROMPT = \"\"\"
You are EliteOmni. You are a precise, factual AI.
MANDATORY ACCURACY RULES:
1. THINK STEP-BY-STEP before answering.
2. If you are not 100% sure, say "I don't have enough data" instead of guessing.
3. Use the [WEB SEARCH CONTEXT] as your primary source of truth.
4. If a calculation is needed, show your work.
\"\"\""""
content = re.sub(r"SYSTEM_PROMPT = \"\"\"(.*?)\"\"\"", prompt_new, content, flags=re.DOTALL)

# 4. Force Tree Search (mode = "think")
content = content.replace('mode = "think" if complexity=="hard" or skill in ("researcher","coder") else "fast"', 'mode = "think"')

with open(path, "w") as f: f.write(content)
print("✅ EliteOmni Accuracy Patch Applied Successfully!")
