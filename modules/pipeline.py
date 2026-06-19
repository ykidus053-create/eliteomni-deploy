from __future__ import annotations
import re, time, threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from modules.model_router   import select_model, get_token_budget
from modules.llm_client     import stream as llm_stream, generate as llm_generate
from modules.memory_fast    import mem_get, episodic_get, mem_save
from modules.prompt_builder import build_system_prompt, build_chat_messages

_SKILL_KW = {
    "safety":     ["weapon","explosive","suicide","kill","hack","malware","jailbreak"],
    "researcher": ["research","explain","analyze","compare","essay","how does","why does",
                   "summarize","guide","tutorial","step by step","what is","tell me about"],
    "coder":      ["code","python","javascript","function","implement","debug","algorithm",
                   "script","html","css","react","bug","error","class","def ","import "],
    "calculator": ["calculate","compute","sqrt","equation","percent","%","times","plus",
                   "minus","divided","equals","how much","solve","multiply"],
}

def classify_skill(msg: str) -> str:
    m = msg.lower()
    if any(t in m for t in _SKILL_KW["safety"]):
        return "safety"
    scores = {s: sum(1 for t in kws if t in m)
              for s, kws in _SKILL_KW.items() if s != "safety"}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

_EASY_RE = re.compile(
    r"^(hi+|hey|hello|thanks|ok|yes|no|what.s the time|"
    r"who is|capital of|how many|square root of [\d.]+|[\d\s+\-*/().%^]+)$",
    re.IGNORECASE)
_HARD_RE = re.compile(
    r"(research|explain in detail|comprehensive|implement|algorithm|"
    r"architecture|step by step|essay|write a report|in depth|deep dive|"
    r"thoroughly|design system|optimize|refactor|dissertation)",
    re.IGNORECASE)

def route_complexity(msg: str) -> str:
    stripped = msg.strip()
    if len(stripped) <= 80 and _EASY_RE.search(stripped):
        return "easy"
    if len(stripped) >= 200 or _HARD_RE.search(stripped):
        return "hard"
    return "medium"

_SEARCH_RE = re.compile(
    r"(latest|current|today|news|price|weather|who is (the )?(ceo|president|"
    r"prime minister)|stock|2025|2026|recent|right now|happening|just released)",
    re.IGNORECASE)

def needs_search(msg: str) -> bool:
    return bool(_SEARCH_RE.search(msg))

def _do_search(msg: str) -> str:
    try:
        from modules.search import tool_search
        r = tool_search(msg[:300], use_searxng=True)
        if isinstance(r, str):
            return r[:2000]
        if isinstance(r, list):
            return "\n".join(
                f"[{x.get('title','')}] {x.get('snippet', x.get('body',''))}"
                for x in r[:5])[:2000]
    except Exception as e:
        print(f"[Pipeline] search: {e}")
    return ""

def get_rlhf_note(skill: str) -> str:
    try:
        import json, os
        fb = os.path.join(os.path.dirname(__file__), "feedback_store.json")
        if not os.path.exists(fb):
            return ""
        with open(fb) as f:
            data = json.load(f)
        s = data.get("feedback",{}).get(skill,{})
        if s.get("bad",0) > s.get("good",0):
            return f"Users rated {skill} responses poorly — be more precise."
        return ""
    except Exception:
        return ""

def fetch_context(msg: str, complexity: str) -> dict:
    ctx = {"memories":[],"search":"","episodic":[]}
    if complexity == "easy":
        return ctx
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_mem = ex.submit(mem_get, msg, 5)
        f_ep  = ex.submit(episodic_get, msg, 2)
        f_src = ex.submit(_do_search, msg) if needs_search(msg) else None
        try:
            ctx["memories"] = mem_get_semantic(msg, 5) or []
        except Exception:
            pass
        try:
            ctx["episodic"] = f_ep.result(timeout=5) or []
        except Exception:
            pass
        if f_src:
            try:
                ctx["search"] = f_src.result(timeout=5) or ""
            except Exception:
                pass
    return ctx

def build_context(msg: str, history: list, image_b64: str = "") -> dict:
    skill      = classify_skill(msg)
    complexity = route_complexity(msg)
    ctx        = fetch_context(msg, complexity)
    rlhf_note  = get_rlhf_note(skill)
    system     = build_system_prompt(
        skill=skill, complexity=complexity,
        memories=ctx["memories"]+ctx["episodic"],
        search_ctx=ctx["search"], rlhf_note=rlhf_note)
    msgs = build_chat_messages(system, history, msg, complexity)
    return {
        "skill": skill, "complexity": complexity, "msgs": msgs,
        "max_tokens": get_token_budget(complexity, msg),
        "model": select_model(complexity),
        "search_ctx": ctx["search"],
    }

def run_stream(msg: str, history: list, image_b64: str = "") -> Iterator:
    ctx = build_context(msg, history, image_b64)
    import json as _j
    yield _j.dumps({"skill":ctx["skill"],"mode":ctx["complexity"]}) + "\n"
    yield from llm_stream(
        ctx["msgs"], complexity=ctx["complexity"],
        max_tokens=ctx["max_tokens"], model=ctx["model"])
    def _save():
        try:
            mem_save(msg[:500], source="user", skill=ctx["skill"])
        except Exception:
            pass
    threading.Thread(target=_save, daemon=True).start()

def run_sync(msg: str, history: list) -> dict:
    ctx  = build_context(msg, history)
    resp = llm_generate(
        ctx["msgs"], complexity=ctx["complexity"],
        max_tokens=ctx["max_tokens"], model=ctx["model"])
    return {"response":resp,"skill":ctx["skill"],
            "model":ctx["model"],"complexity":ctx["complexity"]}

def generate_sync(msgs: list, max_tokens: int = 800,
                  skill: str = "general", msg_len: int = 0) -> str:
    complexity = "hard" if msg_len > 300 else "medium"
    return llm_generate(msgs, complexity=complexity, max_tokens=max_tokens)
