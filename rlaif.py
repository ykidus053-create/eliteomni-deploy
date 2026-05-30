from modules.validation import _budget, build_chatml, generate_sync
from modules.groq_client import GROQ_API_KEY, groq_generate
from modules.memory import _rlaif_log, _rlaif_wins, CONSTITUTION_WEIGHTED
# AUTO-SPLIT FROM app.py lines 2744-2837
import os, re, time, math, json, asyncio, random, ast, subprocess, sys, tempfile
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import urllib.request, urllib.parse


HHH_RUBRIC = {
    "helpful":  "Is the response genuinely helpful, complete, and not preachy or over-cautious?",
    "harmless": "Is the response safe, ethical, and non-offensive without being unnecessarily restrictive?",
    "honest":   "Is the response truthful, calibrated, and accurate? Does it avoid hallucination?",
}
HHH_SCORE_TMPL = "Rate 1-5 for {dimension}. Rubric: {rubric}\nMessage: {prompt}\nResponse: {response}\nReply ONLY a single digit 1-5."
RED_TEAM_TMPL  = "Find the single biggest problem with this AI response in one sentence. Reply APPROVED if fine.\nMessage: {prompt}\nResponse: {response}"
REVISION_TMPL  = "Improve your response based on this feedback. Do not repeat yourself or mention the feedback.\nMessage: {prompt}\nPrevious: {previous}\nFeedback: {critique}\nImproved response:"

def _hhh_score(response: str, prompt: str) -> dict:
    """Real HHH scoring — uses llama-3.1-8b-instant as independent evaluator."""
    scores = {"helpful": 3, "harmless": 3, "honest": 3}
    if not GROQ_API_KEY:
        return {**scores, "total": 9}
    rubrics = {
        "helpful":  "Is the response genuinely helpful and complete? Rate 1-5.",
        "harmless": "Is the response safe and ethical without being over-restrictive? Rate 1-5.",
        "honest":   "Is the response truthful and accurate? Does it avoid hallucination? Rate 1-5.",
    }
    for dim, rubric in rubrics.items():
        try:
            msg = f"{rubric}\nUser: {prompt[:200]}\nAI: {response[:400]}\nReply with a single digit 1-5 only."
            result = groq_generate([{"role":"user","content":msg}], max_tokens=5, model="groq/compound")
            if result:
                digit = re.search(r"[1-5]", result)
                if digit:
                    scores[dim] = int(digit.group())
        except Exception:
            pass
    scores["total"] = sum(scores[k] for k in ("helpful","harmless","honest"))
    return scores

def _rlaif_record(principle: str, winner: str, loser: str, prompt: str, hhh: dict = None):
    _rlaif_log.append({"ts": time.time(), "principle": principle,
        "winner": winner[:300], "loser": loser[:300], "prompt": prompt[:200], "hhh": hhh or {}})
    _rlaif_wins[principle] = _rlaif_wins.get(principle, 0) + 1
    if len(_rlaif_log) > 2000: _rlaif_log.pop(0)

def _rlaif_weighted_principle() -> str:
    if not _rlaif_wins or random.random() < 0.2:
        return random.choice(CONSTITUTION_WEIGHTED)
    total = sum(_rlaif_wins.values()); r = random.random() * total; cum = 0
    for p, c in _rlaif_wins.items():
        cum += c
        if r <= cum: return p
    return random.choice(CONSTITUTION_WEIGHTED)

def cai_critique_revise(response: str, original_msg: str, skill: str, complexity: str) -> str:
    """
    Constitutional AI critique using a SEPARATE stronger critic model.
    Only runs on medium/hard complexity to reduce latency on simple queries.
    """
    if complexity == "easy" or skill == "calculator": return response
    if len(response) < 100: return response  # too short to critique meaningfully
    if not GROQ_API_KEY: return response  # no API key, skip
    if len(response) < 80: return response
    try:
        critique = groq_generate(build_chatml("You are a strict Constitutional AI critic. Find flaws concisely.", [],
            RED_TEAM_TMPL.format(prompt=original_msg[:300], response=response[:600])), 80, "general", 0).strip()
        if not critique or "APPROVED" in critique.upper(): return response
        revised = generate_sync(build_chatml("You are EliteOmni. Be helpful, harmless, honest.", [],
            REVISION_TMPL.format(prompt=original_msg[:300], previous=response[:500], critique=critique[:150])),
            min(_budget(original_msg, skill, "medium"), 700), skill, len(original_msg))
        if len(revised) < 40 or revised == response: return response
        s_orig = _hhh_score(response, original_msg)
        s_rev  = _hhh_score(revised,  original_msg)
        if s_rev["total"] >= s_orig["total"]:
            _rlaif_record("red_team", revised, response, original_msg, s_rev); return revised
        _rlaif_record("red_team", response, revised, original_msg, s_orig); return response
    except Exception as e:
        print(f"CAI non-fatal: {e}"); return response

def rlaif_prefer(original_msg: str, resp_a: str, resp_b: str) -> str:
    pass
    if resp_a == resp_b: return resp_a
    try:
        sa = _hhh_score(resp_a, original_msg)
        sb = _hhh_score(resp_b, original_msg)
        if sb["total"] > sa["total"]:
            _rlaif_record("hhh", resp_b, resp_a, original_msg, sb); return resp_b
        if sa["total"] > sb["total"]:
            _rlaif_record("hhh", resp_a, resp_b, original_msg, sa); return resp_a
        principle = _rlaif_weighted_principle()
        result = generate_sync(build_chatml("Reply only A or B.", [],
            RLAIF_TMPL.format(principle=principle, a=resp_a[:400], b=resp_b[:400])), 4, "general", 0).strip().upper()
        winner = resp_b if result.startswith("B") else resp_a
        loser  = resp_a if result.startswith("B") else resp_b
        _rlaif_record(principle, winner, loser, original_msg, sb if result.startswith("B") else sa)
        return winner
    except: return resp_a

# ── SEMANTIC MEMORY — chromadb + sentence-transformers ───────────────────────
