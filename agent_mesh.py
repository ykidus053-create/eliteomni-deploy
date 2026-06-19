import re, time, json, os, ast, subprocess, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

CHAR_LEVEL_AUDIT_PROMPT = """
<char_level_audit>
Before outputting, scan character by character for:
- Unclosed brackets, parens, braces, quotes
- Off-by-one in loop ranges
- Variable used before assignment
- Return inside loop when it should be outside
Fix any found before responding.
</char_level_audit>
"""

SELF_AUDIT_PATCH = """
<self_audit>
FINAL CHECK before output:
- Does my answer actually solve what was asked?
- Did I complete every sentence and code block?
- Are all variable names consistent throughout?
- Would this run without modification?
If any box fails, fix it now, do not output.
</self_audit>
"""

def inject_cot(system, skill, complexity, msg):
    if complexity == "easy":
        return system
    cot = {
        "coder": "\nBefore code: state invariant, trace one example, check edge cases.",
        "calculator": "\nPATH A rough estimate, PATH B precise, PATH C verify.",
        "researcher": "\nDECOMPOSE, SYNTHESIZE, VERIFY each claim, SUMMARIZE.",
    }.get(skill, "\nThink step by step. State assumptions. Verify before answering.")
    return system + cot

def strip_reasoning_artifacts(text):
    for label in ['INTENT','AMBIGUITY','APPROACH','CONSTRAINTS','PLAN','SELF-CHECK','CORRECTION']:
        text = re.sub(rf'{label}:.*?\n', '', text)
    text = re.sub(r'DRAFT:.*?\n', '', text, flags=re.DOTALL)
    return text.strip()

def cot_complexity_gate(msg, complexity):
    return complexity in ("medium", "hard") and len(msg) > 30

def scan_for_errors(response, skill):
    errors = []
    if skill == "coder":
        if "```" not in response and any(w in response.lower() for w in ["implement","write","create","build"]):
            errors.append("coder_no_code_block")
        if re.search(r'\bTODO\b|\bFIXME\b|\bpass\b\s*#', response):
            errors.append("coder_has_placeholder")
        if re.search(r'except\s*:', response) and "except Exception" not in response:
            errors.append("coder_bare_except")
    if skill == "calculator":
        if not re.search(r'\d', response):
            errors.append("calc_no_number")
    if re.search(r'According to \w+,', response) and "[WEB" not in response:
        errors.append("fake_citation")
    return errors

def record_error(error, skill):
    try:
        import sqlite3
        db = os.path.expanduser("~/eliteomni_errors.db")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE IF NOT EXISTS error_log (ts REAL, skill TEXT, error TEXT)")
        con.execute("INSERT INTO error_log VALUES (?,?,?)", (time.time(), skill, error))
        con.commit(); con.close()
    except Exception:
        pass

def get_error_warnings(skill):
    try:
        import sqlite3
        db = os.path.expanduser("~/eliteomni_errors.db")
        con = sqlite3.connect(db)
        rows = con.execute(
            "SELECT error, COUNT(*) as n FROM error_log WHERE skill=? "
            "GROUP BY error HAVING n >= 3 ORDER BY n DESC LIMIT 3", (skill,)
        ).fetchall()
        con.close()
        if not rows: return ""
        return "[LEARNED WARNINGS]\n" + "\n".join(f"Avoid: {r[0]} (seen {r[1]}x)" for r in rows)
    except Exception:
        return ""

def post_process_check(response, skill):
    if "[WEB" not in response and "SEARCH(" not in response:
        response = re.sub(r"\[\d+\]", "", response)
    response = strip_reasoning_artifacts(response)
    response = re.sub(r'\n{4,}', '\n\n\n', response)
    return response.strip()

def iterative_code_fix(code, task, generate_fn=None, max_rounds=5):
    result = {"code": code, "passed": False, "rounds": 0, "error": ""}
    blocked = re.compile(r'\b(os\.system|subprocess|shutil\.rmtree|socket|__import__)\b', re.IGNORECASE)

    for round_num in range(max_rounds):
        result["rounds"] = round_num + 1
        try:
            ast.parse(code)
        except SyntaxError as e:
            result["error"] = f"SyntaxError line {e.lineno}: {e.msg}"
            if generate_fn and round_num < max_rounds - 1:
                fix = generate_fn([{"role":"user","content":f"Fix this SyntaxError in the code.\nError: {result['error']}\nCode:\n{code}\nReturn only corrected code:"}], max_tokens=1000)
                if fix:
                    code = re.sub(r'^```python\n|^```\n|```$', '', fix.strip(), flags=re.MULTILINE).strip()
                    result["code"] = code
                    continue
            return result

        if blocked.search(code):
            result["error"] = "blocked: restricted operation"
            return result

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code); tmp = f.name
            proc = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=30)
            os.unlink(tmp)
            output = (proc.stdout + proc.stderr).strip()

            if proc.returncode == 0:
                result["passed"] = True
                result["code"] = code
                result["error"] = ""
                return result
            else:
                result["error"] = output[:400]
                if generate_fn and round_num < max_rounds - 1:
                    fix = generate_fn([{"role":"user","content":f"Fix this runtime error.\nError: {result['error']}\nTask: {task}\nCode:\n{code}\nReturn only corrected code:"}], max_tokens=1000)
                    if fix:
                        code = re.sub(r'^```python\n|^```\n|```$', '', fix.strip(), flags=re.MULTILINE).strip()
                        result["code"] = code
        except subprocess.TimeoutExpired:
            result["error"] = "timeout"
            return result
        except Exception as e:
            result["error"] = str(e)
            return result

    return result

def hierarchical_summarize(text, target_length=800, domain="general"):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(text) <= target_length:
        return text
    words_all = set(re.findall(r'\b\w{4,}\b', text.lower()))
    scored = []
    for i, sent in enumerate(sentences):
        if len(sent) < 10:
            continue
        words_sent = set(re.findall(r'\b\w{4,}\b', sent.lower()))
        density = len(words_sent & words_all) / max(len(words_sent), 1)
        position_boost = 0.3 if i < 3 else (0.2 if i >= len(sentences) - 3 else 0)
        scored.append((density + position_boost, sent))
    scored.sort(key=lambda x: -x[0])
    result, chars = [], 0
    for _, sent in scored:
        if chars + len(sent) > target_length:
            break
        result.append(sent)
        chars += len(sent)
    return " ".join(result) if result else text[:target_length]

def _run_critique(response, skill):
    try:
        from modules.core.http_client import mistral_generate
        prompt = [{"role":"user","content":f"Critique this {skill} response in one sentence. Reply APPROVED if fine, or ISSUE: [problem].\n\n{response[:600]}"}]
        result = mistral_generate(prompt, max_tokens=60)
        if result and "APPROVED" not in result.upper():
            return result.strip()
    except ImportError:
        pass
    except Exception:
        pass
    return ""

def prm_check_and_fix(response, msg, skill, generate_fn):
    steps = re.split(r'\n(?=\d+\.\s|Step\s*\d|##\s)', response)
    steps = [s.strip() for s in steps if len(s.strip()) > 40][:6]
    if len(steps) < 2:
        return response
    weak_steps = []
    for i, step in enumerate(steps):
        try:
            raw = generate_fn([{"role":"user","content":f"Rate this reasoning step 1-5 for correctness.\nQuestion: {msg[:100]}\nStep: {step[:200]}\nReply ONLY a digit 1-5:"}], max_tokens=3)
            d = re.search(r'[1-5]', raw or "3")
            score = int(d.group()) if d else 3
            if score <= 2:
                weak_steps.append((i, step, score))
        except Exception:
            pass
    if not weak_steps:
        return response
    weak_text = "\n".join(f"Step {i+1} (score {s}/5): {step[:150]}" for i, step, s in weak_steps)
    fix_prompt = [{"role":"user","content":f"These reasoning steps scored poorly:\n{weak_text}\n\nOriginal question: {msg[:200]}\nRewrite ONLY these steps with correct reasoning. Keep everything else identical. Full response:"}]
    try:
        fixed = generate_fn(fix_prompt, max_tokens=len(response) + 200)
        if fixed and len(fixed) > len(response) * 0.5:
            print(f"[PRM] fixed {len(weak_steps)} weak step(s)")
            return fixed
    except Exception:
        pass
    return response

def enforce_thinking(response, msg, skill, complexity, generate_fn):
    if complexity != "hard" or "<think>" in response or len(response) < 100:
        return response
    try:
        verdict = generate_fn([{"role":"user","content":f"Does this response correctly answer the question? Reply YES or NO: [one sentence reason]\n\nQuestion: {msg[:200]}\nResponse: {response[:500]}"}], max_tokens=40)
        if verdict and verdict.strip().upper().startswith("NO"):
            reason = verdict.strip()[3:].strip()
            fix = generate_fn([{"role":"user","content":f"Your response had this issue: {reason}\nQuestion: {msg[:200]}\nWrite the corrected response:"}], max_tokens=len(response) + 200)
            if fix and len(fix) > 80:
                print(f"[Thinking] self-corrected: {reason[:60]}")
                return fix
    except Exception:
        pass
    return response

AGENT_ROLES = {
    "researcher": {
        "system": "You are the Research Agent. Find facts, verify claims, provide context. Mark claims [VERIFIED] or [UNCERTAIN]. Be concise — you are feeding another agent, not the user.",
        "max_tokens": 600,
    },
    "critic": {
        "system": "You are the Critic Agent. Find the single biggest flaw in the draft: factual error, missing edge case, wrong assumption, or incomplete implementation. Reply: ISSUE: [one sentence] or APPROVED.",
        "max_tokens": 80,
    },
    "executor": {
        "system": "You are the Execution Agent. Extract any code, run it mentally step by step, and report: PASSES or FAILS: [reason].",
        "max_tokens": 100,
    },
    "synthesizer": {
        "system": "You are the Synthesis Agent. You receive: original question, draft response, research context, and critique. Produce the final polished response incorporating all feedback.",
        "max_tokens": 2000,
    },
}

class AgentMesh:
    def __init__(self, generate_fn):
        self._gen = generate_fn
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="agent")

    def _call(self, role, user_content):
        config = AGENT_ROLES[role]
        try:
            return self._gen(
                [{"role":"system","content":config["system"]},
                 {"role":"user","content":user_content}],
                max_tokens=config["max_tokens"]
            ) or ""
        except Exception as e:
            print(f"[AgentMesh] {role} error: {e}")
            return ""

    def run(self, msg, draft, skill, complexity, search_ctx=""):
        if complexity == "easy":
            return draft
        if complexity == "medium":
            return self._medium_pipeline(msg, draft, skill)
        return self._hard_pipeline(msg, draft, skill, search_ctx)

    def _medium_pipeline(self, msg, draft, skill):
        critique = self._call("critic", f"Question: {msg[:200]}\nDraft response:\n{draft[:800]}")
        if not critique or "APPROVED" in critique.upper():
            return draft
        issue = critique.replace("ISSUE:", "").strip()
        print(f"[AgentMesh:medium] critic flagged: {issue[:60]}")
        fixed = self._gen(
            [{"role":"system","content":"Fix the specific issue in this response. Keep everything else identical."},
             {"role":"user","content":f"Issue: {issue}\nQuestion: {msg[:200]}\nResponse to fix:\n{draft[:1200]}\nFixed response:"}],
            max_tokens=len(draft) + 300
        )
        return fixed if fixed and len(fixed) > len(draft) * 0.4 else draft

    def _hard_pipeline(self, msg, draft, skill, search_ctx):
        t0 = time.time()
        research_future = self._pool.submit(self._call, "researcher",
            f"Provide key facts and context for: {msg[:400]}"
            + (f"\n\nSearch context:\n{search_ctx[:600]}" if search_ctx else ""))
        critic_future = self._pool.submit(self._call, "critic",
            f"Question: {msg[:200]}\nDraft:\n{draft[:800]}")
        research_ctx = research_future.result(timeout=20) or ""
        critique = critic_future.result(timeout=20) or ""
        exec_result = ""
        if skill == "coder" and "```" in draft:
            import re, subprocess, tempfile, os
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", draft, re.DOTALL)
            for block in code_blocks[:1]:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                        f.write(block)
                        fname = f.name
                    r = subprocess.run(["python", fname], capture_output=True, text=True, timeout=30)
                    os.unlink(fname)
                    if r.returncode != 0:
                        exec_result = f"FAILS: {r.stderr[:400]}"
                    else:
                        # Check for stub patterns even if code runs
                        import re as _re
                        stubs = _re.findall(r'#\s*In real implementation.*', block)
                        passes = _re.findall(r'print\(f"Replicat', block)
                        if stubs or passes:
                            exec_result = f"FAILS: {len(stubs)} stub(s) found that say 'In real implementation' — these must be replaced with actual socket/HTTP calls: {stubs[:2]}"
                        else:
                            exec_result = "PASSES"
                except Exception as e:
                    exec_result = f"FAILS: {str(e)}"
        has_issue = critique and "APPROVED" not in critique.upper()
        has_exec_fail = exec_result and "FAILS" in exec_result.upper()
        if not has_issue and not has_exec_fail and not research_ctx and skill != "coder":
            print(f"[AgentMesh:hard] all approved in {int((time.time()-t0)*1000)}ms")
            return draft
        synth_input = (
            f"ORIGINAL QUESTION:\n{msg[:400]}\n\n"
            f"DRAFT RESPONSE:\n{draft[:1200]}\n\n"
            + (f"RESEARCH CONTEXT:\n{research_ctx[:400]}\n\n" if research_ctx else "")
            + (f"CRITIC FLAGGED:\n{critique}\n\n" if has_issue else "")
            + (f"EXECUTION CHECK:\n{exec_result}\n\n" if exec_result else "")
            + "Produce the final response incorporating the above feedback."
        )
        final = self._call("synthesizer", synth_input)
        print(f"[AgentMesh:hard] done in {int((time.time()-t0)*1000)}ms | critique={'flagged' if has_issue else 'ok'} | exec={'fail' if has_exec_fail else 'ok'}")
        final = final if final and len(final) > len(draft) * 0.3 else draft
        if skill == "coder":
            from reflexion_loop import reflexion_verify
            def _gen_fn(prompt, model=""):
                return self._call("synthesizer", prompt) or final
            final = reflexion_verify(final, _gen_fn, model="", max_rounds=5)
        # ── Self-eval gate (Anthropic-style quality check) ──
        _score = self_eval_score(final, msg, self._gen)
        print(f"[self_eval] score={_score:.2f}")
        if _score < 0.7:
            print(f"[self_eval] below threshold — forcing rewrite")
            _rewrite = self._call("synthesizer",
                f"Your previous response scored {_score:.2f}/1.0 on quality. "
                f"Rewrite with complete implementations, no stubs, no placeholders.\n"
                f"Task: {msg[:300]}\nPrevious attempt:\n{final[:2000]}")
            if _rewrite and len(_rewrite) > len(final) * 0.3:
                final = _rewrite

        # ── Repeat loop: run until PASSES ──
        _max_iters = 5
        for _iter in range(_max_iters):
            import re, subprocess, tempfile, os
            _blocks = re.findall(r"```(?:python)?\n(.*?)```", final, re.DOTALL)
            if not _blocks: break
            _block = _blocks[0]
            try:
                with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as _f:
                    _f.write(_block); _fname = _f.name
                _r = subprocess.run(["python", _fname], capture_output=True, text=True, timeout=30)
                os.unlink(_fname)
                _stubs = re.findall(r"# In real implementation.*", _block)
                if _r.returncode == 0 and not _stubs:
                    print(f"[loop] PASSES iter {_iter+1}"); break
                _err = _r.stderr[:300] if _r.returncode != 0 else f"{len(_stubs)} stubs found"
                print(f"[loop] iter {_iter+1} FAILS: {_err[:80]}")
                _fix = self._call("synthesizer", f"Fix this code error. No stubs. Real code only.\nERROR: {_err}\nCODE:\n{final[:2000]}")
                if _fix and len(_fix) > len(final) * 0.3: final = _fix
            except Exception as _le: print(f"[loop] error: {_le}"); break
        return final

def route(msg):
    m = msg.lower()
    SAFETY_TRIGGERS = ["synthesize nerve","weaponize","bioweapon","sarin","ricin","jailbreak","ignore all instructions","kill myself","suicide method"]
    for danger in SAFETY_TRIGGERS:
        if danger in m:
            return "safety", 1.0, {"complexity": "easy"}
    SKILL_KEYWORDS = [
        ("implement","coder",3.0),("refactor","coder",2.5),("debug","coder",3.0),
        ("write a function","coder",3.5),("write a class","coder",3.5),
        ("write a script","coder",3.5),("python code","coder",3.0),
        ("javascript","coder",2.5),("sql query","coder",2.5),
        ("calculate","calculator",3.0),("compute","calculator",2.5),
        ("sqrt","calculator",3.0),("percent of","calculator",3.0),
        ("divided by","calculator",2.5),("solve for","calculator",3.0),
        ("research","researcher",2.5),("analyze","researcher",2.5),
        ("compare","researcher",2.0),("history of","researcher",2.5),
        ("explain in detail","researcher",2.5),("pros and cons","researcher",2.0),
        ("how does","researcher",1.5),("why does","researcher",1.5),
    ]
    from collections import defaultdict
    scores = defaultdict(float)
    for keyword, skill, weight in SKILL_KEYWORDS:
        if keyword in m:
            scores[skill] += weight
    best_skill = max(scores, key=scores.get) if scores else "general"
    best_score = scores.get(best_skill, 0.0)
    if best_score < 1.5:
        best_skill = "general"
    confidence = min(0.95, best_score / (best_score + 3.0)) if best_score > 0 else 0.3
    EASY_WORDS = ["hi ","hey ","hello","thanks","yes","no","capital of","2+2","what time","who is"]
    HARD_WORDS = ["research","comprehensive","analyze","implement","algorithm","step by step","essay","explain in detail","design","architecture","distributed","scheduler"]
    if any(w in m for w in EASY_WORDS) and len(msg) < 100:
        complexity = "easy"
    elif len(msg) > 200 or any(w in m for w in HARD_WORDS):
        complexity = "hard"
    else:
        complexity = "medium"
    if best_skill == "coder":
        complexity = "hard"
    return best_skill, confidence, {"complexity": complexity}

def goal_detect_and_save(msg, session_id="default"):
    patterns = [
        r"(?:I want to|I need to|I am trying to|my goal is to|help me)\s+([^.!?]{10,80})",
        r"(?:working on|building|creating|developing)\s+([^.!?]{10,80})",
        r"(?:I am learning|studying|preparing for)\s+([^.!?]{10,80})",
    ]
    signal = None
    for p in patterns:
        match = re.search(p, msg, re.IGNORECASE)
        if match:
            signal = match.group(1).strip()[:100]
            break
    if not signal:
        return None
    try:
        import sqlite3
        db = os.path.expanduser("~/eliteomni_goals.db")
        goal_id = f"g_{int(time.time() * 1000)}"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE IF NOT EXISTS goals (id TEXT PRIMARY KEY, description TEXT, session_id TEXT, status TEXT DEFAULT 'active', created_ts REAL)")
        con.execute("INSERT OR IGNORE INTO goals VALUES (?,?,?,?,?)", (goal_id, signal, session_id, "active", time.time()))
        con.commit(); con.close()
        return goal_id
    except Exception:
        return None

def production_generate(msg, skill, complexity, history, search_ctx, generate_fn, max_tokens=2000):
    t0 = time.time()
    try:
        draft = generate_fn(history + [{"role":"user","content":msg}], max_tokens=max_tokens)
    except Exception as e:
        return f"[Generation error: {e}]"
    if not draft:
        return ""
    if complexity in ("medium","hard") and skill in ("coder","researcher","calculator"):
        draft = prm_check_and_fix(draft, msg, skill, generate_fn)
    if complexity == "hard":
        draft = enforce_thinking(draft, msg, skill, complexity, generate_fn)
    mesh = AgentMesh(generate_fn)
    final = mesh.run(msg, draft, skill, complexity, search_ctx)
    errors = scan_for_errors(final, skill)
    for err in errors:
        record_error(err, skill)
    final = post_process_check(final, skill)
    print(f"[production_generate] skill={skill} complexity={complexity} latency={int((time.time()-t0)*1000)}ms errors={errors}")
    return final

# ── Reflexion patch for _hard_pipeline ──

# ── Self-eval scorer (Anthropic-style quality gate) ──────────────────────────
def self_eval_score(code: str, task: str, generate_fn) -> float:
    """
    Agent scores its own output 0.0-1.0 before returning.
    Below 0.7 triggers another rewrite round.
    Mirrors Anthropic's internal quality gate on Claude Code outputs.
    """
    prompt = f"""Score this code implementation from 0.0 to 1.0.

TASK: {task[:300]}

CODE:
{code[:2000]}

Score criteria:
- 1.0: All requirements implemented, no stubs, no 'In real implementation', runnable
- 0.7: Most requirements met, minor gaps
- 0.4: Structural skeleton only, stubs present
- 0.1: Placeholder code, nothing implemented

Respond with ONLY a float like: 0.8
Score:"""
    try:
        result = generate_fn([{"role": "user", "content": prompt}], max_tokens=10)
        import re
        match = re.search(r'0\.\d+|1\.0', result or "")
        return float(match.group()) if match else 0.5
    except:
        return 0.5
