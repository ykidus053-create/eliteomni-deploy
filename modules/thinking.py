"""EliteOmni Thinking Layer — Part 1"""
import re as _re, time, json, hashlib, threading
from collections import Counter

_clp = threading.Lock()
_vc = {}

def hidden_cot(gen, msgs, complexity="medium"):
    if complexity == "easy":
        a = gen(msgs)
        return (a if isinstance(a, str) else "".join(a)), ""
    umsg = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
    cp = ("Before answering, think step by step inside <thinking></thinking> tags. "
          "Consider: What is being asked? Key components? What could go wrong? "
          "Assumptions? Most precise answer? Then answer outside the tags.\n\n" + umsg)
    cm = []
    for m in msgs:
        if m.get("role") == "user":
            cm.append({"role":"user","content":cp})
            break
        cm.append(m)
    try:
        raw = gen(cm)
        rt = raw if isinstance(raw, str) else "".join(raw)
    except Exception:
        return ("", "")
    tm = _re.search(r"<thinking>(.*?)</thinking>", rt, _re.DOTALL)
    thinking = tm.group(1).strip() if tm else ""
    answer = rt[tm.end():].strip() if tm else rt
    if not answer:
        answer = rt
    if complexity in ("hard","reasoning") and thinking and len(thinking) > 50:
        cm2 = []
        for m in msgs:
            cm2.append(m)
            if m.get("role") == "user":
                break
        cm2.append({"role":"user","content":"[Internal reasoning: " + thinking[:1500] + "]\n\nBased on that, give a direct precise answer to: " + umsg})
        try:
            ca = gen(cm2)
            cj = ca if isinstance(ca, str) else "".join(ca)
            if len(cj) > len(answer) * 0.3:
                answer = cj
        except Exception:
            pass
    return (answer, thinking)

def _norm(text):
    text = _re.sub(r"```.*?```","",text,flags=_re.DOTALL)
    text = _re.sub(r"\*{1,2}","",text)
    text = _re.sub(r"#+\s*","",text)
    text = text.strip().lower()
    def _nm(m):
        try:
            n = float(m.group(0))
            return str(int(n)) if n == int(n) else str(round(n, 4))
        except Exception:
            return m.group(0)
    text = _re.sub(r"-?\d+\.?\d*", _nm, text)
    fl = text.split("\n")[0].strip()
    fs = _re.split(r"[.!?]", fl)[0].strip()
    return fs if len(fs) > 5 else fl

def self_consistency(gen, msgs, n=3, complexity="medium"):
    umsg = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"),"").lower()
    if any(s in umsg for s in ["write a","story","poem","essay","opinion","what do you think","imagine"]):
        a = gen(msgs)
        return (a if isinstance(a,str) else "".join(a)), [], 1.0
    an = {"easy":1,"medium":3,"hard":5,"reasoning":5}.get(complexity, n)
    if an <= 1:
        a = gen(msgs)
        return (a if isinstance(a,str) else "".join(a)), [], 1.0
    ck = hashlib.md5((str(msgs) + str(an)).encode()).hexdigest()[:12]
    with _clp:
        if ck in _vc and time.time() - _vc[ck]["ts"] < 300:
            c = _vc[ck]
            return c["b"], c["a"], c["r"]
    notes = ["Answer directly and precisely.",
             "Think carefully before answering. Be exact.",
             "Consider the question thoroughly. Give the most accurate answer.",
             "Work through this methodically.",
             "Be precise and specific in your answer."]
    answers = []
    for i in range(an):
        try:
            vm = []
            for m in msgs:
                if m.get("role") == "system":
                    vm.append({**m, "content": m.get("content","") + "\n" + notes[i % len(notes)]})
                else:
                    vm.append(m)
            raw = gen(vm)
            a = raw if isinstance(raw,str) else "".join(raw)
            if a and len(a) > 5:
                answers.append(a)
        except Exception as e:
            print("[SelfCons] run " + str(i+1) + " failed: " + str(e))
    if not answers:
        return ("", [], 0.0)
    if len(answers) == 1:
        return (answers[0], answers, 1.0)
    norms = [_norm(a) for a in answers]
    counts = Counter(norms)
    mcn, vc = counts.most_common(1)[0]
    agr = vc / len(answers)
    bi = norms.index(mcn)
    best = answers[bi]
    if agr < 0.5 and len(answers) >= 3:
        best += "\n\n> *Note: low confidence (" + str(round(agr*100)) + "% agreement across " + str(len(answers)) + " attempts)*"
    with _clp:
        if len(_vc) > 200:
            oldest = min(_vc, key=lambda k: _vc[k]["ts"])
            del _vc[oldest]
        _vc[ck] = {"b": best, "a": answers, "r": agr, "ts": time.time()}
    return (best, answers, agr)
"""EliteOmni Thinking Layer — Part 2"""

def progressive_refine(gen, msgs, skill="general", complexity="medium", max_rounds=2):
    if complexity == "easy":
        a = gen(msgs)
        return (a if isinstance(a,str) else "".join(a)), []
    umsg = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
    draft = gen(msgs)
    cur = draft if isinstance(draft,str) else "".join(draft)
    crits = []
    ct = {
        "coder": "Review this code for LOGIC bugs (not syntax). 1) Off-by-one errors 2) Missing edge cases (empty,None,negative,concurrent) 3) Wrong algorithm 4) Missing I/O error handling. If correct reply EXACTLY: APPROVED. If issues: ISSUE: [desc] FIX: [what to change]",
        "researcher": "Fact-check this: 1) Unsupported claims? 2) Wrong dates/names/numbers? 3) Unhedged speculation? 4) Contradictions? If accurate reply EXACTLY: APPROVED. If issues: ISSUE: [claim] CORRECTION: [right answer]",
        "general": "Review: 1) Does it answer what was asked? 2) Anything missing? 3) Logic gaps? 4) Incomplete? If complete reply EXACTLY: APPROVED. If issues: ISSUE: [desc] FIX: [what to add]"
    }
    for rn in range(max_rounds):
        cm = [{"role":"system","content":"You are a strict reviewer. Be thorough but fair."},
              {"role":"user","content":"Question: " + umsg[:500] + "\n\nAnswer:\n" + cur[:2000] + "\n\n" + ct.get(skill, ct["general"])}]
        try:
            cr = gen(cm)
            critique = cr if isinstance(cr,str) else "".join(cr)
        except Exception:
            break
        crits.append(critique)
        if "APPROVED" in critique.upper() and len(critique.strip()) < 50:
            break
        rm = [{"role":"system","content":"Revise your answer based on review feedback. Address EVERY issue. Output the COMPLETE revised answer."},
              {"role":"user","content":"Original question: " + umsg[:500] + "\n\nYour answer:\n" + cur[:2000] + "\n\nFeedback:\n" + critique[:1000] + "\n\nComplete revised answer:"}]
        try:
            rv = gen(rm)
            na = rv if isinstance(rv,str) else "".join(rv)
            if na and len(na) > len(cur) * 0.3:
                cur = na
        except Exception:
            break
    return (cur, crits)

def _is_complex(msg):
    m = msg.lower()
    if m.count("?") >= 3:
        return True
    sigs = ["and also","also tell me","additionally","furthermore","as well as","moreover",
            "first,","second,","third,","compare and contrast","pros and cons",
            "part 1","part 2","step 1","step 2"]
    if sum(1 for s in sigs if s in m) >= 2:
        return True
    if len(msg) > 500 and m.count("?") >= 2:
        return True
    return False

def decompose_and_solve(gen, msgs, complexity="medium"):
    umsg = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
    if not _is_complex(umsg) or complexity == "easy":
        a = gen(msgs)
        return (a if isinstance(a,str) else "".join(a)), [], []
    dm = [{"role":"system","content":"Break this question into sub-questions. Output ONLY a JSON array of strings, no markdown."},
          {"role":"user","content":umsg[:600]}]
    try:
        raw = gen(dm)
        rt = raw if isinstance(raw,str) else "".join(raw)
        match = _re.search(r"\[.*\]", rt, _re.DOTALL)
        if not match:
            a = gen(msgs)
            return (a if isinstance(a,str) else "".join(a)), [], []
        cand = match.group(0)
        if not cand.endswith("]"):
            last = cand.rfind('",')
            cand = (cand[:last+1] + "]") if last != -1 else "[]"
        subs = json.loads(cand)
        if not isinstance(subs, list) or len(subs) < 2:
            a = gen(msgs)
            return (a if isinstance(a,str) else "".join(a)), [], []
    except (json.JSONDecodeError, Exception):
        a = gen(msgs)
        return (a if isinstance(a,str) else "".join(a)), [], []
    sp = next((m.get("content","") for m in msgs if m.get("role")=="system"), "Answer precisely.")
    sa = []
    for sq in subs[:5]:
        try:
            sm = [{"role":"system","content":sp},{"role":"user","content":sq}]
            r = gen(sm)
            sa.append(r if isinstance(r,str) else "".join(r))
        except Exception as e:
            sa.append("[failed: " + str(e) + "]")
    sm2 = [{"role":"system","content":"Synthesize sub-answers into ONE coherent response. Do NOT list as Q&A."},
           {"role":"user","content":"Original: " + umsg[:400] + "\n\nSub-answers:\n" + "\n".join("Q:" + q + "\nA:" + a[:400] for q,a in zip(subs,sa)) + "\n\nUnified answer:"}]
    try:
        f = gen(sm2)
        final = f if isinstance(f,str) else "".join(f)
    except Exception:
        final = "\n\n".join("**" + q + "**\n" + a for q,a in zip(subs,sa))
    return (final, subs, sa)

def calibrate_uncertainty(answer, umsg):
    ml = umsg.lower()
    if any(c in ml for c in ["write a","story","poem","opinion","what do you think"]):
        return answer
    return answer

def think(gen, msgs, skill="general", complexity="medium",
          use_consistency=True, use_cot=True, use_refinement=True, use_decomposition=True):
    umsg = next((m.get("content","") for m in reversed(msgs) if m.get("role")=="user"), "")
    ml = umsg.lower()
    r = {"answer":"","thinking":"","method":"direct","confidence":1.0,"sub_questions":[],"critiques":[]}
    if complexity == "easy" and skill in ("general","calculator"):
        raw = gen(msgs)
        r["answer"] = raw if isinstance(raw,str) else "".join(raw)
        return r
    if use_decomposition and _is_complex(umsg) and skill != "coder":
        a, s, sa = decompose_and_solve(gen, msgs, complexity)
        if s:
            r["answer"] = a
            r["method"] = "decomposition"
            r["sub_questions"] = s
            return r
    if skill == "coder":
        if use_cot:
            a, t = hidden_cot(gen, msgs, complexity)
            r["thinking"] = t
        else:
            raw = gen(msgs)
            a = raw if isinstance(raw,str) else "".join(raw)
        if use_refinement and complexity in ("medium","hard"):
            a, c = progressive_refine(gen, msgs, skill, complexity, 2)
            r["critiques"] = c
        r["answer"] = a
        r["method"] = "coder-cot-refine"
        return r
    if skill == "researcher":
        if use_cot:
            a, t = hidden_cot(gen, msgs, complexity)
            r["thinking"] = t
        else:
            raw = gen(msgs)
            a = raw if isinstance(raw,str) else "".join(raw)
        if use_refinement:
            a, c = progressive_refine(gen, msgs, skill, complexity, 2)
            r["critiques"] = c
        r["answer"] = a
        r["method"] = "researcher-cot-refine"
        return r
    is_factual = any(k in ml for k in ["what is","what was","when did","who invented","who discovered",
        "how many","how much","calculate","solve","what year","define","capital of","formula for","equation"])
    if use_consistency and (is_factual or skill == "calculator"):
        nv = 3 if complexity == "medium" else 5
        a, aa, ag = self_consistency(gen, msgs, n=nv, complexity=complexity)
        if use_cot and complexity in ("hard","reasoning"):
            rf, tk = hidden_cot(gen, msgs, complexity)
            if tk and len(rf) > len(a) * 0.5:
                a = rf
                r["thinking"] = tk
        r["answer"] = a
        r["method"] = "consistency-" + str(nv) + "v"
        r["confidence"] = ag
        return r
    if use_cot:
        a, t = hidden_cot(gen, msgs, complexity)
        r["thinking"] = t
    else:
        raw = gen(msgs)
        a = raw if isinstance(raw,str) else "".join(raw)
    if use_refinement and complexity == "hard":
        a, c = progressive_refine(gen, msgs, skill, complexity, 1)
        r["critiques"] = c
    a = calibrate_uncertainty(a, umsg)
    r["answer"] = a
    r["method"] = "cot" + ("-refine" if r["critiques"] else "")
    return r
