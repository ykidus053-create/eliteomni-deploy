"""
Wire all orphaned modules into app.py.
Run once from eliteomni_app: python wire_orphans.py
"""
import re

PATH = "app.py"
with open(PATH, "r") as f:
    src = f.read()

# ── 1. IMPORTS (add after existing imports block) ────────────────────────────
IMPORTS = """
# ── Orphaned modules wired in ────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from pregen_reasoning import analyze_intent, build_intent_aware_system, should_run_pregen
    _PREGEN_LOADED = True
except Exception as _e: print(f"[wire] pregen_reasoning: {_e}"); _PREGEN_LOADED = False
try:
    from voting_engine import should_use_voting, self_consistent_answer, vote_report
    _VOTING_LOADED = True
except Exception as _e: print(f"[wire] voting_engine: {_e}"); _VOTING_LOADED = False
try:
    from uncertainty_engine import score_response_confidence, strip_overconfidence, inject_confidence_header, should_hedge
    _UNCERTAINTY_LOADED = True
except Exception as _e: print(f"[wire] uncertainty_engine: {_e}"); _UNCERTAINTY_LOADED = False
try:
    from reflection_engine import should_reflect, reflect_on_response, annotate_response
    _REFLECTION_LOADED = True
except Exception as _e: print(f"[wire] reflection_engine: {_e}"); _REFLECTION_LOADED = False
try:
    from cot_engine import inject_cot, strip_reasoning_artifacts, cot_complexity_gate
    _COT_LOADED = True
except Exception as _e: print(f"[wire] cot_engine: {_e}"); _COT_LOADED = False
try:
    from safety_layer import SafetyLayer as _SafetyLayer
    _safety_layer = _SafetyLayer()
    _SAFETY_LAYER_LOADED = True
except Exception as _e: print(f"[wire] safety_layer: {_e}"); _SAFETY_LAYER_LOADED = False
try:
    from error_learner import scan_for_errors, record_error, get_error_warnings, post_process_check
    _ERROR_LEARNER_LOADED = True
except Exception as _e: print(f"[wire] error_learner: {_e}"); _ERROR_LEARNER_LOADED = False
try:
    from goal_engine import goal_detect_and_save, goals_get_context as root_goals_get_context, goal_complete
    _GOAL_ENGINE_LOADED = True
except Exception as _e: print(f"[wire] goal_engine: {_e}"); _GOAL_ENGINE_LOADED = False
try:
    from tool_planner import plan_tools, tool_plan_to_system
    _TOOL_PLANNER_LOADED = True
except Exception as _e: print(f"[wire] tool_planner: {_e}"); _TOOL_PLANNER_LOADED = False
try:
    from reasoning_engine import deliberate as deliberate_v2
    _REASONING_ENGINE_LOADED = True
except Exception as _e: print(f"[wire] reasoning_engine: {_e}"); _REASONING_ENGINE_LOADED = False
# ── End orphan imports ────────────────────────────────────────────────────────
"""

# Insert imports after the last top-level import line
insert_after = "import groq_client_patch  # speed patch"
if insert_after in src and IMPORTS.strip() not in src:
    src = src.replace(insert_after, insert_after + "\n" + IMPORTS)
    print("[wire] ✓ imports injected")
else:
    print("[wire] imports already present or anchor not found")

# ── 2. PREGEN — inject after skill/complexity are set (~line 426) ─────────────
PREGEN_HOOK = """
    # ── Pre-generation reasoning (orphan wired) ──────────────────────────────
    if _PREGEN_LOADED and should_run_pregen(msg, skill, complexity):
        try:
            def _quick_gen(p): return generate_sync(p, 400, skill, len(msg))
            _intent = analyze_intent(msg, _quick_gen)
            system = build_intent_aware_system(system if 'system' in dir() else '', _intent, skill, complexity)
            print(f"[PreGen] intent={_intent.get('true_intent','?')[:60]}")
        except Exception as _pge: print(f"[PreGen] {_pge}")
    # ── Goal tracking (orphan wired) ─────────────────────────────────────────
    if _GOAL_ENGINE_LOADED:
        try:
            goal_detect_and_save(msg, history[0].get('session_id','default') if history else 'default')
            _root_goals = root_goals_get_context()
            if _root_goals and 'system' in dir():
                system = system + "\\n" + _root_goals if system else _root_goals
        except Exception as _ge: print(f"[GoalEngine] {_ge}")
    # ── CoT injection (orphan wired) ─────────────────────────────────────────
    if _COT_LOADED:
        try:
            system = inject_cot(system if 'system' in dir() else '', skill, complexity, msg)
        except Exception as _ce: print(f"[CoT] {_ce}")
    # ── Tool planner (orphan wired) ───────────────────────────────────────────
    if _TOOL_PLANNER_LOADED and complexity in ('medium','hard'):
        try:
            _tool_plan_ctx = tool_plan_to_system(msg, skill, complexity)
            if _tool_plan_ctx and 'system' in dir():
                system = system + "\\n" + _tool_plan_ctx
        except Exception as _tpe: print(f"[ToolPlanner] {_tpe}")
"""

PREGEN_ANCHOR = "if skill == \"calculator\": complexity = max(complexity"
if PREGEN_ANCHOR in src and "PreGen] intent" not in src:
    src = src.replace(PREGEN_ANCHOR, PREGEN_ANCHOR + "\n" + PREGEN_HOOK)
    print("[wire] ✓ pregen+goals+cot+toolplanner injected")
else:
    print("[wire] pregen anchor not found or already wired")

# ── 3. VOTING — inject inside agentic loop where response is first set ────────
VOTING_HOOK = """
        # ── Self-consistency voting (orphan wired) ────────────────────────────
        if _VOTING_LOADED and should_use_voting(msg, skill, complexity) and not response:
            try:
                def _vgen(msgs_): return generate_sync(build_chatml(system, hist_msgs, clean_msg), max_t, skill, len(msg))
                _voted, _vconf = self_consistent_answer(_vgen, [], n_samples=3, max_tokens=max_t)
                if _voted:
                    response = _voted
                    print(f"[Voting] confidence={_vconf:.2f}")
            except Exception as _ve2: print(f"[Voting] {_ve2}")
        # ── Reasoning engine (orphan wired) ──────────────────────────────────
        elif _REASONING_ENGINE_LOADED and complexity == 'hard' and not response:
            try:
                def _rgen(p): return generate_sync(p, max_t, skill, len(msg))
                response = deliberate_v2(msg, system, hist_msgs, _rgen, _routed_model, complexity, skill)
                print(f"[ReasoningEngine] deliberate complete")
            except Exception as _ree: print(f"[ReasoningEngine] {_ree}")
"""

VOTING_ANCHOR = "if complexity == \"hard\" and skill in (\"researcher\", \"coder\"):"
if VOTING_ANCHOR in src and "Self-consistency voting" not in src:
    src = src.replace(VOTING_ANCHOR, VOTING_HOOK + "\n        " + VOTING_ANCHOR)
    print("[wire] ✓ voting+reasoning_engine injected")
else:
    print("[wire] voting anchor not found or already wired")

# ── 4. POST-PROCESSING — inject before final return ──────────────────────────
POST_HOOK = """
    # ── Uncertainty calibration (orphan wired) ───────────────────────────────
    if _UNCERTAINTY_LOADED:
        try:
            final = strip_overconfidence(final)
            _conf = score_response_confidence(final, skill)
            if should_hedge(skill, complexity, has_search):
                final = inject_confidence_header(final, _conf, skill)
        except Exception as _uce: print(f"[Uncertainty] {_uce}")
    # ── Reflection (orphan wired) ─────────────────────────────────────────────
    if _REFLECTION_LOADED and should_reflect(msg, skill, complexity):
        try:
            _issues = reflect_on_response(final, msg, skill)
            if _issues:
                final = annotate_response(final, _issues, skill)
        except Exception as _rfe: print(f"[Reflection] {_rfe}")
    # ── Error learner (orphan wired) ──────────────────────────────────────────
    if _ERROR_LEARNER_LOADED:
        try:
            _errs = scan_for_errors(final, skill)
            if _errs:
                for _err in _errs: record_error(_err, skill)
            final = post_process_check(final, skill)
        except Exception as _ele: print(f"[ErrorLearner] {_ele}")
    # ── CoT artifact strip (orphan wired) ────────────────────────────────────
    if _COT_LOADED:
        try:
            final = strip_reasoning_artifacts(final)
        except Exception: pass
"""

POST_ANCHOR = "latency_ms = int((time.time() - t_start) * 1000)"
if POST_ANCHOR in src and "Uncertainty calibration" not in src:
    src = src.replace(POST_ANCHOR, POST_HOOK + "\n    " + POST_ANCHOR)
    print("[wire] ✓ uncertainty+reflection+error_learner injected")
else:
    print("[wire] post-processing anchor not found or already wired")

# ── Write patched file ────────────────────────────────────────────────────────
with open(PATH + ".pre_wire_orphans.bak", "w") as f:
    f.write(open(PATH).read())
with open(PATH, "w") as f:
    f.write(src)

print("\n✅ Done. Backup at app.py.pre_wire_orphans.bak")
print("Run: python app.py to test")
