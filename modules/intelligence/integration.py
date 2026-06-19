"""
Intelligence Integration Layer
Connects all intelligence modules to the main pipeline.
This is the single entry point for all intelligence upgrades.
"""
import time
from typing import Optional, Tuple

def build_intelligence_context(
    msg: str,
    skill: str,
    complexity: str,
    history: list,
    session_id: str = "default"
) -> Tuple[str, dict]:
    """
    Build the full intelligence context for a request.
    Returns (context_injection, metadata).
    """
    meta = {}
    parts = []

    try:
        from modules.intelligence.complexity_estimator import estimate_complexity
        est_complexity, confidence = estimate_complexity(msg, len(history))
        if confidence > 0.7 and est_complexity != complexity:
            complexity = est_complexity
            meta["complexity_override"] = est_complexity
    except Exception as e:
        print(f"[Intelligence] complexity estimator: {e}")

    try:
        from modules.intelligence.world_model import get_world_model
        wm = get_world_model(session_id)
        snapshot = wm.get_context_snapshot()
        if snapshot:
            parts.append(f"\n<world_model>\n{snapshot}\n</world_model>")
            meta["world_model_active"] = True
    except Exception as e:
        print(f"[Intelligence] world model: {e}")

    try:
        from modules.intelligence.planner import decompose, plan_to_system_injection
        if complexity in ("medium", "hard"):
            plan = decompose(msg, skill, complexity)
            injection = plan_to_system_injection(plan)
            if injection:
                parts.append(injection)
            meta["plan_tasks"] = len(plan.tasks)
    except Exception as e:
        print(f"[Intelligence] planner: {e}")

    try:
        from modules.intelligence.uncertainty_engine import build_uncertainty_injection
        u_injection = build_uncertainty_injection(msg, skill)
        if u_injection:
            parts.append(u_injection)
    except Exception as e:
        print(f"[Intelligence] uncertainty: {e}")

    try:
        from modules.intelligence.meta_learner import build_reasoning_exemplars
        if complexity in ("medium", "hard"):
            exemplars = build_reasoning_exemplars(msg, skill)
            if exemplars:
                parts.append(exemplars)
    except Exception as e:
        print(f"[Intelligence] meta learner: {e}")

    try:
        from modules.intelligence.cognitive_monitor import get_cognitive_state
        state = get_cognitive_state(session_id)
        intervention = state.get_intervention()
        if intervention:
            parts.append(intervention)
            meta["cognitive_intervention"] = True
    except Exception as e:
        print(f"[Intelligence] cognitive monitor: {e}")

    return "\n".join(parts), meta


def post_process_response(
    msg: str,
    response: str,
    skill: str,
    complexity: str,
    session_id: str = "default",
    prm_score: float = 0.8,
    user_rating: int = 0
) -> str:
    """Post-process response: PRM annotation, world model update, corpus storage."""

    try:
        from modules.intelligence.process_reward import prm_annotation
        if complexity in ("medium", "hard") and len(response) > 200:
            response = prm_annotation(response, msg)
    except Exception as e:
        print(f"[Intelligence] PRM: {e}")

    try:
        from modules.intelligence.world_model import get_world_model
        wm = get_world_model(session_id)
        wm.extract_and_update(msg, response)
    except Exception as e:
        print(f"[Intelligence] world model update: {e}")

    try:
        from modules.intelligence.cognitive_monitor import get_cognitive_state
        state = get_cognitive_state(session_id)
        state.update(msg, response)
    except Exception as e:
        print(f"[Intelligence] cognitive monitor update: {e}")

    try:
        from modules.intelligence.meta_learner import store_pattern
        if len(response) > 150:
            store_pattern(msg, response, skill, complexity, prm_score, user_rating)
    except Exception as e:
        print(f"[Intelligence] meta learner store: {e}")

    return response


# ═══════════════════════════════════════════════════
# STREAM HOOK — call this from app.py after each stream completes
# ═══════════════════════════════════════════════════
def on_stream_complete(msg: str, response: str, skill: str,
                        complexity: str, session_id: str,
                        user_rating: int = 0):
    """
    Drop-in hook for app.py to call after each successful stream.
    Drives all post-generation intelligence updates asynchronously.
    """
    import threading
    def _async_update():
        try:
            from modules.intelligence.process_reward import evaluate_reasoning_chain
            prm = evaluate_reasoning_chain(response, msg)
            prm_score = prm.get("avg_score", 0.8)
        except Exception:
            prm_score = 0.8
        post_process_response(msg, response, skill, complexity,
                               session_id, prm_score, user_rating)
    t = threading.Thread(target=_async_update, daemon=True)
    t.start()


def build_intelligence_context_v2(msg, skill, complexity, history, session_id="default"):
    """
    Round 2 intelligence context — adds hypothesis engine, tool policy,
    self-model, causal reasoning, multi-agent detection, reflection.
    Call this INSTEAD of build_intelligence_context for full capability.
    """
    base_ctx, meta = build_intelligence_context(msg, skill, complexity, history, session_id)
    parts = [base_ctx] if base_ctx else []

    try:
        from modules.intelligence.hypothesis_engine import get_hypothesis_engine
        engine = get_hypothesis_engine(session_id)
        hyps = engine.generate_hypotheses(msg, skill)
        inj = engine.to_injection()
        if inj: parts.append(inj)
        meta["hypotheses"] = len(hyps)
    except Exception as e:
        print(f"[Intel v2] hypothesis: {e}")

    try:
        from modules.intelligence.tool_policy import build_tool_injection
        tool_inj = build_tool_injection(msg, skill, complexity)
        if tool_inj: parts.append(tool_inj)
    except Exception as e:
        print(f"[Intel v2] tool_policy: {e}")

    try:
        from modules.intelligence.self_model import get_self_model
        import re
        domain = "general"
        for d in ["medical","legal","financial","math","code","science"]:
            if d in msg.lower(): domain = d; break
        sm_inj = get_self_model().build_self_awareness_injection(skill, domain)
        if sm_inj: parts.append(sm_inj)
    except Exception as e:
        print(f"[Intel v2] self_model: {e}")

    try:
        from modules.intelligence.causal_reasoner import build_causal_injection
        causal_inj = build_causal_injection(msg)
        if causal_inj: parts.append(causal_inj)
    except Exception as e:
        print(f"[Intel v2] causal: {e}")

    try:
        from modules.intelligence.reflection_engine import build_reflection_injection
        refl_inj = build_reflection_injection(msg, skill)
        if refl_inj: parts.append(refl_inj)
    except Exception as e:
        print(f"[Intel v2] reflection: {e}")

    return chr(10).join(p for p in parts if p), meta


def post_process_v2(msg, response, skill, complexity, session_id="default",
                    prm_score=0.8, user_rating=0):
    """Round 2 post-processing — adds reflection storage and self-model update."""
    response = post_process_response(msg, response, skill, complexity,
                                     session_id, prm_score, user_rating)
    def _async():
        try:
            from modules.intelligence.reflection_engine import critique_response, store_reflection
            issues = critique_response(response, msg, skill)
            if issues:
                store_reflection(session_id, msg, response, skill, issues)
        except Exception as e:
            print(f"[Intel v2] reflection_store: {e}")
        try:
            from modules.intelligence.self_model import get_self_model
            domain = "general"
            for d in ["medical","legal","financial","math","code","science"]:
                if d in msg.lower(): domain = d; break
            get_self_model().record_outcome(skill, domain, prm_score > 0.7, prm_score, user_rating)
        except Exception as e:
            print(f"[Intel v2] self_model_update: {e}")
    import threading
    threading.Thread(target=_async, daemon=True).start()
    return response


def build_intelligence_context_v3(msg, skill, complexity, history, session_id="default"):
    """
    Round 3 — Reasoning Core: epistemic state, working memory,
    goal decomposition, adaptive routing, context compression,
    pre-generation verification. All gaps from architecture audit closed.
    """
    # Get v2 context first
    base_ctx, meta = build_intelligence_context_v2(msg, skill, complexity, history, session_id)
    parts = [base_ctx] if base_ctx else []

    try:
        from modules.intelligence.reasoning_core import build_reasoning_core_context
        rc_ctx, rc_meta = build_reasoning_core_context(msg, skill, complexity, session_id)
        if rc_ctx:
            parts.append(rc_ctx)
        meta.update(rc_meta)
    except Exception as e:
        print(f"[Intel v3] reasoning_core: {e}")

    return "\n".join(p for p in parts if p), meta


def post_process_v3(msg, response, skill, complexity, session_id="default",
                    prm_score=0.8, user_rating=0, tool_results=None):
    """Round 3 post-processing — closes active belief revision loop."""
    response = post_process_v2(msg, response, skill, complexity,
                                session_id, prm_score, user_rating)
    try:
        from modules.intelligence.reasoning_core import post_turn_update
        post_turn_update(msg, response, skill, session_id, tool_results)
    except Exception as e:
        print(f"[Intel v3] post_turn_update: {e}")
    return response
