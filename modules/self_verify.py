def self_verify(answer, original_prompt, generate_fn, skill="general", complexity="medium"):
    """Upgraded: uses progressive refinement for hard/researcher tasks."""
    from modules.thinking import progressive_refine
    if skill not in ("researcher","coder") and complexity != "hard":
        return answer
    msgs = [{"role":"user","content":original_prompt}]
    refined, critiques = progressive_refine(generate_fn, msgs, skill=skill, complexity=complexity, max_rounds=2)
    if critiques and len(refined) > len(answer) * 0.5:
        return refined
    return answer
