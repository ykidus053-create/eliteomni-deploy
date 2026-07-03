"""
Integration patch — wires all upgrade modules into app.py.
Run once: python integrate_upgrades.py
Creates a backup then patches in-place.
"""
import os, shutil, sys

SRC = os.path.join(os.path.dirname(__file__), "app.py")
if not os.path.exists(SRC):
    SRC = os.path.expanduser("~/eliteomni_app/app.py")
if not os.path.exists(SRC):
    print("ERROR: app.py not found"); sys.exit(1)

shutil.copy(SRC, SRC + ".pre_upgrade.bak")
print(f"✅ Backup → {SRC}.pre_upgrade.bak")

with open(SRC, "r", encoding="utf-8") as f:
    src = f.read()

applied = []

# ── 1. Add upgrade imports at top ─────────────────────────────────────────────
IMPORT_ANCHOR = "app = FastAPI"
NEW_IMPORTS = '''
# ── Upgrade Module Imports ────────────────────────────────────────────────────
try:
    from working_memory import wm_save, wm_retrieve, wm_build_context
    _WM_LOADED = True
    print("[Upgrade] ✓ Working Memory Engine loaded")
except ImportError as _e:
    _WM_LOADED = False
    print(f"[Upgrade] ✗ WorkingMemory: {_e}")
    def wm_save(t, **kw): pass
    def wm_retrieve(q, k=8, **kw): return []
    def wm_build_context(q): return ""

try:
    from reasoning_engine import deliberate
    _REASONING_LOADED = True
    print("[Upgrade] ✓ Deliberative Reasoning Engine loaded")
except ImportError as _e:
    _REASONING_LOADED = False
    print(f"[Upgrade] ✗ Reasoning: {_e}")
    def deliberate(msg, system, history, generate_fn, model, complexity="medium", skill="general"):
        msgs = [{"role": "system", "content": system}] + history[-12:] + [{"role": "user", "content": msg}]
        return generate_fn(msgs, max_tokens=1200, model=model)

try:
    from planner import create_plan, execute_plan, plan_format_display
    _PLANNER_LOADED = True
    print("[Upgrade] ✓ Hierarchical Planner loaded")
except ImportError as _e:
    _PLANNER_LOADED = False
    print(f"[Upgrade] ✗ Planner: {_e}")

try:
    from skill_router import route as _route_fn
try:
    from memory import classify_skill
except ImportError:
    classify_skill = lambda msg: _route_fn(msg)[0] as _classify_skill_v2, route_complexity as _route_complexity_v2
    classify_skill = _classify_skill_v2
    route_complexity = _route_complexity_v2
    print("[Upgrade] ✓ Semantic Skill Router loaded (overrides keyword matcher)")
except ImportError as _e:
    print(f"[Upgrade] ✗ SkillRouter: {_e}")

try:
    from learning_loop import log_interaction, get_learned_system_addendum, check_drift
    _LEARNING_LOADED = True
    print("[Upgrade] ✓ Online Learning Loop loaded")
except ImportError as _e:
    _LEARNING_LOADED = False
    print(f"[Upgrade] ✗ LearningLoop: {_e}")
    def log_interaction(*a, **kw): pass
    def get_learned_system_addendum(): return ""
    def check_drift(): return None

try:
    from world_model import get_world_model_context, UserModel, ConversationState
    _WM_MODEL_LOADED = True
    print("[Upgrade] ✓ World Model loaded")
except ImportError as _e:
    _WM_MODEL_LOADED = False
    print(f"[Upgrade] ✗ WorldModel: {_e}")
    def get_world_model_context(msg, session_id="default"): return ""

# ── Upgrade API Endpoints ─────────────────────────────────────────────────────

'''

if IMPORT_ANCHOR in src and "from working_memory import" not in src:
    src = src.replace(IMPORT_ANCHOR, NEW_IMPORTS + IMPORT_ANCHOR, 1)
    applied.append("✅ 1. Upgrade module imports added")
else:
    applied.append("⚠️ 1. Imports already present or anchor not found")

# ── 2. Replace world model stub ───────────────────────────────────────────────
OLD_WM = "_wm_ctx = {}  # world_model stubbed — not implemented"
NEW_WM = (
    "_wm_ctx = get_world_model_context(msg) if _WM_MODEL_LOADED else {}  # world model active\n"
    "    _wm_learned = get_learned_system_addendum() if _LEARNING_LOADED else ''"
)
if OLD_WM in src:
    src = src.replace(OLD_WM, NEW_WM)
    applied.append("✅ 2. World model stub replaced with active implementation")
else:
    applied.append("⚠️ 2. World model stub not found")

# ── 3. Wire learned addendum into system prompt ────────────────────────────────
OLD_SYS_BUILD = "system = build_system_prompt(skill, memory, episodic, rlhf_note, ctx_sum or \"\", complexity)"
NEW_SYS_BUILD = (
    "system = build_system_prompt(skill, memory, episodic, rlhf_note, ctx_sum or \"\", complexity)\n"
    "    _learned_addon = get_learned_system_addendum() if _LEARNING_LOADED else ''\n"
    "    if _learned_addon: system += '\\n' + _learned_addon\n"
    "    _wm_ctx_str = get_world_model_context(msg) if _WM_MODEL_LOADED else ''\n"
    "    if _wm_ctx_str: system += '\\n' + _wm_ctx_str"
)
if OLD_SYS_BUILD in src and "_learned_addon" not in src:
    src = src.replace(OLD_SYS_BUILD, NEW_SYS_BUILD, 1)
    applied.append("✅ 3. Learned system addendum + world model wired into prompt")
else:
    applied.append("⚠️ 3. System build already patched or anchor not found")

# ── 4. Wire interaction logging into pipeline_sync return ─────────────────────
OLD_RETURN = '    return {\n        "response":   final,'
NEW_RETURN = (
    '    # Log interaction for learning\n'
    '    if _LEARNING_LOADED:\n'
    '        try:\n'
    '            log_interaction(skill, complexity, msg, final, latency_ms)\n'
    '            wm_save(f"Q:{msg[:80]} A:{final[:160]}", memory_type="episodic")\n'
    '        except Exception as _le:\n'
    '            print(f"[LearningLog] {_le}")\n'
    '    return {\n        "response":   final,'
)
if OLD_RETURN in src and "log_interaction(skill" not in src:
    src = src.replace(OLD_RETURN, NEW_RETURN, 1)
    applied.append("✅ 4. Interaction logging wired into pipeline_sync")
else:
    applied.append("⚠️ 4. Logging already wired or anchor not found")

# ── 5. Add upgrade status endpoints ──────────────────────────────────────────
ENDPOINT_ANCHOR = '@app.get("/effort")'
NEW_ENDPOINTS = '''
@app.get("/upgrades/status")
async def upgrade_status():
    """Status of all upgrade modules."""
    drift = None
    try:
        drift = check_drift() if _LEARNING_LOADED else None
    except Exception:
        pass
    return {
        "working_memory": _WM_LOADED,
        "reasoning_engine": _REASONING_LOADED,
        "planner": _PLANNER_LOADED,
        "learning_loop": _LEARNING_LOADED,
        "world_model": _WM_MODEL_LOADED,
        "behavioral_drift": drift,
    }

@app.get("/upgrades/learning")
async def learning_stats_endpoint():
    """Get online learning statistics."""
    try:
        from learning_loop import get_learning_stats
        return get_learning_stats()
    except Exception as e:
        return {"error": str(e)}

@app.post("/upgrades/deliberate")
async def deliberate_endpoint(req: Request):
    """Test deliberative reasoning on a single question."""
    d = await req.json()
    msg = d.get("msg", "")
    complexity = d.get("complexity", "hard")
    if not msg:
        return {"error": "msg required"}
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: deliberate(
        msg=msg,
        system="You are a highly capable AI assistant.",
        history=[],
        generate_fn=lambda msgs, **kw: groq_generate(msgs, max_tokens=kw.get("max_tokens", 1000)),
        model=GROQ_MODEL,
        complexity=complexity,
        skill=classify_skill(msg)
    ))
    return {"response": result, "complexity": complexity}

'''
if ENDPOINT_ANCHOR in src and "/upgrades/status" not in src:
    src = src.replace(ENDPOINT_ANCHOR, NEW_ENDPOINTS + ENDPOINT_ANCHOR, 1)
    applied.append("✅ 5. Upgrade API endpoints added")
else:
    applied.append("⚠️ 5. Endpoints already exist or anchor not found")

with open(SRC, "w", encoding="utf-8") as f:
    f.write(src)

print(f"\n{'='*60}")
print("UPGRADE INTEGRATION COMPLETE")
print(f"{'='*60}")
for line in applied:
    print(f"  {line}")
print(f"\nTo apply:\n  cd ~/eliteomni_app && python integrate_upgrades.py")
print(f"To verify:\n  curl http://localhost:8080/upgrades/status")
print(f"To test deliberative reasoning:\n  curl -X POST http://localhost:8080/upgrades/deliberate -H 'Content-Type: application/json' -d '{{\"msg\":\"What is the difference between L1 and L2 regularization?\",\"complexity\":\"hard\"}}'")
