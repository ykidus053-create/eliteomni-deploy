"""
capability_watchdog.py
Runs on startup. Tests every AI capability with a real call.
Prints a health report. Logs broken ones to a file.
"""
import time, traceback
from pathlib import Path

LOG = Path.home() / "eliteomni_capability_health.log"
RESULTS = {}

def _test(name, fn):
    try:
        start = time.time()
        result = fn()
        ms = int((time.time() - start) * 1000)
        if result is None or result == [] or result == {} or result == "":
            RESULTS[name] = ("⚠️  EMPTY", ms, "returned empty/None")
        else:
            RESULTS[name] = ("✅ OK", ms, "")
    except Exception as e:
        RESULTS[name] = ("❌ BROKEN", 0, traceback.format_exc(limit=2))

def run_watchdog():
    # MEMORY
    def test_memory():
        from modules.services.memory import mem_save, mem_get
        mem_save("watchdog_test", "test", importance=0.1)
        return mem_get("watchdog_test", k=1)
    _test("MEMORY", test_memory)

    # SEMANTIC MEMORY
    def test_semantic():
        from modules.semantic_mem import SemanticMemory
        sm = SemanticMemory()
        sm.add("watchdog probe", metadata={"source": "watchdog"})
        return sm.search("watchdog probe", k=1)
    _test("SEMANTIC_MEM", test_semantic)

    # REASONING CORE
    def test_reasoning():
        from modules.intelligence.reasoning_core import build_reasoning_core_context
        r, m = build_reasoning_core_context("test query", "general", "easy", "watchdog")
        return r or m
    _test("REASONING_CORE", test_reasoning)

    # SKILL ROUTER
    def test_router():
        from skill_router import classify_skill, route_complexity
        s = classify_skill("write a python function")
        c = route_complexity("write a python function")
        return f"{s}/{c}"
    _test("SKILL_ROUTER", test_router)

    # SEARCH
    def test_search():
        from modules.services.search import tool_search
        return tool_search("python list comprehension", n=1)
    _test("SEARCH", test_search)

    # WORKING MEMORY
    def test_wm():
        from modules.working_memory import wm_retrieve
        return wm_retrieve("test", k=1) or "ok"
    _test("WORKING_MEMORY", test_wm)

    # TOOLS
    def test_tools():
        from modules.core.tool_orchestrator import ToolOrchestrator
        t = ToolOrchestrator()
        return t or "ok"
    _test("TOOLS", test_tools)

    # PIPELINE
    def test_pipeline():
        from modules.services.pipeline import build_system_prompt
        return build_system_prompt("general", [], [], "", "", "easy")
    _test("PIPELINE", test_pipeline)

    # SELF MODEL
    def test_self():
        from modules.intelligence.self_model import get_self_context
        return get_self_context() or "ok"
    _test("SELF_MODEL", test_self)

    # UNCERTAINTY
    def test_uncertainty():
        from modules.uncertainty_engine import estimate_uncertainty
        return estimate_uncertainty("what is 2+2", "general")
    _test("UNCERTAINTY", test_uncertainty)

    # ── Print report ─────────────────────────────────────────────────────────
    lines = ["\n" + "="*55, "  CAPABILITY HEALTH REPORT", "="*55]
    broken = []
    for cap, (status, ms, err) in RESULTS.items():
        ms_str = f"{ms}ms" if ms else ""
        lines.append(f"  {status:<12} {cap:<20} {ms_str}")
        if "BROKEN" in status or "EMPTY" in status:
            broken.append((cap, err))

    if broken:
        lines.append("\n  FAILURES:")
        for cap, err in broken:
            lines.append(f"  ── {cap}")
            for l in err.strip().splitlines()[-3:]:
                lines.append(f"     {l}")
    else:
        lines.append("\n  All capabilities operational ✅")

    lines.append("="*55)
    report = "\n".join(lines)
    print(report)
    LOG.write_text(report)
    return broken

if __name__ == "__main__":
    run_watchdog()
