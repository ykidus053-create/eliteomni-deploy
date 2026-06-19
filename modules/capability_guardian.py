"""
capability_guardian.py
Runs BEFORE server starts. Tests every capability.
If broken — attempts auto-repair. Blocks startup if critical caps fail.
"""
import sys, time, traceback, importlib, subprocess
from pathlib import Path

CRITICAL = {"MEMORY", "PIPELINE", "SKILL_ROUTER"}  # server won't start without these
RESULTS = {}

def _test(name, fn):
    try:
        r = fn()
        if r is None or r == [] or r == {} or r == "":
            RESULTS[name] = ("EMPTY", None)
        else:
            RESULTS[name] = ("OK", None)
    except Exception as e:
        RESULTS[name] = ("BROKEN", traceback.format_exc(limit=3))

def _repair(name, err):
    """Attempt auto-repair based on error type."""
    err_lower = err.lower()

    # Missing pip package
    import re
    pkg = re.search(r"no module named '([\w\-]+)'", err_lower)
    if pkg:
        package = pkg.group(1).replace("_", "-")
        print(f"  [Guardian] installing missing package: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", package,
                        "--break-system-packages", "-q"], timeout=30)
        return True

    # DB corruption
    if "duplicate column" in err_lower or "no such column" in err_lower:
        print(f"  [Guardian] DB schema issue in {name} — patching")
        try:
            import sqlite3, os
            db = os.path.expanduser("~/eliteomni_memory.db")
            conn = sqlite3.connect(db)
            col = re.search(r"no such column: (\w+)", err_lower)
            if col:
                conn.execute(f"ALTER TABLE memory ADD COLUMN {col.group(1)} TEXT")
                conn.commit()
            conn.close()
            return True
        except: pass

    # Name/import error — try reloading module
    if "nameerror" in err_lower or "importerror" in err_lower or "is not defined" in err_lower:
        print(f"  [Guardian] import/name error in {name} — clearing pycache")
        subprocess.run(["find", "/home/kidus/eliteomni_app", "-name", "*.pyc",
                        "-delete"], capture_output=True)
        subprocess.run(["find", "/home/kidus/eliteomni_app", "-name",
                        "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"],
                       capture_output=True)
        return True

    # Syntax error — log it clearly, can't auto-fix
    if "syntaxerror" in err_lower or "indentationerror" in err_lower:
        print(f"  [Guardian] ❌ SYNTAX ERROR in {name} — manual fix required")
        for line in err.strip().splitlines()[-5:]:
            print(f"     {line}")
        return False

    return False

# ── Capability tests ──────────────────────────────────────────────────────────

TESTS = {
    "MEMORY": lambda: __import__(
        "modules.services.memory", fromlist=["mem_save","mem_get"]
    ).mem_get("guardian_probe", k=1) or "ok",

    "PIPELINE": lambda: __import__(
        "modules.services.pipeline", fromlist=["build_system_prompt"]
    ).build_system_prompt("general", [], [], "", "", "easy"),

    "SKILL_ROUTER": lambda: __import__(
        "skill_router", fromlist=["classify_skill"]
    ).classify_skill("write a python script"),

    "REASONING": lambda: __import__(
        "modules.intelligence.reasoning_core",
        fromlist=["build_reasoning_core_context"]
    ).build_reasoning_core_context("test", "general", "easy", "guardian") or "ok",

    "WORKING_MEMORY": lambda: (
        __import__("modules.working_memory", fromlist=["wm_retrieve"])
        .wm_retrieve("test", k=1) or "ok"
    ),

    "SEMANTIC_MEM": lambda: (
        __import__("modules.semantic_mem", fromlist=["SemanticMemory"])
        .SemanticMemory().search("probe", k=1) or "ok"
    ),

    "SEARCH": lambda: (
        __import__("modules.services.search", fromlist=["tool_search"])
        .tool_search("test", n=1) or "ok"
    ),

    "SELF_MODEL": lambda: (
        __import__("modules.intelligence.self_model", fromlist=["get_self_context"])
        .get_self_context() or "ok"
    ),

    "UNCERTAINTY": lambda: (
        __import__("modules.uncertainty_engine", fromlist=["estimate_uncertainty"])
        .estimate_uncertainty("test", "general") or "ok"
    ),

    "SWE_VERIFIER": lambda: (
        __import__("modules.swe_verifier", fromlist=["swe_verify_and_fix"])
        and "ok"
    ),
}

def run_guardian():
    print("\n" + "="*55)
    print("  CAPABILITY GUARDIAN — pre-flight check")
    print("="*55)

    # Round 1: test all
    for name, fn in TESTS.items():
        _test(name, fn)

    # Round 2: repair broken, retest once
    for name, (status, err) in list(RESULTS.items()):
        if status in ("BROKEN", "EMPTY") and err:
            print(f"\n  [Guardian] ⚠️  {name} failed — attempting repair...")
            repaired = _repair(name, err)
            if repaired:
                time.sleep(0.5)
                _test(name, TESTS[name])
                new_status = RESULTS[name][0]
                print(f"  [Guardian] {name} after repair: {new_status}")

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n  RESULT:")
    failed_critical = []
    for name, (status, err) in RESULTS.items():
        icon = "✅" if status == "OK" else ("⚠️ " if status == "EMPTY" else "❌")
        crit = " [CRITICAL]" if name in CRITICAL and status != "OK" else ""
        print(f"  {icon} {name:<20}{crit}")
        if name in CRITICAL and status != "OK":
            failed_critical.append(name)

    print("="*55)

    if failed_critical:
        print(f"\n  ❌ STARTUP BLOCKED — critical capabilities broken: {failed_critical}")
        print("  Fix the errors above then restart.\n")
        sys.exit(1)
    else:
        print("  ✅ All critical capabilities healthy — starting server\n")

if __name__ == "__main__":
    run_guardian()
