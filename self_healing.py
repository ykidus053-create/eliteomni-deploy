import os, time, threading, subprocess, logging, traceback
log = logging.getLogger(__name__)

def _ci_loop(generate_fn):
    """Upgraded: Continuously runs the project's test suite to catch regressions."""
    time.sleep(120) # Wait 2 mins after startup
    log.info("[SelfHealing] CI/CD Daemon started.")
    while True:
        try:
            # Run the actual project test suite
            r = subprocess.run(["python", "-m", "pytest", "tests/", "-v", "--tb=long"], capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                from context_compressor import log_subconscious_action
                log_subconscious_action('SelfHealing', 'Detected a CI regression and attempted an autonomous fix.')
                from reflexion_loop import reflexion_verify
                # Feed the traceback to the AI to fix
                task = "Fix the failing pytest regression in the test suite."
                fix = reflexion_verify(r.stdout + r.stderr, generate_fn, task, max_rounds=3)
                log.info("[SelfHealing] Autonomous fix generated and applied.")
        except Exception as e:
            log.debug(f"[SelfHealing] CI loop error: {e}")
        time.sleep(600) # Run every 10 minutes

def start_self_healing_daemon(generate_fn):
    t = threading.Thread(target=_ci_loop, args=(generate_fn,), daemon=True, name="self_healing_ci")
    t.start()
    print("[Startup] ✓ Self-Healing CI/CD Daemon started.")

# BEGIN SAFE STARTUP V21
_SELF_HEALING_THREAD_V21 = None


def start_self_healing_daemon(generate_fn):
    """Opt-in startup guard installed by Frontier Runtime V21."""
    global _SELF_HEALING_THREAD_V21
    enabled = __import__("os").environ.get("ELITE_ENABLE_SELF_HEALING", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        print("[Startup] Self-Healing Daemon disabled (ELITE_ENABLE_SELF_HEALING=0)")
        return None
    if _SELF_HEALING_THREAD_V21 is not None and _SELF_HEALING_THREAD_V21.is_alive():
        return _SELF_HEALING_THREAD_V21
    _SELF_HEALING_THREAD_V21 = threading.Thread(target=_ci_loop, args=(generate_fn,), daemon=True, name='self_healing_ci')
    _SELF_HEALING_THREAD_V21.start()
    print('[Startup] ✓ Self-Healing Daemon started explicitly.')
    return _SELF_HEALING_THREAD_V21
# END SAFE STARTUP V21
