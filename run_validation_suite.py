import os
import ast
import time
import threading
import sys

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(WORKSPACE_ROOT, "modules")

for execution_path in [WORKSPACE_ROOT, MODULES_DIR]:
    if execution_path not in sys.path:
        sys.path.insert(0, execution_path)

try:
    from modules.tools import numpy_exec_safe
    from modules.circuit_breaker import CircuitBreaker
except ImportError:
    from tools import numpy_exec_safe
    from circuit_breaker import CircuitBreaker

def run_tests():
    print("="*60)
    print("RUNNING SYSTEM CONCURRENCY & HARDENING SUITE")
    print("="*60)

    # 1. Test Negative Exponent, Float Bypasses, and Multiplication Magnitude bounds
    print("Evaluating Math Engine Magnitude Protections...")
    assert "Calculation Error" in numpy_exec_safe("2 ** -100000")
    assert "Calculation Error" in numpy_exec_safe("9999 ** 9999")
    assert "Calculation Error" in numpy_exec_safe("10**39 * 10**39")
    assert "Calculation Error" in numpy_exec_safe("0.000000000000000000000000000000000000000000000000001")
    assert "Calculation Error" in numpy_exec_safe("exp(1000)")
    print("  [PASS] Successfully blocked numerical explosions, float underflows, and NaN/Inf generation.")

    # 2. Test True AST Deep Nesting
    print("\nEvaluating Genuine AST Nesting Restraints...")
    deep_call = "sin(" * 20 + "1" + ")" * 20
    assert "exceeds safe limits" in numpy_exec_safe(deep_call)
    print("  [PASS] Successfully blocked true nested function stack exhaustion.")

    # 3. Test Threshold Enforcement in Circuit Breaker
    print("\nEvaluating Circuit Breaker Threshold Enforcement...")
    def _raise(ex): raise ex
    cb_thresh = CircuitBreaker("threshold-test", failure_threshold=3, recovery_timeout=0.5)
    
    for _ in range(2):
        try: cb_thresh.call(lambda: _raise(IOError("Net Error")))
        except Exception: pass
        assert cb_thresh.state == "closed", "Breaker tripped prematurely!"
        
    try: cb_thresh.call(lambda: _raise(IOError("Net Error")))
    except Exception: pass
    assert cb_thresh.state == "open", "Breaker failed to trip on threshold boundary!"
    print("  [PASS] Circuit breaker strictly respects failure thresholds.")

    # 4. Test Probe Reacquisition and Starvation Prevention
    print("\nEvaluating Circuit Breaker Starvation Prevention...")
    cb_starve = CircuitBreaker("starve-test", failure_threshold=1, recovery_timeout=0.1)
    class NonTrippingAppError(Exception): pass
    
    try: cb_starve.inspect_and_call(lambda: _raise(IOError("Net Fail")), (IOError,))
    except IOError: pass
    assert cb_starve.state == "open"
    
    time.sleep(0.15)
    try: cb_starve.inspect_and_call(lambda: _raise(NonTrippingAppError("App Bug")), (IOError,))
    except NonTrippingAppError: pass
    
    assert cb_starve.state == "open", "Breaker locked open or failed starvation reset!"
    assert cb_starve._probe_in_progress is False, "Probe state leaked tracking metrics!"
    print("  [PASS] Circuit cleanly resets to open following non-tripping probe failures.")

    # 5. Multi-Threaded Concurrency Test (Contention Handling)
    print("\nEvaluating Thread Contention on Half-Open Boundaries...")
    cb_concurrent = CircuitBreaker("concurrency-cb", failure_threshold=1, recovery_timeout=0.1)
    
    try: cb_concurrent.call(lambda: _raise(IOError("Timeout")))
    except Exception: pass
    assert cb_concurrent.state == "open"
    
    time.sleep(0.15)
    
    results = []
    def Worker():
        res = cb_concurrent.call(lambda: (time.sleep(0.05), "SUCCESS")[1])
        results.append(res)

    threads = [threading.Thread(target=Worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert results.count("SUCCESS") == 1
    assert results.count(None) == 9
    print("  [PASS] Concurrency block isolation verified. 1 probe routed, 9 fast-failed.")
    
    print("\n" + "="*60)
    print("ALL MODULES VERIFIED GREEN: ARCHITECTURE IS PRODUCTION READY")
    print("="*60)

if __name__ == "__main__":
    run_tests()
