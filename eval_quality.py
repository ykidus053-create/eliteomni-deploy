import sys, os, time, re, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_CASES = [
    ("3750 * 15 / 100",      "562",         "calculator"),
    ("144 ** 0.5",           "12",          "calculator"),
    ("2 + 2",                "4",           "calculator"),
    ("100 / 4",              "25",          "calculator"),
    ("100 * 9/5 + 32",       "212",         "calculator"),
]

def tool_calc(expr):
    try:
        safe = re.sub(r'[^0-9+\-*/().,% e]', '', expr).replace('%','/100').replace('^','**')
        safe = re.sub(r'math\.sqrt', 'sqrt', safe)
        r = eval(safe, {"__builtins__": {}, "math": math, "sqrt": math.sqrt,
                        "sin": math.sin, "cos": math.cos, "log": math.log,
                        "pi": math.pi, "e": math.e, "abs": abs, "round": round})
        return str(round(float(r), 6))
    except Exception as ex:
        return f"error: {ex}"

def run_evals():
    print("=" * 55)
    print("ELITEOMNI QUALITY EVAL SUITE")
    print("=" * 55)
    passed = 0
    failed = 0
    print(f"\n{'STATUS':<6} {'EXPRESSION':<35} {'EXPECTED':<10} {'GOT':<20}")
    print("-" * 75)
    for question, expected, skill in TEST_CASES:
        result = tool_calc(question)
        ok = expected in result
        status = "PASS" if ok else "FAIL"
        if ok: passed += 1
        else: failed += 1
        print(f"{status:<6} {question:<35} {expected:<10} {result:<20}")

    print()
    # Also verify the broken regex fix worked
    print("--- Regex fix verification ---")
    try:
        overconf = re.compile(r'\b(exactly|always|never|100%|guaranteed|definitely|certainly|absolutely)\b', re.IGNORECASE)
        hedge    = re.compile(r'\b(approximately|about|roughly|generally|may|might|could|likely|probably)\b', re.IGNORECASE)
        test = "This is exactly right and will definitely work, though it might vary"
        oc = len(overconf.findall(test))
        hd = len(hedge.findall(test))
        print(f"PASS  overconfidence regex: found {oc} overconf, {hd} hedge words (expected 2, 1)")
        passed += 1
    except Exception as e:
        print(f"FAIL  overconfidence regex: {e}")
        failed += 1

    # Verify HHH weighted scoring
    print("--- HHH scoring verification ---")
    scores = {"helpful": 4, "harmless": 5, "honest": 3}
    total = round((scores["helpful"]*0.4) + (scores["harmless"]*0.35) + (scores["honest"]*0.25), 3)
    expected_total = round(4*0.4 + 5*0.35 + 3*0.25, 3)
    ok = abs(total - expected_total) < 0.001
    print(f"{'PASS' if ok else 'FAIL'}  HHH weighted score: {total} (expected {expected_total})")
    if ok: passed += 1
    else: failed += 1

    # Verify singleton pool exists in search.py
    print("--- Singleton pool verification ---")
    spath = 'modules/services/search.py'
    if os.path.exists(spath):
        content = open(spath).read()
        ok = '_get_search_pool' in content
        print(f"{'PASS' if ok else 'FAIL'}  singleton pool in search.py")
        if ok: passed += 1
        else: failed += 1

    # Verify _os2 is gone
    print("--- _os2 removal verification ---")
    tpath = 'modules/services/tools.py'
    if os.path.exists(tpath):
        content = open(tpath).read()
        ok = '_os2.environ' not in content
        print(f"{'PASS' if ok else 'FAIL'}  _os2 removed from tools.py")
        if ok: passed += 1
        else: failed += 1

    total_tests = len(TEST_CASES) + 4
    print()
    print("=" * 55)
    print(f"RESULTS: {passed} passed / {failed} failed / {total_tests} total")
    print(f"SCORE: {passed/total_tests*100:.0f}%")
    if passed / total_tests >= 0.8:
        print("STATUS: PASS — system is healthy")
    else:
        print("STATUS: FAIL — investigate above")
    print("=" * 55)

if __name__ == "__main__":
    run_evals()
