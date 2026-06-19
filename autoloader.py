"""
Auto-wires all .py files in the app directory.
Drop a new file → it gets imported on next restart. No manual wiring.
"""
import os, ast, importlib, sys

_SKIP = {
    "app.py", "autoloader.py", "main.py", "config.py",
    "setup.py", "conftest.py", "manage.py",
}
_SKIP_PREFIXES = ("test_", ".", "_")

def autodiscover(base_dir: str, verbose: bool = True) -> dict:
    results = {}
    sys.path.insert(0, base_dir)

    for fname in sorted(os.listdir(base_dir)):
        if not fname.endswith(".py"): continue
        if fname in _SKIP: continue
        if any(fname.startswith(p) for p in _SKIP_PREFIXES): continue

        mod_name = fname[:-3]
        try:
            mod = importlib.import_module(mod_name)
            # Count top-level callables
            names = [n for n in dir(mod)
                     if not n.startswith("_") and callable(getattr(mod, n, None))]
            results[mod_name] = {"status": "ok", "exports": len(names)}
            if verbose:
                print(f"[autoloader] ✅ {mod_name} — {len(names)} exports")
        except Exception as e:
            results[mod_name] = {"status": "error", "error": str(e)}
            if verbose:
                print(f"[autoloader] ❌ {mod_name} — {e}")

    return results

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    autodiscover(base)
