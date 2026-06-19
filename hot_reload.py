"""
Hot-reload watcher — monitors all .py files in the app dir.
Any change (save, new file, edit) → auto-reimports the module instantly.
No restart needed.
"""
import os, sys, time, importlib, threading, hashlib

_SKIP = {"app.py", "hot_reload.py", "main.py", "config.py"}
_SKIP_PREFIXES = ("test_", ".", "_")
_watched: dict = {}  # mod_name -> last_hash
_lock = threading.Lock()

def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""

def _should_watch(fname: str) -> bool:
    if not fname.endswith(".py"): return False
    if fname in _SKIP: return False
    if any(fname.startswith(p) for p in _SKIP_PREFIXES): return False
    return True

def _reload_module(base_dir: str, fname: str):
    mod_name = fname[:-3]
    path = os.path.join(base_dir, fname)
    new_hash = _file_hash(path)
    old_hash = _watched.get(mod_name, "")
    if new_hash == old_hash:
        return
    with _lock:
        try:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
                action = "reloaded"
            else:
                sys.path.insert(0, base_dir)
                importlib.import_module(mod_name)
                action = "loaded"
            _watched[mod_name] = new_hash
            print(f"[hot_reload] ✅ {action}: {mod_name}")
        except Exception as e:
            _watched[mod_name] = new_hash  # don't retry until next change
            print(f"[hot_reload] ❌ {mod_name}: {e}")

def _watch_loop(base_dir: str, interval: float = 1.0):
    print(f"[hot_reload] 👁  watching {base_dir} every {interval}s")
    while True:
        try:
            for fname in os.listdir(base_dir):
                if _should_watch(fname):
                    _reload_module(base_dir, fname)
        except Exception as e:
            print(f"[hot_reload] scan error: {e}")
        time.sleep(interval)

def start(base_dir: str = None, interval: float = 1.0):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    # Initial load of all modules
    for fname in sorted(os.listdir(base_dir)):
        if _should_watch(fname):
            _reload_module(base_dir, fname)
    # Background watcher thread
    t = threading.Thread(target=_watch_loop, args=(base_dir, interval),
                         daemon=True, name="hot_reload")
    t.start()
    return t

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    start(base)
    while True:
        time.sleep(60)
