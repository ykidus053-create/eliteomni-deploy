"""
Self-Wiring Pipeline
Any .py file change → hot-reload → re-index knowledge → inject into prompts
Closes the loop: you write code → model immediately knows about it
"""
import os, sys, time, hashlib, importlib, threading, sqlite3, ast

_BASE = os.path.dirname(os.path.abspath(__file__))
_SKIP = {"app.py", "self_wire.py", "main.py", "config.py", "hot_reload.py", "autoloader.py", "debug_patch.py"}
_SKIP_PREFIXES = ("test_", ".", "_")
_watched: dict = {}  # fname -> hash
_lock = threading.Lock()
_change_callbacks = []  # functions to call on any change

def on_change(fn):
    """Decorator — register a callback for any file change."""
    _change_callbacks.append(fn)
    return fn

def _hash(path):
    try:
        return hashlib.md5(open(path,"rb").read()).hexdigest()
    except: return ""

def _should_watch(fname):
    if not fname.endswith(".py"): return False
    if fname in _SKIP: return False
    if any(fname.startswith(p) for p in _SKIP_PREFIXES): return False
    return True

def _reload(fname):
    path = os.path.join(_BASE, fname)
    h = _hash(path)
    if _watched.get(fname) == h: return False
    mod = fname[:-3]
    with _lock:
        try:
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])
                action = "reloaded"
            else:
                sys.path.insert(0, _BASE)
                importlib.import_module(mod)
                action = "loaded"
            _watched[fname] = h
            print(f"[self_wire] ✅ {action}: {mod}")
            # Fire callbacks in background — don't block reload
            def _run_callbacks(m=mod, p=path):
                for cb in _change_callbacks:
                    try: cb(m, p)
                    except Exception as e: print(f"[self_wire] callback error: {e}")
            threading.Thread(target=_run_callbacks, daemon=True).start()
            return True
        except Exception as e:
            _watched[fname] = h
            print(f"[self_wire] ❌ {mod}: {e}")
            return False

def _scan():
    files = [f for f in os.listdir(_BASE) if _should_watch(f)]
    threads = [threading.Thread(target=_reload, args=(f,), daemon=True) for f in files]
    for t in threads: t.start()
    for t in threads: t.join(timeout=10)  # 10s max per module

def _watch_loop(interval=30.0):
    print(f"[self_wire] 👁  watching {_BASE}")
    while True:
        try: _scan()
        except Exception as e: print(f"[self_wire] scan error: {e}")
        time.sleep(interval)

# ── Callbacks that fire on every change ──────────────────────────────────────

@on_change
def _reindex_knowledge(mod_name, path):
    """Re-index the changed module into knowledge RAG immediately."""
    try:
        from knowledge_rag import _extract_chunks, _DB, _cache
        chunks = _extract_chunks(mod_name)
        if not chunks: return
        _cache.clear()  # invalidate retrieval cache
        con = sqlite3.connect(_DB)
        con.execute("DELETE FROM knowledge WHERE module=?", (mod_name,))
        for c in chunks:
            con.execute(
                "INSERT INTO knowledge(module,name,kind,doc,signature,chunk) VALUES(?,?,?,?,?,?)",
                (c["module"],c["name"],c["kind"],c["doc"],c["signature"],c["chunk"])
            )
        con.commit(); con.close()
        print(f"[self_wire] 📚 re-indexed {mod_name}: {len(chunks)} chunks")
    except Exception as e:
        print(f"[self_wire] reindex error: {e}")

@on_change
def _log_change(mod_name, path):
    """Log every change for the model improvement audit trail."""
    try:
        con = sqlite3.connect(os.path.expanduser("~/eliteomni_changes.db"))
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""CREATE TABLE IF NOT EXISTS changes (
            ts REAL, module TEXT, path TEXT, lines INTEGER
        )""")
        lines = len(open(path).readlines())
        con.execute("INSERT INTO changes VALUES(?,?,?,?)",
                    (time.time(), mod_name, path, lines))
        con.execute("DELETE FROM changes WHERE id NOT IN "
                    "(SELECT id FROM changes ORDER BY ts DESC LIMIT 1000)")
        con.commit(); con.close()
    except: pass

@on_change  
def _extract_sft_demo(mod_name, path):
    """
    Auto-generate SFT training demo from new functions.
    New function = new (instruction, response) pair for fine-tuning.
    """
    try:
        tree = ast.parse(open(path).read())
        demos = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
            if node.col_offset != 0: continue
            if not node.body: continue
            doc = ast.get_docstring(node) or ""
            if len(doc) < 10: continue
            name = node.name
            demo = {
                "instruction": f"Implement or explain: {doc[:200]}",
                "response": f"Here is the implementation of `{name}`: {doc}",
                "source": mod_name,
                "ts": time.time()
            }
            demos.append(demo)
        if not demos: return
        # Append to SFT store
        import json
        sft_path = os.path.expanduser("~/eliteomni_sft_auto.jsonl")
        with open(sft_path, "a") as f:
            for d in demos:
                f.write(json.dumps(d) + "\n")
        print(f"[self_wire] 🎓 {len(demos)} SFT demos generated from {mod_name}")
    except Exception as e:
        print(f"[self_wire] SFT error: {e}")

def start(interval=30.0):
    # Run initial scan in background so uvicorn startup is not blocked
    threading.Thread(target=_scan, daemon=True, name="self_wire_init").start()
    t = threading.Thread(target=_watch_loop, args=(interval,),
                         daemon=True, name="self_wire")
    t.start()
    print("[self_wire] ✅ self-wiring pipeline active")
    return t

if __name__ == "__main__":
    start()
    while True: time.sleep(60)
