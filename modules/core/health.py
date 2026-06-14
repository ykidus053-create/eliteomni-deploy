"""
Real health checks for all dependencies.
"""
import time, os, sqlite3, json, urllib.request

try:
    from debug_patch import _real_urlopen
except ImportError:
    _real_urlopen = urllib.request.urlopen

def check_mistral() -> dict:
    key = os.environ.get("MISTRAL_API_KEY","")
    if not key: return {"ok": False, "reason": "MISTRAL_API_KEY not set"}
    models = [
        "mistral-small-latest",
        "mistral-code-agent-latest",
        "magistral-medium-latest",
        "mistral-ocr-latest",
    ]
    results = {}
    for model in models:
        try:
            payload = json.dumps({"model": model,
                                  "messages":[{"role":"user","content":"hi"}],
                                  "max_tokens":1}).encode()
            req = urllib.request.Request(
                "https://api.mistral.ai/v1/chat/completions",
                data=payload,
                headers={"Authorization":f"Bearer {key}",
                         "Content-Type":"application/json"})
            with _real_urlopen(req, timeout=5) as r:
                results[model] = {"ok": r.status == 200}
        except Exception as e:
            results[model] = {"ok": False, "reason": str(e)[:80]}
    all_ok = all(v["ok"] for v in results.values())
    return {"ok": all_ok, "models": results}

def check_sqlite() -> dict:
    db = os.path.expanduser("~/eliteomni_memory.db")
    try:
        con = sqlite3.connect(db, timeout=3)
        con.execute("INSERT OR REPLACE INTO kv (key,value,ts) VALUES (?,?,?)",
                    ("__health__", "ok", time.time()))
        con.commit()
        con.execute("DELETE FROM kv WHERE key='__health__'")
        con.commit(); con.close()
        return {"ok": True, "path": db}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:80]}

def check_faiss() -> dict:
    try:
        import faiss, numpy as np
        idx = faiss.IndexFlatIP(4)
        idx.add(np.zeros((1,4), dtype="float32"))
        return {"ok": True, "ntotal": idx.ntotal}
    except ImportError:
        return {"ok": False, "reason": "faiss not installed"}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:80]}

def check_searxng() -> dict:
    url = os.environ.get("SEARXNG_URL","http://localhost:8888")
    try:
        with _real_urlopen(f"{url}/healthz", timeout=3) as r:
            return {"ok": r.status == 200, "url": url}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:60], "url": url}

def check_chroma() -> dict:
    try:
        import chromadb
        cc = chromadb.PersistentClient(path=os.path.expanduser("~/eliteomni_chroma"))
        cc.get_or_create_collection("__health__")
        return {"ok": True, "collections": len(cc.list_collections())}
    except ImportError:
        return {"ok": False, "reason": "chromadb not installed"}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:80]}

def full_health() -> dict:
    t0 = time.time()
    checks = {
        "sqlite":  check_sqlite(),
        "faiss":   check_faiss(),
        "searxng": check_searxng(),
        "chroma":  check_chroma(),
        "mistral": check_mistral(),
    }
    all_ok  = all(v["ok"] for v in checks.values())
    elapsed = round((time.time() - t0) * 1000)
    return {
        "status":     "healthy" if all_ok else "degraded",
        "checks":     checks,
        "elapsed_ms": elapsed,
        "ts":         time.time(),
    }
