"""
Production-safe self-wiring pipeline.

The historical implementation imported every top-level Python file during
application startup. Some repository files are executable maintenance scripts,
so importing them rewrote source files, generated training data, and referenced
developer-only paths inside the Railway web process.

The watcher is now disabled unless ELITE_SELF_WIRE=1 is explicitly configured.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


_BASE = os.path.dirname(os.path.abspath(__file__))

_SKIP = {
    "app.py",
    "self_wire.py",
    "main.py",
    "config.py",
    "hot_reload.py",
    "autoloader.py",
    "debug_patch.py",
}

# Files with top-level mutation, migration, repair, training, or developer-path
# behavior must never be imported by a web-process file watcher.
_PRODUCTION_SKIP = {
    "apply_fixes.py",
    "fast_trainer.py",
    "finetune.py",
    "fix_accuracy.py",
    "fix_all.py",
    "fix_errors.py",
    "fix_rendermd.py",
    "integrate_upgrades.py",
    "mistral_finetune.py",
    "split_modules.py",
    "synthetic_trainer.py",
    "wire_orphans.py",
}

_SKIP_PREFIXES = ("test_", ".", "_")
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}

_watched: dict[str, str] = {}
_lock = threading.Lock()
_start_lock = threading.Lock()
_change_callbacks: list[Callable[[str, str], Any]] = []
_started = False
_scan_count = 0
_last_error: str | None = None


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def is_enabled() -> bool:
    """Return whether development self-wiring was explicitly enabled."""
    return _env_enabled("ELITE_SELF_WIRE", False)


def status() -> dict[str, Any]:
    """Return non-secret watcher diagnostics."""
    return {
        "enabled": is_enabled(),
        "started": _started,
        "sft_enabled": _env_enabled("ELITE_SELF_WIRE_SFT", False),
        "base": _BASE,
        "watched_files": len(_watched),
        "scan_count": _scan_count,
        "last_error": _last_error,
    }


def on_change(fn: Callable[[str, str], Any]) -> Callable[[str, str], Any]:
    """Register a callback for a successfully imported file change."""
    _change_callbacks.append(fn)
    return fn


def _hash(path: str) -> str:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _should_watch(fname: str) -> bool:
    if not fname.endswith(".py"):
        return False
    if fname in _SKIP or fname in _PRODUCTION_SKIP:
        return False
    if any(fname.startswith(prefix) for prefix in _SKIP_PREFIXES):
        return False
    return True


def _run_callbacks(mod_name: str, path: str) -> None:
    for callback in tuple(_change_callbacks):
        try:
            callback(mod_name, path)
        except Exception as exc:
            print(f"[self_wire] callback error: {exc}")


def _reload(fname: str) -> bool:
    global _last_error

    if not _should_watch(fname):
        return False

    path = os.path.join(_BASE, fname)
    digest = _hash(path)
    if not digest or _watched.get(fname) == digest:
        return False

    module_name = fname[:-3]

    with _lock:
        try:
            if _BASE not in sys.path:
                sys.path.insert(0, _BASE)

            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                action = "reloaded"
            else:
                importlib.import_module(module_name)
                action = "loaded"

            _watched[fname] = digest
            _last_error = None
            print(f"[self_wire] loaded safely: {module_name} ({action})")

            threading.Thread(
                target=_run_callbacks,
                args=(module_name, path),
                daemon=True,
                name=f"self_wire_callbacks_{module_name}",
            ).start()
            return True
        except Exception as exc:
            _watched[fname] = digest
            _last_error = f"{module_name}: {exc}"
            print(f"[self_wire] skipped {module_name}: {exc}")
            return False


def _scan() -> None:
    global _scan_count, _last_error

    if not is_enabled():
        return

    try:
        files = sorted(
            fname for fname in os.listdir(_BASE) if _should_watch(fname)
        )
        # Sequential loading is deliberate: concurrent arbitrary imports can
        # race over module state and source files.
        for fname in files:
            _reload(fname)
        _scan_count += 1
    except Exception as exc:
        _last_error = str(exc)
        print(f"[self_wire] scan error: {exc}")


def _watch_loop(interval: float = 30.0) -> None:
    interval = max(float(interval), 5.0)
    print(f"[self_wire] development watcher active for {_BASE}")
    while is_enabled():
        _scan()
        time.sleep(interval)
    print("[self_wire] development watcher stopped")


@on_change
def _reindex_knowledge(mod_name: str, path: str) -> None:
    """Re-index a changed module only in explicitly enabled development mode."""
    if not is_enabled():
        return

    try:
        from knowledge_rag import _DB, _cache, _extract_chunks

        chunks = _extract_chunks(mod_name)
        if not chunks:
            return

        db_path = os.path.expanduser(str(_DB))
        con = sqlite3.connect(db_path)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    doc TEXT,
                    signature TEXT,
                    chunk TEXT NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_module "
                "ON knowledge(module)"
            )
            con.execute(
                "DELETE FROM knowledge WHERE module=?",
                (mod_name,),
            )
            con.executemany(
                """
                INSERT INTO knowledge(
                    module, name, kind, doc, signature, chunk
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.get("module", mod_name),
                        chunk.get("name", ""),
                        chunk.get("kind", ""),
                        chunk.get("doc", ""),
                        chunk.get("signature", ""),
                        chunk.get("chunk", ""),
                    )
                    for chunk in chunks
                ],
            )
            con.commit()
        finally:
            con.close()

        try:
            _cache.clear()
        except Exception:
            pass

        print(f"[self_wire] re-indexed {mod_name}: {len(chunks)} chunks")
    except Exception as exc:
        print(f"[self_wire] reindex skipped: {exc}")


@on_change
def _log_change(mod_name: str, path: str) -> None:
    """Keep a bounded local development audit trail."""
    if not is_enabled():
        return

    try:
        db_path = os.path.expanduser("~/eliteomni_changes.db")
        con = sqlite3.connect(db_path)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    module TEXT NOT NULL,
                    path TEXT NOT NULL,
                    lines INTEGER NOT NULL
                )
                """
            )
            try:
                lines = len(Path(path).read_text(encoding="utf-8").splitlines())
            except OSError:
                lines = 0
            con.execute(
                "INSERT INTO changes(ts,module,path,lines) VALUES(?,?,?,?)",
                (time.time(), mod_name, path, lines),
            )
            con.execute(
                """
                DELETE FROM changes
                WHERE id NOT IN (
                    SELECT id FROM changes ORDER BY ts DESC LIMIT 1000
                )
                """
            )
            con.commit()
        finally:
            con.close()
    except Exception:
        pass


@on_change
def _extract_sft_demo(mod_name: str, path: str) -> None:
    """Generate development SFT examples only when separately enabled."""
    if not is_enabled() or not _env_enabled("ELITE_SELF_WIRE_SFT", False):
        return

    try:
        source = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        demos: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.col_offset != 0 or not node.body:
                continue

            doc = ast.get_docstring(node) or ""
            if len(doc) < 10:
                continue

            demos.append(
                {
                    "instruction": f"Implement or explain: {doc[:200]}",
                    "response": (
                        f"Here is the implementation of `{node.name}`: {doc}"
                    ),
                    "source": mod_name,
                    "ts": time.time(),
                }
            )

        if not demos:
            return

        sft_path = Path(
            os.path.expanduser("~/eliteomni_sft_auto.jsonl")
        )
        with sft_path.open("a", encoding="utf-8") as handle:
            for demo in demos:
                handle.write(json.dumps(demo, ensure_ascii=False) + "\n")

        print(f"[self_wire] generated {len(demos)} SFT demos from {mod_name}")
    except Exception as exc:
        print(f"[self_wire] SFT skipped: {exc}")


def start(interval: float = 30.0) -> threading.Thread | None:
    """
    Start the watcher only when ELITE_SELF_WIRE=1.

    Repeated calls are idempotent. Production and tests are safe by default.
    """
    global _started

    if not is_enabled():
        print(
            "[self_wire] disabled "
            "(keep ELITE_SELF_WIRE=0 in production)"
        )
        return None

    with _start_lock:
        if _started:
            print("[self_wire] already active")
            return None
        _started = True

    watcher = threading.Thread(
        target=_watch_loop,
        args=(interval,),
        daemon=True,
        name="self_wire",
    )
    watcher.start()
    return watcher


if __name__ == "__main__":
    thread = start()
    if thread is not None:
        while thread.is_alive():
            time.sleep(60)
