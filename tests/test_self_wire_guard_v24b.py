from __future__ import annotations

import importlib
import sqlite3
import sys
import threading
import types


def _reload(monkeypatch, enabled=None, sft=None):
    if enabled is None:
        monkeypatch.delenv("ELITE_SELF_WIRE", raising=False)
    else:
        monkeypatch.setenv("ELITE_SELF_WIRE", enabled)

    if sft is None:
        monkeypatch.delenv("ELITE_SELF_WIRE_SFT", raising=False)
    else:
        monkeypatch.setenv("ELITE_SELF_WIRE_SFT", sft)

    import self_wire
    return importlib.reload(self_wire)


def test_disabled_by_default(monkeypatch):
    module = _reload(monkeypatch)
    assert module.is_enabled() is False
    assert module.start(interval=0.01) is None
    assert module.status()["started"] is False


def test_mutation_and_training_scripts_are_never_watched(monkeypatch):
    module = _reload(monkeypatch)
    for filename in (
        "fix_errors.py",
        "split_modules.py",
        "wire_orphans.py",
        "fix_rendermd.py",
        "fast_trainer.py",
        "mistral_finetune.py",
        "synthetic_trainer.py",
    ):
        assert module._should_watch(filename) is False


def test_start_is_singleton(monkeypatch):
    module = _reload(monkeypatch, enabled="1")
    calls = []

    class DummyThread:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("name"))
        def start(self):
            return None

    monkeypatch.setattr(threading, "Thread", DummyThread)

    first = module.start(interval=99)
    second = module.start(interval=99)

    assert first is not None
    assert second is None
    assert calls == ["self_wire"]


def test_reindex_creates_missing_knowledge_table(monkeypatch, tmp_path):
    module = _reload(monkeypatch, enabled="1")
    db_path = tmp_path / "knowledge.db"

    fake = types.ModuleType("knowledge_rag")
    fake._DB = str(db_path)
    fake._cache = {}
    fake._extract_chunks = lambda name: [
        {
            "module": name,
            "name": "demo",
            "kind": "function",
            "doc": "demo",
            "signature": "()",
            "chunk": "def demo(): pass",
        }
    ]
    monkeypatch.setitem(sys.modules, "knowledge_rag", fake)

    module._reindex_knowledge("sample", "sample.py")

    con = sqlite3.connect(db_path)
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM knowledge WHERE module='sample'"
        ).fetchone()[0]
    finally:
        con.close()

    assert count == 1


def test_sft_generation_requires_separate_opt_in(monkeypatch, tmp_path):
    module = _reload(monkeypatch, enabled="1", sft="0")
    source = tmp_path / "sample.py"
    source.write_text(
        'def demo():\n'
        '    """This documentation is sufficiently long for a demo."""\n'
        '    return 1\n',
        encoding="utf-8",
    )

    module._extract_sft_demo("sample", str(source))
    assert module.status()["sft_enabled"] is False
