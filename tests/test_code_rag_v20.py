from pathlib import Path

import code_rag


def test_symbol_and_traceback_retrieval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("alpha.py").write_text(
        "def unrelated():\n    return 1\n\n"
        "def compress_history(items):\n"
        "    return items[-3:]\n",
        encoding="utf-8",
    )
    Path("beta.py").write_text(
        "class PaymentLedger:\n"
        "    def rollback(self):\n"
        "        return 'rolled back'\n",
        encoding="utf-8",
    )
    code_rag.invalidate_index()

    result = code_rag.get_relevant_code_context(
        "UnboundLocalError in compress_history alpha.py:4",
        top_k=3,
    )
    assert "alpha.py" in result
    assert "compress_history" in result
    assert "score=" in result


def test_ignored_directories_are_not_indexed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path(".venv").mkdir()
    Path(".venv/secret.py").write_text(
        "def forbidden_symbol(): pass",
        encoding="utf-8",
    )
    Path("visible.py").write_text(
        "def visible_symbol(): return True",
        encoding="utf-8",
    )
    code_rag.invalidate_index()
    result = code_rag.get_relevant_code_context(
        "visible_symbol",
        top_k=3,
    )
    assert "visible.py" in result
    assert "secret.py" not in result
