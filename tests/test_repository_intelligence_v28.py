from pathlib import Path

from modules.repository_intelligence_v28 import (
    analyze_repository,
    build_repository_index,
    format_repository_impact,
)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app.py").write_text(
        "from pkg.service import run\n"
        "def main():\n"
        "    return run()\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "service.py").write_text(
        "from pkg.util import normalize\n"
        "def run():\n"
        "    return normalize('value')\n",
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "util.py").write_text(
        "def normalize(value: str) -> str:\n"
        "    return value.strip().lower()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "from pkg.service import run\n"
        "def test_run():\n"
        "    assert run() == 'value'\n",
        encoding="utf-8",
    )
    return tmp_path


def test_dependency_edges(tmp_path):
    index = build_repository_index(_repo(tmp_path), force=True)
    assert "pkg/util.py" in index.dependencies["pkg/service.py"]
    assert "pkg/service.py" in index.dependents["pkg/util.py"]


def test_traceback_finds_impact_and_tests(tmp_path):
    result = analyze_repository(
        'Traceback File "pkg/service.py", line 3, in run',
        root=_repo(tmp_path),
        max_files=8,
    )
    selected = {item["path"] for item in result["files"]}
    assert "pkg/service.py" in selected
    assert "pkg/util.py" in selected
    assert "tests/test_service.py" in selected


def test_symbol_finds_definition_and_caller(tmp_path):
    result = analyze_repository(
        "Fix normalize without breaking callers",
        root=_repo(tmp_path),
        max_files=8,
    )
    selected = {item["path"] for item in result["files"]}
    assert "pkg/util.py" in selected
    assert "pkg/service.py" in selected
    assert "normalize" in result["symbols"]


def test_formatted_map(tmp_path):
    output = format_repository_impact(
        "Fix pkg/service.py",
        root=_repo(tmp_path),
    )
    assert "REPOSITORY CHANGE-IMPACT MAP" in output
    assert "pkg/service.py" in output
    assert "Likely regression tests" in output


def test_cache_reuse(tmp_path):
    root = _repo(tmp_path)
    first = build_repository_index(root, force=True)
    second = build_repository_index(root)
    assert first is second
