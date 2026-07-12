"""Safe, deployment-aware project context for coding prompts."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _resolve_root() -> Path:
    configured = os.getenv("ELITE_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "app.py").exists():
        return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "app.py").exists():
        return cwd

    return candidate


ROOT = str(_resolve_root())
_CACHE: tuple[float, tuple[int, int], str] | None = None


def _run(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            args,
            cwd=str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return ""


def _signature(root: Path) -> tuple[int, int]:
    count = 0
    newest = 0
    for path in root.rglob("*.py"):
        if any(
            part in {".git", ".venv", "venv", "__pycache__", "site-packages"}
            for part in path.parts
        ):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        newest = max(newest, stat.st_mtime_ns)
    return count, newest


def infer_project_context() -> str:
    global _CACHE

    root = _resolve_root()
    signature = _signature(root)
    ttl = max(5, int(os.getenv("ELITE_PROJECT_CONTEXT_TTL_SECONDS", "60")))
    now = time.monotonic()

    if _CACHE and now < _CACHE[0] and signature == _CACHE[1]:
        return _CACHE[2]

    app_path = root / "app.py"
    try:
        app_source = app_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        app_source = ""

    package_names: list[str] = []
    try:
        package_names = sorted(
            {
                dist.metadata.get("Name", "")
                for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")
            }
        )[:80]
    except Exception:
        pass

    facts = [
        f"Repository root: {root}",
        f"Runtime: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"Platform: {platform.system()} {platform.machine()}",
        f"Python source files: {signature[0]}",
    ]

    if "FastAPI" in app_source:
        facts.append("Framework: FastAPI / ASGI")
    if "async def" in app_source:
        facts.append("Concurrency: mixed async and thread-pool execution")
    if (root / "pytest.ini").exists() or (root / "tests").exists():
        facts.append("Tests: pytest")
    if (root / "Procfile").exists():
        facts.append("Deployment: Procfile-based service")
    if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"):
        facts.append("Hosting: Railway")
    if package_names:
        facts.append("Installed packages: " + ", ".join(package_names[:40]))

    branch = _run(["git", "branch", "--show-current"], root)
    commit = _run(["git", "rev-parse", "--short", "HEAD"], root)
    if branch:
        facts.append(f"Git branch: {branch}")
    if commit:
        facts.append(f"Git commit: {commit}")

    result = "\n".join(facts)
    _CACHE = (now + ttl, signature, result)
    return result
