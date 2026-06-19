from __future__ import annotations
import os, re, subprocess
from functools import lru_cache

ROOT = os.path.expanduser("~/eliteomni_app")

def _read(path: str, maxbytes: int = 4000) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read(maxbytes)
    except Exception:
        return ""

def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=3).strip()
    except Exception:
        return ""

@lru_cache(maxsize=1)
def infer_project_context() -> str:
    facts = []
    pyver = _run("python3 --version")
    if pyver: facts.append(f"Runtime: {pyver}")

    pip_list = _run("pip list --format=columns 2>/dev/null | awk '{print $1}' | tr '\n' ' '")
    if pip_list: facts.append(f"Installed packages: {pip_list[:400]}")

    app_src = _read(os.path.join(ROOT, "app.py"))
    if "FastAPI" in app_src: facts.append("Framework: FastAPI (async)")
    if "aiohttp" in app_src: facts.append("HTTP client: aiohttp (async)")

    async_count = app_src.count("async def")
    sync_count  = app_src.count("\ndef ")
    facts.append("Convention: async-first — all I/O must use await" if async_count > sync_count
                 else "Convention: mixed sync/async — check before adding await")

    if re.search(r'print\(f"\[', app_src):
        facts.append("Logging: print() with [Module] prefix convention")
    elif "loguru" in pip_list:
        facts.append("Logging: loguru")

    uname = _run("uname -a")
    if "microsoft" in uname.lower() or "wsl" in uname.lower():
        facts.append("OS: Ubuntu/WSL — bash only, no Windows paths")

    if "chromadb" in pip_list: facts.append("Vector DB: ChromaDB")
    if "sqlite" in app_src.lower(): facts.append("Relational DB: SQLite")

    modules = _run(f"ls {ROOT}/modules/*.py 2>/dev/null | xargs -I{{}} basename {{}} .py | tr '\n' ', '")
    if modules: facts.append(f"Modules: {modules.strip(', ')}")

    req = _read(os.path.join(ROOT, "requirements.txt"), 600)
    if req: facts.append(f"Pinned deps:\n{req}")

    block = "INFERRED PROJECT CONTEXT (live codebase scan):\n"
    block += "\n".join(f"  - {f}" for f in facts)
    block += "\nAll code must be consistent with the above. No exceptions."
    return block

def get_project_context() -> str:
    return infer_project_context()
