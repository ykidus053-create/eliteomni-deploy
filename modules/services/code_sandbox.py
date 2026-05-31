
"""
OmniSandbox — maximum language coverage, zero apt dependency.

Strategy per language:
  Python  — native subprocess          (always works)
  JS      — execjs pip package         (no node needed)
  Lua     — lupa pip package (LuaJIT)  (no lua needed)
  Bash    — /bin/bash                  (always on Linux)
  Perl    — already installed          (always on Ubuntu)
  C/C++   — gcc/g++ already present   (usually pre-installed)
  SQL     — sqlite3 stdlib             (built into Python)
  Ruby/R/PHP/Go/Java/Rust — graceful fallback: run via Python
    emulation or skip with clear message if binary missing
"""

import subprocess, tempfile, os, sys, re, resource, shutil
from typing import Optional

_CPU_SEC   = 60
_RAM_BYTES = 3 * 1024**3
_OUT_CAP   = 20_000

def _limits():
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SEC, _CPU_SEC))
        resource.setrlimit(resource.RLIMIT_AS,  (_RAM_BYTES, _RAM_BYTES))
    except Exception:
        pass

def _run_proc(cmd, stdin_data=None, timeout=_CPU_SEC, use_limits=True):
    try:
        proc = subprocess.run(
            cmd, input=stdin_data, capture_output=True, text=True,
            timeout=timeout, preexec_fn=_limits if use_limits else None
        )
        return {"stdout": proc.stdout[:_OUT_CAP], "stderr": proc.stderr[:_OUT_CAP//2],
                "success": proc.returncode == 0, "exit_code": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout":"","stderr":f"[TIMEOUT] {timeout}s","success":False,"exit_code":-1}
    except FileNotFoundError as e:
        return {"stdout":"","stderr":f"[NOT FOUND] {e}","success":False,"exit_code":-1}
    except Exception as e:
        return {"stdout":"","stderr":str(e),"success":False,"exit_code":-1}

def _tmpfile(code, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False, encoding="utf-8") as f:
        f.write(code); return f.name


# ══════════════════════════════════════════════════════════════
# PYTHON — always works, auto-install missing packages
# ══════════════════════════════════════════════════════════════
def run_python(code: str, timeout: int = _CPU_SEC) -> dict:
    header = "import warnings; warnings.filterwarnings('ignore')\n"
    tmp = _tmpfile(header + code, ".py")
    try:
        result = _run_proc([sys.executable, "-u", tmp], timeout=timeout)
        if not result["success"]:
            m = re.search(r"No module named '([\w\.]+)'", result["stderr"])
            if m:
                pkg = m.group(1).split(".")[0]
                print(f"  [AUTO-INSTALL] {pkg}")
                subprocess.run([sys.executable, "-m", "pip", "install", pkg,
                                "-q", "--break-system-packages"], capture_output=True)
                result = _run_proc([sys.executable, "-u", tmp], timeout=timeout)
                result["auto_installed"] = pkg
        return result
    finally:
        try: os.unlink(tmp)
        except: pass


# ══════════════════════════════════════════════════════════════
# JAVASCRIPT — execjs (pip) with fallback to node if present
# ══════════════════════════════════════════════════════════════
def run_node(code: str, timeout: int = _CPU_SEC) -> dict:
    # Try real node first
    node = next((p for p in [
        shutil.which("node"),
        shutil.which("nodejs"),
        os.path.expanduser("~/nodeenv/bin/node"),
        os.path.expanduser("~/nenv/bin/node"),
        "/usr/bin/node",
        "/usr/local/bin/node",
    ] if p and os.path.exists(p)), None)
    
    if node:
        tmp = _tmpfile(code, ".js")
        try:
            return _run_proc([node, tmp], timeout=timeout)
        finally:
            try: os.unlink(tmp)
            except: pass

    # Fallback: execjs via pip
    try:
        import execjs
        ctx = execjs.compile(code)
        # execjs needs a function call — wrap bare statements
        try:
            result = execjs.eval(code)
            return {"stdout": str(result), "stderr": "", "success": True, "exit_code": 0}
        except Exception:
            pass
        # wrap in IIFE
        wrapped = f"(function(){{ {code} }})()"
        result = execjs.eval(wrapped)
        return {"stdout": str(result) if result is not None else "",
                "stderr": "", "success": True, "exit_code": 0}
    except ImportError:
        return {"stdout":"","stderr":"[JS] Install node: sudo apt install nodejs","success":False,"exit_code":-1}
    except Exception as e:
        return {"stdout":"","stderr":str(e),"success":False,"exit_code":-1}


# ══════════════════════════════════════════════════════════════
# BASH — always available on Linux
# ══════════════════════════════════════════════════════════════
def run_bash(code: str, timeout: int = _CPU_SEC) -> dict:
    tmp = _tmpfile("#!/bin/bash\n" + code, ".sh")
    os.chmod(tmp, 0o755)
    try:
        return _run_proc(["bash", tmp], timeout=timeout, use_limits=False)
    finally:
        try: os.unlink(tmp)
        except: pass

def run_shell(cmd: str, timeout: int = 30) -> dict:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
        return {"stdout": proc.stdout[:_OUT_CAP], "stderr": proc.stderr[:_OUT_CAP//2],
                "success": proc.returncode == 0, "exit_code": proc.returncode}
    except Exception as e:
        return {"stdout":"","stderr":str(e),"success":False,"exit_code":-1}


# ══════════════════════════════════════════════════════════════
# LUA — lupa embeds LuaJIT (pip install lupa)
# ══════════════════════════════════════════════════════════════
def run_lua(code: str, timeout: int = _CPU_SEC) -> dict:
    # Try system lua first
    lua = shutil.which("lua5.4") or shutil.which("lua5.3") or shutil.which("lua")
    if lua:
        tmp = _tmpfile(code, ".lua")
        try:
            return _run_proc([lua, tmp], timeout=timeout)
        finally:
            try: os.unlink(tmp)
            except: pass
    # Fallback: lupa (LuaJIT embedded in Python)
    try:
        import lupa
        lua_rt = lupa.LuaRuntime(unpack_returned_tuples=True)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        # Redirect Lua print to Python stdout capture
        output = []
        def _py_print(*args):
            output.append("\t".join(str(a) for a in args))
        lua_rt.globals().py_print = _py_print
        lua_rt.execute("print = function(...) local t={...}; local s={}; for i,v in ipairs(t) do s[i]=tostring(v) end; py_print(table.concat(s,\"\\t\")) end")
        lua_rt.execute(code)
        return {"stdout": "\n".join(output), "stderr": "", "success": True, "exit_code": 0}
    except ImportError:
        return {"stdout":"","stderr":"[Lua] Run: pip install lupa --break-system-packages","success":False,"exit_code":-1}
    except Exception as e:
        return {"stdout":"","stderr":str(e),"success":False,"exit_code":-1}


# ══════════════════════════════════════════════════════════════
# PERL — pre-installed on Ubuntu
# ══════════════════════════════════════════════════════════════
def run_perl(code: str, timeout: int = _CPU_SEC) -> dict:
    perl = shutil.which("perl")
    if not perl:
        return {"stdout":"","stderr":"[RUNTIME] perl not found","success":False,"exit_code":-1}
    tmp = _tmpfile(code, ".pl")
    try:
        return _run_proc([perl, tmp], timeout=timeout)
    finally:
        try: os.unlink(tmp)
        except: pass


# ══════════════════════════════════════════════════════════════
# C / C++ — gcc/g++ usually pre-installed
# ══════════════════════════════════════════════════════════════
def run_c(code: str, timeout: int = _CPU_SEC) -> dict:
    gcc = shutil.which("gcc")
    if not gcc:
        return {"stdout":"","stderr":"[C] sudo apt install gcc","success":False,"exit_code":-1}
    src = _tmpfile(code, ".c"); exe = src[:-2]
    try:
        comp = _run_proc(["gcc", "-O2", "-o", exe, src, "-lm"], timeout=30)
        if not comp["success"]:
            return {"stdout":"","stderr":"[COMPILE]\n"+comp["stderr"],"success":False,"exit_code":-1}
        return _run_proc([exe], timeout=timeout)
    finally:
        for f in [src, exe]:
            try: os.unlink(f)
            except: pass

def run_cpp(code: str, timeout: int = _CPU_SEC) -> dict:
    gpp = shutil.which("g++")
    if not gpp:
        return {"stdout":"","stderr":"[C++] sudo apt install g++","success":False,"exit_code":-1}
    src = _tmpfile(code, ".cpp"); exe = src[:-4]
    try:
        comp = _run_proc(["g++", "-O2", "-std=c++17", "-o", exe, src, "-lm"], timeout=30)
        if not comp["success"]:
            return {"stdout":"","stderr":"[COMPILE]\n"+comp["stderr"],"success":False,"exit_code":-1}
        return _run_proc([exe], timeout=timeout)
    finally:
        for f in [src, exe]:
            try: os.unlink(f)
            except: pass


# ══════════════════════════════════════════════════════════════
# SQL — sqlite3 stdlib, zero dependencies
# ══════════════════════════════════════════════════════════════
def run_sql(code: str, timeout: int = _CPU_SEC) -> dict:
    import sqlite3, io
    out = io.StringIO()
    try:
        conn = sqlite3.connect(":memory:")
        cur  = conn.cursor()
        for stmt in [s.strip() for s in code.split(";") if s.strip()]:
            cur.execute(stmt)
            rows = cur.fetchall()
            if rows:
                cols = [d[0] for d in cur.description] if cur.description else []
                out.write("  ".join(cols) + "\n" + "-"*40 + "\n")
                for row in rows[:500]:
                    out.write("  ".join(str(c) for c in row) + "\n")
        conn.commit(); conn.close()
        return {"stdout": out.getvalue()[:_OUT_CAP], "stderr":"","success":True,"exit_code":0}
    except Exception as e:
        return {"stdout": out.getvalue(), "stderr": str(e), "success":False,"exit_code":-1}


# ══════════════════════════════════════════════════════════════
# RUBY / R / PHP / GO / RUST / JAVA
# Use binary if installed, otherwise clear message
# ══════════════════════════════════════════════════════════════
def _binary_runner(code, binary_names, suffix, compile_cmd=None,
                   run_cmd=None, timeout=_CPU_SEC, install_hint=""):
    binary = next((shutil.which(b) for b in binary_names if shutil.which(b)), None)
    if not binary:
        return {"stdout":"","stderr":f"[NOT INSTALLED] {install_hint}","success":False,"exit_code":-1}
    tmp = _tmpfile(code, suffix)
    try:
        if compile_cmd:
            exe = tmp.replace(suffix, "")
            cmd = [c.replace("__SRC__", tmp).replace("__EXE__", exe) for c in compile_cmd]
            comp = _run_proc(cmd, timeout=30)
            if not comp["success"]:
                return {"stdout":"","stderr":"[COMPILE]\n"+comp["stderr"],"success":False,"exit_code":-1}
            run = [c.replace("__EXE__", exe) for c in run_cmd]
            result = _run_proc(run, timeout=timeout)
            try: os.unlink(exe)
            except: pass
            return result
        run = [c.replace("__SRC__", tmp).replace("__BINARY__", binary) for c in run_cmd]
        return _run_proc(run, timeout=timeout)
    finally:
        try: os.unlink(tmp)
        except: pass

def run_ruby(code, timeout=_CPU_SEC):
    return _binary_runner(code, ["ruby"], ".rb",
        run_cmd=["__BINARY__", "__SRC__"], timeout=timeout,
        install_hint="sudo apt install ruby-full")

def run_r(code, timeout=_CPU_SEC):
    return _binary_runner(code, ["Rscript"], ".R",
        run_cmd=["__BINARY__", "--vanilla", "__SRC__"], timeout=timeout,
        install_hint="sudo apt install r-base")

def run_php(code, timeout=_CPU_SEC):
    if not code.strip().startswith("<?"):
        code = "<?php\n" + code
    return _binary_runner(code, ["php"], ".php",
        run_cmd=["__BINARY__", "__SRC__"], timeout=timeout,
        install_hint="sudo apt install php-cli")

def run_go(code, timeout=_CPU_SEC):
    go = shutil.which("go")
    if not go:
        return {"stdout":"","stderr":"[NOT INSTALLED] sudo apt install golang-go","success":False,"exit_code":-1}
    import tempfile as _tf
    d = _tf.mkdtemp(); src = os.path.join(d,"main.go"); exe = os.path.join(d,"main")
    open(src,"w").write(code)
    try:
        comp = _run_proc(["go","build","-o",exe,src], timeout=30)
        if not comp["success"]:
            return {"stdout":"","stderr":"[COMPILE]\n"+comp["stderr"],"success":False,"exit_code":-1}
        return _run_proc([exe], timeout=timeout)
    finally:
        shutil.rmtree(d, ignore_errors=True)

def run_rust(code, timeout=_CPU_SEC):
    rustc = shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc")
    if not os.path.exists(rustc):
        return {"stdout":"","stderr":"[NOT INSTALLED] curl https://sh.rustup.rs | sh","success":False,"exit_code":-1}
    src = _tmpfile(code, ".rs"); exe = src[:-3]
    try:
        comp = _run_proc([rustc, "-o", exe, src], timeout=60)
        if not comp["success"]:
            return {"stdout":"","stderr":"[COMPILE]\n"+comp["stderr"],"success":False,"exit_code":-1}
        return _run_proc([exe], timeout=timeout)
    finally:
        for f in [src, exe]:
            try: os.unlink(f)
            except: pass

def run_java(code, timeout=_CPU_SEC):
    javac = shutil.which("javac")
    if not javac:
        return {"stdout":"","stderr":"[NOT INSTALLED] sudo apt install default-jdk","success":False,"exit_code":-1}
    m = re.search(r"public class (\w+)", code)
    cls = m.group(1) if m else "Main"
    import tempfile as _tf
    d = _tf.mkdtemp(); src = os.path.join(d, f"{cls}.java")
    open(src,"w").write(code)
    try:
        comp = _run_proc(["javac", src], timeout=30)
        if not comp["success"]:
            return {"stdout":"","stderr":"[COMPILE]\n"+comp["stderr"],"success":False,"exit_code":-1}
        return _run_proc(["java","-cp",d,cls], timeout=timeout)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# UNIVERSAL ROUTER
# ══════════════════════════════════════════════════════════════
_LANG_MAP = {
    "python":run_python, "py":run_python,
    "node":run_node,     "js":run_node, "javascript":run_node,
    "bash":run_bash,     "sh":run_bash, "shell":run_bash,
    "ruby":run_ruby,     "rb":run_ruby,
    "lua":run_lua,
    "r":run_r,
    "php":run_php,
    "perl":run_perl,     "pl":run_perl,
    "c":run_c,
    "cpp":run_cpp,       "c++":run_cpp,
    "go":run_go,         "golang":run_go,
    "rust":run_rust,     "rs":run_rust,
    "java":run_java,
    "sql":run_sql,       "sqlite":run_sql,
}

def run(code: str, language: str = "python", timeout: int = _CPU_SEC) -> dict:
    lang = language.lower().strip()
    runner = _LANG_MAP.get(lang)
    if not runner:
        return {"stdout":"","stderr":f"Unknown language: {lang}. Available: {sorted(set(_LANG_MAP))}",
                "success":False,"exit_code":-1}
    return runner(code, timeout=timeout)

def detect_language(code: str) -> str:
    if re.search(r"^\s*(import |from .+ import |def .+:|print\()", code, re.MULTILINE): return "python"
    if re.search(r"console\.log|require\(|const |let |=>", code): return "node"
    if re.search(r"std::|#include <iostream>", code): return "cpp"
    if re.search(r"^#include", code, re.MULTILINE): return "c"
    if re.search(r"^package main|func main\(", code, re.MULTILINE): return "go"
    if re.search(r"fn main\(|println!", code): return "rust"
    if re.search(r"public class|System\.out\.println", code): return "java"
    if re.search(r"SELECT|INSERT|CREATE TABLE", code, re.IGNORECASE): return "sql"
    if re.search(r"^#!.*bash|echo |grep ", code, re.MULTILINE): return "bash"
    if re.search(r"puts |def .+\n.*end", code, re.MULTILINE): return "ruby"
    if re.search(r"^<\?php", code): return "php"
    if re.search(r"^use strict|\$\w+ =", code, re.MULTILINE): return "perl"
    return "python"

def run_auto(code: str, timeout: int = _CPU_SEC) -> dict:
    lang = detect_language(code)
    r = run(code, language=lang, timeout=timeout)
    r["detected_language"] = lang
    return r

def available_runtimes() -> dict:
    checks = {
        "python":  sys.executable,
        "node":    shutil.which("node") or shutil.which("nodejs") or "execjs(pip)",
        "bash":    shutil.which("bash"),
        "ruby":    shutil.which("ruby"),
        "lua":     shutil.which("lua5.4") or shutil.which("lua") or "lupa(pip)",
        "R":       shutil.which("Rscript"),
        "php":     shutil.which("php"),
        "perl":    shutil.which("perl"),
        "gcc/C":   shutil.which("gcc"),
        "g++/C++": shutil.which("g++"),
        "go":      shutil.which("go"),
        "rust":    shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc"),
        "java":    shutil.which("javac"),
        "sqlite":  "built-in",
    }
    return {k: v or "[not installed]" for k, v in checks.items()}

# legacy aliases
run_code_sandbox      = run_python
run_code_auto_install = run_python
def run_data_analysis(csv, q=""):
    code = ("import pandas as pd,numpy as np,io\n"
            "df=pd.read_csv(io.StringIO(" + repr(csv[:200000]) + "))\n"
            "print(df.describe(include='all').to_string())")
    return run_python(code)