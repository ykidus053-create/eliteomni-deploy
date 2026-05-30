"""
claude_code.py — Claude Code knowledge & context features
"""
from __future__ import annotations
import ast, fnmatch, functools, os, re, subprocess, sys, time
from pathlib import Path
from typing import Optional

# ── 1. CLAUDE.md ─────────────────────────────────────────────────────────────
_CLAUDE_MD_CACHE: dict = {}
_CLAUDE_MD_TTL = 30.0

def _find_project_root(start: str = ".") -> Optional[Path]:
    p = Path(start).resolve()
    vcs_root = None
    for ancestor in [p, *p.parents]:
        if (ancestor / ".claude" / "CLAUDE.md").exists():
            return ancestor
        if any((ancestor / m).exists() for m in (".git", ".hg", ".svn")):
            vcs_root = ancestor
    return vcs_root

def load_claude_md(project_root: Optional[str] = None) -> str:
    root = Path(project_root).resolve() if project_root else _find_project_root()
    if root is None: return ""
    md_path = root / ".claude" / "CLAUDE.md"
    if not md_path.exists(): return ""
    mtime = md_path.stat().st_mtime
    cached = _CLAUDE_MD_CACHE.get(str(md_path))
    if cached and (time.time() - cached[0]) < _CLAUDE_MD_TTL and cached[0] >= mtime:
        return cached[1]
    try:
        content = md_path.read_text(encoding="utf-8")
        _CLAUDE_MD_CACHE[str(md_path)] = (time.time(), content)
        print(f"[CLAUDE.md] loaded {len(content)} chars from {md_path}")
        return content
    except Exception as e:
        print(f"[CLAUDE.md] read error: {e}"); return ""

def inject_claude_md(system: str, project_root: Optional[str] = None) -> str:
    rules = load_claude_md(project_root)
    if not rules: return system
    return system + f"\n\n<project_rules>\n{rules.strip()}\n</project_rules>"

def update_claude_md(rule: str, project_root: Optional[str] = None) -> str:
    root = Path(project_root).resolve() if project_root else _find_project_root() or Path(".")
    claude_dir = root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    md_path = claude_dir / "CLAUDE.md"
    existing = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    if rule.strip() in existing:
        return f"[CLAUDE.md] Rule already present: {rule[:60]}"
    with md_path.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"): f.write("\n")
        f.write(f"\n- {rule.strip()}\n")
    _CLAUDE_MD_CACHE.pop(str(md_path), None)
    return f"[CLAUDE.md] Rule saved: {rule[:80]}"

# ── 2. MODULAR SKILLS ────────────────────────────────────────────────────────
def _skills_dir(project_root: Optional[str] = None) -> Optional[Path]:
    # Check global ~/.claude/skills first (Claude Code default location)
    global_dir = Path.home() / ".claude" / "skills"
    if global_dir.is_dir(): return global_dir
    # Then project-local .claude/skills
    root = Path(project_root).resolve() if project_root else _find_project_root()
    if root is None: return None
    d = root / ".claude" / "skills"
    return d if d.is_dir() else None

def list_skills(project_root: Optional[str] = None) -> list:
    d = _skills_dir(project_root)
    if not d: return []
    # flat: skills/django.md  OR  nested: skills/graphify/SKILL.md
    names = [p.stem for p in sorted(d.glob("*.md"))]
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and any(sub.glob("*.md")):
            names.append(sub.name)
    return names

@functools.lru_cache(maxsize=64)
def _load_skill_cached(skill_path: str) -> str:
    try: return Path(skill_path).read_text(encoding="utf-8")
    except Exception as e: return f"[Skill load error: {e}]"

def load_skill(name: str, project_root: Optional[str] = None) -> str:
    d = _skills_dir(project_root)
    if d is None: return ""
    # flat file
    path = d / f"{name}.md"
    if path.exists(): return _load_skill_cached(str(path))
    # subdir/SKILL.md
    for sub in d.iterdir():
        if sub.is_dir() and sub.name.lower() == name.lower():
            for md in sub.glob("*.md"):
                return _load_skill_cached(str(md))
    # case-insensitive flat fallback
    for p in d.glob("*.md"):
        if p.stem.lower() == name.lower(): return _load_skill_cached(str(p))
    return ""

def inject_skill(system: str, skill_name: str, project_root: Optional[str] = None) -> str:
    content = load_skill(skill_name, project_root)
    if not content: return system
    return system + f"\n\n<skill name=\"{skill_name}\">\n{content.strip()}\n</skill>"

def detect_requested_skills(msg: str, project_root: Optional[str] = None) -> list:
    available = list_skills(project_root)
    if not available: return []
    found = []
    for name in re.findall(r'(?:use\s+(?:the\s+)?|/skill\s+|@skill\s+)([\w_-]+)', msg, re.IGNORECASE):
        if name in available and name not in found: found.append(name)
    for name in available:
        if re.search(rf'\b{re.escape(name)}\b', msg, re.IGNORECASE) and name not in found:
            found.append(name)
    return found

def create_skill(name: str, content: str, project_root: Optional[str] = None) -> str:
    root = Path(project_root).resolve() if project_root else _find_project_root() or Path(".")
    skills_dir = root / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / f"{name}.md").write_text(content, encoding="utf-8")
    _load_skill_cached.cache_clear()
    return f"[Skill] created '{name}'"

# ── 3. CODEBASE BROWSER ──────────────────────────────────────────────────────
_AST_GREP_BIN = None

def _find_ast_grep() -> Optional[str]:
    global _AST_GREP_BIN
    if _AST_GREP_BIN is not None: return _AST_GREP_BIN or None
    for candidate in ("ast-grep", "sg"):
        try:
            r = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0: _AST_GREP_BIN = candidate; return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired): pass
    _AST_GREP_BIN = ""; return None

def _ast_grep_search(pattern: str, root: str, lang: str = "python") -> list:
    bin_ = _find_ast_grep()
    if not bin_: return []
    try:
        import json as _j
        result = subprocess.run([bin_, "run", "--pattern", pattern, "--lang", lang, "--json", root],
                                capture_output=True, text=True, timeout=15)
        hits = _j.loads(result.stdout) if result.stdout.strip() else []
        return [{"file": h.get("file",""), "line": h.get("range",{}).get("start",{}).get("line",0),
                 "match": h.get("text","")[:200]} for h in hits]
    except Exception as e: print(f"[ast-grep] {e}"); return []

def _python_ast_find(pattern: str, root: str, max_files: int = 40) -> list:
    pat = re.compile(pattern, re.IGNORECASE)
    results = []
    root_path = Path(root)
    for py_file in sorted(root_path.rglob("*.py"))[:max_files]:
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except Exception: continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if pat.search(node.name):
                    kind = "class" if isinstance(node, ast.ClassDef) else "def"
                    results.append({"file": str(py_file.relative_to(root_path)),
                                    "line": node.lineno, "kind": kind,
                                    "match": f"{kind} {node.name}"})
            elif isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if name and pat.search(name):
                    results.append({"file": str(py_file.relative_to(root_path)),
                                    "line": node.lineno, "kind": "call", "match": f"{name}(...)"})
    return results[:30]

def _regex_grep(pattern: str, root: str, max_results: int = 20) -> list:
    try: pat = re.compile(pattern, re.IGNORECASE)
    except re.error as e: return [{"file":"","line":0,"kind":"error","match":f"Bad regex: {e}"}]
    results = []
    for fpath in sorted(Path(root).rglob("*.py"))[:60]:
        try:
            for lineno, line in enumerate(fpath.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if pat.search(line):
                    results.append({"file": str(fpath.relative_to(root)), "line": lineno,
                                    "kind": "grep", "match": line.strip()[:200]})
                    if len(results) >= max_results: return results
        except Exception: continue
    return results

def browse_codebase(query: str, root: Optional[str] = None, mode: str = "auto") -> str:
    import os
    # Resolve WSL symlinks to actual Windows path where .py files live
    cwd = os.getcwd()
    real = os.path.realpath(cwd)
    root = root or real
    print(f"[CodeBrowser] searching '{query}' in {root}")
    if mode in ("ast-grep", "auto") and _find_ast_grep():
        hits = _ast_grep_search(query, root)
        if hits:
            return "[ast-grep results for '{}']\n".format(query) + "\n".join(
                f"{h['file']}:{h['line']}  {h['match']}" for h in hits)
    if mode in ("ast", "auto"):
        hits = _python_ast_find(query, root)
        if hits:
            return "[AST results for '{}']\n".format(query) + "\n".join(
                f"{h['file']}:{h['line']} [{h['kind']}]  {h['match']}" for h in hits)
    hits = _regex_grep(query, root)
    if hits:
        return "[Grep results for '{}']\n".format(query) + "\n".join(
            f"{h['file']}:{h['line']}  {h['match']}" for h in hits)
    return f"[CodeBrowser] No results for '{query}'"

def map_file_relationships(entry_file: str, root: Optional[str] = None, depth: int = 2) -> str:
    root_path = Path(root).resolve() if root else (_find_project_root() or Path("."))
    seen, lines = set(), []
    def _walk(fpath, indent):
        try: rel = str(fpath.relative_to(root_path))
        except ValueError: rel = str(fpath)
        if rel in seen or indent > depth: return
        seen.add(rel); lines.append("  " * indent + rel)
        try:
            src = fpath.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception: return
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if node.module else [])
                for mod in mods:
                    mp = root_path / Path(*mod.split("."))
                    for suffix in (".py", os.sep + "__init__.py"):
                        c = Path(str(mp) + suffix)
                        if c.exists(): _walk(c, indent + 1); break
    entry = Path(entry_file)
    if not entry.is_absolute(): entry = root_path / entry
    _walk(entry, 0)
    return "[File map]\n" + "\n".join(lines) if lines else f"[FileMap] Could not map {entry_file}"

# ── 4. AGENTIC SELF-CORRECTION LOOP ─────────────────────────────────────────
def _extract_exception(exec_output: str) -> Optional[str]:
    for line in exec_output.splitlines():
        if re.search(r'(Error|Exception|Traceback|FAILED|assert)', line, re.IGNORECASE):
            return line.strip()
    return None

def _run_project_tests(root: str, timeout: int = 30) -> str:
    root_path = Path(root)
    for runner in (["python", "-m", "pytest", "--tb=short", "-q"],
                   ["python", "-m", "unittest", "discover", "-q"]):
        if (root_path / "tests").exists() or list(root_path.glob("test_*.py")):
            try:
                r = subprocess.run(runner, cwd=root, capture_output=True, text=True, timeout=timeout)
                return (r.stdout + r.stderr).strip()[:1200]
            except Exception as e: return f"[Tests] {e}"
    return ""

def agentic_self_correct(generate_fn, build_prompt_fn, system, history, msg,
                          initial_response="", project_root=None, run_tests=False):
    try:
        from modules.services.tools import tool_lint, tool_exec
    except ImportError:
        def tool_lint(c): return "OK"
        def tool_exec(c, timeout=8): return "(no output)"

    root = project_root or str(_find_project_root() or Path("."))
    response = initial_response
    iters_used = 0
    for iteration in range(4):
        iters_used = iteration + 1
        if not response:
            try: response = generate_fn(build_prompt_fn(system, history, msg))
            except Exception as e: response = f"[Gen error: {e}]"; break
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        if not code_blocks: break
        code = code_blocks[0].strip()
        lint = tool_lint(code)
        if lint != "OK":
            print(f"[SelfCorrect iter {iteration+1}] lint: {lint}")
            fb = f"Your code had a lint error:\n{lint}\n\nFix it and output the complete corrected code in a ```python block."
            response = ""
            try: response = generate_fn(build_prompt_fn(system + "\n\n[SELF-CORRECTION MODE] Fix lint error. Output ONLY corrected code.", history, fb))
            except Exception as e: response = f"[Gen error: {e}]"; break
            continue
        exec_out = tool_exec(code)
        exc = _extract_exception(exec_out)
        if exc:
            print(f"[SelfCorrect iter {iteration+1}] exec error: {exc}")
            fb = f"Your code raised an error:\n```\n{exec_out[:600]}\n```\n\nFix the root cause and output the complete corrected code in a ```python block."
            response = ""
            try: response = generate_fn(build_prompt_fn(system + "\n\n[SELF-CORRECTION MODE] Fix the runtime error.", history, fb))
            except Exception as e: response = f"[Gen error: {e}]"; break
            continue
        if run_tests:
            test_out = _run_project_tests(root)
            if test_out and re.search(r'(FAILED|ERROR)', test_out, re.IGNORECASE):
                fb = f"Tests failed:\n```\n{test_out[:800]}\n```\n\nFix the code so all tests pass."
                response = ""
                try: response = generate_fn(build_prompt_fn(system + "\n\n[SELF-CORRECTION MODE] Fix failing tests.", history, fb))
                except Exception as e: response = f"[Gen error: {e}]"; break
                continue
        suffix = f"\n\n✅ **Verified** after {iters_used} iteration(s) · Lint: OK"
        if exec_out and exec_out not in ("(no output)", ""):
            suffix += f"\n```\n{exec_out[:400]}\n```"
        response = response.rstrip() + suffix
        break
    return response, iters_used

# ── Integration helpers ───────────────────────────────────────────────────────
def _detect_code_entity(msg: str) -> Optional[str]:
    m = re.search(r'\b(?:find|search|grep|where\s+is|definition\s+of|locate)\s+[`"\']?([\w_]+)[`"\']?', msg, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'\b([\w_/]+\.py)\b', msg)
    if m: return m.group(1)
    m = re.search(r'`([A-Z][\w]+|[a-z_]+_[a-z_]+)`', msg)
    if m: return m.group(1)
    return None

def enrich_system_prompt(system: str, msg: str, project_root: Optional[str] = None) -> str:
    system = inject_claude_md(system, project_root)
    for skill_name in detect_requested_skills(msg, project_root):
        system = inject_skill(system, skill_name, project_root)
    entity = _detect_code_entity(msg)
    if entity:
        ctx = browse_codebase(entity, root=project_root)
        if ctx and "No results" not in ctx:
            system += f"\n\n<codebase_context>\n{ctx[:1200]}\n</codebase_context>"
    return system

def detect_style_rule(user_msg: str, assistant_response: str) -> Optional[str]:
    for pat in [r'\b(?:always|never|from\s+now\s+on|in\s+this\s+project)\b.{5,80}',
                r'\b(?:use|prefer|avoid|don\'t\s+use|stop\s+using)\s+\w.{5,60}']:
        m = re.search(pat, user_msg, re.IGNORECASE)
        if m:
            rule = m.group(0).strip()
            if len(rule) > 10: return rule
    return None
