import os, py_compile, tempfile

BASE = "/home/kidus/eliteomni_app"
SRC  = os.path.join(BASE, "app.py")
with open(SRC) as f:
    lines = f.readlines()

HEADER = "".join(lines[:22])

sections = {
    "groq_client.py":  (23,  340),
    "config.py":       (340, 449),
    "memory.py":       (449, 903),
    "tools.py":        (903, 1124),
    "search.py":       (1124, 1707),
    "prompts.py":      (1707, 2013),
    "rlaif.py":        (2744, 2837),
    "semantic_mem.py": (2837, 2878),
    "finetune.py":     (2878, 2945),
    "agents.py":       (2945, 3379),
    "mcp.py":          (3379, 3637),
}

os.makedirs(os.path.join(BASE, "modules"), exist_ok=True)

for fname, (start, end) in sections.items():
    path = os.path.join(BASE, "modules", fname)

    # Check if existing file compiles cleanly — if so, skip it
    if os.path.exists(path):
        try:
            py_compile.compile(path, doraise=True)
            print(f"✓ modules/{fname}: {end-start} lines (kept existing clean version)")
            continue
        except Exception:
            pass  # existing file is broken, regenerate

    content = f"# AUTO-SPLIT FROM app.py lines {start}-{end}\n" + HEADER + "".join(lines[start:end])

    # Validate before writing
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(content)
            tname = tf.name
        py_compile.compile(tname, doraise=True)
        os.unlink(tname)
        with open(path, "w") as f:
            f.write(content)
        print(f"✓ modules/{fname}: {end-start} lines")
    except Exception as e:
        print(f"⚠️  modules/{fname}: skipped (syntax error in source range {start}-{end}: {e})")

print(f"\nDone. {len(sections)} modules processed in /home/kidus/eliteomni_app/modules/")
