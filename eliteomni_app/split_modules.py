import os

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
    with open(path, "w") as f:
        f.write(f"# AUTO-SPLIT FROM app.py lines {start}-{end}\n")
        f.write(HEADER)
        f.write("".join(lines[start:end]))
    print(f"✓ modules/{fname}: {end-start} lines")

print(f"\nDone. {len(sections)} modules created in /home/kidus/eliteomni_app/modules/")
print("Next step: update app.py to import from modules/")
