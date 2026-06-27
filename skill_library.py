import os, re, json, logging
log = logging.getLogger(__name__)
SKILL_DIR = os.path.join(os.path.dirname(__file__), "skill_library")
os.makedirs(SKILL_DIR, exist_ok=True)

def save_skill(task_desc: str, code: str, generate_fn):
    """Upgraded: Voyager-style skill abstraction. Saves successful code as reusable functions."""
    prompt = [
        {"role": "system", "content": "You are a code abstraction engine. Given a task and its working implementation, extract the core logic into a single, clean, reusable Python function. Output ONLY the python code, no markdown."},
        {"role": "user", "content": f"Task: {task_desc}\n\nWorking Code:\n{code[:2000]}"}
    ]
    try:
        raw = generate_fn(prompt, max_tokens=1500)
        # Extract code
        match = re.search(r'```python\n(.*?)```', raw, re.DOTALL)
        skill_code = match.group(1).strip() if match else raw.strip()
        
        # Generate a safe filename
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', task_desc.replace(' ', '_').lower())[:40]
        if not safe_name: safe_name = "unnamed_skill"
        
        filepath = os.path.join(SKILL_DIR, f"{safe_name}.py")
        with open(filepath, "w") as f:
            f.write(skill_code)
            
        log.info(f"[SkillLibrary] Saved new skill: {safe_name}.py")
        return safe_name
    except Exception as e:
        log.error(f"[SkillLibrary] Error saving skill: {e}")
        return None

def get_skills_context(task_desc: str) -> str:
    """Returns a list of available skills so the AI can import them instead of coding from scratch."""
    try:
        skills = [f.replace('.py', '') for f in os.listdir(SKILL_DIR) if f.endswith('.py')]
        if not skills: return ""
        return "[AVAILABLE SKILLS]\nYou can import these existing skills instead of writing from scratch:\n- " + "\n- ".join(skills) + "\n[/AVAILABLE SKILLS]"
    except:
        return ""
