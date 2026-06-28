import json, os, time, threading

SUBCONSCIOUS_LOG = "AGI_SUBCONSCIOUS.md"

def log_subconscious_action(daemon: str, action: str):
    """Upgraded: Background daemons log their actions here so the main AI can read them."""
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(SUBCONSCIOUS_LOG, "a") as f:
            f.write(f"- [{ts}] **{daemon}**: {action}\n")
    except: pass

def get_subconscious_context() -> str:
    """The main AI reads this to know what happened while it was idle."""
    try:
        if not os.path.exists(SUBCONSCIOUS_LOG): return ""
        with open(SUBCONSCIOUS_LOG, "r") as f:
            lines = f.readlines()[-10:] # Last 10 actions
        if not lines: return ""
        return "[SUBCONSCIOUS ACTIVITY LOG]\nWhat my background daemons did while I was idle:\n" + "".join(lines) + "\n[/SUBCONSCIOUS ACTIVITY LOG]\n"
    except: return ""

def compress_history(history: list, generate_fn) -> list:
    """Upgraded: Rolling State Compressor. Prevents context degradation."""
    if len(history) < 12: return history
    
    try:
        # Take the oldest 8 messages and compress them into a state object
        old_msgs = history[:-4]
        convo_str = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in old_msgs])
        
        prompt = [
            {"role": "system", "content": "Compress this conversation history into a dense JSON object containing 'project_state', 'active_goals', and 'key_decisions'. Output ONLY the JSON."},
            {"role": "user", "content": convo_str}
        ]
        raw = generate_fn(prompt, max_tokens=300, model="mistral-small-latest")
        
        # Extract JSON
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            state_json = match.group()
            # Replace old history with the compressed state + recent messages
            return [{"role": "system", "content": f"[COMPRESSED HISTORY STATE]\n{state_json}\n[/COMPRESSED HISTORY STATE]"}] + history[-4:]
    except:
        pass
        
    return history
