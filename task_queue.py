import threading, sqlite3, time, os, logging
from uuid import uuid4

log = logging.getLogger(__name__)
DB = os.path.expanduser("~/eliteomni_tasks.db")
_lock = threading.Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, status TEXT, prompt TEXT, logs TEXT, result TEXT, created_ts REAL
    )""")
    con.commit(); con.close()
_init()

def _update_task(task_id, status, logs="", result=""):
    with _lock:
        con = sqlite3.connect(DB)
        con.execute("UPDATE tasks SET status=?, logs=?, result=? WHERE id=?", (status, logs, result, task_id))
        con.commit(); con.close()

def submit_task(prompt: str, history: list, skill: str, complexity: str, search_ctx: str, generate_fn):
    """Upgraded: Spawns a background thread for massive tasks to prevent HTTP timeouts."""
    task_id = str(uuid4())[:8]
    
    with _lock:
        con = sqlite3.connect(DB)
        con.execute("INSERT INTO tasks (id, status, prompt, logs, result, created_ts) VALUES (?,?,?,?,?,?)",
                    (task_id, "running", prompt, "Starting...\n", "", time.time()))
        con.commit(); con.close()
        
    def _run():
        logs = f"Task started: {prompt[:100]}\n"
        try:
            # Dynamically import to avoid circular dependencies
            from agent_mesh import production_generate
            result = production_generate(prompt, skill, complexity, history, search_ctx, generate_fn)
            logs += f"Task completed successfully.\n"
            _update_task(task_id, "completed", logs, result)
        except Exception as e:
            logs += f"Task failed: {str(e)}\n"
            _update_task(task_id, "failed", logs, str(e))
            
    t = threading.Thread(target=_run, daemon=True, name=f"task_{task_id}")
    t.start()
    return task_id

def get_task_status(task_id: str) -> dict:
    with _lock:
        con = sqlite3.connect(DB)
        row = con.execute("SELECT status, prompt, logs, result FROM tasks WHERE id=?", (task_id,)).fetchone()
        con.close()
    if not row: return {"error": "Task not found"}
    return {"id": task_id, "status": row[0], "prompt": row[1], "logs": row[2], "result": row[3]}

def should_use_async_task(prompt: str, complexity: str) -> bool:
    """Routes massive requests to the background queue."""
    if complexity == "hard" and len(prompt.split()) > 20:
        return True
    if any(kw in prompt.lower() for kw in ["refactor", "entire codebase", "rewrite all", "full migration"]):
        return True
    return False
