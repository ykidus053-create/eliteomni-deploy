"""
Mistral Fine-tuning Pipeline
Uploads preference data and triggers DPO fine-tune job automatically.
"""
import os, json, time, requests

API_KEY = os.environ.get("MISTRAL_API_KEY", "")
BASE = "https://api.mistral.ai/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def upload_file(jsonl_path: str) -> str:
    """Upload JSONL file to Mistral and return file_id."""
    with open(jsonl_path, "rb") as f:
        r = requests.post(
            f"{BASE}/files",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": (os.path.basename(jsonl_path), f, "text/plain")},
            data={"purpose": "fine-tune"}
        )
    r.raise_for_status()
    file_id = r.json()["id"]
    print(f"[finetune] uploaded file_id={file_id}")
    return file_id

def create_finetune_job(file_id: str, model: str = "magistral-medium-latest", suffix: str = "eliteomni") -> str:
    """Create a DPO fine-tune job and return job_id."""
    payload = {
        "model": model,
        "training_files": [{"file_id": file_id, "weight": 1}],
        "hyperparameters": {
            "training_steps": 100,
            "learning_rate": 1e-4
        },
        "suffix": suffix
    }
    r = requests.post(f"{BASE}/fine_tuning/jobs", headers=HEADERS, json=payload)
    r.raise_for_status()
    job_id = r.json()["id"]
    print(f"[finetune] job created job_id={job_id}")
    return job_id

def check_job_status(job_id: str) -> dict:
    r = requests.get(f"{BASE}/fine_tuning/jobs/{job_id}", headers=HEADERS)
    r.raise_for_status()
    return r.json()

def wait_for_job(job_id: str, poll_interval: int = 60) -> dict:
    """Poll until job completes. Returns final job dict."""
    while True:
        status = check_job_status(job_id)
        state = status.get("status", "unknown")
        print(f"[finetune] job={job_id} status={state}")
        if state in ("SUCCESS", "FAILED", "CANCELLED"):
            return status
        time.sleep(poll_interval)

def run_finetune_pipeline(jsonl_path: str) -> dict:
    """Full pipeline: upload → create job → wait → return result."""
    print(f"[finetune] starting pipeline with {jsonl_path}")
    file_id = upload_file(jsonl_path)
    job_id = create_finetune_job(file_id)
    print(f"[finetune] job submitted — polling every 60s")
    result = wait_for_job(job_id)
    if result.get("status") == "SUCCESS":
        model_id = result.get("fine_tuned_model")
        print(f"[finetune] SUCCESS — new model: {model_id}")
        # Auto-update config to use new model
        _update_model_config(model_id)
    else:
        print(f"[finetune] FAILED: {result}")
    return result

def _update_model_config(model_id: str):
    """Swap the active model to the newly fine-tuned one."""
    env_path = os.path.expanduser("~/eliteomni_app/.env")
    try:
        with open(env_path) as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            for line in lines:
                if line.startswith("FINETUNED_MODEL="):
                    f.write(f"FINETUNED_MODEL={model_id}\n")
                else:
                    f.write(line)
            else:
                f.write(f"\nFINETUNED_MODEL={model_id}\n")
        print(f"[finetune] updated .env FINETUNED_MODEL={model_id}")
    except Exception as e:
        print(f"[finetune] config update error: {e}")

if __name__ == "__main__":
    # Manual trigger
    from constitutional_rlaif import export_preference_dataset
    path = export_preference_dataset()
    result = run_finetune_pipeline(path)
    print(json.dumps(result, indent=2))

def upload_sft_and_finetune(min_rating: int = 7, model: str = "magistral-medium-latest") -> dict:
    """
    Export SFT data from finetune.db, convert to Mistral format, upload and train.
    Mistral SFT format: {"messages": [{"role": ..., "content": ...}]}
    """
    import sqlite3, json, tempfile, os
    FINETUNE_DB = os.path.expanduser("~/eliteomni_finetune.db")

    # Export from DB
    conn = sqlite3.connect(FINETUNE_DB)
    rows = conn.execute(
        "SELECT system_prompt, user_msg, assistant_response FROM samples WHERE rating>=? ORDER BY ts DESC LIMIT 5000",
        (min_rating,)
    ).fetchall()
    conn.close()

    if len(rows) < 10:
        print(f"[SFT] only {len(rows)} samples — need at least 10")
        return {"error": f"insufficient data: {len(rows)} samples"}

    # Write in Mistral completion format
    path = os.path.expanduser("~/eliteomni_sft.jsonl")
    with open(path, "w") as f:
        for system, user, assistant in rows:
            prompt = (system or "You are a helpful assistant.") + "\n\nUser: " + user + "\nAssistant:"
            record = {"prompt": prompt, "completion": " " + assistant}
            f.write(json.dumps(record) + "\n")
    print(f"[SFT] exported {len(rows)} samples to {path}")

    # Upload and fine-tune
    file_id = upload_file(path)
    payload = {
        "model": model,
        "training_files": [{"file_id": file_id, "weight": 1}],
        "hyperparameters": {"training_steps": 100, "learning_rate": 0.0001},
        "suffix": "eliteomni-sft"
    }
    r = requests.post(f"{BASE}/fine_tuning/jobs", headers=HEADERS, json=payload)
    r.raise_for_status()
    job = r.json()
    print(f"[SFT] job created job_id={job['id']}")
    return job
