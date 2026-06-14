# modules/config.py
import os
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH", "")
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))

# ── Exports required by other modules ────────────────────────────────────────
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))
_faiss_ok = False
try:
    import faiss
    _faiss_ok = True
except ImportError:
    pass

def _tool_exec(tool_name: str, args: dict):
    """Stub tool executor — real implementation in app.py."""
    return {"error": f"_tool_exec not available for {tool_name}"}
