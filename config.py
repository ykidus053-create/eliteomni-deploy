import os, logging
log = logging.getLogger(__name__)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
SEARXNG_URL     = os.getenv("SEARXNG_URL", "http://localhost:8081")
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH", "")
N_THREADS       = int(os.getenv("N_THREADS", "4"))
N_GPU_LAYERS    = int(os.getenv("N_GPU_LAYERS", "0"))
RATE_LIMIT      = int(os.getenv("RATE_LIMIT", "60"))
if not MISTRAL_API_KEY:
    log.warning("MISTRAL_API_KEY is not set — API calls will fail")
_faiss_ok = False
try:
    import faiss; _faiss_ok = True
except ImportError:
    pass
def _tool_exec(tool_name: str, args: dict):
    return {"error": f"_tool_exec not available for {tool_name}"}
