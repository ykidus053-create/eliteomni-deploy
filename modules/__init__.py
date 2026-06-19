"""EliteOmni modules — v2 architecture."""
try:
    from modules.pipeline     import classify_skill, route_complexity, run_stream, run_sync
    from modules.memory_fast  import mem_save, mem_get, episodic_save, episodic_get
    from modules.llm_client   import generate, stream
    from modules.model_router import select_model, record_outcome
except ImportError as e:
    print(f"[modules/__init__] import warning: {e}")
