import threading, time, logging
log = logging.getLogger(__name__)

def _proactive_loop():
    """Background loop that runs every 5 minutes to consolidate memories and index files."""
    time.sleep(15)  # Let app finish booting
    log.info("[ProactiveDaemon] Starting background intelligence loops...")
    
    while True:
        try:
            # 1. Idle-Time Memory Consolidation
            from modules.core.http_client import mistral_generate
            from reflection_engine import consolidate_lessons
            from memory import mem_get
            
            recent_msgs = mem_get(limit=6)
            if recent_msgs and len(recent_msgs) >= 4:
                # Reconstruct a pseudo-history for the consolidator
                history = [{"role": "user" if i%2==0 else "assistant", "content": m} for i, m in enumerate(recent_msgs)]
                consolidate_lessons(history, lambda p, m="": mistral_generate(p, max_tokens=200, model="mistral-small-latest"), "general")
        except Exception as e:
            log.debug(f"[ProactiveDaemon] Memory consolidation error: {e}")
        
        try:
            # 2. Background RAG Indexing (Hot-reloads new files automatically)
            from knowledge_rag import build_index
            build_index(force=False)  # Only updates if files changed
        except Exception as e:
            log.debug(f"[ProactiveDaemon] RAG indexing error: {e}")

        # Sleep for 5 minutes (300s)
        time.sleep(300)

def start_proactive_daemon():
    t = threading.Thread(target=_proactive_loop, daemon=True, name="proactive_daemon")
    t.start()
    print("[Startup] ✓ Proactive Daemon started (Memory Consolidator + RAG Indexer).")
