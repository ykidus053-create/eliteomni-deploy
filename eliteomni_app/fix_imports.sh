#!/bin/bash
M="/home/kidus/eliteomni_app/modules"

# 1. Move _faiss_ok to config.py (memory.py loads before search.py, causing circular import)
grep -q "_faiss_ok" $M/config.py || cat >> $M/config.py << 'PYEOF'

# FAISS availability (shared across modules)
try:
    import faiss as _faiss
    import numpy as np
    _faiss_ok = True
except ImportError:
    _faiss = None
    np = None
    _faiss_ok = False
faiss_index = None
faiss_texts = []
PYEOF

# 2. Fix memory.py - needs _faiss_ok from config (not search, avoids circular import)
sed -i 's/from modules.search import _embed, _faiss_ok, msg//' $M/memory.py
grep -q "from modules.config import" $M/memory.py && \
  sed -i 's/from modules.config import/from modules.config import _faiss_ok, /' $M/memory.py || \
  sed -i '1s/^/from modules.config import _faiss_ok\n/' $M/memory.py

# 3. Fix search.py - add all missing real imports
sed -i 's/from modules.config import _probe_searxng, _background_searxng_watchdog, _ensure_searxng/from modules.config import _probe_searxng, _background_searxng_watchdog, _ensure_searxng, SEARXNG_URL, CTX_WINDOW, MAX_MEM/' $M/search.py
sed -i 's/from modules.memory import _rlaif_log, _rag_index, _rag_store, _load_rag_from_db/from modules.memory import _rlaif_log, _rag_index, _rag_store, _load_rag_from_db, CTX_TOKEN_BUDGET, _DB_PATH, _feedback, _save_feedback, _sft_store/' $M/search.py
grep -q "from modules.finetune import" $M/search.py || sed -i '1s/^/from modules.finetune import FINETUNE_DB\n/' $M/search.py

# 4. Fix validation.py - add missing prompts and config vars
sed -i 's/from modules.config import _probe_searxng, _background_searxng_watchdog/from modules.config import _probe_searxng, _background_searxng_watchdog, GGUF_MODEL_PATH, N_BATCH, N_GPU_LAYERS, N_THREADS, _gen_lock/' $M/validation.py
sed -i 's/from modules.memory import _MCP_SERVERS//' $M/validation.py 2>/dev/null || true

# 5. Fix rlaif.py
sed -i 's/from modules.memory import _rlaif_log/from modules.memory import _rlaif_log, _rlaif_wins, CONSTITUTION_WEIGHTED/' $M/rlaif.py

echo "Done. Restarting..."
pkill -9 -f "python3.*app.py"; sleep 1
