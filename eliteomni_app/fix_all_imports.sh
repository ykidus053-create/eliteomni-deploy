#!/bin/bash
M=/home/kidus/eliteomni_app/modules

echo "=== Fixing all module imports ==="

# search.py
grep -q "from modules.config import.*_ensure_searxng" $M/search.py || \
  sed -i '1s/^/from modules.config import CTX_WINDOW, MAX_MEM, SEARXNG_URL, _background_searxng_watchdog, _ensure_searxng, _probe_searxng, episodic_store, mem_store\n/' $M/search.py

grep -q "from modules.memory import.*_load_rag_from_db" $M/search.py || \
  sed -i '1s/^/from modules.memory import CTX_TOKEN_BUDGET, _DB_PATH, _feedback, _load_rag_from_db, _rag_index, _rag_store, _rlaif_log, _save_feedback, _sft_store\n/' $M/search.py

grep -q "from modules.finetune import.*FINETUNE_DB" $M/search.py || \
  sed -i '1s/^/from modules.finetune import FINETUNE_DB\n/' $M/search.py

# memory.py
grep -q "from modules.config import.*_faiss_ok" $M/memory.py || \
  sed -i '1s/^/from modules.config import _faiss_ok\n/' $M/memory.py

grep -q "from modules.groq_client import.*FEEDBACK_FILE" $M/memory.py || \
  sed -i '1s/^/from modules.groq_client import FEEDBACK_FILE\n/' $M/memory.py

# validation.py
grep -q "from modules.config import.*_probe_searxng" $M/validation.py || \
  sed -i '1s/^/from modules.config import CTX_WINDOW, GGUF_MODEL_PATH, MAX_MEM, N_BATCH, N_GPU_LAYERS, N_THREADS, RATE_LIMIT, SEARXNG_URL, _background_searxng_watchdog, _ensure_searxng, _gen_lock, _probe_searxng\n/' $M/validation.py

grep -q "from modules.groq_client import.*GROQ_API_KEY" $M/validation.py || \
  sed -i '1s/^/from modules.groq_client import GROQ_API_KEY, groq_generate, groq_stream\n/' $M/validation.py

grep -q "from modules.mcp import.*_MCP_SERVERS" $M/validation.py || \
  sed -i '1s/^/from modules.mcp import _MCP_SERVERS, mcp_discover_all\n/' $M/validation.py

grep -q "from modules.memory import.*CONSTITUTION_WEIGHTED" $M/validation.py || \
  sed -i '1s/^/from modules.memory import CONSTITUTION, CONSTITUTION_FLAT, CONSTITUTION_WEIGHTED, EFFORT_LEVEL, HIERARCHY, SKILLS, _DB_PATH, _rlaif_log, _rlaif_wins, tool_calc\n/' $M/validation.py

grep -q "from modules.prompts import.*AGENTIC_EXEMPLARS" $M/validation.py || \
  sed -i '1s/^/from modules.prompts import AGENTIC_EXEMPLARS, ANTI_HALLUCINATION_PROMPT, APPROVER_PROMPT, COMPUTER_USE_PROMPT, DELIBERATION_PROMPT, EXECUTION_SIMULATOR_PROMPT, LONG_SESSION_PROMPT, PARALLEL_CALC_PROMPT, PEVI_LOOP_PROMPT, PROCESS_SUPERVISION_PROMPT, SCIENTIFIC_COMPUTING_PROMPT, SELF_CORRECT_DEBUG_PROMPT, STATE_TRACKING_PROMPT, UNCERTAINTY_PROMPT, _scratchpad, get_effort_prompts\n/' $M/validation.py

grep -q "from modules.tools import.*_extract_code_blocks" $M/validation.py || \
  sed -i '1s/^/from modules.tools import _extract_code_blocks, tool_lint\n/' $M/validation.py

# rlaif.py
grep -q "from modules.groq_client import.*GROQ_API_KEY" $M/rlaif.py || \
  sed -i '1s/^/from modules.groq_client import GROQ_API_KEY, groq_generate\n/' $M/rlaif.py

grep -q "from modules.memory import.*CONSTITUTION_WEIGHTED" $M/rlaif.py || \
  sed -i '1s/^/from modules.memory import CONSTITUTION_WEIGHTED, _rlaif_log, _rlaif_wins\n/' $M/rlaif.py

grep -q "from modules.validation import.*RLAIF_TMPL" $M/rlaif.py || \
  sed -i '1s/^/from modules.validation import CAI_CRITIQUE_TMPL, CAI_REVISE_TMPL, HHH_SCORE_TMPL, RED_TEAM_TMPL, REVISION_TMPL, RLAIF_TMPL, _budget, build_chatml, generate_sync\n/' $M/rlaif.py

# agents.py
grep -q "from modules.groq_client import.*GROQ_API_KEY" $M/agents.py || \
  sed -i '1s/^/from modules.groq_client import GROQ_API_KEY, GROQ_URL, _audit, _get_next_key, _truncate_msgs, groq_generate\n/' $M/agents.py

grep -q "from modules.mcp import.*run_mcp_tools" $M/agents.py || \
  sed -i '1s/^/from modules.mcp import run_mcp_tools\n/' $M/agents.py

grep -q "from modules.memory import.*db_mem_save" $M/agents.py || \
  sed -i '1s/^/from modules.memory import db_mem_save\n/' $M/agents.py

grep -q "from modules.tools import.*_grep_codebase" $M/agents.py || \
  sed -i '1s/^/from modules.tools import _grep_codebase, tool_exec, tool_lint\n/' $M/agents.py

grep -q "from modules.search import.*tool_search" $M/agents.py || \
  sed -i '1s/^/from modules.search import tool_search, tool_web_fetch\n/' $M/agents.py

grep -q "from modules.validation import.*build_chatml" $M/agents.py || \
  sed -i '1s/^/from modules.validation import build_chatml, build_system_prompt, generate_sync\n/' $M/agents.py

grep -q "from modules.config import.*_tool_exec" $M/agents.py || \
  sed -i '1s/^/from modules.config import _tool_exec\n/' $M/agents.py

# finetune.py
grep -q "from modules.config import.*RATE_LIMIT" $M/finetune.py || \
  sed -i '1s/^/from modules.config import RATE_LIMIT\n/' $M/finetune.py

grep -q "from modules.memory import.*_rate_lim" $M/finetune.py || \
  sed -i '1s/^/from modules.memory import _rate_lim\n/' $M/finetune.py

# mcp.py
grep -q "from modules.config import.*_tool_exec" $M/mcp.py || \
  sed -i '1s/^/from modules.config import _tool_exec\n/' $M/mcp.py

# prompts.py
grep -q "from modules.config import.*_gen_lock" $M/prompts.py || \
  sed -i '1s/^/from modules.config import _gen_lock\n/' $M/prompts.py

grep -q "from modules.groq_client import.*GROQ_API_KEY" $M/prompts.py || \
  sed -i '1s/^/from modules.groq_client import GROQ_API_KEY, groq_generate, groq_stream\n/' $M/prompts.py

grep -q "from modules.validation import.*build_chatml" $M/prompts.py || \
  sed -i '1s/^/from modules.validation import _budget, _clean, _lc_kw, build_chatml, build_system_prompt, generate_sync, msg_len\n/' $M/prompts.py

grep -q "from modules.search import.*STATE_TRACKING_PROMPT" $M/prompts.py || \
  sed -i '1s/^/from modules.search import DELIBERATION_PROMPT, STATE_TRACKING_PROMPT\n/' $M/prompts.py

echo ""
echo "=== All imports patched! Testing imports... ==="
cd /home/kidus/eliteomni_app
for f in modules/*.py; do
  python3 -c "import sys; sys.path.insert(0,'.'); mod='${f%.py}'; mod=mod.replace('/','.');import importlib; importlib.import_module(mod)" 2>&1 | grep -v "^$" | grep -v "FutureWarning\|warnings.warn\|HF_HUB" && echo "FAIL $f" || echo "OK   $f"
done

echo ""
echo "=== Restarting app ==="
pkill -9 -f "python3.*app.py"; sleep 2
cd /home/kidus/eliteomni_app && python3 app.py &
sleep 5
echo ""
echo "=== Testing /stream ==="
curl -s -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "session_id": "test"}' && echo
curl -s -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "calculate 10 * 5", "session_id": "test"}' && echo
