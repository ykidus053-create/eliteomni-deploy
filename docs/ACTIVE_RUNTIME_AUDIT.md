# EliteOmni Active Runtime Audit

Generated: 2026-07-11T15:44:50.176207+00:00

## Serving path

- `app.py` is the live FastAPI composition root.
- The active request path uses `pipeline_sync`, `build_system_prompt`, `build_chatml`, `modules.core.http_client`, and `modules.reliability`.
- Quality V18 is installed at the final response boundary so later legacy enhancement passes cannot bypass it.

## Direct imports from app.py

- `agi_emulation_layer`
- `apo_engine`
- `ast`
- `asyncio`
- `base64`
- `code_rag`
- `collections`
- `concurrent.futures`
- `constitutional_rlaif`
- `context_compressor`
- `datetime`
- `debug_patch`
- `docx`
- `error_learner`
- `fastapi`
- `fastapi.middleware.cors`
- `fastapi.responses`
- `functools`
- `goal_engine`
- `god_prompt`
- `groq_client`
- `io`
- `json`
- `knowledge_graph`
- `knowledge_rag`
- `math`
- `memory`
- `model_router`
- `modules.adaptive_memory`
- `modules.claude_code`
- `modules.code_executor`
- `modules.core.constants`
- `modules.core.http_client`
- `modules.groq_client`
- `modules.knowledge_graph`
- `modules.langchain_tracing`
- `modules.loop_engine`
- `modules.meta_cognition`
- `modules.pipeline`
- `modules.quality_kernel`
- `modules.reliability`
- `modules.safety_enterprise`
- `modules.self_improvement`
- `modules.services.agents`
- `modules.services.code_enforcer`
- `modules.services.finetune`
- `modules.services.mcp`
- `modules.services.memory`
- `modules.services.pipeline`
- `modules.services.prompts`
- `modules.services.rlaif`
- `modules.services.search`
- `modules.services.semantic_mem`
- `modules.services.tool_schemas`
- `modules.services.tools`
- `modules.tool_orchestrator`
- `modules.tools`
- `modules.ttft`
- `modules.video_editor`
- `os`
- `playwright.sync_api`
- `proactive_daemon`
- `pypdf`
- `queue`
- `random`
- `re`
- `reasoning_engine`
- `refactor_daemon`
- `reflexion_loop`
- `rlef_engine`
- `secrets`
- `self_healing`
- `self_verify`
- `skill_router`
- `sqlite3`
- `structured_output`
- `subprocess`
- `swarm_orchestrator`
- `sys`
- `system_perception`
- `task_queue`
- `tempfile`
- `threading`
- `time`
- `traceback`
- `urllib.parse`
- `urllib.request`
- `uuid`
- `uvicorn`
- `uvloop`
- `working_memory`
- `world_model`

## Duplicate Python basenames

- `__init__.py`: `modules/__init__.py`, `tests/__init__.py`, `eliteomni_app/modules/__init__.py`, `eliteomni_app/human-eval/human_eval/__init__.py`, `eliteomni_app/human-eval/human-eval/human_eval/__init__.py`, `modules/intelligence/__init__.py`, `modules/api/__init__.py`, `modules/core/__init__.py`, `modules/services/__init__.py`, `modules/api/routers/__init__.py`
- `agents.py`: `eliteomni_app/modules/agents.py`, `modules/services/agents.py`
- `app.py`: `app.py`, `eliteomni_app/app.py`
- `apply_fixes.py`: `apply_fixes.py`, `eliteomni_app/apply_fixes.py`
- `budget_counter.py`: `modules/core/budget_counter.py`, `modules/services/budget_counter.py`
- `circuit_breaker.py`: `modules/circuit_breaker.py`, `modules/core/circuit_breaker.py`
- `code_rag.py`: `code_rag.py`, `modules/code_rag.py`
- `config.py`: `config.py`, `eliteomni_app/config.py`, `modules/config.py`, `eliteomni_app/modules/config.py`
- `data.py`: `eliteomni_app/human-eval/human_eval/data.py`, `eliteomni_app/human-eval/human-eval/human_eval/data.py`
- `debug_patch.py`: `debug_patch.py`, `eliteomni_app/debug_patch.py`
- `deep_think_math.py`: `modules/deep_think_math.py`, `eliteomni_app/modules/deep_think_math.py`
- `evaluate_functional_correctness.py`: `eliteomni_app/human-eval/human_eval/evaluate_functional_correctness.py`, `eliteomni_app/human-eval/human-eval/human_eval/evaluate_functional_correctness.py`
- `evaluation.py`: `eliteomni_app/human-eval/human_eval/evaluation.py`, `eliteomni_app/human-eval/human-eval/human_eval/evaluation.py`
- `execution.py`: `eliteomni_app/human-eval/human_eval/execution.py`, `eliteomni_app/human-eval/human-eval/human_eval/execution.py`
- `finetune.py`: `finetune.py`, `eliteomni_app/finetune.py`, `modules/finetune.py`, `eliteomni_app/modules/finetune.py`, `modules/services/finetune.py`
- `fix_accuracy.py`: `fix_accuracy.py`, `eliteomni_app/fix_accuracy.py`
- `fix_all.py`: `fix_all.py`, `eliteomni_app/fix_all.py`
- `gpt55_style.py`: `modules/gpt55_style.py`, `eliteomni_app/modules/gpt55_style.py`
- `groq_client.py`: `groq_client.py`, `eliteomni_app/groq_client.py`, `modules/groq_client.py`
- `groq_client_patch.py`: `groq_client_patch.py`, `eliteomni_app/groq_client_patch.py`
- `health.py`: `health.py`, `modules/health.py`, `modules/core/health.py`
- `knowledge_graph.py`: `knowledge_graph.py`, `modules/knowledge_graph.py`
- `knowledge_rag.py`: `knowledge_rag.py`, `modules/knowledge_rag.py`
- `mcp.py`: `mcp.py`, `eliteomni_app/mcp.py`, `modules/mcp.py`, `eliteomni_app/modules/mcp.py`, `modules/services/mcp.py`
- `memory.py`: `memory.py`, `eliteomni_app/memory.py`, `modules/memory.py`, `eliteomni_app/modules/memory.py`, `modules/services/memory.py`
- `model_router.py`: `model_router.py`, `modules/model_router.py`
- `opus_engine.py`: `opus_engine.py`, `modules/opus_engine.py`
- `pipeline.py`: `modules/pipeline.py`, `modules/services/pipeline.py`
- `planner.py`: `planner.py`, `modules/intelligence/planner.py`
- `prompts.py`: `eliteomni_app/prompts.py`, `modules/prompts.py`, `eliteomni_app/modules/prompts.py`, `modules/services/prompts.py`
- `reasoning_engine.py`: `reasoning_engine.py`, `modules/services/reasoning_engine.py`
- `reflection_engine.py`: `reflection_engine.py`, `modules/intelligence/reflection_engine.py`
- `reliability.py`: `reliability.py`, `modules/reliability.py`, `eliteomni_app/modules/reliability.py`
- `rlaif.py`: `rlaif.py`, `eliteomni_app/rlaif.py`, `modules/rlaif.py`, `eliteomni_app/modules/rlaif.py`, `modules/services/rlaif.py`
- `search.py`: `search.py`, `eliteomni_app/search.py`, `modules/search.py`, `eliteomni_app/modules/search.py`, `modules/services/search.py`
- `self_verify.py`: `self_verify.py`, `modules/self_verify.py`
- `semantic_mem.py`: `semantic_mem.py`, `eliteomni_app/semantic_mem.py`, `modules/semantic_mem.py`, `eliteomni_app/modules/semantic_mem.py`, `modules/services/semantic_mem.py`
- `setup.py`: `eliteomni_app/human-eval/setup.py`, `eliteomni_app/human-eval/human-eval/setup.py`
- `split_modules.py`: `split_modules.py`, `eliteomni_app/split_modules.py`
- `tools.py`: `eliteomni_app/tools.py`, `modules/tools.py`, `eliteomni_app/modules/tools.py`, `modules/services/tools.py`
- `ttft.py`: `modules/ttft.py`, `eliteomni_app/modules/ttft.py`
- `uncertainty_engine.py`: `uncertainty_engine.py`, `modules/intelligence/uncertainty_engine.py`
- `validation.py`: `eliteomni_app/validation.py`, `eliteomni_app/modules/validation.py`
- `world_model.py`: `world_model.py`, `modules/intelligence/world_model.py`

## Dangerous/static patterns

| File | Pattern | Count |
|---|---:|---:|
| `agent_mesh.py` | bare except | 1 |
| `app.py` | bare except | 25 |
| `app.py` | star import | 11 |
| `apply_fixes.py` | bare except | 1 |
| `apply_fixes.py` | shell=True | 2 |
| `autonomous_agent.py` | shell=True | 1 |
| `book8_gaps.py` | bare except | 1 |
| `book_gaps_impl.py` | bare except | 1 |
| `constitutional_rlaif.py` | bare except | 1 |
| `context_compressor.py` | bare except | 3 |
| `debug_patch.py` | bare except | 3 |
| `eliteomni_app/app.py` | bare except | 6 |
| `eliteomni_app/app.py` | star import | 11 |
| `eliteomni_app/app_patched.py` | bare except | 10 |
| `eliteomni_app/app_patched.py` | shell=True | 1 |
| `eliteomni_app/apply_fixes.py` | bare except | 1 |
| `eliteomni_app/apply_fixes.py` | shell=True | 2 |
| `eliteomni_app/debug_patch.py` | bare except | 4 |
| `eliteomni_app/fix_all.py` | bare except | 1 |
| `eliteomni_app/groq_client.py` | bare except | 4 |
| `eliteomni_app/modules/agents.py` | shell=True | 1 |
| `eliteomni_app/modules/rlaif.py` | bare except | 1 |
| `eliteomni_app/modules/search.py` | bare except | 1 |
| `eliteomni_app/rlaif.py` | bare except | 1 |
| `eliteomni_app/search.py` | bare except | 1 |
| `eliteomni_app/validation.py` | bare except | 4 |
| `fix_all.py` | bare except | 1 |
| `goal_engine.py` | bare except | 1 |
| `god_prompt.py` | bare except | 2 |
| `groq_client.py` | bare except | 4 |
| `hot_reload.py` | bare except | 1 |
| `modules/capability_guardian.py` | bare except | 1 |
| `modules/capability_guardian.py` | runtime pip install | 1 |
| `modules/core/unified_memory.py` | bare except | 2 |
| `modules/intelligence/hypothesis_engine.py` | bare except | 1 |
| `modules/intelligence/reflection_engine.py` | bare except | 1 |
| `modules/intelligence/self_model.py` | bare except | 4 |
| `modules/intelligence/tool_policy.py` | bare except | 1 |
| `modules/project_context.py` | shell=True | 1 |
| `modules/services/code_sandbox.py` | bare except | 10 |
| `modules/services/code_sandbox.py` | runtime pip install | 1 |
| `modules/services/code_sandbox.py` | shell=True | 1 |
| `modules/services/code_validator.py` | runtime pip install | 1 |
| `modules/services/file_reader.py` | runtime pip install | 1 |
| `modules/services/mcp.py` | bare except | 1 |
| `modules/services/rlaif.py` | bare except | 1 |
| `modules/services/search.py` | bare except | 1 |
| `planner.py` | bare except | 1 |
| `pregen_reasoning.py` | bare except | 1 |
| `refactor_daemon.py` | bare except | 1 |
| `reflexion_loop.py` | bare except | 4 |
| `rlef_engine.py` | bare except | 3 |
| `scripts/audit_active_runtime.py` | shell=True | 1 |
| `self_wire.py` | bare except | 2 |
| `skill_library.py` | bare except | 1 |
| `swarm_orchestrator.py` | bare except | 2 |
| `synthetic_trainer.py` | bare except | 1 |
| `system_perception.py` | bare except | 1 |
| `tests/test_active_runtime_wiring_v18.py` | shell=True | 1 |
| `tests/test_agent_core.py` | shell=True | 1 |
| `tests/test_production_guard.py` | bare except | 1 |
| `voting_engine.py` | bare except | 1 |

## Largest physical line lengths

| File | Longest line |
|---|---:|
| `modules/groq_client.py` | 2357 |
| `eliteomni_app/app_patched.py` | 955 |
| `eliteomni_app/app.py` | 900 |
| `app.py` | 900 |
| `modules/services/pipeline.py` | 479 |
| `eliteomni_app/validation.py` | 436 |
| `reasoning_engine.py` | 358 |
| `reflexion_loop.py` | 345 |
| `modules/services/memory.py` | 342 |
| `modules/services/search.py` | 340 |
| `eliteomni_app/modules/memory.py` | 337 |
| `eliteomni_app/memory.py` | 295 |
| `opus_engine.py` | 290 |
| `agentic_loop.py` | 284 |
| `eliteomni_app/modules/prompts.py` | 282 |
| `swarm_orchestrator.py` | 276 |
| `modules/thinking.py` | 274 |
| `cot_engine.py` | 268 |
| `eliteomni_app/debug_patch.py` | 253 |
| `debug_patch.py` | 253 |

## Priority recommendations

1. Gradually split `app.py` into route, orchestration, research, and verification services while keeping Quality V18 as the invariant edge.
2. Remove duplicate root/modules implementations after import-graph tests prove which copies are inactive.
3. Eliminate runtime package installation and shell execution from the sandbox.
4. Configure distinct models through `ELITE_MODEL_*` variables only after confirming provider availability.
5. Add golden end-to-end evaluations for coding, research, reasoning, and tool-grounding before accepting future self-modifying prompts.
