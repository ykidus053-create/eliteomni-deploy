# EliteOmni Coding & Reasoning Core V27

V27 consolidates the active production path instead of adding another generic
agent layer.

## Repairs

- Defines the missing `_truncate_msgs`, `knowledge_boundary_check`, and
  `prefetch_plan` helpers.
- Replaces the extra LLM architect call with a deterministic repository-aware
  plan, preserving the model budget for implementation and evidence-based repair.
- Makes the editor language-aware rather than forcing every task into Python.
- Uses symbol-aware V20 repository retrieval in the coding prompt.
- Performs deterministic syntax/placeholder checks and never labels skipped
  execution as verified.
- Fixes the stale `~/eliteomni_app` project root.
- Makes parallel agent teams opt-in.
- Skips generic loop-engine rewrites for coder responses and disables the
  duplicate second loop by default.
- Keeps Frontier multi-candidate multiplication opt-in for coding.
- Preserves true token streaming by making V18 buffered verification opt-in.
- Parses Cerebras GLM reasoning from the documented streaming `reasoning` field.
- Routes coding to the supported GLM 4.7 model by default.

## Recommended Railway variables

```env
ELITE_PROJECT_ROOT=/app
ELITE_MODEL_CODER=cerebras/zai-glm-4.7
ELITE_ENABLE_AGENT_TEAM=0
ELITE_ENABLE_LOOP_ENGINE=1
ELITE_ENABLE_SECOND_LOOP_ENGINE=0
ELITE_FRONTIER_CODER=0
ELITE_BUFFERED_VERIFICATION_STREAM=0
ELITE_CODER_REPAIR_ON_EVIDENCE=1
ELITE_CODER_MAX_TOKENS=8000
```

