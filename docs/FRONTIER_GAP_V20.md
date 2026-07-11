# Frontier Gap V20

This upgrade targets measurable capability gaps instead of adding another large
system prompt.

## Runtime behavior

`modules/frontier_runtime.py` wraps the final active `pipeline_sync` function.

- Easy requests keep the existing one-pass path.
- Failed verification always triggers an independent retry.
- Hard coding, research, calculation, production, and high-stakes requests use
  verified candidate selection.
- When an OpenAI-compatible frontier endpoint is configured, it performs the
  expert review pass. Without one, EliteOmni uses a second local generation.
- Candidate selection uses the existing V18 artifact verifier, source checks,
  and deterministic task-specific scores.

## Optional frontier provider

Set all three values to enable it:

```env
ELITE_FRONTIER_BASE_URL=https://provider.example
ELITE_FRONTIER_API_KEY=replace_me
ELITE_FRONTIER_MODEL=provider-model-id
```

The endpoint must support the OpenAI-compatible
`POST /v1/chat/completions` format.

## Modes

- `off`: disable V20 escalation
- `balanced`: retry failures and hard coding/research/calculation tasks
- `aggressive`: also compare candidates for hard general reasoning and medium
  coding/research tasks

## Repository intelligence

The root `code_rag.py` now builds a cached hybrid index from:

- Python functions and classes
- module headers and imports
- overlapping source windows
- file paths and symbol names
- traceback `file.py:line` targets

It returns line-numbered, diversified context instead of raw fixed 20-line
term-frequency matches.

## Evaluation

Validate tasks:

```bash
python3 scripts/run_frontier_eval.py --dry-run
```

Run against a local server:

```bash
python3 scripts/run_frontier_eval.py \
  --base-url http://127.0.0.1:8000
```

Results are written to `artifacts/frontier_gap_v20_results.json`.
