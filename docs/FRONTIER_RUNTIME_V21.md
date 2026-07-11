# Frontier Runtime V21

V21 focuses on active-path correctness rather than adding another prompt layer.

## Changes

1. Activates a GLM-4.7 multi-candidate runtime after the V18 verification hook.
2. Generates independent challenger answers only for difficult, rejected,
   production, or high-stakes tasks.
3. Adds a third adversarial synthesis candidate only in aggressive mode or when
   the first candidates are both weak/close.
4. Preserves all internal loop iterations instead of discarding earlier output.
5. Removes the duplicated loop-engine pass.
6. Makes autonomous background daemons opt-in and idempotent.
7. Adds `/runtime/v21` status metadata.

## Recommended configuration

```env
ELITE_FRONTIER_V21_MODE=balanced
ELITE_FRONTIER_V21_CANDIDATES=2
ELITE_AGENTIC_MAX_ITERS=3

ELITE_ENABLE_PROACTIVE_DAEMON=0
ELITE_ENABLE_AGI_EMULATION=0
ELITE_ENABLE_REFACTOR_DAEMON=0
ELITE_ENABLE_SELF_HEALING=0
```

Use `aggressive` and three candidates only when extra latency and inference cost
are acceptable.
