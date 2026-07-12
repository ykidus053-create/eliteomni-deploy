# Secure Code Quality Gate V31

V31 prevents generated coding answers from being released before they pass
deterministic syntax, security, and reliability checks.

## Release flow

1. Extract every fenced code block.
2. Reject malformed Markdown fences.
3. Parse Python with `ast.parse`.
4. Run `node --check` for JavaScript when Node is available.
5. Run `bash -n` for shell scripts.
6. Parse JSON and perform structural SQL checks.
7. Scan Python ASTs for injection, unsafe deserialization, disabled TLS or
   JWT verification, hardcoded secrets, blocking async calls, missing
   production HTTP timeouts, swallowed exceptions, and mutable defaults.
8. Require tests for production coding requests.
9. Ask the model for a complete replacement answer when validation fails.
10. Re-run the full gate after each repair.
11. Fail closed and release no code if mandatory checks still fail.

The legacy code executor no longer appends execution errors or a separate
partial auto-fix to coder responses. V31 validates the final answer after
the existing verification pipeline, so later model rewrites cannot bypass
the gate.

## Environment

```env
ELITE_CODE_GATE_MAX_REPAIR_ROUNDS=2
ELITE_CODE_GATE_FAIL_CLOSED=1
ELITE_CODE_GATE_REQUIRE_TESTS=1
ELITE_CODE_GATE_MAX_BLOCKS=12
ELITE_CODE_GATE_MAX_CANDIDATE_CHARS=80000
```

Fail-closed mode should remain enabled in production.
