# Senior Engineer Production Guard V28.2

V28.2 makes senior-engineer implementation quality mandatory.

- Ambiguous coding requests default to real production scope.
- Reduced teaching/demo scope is allowed only when explicitly requested.
- Even reduced-scope code must remain clean, correct, tested, and maintainable.
- The verifier rejects toy/educational downgrades for production requests.
- The verifier rejects placeholders, fake secrets, incomplete code, and
  production code without tests or exact validation evidence.
- Private model reasoning remains hidden.

```env
ELITE_SENIOR_ENGINEER_DEFAULT=1
ELITE_PRODUCTION_SCOPE_DEFAULT=1
ELITE_EXPOSE_MODEL_REASONING=0
```
