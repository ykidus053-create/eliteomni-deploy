# Production Scope Guard V28.1

V28.1 prevents ambiguous coding requests from being downgraded to toy,
educational, demonstration, or in-memory substitute implementations.

For database requests, the default is a real application using SQLite or
PostgreSQL unless the user explicitly asks to build a database engine.

Raw provider reasoning is hidden by default. The UI may still show a generic
thinking indicator.

```env
ELITE_PRODUCTION_SCOPE_DEFAULT=1
ELITE_EXPOSE_MODEL_REASONING=0
```
