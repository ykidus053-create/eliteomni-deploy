# Repository Intelligence V28

V28 improves coding quality without another model request.

It indexes repository files, definitions, imports, calls, reverse dependents,
traceback targets, and likely regression tests. The V27 coder receives a
change-impact map before implementation.

```env
ELITE_REPO_INTELLIGENCE=1
ELITE_PROJECT_ROOT=/app
ELITE_REPO_MAX_FILES=800
ELITE_REPO_CONTEXT_FILES=10
ELITE_REPO_FILE_MAX_BYTES=262144
```

The index is metadata-cached and rebuilt when source files change.
