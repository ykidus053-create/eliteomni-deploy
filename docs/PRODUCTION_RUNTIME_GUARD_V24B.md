# Production Runtime Guard V24b

V24b replaces the historical startup-time hot importer with a production-safe
implementation.

## Railway settings

```env
ELITE_SELF_WIRE=0
ELITE_SELF_WIRE_SFT=0
```

## Behavior

- Disabled unless `ELITE_SELF_WIRE=1`.
- Starts no duplicate watcher threads.
- Never imports known repair, mutation, migration, or training scripts.
- Loads watched development modules sequentially instead of concurrently.
- Creates the legacy `knowledge` table before optional development reindexing.
- Generates SFT examples only when both self-wire and SFT are explicitly on.
- Preserves the public `on_change`, `start`, `status`, and watcher functions.

Production deployments should receive source changes through Git and Railway,
not by rewriting source inside a running container.
