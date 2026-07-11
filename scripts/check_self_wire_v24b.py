from __future__ import annotations

import json

import self_wire

snapshot = self_wire.status()
print(json.dumps(snapshot, indent=2))

if not snapshot["enabled"]:
    assert self_wire.start() is None
    print("Production-safe: self-wire remained disabled.")
else:
    print("Development self-wire is explicitly enabled.")
