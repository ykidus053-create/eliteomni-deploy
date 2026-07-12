from __future__ import annotations

import json

from modules.coding_reasoning_v27 import (
    architect_plan,
    prefetch_plan,
    runtime_status,
)


print(json.dumps(runtime_status(), indent=2))
print("\nSAMPLE PLAN")
print(architect_plan("Fix modules/services/agents.py:265 without breaking the API"))
print("\nPREFETCH KEYS")
print(sorted(prefetch_plan("Fix modules/services/agents.py:265", "coder")))

