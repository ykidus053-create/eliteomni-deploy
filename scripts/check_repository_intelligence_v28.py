import json

from modules.repository_intelligence_v28 import (
    analyze_repository,
    format_repository_impact,
    runtime_status,
)

query = (
    'Fix NameError in File "modules/services/agents.py", '
    "line 265 without breaking pipeline_sync"
)

print(json.dumps(runtime_status(), indent=2))
print()
print(format_repository_impact(query))
print()
result = analyze_repository(query)
print(
    json.dumps(
        {
            "selected": [item["path"] for item in result["files"]],
            "tests": result["tests"],
            "risks": result["risks"],
        },
        indent=2,
    )
)
