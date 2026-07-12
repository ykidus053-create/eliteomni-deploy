#!/usr/bin/env python3
# Deterministic smoke check for the V31 generated-code release gate.

from __future__ import annotations

import json

from modules.secure_code_gate_v31 import inspect_generated_code


def main() -> int:
    insecure = '''
```python
password = "production-password"
result = eval(user_input)
```
'''
    secure = '''
```python
import os

def add(left: int, right: int) -> int:
    return left + right

API_TOKEN = os.environ["API_TOKEN"]
```
```python
def test_add() -> None:
    assert add(2, 3) == 5
```
Validate with: `pytest -q`
'''

    bad_report = inspect_generated_code(
        insecure,
        "Build a secure production API with tests.",
    )
    good_report = inspect_generated_code(
        secure,
        "Build a secure production API with tests.",
    )

    result = {
        "version": "V31",
        "insecure_code_blocked": not bad_report.passed,
        "secure_code_approved": good_report.passed,
        "insecure_findings": [
            item.code for item in bad_report.issues
        ],
        "policy": {
            "syntax_validation": True,
            "security_ast_scan": True,
            "production_tests_required": True,
            "whole_answer_repair": True,
            "fail_closed": True,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if (
        not bad_report.passed and good_report.passed
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
