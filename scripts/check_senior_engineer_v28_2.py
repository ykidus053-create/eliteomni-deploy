from modules.coding_reasoning_v27 import (
    explicit_nonproduction_request,
    senior_engineer_contract,
)

contract = senior_engineer_contract(
    "Build a database-backed production API."
)

assert "MANDATORY SENIOR ENGINEER" in contract
assert "SQLite" in contract
assert "parameterized" in contract
assert not explicit_nonproduction_request(
    "Build a production service from scratch."
)

print("Senior-engineer production default check passed.")
print("Database production requirements check passed.")
print("Educational downgrade requires an explicit user request.")
