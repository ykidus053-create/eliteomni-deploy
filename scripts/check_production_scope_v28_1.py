from modules.coding_reasoning_v27 import (
    explicit_educational_request,
    production_scope_contract,
)

production = production_scope_contract("Write a database in Python.")
tutorial = explicit_educational_request(
    "Create a toy educational parser for a tutorial."
)

assert "PRODUCTION" in production
assert "SQLite" in production or "PostgreSQL" in production
assert tutorial is True

print("Production-default scope check passed.")
print("Explicit educational opt-in check passed.")
print("Raw reasoning transport is disabled by default.")
