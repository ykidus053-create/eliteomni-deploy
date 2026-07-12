from modules.secure_code_gate_v31 import (
    enforce_secure_code_output,
    inspect_generated_code,
)


def test_python_syntax_error_is_blocked():
    response = "```python\ndef broken(:\n    return 1\n```"
    report = inspect_generated_code(
        response,
        "Implement a secure production function with tests.",
    )
    assert not report.passed
    assert any(
        issue.code == "python.syntax"
        for issue in report.issues
    )


def test_unclosed_fence_is_blocked():
    report = inspect_generated_code(
        "```python\nprint('x')",
        "Write code.",
    )
    assert not report.passed
    assert any(
        issue.code == "markdown.unclosed_fence"
        for issue in report.issues
    )


def test_eval_and_hardcoded_secret_are_blocked():
    response = '''
```python
api_key = "super-secret-live-key"
value = eval(user_input)
```
'''
    report = inspect_generated_code(
        response,
        "Build a secure production API with tests.",
    )
    codes = {issue.code for issue in report.issues}
    assert "python.dynamic_execution" in codes
    assert "python.hardcoded_secret" in codes
    assert not report.passed


def test_sql_interpolation_is_blocked():
    response = '''
```python
def find_user(cursor, user_id):
    return cursor.execute(
        f"SELECT * FROM users WHERE id = {user_id}"
    ).fetchone()
```
```python
def test_find_user():
    assert True
```
'''
    report = inspect_generated_code(
        response,
        "Implement a secure production database query with tests.",
    )
    assert any(
        issue.code == "python.sql_injection"
        for issue in report.issues
    )


def test_async_blocking_sleep_is_blocked():
    response = '''
```python
import time

async def worker():
    time.sleep(1)
```
```python
def test_worker():
    assert True
```
'''
    report = inspect_generated_code(
        response,
        "Build a production async worker with tests.",
    )
    assert any(
        issue.code == "python.blocking_async_sleep"
        for issue in report.issues
    )


def test_production_http_request_requires_timeout():
    response = '''
```python
import requests

def fetch(url: str) -> str:
    return requests.get(url).text
```
```python
def test_fetch():
    assert True
```
'''
    report = inspect_generated_code(
        response,
        "Build a production API integration with tests.",
    )
    assert any(
        issue.code == "python.http_timeout_missing"
        for issue in report.issues
    )


def test_secure_python_with_tests_passes():
    response = '''
```python
import os
import sqlite3

def find_user(
    connection: sqlite3.Connection,
    user_id: int,
) -> tuple[int, str] | None:
    row = connection.execute(
        "SELECT id, name FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return tuple(row) if row else None

API_TOKEN = os.environ["API_TOKEN"]
```
```python
def test_find_user() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE users(id INTEGER, name TEXT)"
    )
    connection.execute(
        "INSERT INTO users VALUES (?, ?)",
        (1, "Ada"),
    )
    assert find_user(connection, 1) == (1, "Ada")
```
Validate with: `pytest -q`
'''
    report = inspect_generated_code(
        response,
        "Implement a secure production database function with tests.",
    )
    assert report.passed, report.to_dict()


def test_repair_loop_replaces_entire_answer():
    bad = "```python\ndef bad(:\n    return 1\n```"
    good = '''
```python
def add(left: int, right: int) -> int:
    return left + right
```
```python
def test_add() -> None:
    assert add(2, 3) == 5
```
Validate with: `pytest -q`
'''
    prompts = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return good

    outcome = enforce_secure_code_output(
        bad,
        "Implement a production add function with tests.",
        generate,
        max_rounds=2,
    )
    assert outcome.released
    assert outcome.repair_rounds == 1
    assert outcome.report.passed
    assert outcome.text.strip() == good.strip()
    assert "COMPLETE REPLACEMENT ANSWER" in prompts[0]


def test_gate_fails_closed_after_bad_repair():
    bad = "```python\ndef bad(:\n    return 1\n```"

    outcome = enforce_secure_code_output(
        bad,
        "Implement production code with tests.",
        lambda _: bad,
        max_rounds=1,
        fail_closed=True,
    )
    assert not outcome.released
    assert "No unvalidated code was returned." in outcome.text
    assert "def bad" not in outcome.text


def test_missing_tests_blocks_production_code():
    response = '''
```python
def add(left: int, right: int) -> int:
    return left + right
```
'''
    report = inspect_generated_code(
        response,
        "Build a production API.",
    )
    assert any(
        issue.code == "general.tests_missing"
        for issue in report.issues
    )


def test_javascript_eval_is_blocked():
    response = '''
```javascript
const result = eval(input);
```
```javascript
test("safe", () => expect(true).toBe(true));
```
'''
    report = inspect_generated_code(
        response,
        "Build a secure production JavaScript service with tests.",
    )
    assert any(
        issue.code == "javascript.eval"
        for issue in report.issues
    )


def test_shell_insecure_tls_is_blocked():
    response = '''
```bash
curl -k https://example.test
```
```bash
test -n "ok"
```
'''
    report = inspect_generated_code(
        response,
        "Write a secure production shell deployment with tests.",
    )
    assert any(
        issue.code == "bash.insecure_tls"
        for issue in report.issues
    )
