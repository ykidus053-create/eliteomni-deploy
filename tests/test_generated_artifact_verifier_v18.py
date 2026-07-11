from modules.quality_kernel import (
    analyze_request,
    audit_answer,
    verify_python_artifact,
)


def test_python_project_with_passing_test_is_verified():
    response = r"""
File: `calculator.py`
```python
def add(a, b):
    return a + b
```

File: `test_calculator.py`
```python
from calculator import add

def test_add():
    assert add(2, 3) == 5
```
"""
    result = verify_python_artifact(response)
    assert result.syntax_ok is True
    assert result.tests_found is True
    assert result.tests_passed is True


def test_python_syntax_failure_is_rejected():
    response = """
```python
def broken(:
    pass
```
"""
    result = verify_python_artifact(response)
    assert result.attempted is True
    assert result.syntax_ok is False


def test_production_code_without_tests_is_rejected():
    response = """
```python
def add(a, b):
    return a + b
```
"""
    profile = analyze_request(
        "Build a complete production-ready Python service"
    )
    audit = audit_answer(
        "Build a complete production-ready Python service",
        response,
        profile,
    )
    assert audit.approved is False
    assert any(issue.code == "code.tests_missing" for issue in audit.issues)
