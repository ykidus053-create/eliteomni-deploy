"""Evidence-based coding-agent benchmark and scoring for EliteOmni V30."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_CODE_FENCE = re.compile(r"```[A-Za-z0-9_+.-]*\n.*?```", re.DOTALL)
_TEST_SIGNAL = re.compile(
    r"\b("
    r"pytest|unittest|test_[A-Za-z0-9_]+|"
    r"npm (?:run )?test|pnpm (?:run )?test|"
    r"go test|cargo test|dotnet test|mvn test|gradle test"
    r")\b",
    re.IGNORECASE,
)
_VALIDATION_SIGNAL = re.compile(
    r"\b("
    r"python(?:3)? -m py_compile|pytest|ruff|mypy|"
    r"npm (?:run )?(?:test|lint|build)|"
    r"docker compose|curl -f|go test|cargo test|"
    r"dotnet test|mvn test|gradle test"
    r")\b",
    re.IGNORECASE,
)
_ERROR_SIGNAL = re.compile(
    r"\b("
    r"try:|except |raise |HTTPException|error handling|"
    r"rollback|finally:|Result<|throws?"
    r")\b",
    re.IGNORECASE,
)
_CONFIG_SIGNAL = re.compile(
    r"\b("
    r"os\.getenv|os\.environ|BaseSettings|environment variable|"
    r"config(?:uration)?|settings|dotenv"
    r")\b",
    re.IGNORECASE,
)
_SECURITY_SIGNAL = re.compile(
    r"\b("
    r"parameterized|prepared statement|authentication|authorization|"
    r"bcrypt|argon2|constant[- ]time|least privilege|"
    r"path traversal|input validation|rate limit|csrf|csp|tls"
    r")\b",
    re.IGNORECASE,
)
_OBSERVABILITY_SIGNAL = re.compile(
    r"\b(logging|logger|metrics|trace|request[_ -]?id|health|readiness)\b",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(
    r"\b(TODO|FIXME|NotImplementedError|left as an exercise|"
    r"implementation omitted|rest of code|your_api_key|changeme)\b",
    re.IGNORECASE,
)
_DOWNGRADE = re.compile(
    r"\b(toy|educational implementation|demo only|proof of concept only|"
    r"not production[- ]ready|for simplicity we use an in-memory)\b",
    re.IGNORECASE,
)
_UNSAFE = re.compile(
    r"\b(eval\(|exec\(|pickle\.loads|shell=True|verify=False)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    required_signals: tuple[str, ...] = ()
    forbidden_signals: tuple[str, ...] = ()
    minimum_score: int = 85


@dataclass(frozen=True)
class Evaluation:
    case_id: str
    score: int
    passed: bool
    critical_failure: bool
    evidence: tuple[str, ...]
    violations: tuple[str, ...]


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid benchmark JSON on line {line_number}: {exc}"
                ) from exc
            cases.append(
                BenchmarkCase(
                    case_id=str(raw["case_id"]),
                    category=str(raw["category"]),
                    prompt=str(raw["prompt"]),
                    required_signals=tuple(
                        str(item)
                        for item in raw.get("required_signals", [])
                    ),
                    forbidden_signals=tuple(
                        str(item)
                        for item in raw.get("forbidden_signals", [])
                    ),
                    minimum_score=int(raw.get("minimum_score", 85)),
                )
            )
    if not cases:
        raise ValueError("benchmark suite is empty")
    return cases


def evaluate_response(
    case: BenchmarkCase,
    response: str,
) -> Evaluation:
    text = response or ""
    score = 0
    evidence: list[str] = []
    violations: list[str] = []
    critical = False

    if _CODE_FENCE.search(text):
        score += 15
        evidence.append("executable code block present")
    else:
        violations.append("no extractable code block")

    if _TEST_SIGNAL.search(text):
        score += 15
        evidence.append("test implementation or command present")
    else:
        violations.append("missing tests")

    if _VALIDATION_SIGNAL.search(text):
        score += 10
        evidence.append("exact validation command present")
    else:
        violations.append("missing validation command")

    if _ERROR_SIGNAL.search(text):
        score += 10
        evidence.append("structured error handling present")
    else:
        violations.append("missing error-handling evidence")

    if _CONFIG_SIGNAL.search(text):
        score += 10
        evidence.append("externalized configuration present")
    else:
        violations.append("configuration is not evidenced")

    if _SECURITY_SIGNAL.search(text):
        score += 10
        evidence.append("security control present")
    else:
        violations.append("security design is not evidenced")

    if _OBSERVABILITY_SIGNAL.search(text):
        score += 5
        evidence.append("observability signal present")
    else:
        violations.append("observability is not evidenced")

    if _PLACEHOLDER.search(text):
        violations.append("placeholder or omitted implementation present")
        critical = True
    else:
        score += 15
        evidence.append("no placeholder language detected")

    if _DOWNGRADE.search(text):
        violations.append("production request was downgraded")
        critical = True
    else:
        score += 10
        evidence.append("no scope downgrade detected")

    if _UNSAFE.search(text):
        violations.append("high-risk unsafe primitive detected")
        critical = True
        score = max(0, score - 30)

    lowered = text.lower()
    for required in case.required_signals:
        if required.lower() not in lowered:
            violations.append(
                f"required signal missing: {required}"
            )
            score = max(0, score - 5)
        else:
            evidence.append(
                f"required signal present: {required}"
            )

    for forbidden in case.forbidden_signals:
        if forbidden.lower() in lowered:
            violations.append(
                f"forbidden signal present: {forbidden}"
            )
            critical = True
            score = max(0, score - 20)

    score = max(0, min(100, score))
    passed = score >= case.minimum_score and not critical
    return Evaluation(
        case_id=case.case_id,
        score=score,
        passed=passed,
        critical_failure=critical,
        evidence=tuple(dict.fromkeys(evidence)),
        violations=tuple(dict.fromkeys(violations)),
    )


def summarize(evaluations: Iterable[Evaluation]) -> dict:
    items = list(evaluations)
    if not items:
        raise ValueError("no evaluations supplied")

    average = round(
        sum(item.score for item in items) / len(items),
        2,
    )
    passed = sum(item.passed for item in items)
    critical = sum(item.critical_failure for item in items)
    return {
        "suite": "EliteOmni Coding Benchmark V30",
        "cases": len(items),
        "passed": passed,
        "failed": len(items) - passed,
        "critical_failures": critical,
        "average_score": average,
        "pass_rate": round(passed / len(items), 4),
        "release_approved": (
            average >= 90
            and passed == len(items)
            and critical == 0
        ),
        "results": [asdict(item) for item in items],
    }


__all__ = [
    "BenchmarkCase",
    "Evaluation",
    "evaluate_response",
    "load_cases",
    "summarize",
]
