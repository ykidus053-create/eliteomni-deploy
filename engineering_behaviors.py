"""
Engineering Behaviors Module
Implements senior-engineer patterns: tests, validation, edge cases,
refactoring, design explanations, iterative agentic workflows,
and synthetic task generation.
"""

from __future__ import annotations

import re
import ast
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CodeReview:
    has_tests: bool = False
    has_validation: bool = False
    has_error_handling: bool = False
    has_logging: bool = False
    has_security: bool = False
    has_types: bool = False
    missing: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class EngineeringPlan:
    goal: str = ""
    assumptions: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    tests_needed: list[str] = field(default_factory=list)
    security_notes: list[str] = field(default_factory=list)
    design_rationale: str = ""


# ---------------------------------------------------------------------------
# 1. Code reviewer — detects missing engineering patterns
# ---------------------------------------------------------------------------

class CodeReviewer:
    """Analyse a code snippet and flag missing engineering patterns."""

    PATTERNS = {
        "has_tests": [
            r"\bdef test_", r"\bassert\b", r"unittest", r"pytest", r"@pytest"
        ],
        "has_validation": [
            r"\bif\b.*\bis None\b", r"\braise ValueError", r"\braise TypeError",
            r"\bisinstance\(", r"\.strip\(\)", r"len\(.*\)\s*[<=>]"
        ],
        "has_error_handling": [
            r"\btry\b", r"\bexcept\b", r"\bfinally\b", r"\braise\b"
        ],
        "has_logging": [
            r"\blogging\.", r"\blogger\.", r"print\(.*error", r"print\(.*warn"
        ],
        "has_security": [
            r"sanitize", r"escape\(", r"hashlib", r"secrets\.", r"hmac\.",
            r"parameterized", r"\.encode\("
        ],
        "has_types": [
            r":\s*(int|str|float|bool|list|dict|tuple|Optional|Union|Any)\b",
            r"->\s*(int|str|float|bool|None|dict|list)"
        ],
    }

    LABELS = {
        "has_tests": "unit tests",
        "has_validation": "input validation",
        "has_error_handling": "error handling",
        "has_logging": "logging",
        "has_security": "security considerations",
        "has_types": "type annotations",
    }

    def review(self, code: str) -> CodeReview:
        r = CodeReview()
        for attr, patterns in self.PATTERNS.items():
            matched = any(re.search(p, code) for p in patterns)
            setattr(r, attr, matched)
            if not matched:
                label = self.LABELS[attr]
                r.missing.append(label)
                r.suggestions.append(self._suggestion(attr))
        r.score = round(
            sum(getattr(r, a) for a in self.PATTERNS) / len(self.PATTERNS) * 10, 1
        )
        return r

    @staticmethod
    def _suggestion(attr: str) -> str:
        return {
            "has_tests": "Add pytest tests covering happy path and edge cases.",
            "has_validation": "Validate inputs at function entry; raise ValueError on bad data.",
            "has_error_handling": "Wrap external calls in try/except and surface meaningful errors.",
            "has_logging": "Add logger.debug/info/warning calls at key decision points.",
            "has_security": "Consider input sanitization, secrets management, and injection risks.",
            "has_types": "Add PEP-484 type hints to all public functions.",
        }.get(attr, "Improve this area.")


# ---------------------------------------------------------------------------
# 2. Engineering planner — think before coding
# ---------------------------------------------------------------------------

class EngineeringPlanner:
    """
    Given a task description, produce a structured engineering plan
    that mirrors how a senior engineer thinks before writing a line of code.
    """

    def plan(self, task: str) -> EngineeringPlan:
        p = EngineeringPlan(goal=task)
        tl = task.lower()

        # Assumptions
        p.assumptions = self._infer_assumptions(tl)

        # Edge cases
        p.edge_cases = self._infer_edge_cases(tl)

        # Steps (generic iterative workflow)
        p.steps = [
            "Clarify requirements and acceptance criteria",
            "Design data structures / interfaces first",
            "Implement core logic with inline comments",
            "Add input validation and error handling",
            "Write unit tests (happy path + edge cases)",
            "Refactor for readability and DRY principles",
            "Add logging and observability hooks",
            "Review security implications",
            "Document public API with docstrings",
            "Run linter and type checker before PR",
        ]

        # Tests needed
        p.tests_needed = [
            "test_happy_path — expected inputs produce expected outputs",
            "test_empty_input — handle None / empty gracefully",
            "test_boundary_values — min, max, off-by-one",
            "test_invalid_types — raise appropriate exceptions",
            "test_concurrent_access — if shared state is involved",
        ]

        # Security notes
        if any(k in tl for k in ["api", "request", "user", "input", "upload", "file"]):
            p.security_notes = [
                "Sanitize all user-supplied strings before use",
                "Never log sensitive values (passwords, tokens)",
                "Use parameterized queries — no raw SQL interpolation",
                "Rate-limit endpoints that accept arbitrary input",
            ]

        # Design rationale placeholder
        p.design_rationale = (
            "Chosen approach prioritises readability and testability over premature optimisation. "
            "Interfaces are defined before implementations to allow mocking in tests."
        )

        return p

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _infer_assumptions(tl: str) -> list[str]:
        base = [
            "Input data is UTF-8 encoded unless otherwise specified",
            "Callers handle returned errors — this module does not swallow exceptions silently",
        ]
        if "database" in tl or " db " in tl or "sql" in tl:
            base.append("Database connection is managed externally and passed in")
        if "async" in tl or "await" in tl:
            base.append("Caller runs an asyncio event loop")
        if "file" in tl or "path" in tl:
            base.append("Paths are absolute or resolved relative to project root")
        return base

    @staticmethod
    def _infer_edge_cases(tl: str) -> list[str]:
        cases = [
            "Empty or None inputs",
            "Inputs at maximum allowed size / length",
            "Unicode and special characters in string fields",
            "Concurrent calls modifying shared state",
        ]
        if "list" in tl or "array" in tl:
            cases.append("Empty list, single-element list, very large list")
        if "number" in tl or "int" in tl or "float" in tl:
            cases.append("Zero, negative numbers, float precision issues")
        if "file" in tl or "path" in tl:
            cases.append("File does not exist, permission denied, file already open")
        if "network" in tl or "http" in tl or "api" in tl:
            cases.append("Timeout, connection refused, 4xx/5xx responses, partial responses")
        return cases


# ---------------------------------------------------------------------------
# 3. Agentic iteration loop
# ---------------------------------------------------------------------------

class AgenticIterator:
    """
    Drives the Read → Analyse → Plan → Edit → Review → Fix cycle
    described in the post-training research notes.
    """

    PHASES = ["read", "analyse", "plan", "edit", "review", "fix"]

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self._history: list[dict] = []

    # -- public API ----------------------------------------------------------

    def run(
        self,
        code: str,
        task: str,
        apply_fn,   # callable(code, plan) -> str  (your LLM call)
    ) -> str:
        """
        Iterate until the code passes review or max_iterations is reached.

        Parameters
        ----------
        code      : starting source code (may be empty string for new files)
        task      : natural-language description of what needs to be done
        apply_fn  : function that receives (current_code, plan_dict) and
                    returns improved code as a string
        """
        reviewer = CodeReviewer()
        planner = EngineeringPlanner()

        for i in range(self.max_iterations):
            phase_log = {"iteration": i + 1, "phases": []}

            # READ
            phase_log["phases"].append("read")
            review = reviewer.review(code)

            # ANALYSE
            phase_log["phases"].append("analyse")
            logger.info(
                "Iteration %d — score %.1f/10 — missing: %s",
                i + 1, review.score, review.missing or "nothing"
            )

            if review.score >= 9.0 and i > 0:
                logger.info("Code meets quality threshold. Stopping.")
                phase_log["result"] = "passed"
                self._history.append(phase_log)
                break

            # PLAN
            phase_log["phases"].append("plan")
            plan = planner.plan(task)
            plan_dict = {
                "goal": plan.goal,
                "missing_patterns": review.missing,
                "suggestions": review.suggestions,
                "edge_cases": plan.edge_cases,
                "steps": plan.steps,
            }

            # EDIT
            phase_log["phases"].append("edit")
            try:
                code = apply_fn(code, plan_dict)
            except Exception as exc:
                logger.error("apply_fn failed: %s", exc)
                phase_log["error"] = str(exc)
                self._history.append(phase_log)
                break

            # REVIEW
            phase_log["phases"].append("review")
            review = reviewer.review(code)

            # FIX (mark remaining issues)
            phase_log["phases"].append("fix")
            phase_log["remaining_issues"] = review.missing
            phase_log["score"] = review.score
            self._history.append(phase_log)

        return code

    def history(self) -> list[dict]:
        return list(self._history)


# ---------------------------------------------------------------------------
# 4. Assumption checker
# ---------------------------------------------------------------------------

class AssumptionChecker:
    """
    Given a function's source, extract and validate its implicit assumptions.
    Returns a list of (assumption, status) pairs.
    """

    def check(self, source: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []

        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError as exc:
            return [("parse_error", str(exc))]

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                results.extend(self._check_function(node, source))

        return results

    @staticmethod
    def _check_function(node: ast.FunctionDef, source: str) -> list[tuple[str, str]]:
        issues = []
        args = [a.arg for a in node.args.args if a.arg != "self"]

        # Check: are arguments validated?
        body_src = ast.unparse(node) if hasattr(ast, "unparse") else source
        has_validation = "raise" in body_src or "isinstance" in body_src or "assert" in body_src

        if args and not has_validation:
            issues.append((
                f"{node.name}: assumes valid inputs for {args}",
                "UNVERIFIED — add input validation"
            ))
        else:
            issues.append((
                f"{node.name}: input validation present",
                "OK"
            ))

        # Check: does it have a docstring?
        has_doc = (
            isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ) if node.body else False

        issues.append((
            f"{node.name}: documented",
            "OK" if has_doc else "MISSING — add a docstring"
        ))

        return issues


# ---------------------------------------------------------------------------
# 5. Refactor suggester
# ---------------------------------------------------------------------------

class RefactorSuggester:
    """Surface simple refactor opportunities via static analysis."""

    def suggest(self, source: str) -> list[str]:
        suggestions: list[str] = []
        lines = source.splitlines()

        # Long functions
        try:
            tree = ast.parse(textwrap.dedent(source))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    length = (node.end_lineno or 0) - node.lineno
                    if length > 40:
                        suggestions.append(
                            f"Function `{node.name}` is {length} lines — consider splitting "
                            "into smaller focused functions."
                        )
        except SyntaxError:
            pass

        # Magic numbers
        if re.search(r"\b(?<!\.)(?!0\b)(?!1\b)\d{2,}\b", source):
            suggestions.append(
                "Magic numbers detected — extract to named constants or config."
            )

        # Repeated string literals
        strings = re.findall(r'"([^"]{4,})"', source)
        seen: dict[str, int] = {}
        for s in strings:
            seen[s] = seen.get(s, 0) + 1
        for s, count in seen.items():
            if count >= 3:
                suggestions.append(
                    f'String "{s[:40]}" appears {count} times — extract to a constant.'
                )

        # Nested ifs > 3 deep
        max_depth = 0
        depth = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("if ") or stripped.startswith("elif "):
                depth += 1
                max_depth = max(max_depth, depth)
            elif stripped.startswith(("return", "break", "continue", "pass")):
                depth = max(0, depth - 1)
        if max_depth >= 4:
            suggestions.append(
                f"Nesting depth of {max_depth} detected — use early returns or extract helpers."
            )

        if not suggestions:
            suggestions.append("No obvious refactor opportunities detected.")

        return suggestions


# ---------------------------------------------------------------------------
# 6. Synthetic task generator (for self-training / evals)
# ---------------------------------------------------------------------------

class SyntheticTaskGenerator:
    """
    Generate synthetic coding tasks for eval loops.
    Mirrors the Anthropic loop: generate task → generate solution →
    generate tests → filter → use for training.
    """

    TEMPLATES = [
        "Implement a function `{name}` that {verb} {noun} with full validation, "
        "error handling, and pytest tests.",
        "Refactor `{name}` to use {pattern} pattern. Keep behaviour identical and "
        "add regression tests.",
        "Add input validation and type hints to `{name}`. Return meaningful errors "
        "for invalid inputs.",
        "Write a class `{name}` that {verb} {noun}. Include __repr__, __eq__, "
        "docstrings, and unit tests.",
        "Debug and fix `{name}` — it fails on edge case: {edge_case}. "
        "Add a regression test.",
    ]

    VERBS = ["parses", "validates", "transforms", "filters", "aggregates",
             "serialises", "caches", "retries", "schedules", "monitors"]
    NOUNS = ["user input", "API responses", "file paths", "configuration",
             "database records", "log entries", "task queues", "event streams"]
    PATTERNS = ["repository", "strategy", "observer", "factory", "decorator",
                "command", "chain-of-responsibility"]
    EDGE_CASES = ["empty string input", "None value", "integer overflow",
                  "concurrent modification", "network timeout", "malformed JSON",
                  "Unicode emoji in text field", "negative index"]

    def generate(self, n: int = 5, seed_name: str = "my_function") -> list[dict]:
        import random
        tasks = []
        for i in range(n):
            tmpl = random.choice(self.TEMPLATES)
            task_str = tmpl.format(
                name=seed_name,
                verb=random.choice(self.VERBS),
                noun=random.choice(self.NOUNS),
                pattern=random.choice(self.PATTERNS),
                edge_case=random.choice(self.EDGE_CASES),
            )
            tasks.append({
                "id": f"synth_{i+1:04d}",
                "task": task_str,
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "tags": random.sample(
                    ["validation", "testing", "refactor", "error-handling",
                     "types", "security", "design-pattern"],
                    k=random.randint(2, 4)
                ),
            })
        return tasks


# ---------------------------------------------------------------------------
# 7. Engineering behavior prompt builder
# ---------------------------------------------------------------------------

class EngineeringPromptBuilder:
    """
    Augments a raw user prompt with senior-engineer context so the LLM
    naturally produces tests, validation, edge-case handling, etc.
    """

    SYSTEM_ADDON = """
You are a senior software engineer with 15+ years of experience.
For every coding task you MUST:
1. State your assumptions before writing code.
2. Handle edge cases explicitly (None, empty, boundary values).
3. Add input validation with meaningful error messages.
4. Include try/except around I/O, network, and DB operations.
5. Write pytest unit tests covering happy path AND edge cases.
6. Add PEP-484 type hints to all public functions.
7. Add a module-level and function-level docstring.
8. Use structured logging (logger.info / logger.error) not print().
9. Flag any security considerations inline as # SECURITY: comments.
10. After code, write a short "Design rationale" section explaining choices.
""".strip()

    def build(self, user_prompt: str, context: dict | None = None) -> dict:
        """
        Returns {"system": ..., "user": ...} ready to pass to your LLM client.
        """
        system = self.SYSTEM_ADDON
        if context:
            plan_text = "\n".join(
                f"- {k}: {v}" for k, v in context.items()
            )
            system += f"\n\nContext from engineering planner:\n{plan_text}"

        return {"system": system, "user": user_prompt}


# ---------------------------------------------------------------------------
# Convenience facade
# ---------------------------------------------------------------------------

class EngineeringBehaviors:
    """Single entry point wiring all components together."""

    def __init__(self):
        self.reviewer = CodeReviewer()
        self.planner = EngineeringPlanner()
        self.iterator = AgenticIterator()
        self.assumption_checker = AssumptionChecker()
        self.refactor_suggester = RefactorSuggester()
        self.task_generator = SyntheticTaskGenerator()
        self.prompt_builder = EngineeringPromptBuilder()

    # Shortcuts
    def review(self, code: str) -> CodeReview:
        return self.reviewer.review(code)

    def plan(self, task: str) -> EngineeringPlan:
        return self.planner.plan(task)

    def iterate(self, code: str, task: str, apply_fn) -> str:
        return self.iterator.run(code, task, apply_fn)

    def check_assumptions(self, source: str):
        return self.assumption_checker.check(source)

    def refactor_hints(self, source: str) -> list[str]:
        return self.refactor_suggester.suggest(source)

    def synthetic_tasks(self, n: int = 5, seed: str = "fn") -> list[dict]:
        return self.task_generator.generate(n, seed)

    def build_prompt(self, user_prompt: str, context: dict | None = None) -> dict:
        return self.prompt_builder.build(user_prompt, context)

    def full_pipeline(self, task: str, code: str, apply_fn) -> dict:
        """
        Run the complete pipeline and return a results dict.
        Useful for wiring into agent_core.py.
        """
        plan = self.plan(task)
        prompt = self.build_prompt(task, {"edge_cases": plan.edge_cases})
        final_code = self.iterate(code, task, apply_fn)
        review = self.review(final_code)
        refactor = self.refactor_hints(final_code)
        assumptions = self.check_assumptions(final_code)
        return {
            "plan": plan,
            "prompt": prompt,
            "final_code": final_code,
            "review": review,
            "refactor_hints": refactor,
            "assumption_checks": assumptions,
            "iteration_history": self.iterator.history(),
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    eb = EngineeringBehaviors()

    sample = '''
def add_user(name, email):
    db.insert(name, email)
    return True
'''

    print("=== CODE REVIEW ===")
    r = eb.review(sample)
    print(f"Score: {r.score}/10")
    print(f"Missing: {r.missing}")
    print(f"Suggestions:\n" + "\n".join(f"  - {s}" for s in r.suggestions))

    print("\n=== ENGINEERING PLAN ===")
    p = eb.plan("Build a user registration API endpoint")
    print(f"Edge cases: {p.edge_cases[:3]}")
    print(f"Steps (first 3): {p.steps[:3]}")

    print("\n=== ASSUMPTION CHECK ===")
    for assumption, status in eb.check_assumptions(sample):
        print(f"  [{status}] {assumption}")

    print("\n=== REFACTOR HINTS ===")
    for hint in eb.refactor_hints(sample):
        print(f"  - {hint}")

    print("\n=== SYNTHETIC TASKS ===")
    for t in eb.synthetic_tasks(2, "add_user"):
        print(f"  [{t['difficulty']}] {t['task'][:80]}...")

    print("\n=== PROMPT BUILDER ===")
    prompt = eb.build_prompt("Write a login function")
    print(prompt["system"][:200] + "...")
