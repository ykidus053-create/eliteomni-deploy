"""Frontier Runtime V21: activate GLM-4.7 multi-candidate deliberation.

This wrapper sits after the V18 verification hook. It uses the existing
pipeline as the capable worker, then asks it for independent alternatives only
when task difficulty or verification evidence justifies the extra cost.
"""
from __future__ import annotations

import contextvars
import dataclasses
import os
import re
from typing import Any, Mapping

from modules.frontier_runtime import (
    Candidate,
    _as_candidate,
    _judge_close_candidates,
)
from modules.quality_kernel import TaskProfile, analyze_request


_ACTIVE = contextvars.ContextVar("elite_frontier_v21_active", default=False)
_INSTALLED = False
_LAST_STATUS: dict[str, Any] = {
    "installed": False,
    "mode": "balanced",
    "last_request": None,
}


def _mode() -> str:
    value = os.getenv("ELITE_FRONTIER_V21_MODE", "balanced").strip().lower()
    return value if value in {"off", "balanced", "aggressive"} else "balanced"


def _candidate_budget() -> int:
    default = 3 if _mode() == "aggressive" else 2
    try:
        value = int(os.getenv("ELITE_FRONTIER_V21_CANDIDATES", str(default)))
    except ValueError:
        value = default
    return max(1, min(value, 3))


def _requirements_contract(message: str) -> str:
    """Extract explicit constraints without inventing hidden requirements."""
    text = message or ""
    clauses = re.split(r"(?<=[.!?])\s+|\n+", text)
    selected = []
    constraint_re = re.compile(
        r"\b(must|need|should|required|include|without|only|exactly|"
        r"do not|don't|no |avoid|preserve|compatible|production|test|"
        r"cite|source|current|latest|deadline|limit)\b",
        re.IGNORECASE,
    )
    for clause in clauses:
        cleaned = " ".join(clause.split()).strip()
        if cleaned and constraint_re.search(cleaned):
            selected.append(cleaned[:600])
        if len(selected) >= 12:
            break

    file_refs = sorted(
        set(
            re.findall(
                r"\b[A-Za-z0-9_.\-/]+\.(?:py|js|ts|tsx|jsx|go|rs|java|"
                r"sql|yaml|yml|json|toml|md)\b",
                text,
            )
        )
    )[:12]
    traceback_refs = sorted(
        set(
            re.findall(
                r"\b[A-Za-z0-9_.\-/]+\.py(?::\d+|[\"'],?\s*line\s+\d+)",
                text,
                re.IGNORECASE,
            )
        )
    )[:8]

    lines = ["Explicit requirements extracted from the original request:"]
    lines.extend(f"- {item}" for item in selected)
    if file_refs:
        lines.append("- Referenced files: " + ", ".join(file_refs))
    if traceback_refs:
        lines.append("- Traceback locations: " + ", ".join(traceback_refs))
    if len(lines) == 1:
        lines.append("- Satisfy the request completely without adding assumptions.")
    return "\n".join(lines)


def _should_deepen(profile: TaskProfile, first: Candidate) -> bool:
    if _mode() == "off" or _candidate_budget() <= 1:
        return False
    quality = first.result.get("quality_v18") or {}
    if (
        not first.audit.approved
        or quality.get("approved") is False
        or "I withheld the generated answer" in first.response
    ):
        return True
    if profile.production_claim or profile.high_stakes:
        return True
    if profile.complexity == "hard":
        return True
    if (
        _mode() == "aggressive"
        and profile.complexity == "medium"
        and profile.skill in {"coder", "researcher", "calculator"}
    ):
        return True
    return False


def _specialist_contract(profile: TaskProfile) -> str:
    if profile.skill == "coder":
        return (
            "Act as an independent senior repository engineer. Re-derive the "
            "solution instead of paraphrasing a prior answer. Inspect interfaces "
            "and failure paths, preserve compatibility, return complete runnable "
            "artifacts, add focused regression tests, and never claim execution "
            "without evidence."
        )
    if profile.skill == "researcher":
        return (
            "Act as an independent research analyst. Triangulate material claims "
            "across multiple sources, keep URLs close to claims, distinguish "
            "reported facts from inference, identify source disagreement, and "
            "avoid relying on model memory for current facts."
        )
    if profile.skill == "calculator":
        return (
            "Solve independently, track units and assumptions, recompute the "
            "result using a second method, and present one consistent answer."
        )
    return (
        "Solve independently from first principles. Check every explicit "
        "constraint, identify unsupported assumptions, and return a complete "
        "answer rather than commentary about the process."
    )


def _challenger_prompt(message: str, profile: TaskProfile) -> str:
    return (
        message
        + "\n\n[GLM-4.7 FRONTIER V21 — INDEPENDENT CHALLENGER]\n"
        + _specialist_contract(profile)
        + "\n\n"
        + _requirements_contract(message)
        + "\n\nReturn only the improved final answer to the original request."
    )


def _review_prompt(
    message: str,
    profile: TaskProfile,
    candidates: list[Candidate],
) -> str:
    rendered = []
    for index, candidate in enumerate(candidates, start=1):
        response = candidate.response
        if len(response) > 9000:
            response = response[:5500] + "\n...[clipped]...\n" + response[-2500:]
        issues = ", ".join(issue.code for issue in candidate.audit.issues) or "none"
        rendered.append(
            f"CANDIDATE {index} score={candidate.score} "
            f"approved={candidate.audit.approved} issues={issues}\n{response}"
        )
    return (
        message
        + "\n\n[GLM-4.7 FRONTIER V21 — ADVERSARIAL SYNTHESIS]\n"
        + "Produce one corrected final answer. Do not mention candidates or this "
        + "review. Keep every correct part, remove contradictions, repair failed "
        + "tests or unsupported claims, and satisfy the original constraints.\n\n"
        + _requirements_contract(message)
        + "\n\n"
        + "\n\n".join(rendered)
    )


def _close(left: Candidate, right: Candidate) -> bool:
    return abs(left.score - right.score) <= 12.0


def runtime_status() -> dict[str, Any]:
    return {
        **_LAST_STATUS,
        "installed": _INSTALLED,
        "mode": _mode(),
        "candidate_budget": _candidate_budget(),
    }


def install_frontier_runtime_v21(namespace: dict[str, Any]) -> None:
    global _INSTALLED, _LAST_STATUS
    if _INSTALLED:
        return

    original = namespace.get("pipeline_sync")
    if not callable(original):
        raise RuntimeError("pipeline_sync is unavailable for Frontier V21")

    def pipeline_sync_v21(message: str, history: list) -> dict[str, Any]:
        global _LAST_STATUS
        if _ACTIVE.get() or _mode() == "off":
            raw = original(message, history)
            return dict(raw) if isinstance(raw, Mapping) else {
                "response": str(raw),
                "skill": "general",
            }

        token = _ACTIVE.set(True)
        try:
            classifier = namespace.get("classify_skill")
            hint = classifier(message) if callable(classifier) else None
            profile = analyze_request(message, hint)

            baseline_raw = original(message, history)
            baseline = _as_candidate(
                "baseline",
                message,
                profile,
                baseline_raw,
            )
            candidates = [baseline]

            if _should_deepen(profile, baseline):
                try:
                    challenger_raw = original(
                        _challenger_prompt(message, profile),
                        history,
                    )
                    candidates.append(
                        _as_candidate(
                            "independent-challenger",
                            message,
                            profile,
                            challenger_raw,
                        )
                    )
                except Exception as exc:
                    print(f"[FrontierV21] challenger failed: {exc}")

            if (
                _candidate_budget() >= 3
                and len(candidates) >= 2
                and (
                    not any(candidate.audit.approved for candidate in candidates)
                    or _close(candidates[0], candidates[1])
                )
            ):
                try:
                    review_raw = original(
                        _review_prompt(message, profile, candidates),
                        history,
                    )
                    candidates.append(
                        _as_candidate(
                            "adversarial-synthesis",
                            message,
                            profile,
                            review_raw,
                        )
                    )
                except Exception as exc:
                    print(f"[FrontierV21] synthesis failed: {exc}")

            selected = max(candidates, key=lambda candidate: candidate.score)
            runner_up = sorted(
                candidates,
                key=lambda candidate: candidate.score,
                reverse=True,
            )[1:2]
            if runner_up and _close(selected, runner_up[0]):
                judged = _judge_close_candidates(
                    namespace,
                    message,
                    selected,
                    runner_up[0],
                )
                if judged:
                    selected = next(
                        candidate
                        for candidate in candidates
                        if candidate.name == judged
                    )

            output = dict(selected.result)
            output["response"] = selected.response
            metadata = {
                "mode": _mode(),
                "profile": dataclasses.asdict(profile),
                "selected": selected.name,
                "candidate_count": len(candidates),
                "candidates": [
                    {
                        "name": candidate.name,
                        "score": candidate.score,
                        "approved": candidate.audit.approved,
                        "issues": [
                            issue.code for issue in candidate.audit.issues
                        ],
                    }
                    for candidate in candidates
                ],
            }
            output["frontier_v21"] = metadata
            _LAST_STATUS = {
                "installed": True,
                "mode": _mode(),
                "last_request": metadata,
            }
            return output
        finally:
            _ACTIVE.reset(token)

    pipeline_sync_v21.__name__ = "pipeline_sync"
    pipeline_sync_v21._frontier_v21_wrapped = True
    namespace["pipeline_sync"] = pipeline_sync_v21
    namespace["FRONTIER_RUNTIME_V21_INSTALLED"] = True
    _INSTALLED = True
    _LAST_STATUS = {
        "installed": True,
        "mode": _mode(),
        "last_request": None,
    }

# BEGIN FRONTIER CODER COST GUARD V27
_FRONTIER_V27_ORIGINAL_SHOULD_DEEPEN = _should_deepen

def _should_deepen(profile, first):
    if (
        getattr(profile, "skill", "") == "coder"
        and os.getenv("ELITE_FRONTIER_CODER", "1") != "1"
    ):
        return False
    return _FRONTIER_V27_ORIGINAL_SHOULD_DEEPEN(profile, first)
# END FRONTIER CODER COST GUARD V27
