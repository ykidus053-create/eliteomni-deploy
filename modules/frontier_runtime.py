"""Evaluation-driven serving wrapper for hard EliteOmni requests."""
from __future__ import annotations

import contextvars
import dataclasses
import os
import re
from typing import Any, Callable, Mapping

from modules.frontier_provider import frontier_enabled, frontier_generate
from modules.quality_kernel import (
    AnswerAudit,
    TaskProfile,
    analyze_request,
    audit_answer,
    finalize_response,
)


@dataclasses.dataclass(frozen=True)
class Candidate:
    name: str
    response: str
    result: dict[str, Any]
    audit: AnswerAudit
    score: float


_ACTIVE = contextvars.ContextVar("elite_frontier_v20_active", default=False)
_INSTALLED = False


def _mode() -> str:
    value = os.getenv("ELITE_FRONTIER_MODE", "balanced").strip().lower()
    return value if value in {"off", "balanced", "aggressive"} else "balanced"


def _urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)\]>]+", text or "")


def _source_domains(text: str) -> set[str]:
    domains = set()
    for url in _urls(text):
        match = re.match(r"https?://([^/]+)", url)
        if match:
            domains.add(match.group(1).lower().removeprefix("www."))
    return domains


def _candidate_score(
    message: str,
    response: str,
    profile: TaskProfile,
    audit: AnswerAudit,
    result: Mapping[str, Any] | None = None,
) -> float:
    score = 0.0
    metadata = dict(result or {})
    quality = metadata.get("quality_v18") or {}

    if audit.approved and quality.get("approved", True):
        score += 100.0
    else:
        score -= 120.0

    if response.strip():
        score += min(12.0, len(response) / 1200.0)
    if "I withheld the generated answer" in response:
        score -= 100.0
    if response.count("```") % 2:
        score -= 35.0

    if profile.skill == "coder":
        artifact = audit.artifact
        if artifact is not None:
            if artifact.syntax_ok:
                score += 18.0
            if artifact.tests_found:
                score += 14.0
            if artifact.tests_passed:
                score += 48.0
            elif artifact.tests_found:
                score -= 35.0
        if re.search(
            r"(?im)^\s*(?:TODO|FIXME|pass|\.\.\.|"
            r"raise\s+NotImplementedError)\s*$",
            response,
        ):
            score -= 30.0
        if "```" in response:
            score += 8.0
        if profile.needs_code_execution and not artifact:
            score -= 18.0

    elif profile.skill == "researcher":
        urls = _urls(response)
        domains = _source_domains(response)
        score += min(36.0, len(urls) * 7.0)
        score += min(18.0, len(domains) * 6.0)
        if profile.requires_sources and not urls:
            score -= 60.0
        if re.search(
            r"\b(inference|I infer|appears|uncertain|evidence is limited)\b",
            response,
            re.IGNORECASE,
        ):
            score += 5.0

    elif profile.skill == "calculator":
        if re.search(r"\d", response):
            score += 10.0
        if re.search(r"=|equation|formula|therefore", response, re.I):
            score += 6.0

    return round(score, 3)


def _should_escalate(
    profile: TaskProfile,
    first: Candidate,
) -> bool:
    mode = _mode()
    if mode == "off":
        return False
    quality = first.result.get("quality_v18") or {}
    if (
        not first.audit.approved
        or quality.get("approved") is False
        or "I withheld the generated answer" in first.response
    ):
        return True
    if profile.high_stakes or profile.production_claim:
        return True
    if profile.complexity == "hard" and profile.skill in {
        "coder",
        "researcher",
        "calculator",
    }:
        return True
    if mode == "aggressive":
        if profile.complexity == "hard":
            return True
        if profile.complexity == "medium" and profile.skill in {
            "coder",
            "researcher",
        }:
            return True
    return False


def _retry_contract(
    original_message: str,
    profile: TaskProfile,
    first: Candidate,
) -> str:
    issues = [issue.code for issue in first.audit.issues]
    artifact = first.audit.artifact
    evidence = artifact.summary if artifact is not None else "none"
    return (
        original_message
        + "\n\n[FRONTIER V20 INDEPENDENT SECOND PASS]\n"
        + "Solve the original request again from first principles. Do not "
        + "refer to this instruction in the answer. Improve correctness rather "
        + "than merely expanding the response.\n"
        + f"Task type: {profile.skill}; complexity: {profile.complexity}.\n"
        + f"Prior verification issues: {issues or ['none']}.\n"
        + f"Prior execution evidence: {evidence}.\n"
        + (
            "For code: inspect interfaces, return complete runnable artifacts, "
            "add focused tests, and verify failure paths.\n"
            if profile.skill == "coder"
            else ""
        )
        + (
            "For research: use multiple independent current sources, attach "
            "URLs near material claims, distinguish fact from inference, and "
            "resolve disagreements.\n"
            if profile.skill == "researcher"
            else ""
        )
        + "Return only the improved final answer."
    )


def _frontier_review(
    message: str,
    profile: TaskProfile,
    first: Candidate,
) -> str:
    issues = "\n".join(
        f"- {issue.code}: {issue.message}" for issue in first.audit.issues
    ) or "- No hard verification failure; improve depth and correctness."
    baseline = first.response
    if len(baseline) > 24000:
        baseline = baseline[:14000] + "\n...[clipped]...\n" + baseline[-8000:]

    system = (
        "You are the final expert reviewer for a high-stakes AI system. "
        "Return only a corrected, self-contained final answer to the original "
        "user. Preserve useful code and valid citations, but repair omissions, "
        "logic errors, unsupported claims, and incomplete tests. Do not claim "
        "execution unless evidence is present. Do not mention reviewing another "
        "answer."
    )
    user = (
        f"ORIGINAL REQUEST:\n{message}\n\n"
        f"TASK PROFILE:\n{profile}\n\n"
        f"VERIFICATION FEEDBACK:\n{issues}\n\n"
        f"BASELINE ANSWER:\n{baseline}\n\n"
        "Produce the improved final answer."
    )
    return frontier_generate(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )


def _as_candidate(
    name: str,
    message: str,
    profile: TaskProfile,
    result: Any,
) -> Candidate:
    if isinstance(result, Mapping):
        data = dict(result)
        response = str(data.get("response", ""))
    else:
        response = str(result)
        data = {"response": response, "skill": profile.skill}

    audit = audit_answer(message, response, profile)
    score = _candidate_score(message, response, profile, audit, data)
    return Candidate(name, response, data, audit, score)


def _judge_close_candidates(
    namespace: dict[str, Any],
    message: str,
    left: Candidate,
    right: Candidate,
) -> str | None:
    if abs(left.score - right.score) > 12:
        return None
    judge = namespace.get("mistral_generate")
    if not callable(judge):
        return None

    a = left.response[:9000]
    b = right.response[:9000]
    prompt = [
        {
            "role": "system",
            "content": (
                "Choose the more correct answer. Prioritize factual and logical "
                "correctness, executable code, explicit assumptions, source "
                "quality, and complete satisfaction of the request. Reply with "
                "exactly A or B."
            ),
        },
        {
            "role": "user",
            "content": (
                f"REQUEST:\n{message[:5000]}\n\n"
                f"ANSWER A:\n{a}\n\nANSWER B:\n{b}"
            ),
        },
    ]
    try:
        verdict = str(judge(prompt, max_tokens=8)).strip().upper()
    except Exception:
        return None
    if verdict.startswith("A"):
        return left.name
    if verdict.startswith("B"):
        return right.name
    return None


def install_frontier_runtime(namespace: dict[str, Any]) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    original = namespace.get("pipeline_sync")
    if not callable(original):
        return

    def pipeline_sync_v20(message: str, history: list) -> dict[str, Any]:
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

            first_raw = original(message, history)
            first = _as_candidate(
                "baseline",
                message,
                profile,
                first_raw,
            )
            candidates = [first]
            provider_used = False

            if _should_escalate(profile, first):
                if frontier_enabled():
                    try:
                        improved = _frontier_review(message, profile, first)
                        final, _ = finalize_response(
                            message,
                            improved,
                            profile,
                        )
                        candidates.append(
                            _as_candidate(
                                "frontier-provider",
                                message,
                                profile,
                                {
                                    **first.result,
                                    "response": final,
                                    "mode": "frontier-provider-v20",
                                },
                            )
                        )
                        provider_used = True
                    except Exception as exc:
                        print(f"[FrontierV20] provider review failed: {exc}")

                if len(candidates) == 1:
                    retry_message = _retry_contract(
                        message,
                        profile,
                        first,
                    )
                    try:
                        retry_raw = original(retry_message, history)
                        candidates.append(
                            _as_candidate(
                                "verified-retry",
                                message,
                                profile,
                                retry_raw,
                            )
                        )
                    except Exception as exc:
                        print(f"[FrontierV20] local retry failed: {exc}")

            selected = max(candidates, key=lambda candidate: candidate.score)
            if len(candidates) == 2:
                judge_choice = _judge_close_candidates(
                    namespace,
                    message,
                    candidates[0],
                    candidates[1],
                )
                if judge_choice:
                    selected = next(
                        candidate
                        for candidate in candidates
                        if candidate.name == judge_choice
                    )

            output = dict(selected.result)
            output["response"] = selected.response
            output["frontier_v20"] = {
                "mode": _mode(),
                "profile": dataclasses.asdict(profile),
                "provider_used": provider_used,
                "selected": selected.name,
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
            return output
        finally:
            _ACTIVE.reset(token)

    pipeline_sync_v20.__name__ = "pipeline_sync"
    pipeline_sync_v20._frontier_v20_wrapped = True
    namespace["pipeline_sync"] = pipeline_sync_v20
    namespace["FRONTIER_RUNTIME_V20_INSTALLED"] = True
    _INSTALLED = True
