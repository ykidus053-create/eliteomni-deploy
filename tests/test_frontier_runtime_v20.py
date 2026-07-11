from modules.frontier_runtime import _candidate_score, _should_escalate
from modules.quality_kernel import analyze_request, audit_answer


def _candidate(message, response):
    profile = analyze_request(message)
    audit = audit_answer(message, response, profile)
    from modules.frontier_runtime import Candidate
    score = _candidate_score(message, response, profile, audit, {})
    return profile, Candidate("x", response, {}, audit, score)


def test_failed_code_candidate_escalates():
    profile, candidate = _candidate(
        "Build production-grade Python code with tests",
        "I withheld the generated answer because it failed final verification.",
    )
    assert _should_escalate(profile, candidate)


def test_research_sources_raise_score():
    message = "Research the latest benchmark with current sources"
    profile = analyze_request(message)
    weak = "There is a benchmark."
    strong = (
        "A current benchmark exists. https://example.org/a "
        "Another source: https://example.net/b"
    )
    weak_audit = audit_answer(message, weak, profile)
    strong_audit = audit_answer(message, strong, profile)
    assert _candidate_score(
        message, strong, profile, strong_audit, {}
    ) > _candidate_score(message, weak, profile, weak_audit, {})
