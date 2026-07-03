"""
eval_rag.py — repeatable retrieval quality check against real content.

Usage:
    python3 eval_rag.py

Ingests a fixed set of real files from ~/knowledge_base, runs a fixed set
of real questions through retrieve(), and scores each result against
required keywords a good answer should contain. Run this after any change
to chunking, retrieval, or reranking to see whether it actually helped —
not just whether the code compiles.

This is intentionally simple (keyword presence, not semantic grading) so
it's fast and has zero extra dependencies. It catches regressions and
obvious wins; it won't catch subtle relevance differences. Treat a passing
score as "didn't break," not as a guarantee of quality.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.knowledge_rag import ingest, retrieve

KB_DIR = os.path.expanduser("~/knowledge_base")

# (filename, source label) — small/medium files chosen so ingestion is fast
# and topics are distinct enough to catch retrieval crossing into the
# wrong document.
FILES_TO_INGEST = [
    ("system_design.txt", "system_design"),
    ("security.txt", "security"),
    ("cs_fundamentals.txt", "cs_fundamentals"),
]

# Each case: a real question + keywords a genuinely relevant top result
# should contain (case-insensitive substring match, OR'd within a group).
# Keep these honest — write them before looking at retrieval results, not
# after, or you'll unconsciously tune the keywords to whatever comes back.
EVAL_CASES = [
    {
        "query": "what is a load balancer used for",
        "expect_any": ["load balanc", "distribut", "traffic"],
        "topic": "system_design",
    },
    {
        "query": "difference between horizontal and vertical scaling",
        "expect_any": ["horizontal", "vertical", "scal"],
        "topic": "system_design",
    },
    {
        "query": "what is a cdn and why use one",
        "expect_any": ["cdn", "content delivery", "cache", "edge"],
        "topic": "system_design",
    },
    {
        "query": "what is sql injection",
        "expect_any": ["sql injection", "injection", "sanitiz", "parameteriz"],
        "topic": "security",
    },
    {
        "query": "what is cross site scripting xss",
        "expect_any": ["xss", "cross-site scripting", "cross site scripting"],
        "topic": "security",
    },
    {
        "query": "how does public key cryptography work",
        "expect_any": ["public key", "private key", "asymmetric"],
        "topic": "security",
    },
    {
        "query": "what is big o notation",
        "expect_any": ["big o", "o(n)", "time complexity", "asymptotic"],
        "topic": "cs_fundamentals",
    },
    {
        "query": "explain how a hash table works",
        "expect_any": ["hash table", "hash function", "bucket", "collision"],
        "topic": "cs_fundamentals",
    },
]


async def setup():
    print(f"Ingesting {len(FILES_TO_INGEST)} files from {KB_DIR}...\n")
    for filename, source in FILES_TO_INGEST:
        path = os.path.join(KB_DIR, filename)
        if not os.path.exists(path):
            print(f"  SKIP {filename}: not found at {path}")
            continue
        with open(path, "r", errors="ignore") as f:
            text = f.read()
        start = time.monotonic()
        n = await ingest(text, source=source)
        elapsed = time.monotonic() - start
        print(f"  {filename}: {n} new chunks ingested ({elapsed:.1f}s)")
    print()


async def run_eval():
    results = []
    for case in EVAL_CASES:
        retrieved = await retrieve(case["query"], top_k=3)
        combined = " ".join(retrieved).lower()
        hit = any(kw.lower() in combined for kw in case["expect_any"])
        results.append({
            "query": case["query"],
            "topic": case["topic"],
            "pass": hit,
            "top_result_preview": retrieved[0][:200] if retrieved else "(no results)",
        })
    return results


def print_report(results):
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"{'='*70}")
    print(f"RESULTS: {passed}/{total} passed")
    print(f"{'='*70}\n")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] ({r['topic']}) {r['query']}")
        if not r["pass"]:
            print(f"       top result was: {r['top_result_preview']}")
        print()


async def main():
    await setup()
    results = await run_eval()
    print_report(results)
    failed = [r for r in results if not r["pass"]]
    if failed:
        print(f"{len(failed)} case(s) failed. This is signal, not necessarily a bug — ")
        print("check whether the keywords were too strict or retrieval is genuinely missing content.")
        sys.exit(1)
    else:
        print("All eval cases passed.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
