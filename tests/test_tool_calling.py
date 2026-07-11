import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from tool_calling import ToolExecutionConfig, ToolOrchestrator


def call(name, arguments, call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def run(coro):
    return asyncio.run(coro)


def test_invalid_json_is_rejected():
    orchestrator = ToolOrchestrator(
        {"web_search": lambda query: query}
    )
    result = run(
        orchestrator._execute_single(
            call("web_search", "{broken")
        )
    )
    assert result.success is False
    assert "Invalid tool arguments" in result.error


def test_unknown_arguments_are_rejected():
    orchestrator = ToolOrchestrator(
        {"web_search": lambda query: query}
    )
    result = run(
        orchestrator._execute_single(
            call(
                "web_search",
                '{"query":"latest AI","extra":true}',
            )
        )
    )
    assert result.success is False
    assert "Unknown arguments" in result.error


def test_sync_tool_is_supported():
    orchestrator = ToolOrchestrator(
        {
            "web_search": lambda query: {
                "query": query,
                "ok": True,
            }
        }
    )
    result = run(
        orchestrator._execute_single(
            call(
                "web_search",
                '{"query":"latest AI"}',
            )
        )
    )
    assert result.success is True
    assert '"ok": true' in result.result


def test_async_tool_is_supported():
    async def retrieve_memory(query):
        return f"memory for {query}"

    orchestrator = ToolOrchestrator(
        {"retrieve_memory": retrieve_memory}
    )
    result = run(
        orchestrator._execute_single(
            call(
                "retrieve_memory",
                '{"query":"old project"}',
            )
        )
    )
    assert result.success is True
    assert result.result == "memory for old project"


def test_transient_failure_is_retried():
    attempts = {"count": 0}

    async def flaky(query):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionError("temporary")
        return "grounded result with enough detail"

    config = ToolExecutionConfig(
        max_retries=2,
        backoff_base=0,
    )
    orchestrator = ToolOrchestrator(
        {"web_search": flaky},
        config,
    )
    result = run(
        orchestrator._execute_single(
            call(
                "web_search",
                '{"query":"latest AI"}',
            )
        )
    )
    assert result.success is True
    assert result.attempt == 2


def test_permanent_value_error_is_not_retried():
    attempts = {"count": 0}

    def bad(query):
        attempts["count"] += 1
        raise ValueError("bad request")

    config = ToolExecutionConfig(
        max_retries=3,
        backoff_base=0,
    )
    orchestrator = ToolOrchestrator(
        {"web_search": bad},
        config,
    )
    result = run(
        orchestrator._execute_single(
            call(
                "web_search",
                '{"query":"latest AI"}',
            )
        )
    )
    assert result.success is False
    assert attempts["count"] == 1


def test_timeout_is_reported():
    async def slow(query):
        await asyncio.sleep(0.05)
        return query

    config = ToolExecutionConfig(
        max_retries=1,
        per_call_timeout=0.01,
        backoff_base=0,
    )
    orchestrator = ToolOrchestrator(
        {"web_search": slow},
        config,
    )
    result = run(
        orchestrator._execute_single(
            call(
                "web_search",
                '{"query":"latest AI"}',
            )
        )
    )
    assert result.success is False
    assert "timed out" in result.error


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "while True: pass",
        "print((1).__class__)",
        "print(2 ** 1000001)",
        'print("x" * 1000001)',
    ],
)
def test_python_preflight_blocks_dangerous_snippets(code):
    orchestrator = ToolOrchestrator(
        {"execute_python": lambda code: code}
    )
    result = run(
        orchestrator._execute_single(
            call(
                "execute_python",
                json.dumps({"code": code}),
            )
        )
    )
    assert result.success is False
    assert "Pre-flight validation failed" in result.error


def test_run_returns_plain_assistant_content():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="final answer",
                    tool_calls=[],
                )
            )
        ]
    )

    class Chat:
        async def complete_async(self, **kwargs):
            return response

    client = SimpleNamespace(chat=Chat())
    orchestrator = ToolOrchestrator({})
    output = run(
        orchestrator.run(
            [{"role": "user", "content": "hello"}],
            client,
        )
    )
    assert output == "final answer"


def test_custom_tool_schema_is_supported():
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    orchestrator = ToolOrchestrator(
        {"add": lambda a, b: a + b},
        tool_definitions=definitions,
    )
    result = run(
        orchestrator._execute_single(
            call("add", '{"a":2,"b":3}')
        )
    )
    assert result.success is True
    assert result.result == "5"


def test_circuit_breaker_recovers_after_cooldown():
    def fail(query):
        raise ConnectionError("offline")

    config = ToolExecutionConfig(
        max_retries=1,
        backoff_base=0,
        failure_threshold=1,
        circuit_reset_seconds=0.1,
    )
    orchestrator = ToolOrchestrator(
        {"web_search": fail},
        config,
    )

    first = run(
        orchestrator._execute_single(
            call(
                "web_search",
                '{"query":"latest AI"}',
            )
        )
    )
    assert first.success is False
    assert orchestrator._circuit_is_open(
        "web_search"
    ) is True

    orchestrator._circuit_opened_at["web_search"] = (
        time.monotonic() - 1
    )
    assert orchestrator._circuit_is_open(
        "web_search"
    ) is False


def test_parallel_tool_calls_preserve_input_order():
    async def echo(query):
        if query == "first":
            await asyncio.sleep(0.02)
        return query

    orchestrator = ToolOrchestrator(
        {"web_search": echo},
        ToolExecutionConfig(
            max_concurrency=2,
            max_retries=1,
        ),
    )
    results = run(
        orchestrator._execute_tool_calls(
            [
                call(
                    "web_search",
                    '{"query":"first"}',
                    "one",
                ),
                call(
                    "web_search",
                    '{"query":"second"}',
                    "two",
                ),
            ]
        )
    )
    assert [result.tool_call_id for result in results] == [
        "one",
        "two",
    ]

