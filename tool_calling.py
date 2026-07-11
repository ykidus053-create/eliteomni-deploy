"""Reliable tool-calling orchestration for EliteOmni.

This module validates model-generated tool calls, supports synchronous and
asynchronous tools, bounds execution, retries transient failures, and preserves
structured execution logs for debugging.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

log = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use for recent events, "
                "current prices, weather, and news. Do not use for stable facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query, 3-8 words",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this search is needed",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute a small, bounded Python calculation. The code must "
                "print its result and cannot import modules or access files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Small Python calculation that prints results",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "What this code computes",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": (
                "Retrieve relevant information from conversation memory when "
                "the user refers to prior conversations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for in memory",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class ToolExecutionResult:
    tool_name: str
    tool_call_id: str
    result: str
    latency_ms: int
    success: bool
    error: Optional[str] = None
    confidence: float = 1.0
    attempt: int = 1


@dataclass
class ToolExecutionConfig:
    max_rounds: int = 5
    per_call_timeout: float = 15.0
    max_retries: int = 3
    backoff_base: float = 0.5
    min_confidence_threshold: float = 0.4
    max_concurrency: int = 4
    max_result_chars: int = 4_000
    failure_threshold: int = 3
    circuit_reset_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.max_rounds = max(1, min(int(self.max_rounds), 20))
        self.per_call_timeout = max(
            0.01, min(float(self.per_call_timeout), 300.0)
        )
        self.max_retries = max(1, min(int(self.max_retries), 5))
        self.backoff_base = max(0.0, min(float(self.backoff_base), 30.0))
        self.min_confidence_threshold = max(
            0.0, min(float(self.min_confidence_threshold), 1.0)
        )
        self.max_concurrency = max(1, min(int(self.max_concurrency), 16))
        self.max_result_chars = max(
            200, min(int(self.max_result_chars), 100_000)
        )
        self.failure_threshold = max(
            1, min(int(self.failure_threshold), 20)
        )
        self.circuit_reset_seconds = max(
            0.1, min(float(self.circuit_reset_seconds), 3_600.0)
        )


class ToolOrchestrator:
    MAX_TOOL_ROUNDS = 5

    def __init__(
        self,
        tool_registry: Dict[str, Callable[..., Any]],
        config: Optional[ToolExecutionConfig] = None,
        tool_definitions: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        self.registry = dict(tool_registry)
        self.execution_log: List[ToolExecutionResult] = []
        self.config = config or ToolExecutionConfig()
        self.tool_definitions = [
            dict(item) for item in (tool_definitions or TOOL_DEFINITIONS)
        ]
        self._definitions_by_name = self._index_definitions(
            self.tool_definitions
        )
        self._consecutive_failures: Dict[str, int] = {}
        self._circuit_opened_at: Dict[str, float] = {}

    async def run(
        self,
        messages: List[Dict[str, Any]],
        mistral_client: Any,
        model: str = "mistral-large-latest",
        max_tokens: int = 2_000,
    ) -> str:
        current_messages = [dict(message) for message in messages]
        bounded_max_tokens = max(1, min(int(max_tokens), 100_000))

        for _round_num in range(self.config.max_rounds):
            response = await self._complete(
                mistral_client,
                model=model,
                messages=current_messages,
                tools=self.tool_definitions,
                tool_choice="auto",
                max_tokens=bounded_max_tokens,
            )
            message = self._first_message(response)
            tool_calls = self._get(message, "tool_calls", None) or []
            content = self._get(message, "content", "") or ""

            if not tool_calls:
                return str(content)

            current_messages.append(
                {
                    "role": "assistant",
                    "content": str(content),
                    "tool_calls": [
                        self._serialize_tool_call(tool_call)
                        for tool_call in tool_calls
                    ],
                }
            )

            for result in await self._execute_tool_calls(tool_calls):
                tool_content = (
                    result.result
                    if result.success
                    else f"Tool error: {result.error or 'unknown failure'}"
                )
                if (
                    result.success
                    and result.confidence < self.config.min_confidence_threshold
                ):
                    tool_content += (
                        "\n[WARNING: low confidence "
                        f"({result.confidence:.2f}); re-ground before answering]"
                    )
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": tool_content,
                    }
                )

        final_response = await self._complete(
            mistral_client,
            model=model,
            messages=current_messages,
            max_tokens=bounded_max_tokens,
        )
        final_message = self._first_message(final_response)
        return str(self._get(final_message, "content", "") or "")

    async def _complete(self, client: Any, **kwargs: Any) -> Any:
        chat = getattr(client, "chat", None)
        if chat is None:
            raise TypeError("mistral_client must expose a chat interface")

        async_method = getattr(chat, "complete_async", None)
        if callable(async_method):
            result = async_method(**kwargs)
            return await result if inspect.isawaitable(result) else result

        sync_method = getattr(chat, "complete", None)
        if not callable(sync_method):
            raise TypeError(
                "chat interface must expose complete_async or complete"
            )
        return await asyncio.to_thread(sync_method, **kwargs)

    @staticmethod
    def _get(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)

    def _first_message(self, response: Any) -> Any:
        choices = self._get(response, "choices", None)
        if not choices:
            raise ValueError("model response did not contain choices")
        message = self._get(choices[0], "message", None)
        if message is None:
            raise ValueError(
                "model response choice did not contain a message"
            )
        return message

    @staticmethod
    def _index_definitions(
        definitions: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Mapping[str, Any]]:
        indexed: Dict[str, Mapping[str, Any]] = {}
        for item in definitions:
            function = item.get("function", {})
            name = function.get("name")
            if isinstance(name, str) and name:
                indexed[name] = function
        return indexed

    def _serialize_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        function = self._get(tool_call, "function", {}) or {}
        arguments = self._get(function, "arguments", "{}")
        if isinstance(arguments, Mapping):
            arguments_text = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            arguments_text = str(arguments or "{}")

        return {
            "id": str(self._get(tool_call, "id", "") or ""),
            "type": str(
                self._get(tool_call, "type", "function") or "function"
            ),
            "function": {
                "name": str(self._get(function, "name", "") or ""),
                "arguments": arguments_text,
            },
        }

    async def _execute_tool_calls(
        self, tool_calls: Sequence[Any]
    ) -> List[ToolExecutionResult]:
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        results = await asyncio.gather(
            *(
                self._execute_single(tool_call, semaphore=semaphore)
                for tool_call in tool_calls
            )
        )
        self.execution_log.extend(results)
        return list(results)

    async def _execute_single(
        self,
        tool_call: Any,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> ToolExecutionResult:
        started = time.monotonic()
        function = self._get(tool_call, "function", {}) or {}
        tool_name = str(self._get(function, "name", "") or "")
        tool_call_id = str(
            self._get(tool_call, "id", "") or tool_name or "unknown"
        )

        if not tool_name:
            return self._failure(
                tool_name="unknown",
                tool_call_id=tool_call_id,
                started=started,
                error="Tool call is missing a function name",
            )

        if self._circuit_is_open(tool_name):
            return self._failure(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                started=started,
                error=(
                    "Tool circuit is temporarily open after "
                    f"{self.get_failure_streak(tool_name)} consecutive failures"
                ),
            )

        raw_arguments = self._get(function, "arguments", "{}")
        try:
            if isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)
            else:
                arguments = json.loads(str(raw_arguments or "{}"))
            if not isinstance(arguments, dict):
                raise ValueError(
                    "tool arguments must decode to a JSON object"
                )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return self._failure(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                started=started,
                error=f"Invalid tool arguments: {exc}",
            )

        tool_fn = self.registry.get(tool_name)
        if tool_fn is None:
            return self._failure(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                started=started,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            validated_args = self._validate_args(tool_name, arguments)
        except ValueError as exc:
            return self._failure(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                started=started,
                error=f"Pre-flight validation failed: {exc}",
            )

        active_semaphore = semaphore or asyncio.Semaphore(1)
        last_error = "All attempts failed"

        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with active_semaphore:
                    value = await asyncio.wait_for(
                        self._invoke(tool_fn, validated_args),
                        timeout=self.config.per_call_timeout,
                    )
                result_text = self._format_result(value)
                confidence = self._post_call_verify(
                    tool_name, result_text
                )
                self._consecutive_failures.pop(tool_name, None)
                self._circuit_opened_at.pop(tool_name, None)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    result=result_text,
                    latency_ms=self._latency_ms(started),
                    success=True,
                    confidence=confidence,
                    attempt=attempt,
                )
            except (ValueError, TypeError, KeyError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
            except asyncio.TimeoutError:
                last_error = (
                    "Execution timed out after "
                    f"{self.config.per_call_timeout:.2f}s"
                )
            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )

            log.warning(
                "[tool] %s attempt %d/%d failed: %s",
                tool_name,
                attempt,
                self.config.max_retries,
                last_error,
            )
            if (
                attempt < self.config.max_retries
                and self.config.backoff_base
            ):
                await asyncio.sleep(
                    self.config.backoff_base * (2 ** (attempt - 1))
                )

        streak = self._consecutive_failures.get(tool_name, 0) + 1
        self._consecutive_failures[tool_name] = streak
        if streak >= self.config.failure_threshold:
            self._circuit_opened_at[tool_name] = time.monotonic()

        return self._failure(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            started=started,
            error=last_error,
            attempt=self.config.max_retries,
        )

    async def _invoke(
        self,
        tool_fn: Callable[..., Any],
        arguments: Dict[str, Any],
    ) -> Any:
        if inspect.iscoroutinefunction(tool_fn):
            return await tool_fn(**arguments)

        value = await asyncio.to_thread(tool_fn, **arguments)
        if inspect.isawaitable(value):
            return await value
        return value

    def _failure(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        started: float,
        error: str,
        attempt: int = 1,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result="",
            latency_ms=self._latency_ms(started),
            success=False,
            error=error,
            attempt=attempt,
        )

    @staticmethod
    def _latency_ms(started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1_000))

    def _format_result(self, value: Any) -> str:
        if (
            isinstance(value, (dict, list, tuple, bool, int, float))
            or value is None
        ):
            try:
                text = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            except (TypeError, ValueError):
                text = str(value)
        else:
            text = str(value)

        if len(text) > self.config.max_result_chars:
            text = (
                text[: self.config.max_result_chars]
                + "\n...[truncated]"
            )
        return text

    def _post_call_verify(self, tool_name: str, result: str) -> float:
        del tool_name
        normalized = result.strip().lower()
        if not normalized:
            return 0.0
        if normalized in {"none", "null", "[]", "{}"}:
            return 0.1
        if normalized.startswith(
            ("error", "tool error", "traceback")
        ):
            return 0.2
        if len(normalized) < 20:
            return 0.5
        return 1.0

    def _validate_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        definition = self._definitions_by_name.get(tool_name)
        if definition is None:
            raise ValueError(
                f"No schema registered for tool: {tool_name}"
            )

        parameters = definition.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        unknown = sorted(set(args) - set(properties))
        if (
            unknown
            and parameters.get("additionalProperties") is False
        ):
            raise ValueError(
                f"Unknown arguments: {', '.join(unknown)}"
            )

        for key in required:
            if key not in args or args[key] in (None, ""):
                raise ValueError(
                    f"Missing required argument: {key}"
                )

        validated: Dict[str, Any] = {}
        for key, value in args.items():
            expected_type = properties.get(key, {}).get("type")
            self._validate_json_type(
                key,
                value,
                expected_type,
            )
            if isinstance(value, str):
                value = value.strip()
                if len(value) > 10_000:
                    raise ValueError(
                        f"Argument '{key}' is too long"
                    )
            validated[key] = value

        if tool_name == "web_search":
            query = validated["query"][:200].strip()
            if len(query) < 2:
                raise ValueError("Search query is too short")
            validated["query"] = query
            if "reason" in validated:
                validated["reason"] = (
                    validated["reason"][:500]
                )

        if tool_name == "retrieve_memory":
            validated["query"] = (
                validated["query"][:500].strip()
            )

        if tool_name == "execute_python":
            code = validated["code"]
            if len(code) > 4_000:
                raise ValueError(
                    "Python snippet exceeds 4,000 characters"
                )
            self._validate_python(code)
            if "purpose" in validated:
                validated["purpose"] = (
                    validated["purpose"][:500]
                )

        return validated

    @staticmethod
    def _validate_json_type(
        key: str,
        value: Any,
        expected_type: Optional[str],
    ) -> None:
        if expected_type is None:
            return

        valid = {
            "string": isinstance(value, str),
            "integer": (
                isinstance(value, int)
                and not isinstance(value, bool)
            ),
            "number": (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ),
            "boolean": isinstance(value, bool),
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
        }.get(expected_type, True)

        if not valid:
            raise ValueError(
                f"Argument '{key}' must be of type {expected_type}"
            )

    @staticmethod
    def _validate_python(code: str) -> None:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise ValueError(
                f"Syntax error in code: {exc.msg}"
            ) from exc

        nodes = list(ast.walk(tree))
        if len(nodes) > 300:
            raise ValueError("Python snippet is too complex")

        blocked_nodes = (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Lambda,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.Try,
            ast.Raise,
            ast.Delete,
            ast.Global,
            ast.Nonlocal,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
        )
        blocked_names = {
            "__import__",
            "breakpoint",
            "compile",
            "delattr",
            "dir",
            "eval",
            "exec",
            "exit",
            "getattr",
            "globals",
            "help",
            "input",
            "locals",
            "memoryview",
            "object",
            "open",
            "quit",
            "setattr",
            "super",
            "type",
            "vars",
        }

        for node in nodes:
            if isinstance(node, blocked_nodes):
                raise ValueError(
                    "Python construct "
                    f"'{type(node).__name__}' is blocked"
                )
            if (
                isinstance(node, ast.Name)
                and node.id in blocked_names
            ):
                raise ValueError(
                    f"Name '{node.id}' is blocked"
                )
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
            ):
                raise ValueError(
                    f"Private attribute '{node.attr}' is blocked"
                )
            if isinstance(node, ast.Constant):
                if (
                    isinstance(node.value, str)
                    and len(node.value) > 10_000
                ):
                    raise ValueError(
                        "String literal is too large"
                    )
                if (
                    isinstance(node.value, int)
                    and abs(node.value) > 10**12
                ):
                    raise ValueError(
                        "Integer literal is too large"
                    )
            if isinstance(node, ast.BinOp):
                ToolOrchestrator._validate_expensive_binop(
                    node
                )

    @staticmethod
    def _validate_expensive_binop(node: ast.BinOp) -> None:
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Constant)
        ):
            exponent = node.right.value
            if (
                isinstance(exponent, (int, float))
                and abs(exponent) > 10_000
            ):
                raise ValueError("Exponent is too large")

        if isinstance(node.op, ast.Mult):
            constants = [
                side.value
                for side in (node.left, node.right)
                if isinstance(side, ast.Constant)
            ]
            if any(
                isinstance(value, int)
                and abs(value) > 1_000_000
                for value in constants
            ):
                raise ValueError(
                    "Sequence repetition is too large"
                )

    def _circuit_is_open(self, tool_name: str) -> bool:
        streak = self.get_failure_streak(tool_name)
        opened_at = self._circuit_opened_at.get(tool_name)
        if (
            streak < self.config.failure_threshold
            or opened_at is None
        ):
            return False

        if (
            time.monotonic() - opened_at
            >= self.config.circuit_reset_seconds
        ):
            self._consecutive_failures.pop(tool_name, None)
            self._circuit_opened_at.pop(tool_name, None)
            return False
        return True

    def get_failure_streak(self, tool_name: str) -> int:
        return self._consecutive_failures.get(tool_name, 0)

