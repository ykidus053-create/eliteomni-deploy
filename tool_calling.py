import json, ast, asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import time
import logging

log = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for: recent events, current prices, weather, news. Do NOT use for stable facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific search query, 3-8 words"},
                    "reason": {"type": "string", "description": "Why this search is needed"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code for calculations, data processing, or verification. Output is captured and returned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Must print results."},
                    "purpose": {"type": "string", "description": "What this code computes"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": "Retrieve relevant information from conversation memory. Use when the user references past conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for in memory"}
                },
                "required": ["query"]
            }
        }
    }
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

class ToolOrchestrator:
    MAX_TOOL_ROUNDS = 5

    def __init__(self, tool_registry: Dict[str, Callable], config: Optional[ToolExecutionConfig] = None):
        self.registry = tool_registry
        self.execution_log: List[ToolExecutionResult] = []
        self.config = config or ToolExecutionConfig()
        self._consecutive_failures: Dict[str, int] = {}

    async def run(self, messages: List[Dict], mistral_client, model: str = "mistral-large-latest", max_tokens: int = 2000) -> str:
        current_messages = list(messages)

        for round_num in range(self.config.max_rounds):
            response = await mistral_client.chat.complete_async(
                model=model, messages=current_messages, tools=TOOL_DEFINITIONS, tool_choice="auto", max_tokens=max_tokens
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            current_messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            })

            tool_results = await self._execute_tool_calls(message.tool_calls)

            for result in tool_results:
                content = result.result if result.success else f"Tool error: {result.error}"
                if result.success and result.confidence < self.config.min_confidence_threshold:
                    content += f"\n[WARNING: low confidence ({result.confidence:.2f}) — consider re-grounding]"
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": content
                })

        final_response = await mistral_client.chat.complete_async(model=model, messages=current_messages, max_tokens=max_tokens)
        return final_response.choices[0].message.content or ""

    async def _execute_tool_calls(self, tool_calls) -> List[ToolExecutionResult]:
        results = []
        for tc in tool_calls:
            result = await self._execute_single(tc)
            results.append(result)
        self.execution_log.extend(results)
        return results

    async def _execute_single(self, tc) -> ToolExecutionResult:
        """Upgraded: pre-flight validation, retry with backoff, post-call verification."""
        t0 = time.time()
        tool_name = tc.function.name

        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}

        tool_fn = self.registry.get(tool_name)
        if not tool_fn:
            return ToolExecutionResult(tool_name, tc.id, "", 0, False, f"Unknown tool: {tool_name}")

        try:
            validated_args = self._validate_args(tool_name, args)
        except ValueError as e:
            return ToolExecutionResult(tool_name, tc.id, "", 0, False, f"Pre-flight validation failed: {e}")

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                result = await asyncio.wait_for(tool_fn(**validated_args), timeout=self.config.per_call_timeout)
                result_str = str(result)
                confidence = self._post_call_verify(tool_name, result_str)
                self._consecutive_failures[tool_name] = 0
                return ToolExecutionResult(
                    tool_name=tool_name, tool_call_id=tc.id,
                    result=result_str[:4000],
                    latency_ms=int((time.time() - t0) * 1000),
                    success=True, confidence=confidence, attempt=attempt
                )
            except asyncio.TimeoutError:
                last_error = f"Execution timed out ({self.config.per_call_timeout}s)"
                log.warning("[tool] %s attempt %d timed out", tool_name, attempt)
            except Exception as e:
                last_error = str(e)[:200]
                log.warning("[tool] %s attempt %d failed: %s", tool_name, attempt, e)

            if attempt < self.config.max_retries:
                backoff = self.config.backoff_base * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

        self._consecutive_failures[tool_name] = self._consecutive_failures.get(tool_name, 0) + 1
        return ToolExecutionResult(
            tool_name=tool_name, tool_call_id=tc.id, result="",
            latency_ms=int((time.time() - t0) * 1000), success=False,
            error=last_error or "All retries exhausted", attempt=self.config.max_retries
        )

    def _post_call_verify(self, tool_name: str, result: str) -> float:
        """NEW: Verify tool result quality. Returns confidence 0.0-1.0."""
        if not result or result.strip() == "":
            return 0.0
        if result.strip().lower() in ("none", "null", "[]", "{}"):
            return 0.1
        if "error" in result.lower()[:50]:
            return 0.3
        if len(result) < 20:
            return 0.5
        return 1.0

    def _validate_args(self, tool_name: str, args: Dict) -> Dict:
        """Upgraded: AST sandboxing + required-arg presence check."""
        for td in TOOL_DEFINITIONS:
            if td["function"]["name"] == tool_name:
                required = td["function"]["parameters"].get("required", [])
                for req in required:
                    if req not in args or not args[req]:
                        raise ValueError(f"Missing required argument: {req}")
                break

        if tool_name == "execute_python":
            code = args.get("code", "")
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        raise ValueError("Imports are blocked in the sandbox.")
                    if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
                        raise ValueError(f"Access to '{node.attr}' is blocked.")
                    if isinstance(node, ast.Call):
                        func = node.func
                        if isinstance(func, ast.Name) and func.id in ['exec', 'eval', 'compile', 'open', 'input']:
                            raise ValueError(f"Call to '{func.id}' is blocked.")
            except SyntaxError as e:
                raise ValueError(f"Syntax error in code: {e}")

        if tool_name == "web_search":
            args["query"] = args.get("query", "")[:200].strip()

        return args

    def get_failure_streak(self, tool_name: str) -> int:
        """NEW: Returns consecutive failure count. Used by compounding-error guard."""
        return self._consecutive_failures.get(tool_name, 0)
