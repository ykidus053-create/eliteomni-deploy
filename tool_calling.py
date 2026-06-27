import json, ast, asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import time

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

class ToolOrchestrator:
    MAX_TOOL_ROUNDS = 3

    def __init__(self, tool_registry: Dict[str, Callable]):
        self.registry = tool_registry
        self.execution_log: List[ToolExecutionResult] = []

    async def run(self, messages: List[Dict], mistral_client, model: str = "mistral-large-latest", max_tokens: int = 2000) -> str:
        current_messages = list(messages)

        for round_num in range(self.MAX_TOOL_ROUNDS):
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
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.result if result.success else f"Tool error: {result.error}"
                })

        final_response = await mistral_client.chat.complete_async(model=model, messages=current_messages, max_tokens=max_tokens)
        return final_response.choices[0].message.content or ""

    async def _execute_tool_calls(self, tool_calls) -> List[ToolExecutionResult]:
        results = []
        for tc in tool_calls:
            t0 = time.time()
            tool_name = tc.function.name

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            tool_fn = self.registry.get(tool_name)

            if not tool_fn:
                results.append(ToolExecutionResult(tool_name, tc.id, "", 0, False, f"Unknown tool: {tool_name}"))
                continue

            try:
                validated_args = self._validate_args(tool_name, args)
                # Upgraded: Added strict 10-second timeout to prevent hanging on bad scripts
                result = await asyncio.wait_for(tool_fn(**validated_args), timeout=10.0)
                
                results.append(ToolExecutionResult(
                    tool_name=tool_name, tool_call_id=tc.id, result=str(result)[:4000],
                    latency_ms=int((time.time() - t0) * 1000), success=True
                ))
            except asyncio.TimeoutError:
                results.append(ToolExecutionResult(
                    tool_name=tool_name, tool_call_id=tc.id, result="", 
                    latency_ms=int((time.time() - t0) * 1000), success=False, error="Execution timed out (10s)"
                ))
            except Exception as e:
                results.append(ToolExecutionResult(
                    tool_name=tool_name, tool_call_id=tc.id, result="",
                    latency_ms=int((time.time() - t0) * 1000), success=False, error=str(e)[:200]
                ))

        self.execution_log.extend(results)
        return results

    def _validate_args(self, tool_name: str, args: Dict) -> Dict:
        """Upgraded: Uses AST parsing for robust Python sandboxing."""
        if tool_name == "execute_python":
            code = args.get("code", "")
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    # Block imports and attribute access to dunder methods
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
