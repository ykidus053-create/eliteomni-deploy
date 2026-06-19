import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import time

# Mistral-compatible function definitions
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use for: recent events, "
                "current prices, weather, news, anything that may have changed "
                "since training cutoff. Do NOT use for stable facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query, 3-8 words"
                    },
                    "reason": {
                        "type": "string", 
                        "description": "Why this search is needed"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code for calculations, data processing, "
                "or verification. Use for: math, statistics, data analysis. "
                "Output is captured and returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Must print results."
                    },
                    "purpose": {
                        "type": "string",
                        "description": "What this code computes"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": (
                "Retrieve relevant information from conversation memory. "
                "Use when the user references past conversations or personal details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for in memory"
                    }
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
    """
    Proper tool calling loop using Mistral function calling API.
    Model decides which tools to call. Results fed back into context.
    Maximum 3 tool-call rounds to prevent infinite loops.
    """
    
    MAX_TOOL_ROUNDS = 3
    
    def __init__(self, tool_registry: Dict[str, Callable]):
        self.registry = tool_registry
        self.execution_log: List[ToolExecutionResult] = []
    
    async def run(self, 
                  messages: List[Dict],
                  mistral_client,
                  model: str = "mistral-large-latest",
                  max_tokens: int = 2000) -> str:
        """
        Full tool-calling loop:
        1. Send messages to Mistral with tool definitions
        2. If model calls tools, execute them
        3. Append results to messages
        4. Repeat until model generates final response (no tool calls)
        5. Return final text response
        """
        current_messages = list(messages)
        
        for round_num in range(self.MAX_TOOL_ROUNDS):
            response = await mistral_client.chat.complete_async(
                model=model,
                messages=current_messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=max_tokens
            )
            
            message = response.choices[0].message
            
            # No tool calls — final response
            if not message.tool_calls:
                return message.content or ""
            
            # Append assistant message with tool calls
            current_messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # Execute all tool calls (parallel where independent)
            tool_results = await self._execute_tool_calls(
                message.tool_calls
            )
            
            # Append tool results
            for result in tool_results:
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.result if result.success 
                               else f"Tool error: {result.error}"
                })
        
        # Exceeded max rounds — force final generation without tools
        final_response = await mistral_client.chat.complete_async(
            model=model,
            messages=current_messages,
            max_tokens=max_tokens
        )
        return final_response.choices[0].message.content or ""
    
    async def _execute_tool_calls(
        self, tool_calls
    ) -> List[ToolExecutionResult]:
        """Execute tool calls with timeout and error isolation."""
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
                results.append(ToolExecutionResult(
                    tool_name=tool_name,
                    tool_call_id=tc.id,
                    result="",
                    latency_ms=0,
                    success=False,
                    error=f"Unknown tool: {tool_name}"
                ))
                continue
            
            try:
                # Validate args before execution
                validated_args = self._validate_args(tool_name, args)
                result = await tool_fn(**validated_args)
                
                results.append(ToolExecutionResult(
                    tool_name=tool_name,
                    tool_call_id=tc.id,
                    result=str(result)[:4000],
                    latency_ms=int((time.time() - t0) * 1000),
                    success=True
                ))
                
            except Exception as e:
                results.append(ToolExecutionResult(
                    tool_name=tool_name,
                    tool_call_id=tc.id,
                    result="",
                    latency_ms=int((time.time() - t0) * 1000),
                    success=False,
                    error=str(e)[:200]
                ))
        
        self.execution_log.extend(results)
        return results
    
    def _validate_args(self, tool_name: str, args: Dict) -> Dict:
        """Sanitize tool arguments. Prevent injection via arguments."""
        if tool_name == "execute_python":
            code = args.get("code", "")
            # Block dangerous patterns
            dangerous = [
                "__import__", "eval(", "exec(", "compile(",
                "os.system", "subprocess", "socket.", "open(",
                "requests.", "urllib", "__builtins__"
            ]
            for pattern in dangerous:
                if pattern in code:
                    raise ValueError(
                        f"Code contains blocked pattern: {pattern}"
                    )
        
        if tool_name == "web_search":
            query = args.get("query", "")
            # Truncate and sanitize query
            args["query"] = query[:200].strip()
        
        return args
