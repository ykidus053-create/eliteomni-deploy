import re, time, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    timeout: int = 15
    retry: int = 2

@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    output: Any
    error: Optional[str] = None
    latency_ms: int = 0

def execute_tool_chain(calls: List[ToolCall], tool_registry: Dict[str, Callable], max_workers: int = 4) -> Dict[str, ToolResult]:
    results: Dict[str, ToolResult] = {}
    pending = {c.name: c for c in calls}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    for _ in range(len(calls) + 2):
        if not pending: break
        ready = [c for c in pending.values() if all(d in results and not results[d].error for d in c.depends_on)]
        if not ready: break
        futures = {}
        for call in ready:
            fn = tool_registry.get(call.name)
            if not fn:
                results[call.name] = ToolResult(call.name, call.name, None, f'Unknown: {call.name}')
                del pending[call.name]; continue
            resolved = {k: (results[v[1:]].output if isinstance(v,str) and v.startswith('$') and v[1:] in results else v)
                        for k, v in call.args.items()}
            futures[executor.submit(_run_retry, fn, resolved, call.retry)] = call
            del pending[call.name]
        for fut, call in futures.items():
            t0 = time.time()
            try: r = ToolResult(call.name, call.name, fut.result(timeout=call.timeout+2), latency_ms=int((time.time()-t0)*1000))
            except Exception as e: r = ToolResult(call.name, call.name, None, str(e))
            results[call.name] = r
    executor.shutdown(wait=False)
    return results

def _run_retry(fn, args, retries):
    for i in range(max(1, retries)):
        try: return fn(**args)
        except Exception as e:
            if i == retries-1: raise
            time.sleep(0.5*(i+1))

def compose_tool_prompt(results: Dict[str, ToolResult]) -> str:
    if not results: return ''
    lines = ['[TOOL RESULTS]']
    for name, r in results.items():
        out = f'ERROR: {r.error}' if r.error else str(r.output)[:400]
        lines.append(f'- {name}: {out}')
    lines.append('[END TOOL RESULTS]')
    return '\n'.join(lines)
