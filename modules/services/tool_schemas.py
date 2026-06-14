"""
tool_schemas.py — JSON-schema tool definitions for Mistral native function calling.
Dispatches via agents._run_one_tool, so all existing tool implementations are reused.
"""

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for current information, news, facts, or anything requiring up-to-date data. Use for questions about recent events, prices, current status of people/companies, or anything that could have changed recently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query, 1-6 words, specific and focused."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Fetch and read the full text content of a specific URL/web page. Use when you need details beyond a search snippet, or when the user provides a link.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to fetch, including https://"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command on the local machine and return stdout/stderr. Use for file operations, running scripts, checking system state, or quick computations. Destructive commands are blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pdf",
            "description": "Extract text content from a PDF file on disk, given its local file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Local filesystem path to the PDF file."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a mathematical expression precisely. Use for arithmetic, algebra, unit conversions, or any calculation where precision matters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to evaluate, e.g. '12 * (3 + 7) / 2'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get current weather conditions for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name or location, e.g. 'Paris, France'"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_lookup",
            "description": "Search the local knowledge base (textbooks, docs, indexed material) for relevant passages on a topic. Use for technical/academic questions where local indexed knowledge might have detailed reference material.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic or question to search the knowledge base for."}
                },
                "required": ["query"]
            }
        }
    },
]


def dispatch_tool_call(name: str, args: dict) -> str:
    """Maps a native tool_call (name + parsed args dict) to the existing tool implementations."""
    from modules.services.agents import _run_one_tool
    name = (name or "").strip().lower()

    if name == "search":
        return _run_one_tool("SEARCH", args.get("query", ""))
    if name == "fetch":
        return _run_one_tool("FETCH", args.get("url", ""))
    if name == "bash":
        return _run_one_tool("BASH", args.get("cmd", ""))
    if name == "pdf":
        return _run_one_tool("PDF", args.get("path", ""))
    if name == "calc":
        return _run_one_tool("CALC", args.get("expression", ""))
    if name == "weather":
        return _run_one_tool("WEATHER", args.get("location", ""))
    if name == "rag_lookup":
        from modules.services.search import rag_get
        hits = rag_get(args.get("query", ""), k=3)
        if not hits:
            return "[No relevant results in knowledge base]"
        return "\n\n".join(f"[{h.get('source','?')}] {h.get('text','')}" for h in hits)

    return f"[Unknown tool: {name}]"
