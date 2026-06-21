"""
Lightweight LangChain-based tracing for EliteOmni.
Logs every LLM call (prompt, response, latency, errors) to eliteomni.db
without changing any existing generation logic.
"""
import time
import sqlite3
import json
import os
from datetime import datetime, timezone

from langchain_core.callbacks.base import BaseCallbackHandler

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eliteomni.db")

def _ensure_table():
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                label TEXT,
                prompt_preview TEXT,
                response_preview TEXT,
                latency_ms REAL,
                token_count INTEGER,
                error TEXT,
                skill TEXT,
                complexity TEXT,
                model TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[LangChainTrace] table setup failed: {e}")

_ensure_table()

class EliteOmniTraceHandler(BaseCallbackHandler):
    """Self-hosted tracing handler — logs LLM calls to SQLite, no external service."""

    def __init__(self, label: str = "", skill: str = "", complexity: str = "", model: str = ""):
        self.label = label
        self.skill = skill
        self.complexity = complexity
        self.model = model
        self.t_start = None
        self.prompt_preview = ""

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.t_start = time.time()
        if prompts:
            self.prompt_preview = str(prompts[0])[:500]
        print(f"[LangChainTrace] start label={self.label} skill={self.skill}")

    def on_llm_end(self, response, **kwargs):
        self._log(response_text=str(response)[:1000], error=None)

    def on_llm_error(self, error, **kwargs):
        self._log(response_text="", error=str(error)[:500])

    def _log(self, response_text: str, error: str):
        latency_ms = (time.time() - self.t_start) * 1000 if self.t_start else 0
        try:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute(
                "INSERT INTO llm_traces (ts, label, prompt_preview, response_preview, latency_ms, token_count, error, skill, complexity, model) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    self.label,
                    self.prompt_preview,
                    response_text,
                    latency_ms,
                    len(response_text.split()) if response_text else 0,
                    error,
                    self.skill,
                    self.complexity,
                    self.model,
                )
            )
            conn.commit()
            conn.close()
            print(f"[LangChainTrace] logged label={self.label} latency={latency_ms:.0f}ms error={bool(error)}")
        except Exception as e:
            print(f"[LangChainTrace] log failed: {e}")


def trace_call(label: str, skill: str = "", complexity: str = "", model: str = "",
                prompt_text: str = "", response_text: str = "", latency_ms: float = 0,
                error: str = None):
    """
    Manual trace function for non-LangChain code paths (like our existing
    mistral_stream generator, which isn't a LangChain LLM object).
    Call this directly after any LLM call to log it the same way.
    """
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(
            "INSERT INTO llm_traces (ts, label, prompt_preview, response_preview, latency_ms, token_count, error, skill, complexity, model) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                label,
                str(prompt_text)[:500],
                str(response_text)[:1000],
                latency_ms,
                len(str(response_text).split()) if response_text else 0,
                error,
                skill,
                complexity,
                model,
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[LangChainTrace] manual log failed: {e}")


def get_recent_traces(limit: int = 50) -> list:
    """Fetch recent traces for a debug dashboard endpoint."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM llm_traces ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[LangChainTrace] fetch failed: {e}")
        return []
