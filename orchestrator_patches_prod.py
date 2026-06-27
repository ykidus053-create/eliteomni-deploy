import re
import logging
import sqlite3
import time
import asyncio
from typing import List, Dict, Callable, Optional, Awaitable, Any
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger("orchestrator.ml_patches")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

DB_PATH = "quality.db"

@contextmanager
def get_db_connection(db_path: str = DB_PATH, timeout: float = 10.0):
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    finally:
        conn.close()

# ── Upgraded: Meta-Controller for Cost & Latency ─────────────────────────────
class MetaController:
    """Tracks API costs and dynamically routes to cheaper/faster models if budget is low."""
    def __init__(self, daily_budget_usd: float = 5.0):
        self.daily_budget = daily_budget_usd
        self.current_spend = 0.0
        self.latency_history = []
        
    def track_cost(self, model: str, input_tokens: int, output_tokens: int):
        cost_map = {"mistral-large-latest": 0.008, "mistral-medium-3.5": 0.002, "mistral-small-latest": 0.0002}
        cost = (input_tokens * cost_map.get(model, 0.002) + output_tokens * cost_map.get(model, 0.002)) / 1000
        self.current_spend += cost
        
    def route_model(self, requested_model: str, complexity: str) -> str:
        # If we are over 80% budget, force downgrade
        if self.current_spend > self.daily_budget * 0.8:
            if requested_model == "mistral-large-latest": return "mistral-medium-3.5"
            if requested_model == "mistral-medium-3.5": return "mistral-small-latest"
        # If latency is high and query is easy, downgrade
        if complexity == "easy" and self.latency_history and np.mean(self.latency_history[-5:]) > 2000:
            return "mistral-small-latest"
        return requested_model

meta_controller = MetaController()

# ── RLAIF Ensemble Averaging ────────────────────────────────────────────────
async def ensemble_rlaif_score(
    response: str,
    prompt: str,
    reward_callables: List[Callable[[str, str], Awaitable[float]]],
    weights: Optional[List[float]] = None
) -> float:
    if not reward_callables: return 0.5
    if weights is None: weights = [1.0 / len(reward_callables)] * len(reward_callables)

    tasks = [rm(response, prompt) for rm in reward_callables]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_scores, valid_weights = [], []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Reward model {i} failed: {res}.")
        else:
            valid_scores.append(res)
            valid_weights.append(weights[i])

    if not valid_scores: return 0.5
    normalized_weights = np.array(valid_weights) / sum(valid_weights)
    return float(np.clip(np.average(valid_scores, weights=normalized_weights), 0.0, 1.0))

def save_sft_example_with_curriculum(msg: str, response: str, complexity: str, skill: str):
    rank_weight = {"easy": 1, "medium": 2, "hard": 3}.get(complexity.lower(), 2)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS sft_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, skill TEXT, prompt TEXT,
                response TEXT, complexity TEXT, rank_weight INTEGER)""")
            cursor.execute("INSERT INTO sft_store (timestamp,skill,prompt,response,complexity,rank_weight) VALUES (?,?,?,?,?,?)",
                           (time.time(), skill, msg, response, complexity, rank_weight))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to save SFT curriculum data: {e}")

def get_optimized_hyperparameters() -> Dict[str, float]:
    defaults = {"temperature": 0.15, "top_p": 0.92, "tree_search_n": 2.0}
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT temperature FROM budget_log WHERE budget > 1024 ORDER BY ts DESC LIMIT 20")
            rows = cursor.fetchall()
            if rows:
                avg_temp = float(np.mean([r[0] for r in rows]))
                defaults["temperature"] = float(np.clip((0.8 * defaults["temperature"]) + (0.2 * avg_temp), 0.05, 0.4))
    except Exception:
        pass
    return defaults

def denoise_query_input(text: str) -> str:
    if not text: return ""
    NORMALIZATION_MAP = {
        re.compile(r"\bcalc\b", re.I): "calculate",
        re.compile(r"\bcod(e|ing)?\b", re.I): "coder",
        re.compile(r"\bsearc(h)?\b", re.I): "search",
    }
    cleaned = text.strip()
    for pattern, replacement in NORMALIZATION_MAP.items():
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned

def dynamic_mcts_branch_factor(score_variance: float, base_n: int = 2) -> int:
    if score_variance < 0.0: return base_n
    if score_variance > 0.7: return min(base_n * 2, 8)
    elif score_variance < 0.2: return max(base_n // 2, 1)
    return base_n

def get_calibrated_confidence(features: Dict[str, Any]) -> float:
    w = np.array([0.15, 0.45, -0.0005])
    b = 0.5
    skill_val = {"coder": 0.8, "researcher": 0.6, "calculator": 0.2}.get(features.get("skill", "general"), 0.4)
    comp_val = {"easy": 0.1, "medium": 0.5, "hard": 0.9}.get(features.get("complexity", "medium"), 0.5)
    text_len = float(features.get("msg_len", 100))
    latent_posterior = (skill_val * w[0]) + (comp_val * w[1]) + (text_len * w[2]) + b
    return float(1.0 / (1.0 + np.exp(-latent_posterior)))
