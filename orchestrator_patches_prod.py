import re
import logging
import sqlite3
import time
import asyncio
from typing import List, Dict, Callable, Optional, Awaitable, Any
import numpy as np
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Production Configuration & Logging
# ---------------------------------------------------------------------------
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
    """Yields a database connection with production safety pragmas enabled."""
    conn = sqlite3.connect(db_path, timeout=timeout)
    # Enable Write-Ahead Logging for better concurrent read/write performance
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# 1. RLAIF Ensemble Averaging (Ch 7.11)
# ---------------------------------------------------------------------------
async def ensemble_rlaif_score(
    response: str, 
    prompt: str, 
    reward_callables: List[Callable[[str, str], Awaitable[float]]],
    weights: Optional[List[float]] = None
) -> float:
    """
    Evaluates multiple reward models concurrently and returns a weighted ensemble score.
    Replaces static heuristic grading with actual AI-feedback bagging.
    """
    if not reward_callables:
        return 0.5  # Neutral fallback

    if weights is None:
        weights = [1.0 / len(reward_callables)] * len(reward_callables)

    # Execute all reward models in parallel
    tasks = [rm(response, prompt) for rm in reward_callables]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_scores = []
    valid_weights = []
    
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Reward model {i} failed: {res}. Excluding from ensemble.")
        else:
            valid_scores.append(res)
            valid_weights.append(weights[i])

    if not valid_scores:
        logger.warning("All reward models failed. Returning default score.")
        return 0.5

    # Re-normalize weights for the successful calls
    normalized_weights = np.array(valid_weights) / sum(valid_weights)
    ensemble_score = np.average(valid_scores, weights=normalized_weights)
    
    return float(np.clip(ensemble_score, 0.0, 1.0))

# ---------------------------------------------------------------------------
# 2. Curriculum Learning Data Sorter (Ch 8.7.1)
# ---------------------------------------------------------------------------
def save_sft_example_with_curriculum(msg: str, response: str, complexity: str, skill: str):
    """
    Saves fine-tuning examples with an integer rank for curriculum training.
    Uses proper DB context management to prevent locks.
    """
    complexity_weights = {"easy": 1, "medium": 2, "hard": 3}
    rank_weight = complexity_weights.get(complexity.lower(), 2)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sft_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    skill TEXT,
                    prompt TEXT,
                    response TEXT,
                    complexity TEXT,
                    rank_weight INTEGER
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rank_weight ON sft_store(rank_weight)
            """)
            cursor.execute(
                "INSERT INTO sft_store (timestamp, skill, prompt, response, complexity, rank_weight) VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), skill, msg, response, complexity, rank_weight)
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to save SFT curriculum data: {e}")

# ---------------------------------------------------------------------------
# 3. Dynamic Hyperparameter Optimization (Ch 11.4)
# ---------------------------------------------------------------------------
def get_optimized_hyperparameters() -> Dict[str, float]:
    """
    Pulls recent high-performing generation budgets to adapt temperature.
    Falls back gracefully to default parameters if DB read fails.
    """
    defaults = {"temperature": 0.15, "top_p": 0.92, "tree_search_n": 2.0}
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # In a real pipeline, filter by a 'success' or 'high_reward' boolean, not just budget
            cursor.execute("SELECT temperature FROM budget_log WHERE budget > 1024 ORDER BY ts DESC LIMIT 20")
            rows = cursor.fetchall()
            
            if rows:
                avg_temp = float(np.mean([r[0] for r in rows]))
                # Apply a smoothing momentum (e.g., 0.8 * default + 0.2 * historical)
                defaults["temperature"] = float(np.clip((0.8 * defaults["temperature"]) + (0.2 * avg_temp), 0.05, 0.4))
    except Exception as e:
        logger.warning(f"Hyperparameter DB lookup failed, using static defaults. Error: {e}")
        
    return defaults

# ---------------------------------------------------------------------------
# 4. Input Robustness via Regex Denoising (Ch 14.5)
# ---------------------------------------------------------------------------
def denoise_query_input(text: str) -> str:
    """
    Normalizes common syntax mutations before routing.
    Compiled regexes for production speed.
    """
    if not text:
        return ""
        
    NORMALIZATION_MAP = {
        re.compile(r"\bcalc\b", re.I): "calculate",
        re.compile(r"\bcod(e|ing)?\b", re.I): "coder",
        re.compile(r"\bsearc(h)?\b", re.I): "search",
        re.compile(r"\bfunc(tion)?\b", re.I): "function",
        re.compile(r"\bimpl(ement)?\b", re.I): "implement",
    }
    
    cleaned = text.strip()
    for pattern, replacement in NORMALIZATION_MAP.items():
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned

# ---------------------------------------------------------------------------
# 5. Importance Sampling & Variational Calibration (Ch 17.2, 19.4)
# ---------------------------------------------------------------------------
def dynamic_mcts_branch_factor(score_variance: float, base_n: int = 2) -> int:
    """Safely calculates branch factor based on uncertainty, bounded to prevent OOM."""
    if score_variance < 0.0:
        return base_n
        
    if score_variance > 0.7:
        return min(base_n * 2, 8)  # Aggressive, but capped
    elif score_variance < 0.2:
        return max(base_n // 2, 1) # Conserve tokens when highly certain
    return base_n

def get_calibrated_confidence(features: Dict[str, Any]) -> float:
    """
    Learned parametric approximation for response confidence.
    In a full production environment, 'w' and 'b' would be loaded from a trained PyTorch/sklearn artifact.
    """
    w = np.array([0.15, 0.45, -0.0005])  # slight negative penalty for extremely long prompts
    b = 0.5 # bias term
    
    # Feature extraction with safe fallbacks
    skill_val = {"coder": 0.8, "researcher": 0.6, "calculator": 0.2}.get(features.get("skill", "general"), 0.4)
    comp_val = {"easy": 0.1, "medium": 0.5, "hard": 0.9}.get(features.get("complexity", "medium"), 0.5)
    text_len = float(features.get("msg_len", 100))
    
    # Compute dot product
    latent_posterior = (skill_val * w[0]) + (comp_val * w[1]) + (text_len * w[2]) + b
    
    # Sigmoid activation to bound between 0 and 1
    calibrated_probability = 1.0 / (1.0 + np.exp(-latent_posterior))
    return float(calibrated_probability)

if __name__ == "__main__":
    logger.info("Production ML orchestrator patches loaded successfully.")
