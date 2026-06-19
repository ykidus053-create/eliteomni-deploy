"""
Implementations from Goodfellow, Bengio & Courville — Deep Learning (MIT Press)
Each function maps directly to a book section.
"""
import math, random, time, sqlite3, os, threading
from typing import List, Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# §7.8  Early Stopping  (Algorithm 7.1)
# ─────────────────────────────────────────────────────────────────
class EarlyStopping:
    """Algorithm 7.1 — patience-based early stopping."""
    def __init__(self, patience: int = 5, n_steps_between_evals: int = 1):
        self.patience   = patience
        self.n          = n_steps_between_evals
        self.best_val   = float('inf')
        self.best_step  = 0
        self.best_params: Optional[Dict] = None
        self.steps      = 0
        self.j          = 0   # consecutive non-improvements

    def step(self, val_error: float, params: Dict) -> bool:
        """Call after every eval. Returns True if training should STOP."""
        self.steps += self.n
        if val_error < self.best_val:
            self.best_val    = val_error
            self.best_step   = self.steps
            self.best_params = dict(params)
            self.j           = 0
        else:
            self.j += 1
        return self.j >= self.patience

    def best(self) -> Tuple[Optional[Dict], int]:
        return self.best_params, self.best_step


# ─────────────────────────────────────────────────────────────────
# §7.11  Bagging / Bootstrap Aggregating
# ─────────────────────────────────────────────────────────────────
def bootstrap_sample(dataset: List, seed: int = None) -> List:
    """Sample with replacement — one bootstrap replica of dataset."""
    rng = random.Random(seed)
    n   = len(dataset)
    return [dataset[rng.randint(0, n - 1)] for _ in range(n)]

class BaggingEnsemble:
    """§7.11 — Train k models on k bootstrap replicas, vote at inference."""
    def __init__(self, k: int = 5):
        self.k       = k
        self.models: List = []

    def fit(self, dataset: List, train_fn):
        """train_fn(dataset) -> model object with .predict(x) method."""
        self.models = []
        for i in range(self.k):
            replica = bootstrap_sample(dataset, seed=i)
            self.models.append(train_fn(replica))
        return self

    def predict(self, x) -> Dict:
        """Majority vote over all k models."""
        votes = [m.predict(x) for m in self.models]
        counts: Dict = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        winner = max(counts, key=counts.get)
        confidence = counts[winner] / len(votes)
        return {"prediction": winner, "confidence": confidence,
                "votes": counts, "n_models": len(self.models)}

    def predict_regression(self, x) -> float:
        """Average prediction for regression tasks."""
        preds = [m.predict(x) for m in self.models]
        return sum(preds) / len(preds)


# ─────────────────────────────────────────────────────────────────
# §7.12  Dropout  (forward pass + weight scaling at inference)
# ─────────────────────────────────────────────────────────────────
def dropout_mask(size: int, keep_prob: float, rng=None) -> List[float]:
    """Bernoulli mask µ — 1 = keep unit, 0 = drop."""
    r = rng or random
    return [1.0 if r.random() < keep_prob else 0.0 for _ in range(size)]

def dropout_forward(activations: List[float],
                    keep_prob: float,
                    training: bool,
                    rng=None) -> Tuple[List[float], List[float]]:
    """
    §7.12 forward pass.
    Training  : apply Bernoulli mask, scale by 1/keep_prob (inverted dropout).
    Inference : return activations unchanged (expectation = weight_scaled sum).
    Returns (masked_activations, mask).
    """
    if not training:
        return list(activations), [1.0] * len(activations)
    mask = dropout_mask(len(activations), keep_prob, rng)
    out  = [a * m / keep_prob for a, m in zip(activations, mask)]
    return out, mask


# ─────────────────────────────────────────────────────────────────
# §8.3  SGD with Momentum  (Algorithm 8.2)
# ─────────────────────────────────────────────────────────────────
class SGDMomentum:
    """
    Algorithm 8.2 — SGD with momentum.
    v ← α*v − ε*∇θ J(θ)
    θ ← θ + v
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr       = lr
        self.momentum = momentum          # α in the book
        self.velocity: Dict[str, float] = {}

    def step(self, params: Dict[str, float],
             grads: Dict[str, float]) -> Dict[str, float]:
        for k in params:
            v_prev          = self.velocity.get(k, 0.0)
            v_new           = self.momentum * v_prev - self.lr * grads.get(k, 0.0)
            self.velocity[k] = v_new
            params[k]       += v_new
        return params

    def reset(self):
        self.velocity.clear()


# ─────────────────────────────────────────────────────────────────
# §8.5.2  RMSProp  (Algorithm 8.5)
# ─────────────────────────────────────────────────────────────────
class RMSProp:
    """
    Algorithm 8.5 — RMSProp.
    r ← ρ*r + (1−ρ)*g⊙g
    ∆θ = −(ε / √(δ+r)) ⊙ g
    """
    def __init__(self, lr: float = 0.001, rho: float = 0.9,
                 delta: float = 1e-6):
        self.lr    = lr
        self.rho   = rho
        self.delta = delta
        self.r: Dict[str, float] = {}

    def step(self, params: Dict[str, float],
             grads: Dict[str, float]) -> Dict[str, float]:
        for k in params:
            g      = grads.get(k, 0.0)
            r_prev = self.r.get(k, 0.0)
            r_new  = self.rho * r_prev + (1 - self.rho) * g * g
            self.r[k]  = r_new
            params[k] -= self.lr / math.sqrt(self.delta + r_new) * g
        return params

    def reset(self):
        self.r.clear()


# ─────────────────────────────────────────────────────────────────
# §8.5.3  Adam  (Algorithm 8.7)
# ─────────────────────────────────────────────────────────────────
class Adam:
    """
    Algorithm 8.7 — Adam optimizer.
    s ← ρ₁*s + (1−ρ₁)*g          (1st moment / momentum)
    r ← ρ₂*r + (1−ρ₂)*g⊙g        (2nd moment)
    ŝ = s/(1−ρ₁ᵗ)                 (bias correction)
    r̂ = r/(1−ρ₂ᵗ)
    ∆θ = −ε * ŝ / (√r̂ + δ)
    """
    def __init__(self, lr: float = 0.001, rho1: float = 0.9,
                 rho2: float = 0.999, delta: float = 1e-8):
        self.lr    = lr
        self.rho1  = rho1
        self.rho2  = rho2
        self.delta = delta
        self.t     = 0
        self.s: Dict[str, float] = {}
        self.r: Dict[str, float] = {}

    def step(self, params: Dict[str, float],
             grads: Dict[str, float]) -> Dict[str, float]:
        self.t += 1
        for k in params:
            g     = grads.get(k, 0.0)
            s_new = self.rho1 * self.s.get(k, 0.0) + (1 - self.rho1) * g
            r_new = self.rho2 * self.r.get(k, 0.0) + (1 - self.rho2) * g * g
            self.s[k] = s_new
            self.r[k] = r_new
            s_hat = s_new / (1 - self.rho1 ** self.t)
            r_hat = r_new / (1 - self.rho2 ** self.t)
            params[k] -= self.lr * s_hat / (math.sqrt(r_hat) + self.delta)
        return params

    def reset(self):
        self.t = 0
        self.s.clear()
        self.r.clear()


# ─────────────────────────────────────────────────────────────────
# §8.4  Glorot / Xavier Initialisation  (eq. 8.23)
# ─────────────────────────────────────────────────────────────────
def glorot_uniform(fan_in: int, fan_out: int) -> float:
    """
    §8.4 eq 8.23 — single weight sample from Glorot uniform.
    W_{i,j} ~ U[-√(6/(m+n)), +√(6/(m+n))]
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(-limit, limit)

def glorot_init_layer(fan_in: int, fan_out: int) -> List[List[float]]:
    """Full weight matrix for one layer."""
    return [[glorot_uniform(fan_in, fan_out) for _ in range(fan_out)]
            for _ in range(fan_in)]


# ─────────────────────────────────────────────────────────────────
# §8.7.1  Batch Normalisation
# ─────────────────────────────────────────────────────────────────
def batch_norm_forward(H: List[float],
                       gamma: float = 1.0,
                       beta: float  = 0.0,
                       eps: float   = 1e-8,
                       training: bool = True,
                       running_mean: float = 0.0,
                       running_var: float  = 1.0,
                       momentum: float     = 0.1):
    """
    §8.7.1 — Batch Normalisation forward pass.
    Training : normalise over mini-batch statistics.
    Inference: use running statistics accumulated during training.
    Returns (normed_H, updated_running_mean, updated_running_var, cache).
    """
    n = len(H)
    if training and n > 1:
        mu    = sum(H) / n
        var   = sum((h - mu) ** 2 for h in H) / n
        H_hat = [(h - mu) / math.sqrt(var + eps) for h in H]
        # update running stats (exponential moving average)
        new_mean = (1 - momentum) * running_mean + momentum * mu
        new_var  = (1 - momentum) * running_var  + momentum * var
    else:
        H_hat    = [(h - running_mean) / math.sqrt(running_var + eps) for h in H]
        new_mean = running_mean
        new_var  = running_var
    out   = [gamma * h + beta for h in H_hat]
    cache = {"H": H, "H_hat": H_hat, "mu": running_mean if not training else sum(H)/n,
             "var": running_var if not training else sum((h - sum(H)/n)**2 for h in H)/n,
             "eps": eps, "gamma": gamma}
    return out, new_mean, new_var, cache


# ─────────────────────────────────────────────────────────────────
# §8.7.6  Curriculum Learning — difficulty sorter
# ─────────────────────────────────────────────────────────────────
def curriculum_sort(examples: List[Dict],
                    difficulty_key: str = "difficulty",
                    strategy: str = "easy_first") -> List[Dict]:
    """
    §8.7.6 — Sort training examples by difficulty.
    strategy='easy_first'  : start easy, ramp to hard (standard curriculum).
    strategy='hard_first'  : anti-curriculum (sometimes useful for fine-tuning).
    strategy='interleaved' : alternate easy/hard.
    """
    scored = sorted(examples, key=lambda x: x.get(difficulty_key, 0.5))
    if strategy == "easy_first":
        return scored
    if strategy == "hard_first":
        return list(reversed(scored))
    if strategy == "interleaved":
        easy = scored[:len(scored)//2]
        hard = list(reversed(scored[len(scored)//2:]))
        out  = []
        for e, h in zip(easy, hard):
            out.extend([e, h])
        return out
    return scored

def score_example_difficulty(text: str, label=None) -> float:
    """
    Heuristic difficulty score 0.0 (easy) – 1.0 (hard).
    Used to build curriculum from raw training pairs.
    """
    score = min(1.0, len(text) / 500)          # length proxy
    score += 0.1 * text.count(",")             # syntactic complexity
    score += 0.05 * len(set(text.split()))     # vocabulary diversity
    return min(1.0, score / 2.0)


# ─────────────────────────────────────────────────────────────────
# §11.4  Hyperparameter Selection — grid & random search
# ─────────────────────────────────────────────────────────────────
def random_log_uniform(low: float, high: float) -> float:
    """Sample from log-uniform distribution — §11.4 recommended for lr, weight decay."""
    log_low  = math.log10(low)
    log_high = math.log10(high)
    return 10 ** random.uniform(log_low, log_high)

def random_search_config(search_space: Dict, n_trials: int = 20,
                          seed: int = 42) -> List[Dict]:
    """
    §11.4 — Random hyperparameter search.
    search_space format:
      {"lr": ("log_uniform", 1e-5, 1e-1),
       "dropout": ("uniform", 0.1, 0.9),
       "layers": ("choice", [1, 2, 3, 4])}
    """
    rng = random.Random(seed)
    configs = []
    for _ in range(n_trials):
        cfg = {}
        for param, spec in search_space.items():
            kind = spec[0]
            if kind == "log_uniform":
                cfg[param] = 10 ** rng.uniform(math.log10(spec[1]),
                                                math.log10(spec[2]))
            elif kind == "uniform":
                cfg[param] = rng.uniform(spec[1], spec[2])
            elif kind == "int_uniform":
                cfg[param] = rng.randint(spec[1], spec[2])
            elif kind == "choice":
                cfg[param] = rng.choice(spec[1])
            else:
                cfg[param] = spec[1]
        configs.append(cfg)
    return configs

class HyperparameterTracker:
    """Persist trial results to SQLite for cross-run comparison (§11.4)."""
    def __init__(self, db_path: str = None):
        self.db = db_path or os.path.expanduser("~/eliteomni_hparam.db")
        self._init()

    def _init(self):
        con = sqlite3.connect(self.db)
        con.execute("""CREATE TABLE IF NOT EXISTS trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, config TEXT, val_loss REAL, val_acc REAL, notes TEXT)""")
        con.commit(); con.close()

    def record(self, config: Dict, val_loss: float, val_acc: float = 0.0,
               notes: str = ""):
        import json
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO trials(ts,config,val_loss,val_acc,notes) VALUES(?,?,?,?,?)",
                    (time.time(), json.dumps(config), val_loss, val_acc, notes))
        con.commit(); con.close()

    def best(self, metric: str = "val_loss") -> Optional[Dict]:
        import json
        con = sqlite3.connect(self.db)
        col = "val_loss" if metric == "val_loss" else "val_acc"
        order = "ASC" if metric == "val_loss" else "DESC"
        row = con.execute(
            f"SELECT config, val_loss, val_acc FROM trials ORDER BY {col} {order} LIMIT 1"
        ).fetchone()
        con.close()
        if row:
            return {"config": json.loads(row[0]), "val_loss": row[1], "val_acc": row[2]}
        return None


# ─────────────────────────────────────────────────────────────────
# §17.2  Importance Sampling
# ─────────────────────────────────────────────────────────────────
def importance_sampling_estimate(samples: List[float],
                                  p_weights: List[float],
                                  q_weights: List[float]) -> float:
    """
    §17.2 — Importance sampling estimator.
    ŝ_q = (1/n) Σ p(x_i)*f(x_i) / q(x_i)
    samples   : f(x) values
    p_weights : p(x) probabilities (target distribution)
    q_weights : q(x) probabilities (proposal distribution, what we sampled from)
    """
    assert len(samples) == len(p_weights) == len(q_weights), "lengths must match"
    n = len(samples)
    total = sum(p * f / max(q, 1e-12)
                for f, p, q in zip(samples, p_weights, q_weights))
    return total / n

def self_normalised_importance_sampling(samples: List[float],
                                         p_weights: List[float],
                                         q_weights: List[float]) -> float:
    """
    Self-normalised IS — more stable when p/q ratio varies widely.
    w_i = p(x_i)/q(x_i);  estimate = Σ w_i f(x_i) / Σ w_i
    """
    ratios = [p / max(q, 1e-12) for p, q in zip(p_weights, q_weights)]
    total_w = sum(ratios)
    if total_w == 0:
        return 0.0
    return sum(w * f for w, f in zip(ratios, samples)) / total_w

def optimal_proposal_distribution(p_weights: List[float],
                                   f_values: List[float]) -> List[float]:
    """
    §17.2 eq 17.13 — optimal q*(x) ∝ p(x)|f(x)|.
    Returns normalised q* weights.
    """
    raw = [p * abs(f) for p, f in zip(p_weights, f_values)]
    z   = sum(raw)
    return [r / z for r in raw] if z > 0 else [1.0 / len(raw)] * len(raw)


# ─────────────────────────────────────────────────────────────────
# Integration shim — drop-in replacements for existing stubs
# ─────────────────────────────────────────────────────────────────
_DEFAULT_ADAM     = Adam(lr=0.001)
_DEFAULT_RMSPROP  = RMSProp(lr=0.001)
_DEFAULT_SGD      = SGDMomentum(lr=0.01, momentum=0.9)

def get_optimizer(name: str = "adam", **kwargs):
    """Factory used by the pipeline to get a book-correct optimiser."""
    name = name.lower()
    if name == "adam":
        return Adam(**{k: v for k, v in kwargs.items()
                       if k in ("lr","rho1","rho2","delta")})
    if name in ("rmsprop", "rms"):
        return RMSProp(**{k: v for k, v in kwargs.items()
                          if k in ("lr","rho","delta")})
    if name in ("sgd", "momentum"):
        return SGDMomentum(**{k: v for k, v in kwargs.items()
                              if k in ("lr","momentum")})
    return Adam()

def build_curriculum(sft_pairs: List[Dict]) -> List[Dict]:
    """
    Score and sort an SFT dataset by difficulty — plug into save_sft_example_with_curriculum.
    sft_pairs: list of {"skill":..., "prompt":..., "response":..., "complexity":...}
    """
    for ex in sft_pairs:
        if "difficulty" not in ex:
            ex["difficulty"] = score_example_difficulty(
                ex.get("prompt", "") + ex.get("response", ""))
    return curriculum_sort(sft_pairs, strategy="easy_first")

def ensemble_rlaif_score_sync(response: str, prompt: str,
                               reward_fns: List, weights: List[float] = None) -> float:
    """
    Synchronous wrapper for ensemble_rlaif_score (§7.11 bagging applied to reward models).
    Replaces the async version in ml_patches.py for contexts without an event loop.
    """
    if not reward_fns:
        return 0.5
    if weights is None:
        weights = [1.0 / len(reward_fns)] * len(reward_fns)
    scores, valid_w = [], []
    for fn, w in zip(reward_fns, weights):
        try:
            scores.append(fn(response, prompt))
            valid_w.append(w)
        except Exception as e:
            print(f"[EnsembleRLAIF] reward fn failed: {e}")
    if not scores:
        return 0.5
    total_w = sum(valid_w)
    return sum(s * w for s, w in zip(scores, valid_w)) / total_w


if __name__ == "__main__":
    # Quick smoke-test of every implementation
    print("Testing dl_book_implementations.py ...")

    # §7.8 Early stopping
    es = EarlyStopping(patience=3)
    for val in [0.9, 0.8, 0.85, 0.87, 0.88, 0.89]:
        stopped = es.step(val, {"w": val})
    assert es.best_val == 0.8, f"early stop best={es.best_val}"
    print("  ✓ §7.8  EarlyStopping")

    # §7.11 Bagging bootstrap
    ds = list(range(10))
    s  = bootstrap_sample(ds, seed=0)
    assert len(s) == 10, "bootstrap length mismatch"
    print("  ✓ §7.11 Bagging / bootstrap_sample")

    # §7.12 Dropout
    acts = [1.0] * 100
    out, mask = dropout_forward(acts, keep_prob=0.5, training=True)
    assert len(out) == 100
    print("  ✓ §7.12 Dropout forward")

    # §8.3 SGD Momentum
    params = {"w": 1.0}; grads = {"w": 0.1}
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    params = opt.step(params, grads)
    assert params["w"] < 1.0
    print("  ✓ §8.3  SGD with Momentum")

    # §8.5.2 RMSProp
    params = {"w": 1.0}; grads = {"w": 0.1}
    rms = RMSProp(lr=0.01)
    params = rms.step(params, grads)
    assert params["w"] < 1.0
    print("  ✓ §8.5.2 RMSProp")

    # §8.5.3 Adam
    params = {"w": 1.0}; grads = {"w": 0.1}
    adam = Adam(lr=0.01)
    params = adam.step(params, grads)
    assert params["w"] < 1.0
    print("  ✓ §8.5.3 Adam")

    # §8.4 Glorot init
    w = glorot_uniform(512, 256)
    limit = math.sqrt(6.0 / (512 + 256))
    assert -limit <= w <= limit
    print("  ✓ §8.4  Glorot/Xavier init")

    # §8.7.1 Batch Norm
    H = [1.0, 2.0, 3.0, 4.0]
    out, rm, rv, _ = batch_norm_forward(H, training=True)
    assert abs(sum(out) / len(out)) < 1e-6, "batch norm mean not ~0"
    print("  ✓ §8.7.1 Batch Normalisation")

    # §8.7.6 Curriculum
    data = [{"text": "hi", "difficulty": 0.1},
            {"text": "explain quantum entanglement in detail", "difficulty": 0.9},
            {"text": "2+2", "difficulty": 0.05}]
    sorted_data = curriculum_sort(data, strategy="easy_first")
    assert sorted_data[0]["difficulty"] <= sorted_data[-1]["difficulty"]
    print("  ✓ §8.7.6 Curriculum Learning")

    # §11.4 Random search
    space = {"lr": ("log_uniform", 1e-4, 1e-1),
             "dropout": ("uniform", 0.1, 0.8),
             "layers": ("choice", [1, 2, 3])}
    configs = random_search_config(space, n_trials=10)
    assert len(configs) == 10
    print("  ✓ §11.4 Hyperparameter random search")

    # §17.2 Importance sampling
    samples  = [1.0, 2.0, 3.0]
    p_w      = [0.5, 0.3, 0.2]
    q_w      = [0.3, 0.4, 0.3]
    est      = importance_sampling_estimate(samples, p_w, q_w)
    assert est > 0
    print("  ✓ §17.2 Importance Sampling")

    print("\nAll tests passed.")
