"""
Deep Learning — Goodfellow, Bengio & Courville  (MIT Press)
Second batch: every major algorithm not yet in dl_book_implementations.py
"""
import math, random, time, collections
from typing import List, Dict, Tuple, Optional, Callable

# ─────────────────────────────────────────────────────────────────
# §4.3  Gradient Descent (vanilla + line-search step)
# ─────────────────────────────────────────────────────────────────
def gradient_descent_step(params: Dict[str, float],
                           grads: Dict[str, float],
                           lr: float = 0.01) -> Dict[str, float]:
    """θ ← θ − ε ∇_θ J(θ)"""
    return {k: params[k] - lr * grads.get(k, 0.0) for k in params}

def numerical_gradient(f: Callable, params: Dict[str, float],
                        eps: float = 1e-5) -> Dict[str, float]:
    """Finite-difference gradient — useful for gradient checking."""
    grads = {}
    for k in params:
        orig = params[k]
        params[k] = orig + eps;  fp = f(params)
        params[k] = orig - eps;  fm = f(params)
        params[k] = orig
        grads[k] = (fp - fm) / (2 * eps)
    return grads


# ─────────────────────────────────────────────────────────────────
# §4.4  KKT Constrained Optimisation (Lagrangian helper)
# ─────────────────────────────────────────────────────────────────
def lagrangian(f_val: float,
               eq_constraints: List[Tuple[float, float]],
               ineq_constraints: List[Tuple[float, float]]) -> float:
    """
    §4.4 Generalised Lagrangian.
    L(x,λ,α) = f(x) + Σ λ_i g_i(x) + Σ α_j h_j(x)
    eq_constraints  : list of (λ_i, g_i(x))  — equality
    ineq_constraints: list of (α_j, h_j(x))  — inequality, α_j ≥ 0
    """
    val = f_val
    for lam, g in eq_constraints:
        val += lam * g
    for alpha, h in ineq_constraints:
        assert alpha >= 0, "KKT: inequality multiplier must be ≥ 0"
        val += alpha * h
    return val

def kkt_satisfied(eq_constraints: List[float],
                  ineq_constraints: List[Tuple[float, float]],
                  tol: float = 1e-6) -> bool:
    """Check KKT feasibility conditions."""
    for g in eq_constraints:
        if abs(g) > tol:
            return False
    for alpha, h in ineq_constraints:
        if alpha < 0 or h > tol or abs(alpha * h) > tol:
            return False
    return True


# ─────────────────────────────────────────────────────────────────
# §5.4  Bias–Variance Decomposition + k-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────
def bias_variance_decompose(predictions_per_model: List[List[float]],
                             targets: List[float]) -> Dict[str, float]:
    """
    §5.4 — empirical bias-variance decomposition.
    predictions_per_model: one list per trained model, each of length n.
    Returns bias², variance, noise (irreducible), total expected error.
    """
    n = len(targets)
    k = len(predictions_per_model)
    mean_pred = [sum(predictions_per_model[m][i] for m in range(k)) / k
                 for i in range(n)]
    bias_sq  = sum((mean_pred[i] - targets[i]) ** 2 for i in range(n)) / n
    variance = sum(
        sum((predictions_per_model[m][i] - mean_pred[i]) ** 2 for m in range(k)) / k
        for i in range(n)) / n
    total_mse = sum(
        sum((predictions_per_model[m][i] - targets[i]) ** 2 for m in range(k)) / k
        for i in range(n)) / n
    noise = max(0.0, total_mse - bias_sq - variance)
    return {"bias_squared": round(bias_sq, 6),
            "variance":     round(variance, 6),
            "noise":        round(noise, 6),
            "total_mse":    round(total_mse, 6)}

def k_fold_split(dataset: List, k: int = 5) -> List[Tuple[List, List]]:
    """
    Algorithm 5.1 — k-fold cross-validation split.
    Returns k (train_set, test_set) pairs.
    """
    n     = len(dataset)
    folds = [dataset[i::k] for i in range(k)]
    splits = []
    for i in range(k):
        test  = folds[i]
        train = [x for j, fold in enumerate(folds) if j != i for x in fold]
        splits.append((train, test))
    return splits


# ─────────────────────────────────────────────────────────────────
# §5.5  Maximum Likelihood Estimation
# ─────────────────────────────────────────────────────────────────
def log_likelihood_gaussian(data: List[float],
                             mu: float, sigma: float) -> float:
    """§5.5 — log-likelihood of data under N(mu, sigma²)."""
    n = len(data)
    ll = -n * math.log(sigma) - n * 0.5 * math.log(2 * math.pi)
    ll -= sum((x - mu) ** 2 for x in data) / (2 * sigma ** 2)
    return ll

def mle_gaussian(data: List[float]) -> Tuple[float, float]:
    """§5.5 — closed-form MLE for Gaussian: returns (mu_hat, sigma_hat)."""
    n  = len(data)
    mu = sum(data) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in data) / n)
    return mu, sigma

def mle_bernoulli(data: List[int]) -> float:
    """§5.5 — MLE for Bernoulli: p_hat = mean(data)."""
    return sum(data) / len(data)


# ─────────────────────────────────────────────────────────────────
# §5.9  Minibatch SGD
# ─────────────────────────────────────────────────────────────────
def minibatch_sgd(dataset: List[Dict],
                  params: Dict[str, float],
                  grad_fn: Callable,
                  lr: float = 0.01,
                  batch_size: int = 32,
                  epochs: int = 10,
                  shuffle: bool = True) -> Dict[str, float]:
    """
    §5.9 — Minibatch SGD.
    grad_fn(batch, params) → Dict[str, float] of parameter gradients.
    """
    data = list(dataset)
    for epoch in range(epochs):
        if shuffle:
            random.shuffle(data)
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            grads = grad_fn(batch, params)
            params = gradient_descent_step(params, grads, lr)
    return params


# ─────────────────────────────────────────────────────────────────
# §6.3  Activation Functions
# ─────────────────────────────────────────────────────────────────
def relu(z: float) -> float:
    """§6.3 — Rectified Linear Unit: max(0, z)."""
    return max(0.0, z)

def relu_grad(z: float) -> float:
    return 1.0 if z > 0 else 0.0

def leaky_relu(z: float, alpha: float = 0.01) -> float:
    """§6.3 — Leaky ReLU: max(αz, z)."""
    return z if z >= 0 else alpha * z

def elu(z: float, alpha: float = 1.0) -> float:
    """§6.3 — Exponential Linear Unit."""
    return z if z >= 0 else alpha * (math.exp(z) - 1)

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def tanh_act(z: float) -> float:
    return math.tanh(z)

def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


# ─────────────────────────────────────────────────────────────────
# §6.5  Backpropagation (scalar chain rule + simple MLP)
# ─────────────────────────────────────────────────────────────────
def backprop_chain_rule(upstream_grad: float,
                         local_grads: List[float]) -> List[float]:
    """§6.5 — multiply upstream gradient by each local Jacobian entry."""
    return [upstream_grad * g for g in local_grads]

class SimpleMLP:
    """
    §6.5 — Two-layer MLP with manual backprop (scalar, educational).
    Input → hidden (ReLU) → output (sigmoid).
    """
    def __init__(self, n_in: int, n_hidden: int):
        scale = math.sqrt(6.0 / (n_in + n_hidden))
        self.W1 = [[random.uniform(-scale, scale) for _ in range(n_hidden)]
                   for _ in range(n_in)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [random.uniform(-scale, scale) for _ in range(n_hidden)]
        self.b2 = 0.0

    def forward(self, x: List[float]) -> Tuple[float, Dict]:
        h_pre = [sum(x[i] * self.W1[i][j] for i in range(len(x))) + self.b1[j]
                 for j in range(len(self.b1))]
        h = [relu(v) for v in h_pre]
        o_pre = sum(h[j] * self.W2[j] for j in range(len(h))) + self.b2
        o = sigmoid(o_pre)
        return o, {"x": x, "h_pre": h_pre, "h": h, "o_pre": o_pre, "o": o}

    def backward(self, cache: Dict, y: float,
                 lr: float = 0.01):
        o = cache["o"]; h = cache["h"]; h_pre = cache["h_pre"]; x = cache["x"]
        # output layer gradient
        d_loss_o = o - y                      # BCE gradient
        d_o_pre  = d_loss_o * o * (1 - o)    # sigmoid backward
        # W2, b2
        for j in range(len(self.W2)):
            self.W2[j] -= lr * d_o_pre * h[j]
        self.b2 -= lr * d_o_pre
        # hidden layer
        d_h = [d_o_pre * self.W2[j] for j in range(len(self.W2))]
        d_h_pre = [d_h[j] * relu_grad(h_pre[j]) for j in range(len(h_pre))]
        for i in range(len(x)):
            for j in range(len(self.b1)):
                self.W1[i][j] -= lr * d_h_pre[j] * x[i]
        for j in range(len(self.b1)):
            self.b1[j] -= lr * d_h_pre[j]


# ─────────────────────────────────────────────────────────────────
# §7.1  L1 and L2 Regularisation (penalty + regularised grad)
# ─────────────────────────────────────────────────────────────────
def l2_penalty(params: Dict[str, float], alpha: float) -> float:
    """§7.1 — L2 weight decay: α/2 * ||w||²."""
    return alpha / 2.0 * sum(v ** 2 for v in params.values())

def l1_penalty(params: Dict[str, float], alpha: float) -> float:
    """§7.1 — L1 sparsity: α * ||w||₁."""
    return alpha * sum(abs(v) for v in params.values())

def l2_grad(params: Dict[str, float], alpha: float) -> Dict[str, float]:
    """Gradient of L2 penalty: α * w."""
    return {k: alpha * v for k, v in params.items()}

def l1_grad(params: Dict[str, float], alpha: float) -> Dict[str, float]:
    """Subgradient of L1 penalty: α * sign(w)."""
    return {k: alpha * (1.0 if v > 0 else -1.0 if v < 0 else 0.0)
            for k, v in params.items()}

def regularised_loss(task_loss: float, params: Dict[str, float],
                     alpha: float, mode: str = "l2") -> float:
    """J̃(θ) = J(θ) + αΩ(θ)"""
    penalty = l2_penalty(params, alpha) if mode == "l2" else l1_penalty(params, alpha)
    return task_loss + penalty


# ─────────────────────────────────────────────────────────────────
# §7.4  Dataset Augmentation
# ─────────────────────────────────────────────────────────────────
def augment_text(text: str, ops: List[str] = None) -> List[str]:
    """
    §7.4 — Text augmentation (analogous to image flips/crops).
    ops: subset of ['swap', 'delete', 'duplicate', 'lowercase']
    """
    ops = ops or ['swap', 'delete', 'lowercase']
    words = text.split()
    results = [text]
    if 'swap' in ops and len(words) >= 2:
        i = random.randint(0, len(words) - 2)
        w = list(words)
        w[i], w[i+1] = w[i+1], w[i]
        results.append(" ".join(w))
    if 'delete' in ops and words:
        i = random.randint(0, len(words) - 1)
        results.append(" ".join(words[:i] + words[i+1:]))
    if 'duplicate' in ops and words:
        i = random.randint(0, len(words) - 1)
        results.append(" ".join(words[:i] + [words[i]] + words[i:]))
    if 'lowercase' in ops:
        results.append(text.lower())
    return results

def augment_numeric(value: float, noise_std: float = 0.05,
                    n: int = 3) -> List[float]:
    """§7.5 — Noise injection augmentation for numeric features."""
    return [value + random.gauss(0, noise_std) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────
# §7.5  Noise Robustness — weight noise injection
# ─────────────────────────────────────────────────────────────────
def inject_weight_noise(params: Dict[str, float],
                         std: float = 0.01) -> Dict[str, float]:
    """§7.5 — Add Gaussian noise to weights during training."""
    return {k: v + random.gauss(0, std) for k, v in params.items()}


# ─────────────────────────────────────────────────────────────────
# §8.5.1  AdaGrad  (Algorithm 8.4)
# ─────────────────────────────────────────────────────────────────
class AdaGrad:
    """
    Algorithm 8.4 — AdaGrad.
    r ← r + g ⊙ g
    ∆θ = −(ε / (δ + √r)) ⊙ g
    """
    def __init__(self, lr: float = 0.01, delta: float = 1e-7):
        self.lr    = lr
        self.delta = delta
        self.r: Dict[str, float] = {}

    def step(self, params: Dict[str, float],
             grads: Dict[str, float]) -> Dict[str, float]:
        for k in params:
            g         = grads.get(k, 0.0)
            self.r[k] = self.r.get(k, 0.0) + g * g
            params[k] -= self.lr / (self.delta + math.sqrt(self.r[k])) * g
        return params

    def reset(self):
        self.r.clear()


# ─────────────────────────────────────────────────────────────────
# §8.7.3  Polyak / Exponential Moving Average of Weights
# ─────────────────────────────────────────────────────────────────
class PolyakAveraging:
    """
    §8.7.3 — θ̂(t) = α θ̂(t-1) + (1−α) θ(t)
    Maintains a running EMA of parameter snapshots.
    Use .averaged_params() for inference.
    """
    def __init__(self, alpha: float = 0.999):
        self.alpha   = alpha
        self._avg:   Dict[str, float] = {}
        self.n_steps = 0

    def update(self, params: Dict[str, float]):
        self.n_steps += 1
        if not self._avg:
            self._avg = dict(params)
        else:
            for k in params:
                self._avg[k] = self.alpha * self._avg.get(k, params[k]) + \
                               (1 - self.alpha) * params[k]

    def averaged_params(self) -> Dict[str, float]:
        return dict(self._avg)


# ─────────────────────────────────────────────────────────────────
# §8.2.4 / §10.11  Gradient Clipping
# ─────────────────────────────────────────────────────────────────
def clip_gradients_by_norm(grads: Dict[str, float],
                            max_norm: float = 1.0) -> Dict[str, float]:
    """
    §8.2.4 / §10.11 — Rescale gradient if global norm exceeds max_norm.
    g ← g * (max_norm / ||g||)  when ||g|| > max_norm.
    """
    global_norm = math.sqrt(sum(v ** 2 for v in grads.values()))
    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-8)
        return {k: v * scale for k, v in grads.items()}
    return dict(grads)

def clip_gradients_by_value(grads: Dict[str, float],
                             clip_val: float = 1.0) -> Dict[str, float]:
    """Element-wise value clipping (alternative to norm clipping)."""
    return {k: max(-clip_val, min(clip_val, v)) for k, v in grads.items()}


# ─────────────────────────────────────────────────────────────────
# §10.10  LSTM Cell (forward pass)
# ─────────────────────────────────────────────────────────────────
def lstm_cell_forward(x: List[float],
                       h_prev: List[float],
                       c_prev: List[float],
                       W_f: List[List[float]], b_f: List[float],
                       W_i: List[List[float]], b_i: List[float],
                       W_c: List[List[float]], b_c: List[float],
                       W_o: List[List[float]], b_o: List[float]
                       ) -> Tuple[List[float], List[float]]:
    """
    §10.10 — LSTM cell.
    Gates: forget f, input i, cell candidate c̃, output o.
    c_t = f ⊙ c_{t-1} + i ⊙ c̃
    h_t = o ⊙ tanh(c_t)
    Returns (h_t, c_t).
    """
    inp = x + h_prev  # concatenated input

    def affine(W, b):
        return [sum(inp[j] * W[j][k] for j in range(len(inp))) + b[k]
                for k in range(len(b))]

    f  = [sigmoid(v) for v in affine(W_f, b_f)]
    i  = [sigmoid(v) for v in affine(W_i, b_i)]
    c_ = [tanh_act(v) for v in affine(W_c, b_c)]
    o  = [sigmoid(v) for v in affine(W_o, b_o)]

    c_t = [f[k] * c_prev[k] + i[k] * c_[k] for k in range(len(c_prev))]
    h_t = [o[k] * tanh_act(c_t[k]) for k in range(len(c_t))]
    return h_t, c_t

def init_lstm_weights(input_size: int,
                       hidden_size: int) -> Dict[str, object]:
    """Glorot-initialised LSTM weight matrices."""
    fan = input_size + hidden_size
    scale = math.sqrt(6.0 / (fan + hidden_size))
    def W():
        return [[random.uniform(-scale, scale) for _ in range(hidden_size)]
                for _ in range(fan)]
    def b():
        return [0.0] * hidden_size
    return {"W_f": W(), "b_f": b(), "W_i": W(), "b_i": b(),
            "W_c": W(), "b_c": b(), "W_o": W(), "b_o": b()}


# ─────────────────────────────────────────────────────────────────
# §14.5  Denoising Autoencoder — corruption + reconstruction loss
# ─────────────────────────────────────────────────────────────────
def corrupt_masking(x: List[float],
                    corruption_rate: float = 0.3) -> List[float]:
    """§14.5 — Masking noise: set each element to 0 with prob corruption_rate."""
    return [0.0 if random.random() < corruption_rate else v for v in x]

def corrupt_gaussian(x: List[float], std: float = 0.1) -> List[float]:
    """§14.5 — Additive Gaussian noise corruption."""
    return [v + random.gauss(0, std) for v in x]

def dae_reconstruction_loss(x_clean: List[float],
                             x_reconstructed: List[float]) -> float:
    """§14.5 — MSE reconstruction loss L(x, g(f(x̃)))."""
    n = len(x_clean)
    return sum((a - b) ** 2 for a, b in zip(x_clean, x_reconstructed)) / n


# ─────────────────────────────────────────────────────────────────
# §15.2  Transfer Learning — layer freezing helper
# ─────────────────────────────────────────────────────────────────
class TransferLearningRegistry:
    """
    §15.2 — Track which layers are frozen (shared) vs fine-tunable.
    Mirrors the 'shared lower layers, task-specific upper layers' pattern.
    """
    def __init__(self):
        self.layers: Dict[str, Dict] = {}

    def register(self, name: str, params: Dict[str, float],
                 frozen: bool = False):
        self.layers[name] = {"params": params, "frozen": frozen}

    def freeze(self, name: str):
        if name in self.layers:
            self.layers[name]["frozen"] = True

    def unfreeze(self, name: str):
        if name in self.layers:
            self.layers[name]["frozen"] = False

    def trainable_params(self) -> Dict[str, float]:
        out = {}
        for name, info in self.layers.items():
            if not info["frozen"]:
                out.update({f"{name}.{k}": v
                            for k, v in info["params"].items()})
        return out

    def apply_grads(self, grads: Dict[str, float], lr: float = 0.001):
        for key, grad in grads.items():
            parts = key.split(".", 1)
            if len(parts) == 2:
                layer, param = parts
                if layer in self.layers and not self.layers[layer]["frozen"]:
                    self.layers[layer]["params"][param] = \
                        self.layers[layer]["params"].get(param, 0.0) - lr * grad


# ─────────────────────────────────────────────────────────────────
# §15.1  Greedy Layer-Wise Pretraining
# ─────────────────────────────────────────────────────────────────
def greedy_layerwise_pretrain(data: List[List[float]],
                               layer_sizes: List[int],
                               train_layer_fn: Callable,
                               encode_fn: Callable) -> List:
    """
    §15.1 / Algorithm 15.1 — Greedy layer-wise unsupervised pretraining.
    train_layer_fn(data, in_size, out_size) → layer_model
    encode_fn(layer_model, data) → transformed data (next layer input)
    Returns list of pretrained layer models.
    """
    models = []
    current_data = data
    for i in range(len(layer_sizes) - 1):
        in_size  = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        print(f"[Pretraining] Layer {i+1}: {in_size} → {out_size}")
        layer = train_layer_fn(current_data, in_size, out_size)
        models.append(layer)
        current_data = encode_fn(layer, current_data)
    return models


# ─────────────────────────────────────────────────────────────────
# §17.3  Markov Chain Monte Carlo (Metropolis-Hastings)
# ─────────────────────────────────────────────────────────────────
def metropolis_hastings(log_prob_fn: Callable,
                         init_state: List[float],
                         n_samples: int = 1000,
                         step_size: float = 0.1,
                         burn_in: int = 100) -> List[List[float]]:
    """
    §17.3 — Metropolis-Hastings MCMC sampler.
    log_prob_fn(x) → log p(x)  (unnormalised OK)
    Returns samples after burn-in.
    """
    state   = list(init_state)
    log_p   = log_prob_fn(state)
    samples = []

    for step in range(n_samples + burn_in):
        proposal = [s + random.gauss(0, step_size) for s in state]
        log_p_new = log_prob_fn(proposal)
        log_accept = log_p_new - log_p
        if math.log(random.random() + 1e-12) < log_accept:
            state = proposal
            log_p = log_p_new
        if step >= burn_in:
            samples.append(list(state))

    return samples


# ─────────────────────────────────────────────────────────────────
# §17.4  Gibbs Sampling
# ─────────────────────────────────────────────────────────────────
def gibbs_sampling(conditional_samplers: List[Callable],
                   init_state: List[float],
                   n_samples: int = 500,
                   burn_in: int = 50) -> List[List[float]]:
    """
    §17.4 — Block Gibbs sampling.
    conditional_samplers[i](state) → new value for variable i.
    """
    state   = list(init_state)
    samples = []
    for step in range(n_samples + burn_in):
        for i, sampler in enumerate(conditional_samplers):
            state[i] = sampler(state)
        if step >= burn_in:
            samples.append(list(state))
    return samples


# ─────────────────────────────────────────────────────────────────
# §19.4  Variational Inference — ELBO and Mean-Field update
# ─────────────────────────────────────────────────────────────────
def elbo(log_joint_fn: Callable,
         log_q_fn: Callable,
         samples: List[List[float]]) -> float:
    """
    §19.4 — Evidence Lower BOund: L = E_q[log p(h,v)] − E_q[log q(h|v)]
                                     = E_q[log p(h,v) − log q(h|v)]
    Monte Carlo estimate using provided samples from q.
    """
    total = sum(log_joint_fn(s) - log_q_fn(s) for s in samples)
    return total / len(samples)

def mean_field_update(mu: List[float],
                      log_factor_fn: Callable,
                      n_mc: int = 50) -> List[float]:
    """
    §19.4 — Mean-field coordinate ascent update (one variable at a time).
    For each dimension, maximise ELBO holding others fixed.
    Returns updated mu.
    """
    new_mu = list(mu)
    for i in range(len(mu)):
        best_val  = mu[i]
        best_elbo = float('-inf')
        for delta in [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]:
            candidate     = list(new_mu)
            candidate[i] += delta
            e = log_factor_fn(candidate)
            if e > best_elbo:
                best_elbo = e
                best_val  = candidate[i]
        new_mu[i] = best_val
    return new_mu


# ─────────────────────────────────────────────────────────────────
# §20  Generative Adversarial Network — minimax objective
# ─────────────────────────────────────────────────────────────────
def gan_discriminator_loss(d_real: List[float],
                            d_fake: List[float]) -> float:
    """
    §20 eq 20.81 — Discriminator loss.
    L_D = −E[log D(x_real)] − E[log(1 − D(x_fake))]
    """
    eps = 1e-8
    real_loss = -sum(math.log(d + eps) for d in d_real) / len(d_real)
    fake_loss = -sum(math.log(1 - d + eps) for d in d_fake) / len(d_fake)
    return real_loss + fake_loss

def gan_generator_loss(d_fake: List[float]) -> float:
    """
    §20 — Generator loss (non-saturating variant).
    L_G = −E[log D(G(z))]
    """
    eps = 1e-8
    return -sum(math.log(d + eps) for d in d_fake) / len(d_fake)

def gan_minimax_value(d_real: List[float],
                       d_fake: List[float]) -> float:
    """§20 eq 20.81 — v(g,d) = E[log D(x)] + E[log(1−D(G(z)))]"""
    eps = 1e-8
    return (sum(math.log(d + eps) for d in d_real) / len(d_real) +
            sum(math.log(1 - d + eps) for d in d_fake) / len(d_fake))


# ─────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing dl_book_implementations2.py ...")

    # §4.3 gradient descent
    p = {"w": 2.0}; g = {"w": 1.0}
    p = gradient_descent_step(p, g, lr=0.1)
    assert p["w"] == 1.9, p["w"]
    print("  ✓ §4.3  Gradient Descent")

    # §4.4 KKT
    assert kkt_satisfied([0.0], [(0.0, -0.5)])
    print("  ✓ §4.4  KKT conditions")

    # §5.4 bias-variance
    preds = [[1.0, 2.0, 3.0], [1.1, 1.9, 3.1]]
    tgts  = [1.0, 2.0, 3.0]
    bv = bias_variance_decompose(preds, tgts)
    assert bv["total_mse"] >= 0
    print("  ✓ §5.4  Bias-Variance decomposition")

    # §5.4 k-fold
    splits = k_fold_split(list(range(10)), k=5)
    assert len(splits) == 5 and len(splits[0][0]) == 8
    print("  ✓ §5.4  k-Fold cross-validation")

    # §5.5 MLE Gaussian
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mu, sigma = mle_gaussian(data)
    assert abs(mu - 3.0) < 1e-9
    print("  ✓ §5.5  MLE Gaussian")

    # §6.3 activations
    assert relu(-1.0) == 0.0 and relu(2.0) == 2.0
    assert abs(sigmoid(0.0) - 0.5) < 1e-9
    sm = softmax([1.0, 2.0, 3.0])
    assert abs(sum(sm) - 1.0) < 1e-9
    print("  ✓ §6.3  Activation functions (ReLU, sigmoid, softmax)")

    # §6.5 MLP backprop
    mlp = SimpleMLP(2, 4)
    out, cache = mlp.forward([0.5, -0.3])
    mlp.backward(cache, 1.0, lr=0.01)
    print("  ✓ §6.5  Backpropagation (SimpleMLP)")

    # §7.1 L1/L2
    params = {"w": 2.0, "b": -1.0}
    assert l2_penalty(params, 0.1) == 0.1/2 * (4+1)
    print("  ✓ §7.1  L1/L2 regularisation")

    # §7.4 augmentation
    aug = augment_text("hello world", ops=["swap","lowercase"])
    assert len(aug) >= 2
    print("  ✓ §7.4  Dataset augmentation")

    # §8.5.1 AdaGrad
    p = {"w": 1.0}; ada = AdaGrad(lr=0.1)
    p = ada.step(p, {"w": 0.5})
    assert p["w"] < 1.0
    print("  ✓ §8.5.1 AdaGrad")

    # §8.7.3 Polyak averaging
    pa = PolyakAveraging(alpha=0.9)
    pa.update({"w": 1.0}); pa.update({"w": 2.0})
    avg = pa.averaged_params()
    assert 1.0 < avg["w"] < 2.0
    print("  ✓ §8.7.3 Polyak averaging")

    # grad clipping
    grads = {"w": 10.0, "b": 10.0}
    clipped = clip_gradients_by_norm(grads, max_norm=1.0)
    norm = math.sqrt(sum(v**2 for v in clipped.values()))
    assert abs(norm - 1.0) < 1e-6
    print("  ✓ §8.2.4 Gradient clipping")

    # §10.10 LSTM
    ws = init_lstm_weights(3, 4)
    h, c = lstm_cell_forward([0.1,0.2,0.3], [0.0]*4, [0.0]*4, **ws)
    assert len(h) == 4 and len(c) == 4
    print("  ✓ §10.10 LSTM cell forward pass")

    # §14.5 Denoising AE
    x = [1.0, 2.0, 3.0]
    x_noisy = corrupt_masking(x, 0.3)
    loss = dae_reconstruction_loss(x, x)
    assert loss == 0.0
    print("  ✓ §14.5 Denoising autoencoder corruption + loss")

    # §15.2 transfer learning
    tlr = TransferLearningRegistry()
    tlr.register("encoder", {"w": 1.0}, frozen=True)
    tlr.register("head",    {"w": 0.5}, frozen=False)
    tp = tlr.trainable_params()
    assert "encoder.w" not in tp and "head.w" in tp
    print("  ✓ §15.2 Transfer learning layer freezing")

    # §17.3 MCMC
    log_p = lambda x: -0.5 * sum(v**2 for v in x)   # standard normal
    samples = metropolis_hastings(log_p, [0.0, 0.0], n_samples=200, burn_in=50)
    assert len(samples) == 200
    print("  ✓ §17.3 MCMC Metropolis-Hastings")

    # §17.4 Gibbs
    samplers = [lambda s: random.gauss(0, 1), lambda s: random.gauss(0, 1)]
    gs = gibbs_sampling(samplers, [0.0, 0.0], n_samples=100, burn_in=10)
    assert len(gs) == 100
    print("  ✓ §17.4 Gibbs Sampling")

    # §20 GAN losses
    d_r = [0.9, 0.85]; d_f = [0.1, 0.15]
    dl = gan_discriminator_loss(d_r, d_f)
    gl = gan_generator_loss(d_f)
    assert dl > 0 and gl > 0
    print("  ✓ §20   GAN minimax losses")

    print("\nAll tests passed.")
