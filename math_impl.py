"""
Mathematics for Machine Learning — full implementation of gaps across all 6 books.
Covers: Linear algebra, calculus, probability, optimization, information theory,
        numerical methods, and ML-specific math not yet in EliteOmni.
"""

import math, random, re, time, json, os
from collections import Counter, defaultdict
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# LINEAR ALGEBRA (MML Ch2, Raschka Ch3, Goodfellow Ch2)
# ─────────────────────────────────────────────────────────────────────────────

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_mul(A, B):
    rows, cols, inner = len(A), len(B[0]), len(B)
    return [[sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]

def mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def mat_scalar(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]

def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def vec_norm(a, p=2):
    if p == 1: return sum(abs(x) for x in a)
    if p == 2: return math.sqrt(sum(x**2 for x in a))
    return max(abs(x) for x in a)

def vec_normalize(a):
    n = vec_norm(a)
    return [x / max(n, 1e-9) for x in a]

def vec_outer(a, b):
    return [[x * y for y in b] for x in a]

def mat_identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def mat_trace(A):
    return sum(A[i][i] for i in range(min(len(A), len(A[0]))))

def mat_frobenius_norm(A):
    return math.sqrt(sum(A[i][j]**2 for i in range(len(A)) for j in range(len(A[0]))))

# Gaussian elimination — solve Ax=b (Goodfellow Ch2, MML Ch2)
def gaussian_elimination(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12: continue
        for row in range(col + 1, n):
            f = M[row][col] / M[col][col]
            M[row] = [M[row][j] - f * M[col][j] for j in range(n + 1)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / max(M[i][i], 1e-12)
    return x

# LU decomposition (MML Ch2)
def lu_decompose(A):
    n = len(A)
    L = mat_identity(n)
    U = [row[:] for row in A]
    for col in range(n):
        for row in range(col + 1, n):
            if abs(U[col][col]) < 1e-12: continue
            f = U[row][col] / U[col][col]
            L[row][col] = f
            U[row] = [U[row][j] - f * U[col][j] for j in range(n)]
    return L, U

# Eigenvalue estimation — power iteration (MML Ch4, Goodfellow Ch2)
def power_iteration(A, n_iter=100):
    n = len(A)
    v = vec_normalize([random.gauss(0, 1) for _ in range(n)])
    for _ in range(n_iter):
        Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        eigenvalue = vec_norm(Av)
        v = vec_normalize(Av)
    return eigenvalue, v

# SVD approximation via power iteration (MML Ch4, Goodfellow Ch2)
def svd_dominant(A, n_iter=50):
    AT = mat_transpose(A)
    ATA = mat_mul(AT, A)
    sigma_sq, v = power_iteration(ATA, n_iter)
    sigma = math.sqrt(max(sigma_sq, 0))
    Av = [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]
    u = vec_normalize(Av)
    return sigma, u, v

# PCA via covariance + power iteration (MML Ch10, Goodfellow Ch2)
def pca(data: list, n_components: int = 2) -> dict:
    n, d = len(data), len(data[0])
    mean = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
    centered = [[data[i][j] - mean[j] for j in range(d)] for i in range(n)]
    CT = mat_transpose(centered)
    cov = mat_scalar(mat_mul(CT, centered), 1.0 / max(n - 1, 1))
    components, explained = [], []
    deflated = [row[:] for row in cov]
    for _ in range(min(n_components, d)):
        eigenval, eigenvec = power_iteration(deflated, n_iter=100)
        components.append(eigenvec)
        explained.append(eigenval)
        outer = vec_outer(eigenvec, eigenvec)
        deflated = mat_sub(deflated, mat_scalar(outer, eigenval))
    projected = [[vec_dot(centered[i], comp) for comp in components] for i in range(n)]
    total_var = sum(explained) or 1
    return {"components": components, "explained_variance": explained,
            "explained_ratio": [round(e / total_var, 4) for e in explained],
            "projected": projected, "mean": mean}

# Cosine similarity (used everywhere in ML)
def cosine_similarity(a, b):
    dot = vec_dot(a, b)
    return dot / max(vec_norm(a) * vec_norm(b), 1e-9)

# ─────────────────────────────────────────────────────────────────────────────
# CALCULUS & OPTIMIZATION (MML Ch5-7, Goodfellow Ch4-8, Raschka Ch4)
# ─────────────────────────────────────────────────────────────────────────────

# Numerical gradient (MML Ch5)
def numerical_gradient(f, x: list, eps: float = 1e-5) -> list:
    grad = []
    for i in range(len(x)):
        xp = x[:]
        xm = x[:]
        xp[i] += eps
        xm[i] -= eps
        grad.append((f(xp) - f(xm)) / (2 * eps))
    return grad

# Gradient descent variants (MML Ch7, Goodfellow Ch8)
def gradient_descent(f, grad_f, x0: list, lr: float = 0.01,
                     n_iter: int = 1000, tol: float = 1e-6) -> dict:
    x = x0[:]
    history = []
    for i in range(n_iter):
        loss = f(x)
        grad = grad_f(x)
        x = [x[j] - lr * grad[j] for j in range(len(x))]
        history.append(loss)
        if vec_norm(grad) < tol:
            break
    return {"x": x, "loss": f(x), "iterations": i + 1, "history": history[-10:]}

def sgd_with_momentum(grads_sequence: list, lr: float = 0.01,
                      momentum: float = 0.9) -> list:
    """Goodfellow Ch8: SGD with momentum."""
    v = [0.0] * len(grads_sequence[0])
    params = [0.0] * len(grads_sequence[0])
    for grad in grads_sequence:
        v = [momentum * v[i] - lr * grad[i] for i in range(len(v))]
        params = [params[i] + v[i] for i in range(len(params))]
    return params

def adam_step(grad: list, m: list, v: list, t: int,
              lr: float = 1e-3, beta1: float = 0.9,
              beta2: float = 0.999, eps: float = 1e-8) -> tuple:
    """Goodfellow Ch8 + MML Ch7: Adam optimizer step."""
    m = [beta1 * m[i] + (1 - beta1) * grad[i] for i in range(len(grad))]
    v = [beta2 * v[i] + (1 - beta2) * grad[i]**2 for i in range(len(grad))]
    m_hat = [m[i] / (1 - beta1**t) for i in range(len(m))]
    v_hat = [v[i] / (1 - beta2**t) for i in range(len(v))]
    update = [lr * m_hat[i] / (math.sqrt(v_hat[i]) + eps) for i in range(len(grad))]
    return update, m, v

def rmsprop_step(grad: list, cache: list, lr: float = 1e-3,
                 decay: float = 0.9, eps: float = 1e-8) -> tuple:
    """Goodfellow Ch8: RMSProp."""
    cache = [decay * cache[i] + (1 - decay) * grad[i]**2 for i in range(len(grad))]
    update = [lr * grad[i] / (math.sqrt(cache[i]) + eps) for i in range(len(grad))]
    return update, cache

# Newton's method (MML Ch7)
def newtons_method(f, grad_f, hess_f, x0: list, n_iter: int = 20) -> dict:
    x = x0[:]
    for i in range(n_iter):
        g = grad_f(x)
        H = hess_f(x)
        try:
            dx = gaussian_elimination(H, [-gi for gi in g])
            x = [x[j] + dx[j] for j in range(len(x))]
        except Exception:
            break
        if vec_norm(g) < 1e-8:
            break
    return {"x": x, "loss": f(x), "iterations": i + 1}

# Automatic differentiation — forward mode (MML Ch5, Raschka Ch4)
class DualNumber:
    """Forward-mode AD using dual numbers: f(a + bε) = f(a) + f'(a)bε"""
    def __init__(self, val, grad=0.0):
        self.val = val
        self.grad = grad
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val + other.val, self.grad + other.grad)
        return DualNumber(self.val + other, self.grad)
    def __radd__(self, other): return self.__add__(other)
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val * other.val,
                              self.val * other.grad + self.grad * other.val)
        return DualNumber(self.val * other, self.grad * other)
    def __rmul__(self, other): return self.__mul__(other)
    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val - other.val, self.grad - other.grad)
        return DualNumber(self.val - other, self.grad)
    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val / other.val,
                              (self.grad * other.val - self.val * other.grad) / other.val**2)
        return DualNumber(self.val / other, self.grad / other)
    def __pow__(self, n):
        return DualNumber(self.val**n, n * self.val**(n-1) * self.grad)
    def __repr__(self): return f"Dual({self.val:.4f}, grad={self.grad:.4f})"

def auto_diff(f, x: float) -> tuple:
    """Compute f(x) and f'(x) using forward-mode AD."""
    d = f(DualNumber(x, 1.0))
    return d.val, d.grad

# Backpropagation (Raschka Ch4, Goodfellow Ch6)
class SimpleNet:
    """2-layer neural net with backprop — from scratch (Raschka Ch4)."""
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr
        self.W1 = [[random.gauss(0, math.sqrt(2/input_dim)) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.gauss(0, math.sqrt(2/hidden_dim)) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim

    def _relu(self, x): return [max(0.0, v) for v in x]
    def _relu_grad(self, x): return [1.0 if v > 0 else 0.0 for v in x]

    def _softmax(self, x):
        mx = max(x)
        exps = [math.exp(v - mx) for v in x]
        s = sum(exps)
        return [e / s for e in exps]

    def forward(self, x):
        self.x = x
        self.z1 = [sum(x[i] * self.W1[i][j] for i in range(len(x))) + self.b1[j]
                   for j in range(len(self.b1))]
        self.a1 = self._relu(self.z1)
        self.z2 = [sum(self.a1[i] * self.W2[i][j] for i in range(len(self.a1))) + self.b2[j]
                   for j in range(len(self.b2))]
        self.out = self._softmax(self.z2)
        return self.out

    def backward(self, y_true: int):
        # Output delta
        dz2 = self.out[:]
        dz2[y_true] -= 1.0
        # Gradients W2, b2
        dW2 = [[self.a1[i] * dz2[j] for j in range(len(dz2))] for i in range(len(self.a1))]
        db2 = dz2
        # Hidden delta
        da1 = [sum(self.W2[i][j] * dz2[j] for j in range(len(dz2))) for i in range(len(self.a1))]
        dz1 = [da1[i] * self._relu_grad(self.z1)[i] for i in range(len(da1))]
        dW1 = [[self.x[i] * dz1[j] for j in range(len(dz1))] for i in range(len(self.x))]
        db1 = dz1
        # Update
        self.W2 = [[self.W2[i][j] - self.lr * dW2[i][j] for j in range(len(self.W2[0]))] for i in range(len(self.W2))]
        self.b2 = [self.b2[j] - self.lr * db2[j] for j in range(len(self.b2))]
        self.W1 = [[self.W1[i][j] - self.lr * dW1[i][j] for j in range(len(self.W1[0]))] for i in range(len(self.W1))]
        self.b1 = [self.b1[j] - self.lr * db1[j] for j in range(len(self.b1))]
        return -math.log(max(self.out[y_true], 1e-9))

# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY & STATISTICS (MML Ch6, Goodfellow Ch3, DMLS Ch4)
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_pdf(x, mu=0.0, sigma=1.0):
    return math.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))

def gaussian_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def kl_divergence(p: list, q: list) -> float:
    """KL(P||Q) — Goodfellow Ch3, MML Ch6, used in VAEs and RLHF."""
    return sum(pi * math.log2(max(pi, 1e-9) / max(qi, 1e-9))
               for pi, qi in zip(p, q) if pi > 0)

def js_divergence(p: list, q: list) -> float:
    """Jensen-Shannon divergence — symmetric KL (MML Ch6)."""
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def entropy(p: list) -> float:
    """Shannon entropy H(P) — MML Ch6, AIE Ch3."""
    return -sum(pi * math.log2(max(pi, 1e-9)) for pi in p if pi > 0)

def cross_entropy(p: list, q: list) -> float:
    """Cross entropy H(P,Q) — loss function for classification (Raschka Ch4)."""
    return -sum(pi * math.log2(max(qi, 1e-9)) for pi, qi in zip(p, q))

def mutual_information(joint: list, p: list, q: list) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y) — MML Ch6, feature selection."""
    flat_joint = [v for row in joint for v in row]
    return entropy(p) + entropy(q) - entropy(flat_joint)

def bayes_update(prior: list, likelihood: list) -> list:
    """Bayes theorem: posterior ∝ likelihood × prior (MML Ch6)."""
    unnorm = [l * p for l, p in zip(likelihood, prior)]
    total = sum(unnorm)
    return [u / max(total, 1e-9) for u in unnorm]

def monte_carlo_estimate(f, n_samples: int = 10000,
                          x_range: tuple = (0, 1)) -> float:
    """Monte Carlo integration (Goodfellow Ch3, MML Ch6)."""
    a, b = x_range
    samples = [f(random.uniform(a, b)) for _ in range(n_samples)]
    return (b - a) * sum(samples) / n_samples

def bootstrap_confidence_interval(data: list, n_boot: int = 1000,
                                   ci: float = 0.95) -> tuple:
    """Bootstrap CI — DMLS Ch4, AIE Ch3 (sample size / significance)."""
    means = []
    n = len(data)
    for _ in range(n_boot):
        sample = [data[random.randint(0, n-1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = int((1 - ci) / 2 * n_boot)
    hi = int((1 + ci) / 2 * n_boot)
    return round(means[lo], 4), round(means[hi], 4)

def t_test_two_sample(a: list, b: list) -> dict:
    """Welch's t-test — AIE Ch3, DMLS Ch4 (A/B test significance)."""
    na, nb = len(a), len(b)
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma)**2 for x in a) / max(na - 1, 1)
    vb = sum((x - mb)**2 for x in b) / max(nb - 1, 1)
    se = math.sqrt(va / na + vb / nb)
    t = (ma - mb) / max(se, 1e-9)
    df = (va/na + vb/nb)**2 / max((va/na)**2/(na-1) + (vb/nb)**2/(nb-1), 1e-9)
    p_approx = 2 * (1 - gaussian_cdf(abs(t)))
    return {"t_stat": round(t, 4), "p_value": round(p_approx, 4),
            "significant": p_approx < 0.05, "df": round(df, 1),
            "mean_a": round(ma, 4), "mean_b": round(mb, 4)}

# ─────────────────────────────────────────────────────────────────────────────
# INFORMATION THEORY (MML Ch6, Goodfellow Ch3, AIE Ch3)
# ─────────────────────────────────────────────────────────────────────────────

def perplexity_from_probs(probs: list) -> float:
    """Perplexity = 2^H(p) — AIE Ch3, language model evaluation."""
    h = entropy(probs)
    return round(2 ** h, 4)

def bits_back_coding_gain(latent_dim: int, prior_bits: float) -> float:
    """Bits-back argument — VAE coding efficiency (Goodfellow Ch20)."""
    return max(0.0, prior_bits - latent_dim * math.log2(math.e))

def information_gain(parent_entropy: float, children: list,
                     weights: list) -> float:
    """Information gain for decision trees — DMLS Ch4."""
    child_entropy = sum(w * entropy(c) for w, c in zip(weights, children))
    return parent_entropy - child_entropy

def pointwise_mutual_info(p_xy: float, p_x: float, p_y: float) -> float:
    """PMI(x,y) = log(P(x,y) / P(x)P(y)) — used in word embeddings (Alammar Ch3)."""
    return math.log2(max(p_xy, 1e-9) / max(p_x * p_y, 1e-9))

# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL METHODS (MML Ch8, Goodfellow Ch4)
# ─────────────────────────────────────────────────────────────────────────────

def numerical_jacobian(f, x: list, eps: float = 1e-5) -> list:
    """Jacobian matrix via finite differences (MML Ch5)."""
    fx = f(x)
    m = len(fx) if isinstance(fx, list) else 1
    n = len(x)
    J = []
    for i in range(n):
        xp = x[:]
        xp[i] += eps
        fxp = f(xp)
        if isinstance(fxp, list):
            J.append([(fxp[j] - fx[j]) / eps for j in range(m)])
        else:
            J.append([(fxp - fx) / eps])
    return J

def conjugate_gradient(A, b, n_iter: int = 100, tol: float = 1e-6) -> list:
    """Conjugate gradient solver — MML Ch8, used in second-order optimization."""
    n = len(b)
    x = [0.0] * n
    r = b[:]
    p = r[:]
    rs_old = vec_dot(r, r)
    for _ in range(n_iter):
        Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]
        alpha = rs_old / max(vec_dot(p, Ap), 1e-12)
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * Ap[i] for i in range(n)]
        rs_new = vec_dot(r, r)
        if math.sqrt(rs_new) < tol:
            break
        beta = rs_new / max(rs_old, 1e-12)
        p = [r[i] + beta * p[i] for i in range(n)]
        rs_old = rs_new
    return x

def line_search_backtrack(f, x: list, grad: list, direction: list,
                           alpha: float = 1.0, rho: float = 0.5,
                           c: float = 1e-4, max_iter: int = 50) -> float:
    """Armijo backtracking line search — MML Ch7."""
    fx = f(x)
    slope = vec_dot(grad, direction)
    for _ in range(max_iter):
        x_new = [x[i] + alpha * direction[i] for i in range(len(x))]
        if f(x_new) <= fx + c * alpha * slope:
            break
        alpha *= rho
    return alpha

# ─────────────────────────────────────────────────────────────────────────────
# ML-SPECIFIC MATH (DMLS, AIE, LLM Handbook, Hands-On LLMs)
# ─────────────────────────────────────────────────────────────────────────────

def softmax(x: list, temperature: float = 1.0) -> list:
    """Softmax with temperature — Raschka Ch4, Alammar Ch2."""
    x_scaled = [v / max(temperature, 1e-9) for v in x]
    mx = max(x_scaled)
    exps = [math.exp(v - mx) for v in x_scaled]
    s = sum(exps)
    return [e / s for e in exps]

def log_softmax(x: list) -> list:
    """Numerically stable log-softmax (Goodfellow Ch6, Raschka Ch4)."""
    mx = max(x)
    log_sum = math.log(sum(math.exp(v - mx) for v in x)) + mx
    return [v - log_sum for v in x]

def gelu(x: float) -> float:
    """GELU activation — used in GPT/BERT (Raschka Ch4, Alammar Ch5)."""
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def swiglu(x: float, gate: float) -> float:
    """SwiGLU — used in LLaMA/Mistral (LLM Handbook Ch4)."""
    return x * gate * (1 / (1 + math.exp(-gate)))

def rope_encoding(pos: int, dim: int, base: float = 10000.0) -> list:
    """RoPE positional encoding — used in Mistral/LLaMA (LLM Handbook Ch4, Raschka Ch5)."""
    result = []
    for i in range(0, dim, 2):
        theta = pos / (base ** (i / dim))
        result.append(math.cos(theta))
        result.append(math.sin(theta))
    return result[:dim]

def attention_entropy(attn_weights: list) -> float:
    """Attention entropy — measures focus vs diffusion (Alammar Ch5, AIE Ch3)."""
    return entropy(attn_weights)

def label_smoothing_loss(logits: list, target: int, smoothing: float = 0.1) -> float:
    """Label smoothing cross-entropy — DMLS Ch4, LLM Handbook Ch5."""
    n = len(logits)
    log_probs = log_softmax(logits)
    smooth_target = [smoothing / (n - 1)] * n
    smooth_target[target] = 1.0 - smoothing
    return -sum(smooth_target[i] * log_probs[i] for i in range(n))

def focal_loss(prob: float, target: int, gamma: float = 2.0) -> float:
    """Focal loss for class imbalance — DMLS Ch4."""
    p_t = prob if target == 1 else 1 - prob
    return -((1 - p_t) ** gamma) * math.log(max(p_t, 1e-9))

def cosine_lr_schedule(step: int, total_steps: int,
                        base_lr: float, min_lr: float = 0.0,
                        warmup_steps: int = 0) -> float:
    """Cosine LR schedule with warmup — LLM Handbook Ch6, Raschka Ch5."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def weight_decay_update(params: list, grads: list, lr: float,
                         wd: float = 0.01) -> list:
    """AdamW-style weight decay (decoupled) — LLM Handbook Ch5, Raschka Ch5."""
    return [p - lr * (g + wd * p) for p, g in zip(params, grads)]

def perplexity_from_loss(avg_nll_loss: float) -> float:
    """PPL = exp(avg NLL) — standard LM eval metric (AIE Ch3, LLM Handbook Ch7)."""
    return round(math.exp(avg_nll_loss), 4)

def bits_per_byte(avg_nll_loss: float) -> float:
    """BPB = NLL / ln(2) — alternative LM metric (LLM Handbook Ch7)."""
    return round(avg_nll_loss / math.log(2), 4)

def flesch_kincaid_grade(text: str) -> float:
    """Readability score — used in output quality eval (AIE Ch3)."""
    sentences = max(len(re.findall(r'[.!?]', text)), 1)
    words = max(len(text.split()), 1)
    syllables = sum(max(len(re.findall(r'[aeiouAEIOU]', w)), 1) for w in text.split())
    return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

def token_fertility(source_text: str, tokenized: list) -> float:
    """Tokens per word — tokenizer efficiency metric (LLM Handbook Ch3)."""
    words = len(source_text.split())
    return round(len(tokenized) / max(words, 1), 3)

# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing math_impl...\n")

    # Linear algebra
    A = [[2,1],[1,3]]
    B = [[1,0],[0,1]]
    assert mat_mul(A, B) == A, "mat_mul identity failed"
    assert vec_norm([3,4]) == 5.0, "vec_norm failed"
    x = gaussian_elimination([[2,1],[1,3]], [5,10])
    assert abs(x[0] - 1.0) < 0.01 and abs(x[1] - 3.0) < 0.01, "gaussian elim failed"
    L, U = lu_decompose([[4,3],[6,3]])
    assert abs(L[1][0] - 1.5) < 0.01, "LU decompose failed"
    data = [[random.gauss(0,1), random.gauss(0,1)] for _ in range(50)]
    pca_result = pca(data, n_components=2)
    assert len(pca_result["components"]) == 2, "PCA failed"
    print("  ✓ Linear algebra (MML Ch2, Goodfellow Ch2, Raschka Ch3)")

    # Calculus & optimization
    f  = lambda x: (x[0]-2)**2 + (x[1]-3)**2
    gf = lambda x: [2*(x[0]-2), 2*(x[1]-3)]
    res = gradient_descent(f, gf, [0.0, 0.0], lr=0.1, n_iter=200)
    assert abs(res["x"][0] - 2.0) < 0.1, "GD failed"
    val, grad = auto_diff(lambda x: x**3 + 2*x, 3.0)
    assert abs(val - 33.0) < 0.01 and abs(grad - 29.0) < 0.01, "AutoDiff failed"
    net = SimpleNet(4, 8, 3, lr=0.05)
    x_in = [0.5, -0.3, 0.8, 0.1]
    out = net.forward(x_in)
    assert abs(sum(out) - 1.0) < 0.001, "SimpleNet forward failed"
    loss = net.backward(1)
    assert loss > 0, "SimpleNet backward failed"
    print("  ✓ Calculus & optimization (MML Ch5-7, Goodfellow Ch4-8, Raschka Ch4)")

    # Probability & statistics
    p = [0.3, 0.5, 0.2]
    q = [0.25, 0.45, 0.3]
    assert kl_divergence(p, q) >= 0, "KL divergence failed"
    assert 0 <= js_divergence(p, q) <= 1, "JS divergence failed"
    assert abs(cross_entropy(p, p) - entropy(p)) < 0.001, "Cross entropy failed"
    posterior = bayes_update([0.5, 0.5], [0.8, 0.3])
    assert abs(sum(posterior) - 1.0) < 0.001, "Bayes update failed"
    ci = bootstrap_confidence_interval([random.gauss(5,1) for _ in range(100)])
    assert ci[0] < ci[1], "Bootstrap CI failed"
    tt = t_test_two_sample([1,2,3,4,5], [3,4,5,6,7])
    assert "p_value" in tt, "t-test failed"
    print("  ✓ Probability & statistics (MML Ch6, Goodfellow Ch3, DMLS Ch4)")

    # Information theory
    probs = [0.25, 0.25, 0.25, 0.25]
    assert abs(entropy(probs) - 2.0) < 0.001, "Entropy failed"
    assert perplexity_from_probs(probs) == 4.0, "Perplexity failed"
    ig = information_gain(entropy([0.5,0.5]), [[0.9,0.1],[0.1,0.9]], [0.5,0.5])
    assert ig > 0, "Information gain failed"
    print("  ✓ Information theory (MML Ch6, Goodfellow Ch3, AIE Ch3)")

    # Numerical methods
    A2 = [[4,1],[1,3]]
    b2 = [1,2]
    x2 = conjugate_gradient(A2, b2)
    assert abs(x2[0]*4 + x2[1]*1 - 1.0) < 0.01, "Conjugate gradient failed"
    print("  ✓ Numerical methods (MML Ch8, Goodfellow Ch4)")

    # ML-specific math
    sm = softmax([1.0, 2.0, 3.0])
    assert abs(sum(sm) - 1.0) < 0.001, "Softmax failed"
    lsm = log_softmax([1.0, 2.0, 3.0])
    assert all(v <= 0 for v in lsm), "Log-softmax failed"
    assert abs(gelu(0.0)) < 0.001, "GELU failed"
    rope = rope_encoding(5, 8)
    assert len(rope) == 8, "RoPE failed"
    lsl = label_smoothing_loss([1.0,2.0,3.0], 2, smoothing=0.1)
    assert lsl > 0, "Label smoothing failed"
    fl = focal_loss(0.9, 1, gamma=2.0)
    assert fl < 0.1, "Focal loss failed"
    lr = cosine_lr_schedule(50, 100, 1e-3, warmup_steps=10)
    assert 0 < lr < 1e-3, "Cosine LR failed"
    ppl = perplexity_from_loss(2.3)
    assert ppl > 1, "PPL from loss failed"
    fk = flesch_kincaid_grade("The cat sat on the mat. It was a good day.")
    assert isinstance(fk, float), "Flesch-Kincaid failed"
    print("  ✓ ML-specific math (all 6 books)")

    print("\n✅ ALL MATH TESTS PASSED")
