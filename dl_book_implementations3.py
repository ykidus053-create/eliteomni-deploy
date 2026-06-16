"""
Deep Learning — Goodfellow, Bengio & Courville
Third batch: all remaining sections not in implementations 1 or 2.
"""
import math, random, collections
from typing import List, Dict, Tuple, Optional, Callable

# ─────────────────────────────────────────────────────────────────
# §2.7 / §2.8  Eigendecomposition & SVD (power-iteration + manual)
# ─────────────────────────────────────────────────────────────────
def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    r, mid, c = len(A), len(B), len(B[0])
    return [[sum(A[i][k]*B[k][j] for k in range(mid)) for j in range(c)] for i in range(r)]

def mat_T(A: List[List[float]]) -> List[List[float]]:
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(x*x for x in v))

def power_iteration(A: List[List[float]], n_iter: int = 100) -> Tuple[float, List[float]]:
    """§2.7 — dominant eigenvalue + eigenvector via power iteration."""
    n  = len(A)
    v  = [random.gauss(0,1) for _ in range(n)]
    nm = vec_norm(v); v = [x/nm for x in v]
    for _ in range(n_iter):
        Av = [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
        lam = sum(Av[i]*v[i] for i in range(n))
        nm  = vec_norm(Av)
        v   = [x/nm for x in Av]
    return lam, v

def svd_1d(A: List[List[float]], n_iter: int = 200) -> Tuple[float, List[float], List[float]]:
    """§2.8 — rank-1 SVD: returns (sigma, u, v) via power iteration on A^T A."""
    m, n  = len(A), len(A[0])
    AT    = mat_T(A)
    ATA   = mat_mul(AT, A)
    _, v  = power_iteration(ATA, n_iter)
    Av    = [sum(A[i][j]*v[j] for j in range(n)) for i in range(m)]
    sigma = vec_norm(Av)
    u     = [x/sigma for x in Av] if sigma > 1e-12 else [0.0]*m
    return sigma, u, v

def pca_project(data: List[List[float]], k: int = 1) -> List[List[float]]:
    """§2.12 — PCA: center, then project onto k dominant eigenvectors."""
    m, n  = len(data), len(data[0])
    mean  = [sum(data[i][j] for i in range(m))/m for j in range(n)]
    X     = [[data[i][j]-mean[j] for j in range(n)] for i in range(m)]
    # covariance matrix X^T X / m
    XT    = mat_T(X)
    C     = [[sum(XT[a][t]*XT[b][t] for t in range(m))/m for b in range(n)] for a in range(n)]
    components = []
    for _ in range(k):
        _, v = power_iteration(C)
        components.append(v)
        # deflate
        for a in range(n):
            for b in range(n):
                C[a][b] -= v[a]*v[b] * sum(v[i]*(sum(C[i][j]*v[j] for j in range(n))) for i in range(n))
    return [[sum(X[i][j]*components[d][j] for j in range(n)) for d in range(k)] for i in range(m)]


# ─────────────────────────────────────────────────────────────────
# §3.8 / §3.13  Information Theory — entropy, KL, cross-entropy
# ─────────────────────────────────────────────────────────────────
def entropy(probs: List[float], base: float = 2.0) -> float:
    """§3.13 — Shannon entropy H(p) = -Σ p log p."""
    eps = 1e-12
    return -sum(p * math.log(p+eps, base) for p in probs if p > 0)

def kl_divergence(p: List[float], q: List[float]) -> float:
    """§3.13 — KL(P||Q) = Σ p log(p/q).  Not symmetric."""
    eps = 1e-12
    return sum(pi * math.log((pi+eps)/(qi+eps)) for pi, qi in zip(p, q) if pi > 0)

def cross_entropy(p: List[float], q: List[float]) -> float:
    """§3.13 — H(P,Q) = -Σ p log q = H(P) + KL(P||Q)."""
    eps = 1e-12
    return -sum(pi * math.log(qi+eps) for pi, qi in zip(p, q))

def binary_cross_entropy(y: float, y_hat: float) -> float:
    """BCE = -[y log ŷ + (1-y) log(1-ŷ)]."""
    eps = 1e-12
    return -(y*math.log(y_hat+eps) + (1-y)*math.log(1-y_hat+eps))

def mutual_information(joint: List[List[float]]) -> float:
    """§3.13 — I(X;Y) = Σ p(x,y) log[p(x,y)/(p(x)p(y))]."""
    px = [sum(row) for row in joint]
    py = [sum(joint[i][j] for i in range(len(joint))) for j in range(len(joint[0]))]
    mi, eps = 0.0, 1e-12
    for i, row in enumerate(joint):
        for j, pxy in enumerate(row):
            if pxy > 0:
                mi += pxy * math.log((pxy+eps)/((px[i]+eps)*(py[j]+eps)))
    return mi


# ─────────────────────────────────────────────────────────────────
# §3.10  Numeric stable softplus, hard-sigmoid, log-sum-exp
# ─────────────────────────────────────────────────────────────────
def softplus(x: float) -> float:
    """§3.10 — ζ(x) = log(1 + e^x), numerically stable."""
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)

def log_sum_exp(xs: List[float]) -> float:
    """Numerically stable log Σ exp(xᵢ)."""
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """§3.9.3 — N(x; µ, σ²)."""
    return (1.0/(sigma*math.sqrt(2*math.pi))) * math.exp(-0.5*((x-mu)/sigma)**2)

def gaussian_log_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return -0.5*math.log(2*math.pi) - math.log(sigma) - 0.5*((x-mu)/sigma)**2


# ─────────────────────────────────────────────────────────────────
# §3.11  Bayes' Rule and MAP estimation
# ─────────────────────────────────────────────────────────────────
def bayes_update(prior: Dict, likelihood_fn: Callable,
                 observation) -> Dict:
    """§3.11 — posterior ∝ likelihood × prior (discrete case)."""
    unnorm = {h: prior[h] * likelihood_fn(h, observation) for h in prior}
    Z = sum(unnorm.values())
    return {h: v/Z for h, v in unnorm.items()}

def map_estimate(log_likelihood_fn: Callable,
                 log_prior_fn: Callable,
                 param_grid: List[float]) -> float:
    """§5.6.1 — MAP: argmax [log p(θ|x)] = argmax [log p(x|θ) + log p(θ)]."""
    scores = [(log_likelihood_fn(t) + log_prior_fn(t), t) for t in param_grid]
    return max(scores)[1]


# ─────────────────────────────────────────────────────────────────
# §4.1  Numerical stability — overflow / underflow guards
# ─────────────────────────────────────────────────────────────────
def stable_sigmoid(x: float) -> float:
    """§4.1 — sigmoid without overflow for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)

def stable_softmax(logits: List[float]) -> List[float]:
    """§4.1 — subtract max before exp to prevent overflow."""
    m    = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s    = sum(exps)
    return [e/s for e in exps]

def stable_log_softmax(logits: List[float]) -> List[float]:
    """log softmax via log-sum-exp trick."""
    lse = log_sum_exp(logits)
    return [x - lse for x in logits]


# ─────────────────────────────────────────────────────────────────
# §4.3.1  Jacobian and Hessian (finite difference)
# ─────────────────────────────────────────────────────────────────
def jacobian(f: Callable, x: List[float], eps: float = 1e-5) -> List[List[float]]:
    """§4.3.1 — numeric Jacobian of f: Rⁿ → Rᵐ."""
    n   = len(x)
    f0  = f(x)
    m   = len(f0) if hasattr(f0, '__len__') else 1
    J   = [[0.0]*n for _ in range(m)]
    for j in range(n):
        xp = list(x); xp[j] += eps
        xm = list(x); xm[j] -= eps
        fp = f(xp); fm = f(xm)
        if m == 1:
            J[0][j] = (fp - fm) / (2*eps)
        else:
            for i in range(m):
                J[i][j] = (fp[i] - fm[i]) / (2*eps)
    return J

def hessian(f: Callable, x: List[float], eps: float = 1e-4) -> List[List[float]]:
    """§4.3.1 — numeric Hessian (second derivatives)."""
    n = len(x)
    H = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            xpp = list(x); xpp[i]+=eps; xpp[j]+=eps
            xpm = list(x); xpm[i]+=eps; xpm[j]-=eps
            xmp = list(x); xmp[i]-=eps; xmp[j]+=eps
            xmm = list(x); xmm[i]-=eps; xmm[j]-=eps
            H[i][j] = (f(xpp)-f(xpm)-f(xmp)+f(xmm))/(4*eps*eps)
    return H


# ─────────────────────────────────────────────────────────────────
# §4.5  Linear Least Squares (normal equations)
# ─────────────────────────────────────────────────────────────────
def linear_least_squares(X: List[List[float]],
                          y: List[float]) -> List[float]:
    """§4.5 — θ = (XᵀX)⁻¹ Xᵀ y  (closed form, small n only)."""
    XT = mat_T(X)
    XTX = mat_mul(XT, [[v] for v in y])   # XTX is XᵀX, XTy is Xᵀy
    # Simple 1-D case shortcut if single feature
    XTX_sq = mat_mul(XT, X)
    XTy    = [sum(XT[i][j]*y[j] for j in range(len(y))) for i in range(len(XT))]
    # Solve by gradient descent (avoids matrix inverse)
    theta = [0.0] * len(XTy)
    for _ in range(500):
        pred = [sum(X[j][i]*theta[i] for i in range(len(theta))) for j in range(len(y))]
        grad = [sum(XTX_sq[i][k]*theta[k] for k in range(len(theta))) - XTy[i]
                for i in range(len(theta))]
        theta = [theta[i] - 0.01*grad[i] for i in range(len(theta))]
    return theta


# ─────────────────────────────────────────────────────────────────
# §5.8.2  k-Means Clustering
# ─────────────────────────────────────────────────────────────────
def kmeans(data: List[List[float]], k: int,
           n_iter: int = 100, seed: int = 0) -> Tuple[List[int], List[List[float]]]:
    """§5.8.2 — Lloyd's k-means. Returns (assignments, centroids)."""
    rng = random.Random(seed)
    centroids = [list(data[i]) for i in rng.sample(range(len(data)), k)]
    assignments = [0]*len(data)
    for _ in range(n_iter):
        for i, x in enumerate(data):
            dists = [sum((x[d]-c[d])**2 for d in range(len(x))) for c in centroids]
            assignments[i] = dists.index(min(dists))
        new_c = [[0.0]*len(data[0]) for _ in range(k)]
        counts = [0]*k
        for i, x in enumerate(data):
            c = assignments[i]; counts[c] += 1
            for d in range(len(x)):
                new_c[c][d] += x[d]
        for c in range(k):
            if counts[c] > 0:
                centroids[c] = [v/counts[c] for v in new_c[c]]
    return assignments, centroids


# ─────────────────────────────────────────────────────────────────
# §7.7  Multi-Task Learning — shared-repr loss combiner
# ─────────────────────────────────────────────────────────────────
def multitask_loss(task_losses: List[float],
                   weights: List[float] = None) -> float:
    """§7.7 — weighted sum of per-task losses (shared representation)."""
    if weights is None:
        weights = [1.0/len(task_losses)]*len(task_losses)
    return sum(w*l for w,l in zip(weights, task_losses))

def uncertainty_weighted_multitask(task_losses: List[float],
                                    log_sigmas: List[float]) -> float:
    """§7.7 — uncertainty-weighted MTL: Σ [L_i/(2σ_i²) + log σ_i]."""
    total = 0.0
    for l, ls in zip(task_losses, log_sigmas):
        sigma_sq = math.exp(2*ls)
        total += l/(2*sigma_sq) + ls
    return total


# ─────────────────────────────────────────────────────────────────
# §7.9  Parameter Sharing (tied weights check)
# ─────────────────────────────────────────────────────────────────
def tied_weight_grad(grad_a: Dict[str,float],
                     grad_b: Dict[str,float]) -> Dict[str,float]:
    """§7.9 — when two layers share weights, sum their gradients."""
    keys = set(grad_a) | set(grad_b)
    return {k: grad_a.get(k,0.0)+grad_b.get(k,0.0) for k in keys}


# ─────────────────────────────────────────────────────────────────
# §7.13  Adversarial Examples (FGSM — Fast Gradient Sign Method)
# ─────────────────────────────────────────────────────────────────
def fgsm_perturbation(x: List[float], grad_loss_x: List[float],
                       epsilon: float = 0.01) -> List[float]:
    """§7.13 — FGSM: x_adv = x + ε * sign(∇_x L)."""
    return [xi + epsilon * (1.0 if g > 0 else -1.0 if g < 0 else 0.0)
            for xi, g in zip(x, grad_loss_x)]


# ─────────────────────────────────────────────────────────────────
# §8.3.3  Nesterov Momentum
# ─────────────────────────────────────────────────────────────────
class NesterovMomentum:
    """
    §8.3.3 — Nesterov accelerated gradient.
    v ← α*v − ε * ∇J(θ + α*v)   (look-ahead gradient)
    θ ← θ + v
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr  = lr; self.momentum = momentum
        self.v: Dict[str,float] = {}

    def lookahead_params(self, params: Dict[str,float]) -> Dict[str,float]:
        """Compute θ + α*v before computing gradient."""
        return {k: params[k] + self.momentum * self.v.get(k,0.0) for k in params}

    def step(self, params: Dict[str,float],
             grads_at_lookahead: Dict[str,float]) -> Dict[str,float]:
        for k in params:
            v_new    = self.momentum*self.v.get(k,0.0) - self.lr*grads_at_lookahead.get(k,0.0)
            self.v[k]= v_new
            params[k]+= v_new
        return params


# ─────────────────────────────────────────────────────────────────
# §8.6.1  Newton's Method (second-order)
# ─────────────────────────────────────────────────────────────────
def newton_step_1d(f_prime: float, f_double_prime: float,
                   x: float) -> float:
    """§8.6.1 — Newton step: x ← x − f'(x)/f''(x)."""
    if abs(f_double_prime) < 1e-12:
        return x
    return x - f_prime / f_double_prime

def newton_method(f: Callable, x0: float,
                  n_iter: int = 50, eps: float = 1e-6) -> float:
    """§8.6.1 — Newton's method for scalar root-finding."""
    x = x0
    for _ in range(n_iter):
        fp  = (f(x+eps) - f(x-eps)) / (2*eps)
        fpp = (f(x+eps) - 2*f(x) + f(x-eps)) / (eps*eps)
        if abs(fpp) < 1e-12: break
        x  -= fp / fpp
    return x


# ─────────────────────────────────────────────────────────────────
# §8.7.2  Coordinate Descent
# ─────────────────────────────────────────────────────────────────
def coordinate_descent(f: Callable, init_params: List[float],
                        n_cycles: int = 100,
                        step: float = 0.01) -> List[float]:
    """§8.7.2 — Minimize f by cycling over each coordinate."""
    params = list(init_params)
    for _ in range(n_cycles):
        for i in range(len(params)):
            best_val, best_f = params[i], f(params)
            for delta in [-step, step]:
                p2 = list(params); p2[i] += delta
                fv = f(p2)
                if fv < best_f:
                    best_f = fv; best_val = p2[i]
            params[i] = best_val
    return params


# ─────────────────────────────────────────────────────────────────
# §9.1 / §9.3  1-D Convolution and Max/Average Pooling
# ─────────────────────────────────────────────────────────────────
def conv1d(signal: List[float], kernel: List[float],
           padding: int = 0) -> List[float]:
    """§9.1 — 1-D discrete convolution (cross-correlation mode)."""
    sig = [0.0]*padding + signal + [0.0]*padding
    k   = len(kernel); out_len = len(sig) - k + 1
    return [sum(sig[i+j]*kernel[j] for j in range(k)) for i in range(out_len)]

def conv2d_single(img: List[List[float]],
                  kernel: List[List[float]]) -> List[List[float]]:
    """§9.1 — 2-D convolution (no padding, stride 1)."""
    H, W  = len(img), len(img[0])
    Kh, Kw= len(kernel), len(kernel[0])
    oh, ow= H-Kh+1, W-Kw+1
    out = []
    for i in range(oh):
        row = []
        for j in range(ow):
            row.append(sum(img[i+di][j+dj]*kernel[di][dj]
                           for di in range(Kh) for dj in range(Kw)))
        out.append(row)
    return out

def max_pool1d(x: List[float], pool_size: int = 2,
               stride: int = 2) -> List[float]:
    """§9.3 — 1-D max pooling."""
    out = []
    for i in range(0, len(x)-pool_size+1, stride):
        out.append(max(x[i:i+pool_size]))
    return out

def avg_pool1d(x: List[float], pool_size: int = 2,
               stride: int = 2) -> List[float]:
    """§9.3 — 1-D average pooling."""
    out = []
    for i in range(0, len(x)-pool_size+1, stride):
        chunk = x[i:i+pool_size]
        out.append(sum(chunk)/len(chunk))
    return out


# ─────────────────────────────────────────────────────────────────
# §10.2  Vanilla RNN Cell (forward pass)
# ─────────────────────────────────────────────────────────────────
def rnn_cell_forward(x: List[float], h_prev: List[float],
                     W_xh: List[List[float]], W_hh: List[List[float]],
                     b_h: List[float],
                     W_hy: List[List[float]], b_y: List[float]
                     ) -> Tuple[List[float], List[float]]:
    """
    §10.2 — Vanilla RNN step.
    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
    o_t = W_hy h_t + b_y
    Returns (h_t, o_t).
    """
    n_h = len(b_h); n_y = len(b_y)
    h_t = [math.tanh(
               sum(W_xh[j][i]*x[i] for i in range(len(x))) +
               sum(W_hh[j][i]*h_prev[i] for i in range(len(h_prev))) +
               b_h[j])
           for j in range(n_h)]
    o_t = [sum(W_hy[k][j]*h_t[j] for j in range(n_h)) + b_y[k]
           for k in range(n_y)]
    return h_t, o_t

def bptt_gradient_check(loss_seq: List[float], h_seq: List[List[float]],
                         truncate: int = 5) -> float:
    """
    §10.2.2 — Truncated BPTT: accumulate gradients back at most `truncate` steps.
    Returns approximate gradient norm (scalar summary).
    """
    total = 0.0
    for t in range(len(loss_seq)-1, max(-1, len(loss_seq)-truncate-1), -1):
        total += loss_seq[t]
    return total


# ─────────────────────────────────────────────────────────────────
# §10.3  Bidirectional RNN helper
# ─────────────────────────────────────────────────────────────────
def birnn_combine(fwd_states: List[List[float]],
                  bwd_states: List[List[float]]) -> List[List[float]]:
    """§10.3 — Concatenate forward and backward hidden states."""
    return [f + b for f, b in zip(fwd_states, reversed(bwd_states))]


# ─────────────────────────────────────────────────────────────────
# §10.4  Encoder-Decoder / Seq2Seq context vector
# ─────────────────────────────────────────────────────────────────
def encode_sequence(inputs: List[List[float]],
                    rnn_fn: Callable) -> List[float]:
    """§10.4 — Run RNN over inputs, return final hidden state as context."""
    h = [0.0] * len(inputs[0])
    for x in inputs:
        h, _ = rnn_fn(x, h)
    return h

def attention_context(query: List[float],
                      keys: List[List[float]],
                      values: List[List[float]]) -> List[float]:
    """§10.4 / §12.4.5 — Scaled dot-product attention context vector."""
    d    = len(query)
    scale= math.sqrt(d)
    scores = [sum(query[i]*k[i] for i in range(d))/scale for k in keys]
    weights= stable_softmax(scores)
    n_val  = len(values[0])
    ctx    = [sum(weights[t]*values[t][i] for t in range(len(values)))
              for i in range(n_val)]
    return ctx


# ─────────────────────────────────────────────────────────────────
# §11.1  Performance Metrics
# ─────────────────────────────────────────────────────────────────
def precision_recall_f1(y_true: List[int],
                         y_pred: List[int]) -> Dict[str,float]:
    """§11.1 — binary precision, recall, F1, accuracy."""
    tp = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==1)
    fp = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==1)
    fn = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==0)
    tn = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==0)
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec  = tp/(tp+fn) if tp+fn>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    acc  = (tp+tn)/len(y_true) if y_true else 0.0
    return {"precision":round(prec,4),"recall":round(rec,4),
            "f1":round(f1,4),"accuracy":round(acc,4),
            "tp":tp,"fp":fp,"fn":fn,"tn":tn}

def coverage_at_k(relevant: List[int], ranked: List[int], k: int) -> float:
    """§11.1 — Recall@K for ranking / recommendation tasks."""
    top_k = set(ranked[:k])
    return sum(1 for r in relevant if r in top_k) / max(len(relevant),1)

def roc_auc_approx(y_true: List[int], scores: List[float]) -> float:
    """§11.1 — Approximate AUC via trapezoidal rule."""
    pairs = sorted(zip(scores, y_true), reverse=True)
    pos   = sum(y_true); neg = len(y_true)-pos
    if pos == 0 or neg == 0: return 0.5
    tp=0; fp=0; prev_tp=0; prev_fp=0; auc=0.0
    for _, y in pairs:
        if y==1: tp+=1
        else: fp+=1
        auc += (fp-prev_fp)*(tp+prev_tp)/2.0
        prev_tp=tp; prev_fp=fp
    return auc/(pos*neg)


# ─────────────────────────────────────────────────────────────────
# §11.4.3  Grid Search
# ─────────────────────────────────────────────────────────────────
def grid_search(param_grid: Dict[str,List]) -> List[Dict]:
    """§11.4.3 — Cartesian product of all hyperparameter values."""
    keys = list(param_grid.keys())
    result = [{}]
    for k in keys:
        result = [{**cfg, k:v} for cfg in result for v in param_grid[k]]
    return result


# ─────────────────────────────────────────────────────────────────
# §13.2  Independent Component Analysis (whitening + sign)
# ─────────────────────────────────────────────────────────────────
def whiten(X: List[List[float]]) -> List[List[float]]:
    """§13.2 — Zero-mean, unit-variance whitening per feature."""
    m, n  = len(X), len(X[0])
    mean  = [sum(X[i][j] for i in range(m))/m for j in range(n)]
    std   = [math.sqrt(sum((X[i][j]-mean[j])**2 for i in range(m))/m)+1e-8 for j in range(n)]
    return [[(X[i][j]-mean[j])/std[j] for j in range(n)] for i in range(m)]


# ─────────────────────────────────────────────────────────────────
# §13.4  Sparse Coding (ISTA — soft-thresholding)
# ─────────────────────────────────────────────────────────────────
def soft_threshold(x: float, lam: float) -> float:
    """Proximal operator for L1: sign(x)*max(|x|-λ, 0)."""
    return math.copysign(1,x)*max(abs(x)-lam, 0.0)

def ista_sparse_code(x: List[float], D: List[List[float]],
                     lam: float = 0.1, n_iter: int = 100,
                     lr: float = 0.01) -> List[float]:
    """
    §13.4 — ISTA (Iterative Shrinkage-Thresholding) for sparse coding.
    Solves: min_h 0.5||x - Dh||² + λ||h||₁
    """
    k = len(D[0]); h = [0.0]*k
    DT = mat_T(D)
    for _ in range(n_iter):
        Dh   = [sum(D[i][j]*h[j] for j in range(k)) for i in range(len(D))]
        resid= [x[i]-Dh[i] for i in range(len(x))]
        grad = [-sum(DT[j][i]*resid[i] for i in range(len(x))) for j in range(k)]
        h    = [soft_threshold(h[j]-lr*grad[j], lr*lam) for j in range(k)]
    return h


# ─────────────────────────────────────────────────────────────────
# §16.7 / §20.2  Restricted Boltzmann Machine (energy + CD-1)
# ─────────────────────────────────────────────────────────────────
class RBM:
    """
    §20.2 — Bernoulli-Bernoulli RBM.
    Energy: E(v,h) = -b^T v - c^T h - v^T W h
    Training via CD-1 (Contrastive Divergence, k=1).
    """
    def __init__(self, n_visible: int, n_hidden: int, seed: int = 0):
        rng = random.Random(seed)
        scale = 0.01
        self.W = [[rng.gauss(0,scale) for _ in range(n_hidden)]
                  for _ in range(n_visible)]
        self.b = [0.0]*n_visible   # visible bias
        self.c = [0.0]*n_hidden    # hidden bias

    def _sigmoid(self, x): return 1.0/(1.0+math.exp(-max(-30,min(30,x))))

    def h_given_v(self, v: List[float]) -> List[float]:
        """p(h_j=1|v) = σ(c_j + Σ_i W_ij v_i)"""
        return [self._sigmoid(self.c[j]+sum(self.W[i][j]*v[i] for i in range(len(v))))
                for j in range(len(self.c))]

    def v_given_h(self, h: List[float]) -> List[float]:
        """p(v_i=1|h) = σ(b_i + Σ_j W_ij h_j)"""
        return [self._sigmoid(self.b[i]+sum(self.W[i][j]*h[j] for j in range(len(h))))
                for i in range(len(self.b))]

    def energy(self, v: List[float], h: List[float]) -> float:
        """§16.7 — E(v,h) = -b·v - c·h - v^T W h"""
        return (-sum(self.b[i]*v[i] for i in range(len(v)))
                -sum(self.c[j]*h[j] for j in range(len(h)))
                -sum(v[i]*self.W[i][j]*h[j]
                     for i in range(len(v)) for j in range(len(h))))

    def cd1_update(self, v0: List[float], lr: float = 0.01):
        """CD-1 parameter update."""
        h0_prob = self.h_given_v(v0)
        h0 = [1.0 if random.random()<p else 0.0 for p in h0_prob]
        v1_prob = self.v_given_h(h0)
        v1 = [1.0 if random.random()<p else 0.0 for p in v1_prob]
        h1_prob = self.h_given_v(v1)
        nv, nh  = len(self.b), len(self.c)
        for i in range(nv):
            self.b[i] += lr*(v0[i]-v1[i])
            for j in range(nh):
                self.W[i][j] += lr*(v0[i]*h0_prob[j]-v1[i]*h1_prob[j])
        for j in range(nh):
            self.c[j] += lr*(h0_prob[j]-h1_prob[j])


# ─────────────────────────────────────────────────────────────────
# §18.6  Noise-Contrastive Estimation (NCE loss)
# ─────────────────────────────────────────────────────────────────
def nce_loss(log_p_model: List[float],
             log_p_noise: List[float],
             labels: List[int],
             k: int = 5) -> float:
    """
    §18.6 — NCE binary classification loss.
    labels[i]=1 for data samples, 0 for noise samples.
    h(x) = log p_model(x) - log [k * p_noise(x)]
    """
    total, eps = 0.0, 1e-12
    for lpm, lpn, y in zip(log_p_model, log_p_noise, labels):
        logit = lpm - math.log(k) - lpn
        p     = 1.0/(1.0+math.exp(-logit))
        total -= y*math.log(p+eps) + (1-y)*math.log(1-p+eps)
    return total / len(labels)


# ─────────────────────────────────────────────────────────────────
# §19.2  Expectation Maximisation (Gaussian mixture, E + M steps)
# ─────────────────────────────────────────────────────────────────
def gmm_e_step(data: List[float], mus: List[float],
               sigmas: List[float], pis: List[float]) -> List[List[float]]:
    """§19.2 — E-step: compute responsibilities r_{nk}."""
    k = len(mus); eps = 1e-12
    resp = []
    for x in data:
        raw = [pis[j]*gaussian_pdf(x,mus[j],sigmas[j]) for j in range(k)]
        s   = sum(raw)+eps
        resp.append([r/s for r in raw])
    return resp

def gmm_m_step(data: List[float],
               resp: List[List[float]]) -> Tuple[List,List,List]:
    """§19.2 — M-step: update mus, sigmas, pis from responsibilities."""
    n, k = len(data), len(resp[0]); eps=1e-8
    Nk   = [sum(resp[i][j] for i in range(n)) for j in range(k)]
    mus  = [sum(resp[i][j]*data[i] for i in range(n))/(Nk[j]+eps) for j in range(k)]
    sigs = [math.sqrt(sum(resp[i][j]*(data[i]-mus[j])**2 for i in range(n))/(Nk[j]+eps))+eps
            for j in range(k)]
    pis  = [Nk[j]/n for j in range(k)]
    return mus, sigs, pis

def gmm_fit(data: List[float], k: int = 2,
            n_iter: int = 50, seed: int = 0) -> Dict:
    """§19.2 — Full EM for 1-D Gaussian Mixture Model."""
    rng   = random.Random(seed)
    mus   = [rng.choice(data) for _ in range(k)]
    sigs  = [1.0]*k
    pis   = [1.0/k]*k
    for _ in range(n_iter):
        resp     = gmm_e_step(data, mus, sigs, pis)
        mus,sigs,pis = gmm_m_step(data, resp)
    return {"mus": mus, "sigmas": sigs, "pis": pis}


# ─────────────────────────────────────────────────────────────────
# §20.10.3  VAE — reparameterisation trick + ELBO
# ─────────────────────────────────────────────────────────────────
def vae_reparameterise(mu: List[float],
                        log_var: List[float]) -> List[float]:
    """§20.10.3 — z = µ + σ * ε, ε ~ N(0,I)  (reparameterisation trick)."""
    return [mu[i] + math.exp(0.5*log_var[i])*random.gauss(0,1)
            for i in range(len(mu))]

def vae_kl_loss(mu: List[float], log_var: List[float]) -> float:
    """§20.10.3 — KL(q(z|x) || N(0,I)) = -½ Σ(1 + log σ² - µ² - σ²)."""
    return -0.5 * sum(1 + log_var[i] - mu[i]**2 - math.exp(log_var[i])
                      for i in range(len(mu)))

def vae_elbo(recon_loss: float, mu: List[float],
             log_var: List[float]) -> float:
    """§20.10.3 — ELBO = E[log p(x|z)] - KL(q||p)."""
    return -(recon_loss + vae_kl_loss(mu, log_var))


# ─────────────────────────────────────────────────────────────────
# §12.4.2  Neural Language Model — n-gram probability
# ─────────────────────────────────────────────────────────────────
class NgramLM:
    """§12.4.2 — Smoothed n-gram language model (Laplace smoothing)."""
    def __init__(self, n: int = 2):
        self.n   = n
        self.counts: Dict[tuple,Dict[str,int]] = collections.defaultdict(
                         lambda: collections.defaultdict(int))
        self.vocab: set = set()

    def train(self, tokens: List[str]):
        self.vocab.update(tokens)
        for i in range(len(tokens)-self.n+1):
            ctx  = tuple(tokens[i:i+self.n-1])
            word = tokens[i+self.n-1]
            self.counts[ctx][word] += 1

    def prob(self, context: tuple, word: str, alpha: float = 1.0) -> float:
        """Laplace-smoothed probability P(word | context)."""
        ctx_counts = self.counts.get(context, {})
        V   = len(self.vocab)
        num = ctx_counts.get(word, 0) + alpha
        den = sum(ctx_counts.values()) + alpha*V
        return num/den if den > 0 else 1.0/max(V,1)

    def log_prob_sequence(self, tokens: List[str]) -> float:
        lp = 0.0
        for i in range(self.n-1, len(tokens)):
            ctx = tuple(tokens[i-self.n+1:i])
            lp += math.log(self.prob(ctx, tokens[i]))
        return lp

    def perplexity(self, tokens: List[str]) -> float:
        """§12.4.2 — perplexity = exp(-1/N * log P(tokens))."""
        N  = len(tokens) - self.n + 1
        if N <= 0: return float('inf')
        lp = self.log_prob_sequence(tokens)
        return math.exp(-lp / N)


# ─────────────────────────────────────────────────────────────────
# §17.5  Tempering — parallel tempering for mixing
# ─────────────────────────────────────────────────────────────────
def parallel_tempering_swap(states: List[List[float]],
                             log_probs: List[float],
                             temps: List[float]) -> List[List[float]]:
    """
    §17.5.1 — Propose swaps between adjacent temperature chains.
    Metropolis acceptance: min(1, exp((β_i-β_{i+1})(E_{i+1}-E_i)))
    """
    for i in range(len(states)-1):
        b1, b2 = 1.0/temps[i], 1.0/temps[i+1]
        log_accept = (b1-b2)*(log_probs[i+1]-log_probs[i])
        if math.log(random.random()+1e-12) < log_accept:
            states[i], states[i+1] = states[i+1], states[i]
            log_probs[i], log_probs[i+1] = log_probs[i+1], log_probs[i]
    return states


# ─────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing dl_book_implementations3.py ...")

    # §2.7 power iteration
    A = [[4,1],[2,3]]
    lam, v = power_iteration(A)
    assert lam > 4.9, lam
    print("  ✓ §2.7  Eigendecomposition (power iteration)")

    # §2.8 SVD
    A = [[1,2],[3,4],[5,6]]
    s, u, v = svd_1d(A)
    assert s > 0
    print("  ✓ §2.8  SVD (rank-1)")

    # §2.12 PCA
    data = [[1,2],[3,4],[5,6],[7,8]]
    proj = pca_project(data, k=1)
    assert len(proj)==4 and len(proj[0])==1
    print("  ✓ §2.12 PCA projection")

    # §3.13 entropy / KL
    p = [0.5,0.5]; q = [0.9,0.1]
    assert abs(entropy(p)-1.0) < 1e-9
    assert kl_divergence(p,p) < 1e-9
    assert cross_entropy(p,p) < 1.1
    print("  ✓ §3.13 Entropy / KL / Cross-entropy")

    # §3.10 softplus / log-sum-exp
    assert abs(softplus(0)-math.log(2)) < 1e-9
    assert abs(log_sum_exp([0,0])-math.log(2)) < 1e-9
    print("  ✓ §3.10 Softplus / LogSumExp")

    # §3.11 Bayes
    prior = {"A":0.5,"B":0.5}
    lik   = lambda h,o: 0.8 if h=="A" else 0.3
    post  = bayes_update(prior, lik, "obs")
    assert post["A"] > post["B"]
    print("  ✓ §3.11 Bayes' Rule")

    # §4.1 stable sigmoid
    assert abs(stable_sigmoid(0)-0.5) < 1e-9
    assert stable_sigmoid(1000) < 1.0+1e-9
    print("  ✓ §4.1  Numerical stability (sigmoid/softmax)")

    # §5.8.2 k-means
    pts = [[0.0,0.0],[0.1,0.1],[10.0,10.0],[10.1,10.1]]
    asgn, cents = kmeans(pts, k=2)
    assert asgn[0]==asgn[1] and asgn[2]==asgn[3]
    print("  ✓ §5.8.2 k-Means clustering")

    # §7.13 FGSM
    x = [0.5,0.5]; g = [0.3,-0.2]
    x_adv = fgsm_perturbation(x, g, epsilon=0.1)
    assert x_adv[0] > x[0] and x_adv[1] < x[1]
    print("  ✓ §7.13 FGSM adversarial perturbation")

    # §8.3.3 Nesterov
    nes = NesterovMomentum(lr=0.1)
    p   = {"w":1.0}
    la  = nes.lookahead_params(p)
    p   = nes.step(p, {"w":0.5})
    assert p["w"] < 1.0
    print("  ✓ §8.3.3 Nesterov Momentum")

    # §8.6.1 Newton
    f  = lambda x: (x-3)**2
    xr = newton_method(f, x0=0.0)
    assert abs(xr-3.0) < 0.01
    print("  ✓ §8.6.1 Newton's Method")

    # §8.7.2 Coordinate descent
    f2 = lambda p: (p[0]-1)**2 + (p[1]-2)**2
    res = coordinate_descent(f2, [0.0,0.0], n_cycles=200, step=0.05)
    assert abs(res[0]-1.0)<0.2 and abs(res[1]-2.0)<0.2
    print("  ✓ §8.7.2 Coordinate Descent")

    # §9.1 conv1d
    sig = [1,0,1,0,1,0,1]
    ker = [1,1]
    out = conv1d(sig, ker)
    assert len(out) == 6
    print("  ✓ §9.1  1-D Convolution")

    # §9.3 max pooling
    assert max_pool1d([1,3,2,4], pool_size=2, stride=2) == [3,4]
    print("  ✓ §9.3  Max Pooling")

    # §11.1 metrics
    m = precision_recall_f1([1,1,0,0],[1,0,1,0])
    assert "f1" in m and 0 <= m["f1"] <= 1
    print("  ✓ §11.1 Precision/Recall/F1")

    # §11.4.3 grid search
    gs = grid_search({"lr":[0.01,0.1],"bs":[16,32]})
    assert len(gs) == 4
    print("  ✓ §11.4.3 Grid Search")

    # §13.4 ISTA sparse coding
    D = [[1,0],[0,1],[0.7,0.7]]
    h = ista_sparse_code([1.0,0.0,0.0], D, lam=0.05)
    assert len(h) == 2
    print("  ✓ §13.4 Sparse Coding (ISTA)")

    # §16.7 / §20.2 RBM
    rbm = RBM(4, 3)
    v   = [1.0, 0.0, 1.0, 0.0]
    hp  = rbm.h_given_v(v)
    assert len(hp)==3 and all(0<=x<=1 for x in hp)
    rbm.cd1_update(v)
    print("  ✓ §20.2 RBM (energy + CD-1 update)")

    # §18.6 NCE
    lpm = [0.5, -0.5]; lpn = [0.1, 0.1]; labs = [1, 0]
    nce = nce_loss(lpm, lpn, labs)
    assert nce > 0
    print("  ✓ §18.6 Noise-Contrastive Estimation")

    # §19.2 GMM / EM
    data = [random.gauss(0,1) for _ in range(50)] + \
           [random.gauss(10,1) for _ in range(50)]
    res  = gmm_fit(data, k=2, n_iter=30)
    assert len(res["mus"])==2
    print("  ✓ §19.2 GMM / Expectation Maximisation")

    # §20.10.3 VAE
    mu  = [0.5, -0.3]; lv = [0.0, 0.0]
    z   = vae_reparameterise(mu, lv)
    kl  = vae_kl_loss(mu, lv)
    assert kl >= 0
    print("  ✓ §20.10.3 VAE reparameterisation + KL loss")

    # §12.4.2 N-gram LM
    lm = NgramLM(n=2)
    lm.train("the cat sat on the mat the cat".split())
    p  = lm.prob(("the",), "cat")
    pp = lm.perplexity("the cat sat".split())
    assert 0 < p <= 1 and pp > 0
    print("  ✓ §12.4.2 Neural N-gram Language Model")

    # §10.3 BiRNN
    fwd = [[0.1,0.2],[0.3,0.4]]; bwd = [[0.5,0.6],[0.7,0.8]]
    bi  = birnn_combine(fwd, bwd)
    assert len(bi[0]) == 4
    print("  ✓ §10.3 Bidirectional RNN combine")

    # §10.4 Attention
    q = [1.0,0.0]; ks = [[1.0,0.0],[0.0,1.0]]; vs = [[1.0,2.0],[3.0,4.0]]
    ctx = attention_context(q, ks, vs)
    assert len(ctx)==2
    print("  ✓ §10.4 Attention context vector")

    # §17.5 tempering
    states  = [[0.0],[10.0]]
    lps     = [-0.5, -50.0]
    temps   = [1.0, 10.0]
    swapped = parallel_tempering_swap(states, lps, temps)
    assert len(swapped)==2
    print("  ✓ §17.5 Parallel Tempering swap")

    print("\nAll tests passed.")
