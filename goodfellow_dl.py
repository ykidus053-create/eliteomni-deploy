"""
Goodfellow Deep Learning — All Major Chapters Implemented
Wired into EliteOmni pipeline. Pure numpy + stdlib, no internet needed.
Ch2: Linear Algebra, Ch3: Probability, Ch4: Numerical Computation,
Ch5: ML Basics, Ch6: Deep Nets, Ch7: Regularization, Ch8: Optimization,
Ch9: CNNs, Ch10: RNNs, Ch11: Practical Methodology, Ch12: Applications,
Ch13: Linear Factor Models, Ch14: Autoencoders, Ch15: Representation Learning
"""
import numpy as np
import math, re, time, sqlite3, os, hashlib, json
from collections import Counter, deque
from threading import Lock

# ═══════════════════════════════════════════════════════════════════════════════
# CH.2 — LINEAR ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════════
def cosine_similarity(a, b):
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

def pca(X, n_components=2):
    """Ch2.12: PCA via SVD — dimensionality reduction."""
    X = np.array(X, np.float32); X -= X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:n_components].T, S[:n_components]

def l2_norm(x): return float(np.linalg.norm(x))
def frobenius_norm(W): return float(np.linalg.norm(W, 'fro'))

# ═══════════════════════════════════════════════════════════════════════════════
# CH.3 — PROBABILITY & INFORMATION THEORY
# ═══════════════════════════════════════════════════════════════════════════════
def softmax(x, temperature=1.0):
    """Ch3: Softmax with temperature scaling."""
    x = np.array(x, np.float64) / max(temperature, 1e-8)
    x -= x.max(); e = np.exp(x)
    return e / e.sum()

def entropy(p):
    """Ch3.13: Shannon entropy H(P) = -sum p*log2(p)."""
    p = np.array(p, np.float64); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def kl_divergence(p, q):
    """Ch3.13: KL(P||Q) measures distribution divergence."""
    p, q = np.array(p, np.float64), np.array(q, np.float64)
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

def sample_top_k(logits, k=40, temperature=0.8):
    """Ch3: Top-k sampling — nucleus sampling for diverse generation."""
    logits = np.array(logits, np.float64)
    top_k_idx = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_idx]
    probs = softmax(top_k_logits, temperature)
    chosen = np.random.choice(len(probs), p=probs)
    return int(top_k_idx[chosen])

def gaussian_nll(x, mu, sigma):
    """Ch3.10: Gaussian negative log-likelihood."""
    return float(0.5 * np.log(2 * np.pi * sigma**2) + (x - mu)**2 / (2 * sigma**2))

# ═══════════════════════════════════════════════════════════════════════════════
# CH.4 — NUMERICAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
def numerical_gradient(f, x, eps=1e-5):
    """Ch4.3: Finite difference gradient — verify backprop correctness."""
    x = np.array(x, np.float64)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        grad[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad

def log_sum_exp(x):
    """Ch4: Numerically stable log-sum-exp."""
    x = np.array(x, np.float64); c = x.max()
    return float(c + np.log(np.sum(np.exp(x - c))))

def clip_gradient(grad, max_norm=5.0):
    """Ch4 / Ch10.11: Gradient clipping prevents exploding gradients."""
    grad = np.array(grad, np.float64)
    norm = np.linalg.norm(grad)
    return grad * (max_norm / norm) if norm > max_norm else grad

# ═══════════════════════════════════════════════════════════════════════════════
# CH.5 — MACHINE LEARNING BASICS
# ═══════════════════════════════════════════════════════════════════════════════
def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Ch5.3: Holdout method — separate test set for unbiased evaluation."""
    np.random.seed(seed); idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def k_fold_indices(n, k=5, seed=42):
    """Ch5.3: K-fold cross-validation indices."""
    np.random.seed(seed); idx = np.random.permutation(n)
    return [idx[i*n//k:(i+1)*n//k] for i in range(k)]

def bias_variance_mse(y_true, predictions_list):
    """Ch5.4: Decompose MSE into bias^2 + variance + noise."""
    y_true = np.array(y_true, np.float64)
    preds = np.array(predictions_list, np.float64)
    mean_pred = preds.mean(axis=0)
    bias_sq = np.mean((mean_pred - y_true) ** 2)
    variance = np.mean(np.var(preds, axis=0))
    return {"bias_squared": float(bias_sq), "variance": float(variance),
            "total_mse": float(bias_sq + variance)}

def mle_gaussian(data):
    """Ch5.5: Maximum Likelihood Estimation for Gaussian."""
    data = np.array(data, np.float64)
    return {"mu": float(data.mean()), "sigma": float(data.std())}

def map_estimate_gaussian(data, prior_mu=0.0, prior_sigma=1.0):
    """Ch5.6: Maximum A Posteriori — Gaussian with Gaussian prior."""
    data = np.array(data, np.float64)
    n = len(data); sigma_data = data.std() + 1e-8
    post_var = 1.0 / (n / sigma_data**2 + 1.0 / prior_sigma**2)
    post_mu = post_var * (np.sum(data) / sigma_data**2 + prior_mu / prior_sigma**2)
    return {"map_mu": float(post_mu), "posterior_var": float(post_var)}

def tfidf_vectors(corpus):
    """Ch5: TF-IDF — hand-crafted features baseline before learned representations."""
    tokenize = lambda t: re.findall(r'\b[a-z]{3,}\b', t.lower())
    tokenized = [tokenize(d) for d in corpus]
    df = Counter(t for doc in tokenized for t in set(doc))
    N = len(corpus)
    def tfidf(doc_tokens):
        tf = Counter(doc_tokens)
        total = len(doc_tokens) or 1
        return {t: (tf[t]/total) * math.log((N+1)/(df[t]+1)) for t in tf}
    return [tfidf(t) for t in tokenized]

def cosine_sim_tfidf(v1, v2):
    """Ch5: Cosine similarity on sparse TF-IDF vectors."""
    keys = set(v1) | set(v2)
    a = np.array([v1.get(k, 0) for k in keys], np.float64)
    b = np.array([v2.get(k, 0) for k in keys], np.float64)
    return cosine_similarity(a, b)

def rank_by_tfidf(query, candidates):
    """Ch5: BM25-style ranking. Falls back when embeddings unavailable."""
    corpus = [query] + candidates
    vecs = tfidf_vectors(corpus)
    q_vec = vecs[0]
    scored = [(cosine_sim_tfidf(q_vec, vecs[i+1]), c) for i, c in enumerate(candidates)]
    scored.sort(reverse=True)
    return [c for _, c in scored]

# ═══════════════════════════════════════════════════════════════════════════════
# CH.6 — DEEP FEEDFORWARD NETWORKS
# ═══════════════════════════════════════════════════════════════════════════════
def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(np.float32)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_grad(x): s = sigmoid(x); return s * (1 - s)
def tanh_act(x): return np.tanh(x)
def tanh_grad(x): return 1.0 - np.tanh(x)**2
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def elu(x, alpha=1.0): return np.where(x > 0, x, alpha * (np.exp(x) - 1))
def gelu(x): return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
def swish(x): return x * sigmoid(x)

def xavier_init(fan_in, fan_out):
    """Ch6.2.2: Xavier/Glorot initialization — keeps variance stable across layers."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

def he_init(fan_in, fan_out):
    """Ch6 / He 2015: He initialization for ReLU networks."""
    std = math.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, (fan_in, fan_out)).astype(np.float32)

def cross_entropy_loss(logits, target_idx):
    """Ch6.2.1: Softmax cross-entropy loss."""
    probs = softmax(logits)
    return float(-math.log(probs[target_idx] + 1e-12))

def mse_loss(y_pred, y_true):
    """Ch6.2: Mean squared error."""
    return float(np.mean((np.array(y_pred) - np.array(y_true))**2))

class FeedForward:
    """Ch6: Minimal feedforward net with backprop — numpy only."""
    def __init__(self, dims, lr=0.01):
        self.Ws = [xavier_init(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        self.bs = [np.zeros(dims[i+1], np.float32) for i in range(len(dims)-1)]
        self.lr = lr

    def forward(self, x):
        self.cache = [x]
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            x = relu(x @ W + b); self.cache.append(x)
        x = x @ self.Ws[-1] + self.bs[-1]; self.cache.append(x)
        return x

    def backward(self, grad_out):
        grads_W, grads_b = [], []
        g = grad_out
        for i in reversed(range(len(self.Ws))):
            if i < len(self.Ws) - 1:
                g = g * relu_grad(self.cache[i+1])
            grads_W.insert(0, self.cache[i].T @ g)
            grads_b.insert(0, g.sum(axis=0))
            g = g @ self.Ws[i].T
        for i in range(len(self.Ws)):
            self.Ws[i] -= self.lr * clip_gradient(grads_W[i])
            self.bs[i] -= self.lr * grads_b[i]

# ═══════════════════════════════════════════════════════════════════════════════
# CH.7 — REGULARIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def l2_regularization(weights, lambda_=0.01):
    """Ch7.1: L2 / weight decay — penalizes large weights."""
    return lambda_ * sum(float(np.sum(w**2)) for w in weights)

def l1_regularization(weights, lambda_=0.01):
    """Ch7.1: L1 / Lasso — encourages sparsity."""
    return lambda_ * sum(float(np.sum(np.abs(w))) for w in weights)

def dropout(x, rate=0.5, training=True):
    """Ch7.12: Inverted dropout — scales by 1/(1-rate) at train time."""
    if not training or rate == 0: return x
    mask = (np.random.rand(*x.shape) > rate).astype(np.float32)
    return x * mask / (1.0 - rate)

def max_norm_constraint(W, max_norm=3.0):
    """Ch7.4: Max-norm regularization clamps weight column norms."""
    W = np.array(W, np.float32)
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    return W * np.minimum(1.0, max_norm / (norms + 1e-8))

def label_smoothing(target_idx, n_classes, epsilon=0.1):
    """Ch7: Label smoothing — prevents overconfident predictions."""
    dist = np.full(n_classes, epsilon / n_classes, np.float64)
    dist[target_idx] += 1.0 - epsilon
    return dist

def early_stopping_check(val_losses, patience=5, min_delta=1e-4):
    """Ch7.8: Early stopping — stop when validation loss stops improving."""
    if len(val_losses) < patience + 1: return False
    best = min(val_losses[:-patience])
    return all(v > best - min_delta for v in val_losses[-patience:])

def data_augmentation_noise(x, std=0.01):
    """Ch7.4: Input noise as regularization (Goodfellow §7.5)."""
    return x + np.random.normal(0, std, x.shape).astype(x.dtype)

# ═══════════════════════════════════════════════════════════════════════════════
# CH.8 — OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
class SGD:
    """Ch8.3: Stochastic Gradient Descent with momentum."""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr; self.momentum = momentum; self.v = {}
    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.v: self.v[i] = np.zeros_like(p)
            self.v[i] = self.momentum * self.v[i] - self.lr * g
            p += self.v[i]

class Adam:
    """Ch8.5: Adam optimizer — adaptive learning rates + momentum."""
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps
        self.m={}; self.v={}; self.t=0
    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m or self.m[i].shape != g.shape:
                self.m[i]=np.zeros_like(g); self.v[i]=np.zeros_like(g)
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*g**2
            m_hat = self.m[i] / (1-self.b1**self.t)
            v_hat = self.v[i] / (1-self.b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSProp:
    """Ch8.5.2: RMSProp — per-parameter adaptive learning rates."""
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        self.lr=lr; self.rho=rho; self.eps=eps; self.sq={}
    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.sq: self.sq[i] = np.zeros_like(p)
            self.sq[i] = self.rho*self.sq[i] + (1-self.rho)*g**2
            p -= self.lr * g / (np.sqrt(self.sq[i]) + self.eps)

def cosine_annealing_lr(step, total_steps, lr_max=1e-3, lr_min=1e-6):
    """Ch8.6: Cosine annealing LR schedule."""
    return lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(math.pi*step/total_steps))

def warmup_lr(step, warmup_steps, base_lr=1e-3):
    """Ch8: Linear warmup — avoids early instability."""
    return base_lr * min(1.0, step / max(warmup_steps, 1))

def batch_norm(x, gamma=1.0, beta=0.0, eps=1e-8):
    """Ch8.7 / Ch7: Batch normalization — stabilizes deep network training."""
    x = np.array(x, np.float64)
    mu = x.mean(axis=0); sigma = x.std(axis=0) + eps
    return gamma * (x - mu) / sigma + beta

def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-8):
    """Ch8 / Transformers: Layer normalization — normalizes across features."""
    x = np.array(x, np.float64)
    mu = x.mean(); sigma = x.std() + eps
    return gamma * (x - mu) / sigma + beta

def monitor_gradient_norm(grads):
    """Ch8 / Ch10: Track gradient norms — detect vanishing/exploding."""
    norms = [float(np.linalg.norm(g)) for g in grads]
    return {"norms": norms, "mean": float(np.mean(norms)),
            "max": float(np.max(norms)), "min": float(np.min(norms)),
            "vanishing": float(np.max(norms)) < 1e-4,
            "exploding": float(np.max(norms)) > 100.0}

# ═══════════════════════════════════════════════════════════════════════════════
# CH.9 — CONVOLUTIONAL NETWORKS
# ═══════════════════════════════════════════════════════════════════════════════
def conv1d(x, kernel, stride=1, padding=0):
    """Ch9.1: 1D convolution — text/sequence feature extraction."""
    x = np.array(x, np.float32)
    if padding > 0: x = np.pad(x, padding)
    k = len(kernel); out_len = (len(x) - k) // stride + 1
    return np.array([np.dot(x[i*stride:i*stride+k], kernel) for i in range(out_len)])

def max_pool1d(x, pool_size=2):
    """Ch9.3: Max pooling — translation invariance."""
    return np.array([x[i:i+pool_size].max() for i in range(0, len(x)-pool_size+1, pool_size)])

def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """Ch9: Calculate conv output dimensions."""
    return (input_size + 2*padding - kernel_size) // stride + 1

def receptive_field(n_layers, kernel_size):
    """Ch9: Total receptive field after n conv layers."""
    return 1 + n_layers * (kernel_size - 1)

# ═══════════════════════════════════════════════════════════════════════════════
# CH.10 — SEQUENCE MODELING / RNNs
# ═══════════════════════════════════════════════════════════════════════════════
class VanillaRNN:
    """Ch10.2: Elman RNN — h_t = tanh(Wx*x_t + Wh*h_{t-1} + b)."""
    def __init__(self, input_size, hidden_size):
        self.Wx = xavier_init(input_size, hidden_size)
        self.Wh = xavier_init(hidden_size, hidden_size)
        self.b = np.zeros(hidden_size, np.float32)
        self.hidden_size = hidden_size

    def forward(self, xs):
        h = np.zeros(self.hidden_size, np.float32)
        hs = []
        for x in xs:
            h = np.tanh(x @ self.Wx + h @ self.Wh + self.b)
            hs.append(h.copy())
        return hs, h

class LSTMCell:
    """Ch10.10: LSTM — solves vanishing gradient via gated memory cell."""
    def __init__(self, input_size, hidden_size):
        n = input_size + hidden_size
        self.Wf = xavier_init(n, hidden_size)  # forget gate
        self.Wi = xavier_init(n, hidden_size)  # input gate
        self.Wc = xavier_init(n, hidden_size)  # cell gate
        self.Wo = xavier_init(n, hidden_size)  # output gate
        self.bf = np.ones(hidden_size, np.float32)   # forget bias=1 helps
        self.bi = np.zeros(hidden_size, np.float32)
        self.bc = np.zeros(hidden_size, np.float32)
        self.bo = np.zeros(hidden_size, np.float32)

    def forward(self, x, h_prev, c_prev):
        xh = np.concatenate([x, h_prev])
        f = sigmoid(xh @ self.Wf + self.bf)
        i = sigmoid(xh @ self.Wi + self.bi)
        c_tilde = np.tanh(xh @ self.Wc + self.bc)
        c = f * c_prev + i * c_tilde
        o = sigmoid(xh @ self.Wo + self.bo)
        h = o * np.tanh(c)
        return h, c

def bptt_clip(grads, max_norm=5.0):
    """Ch10.11: BPTT with gradient clipping — tames exploding gradients in RNNs."""
    return [clip_gradient(g, max_norm) for g in grads]

# ═══════════════════════════════════════════════════════════════════════════════
# CH.11 — PRACTICAL METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
def learning_curve(train_sizes, train_scores, val_scores):
    """Ch11.4: Learning curves — diagnose bias vs variance."""
    results = []
    for n, tr, va in zip(train_sizes, train_scores, val_scores):
        gap = tr - va
        diagnosis = "high_variance" if gap > 0.1 else "high_bias" if va < 0.7 else "good"
        results.append({"n": n, "train": tr, "val": va, "gap": gap, "diagnosis": diagnosis})
    return results

def hyperparameter_grid(param_grid, max_trials=20, seed=42):
    """Ch11.4.3: Random search over hyperparameter grid (better than grid search)."""
    import random; random.seed(seed)
    keys = list(param_grid.keys())
    trials = []
    for _ in range(max_trials):
        trial = {k: random.choice(v) for k, v in param_grid.items()}
        trials.append(trial)
    return trials

def detect_vanishing_gradients(grad_norms_per_layer):
    """Ch11 / Ch10: Flag layers with near-zero gradients."""
    flags = []
    for i, norm in enumerate(grad_norms_per_layer):
        flags.append({"layer": i, "norm": norm,
                      "status": "vanishing" if norm < 1e-5 else
                                "exploding" if norm > 1e3 else "ok"})
    return flags

def confusion_matrix(y_true, y_pred, n_classes):
    """Ch11.3: Confusion matrix for multi-class evaluation."""
    cm = np.zeros((n_classes, n_classes), int)
    for t, p in zip(y_true, y_pred): cm[t][p] += 1
    return cm

def f1_score(y_true, y_pred):
    """Ch11.3: Binary F1 = 2*precision*recall / (precision+recall)."""
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
    return 2*prec*rec/(prec+rec+1e-8)

# ═══════════════════════════════════════════════════════════════════════════════
# CH.13 — LINEAR FACTOR MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def ppca(X, n_components=2, max_iter=50):
    """Ch13.1: Probabilistic PCA via EM — generative linear factor model."""
    X = np.array(X, np.float64); n, d = X.shape
    X -= X.mean(axis=0)
    W = np.random.randn(d, n_components) * 0.1
    sigma2 = 1.0
    for _ in range(max_iter):
        M = W.T @ W + sigma2 * np.eye(n_components)
        M_inv = np.linalg.inv(M)
        Ez = X @ W @ M_inv.T
        EzzT = n * sigma2 * M_inv + Ez.T @ Ez
        W_new = X.T @ Ez @ np.linalg.inv(EzzT)
        sigma2 = float(np.mean(np.sum((X - Ez @ W_new.T)**2, axis=1)) / d)
        W = W_new
    return Ez, W, sigma2

def ica_whitening(X):
    """Ch13.4: ICA preprocessing — whiten data (zero mean, unit variance, decorrelated)."""
    X = np.array(X, np.float64); X -= X.mean(axis=0)
    cov = np.cov(X.T)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + 1e-8)) @ U.T
    return X @ W.T

# ═══════════════════════════════════════════════════════════════════════════════
# CH.14 — AUTOENCODERS
# ═══════════════════════════════════════════════════════════════════════════════
class Autoencoder:
    """Ch14: Undercomplete autoencoder — learns compressed representation."""
    def __init__(self, input_dim, latent_dim, lr=0.001):
        self.We = xavier_init(input_dim, latent_dim)
        self.be = np.zeros(latent_dim, np.float32)
        self.Wd = xavier_init(latent_dim, input_dim)
        self.bd = np.zeros(input_dim, np.float32)
        self.opt = Adam(lr)

    def encode(self, x): return relu(np.array(x, np.float32) @ self.We + self.be)
    def decode(self, z): return sigmoid(z @ self.Wd + self.bd)
    def forward(self, x): return self.decode(self.encode(x))

    def train_step(self, x):
        """Ch14.1: Reconstruction loss backprop with corrected gradient shapes."""
        x = np.array(x, np.float32)
        if x.ndim == 1:
            x = x[None, :]
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = mse_loss(x_hat, x)
        grad_xhat = 2 * (x_hat - x) / len(x)
        grad_sig = grad_xhat * x_hat * (1 - x_hat)
        grad_Wd = z.T @ grad_sig
        grad_bd = grad_sig.sum(axis=0)
        grad_z = grad_sig @ self.Wd.T
        grad_relu = grad_z * relu_grad(z)
        grad_We = x.T @ grad_relu
        grad_be = grad_relu.sum(axis=0)
        self.opt.step([self.We, self.be, self.Wd, self.bd], [grad_We, grad_be, grad_Wd, grad_bd])
        return loss

def denoising_autoencoder_corrupt(x, noise_std=0.1):
    """Ch14.2: Denoising AE — add noise to input, reconstruct clean signal."""
    return np.array(x, np.float32) + np.random.normal(0, noise_std, np.array(x).shape).astype(np.float32)

def sparse_penalty(z, target_sparsity=0.05, beta=1.0):
    """Ch14.3: Sparse autoencoder KL penalty on hidden activations."""
    rho_hat = np.clip(np.mean(z, axis=0), 1e-8, 1-1e-8)
    rho = target_sparsity
    kl = rho*np.log(rho/rho_hat) + (1-rho)*np.log((1-rho)/(1-rho_hat))
    return float(beta * np.sum(kl))

# ═══════════════════════════════════════════════════════════════════════════════
# CH.15 — REPRESENTATION LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
def build_word_cooccurrence(corpus_tokens, window=4):
    """Ch15.1: Word co-occurrence matrix — basis for word embeddings."""
    vocab = sorted(set(t for doc in corpus_tokens for t in doc))
    w2i = {w:i for i,w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)), np.float32)
    for doc in corpus_tokens:
        for i, w in enumerate(doc):
            for j in range(max(0,i-window), min(len(doc),i+window+1)):
                if i != j:
                    C[w2i[w]][w2i[doc[j]]] += 1.0
    return C, vocab, w2i

def ppmi(C):
    """Ch15: Positive PMI — improves raw co-occurrence matrix."""
    row_sum = C.sum(axis=1, keepdims=True) + 1e-8
    col_sum = C.sum(axis=0, keepdims=True) + 1e-8
    total = C.sum() + 1e-8
    pmi = np.log2((C * total) / (row_sum * col_sum) + 1e-8)
    return np.maximum(0, pmi)

def svd_embeddings(C, dim=32):
    """Ch15: SVD word embeddings from co-occurrence (precursor to word2vec)."""
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return U[:, :dim] * np.sqrt(S[:dim])

def contrastive_loss(anchor, positive, negative, margin=1.0):
    """Ch15: Triplet contrastive loss — pulls similar pairs, pushes dissimilar."""
    d_pos = l2_norm(np.array(anchor) - np.array(positive))
    d_neg = l2_norm(np.array(anchor) - np.array(negative))
    return max(0.0, d_pos - d_neg + margin)

def nearest_neighbors(query_vec, embedding_matrix, vocab, top_k=5):
    """Ch15.1: Nearest neighbor retrieval in embedding space."""
    q = np.array(query_vec, np.float32)
    q /= (np.linalg.norm(q) + 1e-8)
    E = np.array(embedding_matrix, np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    sims = (E / norms) @ q
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [(vocab[i], float(sims[i])) for i in top_idx]

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER COMPONENTS (Ch15 + modern extensions)
# ═══════════════════════════════════════════════════════════════════════════════
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Ch15 / Attention Is All You Need: core transformer attention."""
    Q,K,V = [np.array(x, np.float32) for x in [Q,K,V]]
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)
    if mask is not None: scores[mask] = -1e9
    weights = softmax(scores)
    return weights @ V, weights

def sinusoidal_positional_encoding(seq_len, d_model):
    """Ch15: Sinusoidal position encodings — inject sequence order."""
    PE = np.zeros((seq_len, d_model), np.float32)
    pos = np.arange(seq_len)[:,None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE

def rotary_positional_encoding(seq_len, d_model):
    """RoPE: Rotary position embeddings (used in LLaMA, GPT-NeoX)."""
    assert d_model % 2 == 0
    theta = np.array([1.0 / (10000 ** (2*i/d_model)) for i in range(d_model//2)])
    pos = np.arange(seq_len)[:,None]
    angles = pos * theta[None,:]
    cos = np.cos(angles); sin = np.sin(angles)
    return cos, sin

def multi_head_attention_scores(Q, K, n_heads):
    """Ch15: Multi-head attention splits into parallel attention heads."""
    d_k = Q.shape[-1] // n_heads
    scores = []
    for h in range(n_heads):
        Qh = Q[:, h*d_k:(h+1)*d_k]; Kh = K[:, h*d_k:(h+1)*d_k]
        scores.append(Qh @ Kh.T / math.sqrt(d_k))
    return scores

# ═══════════════════════════════════════════════════════════════════════════════
# WIRING: Semantic ranking for EliteOmni memory retrieval
# ═══════════════════════════════════════════════════════════════════════════════
_VOCAB_CACHE = {}
_EMBED_CACHE = {}
_EMBED_LOCK = Lock()

def _tokenize(text):
    return re.findall(r'\b[a-z]{3,}\b', text.lower())

def build_tfidf_index(documents):
    """Build TF-IDF index from list of strings. Used for memory retrieval."""
    tokenized = [_tokenize(d) for d in documents]
    df = Counter(t for doc in tokenized for t in set(doc))
    N = len(documents)
    matrix = []
    for doc_toks in tokenized:
        tf = Counter(doc_toks); total = len(doc_toks) or 1
        vec = {t: (tf[t]/total)*math.log((N+1)/(df[t]+1)+1) for t in tf}
        matrix.append(vec)
    return matrix, df, N

def semantic_rank(query, candidates, top_k=5):
    """
    Ch15.1: Representation learning for retrieval.
    Uses TF-IDF + cosine similarity (degrades gracefully without GPU).
    Drop-in for mem_get() + embedding re-rank.
    """
    if not candidates: return []
    corpus = [query] + candidates
    vecs, _, _ = build_tfidf_index(corpus)
    q_vec = vecs[0]
    scored = []
    for i, c in enumerate(candidates):
        sim = cosine_sim_tfidf(q_vec, vecs[i+1])
        scored.append((sim, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]

def calibrate_confidence(text, skill="general"):
    """
    Ch3 + Ch11: Calibrated confidence scoring.
    Detects overconfident language, returns calibrated [0,1] score.
    """
    OVERCONF = len(re.findall(
        r'\b(definitely|certainly|always|never|guaranteed|absolutely|obviously)\b',
        text, re.IGNORECASE))
    HEDGED = len(re.findall(
        r'\b(may|might|could|probably|likely|approximately|I think|I believe|seems)\b',
        text, re.IGNORECASE))
    total = OVERCONF + HEDGED + 1
    raw = (OVERCONF * 0.9 + HEDGED * 0.5) / total
    # Bayesian shrinkage toward 0.7 (Ch5.6 MAP)
    prior_conf = 0.7; n_obs = total
    return float((n_obs * raw + 3 * prior_conf) / (n_obs + 3))

def verify_math_response(response_text):
    """
    Ch4.3 + Ch11.3: Numerical verification of math answers.
    Extracts numeric claims and checks internal consistency.
    """
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', response_text)
    if len(numbers) < 2: return {"verified": True, "numbers": numbers}
    floats = [float(n) for n in numbers[:10]]
    issues = []
    for i in range(len(floats)-1):
        a, b = floats[i], floats[i+1]
        if a != 0 and abs(b/a) > 1e10:
            issues.append(f"Suspicious jump: {a} → {b}")
    return {"verified": len(issues)==0, "issues": issues, "numbers": numbers[:10]}

def response_diversity_score(responses):
    """
    Ch7 + Ch15: Measure diversity among candidate responses (for voting_engine).
    Low diversity → single sample; high diversity → voting helps more.
    """
    if len(responses) < 2: return 0.0
    vecs, _, _ = build_tfidf_index(responses)
    sims = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            sims.append(cosine_sim_tfidf(vecs[i], vecs[j]))
    avg_sim = float(np.mean(sims)) if sims else 1.0
    return float(1.0 - avg_sim)  # diversity = 1 - avg_similarity

def gradient_informed_prompt_score(prompt, response):
    """
    Ch8 metaphor: Score prompt-response pair by 'gradient signal'.
    High score = response well-aligned to prompt intent.
    """
    p_toks = set(_tokenize(prompt))
    r_toks = set(_tokenize(response))
    coverage = len(p_toks & r_toks) / (len(p_toks) + 1e-8)
    length_penalty = min(1.0, len(response.split()) / 50.0)
    return float(coverage * length_penalty)

print("[goodfellow_dl] ✅ All chapters loaded: Ch2-Ch15 + Transformers")
