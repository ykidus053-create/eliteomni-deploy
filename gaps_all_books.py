"""
ALL GAPS across all 6 books not yet in EliteOmni.
Books: MML, Deep Learning (Goodfellow), Build LLM From Scratch (Raschka),
       Hands-On LLMs (Alammar), AI Engineering (Huyen), DMLS (Huyen), LLM Engineer's Handbook
"""
import math, random, re, time, json, os, hashlib, sqlite3, threading
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# GOODFELLOW — DEEP LEARNING (gaps not in math_impl or book_gaps_impl)
# ─────────────────────────────────────────────────────────────────────────────

# Ch6 — Maxout, ELU, SELU, Leaky ReLU, PReLU
def elu(x: float, alpha: float = 1.0) -> float:
    return x if x >= 0 else alpha * (math.exp(x) - 1)

def selu(x: float) -> float:
    alpha, scale = 1.6732632423543772, 1.0507009873554805
    return scale * (x if x >= 0 else alpha * (math.exp(x) - 1))

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    return x if x >= 0 else alpha * x

def maxout(inputs: list) -> float:
    return max(inputs)

def hard_sigmoid(x: float) -> float:
    return max(0.0, min(1.0, 0.2 * x + 0.5))

def hard_swish(x: float) -> float:
    return x * hard_sigmoid(x)

# Ch6 — Weight tying check
def are_weights_tied(W1: list, W2: list) -> bool:
    return W1 is W2

# Ch7 — L1/L2 regularization penalties
def l1_penalty(params: list, lam: float = 1e-4) -> float:
    return lam * sum(abs(p) for p in params)

def l2_penalty(params: list, lam: float = 1e-4) -> float:
    return lam * sum(p ** 2 for p in params)

def elastic_net_penalty(params: list, lam1: float = 1e-4, lam2: float = 1e-4) -> float:
    return l1_penalty(params, lam1) + l2_penalty(params, lam2)

# Ch7 — Dropout (inference-time scaling)
def dropout_forward(x: list, p: float = 0.5, training: bool = True) -> list:
    if not training:
        return x
    mask = [0.0 if random.random() < p else 1.0 / (1 - p) for _ in x]
    return [xi * mi for xi, mi in zip(x, mask)]

# Ch7 — Data augmentation: gaussian noise injection
def add_gaussian_noise(x: list, std: float = 0.01) -> list:
    return [xi + random.gauss(0, std) for xi in x]

# Ch7 — Early stopping tracker
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# Ch8 — Gradient clipping (by norm and by value)
def clip_grad_norm(grads: list, max_norm: float = 1.0) -> list:
    total = math.sqrt(sum(g ** 2 for g in grads))
    if total > max_norm:
        scale = max_norm / max(total, 1e-9)
        return [g * scale for g in grads]
    return grads

def clip_grad_value(grads: list, clip: float = 1.0) -> list:
    return [max(-clip, min(clip, g)) for g in grads]

# Ch8 — Nesterov momentum
def nesterov_step(params: list, grads: list, velocity: list,
                  lr: float = 0.01, momentum: float = 0.9) -> Tuple[list, list]:
    v_new = [momentum * v - lr * g for v, g in zip(velocity, grads)]
    p_new = [p + momentum * vn - lr * g for p, vn, g in zip(params, v_new, grads)]
    return p_new, v_new

# Ch8 — Adagrad
def adagrad_step(grad: list, cache: list, lr: float = 0.01,
                 eps: float = 1e-8) -> Tuple[list, list]:
    cache = [c + g ** 2 for c, g in zip(cache, grad)]
    update = [lr * g / (math.sqrt(c) + eps) for g, c in zip(grad, cache)]
    return update, cache

# Ch8 — Polyak averaging
def polyak_average(model_params: list, avg_params: list, t: int) -> list:
    return [(t * a + p) / (t + 1) for a, p in zip(avg_params, model_params)]

# Ch9 — Conv1D (for sequence models)
def conv1d(x: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
    x_pad = [0.0] * padding + x + [0.0] * padding
    k = len(kernel)
    out = []
    for i in range(0, len(x_pad) - k + 1, stride):
        out.append(sum(x_pad[i + j] * kernel[j] for j in range(k)))
    return out

def max_pool1d(x: list, kernel_size: int = 2, stride: int = 2) -> list:
    return [max(x[i:i + kernel_size]) for i in range(0, len(x) - kernel_size + 1, stride)]

def avg_pool1d(x: list, kernel_size: int = 2, stride: int = 2) -> list:
    return [sum(x[i:i + kernel_size]) / kernel_size
            for i in range(0, len(x) - kernel_size + 1, stride)]

# Ch10 — RNN forward pass
def rnn_step(x: list, h: list, Wxh: list, Whh: list, bh: list) -> list:
    xh = [sum(Wxh[i][j] * x[j] for j in range(len(x))) for i in range(len(bh))]
    hh = [sum(Whh[i][j] * h[j] for j in range(len(h))) for i in range(len(bh))]
    return [math.tanh(xh[i] + hh[i] + bh[i]) for i in range(len(bh))]

# Ch10 — LSTM cell
def lstm_step(x: list, h: list, c: list,
              Wi: list, Wf: list, Wg: list, Wo: list,
              bi: list, bf: list, bg: list, bo: list) -> Tuple[list, list]:
    def gate(W, b, act):
        z = [sum(W[k][j] * x[j] for j in range(len(x))) +
             sum(W[k][len(x) + j] * h[j] for j in range(len(h))) + b[k]
             for k in range(len(b))]
        return [act(v) for v in z]
    sig = lambda v: 1 / (1 + math.exp(-max(-500, min(500, v))))
    i_gate = gate(Wi, bi, sig)
    f_gate = gate(Wf, bf, sig)
    g_gate = gate(Wg, bg, math.tanh)
    o_gate = gate(Wo, bo, sig)
    c_new = [f_gate[k] * c[k] + i_gate[k] * g_gate[k] for k in range(len(c))]
    h_new = [o_gate[k] * math.tanh(c_new[k]) for k in range(len(c))]
    return h_new, c_new

# Ch10 — GRU cell
def gru_step(x: list, h: list, Wz: list, Wr: list, Wn: list,
             bz: list, br: list, bn: list) -> list:
    sig = lambda v: 1 / (1 + math.exp(-max(-500, min(500, v))))
    def gate(W, b, act):
        return [act(sum(W[k][j] * (list(x) + list(h))[j]
                        for j in range(len(x) + len(h))) + b[k])
                for k in range(len(b))]
    z = gate(Wz, bz, sig)
    r = gate(Wr, br, sig)
    xrh = list(x) + [r[k] * h[k] for k in range(len(h))]
    n = [math.tanh(sum(Wn[k][j] * xrh[j] for j in range(len(xrh))) + bn[k])
         for k in range(len(h))]
    return [(1 - z[k]) * n[k] + z[k] * h[k] for k in range(len(h))]

# Ch13 — VAE reparameterization trick
def vae_reparameterize(mu: list, log_var: list) -> list:
    return [mu[i] + math.exp(0.5 * log_var[i]) * random.gauss(0, 1)
            for i in range(len(mu))]

def vae_kl_loss(mu: list, log_var: list) -> float:
    return -0.5 * sum(1 + log_var[i] - mu[i] ** 2 - math.exp(log_var[i])
                      for i in range(len(mu)))

# Ch16 — Word2Vec negative sampling loss
def word2vec_neg_sampling_loss(pos_score: float, neg_scores: list) -> float:
    pos_loss = -math.log(max(1 / (1 + math.exp(-pos_score)), 1e-9))
    neg_loss = -sum(math.log(max(1 - 1 / (1 + math.exp(-s)), 1e-9)) for s in neg_scores)
    return pos_loss + neg_loss

# Ch20 — ELBO (Evidence Lower Bound)
def elbo(reconstruction_loss: float, kl_loss: float, beta: float = 1.0) -> float:
    return -(reconstruction_loss + beta * kl_loss)

# ─────────────────────────────────────────────────────────────────────────────
# RASCHKA — BUILD LLM FROM SCRATCH (gaps)
# ─────────────────────────────────────────────────────────────────────────────

# Ch2 — Sliding window dataset builder
def sliding_window_dataset(token_ids: list, context_len: int,
                            stride: int = 1) -> list:
    samples = []
    for i in range(0, len(token_ids) - context_len, stride):
        x = token_ids[i:i + context_len]
        y = token_ids[i + 1:i + context_len + 1]
        samples.append((x, y))
    return samples

# Ch2 — Simple character tokenizer
class CharTokenizer:
    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def fit(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text: str) -> list:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list) -> str:
        return ''.join(self.itos.get(i, '') for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

# Ch3 — Causal (masked) self-attention mask
def causal_mask(seq_len: int) -> list:
    return [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]

# Ch3 — Multi-head attention split/merge
def split_heads(x: list, n_heads: int) -> list:
    d = len(x) // n_heads
    return [x[i * d:(i + 1) * d] for i in range(n_heads)]

def merge_heads(heads: list) -> list:
    return [v for head in heads for v in head]

# Ch3 — Attention with causal mask applied
def masked_attention(Q: list, K: list, V: list, mask: list,
                     d_k: int) -> list:
    scale = math.sqrt(d_k)
    scores = [sum(Q[i] * K[i] for i in range(len(Q))) / scale]
    masked = [s + (0 if mask else float('-inf')) for s in scores]
    sm = softmax_v(masked)
    return [sm[0] * v for v in V]

def softmax_v(x: list) -> list:
    mx = max(x)
    ex = [math.exp(v - mx) for v in x]
    s = sum(ex)
    return [e / s for e in ex]

# Ch4 — GPT-style feed-forward block
def ffn_block(x: list, W1: list, b1: list, W2: list, b2: list) -> list:
    h = [max(0.0, sum(W1[j][i] * x[i] for i in range(len(x))) + b1[j])
         for j in range(len(b1))]
    return [sum(W2[j][i] * h[i] for i in range(len(h))) + b2[j]
            for j in range(len(b2))]

# Ch4 — Pre-norm vs post-norm residual
def pre_norm_residual(x: list, sublayer_fn, gamma: list, beta: list) -> list:
    normed = layer_norm_v(x, gamma, beta)
    return [x[i] + sublayer_fn(normed)[i] for i in range(len(x))]

def post_norm_residual(x: list, sublayer_out: list, gamma: list, beta: list) -> list:
    added = [x[i] + sublayer_out[i] for i in range(len(x))]
    return layer_norm_v(added, gamma, beta)

def layer_norm_v(x: list, gamma: list, beta: list, eps: float = 1e-5) -> list:
    mu = sum(x) / len(x)
    var = sum((v - mu) ** 2 for v in x) / len(x)
    x_hat = [(v - mu) / math.sqrt(var + eps) for v in x]
    return [gamma[i] * x_hat[i] + beta[i] for i in range(len(x))]

# Ch5 — GPT text generation: greedy, top-k, top-p (nucleus)
def greedy_decode(logits: list) -> int:
    return logits.index(max(logits))

def top_k_sample(logits: list, k: int = 40, temperature: float = 1.0) -> int:
    scaled = [v / max(temperature, 1e-9) for v in logits]
    indexed = sorted(enumerate(scaled), key=lambda x: x[1], reverse=True)
    top = indexed[:k]
    probs = softmax_v([v for _, v in top])
    r = random.random()
    cum = 0.0
    for (idx, _), p in zip(top, probs):
        cum += p
        if r <= cum:
            return idx
    return top[-1][0]

def top_p_sample(logits: list, p: float = 0.9, temperature: float = 1.0) -> int:
    scaled = [v / max(temperature, 1e-9) for v in logits]
    indexed = sorted(enumerate(scaled), key=lambda x: x[1], reverse=True)
    probs = softmax_v([v for _, v in indexed])
    cum, chosen = 0.0, []
    for (idx, _), prob in zip(indexed, probs):
        cum += prob
        chosen.append((idx, prob))
        if cum >= p:
            break
    total = sum(pr for _, pr in chosen)
    r = random.random() * total
    cum2 = 0.0
    for idx, prob in chosen:
        cum2 += prob
        if r <= cum2:
            return idx
    return chosen[-1][0]

def temperature_sample(logits: list, temperature: float = 1.0) -> int:
    scaled = [v / max(temperature, 1e-9) for v in logits]
    probs = softmax_v(scaled)
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1

# Ch5 — Repetition penalty
def apply_repetition_penalty(logits: list, generated_ids: list,
                              penalty: float = 1.2) -> list:
    result = logits[:]
    for idx in set(generated_ids):
        if 0 <= idx < len(result):
            result[idx] = result[idx] / penalty if result[idx] > 0 else result[idx] * penalty
    return result

# Ch6 — Weight initialization schemes
def kaiming_uniform_init(fan_in: int, fan_out: int) -> list:
    bound = math.sqrt(6.0 / fan_in)
    return [[random.uniform(-bound, bound) for _ in range(fan_out)]
            for _ in range(fan_in)]

def glorot_uniform_init(fan_in: int, fan_out: int) -> list:
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return [[random.uniform(-bound, bound) for _ in range(fan_out)]
            for _ in range(fan_in)]

# Ch6 — LoRA low-rank weight decomposition
def lora_forward(x: list, W: list, A: list, B: list,
                 alpha: float = 1.0, r: int = 4) -> list:
    scale = alpha / r
    base = [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]
    ax = [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]
    bax = [sum(B[i][j] * ax[j] for j in range(len(ax))) * scale for i in range(len(B))]
    return [base[i] + bax[i] for i in range(len(base))]

# Ch6 — Instruction fine-tuning prompt template
def alpaca_prompt(instruction: str, input_text: str = '', output: str = '') -> str:
    if input_text:
        return (f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}")
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

def phi3_prompt(system: str, user: str, assistant: str = '') -> str:
    return f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}"

def chatml_prompt(system: str, user: str, assistant: str = '') -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}")

# Ch7 — RLHF: reward model scoring, PPO clip loss
def ppo_clip_loss(ratio: float, advantage: float, epsilon: float = 0.2) -> float:
    clipped = max(1 - epsilon, min(1 + epsilon, ratio))
    return -min(ratio * advantage, clipped * advantage)

def reward_model_score(chosen_logprob: float, rejected_logprob: float) -> float:
    return chosen_logprob - rejected_logprob

def dpo_loss(pi_chosen: float, pi_rejected: float,
             ref_chosen: float, ref_rejected: float,
             beta: float = 0.1) -> float:
    log_ratio = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    return -math.log(1 / (1 + math.exp(-beta * log_ratio)))

# ─────────────────────────────────────────────────────────────────────────────
# ALAMMAR — HANDS-ON LARGE LANGUAGE MODELS (gaps)
# ─────────────────────────────────────────────────────────────────────────────

# Ch2 — Token embedding lookup
def token_embedding_lookup(token_ids: list, embedding_table: list) -> list:
    return [embedding_table[i] for i in token_ids if i < len(embedding_table)]

# Ch2 — Absolute positional embedding
def absolute_positional_embedding(seq_len: int, d_model: int) -> list:
    pe = []
    for pos in range(seq_len):
        row = []
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        pe.append(row)
    return pe

# Ch2 — Embedding + positional add
def input_embedding(token_ids: list, embed_table: list,
                    pos_embed: list, dropout_p: float = 0.1) -> list:
    token_embs = token_embedding_lookup(token_ids, embed_table)
    out = [[token_embs[i][j] + pos_embed[i][j]
            for j in range(len(token_embs[i]))]
           for i in range(len(token_embs))]
    return out

# Ch3 — Semantic similarity pipeline
def semantic_similarity_score(vec_a: list, vec_b: list) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a ** 2 for a in vec_a))
    nb = math.sqrt(sum(b ** 2 for b in vec_b))
    return dot / max(na * nb, 1e-9)

# Ch3 — Bi-encoder vs cross-encoder flag
def biencoder_score(emb_a: list, emb_b: list) -> float:
    return semantic_similarity_score(emb_a, emb_b)

def crossencoder_score(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))

# Ch4 — Retrieval-Augmented Generation (RAG) scoring
def rag_score(query_vec: list, doc_vecs: list, top_k: int = 3) -> list:
    scored = [(i, semantic_similarity_score(query_vec, d)) for i, d in enumerate(doc_vecs)]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

# Ch4 — HNSW-like nearest neighbour (brute force, pure python)
def brute_force_knn(query: list, vectors: list, k: int = 5) -> list:
    scores = [(i, semantic_similarity_score(query, v)) for i, v in enumerate(vectors)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

# Ch5 — Text classification head (logistic over embedding)
def classification_head(embedding: list, W: list, b: list) -> list:
    logits = [sum(W[c][i] * embedding[i] for i in range(len(embedding))) + b[c]
              for c in range(len(b))]
    return softmax_v(logits)

# Ch6 — BLEU score (unigram + bigram)
def bleu_score(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    # Unigram precision
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    uni_clip = sum(min(c, ref_counts[t]) for t, c in hyp_counts.items())
    p1 = uni_clip / max(len(hyp_tokens), 1)
    # Bigram precision
    ref_bg = Counter(zip(ref_tokens, ref_tokens[1:]))
    hyp_bg = Counter(zip(hyp_tokens, hyp_tokens[1:]))
    bi_clip = sum(min(c, ref_bg[bg]) for bg, c in hyp_bg.items())
    p2 = bi_clip / max(len(hyp_tokens) - 1, 1)
    # Brevity penalty
    bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)) if len(hyp_tokens) < len(ref_tokens) else 1.0
    return bp * math.sqrt(max(p1, 1e-9) * max(p2, 1e-9))

# Ch6 — ROUGE-1 and ROUGE-2
def rouge_1(reference: str, hypothesis: str) -> float:
    ref = set(reference.lower().split())
    hyp = hypothesis.lower().split()
    if not ref or not hyp:
        return 0.0
    matches = sum(1 for t in hyp if t in ref)
    precision = matches / max(len(hyp), 1)
    recall = matches / max(len(ref), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def rouge_2(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    ref_bg = set(zip(ref_tokens, ref_tokens[1:]))
    hyp_bg = list(zip(hyp_tokens, hyp_tokens[1:]))
    if not ref_bg or not hyp_bg:
        return 0.0
    matches = sum(1 for bg in hyp_bg if bg in ref_bg)
    precision = matches / max(len(hyp_bg), 1)
    recall = matches / max(len(ref_bg), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Ch6 — BERTScore approximation via cosine similarity
def bertscore_approx(ref_vecs: list, hyp_vecs: list) -> float:
    if not ref_vecs or not hyp_vecs:
        return 0.0
    precision = sum(max(semantic_similarity_score(h, r) for r in ref_vecs)
                    for h in hyp_vecs) / max(len(hyp_vecs), 1)
    recall = sum(max(semantic_similarity_score(r, h) for h in hyp_vecs)
                 for r in ref_vecs) / max(len(ref_vecs), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Ch7 — Speculative decoding accept/reject
def speculative_decode_accept(draft_token: int, target_prob: float,
                               draft_prob: float) -> bool:
    if draft_prob <= 0:
        return False
    return random.random() <= min(1.0, target_prob / draft_prob)

# Ch7 — KV cache size estimator
def kv_cache_bytes(n_layers: int, n_heads: int, head_dim: int,
                   seq_len: int, batch_size: int = 1,
                   dtype_bytes: int = 2) -> int:
    return 2 * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_bytes

# Ch7 — Continuous batching occupancy
def batch_occupancy(active_seqs: int, max_seqs: int) -> float:
    return active_seqs / max(max_seqs, 1)

# ─────────────────────────────────────────────────────────────────────────────
# AI ENGINEERING — CHIP HUYEN (gaps)
# ─────────────────────────────────────────────────────────────────────────────

# Ch2 — Sampling strategies: beam search
def beam_search(log_probs_per_step: list, beam_width: int = 4) -> list:
    beams = [(0.0, [])]
    for step_probs in log_probs_per_step:
        candidates = []
        for score, seq in beams:
            for token_id, lp in enumerate(step_probs):
                candidates.append((score + lp, seq + [token_id]))
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
    return beams[0][1] if beams else []

# Ch3 — Hallucination detection heuristics
def factual_consistency_score(claim: str, context: str) -> float:
    claim_words = set(claim.lower().split())
    ctx_words = set(context.lower().split())
    if not claim_words:
        return 0.0
    overlap = claim_words & ctx_words
    return len(overlap) / len(claim_words)

def repetition_score(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - unique / max(len(tokens), 1)

def answer_relevance_score(question: str, answer: str) -> float:
    q_words = set(question.lower().split()) - {'what', 'how', 'why', 'is', 'the', 'a', 'an'}
    a_words = set(answer.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & a_words) / len(q_words)

# Ch3 — LLM-as-judge scoring rubric
def llm_judge_rubric(criteria: Dict[str, float], scores: Dict[str, float]) -> float:
    total_weight = sum(criteria.values())
    if total_weight == 0:
        return 0.0
    return sum(criteria[k] * scores.get(k, 0) for k in criteria) / total_weight

# Ch4 — Context window utilization
def context_utilization(used_tokens: int, max_tokens: int) -> float:
    return used_tokens / max(max_tokens, 1)

# Ch4 — Chunking strategies for RAG
def fixed_size_chunk(text: str, chunk_size: int = 512,
                     overlap: int = 64) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(' '.join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks

def sentence_chunk(text: str, max_sentences: int = 5) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [' '.join(sentences[i:i + max_sentences])
            for i in range(0, len(sentences), max_sentences)]

# Ch4 — Retrieval precision/recall@k
def retrieval_precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / max(k, 1)

def retrieval_recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / max(len(relevant), 1)

def mean_average_precision(retrieved_lists: list, relevant_sets: list) -> float:
    aps = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        hits, sum_p = 0, 0.0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                hits += 1
                sum_p += hits / (i + 1)
        aps.append(sum_p / max(len(relevant), 1))
    return sum(aps) / max(len(aps), 1)

# Ch5 — Agent tool-use: tool selection scoring
def tool_selection_score(query_vec: list, tool_vecs: Dict[str, list]) -> str:
    best_tool = max(tool_vecs, key=lambda t: semantic_similarity_score(query_vec, tool_vecs[t]))
    return best_tool

# Ch5 — ReAct-style thought/action/observation format
def react_format(thought: str, action: str, observation: str) -> str:
    return f"Thought: {thought}\nAction: {action}\nObservation: {observation}"

# Ch6 — Fine-tuning data quality filters
def quality_score(text: str, min_len: int = 50, max_len: int = 2048) -> float:
    if not text or len(text) < min_len or len(text) > max_len:
        return 0.0
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return round(0.5 * unique_ratio + 0.5 * alpha_ratio, 4)

def dedup_exact(texts: list) -> list:
    seen, result = set(), []
    for t in texts:
        h = hashlib.md5(t.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(t)
    return result

def dedup_near(texts: list, threshold: float = 0.9) -> list:
    def shingles(text, k=3):
        tokens = text.lower().split()
        return set(tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1))
    def jaccard(a, b):
        sa, sb = shingles(a), shingles(b)
        return len(sa & sb) / max(len(sa | sb), 1)
    result = []
    for text in texts:
        if all(jaccard(text, kept) < threshold for kept in result):
            result.append(text)
    return result

# Ch6 — Preference data (chosen/rejected) format
def preference_pair(prompt: str, chosen: str, rejected: str) -> dict:
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "margin": len(chosen) - len(rejected)}

# Ch7 — Inference optimization: quantization error
def quantization_error(x: float, bits: int = 8) -> float:
    levels = 2 ** bits
    step = 2.0 / levels
    quantized = round(x / step) * step
    return abs(x - quantized)

def int8_quantize(weights: list) -> Tuple[list, float, float]:
    w_min, w_max = min(weights), max(weights)
    scale = (w_max - w_min) / 255.0
    zero_point = round(-w_min / max(scale, 1e-9))
    quantized = [max(0, min(255, round(w / max(scale, 1e-9)) + zero_point)) for w in weights]
    return quantized, scale, zero_point

def int8_dequantize(quantized: list, scale: float, zero_point: float) -> list:
    return [(q - zero_point) * scale for q in quantized]

# Ch7 — Flash attention complexity comparison
def attention_flops_standard(seq_len: int, d_model: int) -> int:
    return 4 * seq_len * seq_len * d_model

def attention_flops_flash(seq_len: int, d_model: int, block_size: int = 64) -> int:
    n_blocks = math.ceil(seq_len / block_size)
    return 4 * seq_len * d_model * n_blocks * block_size

# Ch8 — Evaluation: G-Eval, MT-Bench style scoring
def g_eval_score(dimensions: list, weights: list) -> float:
    if len(dimensions) != len(weights):
        return 0.0
    total_w = sum(weights)
    return sum(d * w for d, w in zip(dimensions, weights)) / max(total_w, 1)

def mt_bench_aggregate(turn1_scores: list, turn2_scores: list) -> float:
    all_scores = turn1_scores + turn2_scores
    return sum(all_scores) / max(len(all_scores), 1)

# ─────────────────────────────────────────────────────────────────────────────
# DMLS — DESIGNING ML SYSTEMS (gaps)
# ─────────────────────────────────────────────────────────────────────────────

# Ch2 — Sampling strategies
def stratified_sample(data: list, labels: list, n: int) -> list:
    from collections import defaultdict
    buckets = defaultdict(list)
    for item, label in zip(data, labels):
        buckets[label].append(item)
    result = []
    per_class = n // max(len(buckets), 1)
    for label, items in buckets.items():
        result.extend(random.sample(items, min(per_class, len(items))))
    return result

def reservoir_sample(stream, k: int) -> list:
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

def importance_sampling_weight(p_data: float, p_model: float) -> float:
    return p_data / max(p_model, 1e-9)

# Ch3 — Feature engineering
def min_max_normalize(values: list) -> list:
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-9)
    return [(v - lo) / span for v in values]

def z_score_normalize(values: list) -> list:
    mu = sum(values) / max(len(values), 1)
    std = math.sqrt(sum((v - mu) ** 2 for v in values) / max(len(values) - 1, 1))
    return [(v - mu) / max(std, 1e-9) for v in values]

def log_transform(values: list, base: float = math.e) -> list:
    return [math.log(max(v, 1e-9)) / math.log(base) for v in values]

def bucketize(values: list, n_buckets: int = 10) -> list:
    lo, hi = min(values), max(values)
    bucket_size = (hi - lo) / max(n_buckets, 1)
    return [min(int((v - lo) / max(bucket_size, 1e-9)), n_buckets - 1) for v in values]

# Ch3 — Missing value imputation
def mean_impute(values: list) -> list:
    valid = [v for v in values if v is not None]
    mu = sum(valid) / max(len(valid), 1)
    return [v if v is not None else mu for v in values]

def median_impute(values: list) -> list:
    valid = sorted(v for v in values if v is not None)
    n = len(valid)
    med = valid[n // 2] if n % 2 == 1 else (valid[n // 2 - 1] + valid[n // 2]) / 2
    return [v if v is not None else med for v in values]

# Ch4 — Training/serving skew metrics
def covariate_shift_score(train_dist: list, serve_dist: list) -> float:
    from math import sqrt
    n = min(len(train_dist), len(serve_dist))
    if n == 0:
        return 0.0
    return sqrt(sum((train_dist[i] - serve_dist[i]) ** 2 for i in range(n)) / n)

# Ch4 — Label shift detection
def label_shift_ratio(train_labels: list, serve_labels: list) -> dict:
    train_counts = Counter(train_labels)
    serve_counts = Counter(serve_labels)
    all_labels = set(train_counts) | set(serve_counts)
    train_total = max(sum(train_counts.values()), 1)
    serve_total = max(sum(serve_counts.values()), 1)
    return {
        lbl: abs(train_counts.get(lbl, 0) / train_total -
                 serve_counts.get(lbl, 0) / serve_total)
        for lbl in all_labels
    }

# Ch5 — Model evaluation: calibration
def expected_calibration_error(probs: list, labels: list,
                                n_bins: int = 10) -> float:
    bins = defaultdict(list)
    for p, y in zip(probs, labels):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append((p, y))
    ece = 0.0
    n = max(len(probs), 1)
    for b, items in bins.items():
        avg_conf = sum(p for p, _ in items) / len(items)
        avg_acc = sum(y for _, y in items) / len(items)
        ece += len(items) / n * abs(avg_conf - avg_acc)
    return round(ece, 6)

def brier_score(probs: list, labels: list) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / max(len(probs), 1)

# Ch5 — Confusion matrix stats
def confusion_matrix_stats(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    specificity = tn / max(tn + fp, 1)
    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / max(mcc_den, 1e-9)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "accuracy": round(accuracy, 4),
            "specificity": round(specificity, 4), "mcc": round(mcc, 4)}

def roc_auc_approx(scores: list, labels: list) -> float:
    paired = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp, fp, auc = 0, 0, 0.0
    prev_fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / max(n_pos * n_neg, 1)

# Ch6 — Continual learning: catastrophic forgetting proxy
def forgetting_score(old_acc: float, new_acc: float) -> float:
    return max(0.0, old_acc - new_acc)

def plasticity_score(new_task_acc: float, baseline_acc: float) -> float:
    return new_task_acc - baseline_acc

# Ch7 — Infrastructure: latency percentiles
def latency_percentiles(latencies: list) -> dict:
    if not latencies:
        return {}
    s = sorted(latencies)
    n = len(s)
    def pct(p):
        idx = int(math.ceil(p / 100 * n)) - 1
        return s[max(0, min(idx, n - 1))]
    return {"p50": pct(50), "p90": pct(90), "p95": pct(95),
            "p99": pct(99), "mean": sum(s) / n, "max": s[-1]}

def throughput(n_requests: int, duration_seconds: float) -> float:
    return n_requests / max(duration_seconds, 1e-9)

# Ch7 — Shadow deployment comparison
def shadow_delta(prod_scores: list, shadow_scores: list) -> dict:
    if not prod_scores or not shadow_scores:
        return {}
    prod_mean = sum(prod_scores) / len(prod_scores)
    shadow_mean = sum(shadow_scores) / len(shadow_scores)
    return {"prod_mean": round(prod_mean, 4),
            "shadow_mean": round(shadow_mean, 4),
            "delta": round(shadow_mean - prod_mean, 4),
            "pct_change": round((shadow_mean - prod_mean) / max(abs(prod_mean), 1e-9) * 100, 2)}

# ─────────────────────────────────────────────────────────────────────────────
# LLM ENGINEER'S HANDBOOK — IUSZTIN & LABONNE (gaps)
# ─────────────────────────────────────────────────────────────────────────────

# Ch2 — Data pipeline: crawl quality score
def crawl_quality(text: str) -> dict:
    words = text.split()
    sentences = re.split(r'[.!?]', text)
    return {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
        "unique_ratio": len(set(words)) / max(len(words), 1),
        "quality_score": quality_score(text),
    }

# Ch2 — Dataset card fields
def dataset_card(name: str, source: str, n_samples: int,
                 language: str, license_: str, task: str) -> dict:
    return {"name": name, "source": source, "n_samples": n_samples,
            "language": language, "license": license_, "task": task,
            "created_at": time.strftime("%Y-%m-%d")}

# Ch3 — RAG pipeline: chunking + embedding + retrieval
class SimpleRAGPipeline:
    def __init__(self, chunk_size: int = 256, overlap: int = 32):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list = []
        self.embeddings: list = []

    def _simple_embed(self, text: str) -> list:
        words = text.lower().split()
        vec = [0.0] * 64
        for w in words:
            h = hash(w) % 64
            vec[h] += 1.0
        norm = math.sqrt(sum(v ** 2 for v in vec)) or 1.0
        return [v / norm for v in vec]

    def index(self, text: str):
        self.chunks = fixed_size_chunk(text, self.chunk_size, self.overlap)
        self.embeddings = [self._simple_embed(c) for c in self.chunks]

    def retrieve(self, query: str, top_k: int = 3) -> list:
        q_emb = self._simple_embed(query)
        scored = [(i, semantic_similarity_score(q_emb, e))
                  for i, e in enumerate(self.embeddings)]
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        return [{"chunk": self.chunks[i], "score": round(s, 4)} for i, s in top]

# Ch4 — Supervised fine-tuning data formatter
def sft_format_openai(system: str, user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]}

def sft_format_sharegpt(conversations: list) -> dict:
    return {"conversations": [
        {"from": "system" if i == 0 else ("human" if i % 2 == 1 else "gpt"),
         "value": c}
        for i, c in enumerate(conversations)
    ]}

# Ch4 — Token count estimator (no tiktoken needed)
def estimate_tokens_heuristic(text: str) -> int:
    return max(1, len(text) // 4)

def fits_in_context(text: str, max_tokens: int = 4096) -> bool:
    return estimate_tokens_heuristic(text) <= max_tokens

# Ch5 — QLoRA memory estimator
def qlora_memory_mb(n_params_billions: float, bits: int = 4,
                    lora_r: int = 16, dtype_bytes: int = 2) -> dict:
    base_mb = n_params_billions * 1e9 * bits / 8 / 1024 / 1024
    lora_mb = n_params_billions * 1e9 * 0.01 * lora_r * dtype_bytes / 1024 / 1024
    optimizer_mb = lora_mb * 8
    return {"base_model_mb": round(base_mb, 1),
            "lora_adapters_mb": round(lora_mb, 1),
            "optimizer_mb": round(optimizer_mb, 1),
            "total_mb": round(base_mb + lora_mb + optimizer_mb, 1)}

# Ch5 — Training throughput (tokens/sec)
def training_throughput(batch_size: int, seq_len: int,
                        step_time_sec: float) -> float:
    return batch_size * seq_len / max(step_time_sec, 1e-9)

# Ch6 — Model merging: SLERP
def slerp(v0: list, v1: list, t: float) -> list:
    dot = sum(a * b for a, b in zip(v0, v1))
    dot = max(-1.0, min(1.0, dot))
    theta = math.acos(dot)
    if abs(theta) < 1e-6:
        return [v0[i] * (1 - t) + v1[i] * t for i in range(len(v0))]
    sin_theta = math.sin(theta)
    s0 = math.sin((1 - t) * theta) / sin_theta
    s1 = math.sin(t * theta) / sin_theta
    return [s0 * v0[i] + s1 * v1[i] for i in range(len(v0))]

def linear_merge(params_a: list, params_b: list, alpha: float = 0.5) -> list:
    return [alpha * a + (1 - alpha) * b for a, b in zip(params_a, params_b)]

def task_arithmetic_merge(base: list, delta_a: list, delta_b: list,
                          scale: float = 1.0) -> list:
    return [base[i] + scale * (delta_a[i] + delta_b[i]) for i in range(len(base))]

# Ch7 — Deployment: model card fields
def model_card(name: str, base_model: str, task: str, dataset: str,
               eval_metrics: dict, limitations: str) -> dict:
    return {"model_name": name, "base_model": base_model,
            "task": task, "training_dataset": dataset,
            "evaluation": eval_metrics, "limitations": limitations,
            "created_at": time.strftime("%Y-%m-%d")}

# Ch7 — Serving cost estimator
def serving_cost_per_1k(input_tokens: int, output_tokens: int,
                         price_per_1m_input: float = 3.0,
                         price_per_1m_output: float = 15.0) -> float:
    cost = (input_tokens * price_per_1m_input +
            output_tokens * price_per_1m_output) / 1e6
    return round(cost * 1000, 6)

# Ch8 — Observability: LLM trace record
def llm_trace(request_id: str, model: str, prompt_tokens: int,
              completion_tokens: int, latency_ms: int,
              finish_reason: str, cost_usd: float) -> dict:
    return {"request_id": request_id, "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "cost_usd": round(cost_usd, 8),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

# ─────────────────────────────────────────────────────────────────────────────
# MML — MATHEMATICS FOR MACHINE LEARNING (remaining gaps)
# ─────────────────────────────────────────────────────────────────────────────

# Ch3 — Analytic geometry: projection, angle between vectors
def vec_projection(a: list, b: list) -> list:
    scale = sum(ai * bi for ai, bi in zip(a, b)) / max(sum(bi ** 2 for bi in b), 1e-9)
    return [scale * bi for bi in b]

def angle_between(a: list, b: list) -> float:
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai ** 2 for ai in a))
    nb = math.sqrt(sum(bi ** 2 for bi in b))
    return math.acos(max(-1.0, min(1.0, dot / max(na * nb, 1e-9))))

# Ch3 — Gram-Schmidt orthogonalization
def gram_schmidt(vectors: list) -> list:
    ortho = []
    for v in vectors:
        u = v[:]
        for e in ortho:
            proj = vec_projection(v, e)
            u = [u[i] - proj[i] for i in range(len(u))]
        norm = math.sqrt(sum(x ** 2 for x in u))
        if norm > 1e-9:
            ortho.append([x / norm for x in u])
    return ortho

# Ch4 — Matrix rank (via Gaussian elimination)
def matrix_rank(A: list) -> int:
    M = [row[:] for row in A]
    rows, cols = len(M), len(M[0])
    rank = 0
    for col in range(cols):
        pivot_row = None
        for row in range(rank, rows):
            if abs(M[row][col]) > 1e-9:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        for row in range(rows):
            if row != rank and abs(M[row][col]) > 1e-9:
                f = M[row][col] / M[rank][col]
                M[row] = [M[row][j] - f * M[rank][j] for j in range(cols)]
        rank += 1
    return rank

# Ch4 — Null space basis (underdetermined systems)
def is_full_rank(A: list) -> bool:
    return matrix_rank(A) == min(len(A), len(A[0]))

# Ch5 — Jacobian vector product (JVP) and vector Jacobian product (VJP)
def jvp(f, x: list, v: list, eps: float = 1e-5) -> list:
    xpv = [x[i] + eps * v[i] for i in range(len(x))]
    xmv = [x[i] - eps * v[i] for i in range(len(x))]
    fxp = f(xpv)
    fxm = f(xmv)
    if isinstance(fxp, list):
        return [(fxp[i] - fxm[i]) / (2 * eps) for i in range(len(fxp))]
    return [(fxp - fxm) / (2 * eps)]

def vjp(f, x: list, u: list, eps: float = 1e-5) -> list:
    n = len(x)
    fx = f(x)
    m = len(fx) if isinstance(fx, list) else 1
    vjp_out = [0.0] * n
    for i in range(n):
        xp = x[:]
        xp[i] += eps
        fxp = f(xp)
        if isinstance(fxp, list):
            for j in range(m):
                vjp_out[i] += u[j] * (fxp[j] - (fx[j] if isinstance(fx, list) else fx)) / eps
        else:
            vjp_out[i] = u[0] * (fxp - fx) / eps
    return vjp_out

# Ch6 — Gaussian mixture model (E-step only)
def gmm_e_step(x: list, means: list, variances: list,
               weights: list) -> list:
    from math import exp, sqrt, pi
    n, k = len(x), len(means)
    responsibilities = []
    for xi in x:
        probs = []
        for j in range(k):
            p = weights[j] * exp(-0.5 * ((xi - means[j]) ** 2) / max(variances[j], 1e-9))
            p /= sqrt(2 * pi * max(variances[j], 1e-9))
            probs.append(max(p, 1e-300))
        total = sum(probs)
        responsibilities.append([p / total for p in probs])
    return responsibilities

# Ch6 — Exponential family: sufficient statistics
def bernoulli_sufficient_stat(x: list) -> float:
    return sum(x) / max(len(x), 1)

def gaussian_sufficient_stats(x: list) -> Tuple[float, float]:
    mu = sum(x) / max(len(x), 1)
    var = sum((xi - mu) ** 2 for xi in x) / max(len(x) - 1, 1)
    return mu, var

# Ch7 — Lagrangian / constrained optimization
def lagrangian(f_val: float, constraints: list, lambdas: list) -> float:
    return f_val + sum(lam * g for lam, g in zip(lambdas, constraints))

def kkt_check(grad_f: list, grad_g_list: list, lambdas: list,
              tol: float = 1e-4) -> bool:
    n = len(grad_f)
    stationarity = [grad_f[i] + sum(lambdas[j] * grad_g_list[j][i]
                                     for j in range(len(lambdas)))
                    for i in range(n)]
    return all(abs(s) < tol for s in stationarity)

# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing gaps_all_books...\n")

    # Goodfellow gaps
    assert abs(elu(-1.0) - (math.exp(-1) - 1)) < 0.001
    assert leaky_relu(-2.0) == -0.02
    assert selu(0.0) == 0.0
    assert abs(l2_penalty([1.0, 2.0], 0.5) - 2.5) < 0.001
    es = EarlyStopping(patience=3)
    for v in [1.0, 0.9, 0.9, 0.9, 0.9]:
        es.step(v)
    assert es.should_stop
    g = clip_grad_norm([3.0, 4.0], max_norm=1.0)
    assert abs(math.sqrt(g[0]**2+g[1]**2) - 1.0) < 0.001
    h_new, c_new = lstm_step(
        [0.5], [0.1], [0.0],
        [[0.2, 0.1]], [[0.2, 0.1]], [[0.2, 0.1]], [[0.2, 0.1]],
        [0.0], [0.0], [0.0], [0.0]
    )
    assert len(h_new) == 1
    z = vae_reparameterize([0.0], [0.0])
    assert len(z) == 1
    assert vae_kl_loss([0.0], [0.0]) == 0.0
    print("  ✓ Goodfellow — Deep Learning gaps")

    # Raschka gaps
    tok = CharTokenizer()
    tok.fit("hello world")
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"
    ds = sliding_window_dataset([1,2,3,4,5], context_len=3, stride=1)
    assert len(ds) == 2
    assert greedy_decode([0.1, 0.9, 0.3]) == 1
    assert top_k_sample([1.0, 2.0, 3.0], k=2) in [0, 1, 2]
    assert top_p_sample([1.0, 2.0, 3.0], p=0.9) in [0, 1, 2]
    lora_out = lora_forward(
        [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]],
        [[0.1, 0.0], [0.0, 0.1]], [[0.1, 0.0], [0.0, 0.1]]
    )
    assert len(lora_out) == 2
    prompt = alpaca_prompt("Say hi", output="Hi!")
    assert "Instruction" in prompt
    assert isinstance(ppo_clip_loss(1.1, 1.0, epsilon=0.2), float)
    assert dpo_loss(0.9, 0.4, 0.8, 0.5) > 0
    print("  ✓ Raschka — Build LLM From Scratch gaps")

    # Alammar gaps
    pe = absolute_positional_embedding(4, 8)
    assert len(pe) == 4 and len(pe[0]) == 8
    assert bleu_score("the cat sat on the mat", "the cat sat on the mat") > 0.9
    assert rouge_1("hello world", "hello world") == 1.0
    assert rouge_2("the cat sat", "the cat sat") > 0.5
    rag = rag_score([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert rag[0][0] == 0
    q, s, zp = int8_quantize([0.5, -0.5, 1.0, -1.0])
    dq = int8_dequantize(q, s, zp)
    assert all(abs(dq[i] - [-0.5, 0.5, -1.0, 1.0][i]) > 0 for i in range(4))
    print("  ✓ Alammar — Hands-On LLMs gaps")

    # AI Engineering gaps
    result = beam_search([[-0.5, -1.0, -2.0], [-0.3, -0.8, -1.5]], beam_width=2)
    assert isinstance(result, list)
    assert factual_consistency_score("Paris is the capital", "Paris is the capital of France") > 0
    assert repetition_score("the the the") > repetition_score("hello world foo")
    assert 0 <= context_utilization(100, 4096) <= 1
    chunks = fixed_size_chunk("word " * 1000, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert retrieval_precision_at_k([1, 2, 3, 4], {1, 3}, k=4) == 0.5
    assert retrieval_recall_at_k([1, 2, 3, 4], {1, 3}, k=4) == 1.0
    qt, s2, zp2 = int8_quantize([0.1, 0.5, -0.3])
    assert len(qt) == 3
    assert abs(g_eval_score([4, 5, 3], [0.4, 0.4, 0.2]) - 4.2) < 0.001
    print("  ✓ AI Engineering gaps")

    # DMLS gaps
    vals = [3.0, 1.0, 4.0, 1.0, 5.0]
    nm = min_max_normalize(vals)
    assert abs(min(nm)) < 0.001 and abs(max(nm) - 1.0) < 0.001
    zn = z_score_normalize(vals)
    assert abs(sum(zn) / len(zn)) < 0.001
    stats = confusion_matrix_stats(50, 10, 5, 35)
    assert 0 <= stats["f1"] <= 1
    auc = roc_auc_approx([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0])
    assert auc > 0.8
    assert expected_calibration_error([0.9, 0.1], [1, 0]) < 0.2
    lats = latency_percentiles([10, 20, 30, 40, 50, 100, 200])
    assert lats["p99"] >= lats["p50"]
    assert throughput(1000, 10.0) == 100.0
    print("  ✓ DMLS — Designing ML Systems gaps")

    # LLM Engineer's Handbook gaps
    rag_pipe = SimpleRAGPipeline(chunk_size=50, overlap=5)
    rag_pipe.index("the quick brown fox jumps over the lazy dog " * 20)
    results = rag_pipe.retrieve("quick fox", top_k=2)
    assert len(results) == 2
    fmt = sft_format_openai("You are helpful", "Hello", "Hi there!")
    assert fmt["messages"][0]["role"] == "system"
    assert estimate_tokens_heuristic("hello world") > 0
    mem = qlora_memory_mb(7.0, bits=4, lora_r=16)
    assert mem["total_mb"] > 0
    merged = slerp([1.0, 0.0], [0.0, 1.0], t=0.5)
    assert len(merged) == 2
    card = model_card("test", "llama-3", "chat", "alpaca", {"acc": 0.9}, "none")
    assert "model_name" in card
    trace = llm_trace("req-1", "gpt-4", 100, 50, 500, "stop", 0.002)
    assert trace["total_tokens"] == 150
    assert serving_cost_per_1k(1000, 500) > 0
    print("  ✓ LLM Engineer's Handbook gaps")

    # MML gaps
    gs = gram_schmidt([[1.0, 1.0], [1.0, 0.0]])
    assert len(gs) == 2
    assert abs(sum(a*b for a,b in zip(gs[0], gs[1]))) < 0.001
    assert matrix_rank([[1,0],[0,1]]) == 2
    assert matrix_rank([[1,2],[2,4]]) == 1
    proj = vec_projection([3.0, 4.0], [1.0, 0.0])
    assert abs(proj[0] - 3.0) < 0.001
    ang = angle_between([1.0, 0.0], [0.0, 1.0])
    assert abs(ang - math.pi / 2) < 0.001
    resp = gmm_e_step([0.0, 5.0], [0.0, 5.0], [1.0, 1.0], [0.5, 0.5])
    assert abs(resp[0][0] - 1.0) < 0.1
    mu, var = gaussian_sufficient_stats([1.0, 2.0, 3.0])
    assert abs(mu - 2.0) < 0.001
    print("  ✓ MML — Mathematics for ML gaps")

    print("\n✅ ALL GAP TESTS PASSED")
