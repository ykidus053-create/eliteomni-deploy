"""
All 8 books — gaps not yet in EliteOmni.
Blueprint: AI Engineering, DMLS, LLM Handbook, Hands-On LLMs,
Build LLM From Scratch, NLP with Transformers, Deep Learning, MML, ML Engineering
"""
import math, random, re, time, json, hashlib, os, sqlite3, threading
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# BOOK 8 — ML ENGINEERING (Andriy Burkov) — NOT YET IN ELITEOMNI
# ─────────────────────────────────────────────────────────────────────────────

# Ch1 — Problem framing: feasibility check
def ml_feasibility_check(has_data: bool, has_labels: bool,
                          data_size: int, n_features: int) -> dict:
    score = 0
    issues = []
    if has_data: score += 2
    else: issues.append("No data available")
    if has_labels: score += 2
    else: issues.append("No labels — consider self-supervised or weak supervision")
    if data_size >= 1000: score += 2
    elif data_size >= 100: score += 1; issues.append("Small dataset — use transfer learning")
    else: issues.append("Too little data")
    if n_features >= 5: score += 1
    return {"feasible": score >= 5, "score": score,
            "max_score": 7, "issues": issues}

# Ch2 — Data collection: estimating required sample size
def required_sample_size(margin_error: float = 0.05,
                          confidence: float = 0.95,
                          p: float = 0.5) -> int:
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    return math.ceil((z ** 2 * p * (1 - p)) / (margin_error ** 2))

# Ch2 — Data versioning with lineage
def data_lineage_record(dataset_id: str, parent_ids: list,
                         transform: str, row_count: int) -> dict:
    return {"dataset_id": dataset_id,
            "parent_ids": parent_ids,
            "transform": transform,
            "row_count": row_count,
            "hash": hashlib.md5(f"{dataset_id}{parent_ids}{transform}".encode()).hexdigest()[:12],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

# Ch2 — Train/val/test split with stratification check
def split_sizes(n: int, train: float = 0.7,
                val: float = 0.15, test: float = 0.15) -> dict:
    assert abs(train + val + test - 1.0) < 1e-9, "Splits must sum to 1"
    return {"train": int(n * train), "val": int(n * val),
            "test": int(n * test),
            "warning": "Consider stratified split for imbalanced classes"}

# Ch3 — Feature pipeline: schema validation
def validate_feature_schema(record: dict, schema: dict) -> Tuple[bool, list]:
    errors = []
    for field, spec in schema.items():
        if spec.get("required", False) and field not in record:
            errors.append(f"Missing required field: {field}")
            continue
        if field in record:
            val = record[field]
            if "type" in spec and not isinstance(val, spec["type"]):
                errors.append(f"Field {field}: expected {spec['type'].__name__}, got {type(val).__name__}")
            if "min" in spec and val < spec["min"]:
                errors.append(f"Field {field}: {val} < min {spec['min']}")
            if "max" in spec and val > spec["max"]:
                errors.append(f"Field {field}: {val} > max {spec['max']}")
            if "choices" in spec and val not in spec["choices"]:
                errors.append(f"Field {field}: {val} not in {spec['choices']}")
    return len(errors) == 0, errors

# Ch3 — Feature store: point-in-time correctness check
def point_in_time_correct(feature_ts: float, label_ts: float) -> bool:
    return feature_ts < label_ts

# Ch4 — Model selection: cross-validation score aggregator
def cv_score_summary(fold_scores: list) -> dict:
    n = len(fold_scores)
    mean = sum(fold_scores) / max(n, 1)
    variance = sum((s - mean) ** 2 for s in fold_scores) / max(n - 1, 1)
    std = math.sqrt(variance)
    return {"mean": round(mean, 4), "std": round(std, 4),
            "min": round(min(fold_scores), 4),
            "max": round(max(fold_scores), 4),
            "ci_95": (round(mean - 1.96 * std / math.sqrt(n), 4),
                      round(mean + 1.96 * std / math.sqrt(n), 4))}

# Ch4 — Hyperparameter search: random search sampler
def random_search_sample(param_space: dict, n_trials: int = 10) -> list:
    trials = []
    for _ in range(n_trials):
        trial = {}
        for param, spec in param_space.items():
            if spec["type"] == "float":
                if spec.get("log", False):
                    trial[param] = math.exp(random.uniform(math.log(spec["min"]), math.log(spec["max"])))
                else:
                    trial[param] = random.uniform(spec["min"], spec["max"])
            elif spec["type"] == "int":
                trial[param] = random.randint(spec["min"], spec["max"])
            elif spec["type"] == "choice":
                trial[param] = random.choice(spec["values"])
        trials.append(trial)
    return trials

# Ch4 — Bayesian optimization: expected improvement
def expected_improvement(mu: float, sigma: float,
                          best_so_far: float, xi: float = 0.01) -> float:
    if sigma <= 0:
        return 0.0
    z = (mu - best_so_far - xi) / sigma
    norm_cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    norm_pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return (mu - best_so_far - xi) * norm_cdf + sigma * norm_pdf

# Ch5 — Training infrastructure: GPU memory estimator
def gpu_memory_estimate_mb(model_params: int, batch_size: int,
                             seq_len: int, dtype_bytes: int = 4,
                             optimizer: str = "adam") -> dict:
    param_mb = model_params * dtype_bytes / 1024 / 1024
    grad_mb = param_mb
    optimizer_mb = param_mb * {"adam": 2, "sgd": 1, "adafactor": 0.5}.get(optimizer, 2)
    activation_mb = batch_size * seq_len * 64 * dtype_bytes / 1024 / 1024
    total = param_mb + grad_mb + optimizer_mb + activation_mb
    return {"params_mb": round(param_mb, 1), "grads_mb": round(grad_mb, 1),
            "optimizer_mb": round(optimizer_mb, 1),
            "activations_mb": round(activation_mb, 1),
            "total_mb": round(total, 1),
            "total_gb": round(total / 1024, 2)}

# Ch5 — Training: gradient accumulation step count
def gradient_accumulation_steps(target_batch: int, gpu_batch: int) -> int:
    return max(1, math.ceil(target_batch / gpu_batch))

# Ch5 — Mixed precision: loss scaling
def loss_scale_update(scale: float, overflow: bool,
                       scale_factor: float = 2.0,
                       growth_interval: int = 2000,
                       step: int = 1) -> float:
    if overflow:
        return scale / scale_factor
    if step % growth_interval == 0:
        return scale * scale_factor
    return scale

# Ch6 — Model evaluation: McNemar's test for model comparison
def mcnemar_test(model_a_correct: list, model_b_correct: list) -> dict:
    n01 = sum(1 for a, b in zip(model_a_correct, model_b_correct) if not a and b)
    n10 = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a and not b)
    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    statistic = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    p_approx = math.exp(-0.5 * statistic)
    return {"statistic": round(statistic, 4), "n01": n01, "n10": n10,
            "p_value": round(p_approx, 4), "significant": p_approx < 0.05}

# Ch6 — Error analysis: confusion matrix per class
def per_class_metrics(y_true: list, y_pred: list) -> dict:
    classes = sorted(set(y_true) | set(y_pred))
    result = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p != cls)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        result[cls] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "precision": round(precision, 4),
                       "recall": round(recall, 4),
                       "f1": round(f1, 4),
                       "support": tp + fn}
    return result

# Ch6 — Ablation study tracker
class AblationTracker:
    def __init__(self):
        self.runs = []
    def add(self, config: dict, metric: float, metric_name: str = "accuracy"):
        self.runs.append({"config": config, "metric": metric,
                          "metric_name": metric_name, "ts": time.time()})
    def best(self) -> dict:
        if not self.runs: return {}
        return max(self.runs, key=lambda r: r["metric"])
    def summary(self) -> list:
        return sorted(self.runs, key=lambda r: r["metric"], reverse=True)
    def feature_importance(self, baseline_config: dict) -> dict:
        baseline = next((r for r in self.runs if r["config"] == baseline_config), None)
        if not baseline: return {}
        base_score = baseline["metric"]
        return {str(r["config"]): round(r["metric"] - base_score, 4)
                for r in self.runs if r["config"] != baseline_config}

# Ch7 — Serving: request batching with timeout
class DynamicBatcher:
    def __init__(self, max_batch: int = 32, timeout_ms: float = 50.0):
        self.max_batch = max_batch
        self.timeout_ms = timeout_ms
        self.queue = []
        self._lock = threading.Lock()

    def add(self, request: dict) -> int:
        with self._lock:
            self.queue.append({**request, "ts": time.time()})
            return len(self.queue)

    def get_batch(self) -> list:
        with self._lock:
            if not self.queue:
                return []
            now = time.time()
            oldest_age_ms = (now - self.queue[0]["ts"]) * 1000
            if len(self.queue) >= self.max_batch or oldest_age_ms >= self.timeout_ms:
                batch = self.queue[:self.max_batch]
                self.queue = self.queue[self.max_batch:]
                return batch
            return []

    def stats(self) -> dict:
        with self._lock:
            return {"queue_size": len(self.queue),
                    "max_batch": self.max_batch,
                    "timeout_ms": self.timeout_ms}

# Ch7 — Serving: SLA monitoring
def sla_check(latency_p99_ms: float, error_rate: float,
              sla_latency_ms: float = 500.0,
              sla_error_rate: float = 0.01) -> dict:
    latency_ok = latency_p99_ms <= sla_latency_ms
    error_ok = error_rate <= sla_error_rate
    return {"sla_met": latency_ok and error_ok,
            "latency_ok": latency_ok,
            "error_ok": error_ok,
            "latency_headroom_ms": round(sla_latency_ms - latency_p99_ms, 1),
            "error_headroom": round(sla_error_rate - error_rate, 4)}

# Ch7 — Online serving: model warm-up check
def warmup_requests(n_warmup: int = 10) -> list:
    return [{"input": f"warmup_{i}", "skip_logging": True}
            for i in range(n_warmup)]

# Ch8 — Monitoring: PSI (Population Stability Index)
def population_stability_index(expected: list, actual: list,
                                n_bins: int = 10) -> float:
    eps = 1e-9
    all_vals = expected + actual
    lo, hi = min(all_vals), max(all_vals)
    if abs(hi - lo) < eps: return 0.0
    bin_size = (hi - lo) / n_bins
    def bin_counts(data):
        counts = [0] * n_bins
        for v in data:
            b = min(int((v - lo) / bin_size), n_bins - 1)
            counts[b] += 1
        total = max(len(data), 1)
        return [c / total for c in counts]
    exp_pct = bin_counts(expected)
    act_pct = bin_counts(actual)
    psi = sum((a - e) * math.log(max(a, eps) / max(e, eps))
              for e, a in zip(exp_pct, act_pct))
    return round(psi, 4)

# Ch8 — Monitoring: concept drift detection (DDM)
class DDMDriftDetector:
    def __init__(self, alpha_warning: float = 2.0, alpha_drift: float = 3.0):
        self.alpha_warning = alpha_warning
        self.alpha_drift = alpha_drift
        self.n = 0
        self.p = 1.0
        self.s = 0.0
        self.p_min = 1.0
        self.s_min = 0.0

    def update(self, error: int) -> str:
        self.n += 1
        self.p = self.p + (error - self.p) / self.n
        self.s = math.sqrt(self.p * (1 - self.p) / self.n)
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        level = self.p + self.s
        threshold_warning = self.p_min + self.alpha_warning * self.s_min
        threshold_drift = self.p_min + self.alpha_drift * self.s_min
        if level > threshold_drift:
            return "DRIFT"
        if level > threshold_warning:
            return "WARNING"
        return "OK"

    def reset(self):
        self.n = 0; self.p = 1.0; self.s = 0.0
        self.p_min = 1.0; self.s_min = 0.0

# Ch8 — Monitoring: data quality score
def data_quality_score(df_stats: dict) -> dict:
    missing_pct = df_stats.get("missing_pct", 0)
    duplicate_pct = df_stats.get("duplicate_pct", 0)
    outlier_pct = df_stats.get("outlier_pct", 0)
    schema_errors = df_stats.get("schema_errors", 0)
    score = 1.0
    score -= missing_pct * 0.3
    score -= duplicate_pct * 0.2
    score -= outlier_pct * 0.1
    score -= min(schema_errors * 0.05, 0.3)
    return {"quality_score": round(max(0.0, score), 4),
            "missing_penalty": round(missing_pct * 0.3, 4),
            "duplicate_penalty": round(duplicate_pct * 0.2, 4),
            "grade": "A" if score > 0.9 else "B" if score > 0.7 else "C" if score > 0.5 else "F"}

# Ch9 — ML project management: velocity tracking
def sprint_velocity(stories_completed: list) -> dict:
    if not stories_completed: return {}
    avg = sum(stories_completed) / len(stories_completed)
    recent = stories_completed[-3:] if len(stories_completed) >= 3 else stories_completed
    trend = sum(recent) / len(recent) - avg
    return {"average_velocity": round(avg, 2),
            "recent_velocity": round(sum(recent) / len(recent), 2),
            "trend": round(trend, 2),
            "predictable": max(stories_completed) - min(stories_completed) < avg * 0.5}

# ─────────────────────────────────────────────────────────────────────────────
# BOOK 5 — NLP WITH TRANSFORMERS (Tunstall, von Werra, Wolf) — GAPS
# ─────────────────────────────────────────────────────────────────────────────

# Ch2 — Tokenization: WordPiece tokenizer simulation
def wordpiece_tokenize(text: str, vocab: set,
                        unk_token: str = "[UNK]") -> list:
    tokens = []
    for word in text.split():
        if word in vocab:
            tokens.append(word)
            continue
        chars = list(word)
        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            start = end
        tokens.extend([unk_token] if is_bad else sub_tokens)
    return tokens

# Ch2 — Subword tokenization: BPE merge step
def bpe_merge_step(vocab_counts: dict, pair: tuple) -> dict:
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, count in vocab_counts.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = count
    return new_vocab

def bpe_get_pairs(vocab: dict) -> dict:
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

# Ch3 — Transformer: scaled dot-product attention (pure python)
def scaled_dot_product_attention_full(Q: list, K: list, V: list,
                                       mask: list = None) -> list:
    d_k = len(Q[0])
    scale = math.sqrt(d_k)
    scores = [[sum(Q[i][k] * K[j][k] for k in range(d_k)) / scale
               for j in range(len(K))] for i in range(len(Q))]
    if mask:
        scores = [[scores[i][j] + (0 if mask[i][j] else float('-inf'))
                   for j in range(len(scores[0]))] for i in range(len(scores))]
    attn_weights = []
    for row in scores:
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = sum(exps)
        attn_weights.append([e / s for e in exps])
    output = [[sum(attn_weights[i][j] * V[j][k]
                   for j in range(len(V)))
               for k in range(len(V[0]))]
              for i in range(len(Q))]
    return output, attn_weights

# Ch3 — Position-wise feed-forward network
def position_wise_ffn(x: list, W1: list, b1: list,
                       W2: list, b2: list) -> list:
    h = [max(0.0, sum(W1[j][i] * x[i] for i in range(len(x))) + b1[j])
         for j in range(len(b1))]
    return [sum(W2[j][i] * h[i] for i in range(len(h))) + b2[j]
            for j in range(len(b2))]

# Ch4 — Text classification: zero-shot with NLI
def zero_shot_nli_score(premise_emb: list, hypothesis_emb: list) -> float:
    dot = sum(a * b for a, b in zip(premise_emb, hypothesis_emb))
    na = math.sqrt(sum(a ** 2 for a in premise_emb))
    nb = math.sqrt(sum(b ** 2 for b in hypothesis_emb))
    return dot / max(na * nb, 1e-9)

# Ch4 — Token classification: CRF Viterbi decode (simplified)
def viterbi_decode(emissions: list, transitions: list,
                   start_scores: list, end_scores: list) -> list:
    n_steps = len(emissions)
    n_tags = len(emissions[0])
    viterbi = [[float('-inf')] * n_tags for _ in range(n_steps)]
    backpointer = [[0] * n_tags for _ in range(n_steps)]
    for tag in range(n_tags):
        viterbi[0][tag] = start_scores[tag] + emissions[0][tag]
    for step in range(1, n_steps):
        for tag in range(n_tags):
            trans_scores = [viterbi[step - 1][prev] + transitions[prev][tag]
                            for prev in range(n_tags)]
            best_prev = trans_scores.index(max(trans_scores))
            viterbi[step][tag] = max(trans_scores) + emissions[step][tag]
            backpointer[step][tag] = best_prev
    best_last = max(range(n_tags),
                    key=lambda t: viterbi[-1][t] + end_scores[t])
    path = [best_last]
    for step in range(n_steps - 1, 0, -1):
        path.insert(0, backpointer[step][path[0]])
    return path

# Ch5 — Text generation: repetition penalty
def repetition_penalty_logits(logits: list, generated: list,
                               penalty: float = 1.3) -> list:
    result = list(logits)
    for idx in set(generated):
        if 0 <= idx < len(result):
            if result[idx] > 0:
                result[idx] /= penalty
            else:
                result[idx] *= penalty
    return result

# Ch5 — Text generation: length penalty for beam search
def length_penalty(length: int, alpha: float = 0.6) -> float:
    return ((5 + length) / 6) ** alpha

# Ch6 — Summarization: extractive sentence scoring (TextRank-like)
def textrank_scores(sentences: list, sim_fn=None) -> list:
    n = len(sentences)
    if n == 0: return []
    if sim_fn is None:
        def sim_fn(a, b):
            wa = set(a.lower().split())
            wb = set(b.lower().split())
            return len(wa & wb) / max(math.log(len(wa) + 1) + math.log(len(wb) + 1), 1)
    graph = [[sim_fn(sentences[i], sentences[j]) if i != j else 0.0
              for j in range(n)] for i in range(n)]
    scores = [1.0 / n] * n
    for _ in range(50):
        new_scores = []
        for i in range(n):
            row_sum = sum(graph[i])
            incoming = sum(graph[j][i] / max(sum(graph[j]), 1e-9) * scores[j]
                           for j in range(n) if i != j)
            new_scores.append(0.15 / n + 0.85 * incoming)
        scores = new_scores
    return scores

# Ch7 — QA: exact match and F1
def qa_exact_match(prediction: str, ground_truth: str) -> bool:
    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = re.sub(r'[^\w\s]', '', s)
        return re.sub(r'\s+', ' ', s).strip()
    return normalize(prediction) == normalize(ground_truth)

def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    n_common = sum(common.values())
    if n_common == 0: return 0.0
    precision = n_common / max(len(pred_tokens), 1)
    recall = n_common / max(len(gt_tokens), 1)
    return 2 * precision * recall / (precision + recall)

# Ch8 — Model distillation: KL loss for soft labels
def distillation_loss(student_logits: list, teacher_logits: list,
                       hard_labels: list, temperature: float = 2.0,
                       alpha: float = 0.5) -> float:
    def softmax_t(logits, T):
        scaled = [v / T for v in logits]
        mx = max(scaled)
        exps = [math.exp(v - mx) for v in scaled]
        s = sum(exps)
        return [e / s for e in exps]
    teacher_soft = softmax_t(teacher_logits, temperature)
    student_soft = softmax_t(student_logits, temperature)
    kl = sum(t * math.log(max(t, 1e-9) / max(s, 1e-9))
             for t, s in zip(teacher_soft, student_soft))
    kl *= temperature ** 2
    target_idx = hard_labels[0] if hard_labels else 0
    ce_hard = -math.log(max(student_soft[target_idx], 1e-9))
    return alpha * kl + (1 - alpha) * ce_hard

# Ch8 — Knowledge distillation: intermediate layer matching
def layer_matching_loss(student_hidden: list, teacher_hidden: list) -> float:
    return sum((s - t) ** 2 for s, t in zip(student_hidden, teacher_hidden)) / max(len(student_hidden), 1)

# Ch9 — Efficient transformers: linear attention approximation
def linear_attention(Q: list, K: list, V: list,
                     feature_map_fn=None) -> list:
    if feature_map_fn is None:
        feature_map_fn = lambda x: [max(0.0, v) for v in x]
    Q_feat = [feature_map_fn(q) for q in Q]
    K_feat = [feature_map_fn(k) for k in K]
    d = len(V[0])
    kv = [[sum(K_feat[i][j] * V[i][k] for i in range(len(K)))
           for k in range(d)]
          for j in range(len(K_feat[0]))]
    k_sum = [sum(K_feat[i][j] for i in range(len(K)))
             for j in range(len(K_feat[0]))]
    output = []
    for q in Q_feat:
        num = [sum(q[j] * kv[j][k] for j in range(len(q))) for k in range(d)]
        denom = max(sum(q[j] * k_sum[j] for j in range(len(q))), 1e-9)
        output.append([v / denom for v in num])
    return output

# Ch9 — Efficient transformers: sliding window attention mask
def sliding_window_mask(seq_len: int, window_size: int = 4) -> list:
    return [[1 if abs(i - j) <= window_size // 2 else 0
             for j in range(seq_len)] for i in range(seq_len)]

# Ch10 — Training at scale: gradient checkpointing simulation
def checkpoint_memory_savings(n_layers: int, d_model: int,
                               seq_len: int, batch_size: int,
                               checkpoint_every: int = 1) -> dict:
    activation_per_layer = batch_size * seq_len * d_model * 4 / 1024 / 1024
    full_mb = n_layers * activation_per_layer
    checkpointed_layers = math.ceil(n_layers / checkpoint_every)
    saved_mb = full_mb - checkpointed_layers * activation_per_layer
    recompute_mb = (checkpoint_every - 1) * activation_per_layer
    return {"full_activation_mb": round(full_mb, 2),
            "checkpointed_mb": round(checkpointed_layers * activation_per_layer, 2),
            "saved_mb": round(saved_mb, 2),
            "recompute_cost_mb": round(recompute_mb, 2),
            "savings_pct": round(saved_mb / max(full_mb, 1) * 100, 1)}

# ─────────────────────────────────────────────────────────────────────────────
# MISSING FROM AI ENGINEERING (Chip Huyen) — deeper gaps
# ─────────────────────────────────────────────────────────────────────────────

# Ch4 — Context management: lost in the middle detection
def lost_in_middle_score(position: int, total_length: int) -> float:
    relative = position / max(total_length, 1)
    if relative < 0.1 or relative > 0.9:
        return 1.0
    center_dist = abs(relative - 0.5)
    return 0.5 + center_dist

# Ch4 — RAG: parent-child chunking
def parent_child_chunk(text: str, parent_size: int = 1024,
                        child_size: int = 128) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        parent = " ".join(words[i:i + parent_size])
        children = []
        for j in range(i, min(i + parent_size, len(words)), child_size):
            children.append(" ".join(words[j:j + child_size]))
        chunks.append({"parent": parent, "children": children,
                        "parent_start": i, "parent_end": min(i + parent_size, len(words))})
        i += parent_size
    return chunks

# Ch4 — RAG: re-ranking with cross-encoder score
def rerank_with_scores(query: str, docs: list,
                        score_fn=None) -> list:
    if score_fn is None:
        def score_fn(q, d):
            q_words = set(q.lower().split())
            d_words = set(d.lower().split())
            return len(q_words & d_words) / max(len(q_words | d_words), 1)
    scored = [(score_fn(query, doc), doc) for doc in docs]
    return [{"doc": doc, "score": round(score, 4)}
            for score, doc in sorted(scored, reverse=True)]

# Ch5 — Agents: tool use with retry and fallback
def tool_call_with_fallback(tool_fn, args: dict,
                             fallback_fn=None,
                             max_retries: int = 2) -> dict:
    for attempt in range(max_retries):
        try:
            result = tool_fn(**args)
            return {"success": True, "result": result, "attempts": attempt + 1}
        except Exception as e:
            if attempt == max_retries - 1:
                if fallback_fn:
                    try:
                        return {"success": True, "result": fallback_fn(**args),
                                "attempts": attempt + 1, "used_fallback": True}
                    except Exception as fe:
                        return {"success": False, "error": str(fe), "attempts": attempt + 1}
                return {"success": False, "error": str(e), "attempts": attempt + 1}
    return {"success": False, "error": "max retries exceeded"}

# Ch5 — Agents: action space validation
def validate_action(action: dict, allowed_actions: list) -> Tuple[bool, str]:
    action_type = action.get("type", "")
    if action_type not in allowed_actions:
        return False, f"Action '{action_type}' not in allowed: {allowed_actions}"
    required_fields = {"search": ["query"], "code": ["code"],
                        "write": ["path", "content"], "read": ["path"]}
    required = required_fields.get(action_type, [])
    for field in required:
        if field not in action:
            return False, f"Missing required field '{field}' for action '{action_type}'"
    return True, "ok"

# Ch6 — RLHF: Bradley-Terry preference model
def bradley_terry_score(wins: dict, losses: dict, n_iter: int = 100) -> dict:
    players = list(set(list(wins.keys()) + list(losses.keys())))
    scores = {p: 1.0 for p in players}
    for _ in range(n_iter):
        new_scores = {}
        for p in players:
            w = wins.get(p, 0)
            total_games = w + losses.get(p, 0)
            if total_games == 0:
                new_scores[p] = scores[p]
                continue
            denom = sum(scores[p] / (scores[p] + scores.get(opp, 1.0))
                        for opp in players if opp != p)
            new_scores[p] = w / max(denom, 1e-9)
        total = sum(new_scores.values())
        scores = {p: v / total for p, v in new_scores.items()}
    return {p: round(s, 4) for p, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

# Ch6 — RLHF: reward model margin loss
def reward_margin_loss(r_chosen: float, r_rejected: float,
                        margin: float = 0.0) -> float:
    return max(0.0, margin - (r_chosen - r_rejected))

# Ch7 — Inference: continuous batching simulation
class ContinuousBatcher:
    def __init__(self, max_tokens_in_flight: int = 4096):
        self.max_tokens = max_tokens_in_flight
        self.active = []
        self.completed = []

    def add_request(self, req_id: str, prompt_len: int, max_gen: int):
        self.active.append({"id": req_id, "prompt_len": prompt_len,
                             "max_gen": max_gen, "generated": 0})

    def step(self) -> list:
        tokens_used = sum(r["prompt_len"] + r["generated"] for r in self.active)
        finished = []
        for req in self.active:
            req["generated"] += 1
            if req["generated"] >= req["max_gen"]:
                finished.append(req)
        for req in finished:
            self.active.remove(req)
            self.completed.append(req)
        return finished

    def utilization(self) -> float:
        tokens_used = sum(r["prompt_len"] + r["generated"] for r in self.active)
        return tokens_used / max(self.max_tokens, 1)

# Ch7 — Inference: speculative decoding acceptance rate
def speculative_acceptance_rate(draft_tokens: list, target_probs: list,
                                  draft_probs: list) -> float:
    accepted = sum(1 for tp, dp in zip(target_probs, draft_probs)
                   if random.random() <= min(1.0, tp / max(dp, 1e-9)))
    return accepted / max(len(draft_tokens), 1)

# Ch8 — Evaluation: LLM-as-judge with criteria weighting
def llm_judge_weighted(scores: dict, weights: dict) -> float:
    total_w = sum(weights.values())
    if total_w == 0: return 0.0
    return sum(scores.get(k, 0) * w for k, w in weights.items()) / total_w

# Ch8 — Evaluation: G-Eval normalized score
def g_eval_normalized(raw_scores: list, scale: float = 5.0) -> dict:
    if not raw_scores: return {}
    normalized = [s / scale for s in raw_scores]
    avg = sum(normalized) / len(normalized)
    return {"raw_scores": raw_scores,
            "normalized": [round(n, 3) for n in normalized],
            "mean": round(avg, 3),
            "grade": "excellent" if avg > 0.8 else "good" if avg > 0.6 else "needs_improvement"}

# ─────────────────────────────────────────────────────────────────────────────
# MISSING FROM DMLS (Chip Huyen) — deeper gaps
# ─────────────────────────────────────────────────────────────────────────────

# Ch5 — ML system design: two-tower model scoring
def two_tower_score(user_emb: list, item_emb: list) -> float:
    dot = sum(u * i for u, i in zip(user_emb, item_emb))
    nu = math.sqrt(sum(u ** 2 for u in user_emb))
    ni = math.sqrt(sum(i ** 2 for i in item_emb))
    return dot / max(nu * ni, 1e-9)

# Ch5 — Recommendation: NDCG at K
def ndcg_at_k(relevance_scores: list, k: int = 10) -> float:
    def dcg(scores, k):
        return sum(s / math.log2(i + 2)
                   for i, s in enumerate(scores[:k]))
    actual = dcg(relevance_scores, k)
    ideal = dcg(sorted(relevance_scores, reverse=True), k)
    return actual / max(ideal, 1e-9)

# Ch6 — Deployment: blue/green deployment health
def blue_green_health(blue_metrics: dict, green_metrics: dict,
                       thresholds: dict) -> dict:
    green_better = {}
    for metric, threshold in thresholds.items():
        b = blue_metrics.get(metric, 0)
        g = green_metrics.get(metric, 0)
        green_better[metric] = {"blue": b, "green": g,
                                 "green_wins": g >= b - threshold}
    promote = all(v["green_wins"] for v in green_better.values())
    return {"promote_green": promote, "metrics": green_better}

# Ch7 — Infrastructure: auto-scaling policy
def autoscale_decision(current_replicas: int, avg_cpu_pct: float,
                        avg_latency_ms: float,
                        target_cpu: float = 70.0,
                        target_latency_ms: float = 200.0,
                        min_replicas: int = 1,
                        max_replicas: int = 20) -> dict:
    cpu_ratio = avg_cpu_pct / max(target_cpu, 1)
    latency_ratio = avg_latency_ms / max(target_latency_ms, 1)
    scale_factor = max(cpu_ratio, latency_ratio)
    desired = math.ceil(current_replicas * scale_factor)
    desired = max(min_replicas, min(max_replicas, desired))
    action = "scale_up" if desired > current_replicas else "scale_down" if desired < current_replicas else "maintain"
    return {"current": current_replicas, "desired": desired,
            "action": action, "cpu_ratio": round(cpu_ratio, 2),
            "latency_ratio": round(latency_ratio, 2)}

# ─────────────────────────────────────────────────────────────────────────────
# MISSING FROM LLM ENGINEER'S HANDBOOK — deeper gaps
# ─────────────────────────────────────────────────────────────────────────────

# Ch3 — RAG: multi-query retrieval
def multi_query_retrieve(query: str, generate_queries_fn,
                          retrieve_fn, top_k: int = 3) -> list:
    try:
        queries = generate_queries_fn(query)
    except Exception:
        queries = [query]
    all_results = []
    seen = set()
    for q in [query] + queries:
        for result in retrieve_fn(q, top_k):
            key = result.get("id", result.get("text", "")[:50])
            if key not in seen:
                seen.add(key)
                all_results.append(result)
    return all_results[:top_k * 2]

# Ch5 — Fine-tuning: data mixing ratio
def data_mixing_ratio(datasets: dict, strategy: str = "proportional") -> dict:
    total = sum(datasets.values())
    if strategy == "proportional":
        return {k: round(v / total, 4) for k, v in datasets.items()}
    if strategy == "sqrt":
        sqrt_counts = {k: math.sqrt(v) for k, v in datasets.items()}
        total_sqrt = sum(sqrt_counts.values())
        return {k: round(v / total_sqrt, 4) for k, v in sqrt_counts.items()}
    return {k: round(1.0 / len(datasets), 4) for k in datasets}

# Ch6 — Model merging: DARE (weight sparsification before merge)
def dare_sparsify(delta: list, density: float = 0.5,
                  scale: float = 1.0) -> list:
    n = len(delta)
    n_keep = max(1, int(n * density))
    threshold = sorted([abs(v) for v in delta], reverse=True)[n_keep - 1]
    return [v * scale / density if abs(v) >= threshold else 0.0 for v in delta]

# Ch8 — LLMOps: experiment registry
class ExperimentRegistry:
    def __init__(self):
        self.experiments = {}

    def register(self, name: str, config: dict,
                  metrics: dict, artifacts: dict = None) -> str:
        exp_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        self.experiments[exp_id] = {
            "name": name, "config": config, "metrics": metrics,
            "artifacts": artifacts or {},
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "completed"}
        return exp_id

    def compare(self, exp_ids: list, metric: str) -> list:
        results = []
        for eid in exp_ids:
            exp = self.experiments.get(eid)
            if exp:
                results.append({"id": eid, "name": exp["name"],
                                 "value": exp["metrics"].get(metric)})
        return sorted(results, key=lambda x: x["value"] or 0, reverse=True)

    def best(self, metric: str) -> dict:
        if not self.experiments: return {}
        return max(self.experiments.values(),
                   key=lambda e: e["metrics"].get(metric, float('-inf')))

# ─────────────────────────────────────────────────────────────────────────────
# MISSING FROM BUILD LLM FROM SCRATCH — deeper gaps
# ─────────────────────────────────────────────────────────────────────────────

# Ch3 — RoPE: apply rotation to Q/K vectors
def apply_rope(x: list, pos: int, base: float = 10000.0) -> list:
    d = len(x)
    result = list(x)
    for i in range(0, d, 2):
        theta = pos / (base ** (i / d))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        if i + 1 < d:
            result[i] = x[i] * cos_t - x[i + 1] * sin_t
            result[i + 1] = x[i] * sin_t + x[i + 1] * cos_t
    return result

# Ch4 — Grouped-query attention (GQA) head grouping
def gqa_head_groups(n_heads: int, n_kv_heads: int) -> list:
    assert n_heads % n_kv_heads == 0
    group_size = n_heads // n_kv_heads
    return [[i * group_size + j for j in range(group_size)]
            for i in range(n_kv_heads)]

# Ch4 — SwiGLU forward
def swiglu_forward(x: list, W_gate: list, W_up: list,
                    W_down: list) -> list:
    def silu(v): return v / (1 + math.exp(-v))
    gate = [silu(sum(W_gate[i][j] * x[j] for j in range(len(x))))
            for i in range(len(W_gate))]
    up = [sum(W_up[i][j] * x[j] for j in range(len(x)))
          for i in range(len(W_up))]
    hidden = [gate[i] * up[i] for i in range(len(gate))]
    return [sum(W_down[i][j] * hidden[j] for j in range(len(hidden)))
            for i in range(len(W_down))]

# Ch5 — Decoding: contrastive search
def contrastive_search(logits: list, prev_embeddings: list,
                        candidate_embeddings: list,
                        alpha: float = 0.6, k: int = 5) -> int:
    probs = []
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    probs = [e / s for e in exps]
    top_k_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    def cos_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x ** 2 for x in a))
        nb = math.sqrt(sum(x ** 2 for x in b))
        return dot / max(na * nb, 1e-9)
    best_idx, best_score = top_k_idx[0], float('-inf')
    for idx in top_k_idx:
        if idx >= len(candidate_embeddings):
            continue
        model_conf = probs[idx]
        if prev_embeddings:
            degeneration = max(cos_sim(candidate_embeddings[idx], pe)
                               for pe in prev_embeddings)
        else:
            degeneration = 0.0
        score = (1 - alpha) * model_conf - alpha * degeneration
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx

# Ch7 — ORPO loss (odds ratio preference optimization)
def orpo_loss(chosen_logprob: float, rejected_logprob: float,
               beta: float = 0.1) -> float:
    log_odds_chosen = chosen_logprob - math.log(max(1 - math.exp(chosen_logprob), 1e-9))
    log_odds_rejected = rejected_logprob - math.log(max(1 - math.exp(rejected_logprob), 1e-9))
    ratio = log_odds_chosen - log_odds_rejected
    return -math.log(1 / (1 + math.exp(-beta * ratio)))

# ─────────────────────────────────────────────────────────────────────────────
# MISSING FROM HANDS-ON LLMS — deeper gaps
# ─────────────────────────────────────────────────────────────────────────────

# Ch5 — Semantic search: approximate nearest neighbor with LSH
class LSHIndex:
    def __init__(self, d: int, n_planes: int = 8):
        self.planes = [[random.gauss(0, 1) for _ in range(d)]
                       for _ in range(n_planes)]
        self.buckets = defaultdict(list)

    def _hash(self, vec: list) -> str:
        return "".join("1" if sum(vec[j] * self.planes[i][j]
                                  for j in range(len(vec))) > 0 else "0"
                       for i in range(len(self.planes)))

    def add(self, vec: list, doc_id: str):
        h = self._hash(vec)
        self.buckets[h].append((doc_id, vec))

    def query(self, vec: list, top_k: int = 5) -> list:
        h = self._hash(vec)
        candidates = self.buckets.get(h, [])
        if not candidates:
            candidates = [item for bucket in self.buckets.values()
                          for item in bucket]
        def cos_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x ** 2 for x in a))
            nb = math.sqrt(sum(y ** 2 for y in b))
            return dot / max(na * nb, 1e-9)
        scored = [(cos_sim(vec, v), doc_id) for doc_id, v in candidates]
        return [{"id": d, "score": round(s, 4)}
                for s, d in sorted(scored, reverse=True)[:top_k]]

# Ch6 — Fine-tuning: instruction following quality
def instruction_following_score(response: str, instruction: str) -> dict:
    instr_lower = instruction.lower()
    resp_lower = response.lower()
    format_checks = {
        "json_requested": "json" in instr_lower,
        "list_requested": any(w in instr_lower for w in ["list", "enumerate", "bullet"]),
        "short_requested": any(w in instr_lower for w in ["brief", "short", "concise", "one sentence"]),
        "code_requested": any(w in instr_lower for w in ["code", "function", "implement"]),
    }
    format_scores = {}
    if format_checks["json_requested"]:
        try: json.loads(resp_lower); format_scores["json"] = 1.0
        except: format_scores["json"] = 0.0
    if format_checks["list_requested"]:
        has_list = bool(re.search(r"^\s*[-*•\d]", response, re.MULTILINE))
        format_scores["list"] = 1.0 if has_list else 0.3
    if format_checks["short_requested"]:
        format_scores["brevity"] = 1.0 if len(response.split()) < 50 else 0.0
    if format_checks["code_requested"]:
        format_scores["code"] = 1.0 if "```" in response or "def " in response else 0.0
    overall = sum(format_scores.values()) / max(len(format_scores), 1) if format_scores else 0.7
    return {"overall": round(overall, 3), "format_scores": format_scores}

# Ch7 — Quantization: GGUF-style block quantization error
def block_quantization_error(weights: list, block_size: int = 32,
                              bits: int = 4) -> dict:
    n_blocks = math.ceil(len(weights) / block_size)
    total_error = 0.0
    for b in range(n_blocks):
        block = weights[b * block_size:(b + 1) * block_size]
        if not block: continue
        w_min, w_max = min(block), max(block)
        scale = (w_max - w_min) / max(2 ** bits - 1, 1)
        for w in block:
            q = round((w - w_min) / max(scale, 1e-9))
            w_dq = w_min + q * scale
            total_error += (w - w_dq) ** 2
    mse = total_error / max(len(weights), 1)
    return {"mse": round(mse, 8), "rmse": round(math.sqrt(mse), 8),
            "bits": bits, "block_size": block_size, "n_blocks": n_blocks}

# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing book8_gaps...\n")

    # ML Engineering (Burkov)
    feas = ml_feasibility_check(True, True, 5000, 20)
    assert feas["feasible"]
    assert required_sample_size(0.05, 0.95) > 300
    sizes = split_sizes(1000)
    assert sizes["train"] + sizes["val"] + sizes["test"] == 1000
    ok, errs = validate_feature_schema({"age": 25, "name": "Alice"},
                                        {"age": {"type": int, "min": 0, "max": 120, "required": True},
                                         "name": {"type": str, "required": True}})
    assert ok and errs == []
    trials = random_search_sample({"lr": {"type": "float", "min": 1e-5, "max": 1e-1, "log": True},
                                    "layers": {"type": "int", "min": 1, "max": 8}}, n_trials=5)
    assert len(trials) == 5
    ei = expected_improvement(0.85, 0.1, 0.80)
    assert ei >= 0
    mem = gpu_memory_estimate_mb(7_000_000_000, 8, 512)
    assert mem["total_mb"] > 0
    psi = population_stability_index([random.gauss(0, 1) for _ in range(100)],
                                      [random.gauss(0.1, 1) for _ in range(100)])
    assert psi >= 0
    det = DDMDriftDetector()
    for e in [0]*50 + [1]*20: det.update(e)
    dq = data_quality_score({"missing_pct": 0.02, "duplicate_pct": 0.01, "outlier_pct": 0.05, "schema_errors": 0})
    assert dq["quality_score"] > 0.8
    cm = per_class_metrics([0,1,1,0,1], [0,1,0,0,1])
    assert 0 in cm and 1 in cm
    ab = AblationTracker()
    ab.add({"layers": 4}, 0.85); ab.add({"layers": 8}, 0.90)
    assert ab.best()["metric"] == 0.90
    sla = sla_check(300.0, 0.005)
    assert sla["sla_met"]
    print("  ✓ ML Engineering (Burkov) — all gaps")

    # NLP with Transformers
    vocab = {"hello", "world", "##ing", "##ly", "[UNK]"}
    tokens = wordpiece_tokenize("hello worldly", vocab)
    assert "hello" in tokens
    Q = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 2.0], [3.0, 4.0]]
    out, attn = scaled_dot_product_attention_full(Q, K, V)
    assert len(out) == 2
    path = viterbi_decode([[0.8, 0.2], [0.3, 0.7]], [[0.7, 0.3], [0.4, 0.6]], [0.6, 0.4], [0.5, 0.5])
    assert len(path) == 2
    assert qa_exact_match("the cat", "the cat")
    assert qa_f1_score("the cat sat", "the cat sat on the mat") > 0.5
    dl = distillation_loss([1.0, 2.0, 3.0], [0.5, 2.5, 3.5], [2], temperature=2.0)
    assert dl > 0
    scores = textrank_scores(["The cat sat.", "A dog ran.", "The cat ran fast."])
    assert len(scores) == 3
    ck = checkpoint_memory_savings(24, 768, 512, 8, checkpoint_every=4)
    assert ck["savings_pct"] > 0
    print("  ✓ NLP with Transformers (Tunstall et al.) — all gaps")

    # AI Engineering deeper gaps
    chunks = parent_child_chunk("word " * 200, parent_size=50, child_size=10)
    assert len(chunks) > 0 and "children" in chunks[0]
    reranked = rerank_with_scores("cat food", ["cat food recipe", "dog toys", "cat treats"])
    assert reranked[0]["score"] >= reranked[1]["score"]
    bt = bradley_terry_score({"A": 3, "B": 1}, {"A": 1, "B": 3})
    assert "A" in bt and "B" in bt
    batcher = ContinuousBatcher(max_tokens_in_flight=1024)
    batcher.add_request("r1", 100, 50)
    batcher.add_request("r2", 200, 30)
    finished = batcher.step()
    assert batcher.utilization() > 0
    gn = g_eval_normalized([4.0, 3.5, 4.5], scale=5.0)
    assert 0 < gn["mean"] < 1
    print("  ✓ AI Engineering (Huyen) — deeper gaps")

    # DMLS deeper gaps
    u = [0.3, 0.5, 0.2]; it = [0.3, 0.5, 0.2]
    assert abs(two_tower_score(u, it) - 1.0) < 0.01
    ndcg = ndcg_at_k([3, 2, 3, 0, 1, 2], k=6)
    assert 0 < ndcg <= 1
    scale_dec = autoscale_decision(5, 85.0, 350.0)
    assert scale_dec["action"] in ("scale_up", "scale_down", "maintain")
    print("  ✓ DMLS (Huyen) — deeper gaps")

    # LLM Handbook deeper gaps
    ratios = data_mixing_ratio({"general": 10000, "code": 2000, "math": 1000})
    assert abs(sum(ratios.values()) - 1.0) < 0.01
    delta = [0.1 * i - 0.5 for i in range(20)]
    sparsified = dare_sparsify(delta, density=0.5)
    assert sum(1 for v in sparsified if v == 0) >= 5
    reg = ExperimentRegistry()
    eid = reg.register("exp1", {"lr": 1e-4}, {"acc": 0.92, "f1": 0.90})
    assert reg.best("acc")["metrics"]["acc"] == 0.92
    print("  ✓ LLM Engineer's Handbook — deeper gaps")

    # Build LLM From Scratch deeper gaps
    rotated = apply_rope([1.0, 0.0, 0.5, 0.5], pos=5)
    assert len(rotated) == 4
    groups = gqa_head_groups(8, 2)
    assert len(groups) == 2 and len(groups[0]) == 4
    token = contrastive_search([0.1, 0.8, 0.5, 0.3],
                                [[0.9, 0.1], [0.2, 0.8]],
                                [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5], [0.1, 0.9]])
    assert 0 <= token <= 3
    orpo = orpo_loss(-0.5, -1.5)
    assert orpo > 0
    print("  ✓ Build LLM From Scratch (Raschka) — deeper gaps")

    # Hands-On LLMs deeper gaps
    lsh = LSHIndex(d=4, n_planes=4)
    lsh.add([1.0, 0.0, 0.5, 0.5], "doc1")
    lsh.add([0.0, 1.0, 0.2, 0.8], "doc2")
    results = lsh.query([0.9, 0.1, 0.4, 0.6], top_k=2)
    assert len(results) > 0
    ifs = instruction_following_score("```python\ndef foo(): pass\n```", "write a python function")
    assert ifs["overall"] > 0
    bqe = block_quantization_error([random.gauss(0, 1) for _ in range(64)], block_size=32, bits=4)
    assert bqe["mse"] >= 0
    print("  ✓ Hands-On LLMs (Alammar) — deeper gaps")

    print("\n✅ ALL BOOK8_GAPS TESTS PASSED")
