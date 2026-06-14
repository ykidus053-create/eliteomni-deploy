# AUTO-SPLIT FROM app.py lines 23-340
# ── Parallel module loader — imports all heavy modules concurrently ──────────
import importlib, threading as _th
_mod_results = {}
_mod_errors  = {}

def _load(name):
    try:
        _mod_results[name] = importlib.import_module(name)
    except Exception as e:
        _mod_errors[name] = e

_threads = [_th.Thread(target=_load, args=(m,), daemon=True)
            for m in ("book_gaps_impl","aie_book_impl","final_gaps","book8_gaps")]
for t in _threads: t.start()
for t in _threads: t.join()

# Inject symbols into global namespace exactly as before
def _inject(mod_name, symbols):
    mod = _mod_results.get(mod_name)
    if mod is None:
        print(f"[{mod_name}] ❌ {_mod_errors.get(mod_name)}")
        return False
    for s in symbols:
        if hasattr(mod, s): globals()[s] = getattr(mod, s)
    print(f"[{mod_name}] ✅ loaded")
    return True

_GAPS_LOADED = _inject("book_gaps_impl", [
    "detect_data_drift","detect_training_serving_skew","feature_store_set","feature_store_get",
    "monitor_metric","check_continual_learning_trigger","compute_class_weights","oversample_minority",
    "ExperimentTracker","registry_register","registry_promote","registry_get_production",
    "pipeline_log","LLMTwinProfile","sts_benchmark","token_importance","concept_vector_score",
    "cluster_texts","BPETokenizer","sinusoidal_positional_encoding","scaled_dot_product_attention",
    "layer_norm","multi_head_attention_scores","monitor_gradient_norm","detect_vanishing_gradients",
    "LRScheduler","DropoutTracker","batch_norm","xavier_init","he_init",
    "eval_multiturn","HumanInLoopBatchScorer","PromptChain","gap_dashboard",
])

_AIE_LOADED = _inject("aie_book_impl", [
    "bleu_score","rouge_l","embedding_similarity","llm_judge_score","estimate_perplexity",
    "build_scoring_rubric","required_sample_size","golden_set_add","run_eval_suite","get_eval_report",
    "prompt_save","prompt_load","prompt_compare","harden_input","enforce_output_format",
    "rag_index_document","bm25_retrieve","hybrid_retrieve","rerank_results","query_rewrite",
    "mrr_score","ndcg_score","sft_sample_add","dpo_pair_add","export_sft_jsonl","get_lora_config",
    "dedup_dataset","quality_filter","diversity_sample","get_annotation_guidelines",
    "check_annotation_quality","build_synthetic_dataset","InferenceMetricsTracker",
    "get_inference_report","get_prompt_cache_tracker","check_quantization_readiness",
    "record_user_feedback","ab_test_record","ab_test_results","model_selection_pipeline",
    "get_feedback_report","get_guardrail","aie_dashboard",
])

_inject("final_gaps", [
    "PReLU","sigmoid","sigmoid_grad","tanh_grad","softplus","mish",
    "inverted_dropout","max_norm_constraint","bagging_predict","cyclical_lr","gradient_noise",
    "receptive_field","conv_output_size","reconstruction_loss_mse","reconstruction_loss_bce",
    "log_sum_exp","KVCache","gpt_param_count","min_p_sample","gradient_checkpointing_memory",
    "ia3_forward","sft_loss_masked","kl_from_reference","mean_pooling","cls_pooling",
    "max_pooling","ner_decode","extractive_coverage","gptq_block_error","throughput_latency_tradeoff",
    "eta_sample","detect_prompt_injection","validate_json_output","AgentLoopGuard",
    "build_tool_schema","dedup_tool_calls","cohens_kappa","majority_vote_label","feature_cross",
    "target_encode","data_version_hash","slice_metrics","canary_health_check","traffic_split",
    "pack_sequences","format_multiturn","peft_efficiency","ties_merge","deployment_readiness",
    "compare_experiments","rag_faithfulness","det2","det3","mat_inv2","is_positive_definite",
    "empirical_distribution","is_convex_1d",
])

_inject("book8_gaps", [
    "ml_feasibility_check","required_sample_size","data_lineage_record","validate_feature_schema",
    "cv_score_summary","random_search_sample","expected_improvement","gpu_memory_estimate_mb",
    "gradient_accumulation_steps","population_stability_index","DDMDriftDetector","data_quality_score",
    "per_class_metrics","AblationTracker","DynamicBatcher","sla_check","mcnemar_test",
    "parent_child_chunk","rerank_with_scores","tool_call_with_fallback","validate_action",
    "bradley_terry_score","reward_margin_loss","ContinuousBatcher","speculative_acceptance_rate",
    "llm_judge_weighted","g_eval_normalized","two_tower_score","ndcg_at_k","blue_green_health",
    "autoscale_decision","wordpiece_tokenize","scaled_dot_product_attention_full","viterbi_decode",
    "qa_exact_match","qa_f1_score","distillation_loss","textrank_scores","checkpoint_memory_savings",
    "apply_rope","gqa_head_groups","swiglu_forward","contrastive_search","orpo_loss",
    "data_mixing_ratio","dare_sparsify","ExperimentRegistry","LSHIndex","instruction_following_score",
    "block_quantization_error","length_penalty","repetition_penalty_logits","sliding_window_mask",
    "linear_attention","zero_shot_nli_score","layer_matching_loss","sprint_velocity","loss_scale_update",
])

# ── SELF-WIRING PIPELINE — change anything → auto-reload+index+SFT ──────────
try:
    from self_wire import start as _self_wire_start
    _self_wire_start(interval=30.0)
except Exception as _swe:
    print(f"[self_wire] ❌ {_swe}")
# ─────────────────────────────────────────────────────────────────────────────

# ── KNOWLEDGE RAG — book functions injected into prompts ─────────────────────
try:
    from knowledge_rag import start_background_indexer as _kb_start, get_knowledge_context
    _kb_start()
    print("[knowledge_rag] ✅ background indexer started")
except Exception as _kre:
    print(f"[knowledge_rag] ❌ {_kre}")
    def get_knowledge_context(q, top_k=5): return ""
# ─────────────────────────────────────────────────────────────────────────────

# ── HOT RELOAD — auto-reimports any changed .py file instantly ───────────────
if False:  # disabled — self_wire already handles hot reload
    pass
if False:
    import os as _os
    from hot_reload import start as _hot_reload_start
    _hot_reload_start(_os.path.dirname(_os.path.abspath(__file__)), interval=1.0)
    print("[hot_reload] ✅ watching for file changes")
# ─────────────────────────────────────────────────────────────────────────────

# ── AUTODISCOVERY — any new .py file gets auto-imported on restart ────────────
if False:  # disabled — self_wire already handles this
    from autoloader import autodiscover as _autodiscover
    import os as _os
    _autodiscover(_os.path.dirname(_os.path.abspath(__file__)), verbose=True)
# ─────────────────────────────────────────────────────────────────────────────

# ── AUTO-WIRED MODULES (unwired files) ──────────────────────────────────────

try:
    from dl_book_implementations import (EarlyStopping, bootstrap_sample, BaggingEnsemble, dropout_mask, dropout_forward, SGDMomentum, RMSProp, Adam, glorot_uniform, glorot_init_layer, batch_norm_forward, curriculum_sort, score_example_difficulty, random_log_uniform, random_search_config, HyperparameterTracker, importance_sampling_estimate, self_normalised_importance_sampling, optimal_proposal_distribution, get_optimizer, build_curriculum, ensemble_rlaif_score_sync)
    print("[dl_book_implementations] ✅ 22 functions loaded")
except Exception as _e:
    print(f"[dl_book_implementations] ❌ {_e}")

try:
    from dl_book_implementations2 import (gradient_descent_step, numerical_gradient, lagrangian, kkt_satisfied, bias_variance_decompose, k_fold_split, log_likelihood_gaussian, mle_gaussian, mle_bernoulli, minibatch_sgd, relu, relu_grad, leaky_relu, elu, sigmoid, tanh_act, softmax, backprop_chain_rule, SimpleMLP, l2_penalty, l1_penalty, l2_grad, l1_grad, regularised_loss, augment_text, augment_numeric, inject_weight_noise, AdaGrad, PolyakAveraging, clip_gradients_by_norm, clip_gradients_by_value, lstm_cell_forward, init_lstm_weights, corrupt_masking, corrupt_gaussian, dae_reconstruction_loss, TransferLearningRegistry, greedy_layerwise_pretrain, metropolis_hastings, gibbs_sampling, elbo, mean_field_update, gan_discriminator_loss, gan_generator_loss, gan_minimax_value)
    print("[dl_book_implementations2] ✅ 45 functions loaded")
except Exception as _e:
    print(f"[dl_book_implementations2] ❌ {_e}")

try:
    from dl_book_implementations3 import (mat_mul, mat_T, vec_norm, power_iteration, svd_1d, pca_project, entropy, kl_divergence, cross_entropy, binary_cross_entropy, mutual_information, softplus, log_sum_exp, gaussian_pdf, gaussian_log_pdf, bayes_update, map_estimate, stable_sigmoid, stable_softmax, stable_log_softmax, jacobian, hessian, linear_least_squares, kmeans, multitask_loss, uncertainty_weighted_multitask, tied_weight_grad, fgsm_perturbation, NesterovMomentum, newton_step_1d, newton_method, coordinate_descent, conv1d, conv2d_single, max_pool1d, avg_pool1d, rnn_cell_forward, bptt_gradient_check, birnn_combine, encode_sequence, attention_context, precision_recall_f1, coverage_at_k, roc_auc_approx, grid_search, whiten, soft_threshold, ista_sparse_code, RBM, nce_loss, gmm_e_step, gmm_m_step, gmm_fit, vae_reparameterise, vae_kl_loss, vae_elbo, NgramLM, parallel_tempering_swap)
    print("[dl_book_implementations3] ✅ 58 functions loaded")
except Exception as _e:
    print(f"[dl_book_implementations3] ❌ {_e}")

try:
    from goodfellow_dl import (cosine_similarity, pca, l2_norm, frobenius_norm, softmax, entropy, kl_divergence, sample_top_k, gaussian_nll, numerical_gradient, log_sum_exp, clip_gradient, train_test_split, k_fold_indices, bias_variance_mse, mle_gaussian, map_estimate_gaussian, tfidf_vectors, cosine_sim_tfidf, rank_by_tfidf, relu, relu_grad, sigmoid, sigmoid_grad, tanh_act, tanh_grad, leaky_relu, elu, gelu, swish, xavier_init, he_init, cross_entropy_loss, mse_loss, FeedForward, l2_regularization, l1_regularization, dropout, max_norm_constraint, label_smoothing, early_stopping_check, data_augmentation_noise, SGD, Adam, RMSProp, cosine_annealing_lr, warmup_lr, batch_norm, layer_norm, monitor_gradient_norm, conv1d, max_pool1d, conv_output_size, receptive_field, VanillaRNN, LSTMCell, bptt_clip, learning_curve, hyperparameter_grid, detect_vanishing_gradients, confusion_matrix, f1_score, ppca, ica_whitening, Autoencoder, denoising_autoencoder_corrupt, sparse_penalty, build_word_cooccurrence, ppmi, svd_embeddings, contrastive_loss, nearest_neighbors, scaled_dot_product_attention, sinusoidal_positional_encoding, rotary_positional_encoding, multi_head_attention_scores, _tokenize, build_tfidf_index, semantic_rank, calibrate_confidence, verify_math_response, response_diversity_score, gradient_informed_prompt_score)
    print("[goodfellow_dl] ✅ 83 functions loaded")
except Exception as _e:
    print(f"[goodfellow_dl] ❌ {_e}")

try:
    from gaps_all_books import (elu, selu, leaky_relu, maxout, hard_sigmoid, hard_swish, are_weights_tied, l1_penalty, l2_penalty, elastic_net_penalty, dropout_forward, add_gaussian_noise, EarlyStopping, clip_grad_norm, clip_grad_value, nesterov_step, adagrad_step, polyak_average, conv1d, max_pool1d, avg_pool1d, rnn_step, lstm_step, gru_step, vae_reparameterize, vae_kl_loss, word2vec_neg_sampling_loss, elbo, sliding_window_dataset, CharTokenizer, causal_mask, split_heads, merge_heads, masked_attention, softmax_v, ffn_block, pre_norm_residual, post_norm_residual, layer_norm_v, greedy_decode, top_k_sample, top_p_sample, temperature_sample, apply_repetition_penalty, kaiming_uniform_init, glorot_uniform_init, lora_forward, alpaca_prompt, phi3_prompt, chatml_prompt, ppo_clip_loss, reward_model_score, dpo_loss, token_embedding_lookup, absolute_positional_embedding, input_embedding, semantic_similarity_score, biencoder_score, crossencoder_score, rag_score, brute_force_knn, classification_head, bleu_score, rouge_1, rouge_2, bertscore_approx, speculative_decode_accept, kv_cache_bytes, batch_occupancy, beam_search, factual_consistency_score, repetition_score, answer_relevance_score, llm_judge_rubric, context_utilization, fixed_size_chunk, sentence_chunk, retrieval_precision_at_k, retrieval_recall_at_k, mean_average_precision, tool_selection_score, react_format, quality_score, dedup_exact, dedup_near, preference_pair, quantization_error, int8_quantize, int8_dequantize, attention_flops_standard, attention_flops_flash, g_eval_score, mt_bench_aggregate, stratified_sample, reservoir_sample, importance_sampling_weight, min_max_normalize, z_score_normalize, log_transform, bucketize, mean_impute, median_impute, covariate_shift_score, label_shift_ratio, expected_calibration_error, brier_score, confusion_matrix_stats, roc_auc_approx, forgetting_score, plasticity_score, latency_percentiles, throughput, shadow_delta, crawl_quality, dataset_card, SimpleRAGPipeline, sft_format_openai, sft_format_sharegpt, estimate_tokens_heuristic, fits_in_context, qlora_memory_mb, training_throughput, slerp, linear_merge, task_arithmetic_merge, model_card, serving_cost_per_1k, llm_trace, vec_projection, angle_between, gram_schmidt, matrix_rank, is_full_rank, jvp, vjp, gmm_e_step, bernoulli_sufficient_stat, gaussian_sufficient_stats, lagrangian, kkt_check)
    print("[gaps_all_books] ✅ 140 functions loaded")
except Exception as _e:
    print(f"[gaps_all_books] ❌ {_e}")

try:
    from math_impl import (mat_add, mat_sub, mat_mul, mat_transpose, mat_scalar, vec_dot, vec_norm, vec_normalize, vec_outer, mat_identity, mat_trace, mat_frobenius_norm, gaussian_elimination, lu_decompose, power_iteration, svd_dominant, pca, cosine_similarity, numerical_gradient, gradient_descent, sgd_with_momentum, adam_step, rmsprop_step, newtons_method, DualNumber, auto_diff, SimpleNet, gaussian_pdf, gaussian_cdf, kl_divergence, js_divergence, entropy, cross_entropy, mutual_information, bayes_update, monte_carlo_estimate, bootstrap_confidence_interval, t_test_two_sample, perplexity_from_probs, bits_back_coding_gain, information_gain, pointwise_mutual_info, numerical_jacobian, conjugate_gradient, line_search_backtrack, softmax, log_softmax, gelu, swiglu, rope_encoding, attention_entropy, label_smoothing_loss, focal_loss, cosine_lr_schedule, weight_decay_update, perplexity_from_loss, bits_per_byte, flesch_kincaid_grade, token_fertility)
    print("[math_impl] ✅ 59 functions loaded")
except Exception as _e:
    print(f"[math_impl] ❌ {_e}")

try:
    from orchestrator_patches_prod import (get_db_connection, ensemble_rlaif_score, save_sft_example_with_curriculum, get_optimized_hyperparameters, denoise_query_input, dynamic_mcts_branch_factor, get_calibrated_confidence)
    print("[orchestrator_patches_prod] ✅ 7 functions loaded")
except Exception as _e:
    print(f"[orchestrator_patches_prod] ❌ {_e}")


import modules.opus_engine as opus_engine
import sys
try:
    import uvloop
    uvloop.install()
    print("[TTFT] uvloop event loop installed — async 2-4x faster")
except ImportError:
    pass

FEEDBACK_FILE = "modules/feedback_store.json"

