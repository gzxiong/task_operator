"""Public API for src_final_v2."""

from .data import (
    ALGORITHMIC_TASKS,
    ALL_TASKS,
    LEXICAL_TASKS,
    REASONING_TASKS,
    assert_splits_exist,
    load_split,
    max_new_tokens_for_category,
    split_path,
    stop_strings_for_category,
    task_category,
)
from .evaluation import (
    evaluate_generation,
    is_equiv_gpqa,
    is_equiv_gsm8k,
    is_equiv_math500,
    parse_answer,
)
from .io import (
    OUTPUTS_DIR,
    append_row,
    atomic_write_json,
    baselines_paths,
    diff_settings,
    done_test_indices,
    fingerprint,
    is_shard_complete,
    load_or_init_shard,
    nto_paths,
    predictions_path,
    safe_model_name,
    search_paths,
    task_operator_paths,
)
from .model import (
    MODEL_REGISTRY,
    detect_model_family,
    get_decoder_layers,
    get_modeling_backend,
    get_text_config,
    load_hf_model,
)
from .prompting import (
    COT_DEMO_TEMPLATE,
    COT_QUERY_TEMPLATE,
    DEMO_TEMPLATE,
    QUERY_TEMPLATE,
    build_full_icl_prompt,
    build_zsl_prompt,
)
from .spans import (
    find_token_span_by_substring,
    identify_context_query,
    resolve_query_positions,
    token_char_spans,
)
from .tokenization import encode_prompt_ids, leading_sink_offset, resolve_sink_token_id
from .attention import patched_attention
from .generation import generate_with_hooks
from .task_operator import (
    TaskOperatorConfig,
    apply_full_icl,
    apply_task_operator,
    apply_zero_shot,
    extract_knowledge,
)
from .nto import (
    NTOConfig,
    apply_canonical_task_operator,
    extract_canonical_batch_state,
    extract_canonical_knowledge,
    extract_canonical_knowledge_iterative,
    extract_canonical_knowledge_iterative_hybrid,
    extract_hybrid_batch_state,
    nto_config_from_default,
    reparameterize_nto_wb,
)
from .gap_analysis import (
    extract_gap_stats,
    filter_mean_circuit_by_gap_topp,
    flatten_gap_sites,
)
from .nll_site_analysis import (
    compute_recovery_scores,
    compute_recovery_scores_icl,
    compute_rrf_scores,
    extract_nll_icl_per_sample,
    extract_nll_site_stats,
    filter_mean_circuit_by_score_topp,
    flatten_score_sites,
)
from .stability import (
    compare_circuits,
    compute_cross_sample_stats,
    compute_cross_token_within_prompt_cv,
)
from .search import SearchSettings, run_search
from .baselines import (
    ConceptorConfig,
    FVConfig,
    I2CLConfig,
    ICVConfig,
    TVConfig,
    apply_baseline,
    extract_baseline,
)
from .experiments import (
    DEFAULT_REP_PENALTY,
    N_DEMOS,
    N_VALIDATION,
    config_from_searched,
    iter_model_task_pairs,
    load_searched_config,
    load_validation_icl_outputs,
    make_default_config,
    make_nto_config,
    run_baseline_setting,
    run_nto_setting,
    run_predictions_shard,
    run_task_operator_setting,
)
