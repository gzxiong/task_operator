"""task_operator: Capturing In-Context Learning Dynamics with Task Operators.

The Task Operator (TO) extracts an analytic per-(layer, head, position) affine
operator `(w, b)` from an ICL forward pass, then replays it as a dynamic update
to the attention output projection during zero-shot inference. See `core.py`
for extraction + replay, `baselines.py` for TV/FV/ICV/Conceptors, and
`nll_site_analysis.py` for the NLL-recovery sparsity analysis.
"""

# Core method
from .core import (
    TaskOperatorConfig,
    apply_full_icl,
    apply_task_operator,
    apply_zero_shot,
    extract_knowledge,
)

# Data + tasks
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

# Model + prompting + generation
from .model import (
    MODEL_REGISTRY,
    detect_model_family,
    get_decoder_layers,
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
from .generation import generate_with_hooks

# Tokenization + spans
from .tokenization import encode_prompt_ids, leading_sink_offset
from .spans import (
    find_token_span_by_substring,
    identify_context_query,
    resolve_query_positions,
    token_char_spans,
)

# Evaluation
from .evaluation import (
    evaluate_generation,
    is_equiv_gpqa,
    is_equiv_gsm8k,
    is_equiv_math500,
    parse_answer,
)

# Baselines (TV / FV / ICV / Conceptors)
from .baselines import (
    ConceptorConfig,
    FVConfig,
    ICVConfig,
    TVConfig,
    apply_baseline,
    extract_baseline,
)

# Sparsity analysis (Section 3.3)
from .nll_site_analysis import (
    compute_recovery_scores,
    compute_recovery_scores_icl,
    compute_rrf_scores,
    extract_nll_icl_per_sample,
    extract_nll_site_stats,
    filter_mean_circuit_by_score_topp,
    flatten_score_sites,
)

# IO + experiments
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
    predictions_path,
    safe_model_name,
    task_operator_paths,
)
from .experiments import (
    DEFAULT_REP_PENALTY,
    N_DEMOS,
    N_VALIDATION,
    iter_model_task_pairs,
    load_validation_icl_outputs,
    make_allpos_config,
    make_default_config,
    run_baseline_setting,
    run_predictions_shard,
    run_task_operator_setting,
)
