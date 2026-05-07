"""nto: Normalized Task Operator (canonical (theta, mu_C) reparameterization).

Mirrors `src_final_v2/task_operator.py` but stores per-head canonical
coordinates per (layer, slot, head):

    theta = log((1-w)/w) - log K_eff   = log D_ctx - log D_q - log K_eff
    mu_C  = N_ctx / D_ctx              (context-only renormalization;
                                       equivalent to ch / (1-w) but
                                       computed directly to avoid the
                                       1/(1-w) blow-up at low context mass)

where (w, ch) are the parent paper's affine factors:

    w  = D_q   / (D_ctx + D_q)
    ch = N_ctx / (D_ctx + D_q)

with D_ctx = sum_C exp(s), D_q = sum_{!=C} exp(s), N_ctx = sum_C exp(s)*v.

At replay, given a reference demo count K_0:

    p_tilde = sigmoid(theta + log K_0)
    w_tilde = 1 - p_tilde
    b_tilde = p_tilde * mu_C

These (w_tilde, b_tilde) are dropped into the same forward hook used by
the standard TO (`src_final_v2/generation.py:_make_prompt_replay_hook`),
substituting for (w, ch).

Identity at K_0 = K_extraction: theta + log K_0 = log((1-w)/w), so
p_tilde = 1-w and (w_tilde, b_tilde) = (w, ch) exactly. The canonical
operator at the boundary reproduces the standard TO.

This module is independent of `task_operator.py` so the two methods can
be A/B compared without code-coupling. The hook in `generation.py` is
unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .attention import patched_attention
from .generation import generate_with_hooks
from .model import get_decoder_layers, get_text_config
from .prompting import DEMO_TEMPLATE, QUERY_TEMPLATE, build_full_icl_prompt, build_zsl_prompt
from .spans import identify_context_query, resolve_query_positions
from .tokenization import encode_prompt_ids


# ----------------------------------------------------------------------------- config
@dataclass
class NTOConfig:
    """Mirrors TaskOperatorConfig with one canonical-method addition: K_0.

    K_0 is the reference demonstration count used at replay; it controls
    intervention strength via p_tilde = sigmoid(theta + log K_0). At
    K_0 = K_extraction the canonical operator reproduces the standard TO.
    """

    attention_sink: bool = True
    demo_template: str = DEMO_TEMPLATE
    query_template: str = QUERY_TEMPLATE

    n_template: int = -1
    n_content: int = -1
    n_output: int = -1
    template_active_ranks: list[int] | None = None
    active_layers_per_slot: dict[str, list[int]] | None = None

    repetition_penalty: float = 1.1
    max_new_tokens: int = 64
    stopping_strings: list[str] | None = None
    answer_phrases: tuple[str, ...] = ()

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    K_0: float = 8.0


# -------------------------------------------------------------------- position helpers
def _select_template(template_positions: list[int], config: NTOConfig) -> tuple[list[int], list[int]]:
    n = len(template_positions)
    if config.template_active_ranks is not None:
        ranks = [int(r) for r in config.template_active_ranks if 0 <= int(r) < n]
    elif config.n_template == -1:
        ranks = list(range(n))
    elif config.n_template == 0:
        ranks = []
    else:
        ranks = list(range(min(int(config.n_template), n)))
    return [template_positions[r] for r in ranks], ranks


def _select_first_k(positions: list[int], k: int) -> list[int]:
    if k == -1:
        return list(positions)
    if k == 0:
        return []
    return list(positions)[: int(k)]


# ------------------------------------------------------------- canonical factor extraction
def _native_dtype(value, dtype):
    return value if value.dtype == dtype else value.to(dtype=dtype)


def _extract_canonical_factors(batch_scores, value_states, context_span, K_eff: int):
    """Per-prompt canonical coordinates (theta, mu_C) plus raw (w, ch) for diagnostics.

    batch_scores : Tensor[n_pos, n_heads, n_keys]   (pre-softmax attention scores)
    value_states : Tensor[n_heads, n_keys, head_dim]
    context_span : (start, end)
    K_eff        : number of demonstration blocks in the extraction prompt

    Returns (theta, mu_C, w, context_heads):
        theta         : Tensor[n_pos, n_heads]
        mu_C          : Tensor[n_pos, n_heads, head_dim]
        w             : Tensor[n_pos, n_heads]              (paper's w)
        context_heads : Tensor[n_pos, n_heads, head_dim]    (paper's b)
    """

    cs, ce = int(context_span[0]), int(context_span[1])
    context_scores = batch_scores[:, :, cs:ce]                          # [P, H, C]
    context_values = value_states[:, cs:ce, :]                          # [H, C, D]

    log_ctx = torch.logsumexp(context_scores, dim=-1)                   # [P, H]
    log_full = torch.logsumexp(batch_scores, dim=-1)                    # [P, H]

    # Stable log D_q via log-sub-exp:
    #   log D_q = log(exp(log_full) - exp(log_ctx))
    #          = log_full + log1p(-exp(log_ctx - log_full))
    diff = (log_ctx - log_full).clamp_(max=-1e-7)                       # in (-inf, -1e-7]
    log_q = log_full + torch.log1p(-torch.exp(diff))                    # [P, H]

    theta = log_ctx - log_q - math.log(float(K_eff))                    # [P, H]

    # Context-conditioned value prototype via direct context-only renormalization
    # (matches GPT writeup eq 28; avoids the b/(1-w) division entirely).
    ctx_probs_C = torch.exp(context_scores - log_ctx.unsqueeze(-1))     # [P, H, C]
    mu_C = torch.einsum("phc,hcd->phd", ctx_probs_C, context_values)    # [P, H, D]

    # Raw (w, ch) for diagnostic comparison only — not used at replay.
    w = (1 - torch.exp(log_ctx - log_full)).clamp_(0, 1)                # [P, H]
    full_probs_C = torch.exp(context_scores - log_full.unsqueeze(-1))   # full softmax restricted to C
    context_heads = torch.einsum("phc,hcd->phd", full_probs_C, context_values)  # [P, H, D]

    return theta, mu_C, w, context_heads


def _per_layer_value_states(layer, hidden_state, num_kv_heads: int, head_dim: int, model_dtype) -> torch.Tensor:
    norm_dtype = getattr(layer.input_layernorm.weight, "dtype", model_dtype)
    attn_input = layer.input_layernorm(_native_dtype(hidden_state, norm_dtype))
    v_linear = layer.self_attn.v_proj(attn_input)
    v_states = v_linear.view(v_linear.shape[0], num_kv_heads, head_dim).permute(1, 0, 2)
    return v_states.repeat_interleave(int(layer.self_attn.num_key_value_groups), dim=0)


# --------------------------------------------------------- per-sample circuit extraction
def _extract_one_sample_canonical(
    model, tokenizer, demos, record, icl_output_text, config: NTOConfig,
    *,
    keep_on_device: bool = False,
) -> dict | None:
    """Run the extended ICL forward for one (record, icl_output) and return per-layer
    canonical factors (theta, mu_C, w_raw, ch_raw) for the selected positions.

    K_eff for this sample = len(demos).
    """

    K_eff = int(len(demos))
    if K_eff <= 0:
        return None

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    n_heads = int(text_cfg.num_attention_heads)
    n_kv_heads = int(getattr(text_cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // n_heads))

    icl_prompt = build_full_icl_prompt(
        demos, record["input"],
        demo_template=config.demo_template, query_template=config.query_template,
    )
    zsl_prompt = build_zsl_prompt(record["input"], query_template=config.query_template)

    icl_ids = encode_prompt_ids(
        tokenizer, icl_prompt, attention_sink=config.attention_sink, device=device,
    )
    target_token_ids = tokenizer(icl_output_text or "", add_special_tokens=False).input_ids
    if not target_token_ids:
        return None
    target_ids = torch.tensor([target_token_ids], device=device)
    extended_ids = torch.cat([icl_ids, target_ids], dim=1)

    context_span, query_span = identify_context_query(
        tokenizer, icl_ids, zsl_prompt, prompt_text=icl_prompt,
    )
    resolved = resolve_query_positions(
        tokenizer, icl_ids, query_span, record["input"], prompt_text=icl_prompt,
    )
    template_positions = list(resolved["template_positions"])
    content_positions = list(resolved["content_positions"])
    icl_prompt_len = int(icl_ids.shape[1])
    n_target = int(target_ids.shape[1])
    gen_positions = list(range(icl_prompt_len, icl_prompt_len + n_target))

    active_template, active_template_ranks = _select_template(template_positions, config)
    active_content = _select_first_k(content_positions, config.n_content)
    active_output = _select_first_k(gen_positions, config.n_output)
    active_all = active_template + active_content + active_output
    analysis_all = template_positions + content_positions + gen_positions

    mask_spec = []
    if active_all:
        active_set = set(active_all)
        inactive = [p for p in analysis_all if p not in active_set]
        if inactive and context_span[1] > context_span[0]:
            mask_spec.append({
                "layers": list(range(n_layers)),
                "query_rows": inactive,
                "key_span": [int(context_span[0]), int(context_span[1])],
                "alpha": 0.0,
            })

    attn_trace: dict = {}
    if keep_on_device:
        attn_trace["_keep_on_device"] = True
    with torch.inference_mode():
        with patched_attention(model, mask_spec=mask_spec, capture_store=attn_trace):
            outputs = model(
                input_ids=extended_ids, use_cache=False,
                output_hidden_states=True, return_dict=True,
            )

    decoder_layers = get_decoder_layers(model)
    layer_records = []
    for layer_idx, layer in enumerate(decoder_layers):
        layer_hidden = outputs.hidden_states[layer_idx][0]
        v_states = _per_layer_value_states(layer, layer_hidden, n_kv_heads, head_dim, model_dtype)
        if keep_on_device:
            scores = attn_trace[layer_idx]["scores_post"]
        else:
            scores = attn_trace[layer_idx]["scores_post"].to(device=device, dtype=model_dtype)

        def _factors_at(positions: list[int]):
            if not positions:
                return None
            s = scores[:, positions, :].permute(1, 0, 2).contiguous()
            out = _extract_canonical_factors(s, v_states, context_span, K_eff)
            if keep_on_device:
                return tuple(t.detach() for t in out)
            return tuple(t.detach().cpu() for t in out)

        layer_records.append({
            "template": _factors_at(active_template),
            "content": _factors_at(active_content),
            "gen": _factors_at(active_output),
        })

    return {
        "layers": layer_records,
        "n_template": len(active_template),
        "n_content": len(active_content),
        "n_gen": len(active_output),
        "active_template_ranks": active_template_ranks,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "K_eff": K_eff,
    }


# ------------------------------------------------------------ aggregation across samples
def _stack_mean(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(tensors), dim=0).mean(dim=0)


def _aggregate_canonical_circuits(raw: list[dict], config: NTOConfig) -> dict:
    """Average per-(layer, slot) canonical factors across validation samples.

    Returns a mean_circuit_nto with the same record schema as the standard
    mean_circuit, plus theta and mu_C fields (the canonical coordinates
    that drive replay):

        record = {
            "position_type", "rank", "broadcast",
            "theta":         Tensor[n_heads],
            "mu_C":          Tensor[n_heads, head_dim],
            "w":             Tensor[n_heads],            # diagnostic
            "context_heads": Tensor[n_heads, head_dim],  # diagnostic
        }

    Tuple layout from _extract_one_sample_canonical: (theta, mu_C, w, ch).
    Aggregation order matches task_operator._aggregate_circuits exactly so
    record indexing is portable.
    """

    if not raw:
        raise ValueError("no validation samples were processed")
    n_layers = raw[0]["n_layers"]
    n_heads = raw[0]["n_heads"]
    head_dim = raw[0]["head_dim"]

    K_eff_set = sorted({int(c["K_eff"]) for c in raw})

    common_template_ranks = sorted(set.intersection(*[set(c["active_template_ranks"]) for c in raw]))
    min_n_content = min(c["n_content"] for c in raw)
    min_n_gen = min(c["n_gen"] for c in raw)

    records_by_layer: dict[int, list[dict]] = {l: [] for l in range(n_layers)}

    def _make_record(position_type, rank, broadcast,
                     theta_stack, mu_stack, w_stack, ch_stack):
        return {
            "position_type": position_type,
            "rank": rank,
            "broadcast": broadcast,
            "theta": _stack_mean(theta_stack),
            "mu_C":  _stack_mean(mu_stack),
            "w":             _stack_mean(w_stack),
            "context_heads": _stack_mean(ch_stack),
        }

    # ---- template (per-rank) ----
    if common_template_ranks:
        for layer_idx in range(n_layers):
            for rank in common_template_ranks:
                theta_stack, mu_stack, w_stack, ch_stack = [], [], [], []
                for c in raw:
                    rank_idx = c["active_template_ranks"].index(rank)
                    t = c["layers"][layer_idx]["template"]
                    # tuple is (theta, mu_C, w, ch) per _extract_canonical_factors
                    theta_stack.append(t[0][rank_idx])
                    mu_stack.append(t[1][rank_idx])
                    w_stack.append(t[2][rank_idx])
                    ch_stack.append(t[3][rank_idx])
                records_by_layer[layer_idx].append(_make_record(
                    "template", int(rank), False,
                    theta_stack, mu_stack, w_stack, ch_stack,
                ))

    # ---- content ----
    if config.n_content == 0:
        n_content_recs = 0
    elif config.n_content == -1:
        for layer_idx in range(n_layers):
            theta_stack, mu_stack, w_stack, ch_stack = [], [], [], []
            for c in raw:
                t = c["layers"][layer_idx]["content"]
                if t is None:
                    continue
                theta_stack.append(t[0].mean(dim=0))
                mu_stack.append(t[1].mean(dim=0))
                w_stack.append(t[2].mean(dim=0))
                ch_stack.append(t[3].mean(dim=0))
            if theta_stack:
                records_by_layer[layer_idx].append(_make_record(
                    "content", None, True,
                    theta_stack, mu_stack, w_stack, ch_stack,
                ))
        n_content_recs = 1
    else:
        k_max = int(config.n_content)
        produced_ranks: list[int] = []
        for rank in range(k_max):
            if any(int(c["n_content"]) > rank for c in raw):
                produced_ranks.append(rank)
        for layer_idx in range(n_layers):
            for rank in produced_ranks:
                theta_stack, mu_stack, w_stack, ch_stack = [], [], [], []
                for c in raw:
                    if rank >= int(c["n_content"]):
                        continue
                    t = c["layers"][layer_idx]["content"]
                    if t is None:
                        continue
                    theta_stack.append(t[0][rank])
                    mu_stack.append(t[1][rank])
                    w_stack.append(t[2][rank])
                    ch_stack.append(t[3][rank])
                if not theta_stack:
                    continue
                records_by_layer[layer_idx].append(_make_record(
                    "content", int(rank), False,
                    theta_stack, mu_stack, w_stack, ch_stack,
                ))
        n_content_recs = len(produced_ranks)

    # ---- output ----
    if config.n_output == 0:
        n_output_recs = 0
    elif config.n_output == -1:
        for layer_idx in range(n_layers):
            theta_stack, mu_stack, w_stack, ch_stack = [], [], [], []
            for c in raw:
                t = c["layers"][layer_idx]["gen"]
                if t is None:
                    continue
                theta_stack.append(t[0].mean(dim=0))
                mu_stack.append(t[1].mean(dim=0))
                w_stack.append(t[2].mean(dim=0))
                ch_stack.append(t[3].mean(dim=0))
            if theta_stack:
                records_by_layer[layer_idx].append(_make_record(
                    "output", None, True,
                    theta_stack, mu_stack, w_stack, ch_stack,
                ))
        n_output_recs = 1
    else:
        k_max = int(config.n_output)
        produced_ranks_out: list[int] = []
        for rank in range(k_max):
            if any(int(c["n_gen"]) > rank for c in raw):
                produced_ranks_out.append(rank)
        for layer_idx in range(n_layers):
            for rank in produced_ranks_out:
                theta_stack, mu_stack, w_stack, ch_stack = [], [], [], []
                for c in raw:
                    if rank >= int(c["n_gen"]):
                        continue
                    t = c["layers"][layer_idx]["gen"]
                    if t is None:
                        continue
                    theta_stack.append(t[0][rank])
                    mu_stack.append(t[1][rank])
                    w_stack.append(t[2][rank])
                    ch_stack.append(t[3][rank])
                if not theta_stack:
                    continue
                records_by_layer[layer_idx].append(_make_record(
                    "output", int(rank), False,
                    theta_stack, mu_stack, w_stack, ch_stack,
                ))
        n_output_recs = len(produced_ranks_out)

    return {
        "records_by_layer": records_by_layer,
        "meta": {
            "method": "nto",
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "n_template_ranks": list(common_template_ranks),
            "n_content_recs": int(n_content_recs),
            "n_output_recs": int(n_output_recs),
            "n_val_samples": int(len(raw)),
            "K_extraction": int(K_eff_set[0]) if len(K_eff_set) == 1 else K_eff_set,
            "config": {
                "n_template": int(config.n_template),
                "n_content": int(config.n_content),
                "n_output": int(config.n_output),
                "template_active_ranks": list(config.template_active_ranks)
                if config.template_active_ranks is not None else None,
            },
        },
    }


def extract_canonical_knowledge(
    model,
    tokenizer,
    demos,
    validation_records,
    validation_icl_outputs,
    config: NTOConfig,
) -> dict:
    """Extract a canonical mean_circuit (theta, mu_C) averaged across validation samples.

    Tracks failed samples in meta["skipped"] like the standard extractor.
    """

    raw, skipped = [], []
    for record, icl_output in zip(validation_records, validation_icl_outputs):
        try:
            sample = _extract_one_sample_canonical(
                model, tokenizer, demos, record, icl_output, config,
            )
            reason = None if sample is not None else "empty_target_or_span_mismatch"
        except Exception as exc:
            sample = None
            reason = f"{type(exc).__name__}: {exc}"
        if sample is None:
            skipped.append({"example_id": record.get("example_id"), "reason": reason})
        else:
            raw.append(sample)
    if not raw:
        raise RuntimeError(
            f"all {len(skipped)} validation samples failed; first 3: {skipped[:3]}"
        )
    mean_circuit = _aggregate_canonical_circuits(raw, config)
    mean_circuit["meta"]["n_skipped"] = len(skipped)
    mean_circuit["meta"]["skipped"] = skipped

    if config.active_layers_per_slot is not None:
        available_slot_ids = {
            _slot_id_for_record(rec)
            for recs in mean_circuit["records_by_layer"].values()
            for rec in recs
        }
        missing = [sid for sid in config.active_layers_per_slot if sid not in available_slot_ids]
        if missing:
            template_missing = [sid for sid in missing if sid.startswith("template:")]
            other_missing = [sid for sid in missing if not sid.startswith("template:")]
            if template_missing:
                raise ValueError(
                    f"template slot(s) in active_layers_per_slot not present in "
                    f"mean_circuit: {template_missing}."
                )
            if other_missing:
                print(
                    f"  [warn] active_layers_per_slot has {len(other_missing)} content/"
                    f"output slot(s) with no matching mean_circuit records: {other_missing}."
                )
                mean_circuit["meta"]["skipped_active_layers_per_slot_keys"] = list(other_missing)
    return mean_circuit


# ============================================================================
#   ITERATIVE CANONICAL BATCH MERGE
# ============================================================================
#
# Per-batch sufficient statistics (log_D_ctx, mu_C, log_D_q, K_batch) are
# extracted from disjoint short ICL prompts and recombined online into a
# single canonical operator. Replaces the (rho, eta) batch merge from
# notebook 12, whose w_merged collapsed to 0 at large K_eff because
# rho_total accumulates linearly with N. Here we hold theta in a stable
# regime via the K_total normalization; mu_C is a context-mass-weighted
# running mean.
#
# Math (TODO_0429_1500 §3): for each (layer, head, position) site,
#     log_D_ctx_new = logaddexp(log_D_ctx, log_D_ctx_j)
#     mu_C_new      = α_old * mu_C + α_new * mu_C_j  with αs from softmax
#     log_D_q       = running mean
#     K_total      += K_j
# At finalize:
#     theta_merged = log_D_ctx_total - log_D_q_avg - log K_total
#     mu_C_merged  = mu_C_running_mean
#
# The replay path in `_reconstruct_factors_canonical` reads only
# (theta, mu_C) so the merged operator plugs into the existing hook with
# no changes.


def _extract_batch_state_factors(batch_scores, value_states, context_span):
    """Per-batch sufficient statistics (no K-normalization).

    batch_scores : Tensor[n_pos, n_heads, n_keys]   (pre-softmax attention scores)
    value_states : Tensor[n_heads, n_keys, head_dim]
    context_span : (start, end)

    Returns (log_ctx, mu_C, log_q):
        log_ctx : Tensor[n_pos, n_heads]
        mu_C    : Tensor[n_pos, n_heads, head_dim]    context-only renormalized
        log_q   : Tensor[n_pos, n_heads]              stable log-sub-exp
    """

    cs, ce = int(context_span[0]), int(context_span[1])
    context_scores = batch_scores[:, :, cs:ce]
    context_values = value_states[:, cs:ce, :]

    log_ctx = torch.logsumexp(context_scores, dim=-1)
    log_full = torch.logsumexp(batch_scores, dim=-1)

    diff = (log_ctx - log_full).clamp_(max=-1e-7)
    log_q = log_full + torch.log1p(-torch.exp(diff))

    ctx_probs_C = torch.exp(context_scores - log_ctx.unsqueeze(-1))
    mu_C = torch.einsum("phc,hcd->phd", ctx_probs_C, context_values)

    return log_ctx, mu_C, log_q


def _extract_one_sample_batch_state(
    model, tokenizer, demos, record, icl_output_text, config: NTOConfig,
    *,
    keep_on_device: bool = False,
) -> dict | None:
    """Run the extended ICL forward for one (record, icl_output) and return per-layer
    batch-state factors (log_D_ctx, mu_C, log_D_q) for the selected positions.

    K_batch = len(demos) is recorded in the returned dict but not folded into the
    extracted tensors — the merge step finalizes K_total = sum_j K_batch_j and
    normalizes once at the end.
    """

    K_batch = int(len(demos))
    if K_batch <= 0:
        return None

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    n_heads = int(text_cfg.num_attention_heads)
    n_kv_heads = int(getattr(text_cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // n_heads))

    icl_prompt = build_full_icl_prompt(
        demos, record["input"],
        demo_template=config.demo_template, query_template=config.query_template,
    )
    zsl_prompt = build_zsl_prompt(record["input"], query_template=config.query_template)

    icl_ids = encode_prompt_ids(
        tokenizer, icl_prompt, attention_sink=config.attention_sink, device=device,
    )
    target_token_ids = tokenizer(icl_output_text or "", add_special_tokens=False).input_ids
    if not target_token_ids:
        return None
    target_ids = torch.tensor([target_token_ids], device=device)
    extended_ids = torch.cat([icl_ids, target_ids], dim=1)

    context_span, query_span = identify_context_query(
        tokenizer, icl_ids, zsl_prompt, prompt_text=icl_prompt,
    )
    resolved = resolve_query_positions(
        tokenizer, icl_ids, query_span, record["input"], prompt_text=icl_prompt,
    )
    template_positions = list(resolved["template_positions"])
    content_positions = list(resolved["content_positions"])
    icl_prompt_len = int(icl_ids.shape[1])
    n_target = int(target_ids.shape[1])
    gen_positions = list(range(icl_prompt_len, icl_prompt_len + n_target))

    active_template, active_template_ranks = _select_template(template_positions, config)
    active_content = _select_first_k(content_positions, config.n_content)
    active_output = _select_first_k(gen_positions, config.n_output)
    active_all = active_template + active_content + active_output
    analysis_all = template_positions + content_positions + gen_positions

    mask_spec = []
    if active_all:
        active_set = set(active_all)
        inactive = [p for p in analysis_all if p not in active_set]
        if inactive and context_span[1] > context_span[0]:
            mask_spec.append({
                "layers": list(range(n_layers)),
                "query_rows": inactive,
                "key_span": [int(context_span[0]), int(context_span[1])],
                "alpha": 0.0,
            })

    attn_trace: dict = {}
    if keep_on_device:
        attn_trace["_keep_on_device"] = True
    with torch.inference_mode():
        with patched_attention(model, mask_spec=mask_spec, capture_store=attn_trace):
            outputs = model(
                input_ids=extended_ids, use_cache=False,
                output_hidden_states=True, return_dict=True,
            )

    decoder_layers = get_decoder_layers(model)
    layer_records = []
    for layer_idx, layer in enumerate(decoder_layers):
        layer_hidden = outputs.hidden_states[layer_idx][0]
        v_states = _per_layer_value_states(layer, layer_hidden, n_kv_heads, head_dim, model_dtype)
        if keep_on_device:
            scores = attn_trace[layer_idx]["scores_post"]
        else:
            scores = attn_trace[layer_idx]["scores_post"].to(device=device, dtype=model_dtype)

        def _factors_at(positions: list[int]):
            if not positions:
                return None
            s = scores[:, positions, :].permute(1, 0, 2).contiguous()
            out = _extract_batch_state_factors(s, v_states, context_span)
            if keep_on_device:
                return tuple(t.detach() for t in out)
            return tuple(t.detach().cpu() for t in out)

        layer_records.append({
            "template": _factors_at(active_template),
            "content": _factors_at(active_content),
            "gen": _factors_at(active_output),
        })

    return {
        "layers": layer_records,
        "n_template": len(active_template),
        "n_content": len(active_content),
        "n_gen": len(active_output),
        "active_template_ranks": active_template_ranks,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "K_batch": K_batch,
    }


def _aggregate_batch_state_canonical(raw: list[dict], config: NTOConfig) -> dict:
    """Average per-(layer, slot) batch-state factors across m validation samples.

    Each record carries (log_D_ctx, mu_C, log_D_q) — the merge step recombines
    these across batches. Mirrors `_aggregate_canonical_circuits` shape but
    holds the raw partial-softmax coordinates rather than the K-normalized form.
    """

    if not raw:
        raise ValueError("no validation samples were processed")
    n_layers = raw[0]["n_layers"]
    n_heads = raw[0]["n_heads"]
    head_dim = raw[0]["head_dim"]
    K_batch_set = sorted({int(c["K_batch"]) for c in raw})

    common_template_ranks = sorted(set.intersection(*[set(c["active_template_ranks"]) for c in raw]))

    records_by_layer: dict[int, list[dict]] = {l: [] for l in range(n_layers)}

    def _make_record(position_type, rank, broadcast,
                     log_ctx_stack, mu_stack, log_q_stack):
        return {
            "position_type": position_type,
            "rank": rank,
            "broadcast": broadcast,
            "log_D_ctx": _stack_mean(log_ctx_stack),
            "mu_C":      _stack_mean(mu_stack),
            "log_D_q":   _stack_mean(log_q_stack),
        }

    if common_template_ranks:
        for layer_idx in range(n_layers):
            for rank in common_template_ranks:
                lc, mu, lq = [], [], []
                for c in raw:
                    rank_idx = c["active_template_ranks"].index(rank)
                    t = c["layers"][layer_idx]["template"]
                    lc.append(t[0][rank_idx])
                    mu.append(t[1][rank_idx])
                    lq.append(t[2][rank_idx])
                records_by_layer[layer_idx].append(_make_record(
                    "template", int(rank), False, lc, mu, lq,
                ))

    if config.n_content == 0:
        n_content_recs = 0
    elif config.n_content == -1:
        for layer_idx in range(n_layers):
            lc, mu, lq = [], [], []
            for c in raw:
                t = c["layers"][layer_idx]["content"]
                if t is None:
                    continue
                lc.append(t[0].mean(dim=0))
                mu.append(t[1].mean(dim=0))
                lq.append(t[2].mean(dim=0))
            if lc:
                records_by_layer[layer_idx].append(_make_record(
                    "content", None, True, lc, mu, lq,
                ))
        n_content_recs = 1
    else:
        k_max = int(config.n_content)
        produced_ranks: list[int] = []
        for rank in range(k_max):
            if any(int(c["n_content"]) > rank for c in raw):
                produced_ranks.append(rank)
        for layer_idx in range(n_layers):
            for rank in produced_ranks:
                lc, mu, lq = [], [], []
                for c in raw:
                    if rank >= int(c["n_content"]):
                        continue
                    t = c["layers"][layer_idx]["content"]
                    if t is None:
                        continue
                    lc.append(t[0][rank])
                    mu.append(t[1][rank])
                    lq.append(t[2][rank])
                if not lc:
                    continue
                records_by_layer[layer_idx].append(_make_record(
                    "content", int(rank), False, lc, mu, lq,
                ))
        n_content_recs = len(produced_ranks)

    if config.n_output == 0:
        n_output_recs = 0
    elif config.n_output == -1:
        for layer_idx in range(n_layers):
            lc, mu, lq = [], [], []
            for c in raw:
                t = c["layers"][layer_idx]["gen"]
                if t is None:
                    continue
                lc.append(t[0].mean(dim=0))
                mu.append(t[1].mean(dim=0))
                lq.append(t[2].mean(dim=0))
            if lc:
                records_by_layer[layer_idx].append(_make_record(
                    "output", None, True, lc, mu, lq,
                ))
        n_output_recs = 1
    else:
        k_max = int(config.n_output)
        produced_ranks_out: list[int] = []
        for rank in range(k_max):
            if any(int(c["n_gen"]) > rank for c in raw):
                produced_ranks_out.append(rank)
        for layer_idx in range(n_layers):
            for rank in produced_ranks_out:
                lc, mu, lq = [], [], []
                for c in raw:
                    if rank >= int(c["n_gen"]):
                        continue
                    t = c["layers"][layer_idx]["gen"]
                    if t is None:
                        continue
                    lc.append(t[0][rank])
                    mu.append(t[1][rank])
                    lq.append(t[2][rank])
                if not lc:
                    continue
                records_by_layer[layer_idx].append(_make_record(
                    "output", int(rank), False, lc, mu, lq,
                ))
        n_output_recs = len(produced_ranks_out)

    return {
        "records_by_layer": records_by_layer,
        "meta": {
            "method": "nto-batch-state",
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "n_template_ranks": list(common_template_ranks),
            "n_content_recs": int(n_content_recs),
            "n_output_recs": int(n_output_recs),
            "n_val_samples": int(len(raw)),
            "K_batch": int(K_batch_set[0]) if len(K_batch_set) == 1 else K_batch_set,
            "config": {
                "n_template": int(config.n_template),
                "n_content": int(config.n_content),
                "n_output": int(config.n_output),
                "template_active_ranks": list(config.template_active_ranks)
                if config.template_active_ranks is not None else None,
            },
        },
    }


def extract_canonical_batch_state(
    model,
    tokenizer,
    demos,
    validation_records,
    validation_icl_outputs,
    config: NTOConfig,
) -> dict:
    """Public per-batch entry. Runs one short ICL forward pass per validation
    sample and returns averaged batch-state factors (log_D_ctx, mu_C, log_D_q)
    plus the per-batch K_batch. The merge step combines N of these.
    """

    raw, skipped = [], []
    for record, icl_output in zip(validation_records, validation_icl_outputs):
        try:
            sample = _extract_one_sample_batch_state(
                model, tokenizer, demos, record, icl_output, config,
            )
            reason = None if sample is not None else "empty_target_or_span_mismatch"
        except Exception as exc:
            sample = None
            reason = f"{type(exc).__name__}: {exc}"
        if sample is None:
            skipped.append({"example_id": record.get("example_id"), "reason": reason})
        else:
            raw.append(sample)
    if not raw:
        raise RuntimeError(
            f"all {len(skipped)} validation samples failed; first 3: {skipped[:3]}"
        )
    state = _aggregate_batch_state_canonical(raw, config)
    state["meta"]["n_skipped"] = len(skipped)
    state["meta"]["skipped"] = skipped
    return state


def _init_merge_state_from_batch(batch_state: dict) -> dict:
    """Initialize the merge state from the first batch's averaged state.
    The state has the same record schema but represents an accumulator over
    `n_batches` batches with a running K_total.
    """

    n_layers = batch_state["meta"]["n_layers"]
    state = {"records_by_layer": {l: [] for l in range(n_layers)}, "meta": {}}
    for L in range(n_layers):
        for rec in batch_state["records_by_layer"].get(L, []):
            state["records_by_layer"][L].append({
                "position_type": rec["position_type"],
                "rank":          rec["rank"],
                "broadcast":     rec["broadcast"],
                "log_D_ctx":     rec["log_D_ctx"].clone().to(dtype=torch.float32),
                "mu_C":          rec["mu_C"].clone().to(dtype=torch.float32),
                "log_D_q":       rec["log_D_q"].clone().to(dtype=torch.float32),
            })
    state["meta"]["method"]      = "nto-iter"
    state["meta"]["n_layers"]    = batch_state["meta"]["n_layers"]
    state["meta"]["n_heads"]     = batch_state["meta"]["n_heads"]
    state["meta"]["head_dim"]    = batch_state["meta"]["head_dim"]
    state["meta"]["n_template_ranks"] = list(batch_state["meta"]["n_template_ranks"])
    state["meta"]["n_content_recs"]   = batch_state["meta"]["n_content_recs"]
    state["meta"]["n_output_recs"]    = batch_state["meta"]["n_output_recs"]
    state["meta"]["n_val_samples"]    = batch_state["meta"]["n_val_samples"]
    state["meta"]["n_batches"]   = 1
    state["meta"]["K_per_batch"] = int(batch_state["meta"]["K_batch"])
    state["meta"]["K_total"]     = int(batch_state["meta"]["K_batch"])
    state["meta"]["config"]      = dict(batch_state["meta"]["config"])
    return state


def _update_merge_state(state: dict, batch_state: dict) -> dict:
    """In-place online recombination of one new batch into the running state.

    For each (layer, slot, head):
        log_D_ctx_new = logaddexp(state.log_D_ctx, batch.log_D_ctx)
        α_old = exp(state.log_D_ctx - log_D_ctx_new)
        α_new = exp(batch.log_D_ctx - log_D_ctx_new)
        mu_C  ← α_old·mu_C + α_new·batch.mu_C
        log_D_ctx ← log_D_ctx_new
        log_D_q   ← (n·log_D_q + batch.log_D_q) / (n + 1)

    All accumulator math is done in fp32. K_total += K_batch.
    Asserts identical record counts per layer across batches.
    """

    n_layers = state["meta"]["n_layers"]
    n = int(state["meta"]["n_batches"])
    K_b = int(batch_state["meta"]["K_batch"])
    if int(state["meta"]["K_per_batch"]) != K_b:
        raise ValueError(
            f"K_per_batch mismatch: state has {state['meta']['K_per_batch']}, "
            f"batch has {K_b}"
        )

    for L in range(n_layers):
        srecs = state["records_by_layer"].get(L, [])
        brecs = batch_state["records_by_layer"].get(L, [])
        if len(srecs) != len(brecs):
            raise ValueError(
                f"layer {L}: record count differs between state ({len(srecs)}) "
                f"and batch ({len(brecs)})"
            )
        for sr, br in zip(srecs, brecs):
            if sr["position_type"] != br["position_type"] or sr["rank"] != br["rank"]:
                raise ValueError(
                    f"layer {L}: record schema differs (state={sr['position_type']}/{sr['rank']}, "
                    f"batch={br['position_type']}/{br['rank']})"
                )
            log_D_ctx_b = br["log_D_ctx"].to(dtype=torch.float32)
            mu_C_b      = br["mu_C"].to(dtype=torch.float32)
            log_D_q_b   = br["log_D_q"].to(dtype=torch.float32)

            log_D_ctx_new = torch.logaddexp(sr["log_D_ctx"], log_D_ctx_b)
            alpha_old = torch.exp(sr["log_D_ctx"] - log_D_ctx_new).unsqueeze(-1)
            alpha_new = torch.exp(log_D_ctx_b      - log_D_ctx_new).unsqueeze(-1)
            sr["mu_C"]      = alpha_old * sr["mu_C"] + alpha_new * mu_C_b
            sr["log_D_ctx"] = log_D_ctx_new
            sr["log_D_q"]   = (float(n) * sr["log_D_q"] + log_D_q_b) / float(n + 1)

    state["meta"]["n_batches"] = n + 1
    state["meta"]["K_total"]   = int(state["meta"]["K_total"]) + K_b
    return state


def _finalize_merge_state(state: dict, *, dtype: torch.dtype = torch.float32) -> dict:
    """Convert the running state to a `mean_circuit_nto` operator usable by
    `apply_canonical_task_operator`.

        theta_merged = log_D_ctx_total - log_D_q_running_mean - log K_total
        mu_C_merged  = mu_C_running_mean

    Diagnostic fields (`log_D_ctx`, `log_D_q`) are kept on each record for
    later analysis. The replay hook reads only `theta` and `mu_C`.
    """

    n_layers = state["meta"]["n_layers"]
    K_total = int(state["meta"]["K_total"])
    log_K_total = math.log(float(K_total))

    out_records: dict[int, list[dict]] = {l: [] for l in range(n_layers)}
    for L in range(n_layers):
        for sr in state["records_by_layer"].get(L, []):
            theta = sr["log_D_ctx"] - sr["log_D_q"] - log_K_total
            out_records[L].append({
                "position_type": sr["position_type"],
                "rank":          sr["rank"],
                "broadcast":     sr["broadcast"],
                "theta":         theta.to(dtype=dtype),
                "mu_C":          sr["mu_C"].to(dtype=dtype),
                "log_D_ctx":     sr["log_D_ctx"].to(dtype=dtype),
                "log_D_q":       sr["log_D_q"].to(dtype=dtype),
            })

    out_meta = dict(state["meta"])
    out_meta["method"]        = "nto-iter"
    out_meta["K_extraction"]  = K_total
    return {"records_by_layer": out_records, "meta": out_meta}


def extract_canonical_knowledge_iterative(
    model,
    tokenizer,
    demos_pool,
    validation_records,
    validation_icl_outputs,
    config: NTOConfig,
    *,
    K_per_batch: int,
    n_batches: int | None = None,
) -> dict:
    """Iterative canonical batch merge.

    Splits `demos_pool` into disjoint batches of size `K_per_batch` and runs
    one short ICL forward per batch. Per-batch sufficient statistics
    (log_D_ctx, mu_C, log_D_q, K_batch) are merged online into a single
    canonical operator.

    If `n_batches` is given, uses exactly that many disjoint batches (raises
    if the pool is too small). Otherwise uses `len(demos_pool) // K_per_batch`.
    """

    if K_per_batch <= 0:
        raise ValueError("K_per_batch must be positive")
    pool = list(demos_pool)
    max_b = len(pool) // K_per_batch
    if n_batches is None:
        N = max_b
    else:
        if n_batches > max_b:
            raise ValueError(
                f"n_batches={n_batches} exceeds available batches "
                f"({max_b}) in demos_pool of size {len(pool)} at K_per_batch={K_per_batch}"
            )
        N = int(n_batches)
    if N <= 0:
        raise ValueError(f"need at least one batch; got demos_pool size={len(pool)}")

    state = None
    for b in range(N):
        demos_b = pool[K_per_batch * b: K_per_batch * (b + 1)]
        batch_state = extract_canonical_batch_state(
            model, tokenizer, demos_b, validation_records, validation_icl_outputs, config,
        )
        if state is None:
            state = _init_merge_state_from_batch(batch_state)
        else:
            _update_merge_state(state, batch_state)

    return _finalize_merge_state(state)


# --------------------------------------------------- hybrid wb-iter merge
# Hybrid variant: per-batch sample aggregation uses the wb fix (linear means
# of (w, ch)), removing within-batch Jensen drift. Cross-batch composition
# uses the canonical log-sum-exp identity on D_ctx, preserving the
# "longer effective context" semantics. The boundary identity at N=1 with
# K_0 = K_b reproduces the wb-fix on a single batch (= standard TO
# per-aggregate). For N >= 2 it inherits canonical-iter's context composition.


def _extract_batch_state_hybrid_factors(batch_scores, value_states, context_span):
    """Per-batch hybrid sufficient statistics.

    Returns (log_ctx, mu_C, log_q, w, context_heads):
        log_ctx       : Tensor[n_pos, n_heads]              log D_ctx
        mu_C          : Tensor[n_pos, n_heads, head_dim]    context-only renorm. expectation (canonical track)
        log_q         : Tensor[n_pos, n_heads]              log D_q via stable log-sub-exp
        w             : Tensor[n_pos, n_heads]              paper's w = D_q / (D_ctx + D_q)
        context_heads : Tensor[n_pos, n_heads, head_dim]    paper's b = full-softmax expectation on context (wb track)
    """

    cs, ce = int(context_span[0]), int(context_span[1])
    context_scores = batch_scores[:, :, cs:ce]
    context_values = value_states[:, cs:ce, :]

    log_ctx = torch.logsumexp(context_scores, dim=-1)
    log_full = torch.logsumexp(batch_scores, dim=-1)

    diff = (log_ctx - log_full).clamp_(max=-1e-7)
    log_q = log_full + torch.log1p(-torch.exp(diff))

    ctx_probs_C = torch.exp(context_scores - log_ctx.unsqueeze(-1))
    mu_C = torch.einsum("phc,hcd->phd", ctx_probs_C, context_values)

    w = (1 - torch.exp(log_ctx - log_full)).clamp_(0, 1)
    full_probs_C = torch.exp(context_scores - log_full.unsqueeze(-1))
    context_heads = torch.einsum("phc,hcd->phd", full_probs_C, context_values)

    return log_ctx, mu_C, log_q, w, context_heads


def _extract_one_sample_batch_state_hybrid(
    model, tokenizer, demos, record, icl_output_text, config: NTOConfig,
    *,
    keep_on_device: bool = False,
) -> dict | None:
    """Run the extended ICL forward for one (record, icl_output) and return
    per-layer hybrid factors (log_D_ctx, mu_C, log_D_q, w, ch) for the
    selected positions. Mirrors `_extract_one_sample_batch_state` but the
    factor function returns five tensors instead of three.
    """

    K_batch = int(len(demos))
    if K_batch <= 0:
        return None

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    n_heads = int(text_cfg.num_attention_heads)
    n_kv_heads = int(getattr(text_cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // n_heads))

    icl_prompt = build_full_icl_prompt(
        demos, record["input"],
        demo_template=config.demo_template, query_template=config.query_template,
    )
    zsl_prompt = build_zsl_prompt(record["input"], query_template=config.query_template)

    icl_ids = encode_prompt_ids(
        tokenizer, icl_prompt, attention_sink=config.attention_sink, device=device,
    )
    target_token_ids = tokenizer(icl_output_text or "", add_special_tokens=False).input_ids
    if not target_token_ids:
        return None
    target_ids = torch.tensor([target_token_ids], device=device)
    extended_ids = torch.cat([icl_ids, target_ids], dim=1)

    context_span, query_span = identify_context_query(
        tokenizer, icl_ids, zsl_prompt, prompt_text=icl_prompt,
    )
    resolved = resolve_query_positions(
        tokenizer, icl_ids, query_span, record["input"], prompt_text=icl_prompt,
    )
    template_positions = list(resolved["template_positions"])
    content_positions = list(resolved["content_positions"])
    icl_prompt_len = int(icl_ids.shape[1])
    n_target = int(target_ids.shape[1])
    gen_positions = list(range(icl_prompt_len, icl_prompt_len + n_target))

    active_template, active_template_ranks = _select_template(template_positions, config)
    active_content = _select_first_k(content_positions, config.n_content)
    active_output = _select_first_k(gen_positions, config.n_output)
    active_all = active_template + active_content + active_output
    analysis_all = template_positions + content_positions + gen_positions

    mask_spec = []
    if active_all:
        active_set = set(active_all)
        inactive = [p for p in analysis_all if p not in active_set]
        if inactive and context_span[1] > context_span[0]:
            mask_spec.append({
                "layers": list(range(n_layers)),
                "query_rows": inactive,
                "key_span": [int(context_span[0]), int(context_span[1])],
                "alpha": 0.0,
            })

    attn_trace: dict = {}
    if keep_on_device:
        attn_trace["_keep_on_device"] = True
    with torch.inference_mode():
        with patched_attention(model, mask_spec=mask_spec, capture_store=attn_trace):
            outputs = model(
                input_ids=extended_ids, use_cache=False,
                output_hidden_states=True, return_dict=True,
            )

    decoder_layers = get_decoder_layers(model)
    layer_records = []
    for layer_idx, layer in enumerate(decoder_layers):
        layer_hidden = outputs.hidden_states[layer_idx][0]
        v_states = _per_layer_value_states(layer, layer_hidden, n_kv_heads, head_dim, model_dtype)
        if keep_on_device:
            scores = attn_trace[layer_idx]["scores_post"]
        else:
            scores = attn_trace[layer_idx]["scores_post"].to(device=device, dtype=model_dtype)

        def _factors_at(positions: list[int]):
            if not positions:
                return None
            s = scores[:, positions, :].permute(1, 0, 2).contiguous()
            out = _extract_batch_state_hybrid_factors(s, v_states, context_span)
            if keep_on_device:
                return tuple(t.detach() for t in out)
            return tuple(t.detach().cpu() for t in out)

        layer_records.append({
            "template": _factors_at(active_template),
            "content": _factors_at(active_content),
            "gen": _factors_at(active_output),
        })

    return {
        "layers": layer_records,
        "n_template": len(active_template),
        "n_content": len(active_content),
        "n_gen": len(active_output),
        "active_template_ranks": active_template_ranks,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "K_batch": K_batch,
    }


def _aggregate_batch_state_hybrid(raw: list[dict], config: NTOConfig) -> dict:
    """Average per-(layer, slot) hybrid factors across m validation samples.

    All five fields (log_D_ctx, mu_C, log_D_q, w, context_heads) are linear
    means via `_stack_mean`. The wb track (w, context_heads) is the linear
    mean by construction; the canonical track (log_D_ctx, mu_C, log_D_q) is
    sample-averaged identically to `_aggregate_batch_state_canonical` and
    kept for diagnostic / canonical-iter comparison.
    """

    if not raw:
        raise ValueError("no validation samples were processed")
    n_layers = raw[0]["n_layers"]
    n_heads = raw[0]["n_heads"]
    head_dim = raw[0]["head_dim"]
    K_batch_set = sorted({int(c["K_batch"]) for c in raw})

    common_template_ranks = sorted(set.intersection(*[set(c["active_template_ranks"]) for c in raw]))

    records_by_layer: dict[int, list[dict]] = {l: [] for l in range(n_layers)}

    def _make_record(position_type, rank, broadcast,
                     log_ctx_stack, mu_stack, log_q_stack, w_stack, ch_stack):
        return {
            "position_type": position_type,
            "rank": rank,
            "broadcast": broadcast,
            "log_D_ctx":     _stack_mean(log_ctx_stack),
            "mu_C":          _stack_mean(mu_stack),
            "log_D_q":       _stack_mean(log_q_stack),
            "w":             _stack_mean(w_stack),
            "context_heads": _stack_mean(ch_stack),
        }

    if common_template_ranks:
        for layer_idx in range(n_layers):
            for rank in common_template_ranks:
                lc, mu, lq, ws, ch = [], [], [], [], []
                for c in raw:
                    rank_idx = c["active_template_ranks"].index(rank)
                    t = c["layers"][layer_idx]["template"]
                    lc.append(t[0][rank_idx])
                    mu.append(t[1][rank_idx])
                    lq.append(t[2][rank_idx])
                    ws.append(t[3][rank_idx])
                    ch.append(t[4][rank_idx])
                records_by_layer[layer_idx].append(_make_record(
                    "template", int(rank), False, lc, mu, lq, ws, ch,
                ))

    if config.n_content == 0:
        n_content_recs = 0
    elif config.n_content == -1:
        for layer_idx in range(n_layers):
            lc, mu, lq, ws, ch = [], [], [], [], []
            for c in raw:
                t = c["layers"][layer_idx]["content"]
                if t is None:
                    continue
                lc.append(t[0].mean(dim=0))
                mu.append(t[1].mean(dim=0))
                lq.append(t[2].mean(dim=0))
                ws.append(t[3].mean(dim=0))
                ch.append(t[4].mean(dim=0))
            if lc:
                records_by_layer[layer_idx].append(_make_record(
                    "content", None, True, lc, mu, lq, ws, ch,
                ))
        n_content_recs = 1
    else:
        k_max = int(config.n_content)
        produced_ranks: list[int] = []
        for rank in range(k_max):
            if any(int(c["n_content"]) > rank for c in raw):
                produced_ranks.append(rank)
        for layer_idx in range(n_layers):
            for rank in produced_ranks:
                lc, mu, lq, ws, ch = [], [], [], [], []
                for c in raw:
                    if rank >= int(c["n_content"]):
                        continue
                    t = c["layers"][layer_idx]["content"]
                    if t is None:
                        continue
                    lc.append(t[0][rank])
                    mu.append(t[1][rank])
                    lq.append(t[2][rank])
                    ws.append(t[3][rank])
                    ch.append(t[4][rank])
                if not lc:
                    continue
                records_by_layer[layer_idx].append(_make_record(
                    "content", int(rank), False, lc, mu, lq, ws, ch,
                ))
        n_content_recs = len(produced_ranks)

    if config.n_output == 0:
        n_output_recs = 0
    elif config.n_output == -1:
        for layer_idx in range(n_layers):
            lc, mu, lq, ws, ch = [], [], [], [], []
            for c in raw:
                t = c["layers"][layer_idx]["gen"]
                if t is None:
                    continue
                lc.append(t[0].mean(dim=0))
                mu.append(t[1].mean(dim=0))
                lq.append(t[2].mean(dim=0))
                ws.append(t[3].mean(dim=0))
                ch.append(t[4].mean(dim=0))
            if lc:
                records_by_layer[layer_idx].append(_make_record(
                    "output", None, True, lc, mu, lq, ws, ch,
                ))
        n_output_recs = 1
    else:
        k_max = int(config.n_output)
        produced_ranks_out: list[int] = []
        for rank in range(k_max):
            if any(int(c["n_gen"]) > rank for c in raw):
                produced_ranks_out.append(rank)
        for layer_idx in range(n_layers):
            for rank in produced_ranks_out:
                lc, mu, lq, ws, ch = [], [], [], [], []
                for c in raw:
                    if rank >= int(c["n_gen"]):
                        continue
                    t = c["layers"][layer_idx]["gen"]
                    if t is None:
                        continue
                    lc.append(t[0][rank])
                    mu.append(t[1][rank])
                    lq.append(t[2][rank])
                    ws.append(t[3][rank])
                    ch.append(t[4][rank])
                if not lc:
                    continue
                records_by_layer[layer_idx].append(_make_record(
                    "output", int(rank), False, lc, mu, lq, ws, ch,
                ))
        n_output_recs = len(produced_ranks_out)

    return {
        "records_by_layer": records_by_layer,
        "meta": {
            "method": "nto-hybrid-batch-state",
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "n_template_ranks": list(common_template_ranks),
            "n_content_recs": int(n_content_recs),
            "n_output_recs": int(n_output_recs),
            "n_val_samples": int(len(raw)),
            "K_batch": int(K_batch_set[0]) if len(K_batch_set) == 1 else K_batch_set,
            "config": {
                "n_template": int(config.n_template),
                "n_content": int(config.n_content),
                "n_output": int(config.n_output),
                "template_active_ranks": list(config.template_active_ranks)
                if config.template_active_ranks is not None else None,
            },
        },
    }


def extract_hybrid_batch_state(
    model,
    tokenizer,
    demos,
    validation_records,
    validation_icl_outputs,
    config: NTOConfig,
) -> dict:
    """Public per-batch entry for the hybrid wb-iter merge. Runs one
    extended-ICL forward per validation sample and returns per-batch
    hybrid factors averaged across samples (linear means).
    """

    raw, skipped = [], []
    for record, icl_output in zip(validation_records, validation_icl_outputs):
        try:
            sample = _extract_one_sample_batch_state_hybrid(
                model, tokenizer, demos, record, icl_output, config,
            )
            reason = None if sample is not None else "empty_target_or_span_mismatch"
        except Exception as exc:
            sample = None
            reason = f"{type(exc).__name__}: {exc}"
        if sample is None:
            skipped.append({"example_id": record.get("example_id"), "reason": reason})
        else:
            raw.append(sample)
    if not raw:
        raise RuntimeError(
            f"all {len(skipped)} validation samples failed; first 3: {skipped[:3]}"
        )
    state = _aggregate_batch_state_hybrid(raw, config)
    state["meta"]["n_skipped"] = len(skipped)
    state["meta"]["skipped"] = skipped
    return state


def _wb_derived_per_batch(rec: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert per-batch wb aggregate (bar_w, bar_ch) into operator-side stats
    that drive the canonical cross-batch recurrence:

        ~log_D_ctx_b ≡ logit(1 - bar_w_b) = log((1-bar_w_b)/bar_w_b)
                       (relative to a fixed log_D_q_const that cancels in the
                       cross-batch logaddexp + finalize subtraction)
        ~mu_C_b      = bar_ch_b / (1 - bar_w_b)

    Numerical guards on (1 - bar_w):
      - lower floor at _WB_FLOOR avoids div-by-zero at bar_w → 1 (saturated to query)
      - upper ceiling at 1 - _WB_FLOOR avoids log(0) at bar_w → 0 (saturated to context)
    The upper ceiling matters specifically here because the cross-batch
    logaddexp + α-weighting needs FINITE log_D_ctx values: with +inf,
    `α_old = exp(state.log_D_ctx - logaddexp(state.log_D_ctx, batch.log_D_ctx))`
    becomes `exp(inf - inf) = NaN` and propagates through mu_C.

    We deliberately do NOT use bar_log_D_q from the canonical track here —
    at fully saturated heads bar_log_D_q can collapse to -inf (since per-sample
    log_D_q can be -inf when w_i ≈ 0), and routing log_D_q through both
    log_D_ctx and the finalize subtraction produces -inf - (-inf) = NaN.
    The query-tail scale is approximately constant across batches modulo
    RoPE phase, so the fixed-baseline reparameterization is equivalent at
    N=1 and a tight approximation for N > 1.
    """

    bar_w = rec["w"].to(dtype=torch.float32)
    bar_ch = rec["context_heads"].to(dtype=torch.float32)

    one_minus_w = (1.0 - bar_w).clamp_(min=_WB_FLOOR, max=1.0 - _WB_FLOOR)
    w_safe = 1.0 - one_minus_w  # in [_WB_FLOOR, 1.0 - _WB_FLOOR]
    log_D_ctx_tilde = torch.log(one_minus_w) - torch.log(w_safe)
    mu_C_tilde = bar_ch / one_minus_w.unsqueeze(-1)
    return log_D_ctx_tilde, mu_C_tilde


def _init_hybrid_merge_state_from_batch(batch_state: dict) -> dict:
    """Initialize the cross-batch hybrid accumulator from the first batch.
    Uses wb-derived per-batch operator-side stats. State carries log_D_ctx
    (relative to the implicit log_D_q baseline) and mu_C; finalize reads
    only these.
    """

    n_layers = batch_state["meta"]["n_layers"]
    state = {"records_by_layer": {l: [] for l in range(n_layers)}, "meta": {}}
    for L in range(n_layers):
        for rec in batch_state["records_by_layer"].get(L, []):
            log_D_ctx_tilde, mu_C_tilde = _wb_derived_per_batch(rec)
            state["records_by_layer"][L].append({
                "position_type": rec["position_type"],
                "rank":          rec["rank"],
                "broadcast":     rec["broadcast"],
                "log_D_ctx":     log_D_ctx_tilde.clone(),
                "mu_C":          mu_C_tilde.clone(),
            })
    state["meta"]["method"]      = "nto-hybrid-iter"
    state["meta"]["n_layers"]    = batch_state["meta"]["n_layers"]
    state["meta"]["n_heads"]     = batch_state["meta"]["n_heads"]
    state["meta"]["head_dim"]    = batch_state["meta"]["head_dim"]
    state["meta"]["n_template_ranks"] = list(batch_state["meta"]["n_template_ranks"])
    state["meta"]["n_content_recs"]   = batch_state["meta"]["n_content_recs"]
    state["meta"]["n_output_recs"]    = batch_state["meta"]["n_output_recs"]
    state["meta"]["n_val_samples"]    = batch_state["meta"]["n_val_samples"]
    state["meta"]["n_batches"]   = 1
    state["meta"]["K_per_batch"] = int(batch_state["meta"]["K_batch"])
    state["meta"]["K_total"]     = int(batch_state["meta"]["K_batch"])
    state["meta"]["config"]      = dict(batch_state["meta"]["config"])
    return state


def _update_hybrid_merge_state(state: dict, batch_state: dict) -> dict:
    """In-place online recombination: derive ~log_D_ctx_b, ~mu_C_b from the
    per-batch wb aggregate, then apply the canonical recurrence
    (logaddexp on log_D_ctx, α-weighted mu_C update).
    """

    n_layers = state["meta"]["n_layers"]
    K_b = int(batch_state["meta"]["K_batch"])
    if int(state["meta"]["K_per_batch"]) != K_b:
        raise ValueError(
            f"K_per_batch mismatch: state has {state['meta']['K_per_batch']}, "
            f"batch has {K_b}"
        )

    for L in range(n_layers):
        srecs = state["records_by_layer"].get(L, [])
        brecs = batch_state["records_by_layer"].get(L, [])
        if len(srecs) != len(brecs):
            raise ValueError(
                f"layer {L}: record count differs between state ({len(srecs)}) "
                f"and batch ({len(brecs)})"
            )
        for sr, br in zip(srecs, brecs):
            if sr["position_type"] != br["position_type"] or sr["rank"] != br["rank"]:
                raise ValueError(
                    f"layer {L}: record schema differs (state={sr['position_type']}/{sr['rank']}, "
                    f"batch={br['position_type']}/{br['rank']})"
                )
            log_D_ctx_b, mu_C_b = _wb_derived_per_batch(br)

            log_D_ctx_new = torch.logaddexp(sr["log_D_ctx"], log_D_ctx_b)
            alpha_old = torch.exp(sr["log_D_ctx"] - log_D_ctx_new).unsqueeze(-1)
            alpha_new = torch.exp(log_D_ctx_b      - log_D_ctx_new).unsqueeze(-1)
            sr["mu_C"]      = alpha_old * sr["mu_C"] + alpha_new * mu_C_b
            sr["log_D_ctx"] = log_D_ctx_new

    state["meta"]["n_batches"] = int(state["meta"]["n_batches"]) + 1
    state["meta"]["K_total"]   = int(state["meta"]["K_total"]) + K_b
    return state


def _finalize_hybrid_merge_state(state: dict, *, dtype: torch.dtype = torch.float32) -> dict:
    """Convert the running hybrid state to a `mean_circuit_nto`.
        theta = log_D_ctx - log K_total      (log_D_q baseline cancels)
        mu_C  = state.mu_C                   (cross-batch α-weighted)
    """

    n_layers = state["meta"]["n_layers"]
    K_total = int(state["meta"]["K_total"])
    log_K_total = math.log(float(K_total))

    out_records: dict[int, list[dict]] = {l: [] for l in range(n_layers)}
    for L in range(n_layers):
        for sr in state["records_by_layer"].get(L, []):
            theta = sr["log_D_ctx"] - log_K_total
            out_records[L].append({
                "position_type": sr["position_type"],
                "rank":          sr["rank"],
                "broadcast":     sr["broadcast"],
                "theta":         theta.to(dtype=dtype),
                "mu_C":          sr["mu_C"].to(dtype=dtype),
                "log_D_ctx":     sr["log_D_ctx"].to(dtype=dtype),
            })

    out_meta = dict(state["meta"])
    out_meta["method"]        = "nto-hybrid-iter"
    out_meta["K_extraction"]  = K_total
    return {"records_by_layer": out_records, "meta": out_meta}


def extract_canonical_knowledge_iterative_hybrid(
    model,
    tokenizer,
    demos_pool,
    validation_records,
    validation_icl_outputs,
    config: NTOConfig,
    *,
    K_per_batch: int,
    n_batches: int | None = None,
) -> dict:
    """Iterative hybrid wb-iter merge.

    Splits `demos_pool` into disjoint batches of size `K_per_batch`. For each
    batch, runs `extract_hybrid_batch_state` (wb-fix within batch). The
    per-batch wb aggregates are then merged online via the canonical
    cross-batch recurrence, and finalized to a single `mean_circuit_nto`
    that plugs into `apply_canonical_task_operator` unchanged.
    """

    if K_per_batch <= 0:
        raise ValueError("K_per_batch must be positive")
    pool = list(demos_pool)
    max_b = len(pool) // K_per_batch
    if n_batches is None:
        N = max_b
    else:
        if n_batches > max_b:
            raise ValueError(
                f"n_batches={n_batches} exceeds available batches "
                f"({max_b}) in demos_pool of size {len(pool)} at K_per_batch={K_per_batch}"
            )
        N = int(n_batches)
    if N <= 0:
        raise ValueError(f"need at least one batch; got demos_pool size={len(pool)}")

    state = None
    for b in range(N):
        demos_b = pool[K_per_batch * b: K_per_batch * (b + 1)]
        batch_state = extract_hybrid_batch_state(
            model, tokenizer, demos_b, validation_records, validation_icl_outputs, config,
        )
        if state is None:
            state = _init_hybrid_merge_state_from_batch(batch_state)
        else:
            _update_hybrid_merge_state(state, batch_state)

    return _finalize_hybrid_merge_state(state)


# ------------------------------------------------------------ slot-id helpers
def _slot_id_for_record(rec: dict) -> str:
    pt = rec["position_type"]
    if rec.get("broadcast"):
        return f"{pt}:block"
    return f"{pt}:{int(rec['rank'])}"


def _layer_active_for_slot(slot_id: str, layer_idx: int, layers_per_slot: dict | None) -> bool:
    if layers_per_slot is None:
        return True
    return layer_idx in {int(l) for l in layers_per_slot.get(slot_id, [])}


# ------------------------------------------------------------ K_0 reconstruction
def _reconstruct_factors_canonical(rec: dict, K_0: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct (w_tilde, b_tilde) from canonical (theta, mu_C) at reference K_0.

        p_tilde = sigmoid(theta + log K_0)
        w_tilde = 1 - p_tilde
        b_tilde = p_tilde * mu_C

    Computed in fp32 for numerical safety, cast back to the stored dtype.
    """

    log_K0 = math.log(float(K_0))
    theta = rec["theta"]
    mu_C = rec["mu_C"]
    theta_f = theta.to(dtype=torch.float32)
    p_tilde = torch.sigmoid(theta_f + log_K0)                 # [n_heads], fp32
    w_tilde = (1.0 - p_tilde).to(dtype=theta.dtype)           # [n_heads]
    b_tilde = (p_tilde.unsqueeze(-1) * mu_C.to(dtype=torch.float32)).to(dtype=mu_C.dtype)
    return w_tilde, b_tilde


def _build_prompt_records_canonical(
    mean_circuit: dict,
    zsl_template_positions: list[int],
    zsl_content_positions: list[int],
    config: NTOConfig,
) -> dict[int, list[dict]]:
    """Per-layer prompt-time records: list of {token_idx, w, context_heads},
    where (w, context_heads) are the canonical reconstructions at K_0.

    Same routing logic as task_operator._build_prompt_records, swapping in the
    K_0 reconstruction. The hook in generation.py reads `r["w"]` and
    `r["context_heads"]` so the canonical (w_tilde, b_tilde) need to be packed
    under those keys.
    """

    rb = mean_circuit["records_by_layer"]
    n_layers = mean_circuit["meta"]["n_layers"]
    out: dict[int, list[dict]] = {l: [] for l in range(n_layers)}

    template_rank_set = (
        {int(r) for r in config.template_active_ranks}
        if config.template_active_ranks is not None
        else None
    )
    n_template_zsl = len(zsl_template_positions)
    n_content_zsl = len(zsl_content_positions)

    K_0 = float(config.K_0)
    for layer_idx in range(n_layers):
        for rec in rb[layer_idx]:
            pt = rec["position_type"]
            slot_id = _slot_id_for_record(rec)
            if pt == "template":
                rank = int(rec["rank"])
                if template_rank_set is not None and rank not in template_rank_set:
                    continue
                if not _layer_active_for_slot(slot_id, layer_idx, config.active_layers_per_slot):
                    continue
                if rank < n_template_zsl:
                    w_used, ch_used = _reconstruct_factors_canonical(rec, K_0)
                    out[layer_idx].append({
                        "token_idx": int(zsl_template_positions[rank]),
                        "w": w_used, "context_heads": ch_used,
                    })
            elif pt == "content":
                if not _layer_active_for_slot(slot_id, layer_idx, config.active_layers_per_slot):
                    continue
                if rec.get("broadcast"):
                    w_used, ch_used = _reconstruct_factors_canonical(rec, K_0)
                    for token_idx in zsl_content_positions:
                        out[layer_idx].append({
                            "token_idx": int(token_idx),
                            "w": w_used, "context_heads": ch_used,
                        })
                else:
                    rank = int(rec["rank"])
                    if rank < n_content_zsl:
                        w_used, ch_used = _reconstruct_factors_canonical(rec, K_0)
                        out[layer_idx].append({
                            "token_idx": int(zsl_content_positions[rank]),
                            "w": w_used, "context_heads": ch_used,
                        })
    return out


def _build_gen_records_canonical(mean_circuit: dict, config: NTOConfig) -> dict[int, list[dict]]:
    """Per-layer gen-step records for output position-type with canonical reconstruction."""

    rb = mean_circuit["records_by_layer"]
    n_layers = mean_circuit["meta"]["n_layers"]
    out: dict[int, list[dict]] = {l: [] for l in range(n_layers)}
    K_0 = float(config.K_0)
    for layer_idx in range(n_layers):
        for rec in rb[layer_idx]:
            if rec["position_type"] != "output":
                continue
            slot_id = _slot_id_for_record(rec)
            if not _layer_active_for_slot(slot_id, layer_idx, config.active_layers_per_slot):
                continue
            w_used, ch_used = _reconstruct_factors_canonical(rec, K_0)
            out[layer_idx].append({
                "broadcast": bool(rec.get("broadcast")),
                "step": int(rec["rank"]) if rec.get("rank") is not None else None,
                "w": w_used,
                "context_heads": ch_used,
            })
    return out


def apply_canonical_task_operator(
    model,
    tokenizer,
    test_record,
    mean_circuit: dict,
    config: NTOConfig,
) -> dict:
    """Run the normalized task operator on one test record. Returns generation payload.

    Reconstructs (w_tilde, b_tilde) from (theta, mu_C) at config.K_0 and applies
    them via the same forward hook used by the standard TO.
    """

    device = next(model.parameters()).device
    zsl_prompt = build_zsl_prompt(test_record["input"], query_template=config.query_template)
    zsl_ids = encode_prompt_ids(
        tokenizer, zsl_prompt, attention_sink=config.attention_sink, device=device,
    )

    from .tokenization import leading_sink_offset
    sink_off = leading_sink_offset(tokenizer, zsl_ids)
    zsl_prompt_len = int(zsl_ids.shape[1])
    zsl_query_span = (sink_off, zsl_prompt_len)
    resolved = resolve_query_positions(
        tokenizer, zsl_ids, zsl_query_span, test_record["input"], prompt_text=zsl_prompt,
    )
    zsl_template_positions = list(resolved["template_positions"])
    zsl_content_positions = list(resolved["content_positions"])

    prompt_records = _build_prompt_records_canonical(
        mean_circuit, zsl_template_positions, zsl_content_positions, config,
    )
    gen_records = _build_gen_records_canonical(mean_circuit, config)

    payload = generate_with_hooks(
        model, tokenizer, zsl_prompt,
        prompt_records=prompt_records,
        gen_records=gen_records,
        attention_sink=config.attention_sink,
        max_new_tokens=config.max_new_tokens,
        stop_strings=config.stopping_strings,
        answer_phrases=config.answer_phrases,
        repetition_penalty=config.repetition_penalty,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    n_prompt_recs = sum(len(v) for v in prompt_records.values())
    n_gen_recs = sum(len(v) for v in gen_records.values())
    payload["n_prompt_records_total"] = int(n_prompt_recs)
    payload["n_gen_records_total"] = int(n_gen_recs)
    payload["n_template_positions"] = len(zsl_template_positions)
    payload["n_content_positions"] = len(zsl_content_positions)
    payload["K_0"] = float(config.K_0)
    return payload


# ============================================================================
#   wb-REPARAMETERIZATION (Proposal 1)
# ============================================================================
#
# The default NTO aggregator averages (theta, mu_C) per-sample and stores
# those means. At K_0 = K_extraction the boundary identity then holds only
# per-sample, not per-aggregate, because sigma(mean(logit(1-w_i))) != mean(1-w_i)
# (Jensen) and p_tilde * mean(mu_C_i) != mean((1-w_i) * mu_C_i) (non-linear
# product). Empirically this means NTO@K_0=K_extraction trails the standard TO
# at the aggregate level on most non-saturated tasks.
#
# wb-reparameterization fixes this. Aggregate (w, ch) arithmetically (TO's
# order), then reparameterize once into the canonical coordinates:
#
#     theta_wb = logit(1 - bar_w) - log K_extraction
#     mu_C_wb  = bar_ch / (1 - bar_w)
#
# At replay K_0 = K_extraction:
#     p_tilde = sigma(theta_wb + log K_extraction) = sigma(logit(1-bar_w)) = 1-bar_w
#     w_tilde = bar_w
#     b_tilde = (1-bar_w) * (bar_ch / (1-bar_w)) = bar_ch
# i.e. recovers the standard TO operator exactly (per-aggregate). At K_0 != K_extraction
# the K_0 knob acts as a logit shift around the TO baseline.


_WB_FLOOR = 1e-7


def _reparameterize_wb_to_canonical(
    bar_w: torch.Tensor,
    bar_ch: torch.Tensor,
    K_extraction: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert aggregated (bar_w, bar_ch) into canonical (theta, mu_C) such that
    NTO replay at K_0 = K_extraction reproduces (bar_w, bar_ch) exactly.

    bar_w  : Tensor[..., n_heads]
    bar_ch : Tensor[..., n_heads, head_dim]
    K_extraction : extraction shot count (scalar)

    Numerical guards on (1 - bar_w):
      - lower floor at _WB_FLOOR avoids div-by-zero at bar_w → 1.
      - upper ceiling at 1 - _WB_FLOOR avoids log(0) at bar_w → 0; without it
        theta = +inf at fully-context-saturated slots, which is harmless at
        single-batch replay (sigma(+inf) = 1) but produces NaN under the
        cross-batch α-weighting in the iterative hybrid merge.
    Both clamps are well within bf16's representable range and have no
    functional effect at moderately saturated slots.
    """

    one_minus_w = (1.0 - bar_w).clamp_(min=_WB_FLOOR, max=1.0 - _WB_FLOOR)
    w_safe = 1.0 - one_minus_w  # in [_WB_FLOOR, 1.0 - _WB_FLOOR]
    theta = torch.log(one_minus_w) - torch.log(w_safe) - math.log(float(K_extraction))
    mu_C = bar_ch / one_minus_w.unsqueeze(-1)
    return theta, mu_C


def reparameterize_nto_wb(mean_circuit: dict, *, K_extraction: int | None = None) -> dict:
    """Take an existing canonical mean_circuit (with diagnostic `w` and
    `context_heads` fields) and return a new mean_circuit whose `theta` and
    `mu_C` are derived from (bar_w, bar_ch) via the wb-reparameterization.

    The original diagnostic fields are preserved so replay still reads
    (theta, mu_C) but the boundary identity at K_0 = K_extraction now holds
    per-aggregate (i.e. matches the standard TO aggregator's output exactly).
    """

    K = int(K_extraction if K_extraction is not None else mean_circuit["meta"]["K_extraction"])
    new_records: dict[int, list[dict]] = {}
    for layer_idx, recs in mean_circuit["records_by_layer"].items():
        new_recs = []
        for rec in recs:
            if "w" not in rec or "context_heads" not in rec:
                raise KeyError(
                    f"record at layer {layer_idx} (slot {rec.get('position_type')}/{rec.get('rank')}) "
                    "is missing diagnostic 'w' or 'context_heads'; "
                    "wb-reparameterization needs both."
                )
            theta_wb, mu_C_wb = _reparameterize_wb_to_canonical(
                rec["w"].to(dtype=torch.float32),
                rec["context_heads"].to(dtype=torch.float32),
                K,
            )
            new_recs.append({
                "position_type": rec["position_type"],
                "rank": rec["rank"],
                "broadcast": rec["broadcast"],
                "theta":         theta_wb.to(dtype=rec["theta"].dtype if "theta" in rec else torch.float32),
                "mu_C":          mu_C_wb.to(dtype=rec["mu_C"].dtype  if "mu_C"  in rec else torch.float32),
                "w":             rec["w"],
                "context_heads": rec["context_heads"],
            })
        new_records[int(layer_idx)] = new_recs

    new_meta = dict(mean_circuit["meta"])
    new_meta["method"] = "nto-wb"
    new_meta["aggregation"] = "wb-arithmetic-then-reparam"
    new_meta["K_extraction"] = K
    return {"records_by_layer": new_records, "meta": new_meta}


def nto_config_from_default(
    task: str,
    *,
    K_0: float = 8.0,
    repetition_penalty: float = 1.1,
) -> NTOConfig:
    """Build an NTOConfig with the same task-category defaults as
    `experiments.make_default_config`, plus the canonical-method K_0 knob.

    All position knobs default to "all" (n_template=-1, n_content=-1,
    n_output=-1) which matches the `allpos` setting on the standard TO.
    """

    from .experiments import make_default_config

    base = make_default_config(task, repetition_penalty=repetition_penalty)
    return NTOConfig(
        attention_sink=base.attention_sink,
        demo_template=base.demo_template,
        query_template=base.query_template,
        n_template=base.n_template,
        n_content=base.n_content,
        n_output=base.n_output,
        template_active_ranks=base.template_active_ranks,
        active_layers_per_slot=base.active_layers_per_slot,
        repetition_penalty=base.repetition_penalty,
        max_new_tokens=base.max_new_tokens,
        stopping_strings=base.stopping_strings,
        answer_phrases=base.answer_phrases,
        do_sample=base.do_sample,
        temperature=base.temperature,
        top_p=base.top_p,
        top_k=base.top_k,
        K_0=float(K_0),
    )
