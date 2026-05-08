"""task_operator: knowledge extraction + steered generation.

The method extracts an affine replay circuit `(w, context_heads)` per (layer, slot)
from a set of validation samples under K=8 demonstrations, then applies the circuit
at o_proj forward passes during zero-shot generation on test queries.

Slot grammar (matching v1's notebook conventions):
    template:rank      — per-rank template position (rank ∈ [0, n_template))
    content:rank       — per-rank content position (only when n_content > 0)
    content:block      — broadcast across all content positions (n_content == -1)
    output:rank        — per-rank generated position (only when n_output > 0)
    output:block       — broadcast across all generated positions (n_output == -1)

Position knobs:
    n_template / n_content / n_output  — -1 (all/broadcast), 0 (skip), k>0 (first-k per-rank)
    template_active_ranks               — explicit list of template ranks; if set,
                                          overrides n_template (and the search-time slot
                                          restriction `template:i` for i in this list)
    active_layers_per_slot              — dict[slot_id -> list of layer indices]; if set,
                                          a slot is only patched at those layers; None ⇒
                                          all layers active for every slot
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class TaskOperatorConfig:
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
    # Generation stops at the first newline after any phrase in this tuple
    # appears (case-insensitive; ≥1 non-newline char between phrase and newline).
    # Reasoning tasks default to ("answer is",) which covers "The answer is",
    # "the correct answer is", "Therefore, the answer is", etc. — the literal
    # match is case-insensitive via re.IGNORECASE so a single entry suffices.
    answer_phrases: tuple[str, ...] = ()

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    # ---- ablation knobs ----------------------------------------------------
    # Selects which (w, ch) tensors get used at apply time:
    #   "full"   — existing operator (w, context_heads).
    #   "w_only" — multiplicative only: (w_refit, 0). Requires compute_refit=True at extraction.
    #   "b_only" — additive only: (1, ch_refit). Requires compute_refit=True at extraction.
    ablation_mode: str = "full"
    # When True, extraction also computes per-sample x_full and x_zsl =
    # (x_full - ch) / w (perturbed-ICL representation), and aggregation
    # produces `w_refit` and `ch_refit` per slot. Off by default to preserve
    # legacy behaviour for search.py (which does not need refits and where
    # the extra per-sample tensors would inflate the extraction cache on
    # reasoning tasks with long output sequences).
    compute_refit: bool = False


# -------------------------------------------------------------------- position helpers
def _select_template(template_positions: list[int], config: TaskOperatorConfig) -> tuple[list[int], list[int]]:
    """Return (positions, ranks). Both lists are aligned: positions[i] is the token index
    of the i-th selected template position; ranks[i] is its rank in template_positions."""

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


# ------------------------------------------------------------- replay-factor extraction
def _native_dtype(value, dtype):
    return value if value.dtype == dtype else value.to(dtype=dtype)


def _extract_replay_factors(
    batch_scores, batch_probs, value_states, context_span, *, compute_refit: bool = False,
):
    """Return (w, context_heads[, x_full, x_zsl]) per (position, head).

    batch_scores / batch_probs : Tensor[n_pos, n_heads, n_keys]
    value_states               : Tensor[n_heads, n_keys, head_dim]
    context_span               : (start, end) of context keys

    When `compute_refit=True`, additionally returns:
      x_full = einsum("phk,hkd->phd", batch_probs, value_states)
            (= the full attention output at each position before o_proj)
      x_zsl  = (x_full - context_heads) / w  (clamped at 1e-6 to guard /0)
            (= the perturbed-ICL representation: what the attention output
             would be if attention to the context span were zeroed out).
    These satisfy the identity x_full = w * x_zsl + context_heads by
    construction. Used by the w/b ablation refits.
    """

    cs, ce = int(context_span[0]), int(context_span[1])
    context_scores = batch_scores[:, :, cs:ce]
    context_probs = batch_probs[:, :, cs:ce]
    context_values = value_states[:, cs:ce, :]

    log_context = torch.logsumexp(context_scores, dim=-1)
    log_full = torch.logsumexp(batch_scores, dim=-1)
    w = (1 - torch.exp(log_context - log_full)).clamp_(0, 1)
    context_heads = torch.einsum("phc,hcd->phd", context_probs, context_values)
    if not compute_refit:
        return w, context_heads
    x_full = torch.einsum("phk,hkd->phd", batch_probs, value_states)
    x_zsl = (x_full - context_heads) / w.unsqueeze(-1).clamp(min=1e-6)
    return w, context_heads, x_full, x_zsl


def _per_layer_value_states(layer, hidden_state, num_kv_heads: int, head_dim: int, model_dtype) -> torch.Tensor:
    """Replicate the v_proj path of the layer for one captured layer-input hidden state."""

    norm_dtype = getattr(layer.input_layernorm.weight, "dtype", model_dtype)
    attn_input = layer.input_layernorm(_native_dtype(hidden_state, norm_dtype))
    v_linear = layer.self_attn.v_proj(attn_input)
    v_states = v_linear.view(v_linear.shape[0], num_kv_heads, head_dim).permute(1, 0, 2)
    return v_states.repeat_interleave(int(layer.self_attn.num_key_value_groups), dim=0)


# --------------------------------------------------------- per-sample circuit extraction
def _extract_one_sample(
    model, tokenizer, demos, record, icl_output_text, config: TaskOperatorConfig,
    *,
    keep_on_device: bool = False,
    compute_refit: bool = False,
) -> dict | None:
    """Run the extended ICL forward for one (record, icl_output) and return per-layer
    (w, context_heads) tensors for the selected positions.

    Returns None if the sample can't be processed (empty target, span mismatch, etc.).

    `keep_on_device=True` skips per-layer GPU↔CPU sync of attention scores/probs and
    of the returned (w, ch). Used by callers that aggregate immediately on-device
    (e.g. search.py's reconstruction-NLL evaluator). Default False keeps the legacy
    behavior bit-identical for `extract_knowledge` and other downstream consumers.

    `compute_refit=True` makes per-slot returns 4-tuples (w, ch, x_full, x_zsl)
    instead of 2-tuples (w, ch); needed for the w/b ablation refits computed
    later in `_aggregate_circuits`. Default False matches search.py's path.
    """

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
        # Tells attention.py's patched eager forward to skip per-layer .cpu() on the
        # captured scores/probs (the flag is read inside `patched_attention(...)`).
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
            probs = attn_trace[layer_idx]["attn_probs"]
        else:
            scores = attn_trace[layer_idx]["scores_post"].to(device=device, dtype=model_dtype)
            probs = attn_trace[layer_idx]["attn_probs"].to(device=device, dtype=model_dtype)

        def _factors_at(positions: list[int]):
            if not positions:
                return None
            s = scores[:, positions, :].permute(1, 0, 2).contiguous()
            p = probs[:, positions, :].permute(1, 0, 2).contiguous()
            out = _extract_replay_factors(
                s, p, v_states, context_span, compute_refit=compute_refit,
            )
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
    }


# ------------------------------------------------------------ aggregation across samples
def _stack_mean(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(tensors), dim=0).mean(dim=0)


def _maybe_refit(xf_stack: list, xz_stack: list) -> dict:
    """Compute aggregated (w_refit, ch_refit) from per-sample x_full / x_zsl
    stacks, both [n_samples, ..., n_heads, head_dim]. Returns an empty dict if
    the stacks are empty (indicating compute_refit was off during extraction).

    Refit formulas (per layer / head / position):
      ch_refit = E_v[x_full - x_zsl]                 (best fixed additive correction)
      w_refit  = sum_v <x_full,x_zsl> / sum_v <x_zsl,x_zsl>   (OLS scaling)
    """

    if not xf_stack or not xz_stack:
        return {}
    xf = torch.stack(xf_stack, dim=0)
    xz = torch.stack(xz_stack, dim=0)
    ch_refit = (xf - xz).mean(dim=0)
    num = (xf * xz).sum(dim=-1).sum(dim=0)
    denom = (xz * xz).sum(dim=-1).sum(dim=0)
    w_refit = num / (denom + 1e-12)
    return {"w_refit": w_refit, "ch_refit": ch_refit}


def _aggregate_circuits(raw: list[dict], config: TaskOperatorConfig) -> dict:
    """Average per-(layer, slot) factors across validation samples to build a mean_circuit."""

    if not raw:
        raise ValueError("no validation samples were processed")
    n_layers = raw[0]["n_layers"]
    n_heads = raw[0]["n_heads"]
    head_dim = raw[0]["head_dim"]

    # Common template ranks across all samples (intersection); usually identical.
    common_template_ranks = sorted(set.intersection(*[set(c["active_template_ranks"]) for c in raw]))

    min_n_content = min(c["n_content"] for c in raw)
    min_n_gen = min(c["n_gen"] for c in raw)

    records_by_layer: dict[int, list[dict]] = {l: [] for l in range(n_layers)}

    # ---- template (per-rank) ----
    if common_template_ranks:
        for layer_idx in range(n_layers):
            for rank in common_template_ranks:
                w_stack, ch_stack = [], []
                xf_stack, xz_stack = [], []
                for c in raw:
                    rank_idx = c["active_template_ranks"].index(rank)
                    t = c["layers"][layer_idx]["template"]
                    w_stack.append(t[0][rank_idx])
                    ch_stack.append(t[1][rank_idx])
                    if len(t) >= 4:
                        xf_stack.append(t[2][rank_idx])
                        xz_stack.append(t[3][rank_idx])
                rec = {
                    "position_type": "template",
                    "rank": int(rank),
                    "broadcast": False,
                    "w": _stack_mean(w_stack),
                    "context_heads": _stack_mean(ch_stack),
                }
                rec.update(_maybe_refit(xf_stack, xz_stack))
                records_by_layer[layer_idx].append(rec)

    # ---- content ----
    if config.n_content == 0:
        n_content_recs = 0
    elif config.n_content == -1:
        # broadcast: per-sample mean across content positions, then cross-sample mean
        for layer_idx in range(n_layers):
            w_stack, ch_stack = [], []
            xf_stack, xz_stack = [], []
            for c in raw:
                t = c["layers"][layer_idx]["content"]
                if t is None:
                    continue
                w_stack.append(t[0].mean(dim=0))
                ch_stack.append(t[1].mean(dim=0))
                if len(t) >= 4:
                    xf_stack.append(t[2].mean(dim=0))
                    xz_stack.append(t[3].mean(dim=0))
            if w_stack:
                rec = {
                    "position_type": "content",
                    "rank": None,
                    "broadcast": True,
                    "w": _stack_mean(w_stack),
                    "context_heads": _stack_mean(ch_stack),
                }
                rec.update(_maybe_refit(xf_stack, xz_stack))
                records_by_layer[layer_idx].append(rec)
        n_content_recs = 1
    else:
        # Per-rank availability: produce content:rank for every rank in [0, n_content)
        # that at least one sample provided (sample's n_content > rank). This matches
        # the search's behavior — search emits slot keys content:0..n_content-1 even
        # when some samples had fewer content positions, and phase-3 layer pruning
        # evaluates each slot only on samples that actually have it. We mirror that
        # here so the searched active_layers_per_slot keys land in mean_circuit even
        # when min_n_content is small.
        k_max = int(config.n_content)
        produced_ranks: list[int] = []
        for rank in range(k_max):
            if any(int(c["n_content"]) > rank for c in raw):
                produced_ranks.append(rank)
        for layer_idx in range(n_layers):
            for rank in produced_ranks:
                w_stack, ch_stack = [], []
                xf_stack, xz_stack = [], []
                for c in raw:
                    if rank >= int(c["n_content"]):
                        continue
                    t = c["layers"][layer_idx]["content"]
                    if t is None:
                        continue
                    w_stack.append(t[0][rank])
                    ch_stack.append(t[1][rank])
                    if len(t) >= 4:
                        xf_stack.append(t[2][rank])
                        xz_stack.append(t[3][rank])
                if not w_stack:
                    continue
                rec = {
                    "position_type": "content",
                    "rank": int(rank),
                    "broadcast": False,
                    "w": _stack_mean(w_stack),
                    "context_heads": _stack_mean(ch_stack),
                }
                rec.update(_maybe_refit(xf_stack, xz_stack))
                records_by_layer[layer_idx].append(rec)
        n_content_recs = len(produced_ranks)

    # ---- output ----
    if config.n_output == 0:
        n_output_recs = 0
    elif config.n_output == -1:
        for layer_idx in range(n_layers):
            w_stack, ch_stack = [], []
            xf_stack, xz_stack = [], []
            for c in raw:
                t = c["layers"][layer_idx]["gen"]
                if t is None:
                    continue
                w_stack.append(t[0].mean(dim=0))
                ch_stack.append(t[1].mean(dim=0))
                if len(t) >= 4:
                    xf_stack.append(t[2].mean(dim=0))
                    xz_stack.append(t[3].mean(dim=0))
            if w_stack:
                rec = {
                    "position_type": "output",
                    "rank": None,
                    "broadcast": True,
                    "w": _stack_mean(w_stack),
                    "context_heads": _stack_mean(ch_stack),
                }
                rec.update(_maybe_refit(xf_stack, xz_stack))
                records_by_layer[layer_idx].append(rec)
        n_output_recs = 1
    else:
        # Per-rank availability for output, same as content above.
        k_max = int(config.n_output)
        produced_ranks_out: list[int] = []
        for rank in range(k_max):
            if any(int(c["n_gen"]) > rank for c in raw):
                produced_ranks_out.append(rank)
        for layer_idx in range(n_layers):
            for rank in produced_ranks_out:
                w_stack, ch_stack = [], []
                xf_stack, xz_stack = [], []
                for c in raw:
                    if rank >= int(c["n_gen"]):
                        continue
                    t = c["layers"][layer_idx]["gen"]
                    if t is None:
                        continue
                    w_stack.append(t[0][rank])
                    ch_stack.append(t[1][rank])
                    if len(t) >= 4:
                        xf_stack.append(t[2][rank])
                        xz_stack.append(t[3][rank])
                if not w_stack:
                    continue
                rec = {
                    "position_type": "output",
                    "rank": int(rank),
                    "broadcast": False,
                    "w": _stack_mean(w_stack),
                    "context_heads": _stack_mean(ch_stack),
                }
                rec.update(_maybe_refit(xf_stack, xz_stack))
                records_by_layer[layer_idx].append(rec)
        n_output_recs = len(produced_ranks_out)

    return {
        "records_by_layer": records_by_layer,
        "meta": {
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "n_template_ranks": list(common_template_ranks),
            "n_content_recs": int(n_content_recs),
            "n_output_recs": int(n_output_recs),
            "n_val_samples": int(len(raw)),
            "config": {
                "n_template": int(config.n_template),
                "n_content": int(config.n_content),
                "n_output": int(config.n_output),
                "template_active_ranks": list(config.template_active_ranks)
                if config.template_active_ranks is not None else None,
            },
        },
    }


def extract_knowledge(
    model,
    tokenizer,
    demos,
    validation_records,
    validation_icl_outputs,
    config: TaskOperatorConfig,
    *,
    return_per_sample: bool = False,
) -> dict:
    """Extract a mean_circuit averaged across validation samples.

    For each (record, icl_output_text), runs an extended ICL forward of
    icl_prompt + icl_output_text with attention masking that exempts the active
    template/content/output positions from blocking, captures per-layer attention
    logits + probs, and computes (w, context_heads) per active position.

    Tracks samples that fail to process (empty target, span mismatch, hook errors) in
    `meta["skipped"]` rather than silently dropping them. When `config.active_layers_per_slot`
    is provided, asserts that every slot key appears in the aggregated mean_circuit so a
    rank/ordinal mismatch (cf. F1) is caught explicitly rather than silently disabling
    patches.

    When `return_per_sample=True`, returns `(mean_circuit, raw)` where `raw` is the
    list of per-sample dicts produced by `_extract_one_sample` — useful for
    cross-sample / cross-token analyses before aggregation.
    """

    raw, skipped = [], []
    for record, icl_output in zip(validation_records, validation_icl_outputs):
        try:
            sample = _extract_one_sample(
                model, tokenizer, demos, record, icl_output, config,
                compute_refit=bool(getattr(config, "compute_refit", False)),
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
    mean_circuit = _aggregate_circuits(raw, config)
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
            # Template misses indicate a logic bug (search and task_operator disagree on
            # how to encode template ranks — the F1 issue); raise loudly. Content/output
            # misses are a data-availability shortfall (no validation sample provided
            # enough positions for that rank); the runtime in _build_prompt_records and
            # _build_gen_records skips records that aren't in mean_circuit, so warn and
            # continue. Record the skipped slots in meta for downstream inspection.
            template_missing = [sid for sid in missing if sid.startswith("template:")]
            other_missing = [sid for sid in missing if not sid.startswith("template:")]
            if template_missing:
                raise ValueError(
                    f"template slot(s) in active_layers_per_slot not present in "
                    f"mean_circuit: {template_missing}. "
                    f"available sample: {sorted(available_slot_ids)[:8]}... "
                    "This indicates a search/apply rank-encoding mismatch (cf. F1)."
                )
            if other_missing:
                print(
                    f"  [warn] active_layers_per_slot has {len(other_missing)} content/"
                    f"output slot(s) with no matching mean_circuit records "
                    f"(no validation sample had enough positions): {other_missing}. "
                    "These slots are skipped at apply time."
                )
                mean_circuit["meta"]["skipped_active_layers_per_slot_keys"] = list(other_missing)
    if return_per_sample:
        return mean_circuit, raw
    return mean_circuit


# ------------------------------------------------------------ slot-id and record routing
def _slot_id_for_record(rec: dict) -> str:
    pt = rec["position_type"]
    if rec.get("broadcast"):
        return f"{pt}:block"
    return f"{pt}:{int(rec['rank'])}"


def _layer_active_for_slot(slot_id: str, layer_idx: int, layers_per_slot: dict | None) -> bool:
    if layers_per_slot is None:
        return True
    return layer_idx in {int(l) for l in layers_per_slot.get(slot_id, [])}


def _select_factors(rec: dict, ablation_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick which (w, context_heads) tensors get applied for this record under the given
    ablation mode.

    Two ablation flavours, both available at apply time:
      * `*_refit`  — the OLS-optimal refit computed at extraction time using the
        analytical perturbed-ICL representation x_zsl = (x_full - ch) / w. Requires
        `w_refit` / `ch_refit` to be present in `rec` (compute_refit=True at extract).
      * `*_lit`    — literal: keep the original component, zero the other. No refit
        needed; works on legacy mean_circuit blobs.

    Bare `w_only` / `b_only` resolve to the refit variant when refit fields are
    available; otherwise they fall back to literal. The naming `_lit` / `_refit` is
    used in the predictions setting name to distinguish output dirs."""

    if ablation_mode in ("w_only", "w_only_refit"):
        if "w_refit" in rec and "ch_refit" in rec:
            return rec["w_refit"], torch.zeros_like(rec["ch_refit"])
        # Fall back to literal if refit fields aren't present
        return rec["w"], torch.zeros_like(rec["context_heads"])
    if ablation_mode == "w_only_lit":
        return rec["w"], torch.zeros_like(rec["context_heads"])
    if ablation_mode in ("b_only", "b_only_refit"):
        if "ch_refit" in rec and "w_refit" in rec:
            return torch.ones_like(rec["w_refit"]), rec["ch_refit"]
        return torch.ones_like(rec["w"]), rec["context_heads"]
    if ablation_mode == "b_only_lit":
        return torch.ones_like(rec["w"]), rec["context_heads"]
    return rec["w"], rec["context_heads"]


def _build_prompt_records(
    mean_circuit: dict,
    zsl_template_positions: list[int],
    zsl_content_positions: list[int],
    config: TaskOperatorConfig,
) -> dict[int, list[dict]]:
    """Per-layer prompt-time records: list of {token_idx, w, context_heads}."""

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

    abl = getattr(config, "ablation_mode", "full") or "full"
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
                    w_used, ch_used = _select_factors(rec, abl)
                    out[layer_idx].append({
                        "token_idx": int(zsl_template_positions[rank]),
                        "w": w_used, "context_heads": ch_used,
                    })
            elif pt == "content":
                if not _layer_active_for_slot(slot_id, layer_idx, config.active_layers_per_slot):
                    continue
                if rec.get("broadcast"):
                    w_used, ch_used = _select_factors(rec, abl)
                    for token_idx in zsl_content_positions:
                        out[layer_idx].append({
                            "token_idx": int(token_idx),
                            "w": w_used, "context_heads": ch_used,
                        })
                else:
                    rank = int(rec["rank"])
                    if rank < n_content_zsl:
                        w_used, ch_used = _select_factors(rec, abl)
                        out[layer_idx].append({
                            "token_idx": int(zsl_content_positions[rank]),
                            "w": w_used, "context_heads": ch_used,
                        })
            # output records handled by _build_gen_records
    return out


def _build_gen_records(mean_circuit: dict, config: TaskOperatorConfig) -> dict[int, list[dict]]:
    """Per-layer gen-step records for output position-type."""

    rb = mean_circuit["records_by_layer"]
    n_layers = mean_circuit["meta"]["n_layers"]
    out: dict[int, list[dict]] = {l: [] for l in range(n_layers)}
    abl = getattr(config, "ablation_mode", "full") or "full"
    for layer_idx in range(n_layers):
        for rec in rb[layer_idx]:
            if rec["position_type"] != "output":
                continue
            slot_id = _slot_id_for_record(rec)
            if not _layer_active_for_slot(slot_id, layer_idx, config.active_layers_per_slot):
                continue
            w_used, ch_used = _select_factors(rec, abl)
            out[layer_idx].append({
                "broadcast": bool(rec.get("broadcast")),
                "step": int(rec["rank"]) if rec.get("rank") is not None else None,
                "w": w_used,
                "context_heads": ch_used,
            })
    return out


def apply_task_operator(
    model,
    tokenizer,
    test_record,
    mean_circuit: dict,
    config: TaskOperatorConfig,
) -> dict:
    """Run task_operator on one test record. Returns the generation payload."""

    device = next(model.parameters()).device
    zsl_prompt = build_zsl_prompt(test_record["input"], query_template=config.query_template)
    zsl_ids = encode_prompt_ids(
        tokenizer, zsl_prompt, attention_sink=config.attention_sink, device=device,
    )

    # On the ZSL prompt, the entire prompt is the query (no context).
    from .tokenization import leading_sink_offset
    sink_off = leading_sink_offset(tokenizer, zsl_ids)
    zsl_prompt_len = int(zsl_ids.shape[1])
    zsl_query_span = (sink_off, zsl_prompt_len)
    resolved = resolve_query_positions(
        tokenizer, zsl_ids, zsl_query_span, test_record["input"], prompt_text=zsl_prompt,
    )
    zsl_template_positions = list(resolved["template_positions"])
    zsl_content_positions = list(resolved["content_positions"])

    prompt_records = _build_prompt_records(mean_circuit, zsl_template_positions, zsl_content_positions, config)
    gen_records = _build_gen_records(mean_circuit, config)

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
    return payload


def apply_full_icl(
    model,
    tokenizer,
    demos,
    test_record,
    config: TaskOperatorConfig,
) -> dict:
    """Generate the K-demo full-ICL completion for one test record (no hooks).

    Used in notebook 02 for both the K=8 ICL test condition and the validation full-ICL
    teacher-forcing source for notebooks 03+.
    """

    icl_prompt = build_full_icl_prompt(
        demos, test_record["input"],
        demo_template=config.demo_template, query_template=config.query_template,
    )
    return generate_with_hooks(
        model, tokenizer, icl_prompt,
        prompt_records=None, gen_records=None,
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


def apply_zero_shot(
    model,
    tokenizer,
    test_record,
    config: TaskOperatorConfig,
) -> dict:
    zsl_prompt = build_zsl_prompt(test_record["input"], query_template=config.query_template)
    return generate_with_hooks(
        model, tokenizer, zsl_prompt,
        prompt_records=None, gen_records=None,
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
