"""Worstcase greedy minimal-active-set search.

For one (model, task), runs the v1 `causal_patching_attn_search_per_position_worstcase`
procedure on validation samples (not test samples) with the actual ICL output as the
teacher-forced target. The search target is o_proj outputs, the recovery aggregation is
worst-case (`min` across samples), and the search yields:

    Phase 1 + 2: a config (template_active_ranks, n_content, n_output)
    Phase 3:     per-slot active layer subsets

Both are persisted to the trajectory file and the final_config file. The procedure is
idempotent: rerunning skips probes whose `signature` is already in the trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .attention import patched_attention
from .io import atomic_write_json
from .model import get_decoder_layers, get_text_config
from .prompting import build_full_icl_prompt, build_zsl_prompt
from .spans import identify_context_query, resolve_query_positions
from .task_operator import (
    TaskOperatorConfig,
    _aggregate_circuits,
    _extract_one_sample,
    _slot_id_for_record,
)
from .tokenization import encode_prompt_ids, leading_sink_offset


# ---------------------------------------------------------------------------- settings
@dataclass
class SearchSettings:
    recovery_threshold: float = 0.95
    recovery_decimals: int = 3
    batch_size: int = 8
    output_search_values: list[int] = field(default_factory=lambda: [8, 4, 2, 1])
    content_search_values: list[int] = field(default_factory=lambda: [8, 4, 2, 1])
    layer_search_pt_order: list[str] = field(default_factory=lambda: ["output", "content", "template"])
    n_validation: int | None = None  # None ⇒ use all provided
    # NLL metric used during search probes:
    #   'reconstruction' (default): patches o_proj output with the apply-time-aligned
    #     affine reconstruction `(x_zsl[pos] · scale_w + ch.flat) @ W_o^T`, where
    #     (w, context_heads) come from `task_operator._extract_one_sample`. Matches
    #     what `task_operator.apply_task_operator` actually does at test time.
    #   'substitution' (legacy): patches o_proj output with the actual ICL output
    #     captured from the K-shot ICL forward. Reproduces the pre-2026-04-27 search.
    metric: str = "reconstruction"


# ------------------------------------------------------------------ sample preparation
def prepare_sample(
    tokenizer,
    demos,
    record,
    icl_output_text,
    *,
    demo_template,
    query_template,
    attention_sink,
    device,
) -> dict | None:
    """Build per-sample objects used during the search. Returns None on bad samples."""

    if not icl_output_text:
        return None
    target_token_ids = tokenizer(icl_output_text, add_special_tokens=False).input_ids
    if not target_token_ids:
        return None
    target_ids = torch.tensor([target_token_ids], device=device)
    n_target = int(target_ids.shape[1])

    icl_prompt = build_full_icl_prompt(
        demos, record["input"],
        demo_template=demo_template, query_template=query_template,
    )
    zsl_prompt = build_zsl_prompt(record["input"], query_template=query_template)
    icl_ids = encode_prompt_ids(tokenizer, icl_prompt, attention_sink=attention_sink, device=device)
    zsl_ids = encode_prompt_ids(tokenizer, zsl_prompt, attention_sink=attention_sink, device=device)

    context_span, query_span = identify_context_query(
        tokenizer, icl_ids, zsl_prompt, prompt_text=icl_prompt,
    )
    resolved = resolve_query_positions(
        tokenizer, icl_ids, query_span, record["input"], prompt_text=icl_prompt,
    )
    template_positions = list(resolved["template_positions"])
    content_positions = list(resolved["content_positions"])

    icl_prompt_len = int(icl_ids.shape[1])
    zsl_prompt_len = int(zsl_ids.shape[1])
    icl_target_positions = list(range(icl_prompt_len, icl_prompt_len + n_target))
    zsl_target_positions = list(range(zsl_prompt_len, zsl_prompt_len + n_target))

    n_q_icl = int(query_span[1]) - int(query_span[0])
    sink_offset = leading_sink_offset(tokenizer, zsl_ids)
    n_q_zsl = zsl_prompt_len - sink_offset
    if n_q_icl != n_q_zsl:
        return None

    return {
        "icl_ids": icl_ids,
        "zsl_ids": zsl_ids,
        "target_ids": target_ids,
        "icl_prompt_len": icl_prompt_len,
        "zsl_prompt_len": zsl_prompt_len,
        "n_target": n_target,
        "template_positions": template_positions,
        "content_positions": content_positions,
        "icl_target_positions": icl_target_positions,
        "zsl_target_positions": zsl_target_positions,
        "context_span": (int(context_span[0]), int(context_span[1])),
        "query_span": (int(query_span[0]), int(query_span[1])),
        "sink_offset": int(sink_offset),
    }


def _zsl_offset(sample, position_kind: str) -> int:
    if position_kind == "query":
        return sample["sink_offset"] - sample["query_span"][0]
    if position_kind == "output":
        return sample["zsl_prompt_len"] - sample["icl_prompt_len"]
    raise ValueError(position_kind)


# ------------------------------------------------------------------ position selection
def _select_template_positions(template_positions: list[int], active_indices: list[int] | None) -> list[int]:
    if active_indices is None:
        return list(template_positions)
    n = len(template_positions)
    return [template_positions[i] for i in active_indices if 0 <= int(i) < n]


def _select_first_k(positions: list[int], k: int) -> list[int]:
    if k == -1:
        return list(positions)
    if k == 0:
        return []
    return list(positions)[: int(k)]


def _compute_active_for_sample(sample, T_active, c_active, o_active):
    active_template = _select_template_positions(sample["template_positions"], T_active)
    active_content = _select_first_k(sample["content_positions"], c_active)
    active_output = _select_first_k(sample["icl_target_positions"], o_active)
    analysis_icl = (
        list(sample["template_positions"])
        + list(sample["content_positions"])
        + list(sample["icl_target_positions"])
    )
    return active_template, active_content, active_output, analysis_icl


def _build_mask_spec(context_span, analysis_positions, active_positions, n_layers):
    active_set = {int(p) for p in active_positions}
    inactive = [int(p) for p in analysis_positions if int(p) not in active_set]
    if not inactive:
        return []
    return [{
        "layers": list(range(n_layers)),
        "query_rows": inactive,
        "key_span": [int(context_span[0]), int(context_span[1])],
        "alpha": 0.0,
    }]


# -------------------------------------------------------------------- NLL computation
def _per_sample_nll_from_logits(logits, prompt_lens, target_ids_padded, target_lens) -> list[float]:
    out = []
    for i in range(logits.shape[0]):
        n = int(target_lens[i])
        start = int(prompt_lens[i]) - 1
        slice_logits = logits[i, start : start + n, :].float()
        log_probs = F.log_softmax(slice_logits, dim=-1)
        target = target_ids_padded[i].to(log_probs.device, dtype=torch.long)
        nll_per_token = -log_probs.gather(1, target.unsqueeze(-1)).squeeze(-1)
        out.append(float(nll_per_token.mean().item()))
    return out


def _pad_and_stack(tensor_list, pad_value, device):
    L_max = max(int(t.shape[1]) for t in tensor_list)
    out = torch.full(
        (len(tensor_list), L_max), pad_value,
        dtype=tensor_list[0].dtype, device=device,
    )
    attn_mask = torch.zeros((len(tensor_list), L_max), dtype=torch.long, device=device)
    for i, t in enumerate(tensor_list):
        L = int(t.shape[1])
        out[i, :L] = t[0, :L]
        attn_mask[i, :L] = 1
    return out, attn_mask


# ------------------------------------------------------- granular per-stage o_proj hooks
def _make_oproj_substitute_hook(batch_idx, pos_idx, value):
    def hook(module, inputs, output):
        if output.shape[1] == 1:
            return output
        out = output.clone()
        out[batch_idx, pos_idx, :] = value.to(out.device, out.dtype)
        return out
    return hook


def _register_per_stage_hooks(model, per_stage_batch_idx, per_stage_pos_idx, per_stage_values):
    handles = []
    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    assert len(per_stage_batch_idx) == n_layers, (len(per_stage_batch_idx), n_layers)
    for k in range(n_layers):
        if len(per_stage_batch_idx[k]) == 0:
            continue
        h = decoder_layers[k].self_attn.o_proj.register_forward_hook(
            _make_oproj_substitute_hook(per_stage_batch_idx[k], per_stage_pos_idx[k], per_stage_values[k])
        )
        handles.append(h)
    return handles


def _clear_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# ----------------------------------------------------------------- batched ZSL evaluator
def _batched_zsl_nll(
    model, tokenizer, sample_indices, samples,
    *,
    per_sample_per_stage_active=None,
    per_sample_per_stage_values=None,
    batch_size=8,
):
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    n_layers = len(get_decoder_layers(model))
    nlls: list[float | None] = [None] * len(sample_indices)
    for batch_start in range(0, len(sample_indices), batch_size):
        local = list(range(batch_start, min(batch_start + batch_size, len(sample_indices))))
        global_ = [sample_indices[i] for i in local]
        batch = [samples[g] for g in global_]
        extended = [torch.cat([s["zsl_ids"], s["target_ids"]], dim=1) for s in batch]
        input_ids, attn_mask = _pad_and_stack(extended, pad_id, device)

        handles = []
        if per_sample_per_stage_active is not None:
            per_stage_batch_idx: list[Any] = [[]] * n_layers
            per_stage_pos_idx: list[Any] = [[]] * n_layers
            per_stage_values: list[Any] = [None] * n_layers
            for k in range(n_layers):
                bi, pi, vs = [], [], []
                for local_idx, global_idx in enumerate(global_):
                    pos_at_k = per_sample_per_stage_active[global_idx][k]
                    val_at_k = per_sample_per_stage_values[global_idx][k]
                    if not pos_at_k:
                        continue
                    bi.extend([local_idx] * len(pos_at_k))
                    pi.extend(int(p) for p in pos_at_k)
                    vs.append(val_at_k)
                if bi:
                    per_stage_batch_idx[k] = torch.tensor(bi, dtype=torch.long, device=device)
                    per_stage_pos_idx[k] = torch.tensor(pi, dtype=torch.long, device=device)
                    per_stage_values[k] = torch.cat(vs, dim=0)
                else:
                    per_stage_batch_idx[k] = []
                    per_stage_pos_idx[k] = []
                    per_stage_values[k] = None
            handles = _register_per_stage_hooks(model, per_stage_batch_idx, per_stage_pos_idx, per_stage_values)

        try:
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids, attention_mask=attn_mask,
                    use_cache=False, output_hidden_states=False, return_dict=True,
                )
            prompt_lens = [s["zsl_prompt_len"] for s in batch]
            target_padded = [s["target_ids"][0] for s in batch]
            target_lens = [s["n_target"] for s in batch]
            batch_nlls = _per_sample_nll_from_logits(out.logits, prompt_lens, target_padded, target_lens)
        finally:
            _clear_hooks(handles)
        for li, val in zip(local, batch_nlls):
            nlls[li] = val
        del out
    return [float(x) for x in nlls if x is not None]


def _batched_icl_nll_no_hooks(model, sample_indices, samples, *, batch_size=8):
    device = next(model.parameters()).device
    pad_id = 0
    nlls: list[float | None] = [None] * len(sample_indices)
    for batch_start in range(0, len(sample_indices), batch_size):
        local = list(range(batch_start, min(batch_start + batch_size, len(sample_indices))))
        global_ = [sample_indices[i] for i in local]
        batch = [samples[g] for g in global_]
        extended = [torch.cat([s["icl_ids"], s["target_ids"]], dim=1) for s in batch]
        input_ids, attn_mask = _pad_and_stack(extended, pad_id, device)
        with torch.inference_mode():
            out = model(
                input_ids=input_ids, attention_mask=attn_mask,
                use_cache=False, output_hidden_states=False, return_dict=True,
            )
        prompt_lens = [s["icl_prompt_len"] for s in batch]
        target_padded = [s["target_ids"][0] for s in batch]
        target_lens = [s["n_target"] for s in batch]
        batch_nlls = _per_sample_nll_from_logits(out.logits, prompt_lens, target_padded, target_lens)
        for li, val in zip(local, batch_nlls):
            nlls[li] = val
        del out
    return [float(x) for x in nlls if x is not None]


# ============================================================================
# Reconstruction-NLL evaluator (apply-time-aligned)
# ============================================================================
# Mirrors `task_operator.apply_task_operator`'s residual update at o_proj:
#   y[pos, :] = (x[pos, :] * scale_w + ch.flat) @ W_o^T
# where (w, context_heads) come from `_extract_one_sample` and are aggregated
# across validation samples via `_aggregate_circuits` (so per-rank slots use the
# cross-sample mean per rank, broadcast slots use the mean over both samples and
# positions — matching apply-time exactly).

def _make_oproj_recon_hook(batch_idx: torch.Tensor, pos_idx: torch.Tensor,
                            w_stack: torch.Tensor, ch_stack: torch.Tensor):
    """Forward hook on `decoder_layers[ℓ].self_attn.o_proj`. On forward passes
    with seq_len > 1, replaces output[batch_idx[i], pos_idx[i], :] with
    `(x[batch_idx[i], pos_idx[i], :] * scale_w + ch.flat) @ W_o^T`."""

    def hook(module, inputs, output):
        if output.shape[1] == 1:
            return output
        x = inputs[0]
        out = output.clone()
        N = batch_idx.shape[0]
        head_dim = ch_stack.shape[-1]
        b = batch_idx.to(x.device)
        p = pos_idx.to(x.device)
        x_at = x[b, p, :]                                                       # (N, n_heads*head_dim)
        scale = w_stack.repeat_interleave(head_dim, dim=-1).to(x.device, x.dtype)
        ch_flat = ch_stack.reshape(N, -1).to(x.device, x.dtype)
        recon = (x_at * scale + ch_flat) @ module.weight.T.to(x.device, x.dtype)
        out[b, p, :] = recon
        return out
    return hook


def _zsl_positions_for_record(sample: dict, rec: dict) -> list[int]:
    """ZSL positions where `rec` (a `mean_circuit` record) fires for `sample`. Same
    mapping `task_operator._build_prompt_records` / `_build_gen_records` use, but for
    the combined `cat([zsl_ids, target_ids])` forward used in NLL eval."""

    pt = rec["position_type"]
    off_q = _zsl_offset(sample, "query")
    off_o = _zsl_offset(sample, "output")
    if pt == "template":
        rank = int(rec["rank"])
        if 0 <= rank < len(sample["template_positions"]):
            return [int(sample["template_positions"][rank] + off_q)]
        return []
    if pt == "content":
        if rec.get("broadcast"):
            return [int(p + off_q) for p in sample["content_positions"]]
        rank = int(rec["rank"])
        if 0 <= rank < len(sample["content_positions"]):
            return [int(sample["content_positions"][rank] + off_q)]
        return []
    if pt == "output":
        if rec.get("broadcast"):
            return [int(p + off_o) for p in sample["icl_target_positions"]]
        rank = int(rec["rank"])
        if 0 <= rank < len(sample["icl_target_positions"]):
            return [int(sample["icl_target_positions"][rank] + off_o)]
        return []
    raise ValueError(pt)


def _build_recon_factors_for_batch(
    batch_samples: list[dict],
    mean_circuit: dict,
    n_layers: int,
    active_stages_per_slot: dict | None = None,
):
    """Per layer, return (batch_idx_list, pos_idx_list, w_list, ch_list) — flattened
    across (sample, record). Layer rows with no entries are skipped at hook time."""

    rb = mean_circuit["records_by_layer"]
    out_b: list[list[int]] = [[] for _ in range(n_layers)]
    out_p: list[list[int]] = [[] for _ in range(n_layers)]
    out_w: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    out_ch: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    for layer_idx in range(n_layers):
        for rec in rb[layer_idx]:
            slot_id = _slot_id_for_record(rec)
            if active_stages_per_slot is not None:
                stages = active_stages_per_slot.get(slot_id, [])
                if layer_idx not in {int(s) for s in stages}:
                    continue
            for batch_pos, sample in enumerate(batch_samples):
                zsl_ps = _zsl_positions_for_record(sample, rec)
                for p in zsl_ps:
                    out_b[layer_idx].append(int(batch_pos))
                    out_p[layer_idx].append(int(p))
                    out_w[layer_idx].append(rec["w"])
                    out_ch[layer_idx].append(rec["context_heads"])
    return out_b, out_p, out_w, out_ch


def _batched_zsl_nll_recon(
    model,
    tokenizer,
    sample_indices: list[int],
    samples: list[dict],
    *,
    mean_circuit: dict,
    active_stages_per_slot: dict | None = None,
    batch_size: int = 8,
) -> list[float]:
    """Mirror of `_batched_zsl_nll` using the apply-time-aligned recon hook."""

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    decoder_layers = get_decoder_layers(model)
    n_layers = len(decoder_layers)
    nlls: list[float | None] = [None] * len(sample_indices)

    for batch_start in range(0, len(sample_indices), batch_size):
        local = list(range(batch_start, min(batch_start + batch_size, len(sample_indices))))
        global_ = [sample_indices[i] for i in local]
        batch = [samples[g] for g in global_]
        extended = [torch.cat([s["zsl_ids"], s["target_ids"]], dim=1) for s in batch]
        input_ids, attn_mask = _pad_and_stack(extended, pad_id, device)

        bs_b, bs_p, bs_w, bs_ch = _build_recon_factors_for_batch(
            batch, mean_circuit, n_layers, active_stages_per_slot,
        )
        handles = []
        for k in range(n_layers):
            if not bs_b[k]:
                continue
            b_t = torch.tensor(bs_b[k], dtype=torch.long, device=device)
            p_t = torch.tensor(bs_p[k], dtype=torch.long, device=device)
            w_t = torch.stack(bs_w[k], dim=0)        # (N, n_heads)
            ch_t = torch.stack(bs_ch[k], dim=0)      # (N, n_heads, head_dim)
            handles.append(decoder_layers[k].self_attn.o_proj.register_forward_hook(
                _make_oproj_recon_hook(b_t, p_t, w_t, ch_t)
            ))
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids, attention_mask=attn_mask,
                    use_cache=False, output_hidden_states=False, return_dict=True,
                )
            prompt_lens = [s["zsl_prompt_len"] for s in batch]
            target_padded = [s["target_ids"][0] for s in batch]
            target_lens = [s["n_target"] for s in batch]
            batch_nlls = _per_sample_nll_from_logits(out.logits, prompt_lens, target_padded, target_lens)
        finally:
            _clear_hooks(handles)

        for li, val in zip(local, batch_nlls):
            nlls[li] = val
        del out
    return [float(x) for x in nlls if x is not None]


def _build_to_config_for_recon(
    T_active, c_active: int, o_active: int,
    *, demo_template: str, query_template: str, attention_sink: bool,
) -> TaskOperatorConfig:
    """Construct a TaskOperatorConfig matching the trial config, used by the recon
    evaluator's `_extract_one_sample` and `_aggregate_circuits` calls.

    `template_active_ranks` is set from T_active (None → all template positions;
    [] → drop template; [r1, r2, ...] → keep those ranks). `n_template` is left at
    -1 (ignored when `template_active_ranks` is non-None).
    """
    return TaskOperatorConfig(
        attention_sink=attention_sink,
        demo_template=demo_template,
        query_template=query_template,
        n_template=-1,
        n_content=int(c_active),
        n_output=int(o_active),
        template_active_ranks=list(T_active) if T_active is not None else None,
        active_layers_per_slot=None,
    )


def _evaluate_config_recon(
    model, tokenizer,
    samples: list[dict],
    sample_records: list[dict],
    sample_icl_outputs: list[str],
    n_layers: int,
    T_active, c_active: int, o_active: int,
    *,
    demos,
    demo_template: str,
    query_template: str,
    attention_sink: bool,
    active_stages_per_slot: dict | None = None,
    extraction_cache: list | None = None,
    batch_size: int = 8,
):
    """Reconstruction-NLL counterpart of `_evaluate_config`. Returns
    `(per_sample_nll, raw_extractions)`.

    `extraction_cache`: list aligned with `samples`; entries are per-sample raw
    extractions (output of `_extract_one_sample`). When present, skips re-extraction
    — used in Phase 3 where (T_active, c_active, o_active) is fixed.
    """

    cfg_for_extract = _build_to_config_for_recon(
        T_active, c_active, o_active,
        demo_template=demo_template, query_template=query_template, attention_sink=attention_sink,
    )
    raw: list[dict | None] = []
    for i, _ in enumerate(samples):
        if extraction_cache is not None and extraction_cache[i] is not None:
            raw.append(extraction_cache[i])
        else:
            try:
                # keep_on_device=False forces per-layer CPU offload of attention
                # scores/probs inside `patched_attention`. Slower per call but caps
                # GPU memory at ~one layer's attention buffers (~1 GB on long
                # sequences) instead of all-layers (~38 GB on reasoning tasks
                # with seq_len ~2900). Required to avoid OOM on gpqa/gsm8k/math500.
                extr = _extract_one_sample(
                    model, tokenizer, demos, sample_records[i], sample_icl_outputs[i],
                    cfg_for_extract, keep_on_device=False,
                )
            except Exception:
                extr = None
            raw.append(extr)

    valid_idx = [i for i, e in enumerate(raw) if e is not None]
    if not valid_idx:
        return [], raw
    valid_raw = [raw[i] for i in valid_idx]
    valid_samples = [samples[i] for i in valid_idx]

    cfg = _build_to_config_for_recon(
        T_active, c_active, o_active,
        demo_template=demo_template, query_template=query_template, attention_sink=attention_sink,
    )
    if active_stages_per_slot is not None:
        cfg.active_layers_per_slot = {sid: list(stgs) for sid, stgs in active_stages_per_slot.items()}
    mean_circuit = _aggregate_circuits(valid_raw, cfg)

    nlls = _batched_zsl_nll_recon(
        model, tokenizer, list(range(len(valid_samples))), valid_samples,
        mean_circuit=mean_circuit, active_stages_per_slot=active_stages_per_slot,
        batch_size=batch_size,
    )
    return nlls, raw


# ------------------------------------------------------------- per-sample ICL extraction
def _extract_icl_for_sample(model, sample, T_active, c_active, o_active, n_layers):
    """Run the extended ICL forward for one sample under the masking that exempts the
    selected active positions; capture o_proj output at every layer and slice the active
    rows. Returns the per-stage active-position list (in ICL coordinates), the active
    counts per pt, and the per-stage captured tensors at the active positions."""

    decoder_layers = get_decoder_layers(model)
    assert n_layers == len(decoder_layers)

    active_template, active_content, active_output, analysis_icl = _compute_active_for_sample(
        sample, T_active, c_active, o_active,
    )
    template_ranks = (
        [int(r) for r in T_active]
        if T_active is not None
        else list(range(len(sample["template_positions"])))
    )
    icl_active = list(active_template) + list(active_content) + list(active_output)

    extended_icl = torch.cat([sample["icl_ids"], sample["target_ids"]], dim=1)
    mask_spec = _build_mask_spec(sample["context_span"], analysis_icl, icl_active, n_layers)

    captured: dict[int, torch.Tensor] = {}

    def make_capture_hook(layer_idx):
        def hook(module, inputs, output):
            captured[layer_idx] = output[0].detach()
            return output
        return hook

    handles = []
    if icl_active:
        for k, layer in enumerate(decoder_layers):
            handles.append(layer.self_attn.o_proj.register_forward_hook(make_capture_hook(k)))

    try:
        with torch.inference_mode(), patched_attention(model, mask_spec=mask_spec):
            model(input_ids=extended_icl, use_cache=False, return_dict=True)
    finally:
        _clear_hooks(handles)

    if icl_active:
        per_stage_values = [captured[k][icl_active, :].clone() for k in range(n_layers)]
    else:
        per_stage_values = [None] * n_layers

    return {
        "per_stage_values": per_stage_values,
        "icl_active": icl_active,
        "active_template": list(active_template),
        "active_content": list(active_content),
        "active_output": list(active_output),
        "template_ranks": template_ranks,
    }


def _project_icl_to_zsl(sample, extraction):
    off_q = _zsl_offset(sample, "query")
    off_o = _zsl_offset(sample, "output")
    return {
        "zsl_template": [int(p + off_q) for p in extraction["active_template"]],
        "zsl_content": [int(p + off_q) for p in extraction["active_content"]],
        "zsl_output": [int(p + off_o) for p in extraction["active_output"]],
        "zsl_active": [
            *(p + off_q for p in extraction["active_template"]),
            *(p + off_q for p in extraction["active_content"]),
            *(p + off_o for p in extraction["active_output"]),
        ],
    }


def _build_per_stage_active_uniform(extraction, projection, n_stages):
    if not extraction["icl_active"]:
        return [[] for _ in range(n_stages)], [None] * n_stages
    pos = [list(projection["zsl_active"]) for _ in range(n_stages)]
    val = [extraction["per_stage_values"][k] for k in range(n_stages)]
    return pos, val


def _build_per_stage_active_per_slot(extraction, projection, active_stages_per_slot, n_stages):
    """Map per-(slot, stage) selections to ZSL positions and o_proj-row tensors.

    Template slot IDs are `template:{actual_rank}` — translate rank → ordinal for the row
    index into `per_stage_values` (rows are stored in T_active order, not rank order).
    Content/output kinds are already ordinals, no translation needed.
    """

    n_t = len(extraction["active_template"])
    n_c = len(extraction["active_content"])
    n_o = len(extraction["active_output"])
    pt_offset = {"template": 0, "content": n_t, "output": n_t + n_c}
    pt_count = {"template": n_t, "content": n_c, "output": n_o}
    pt_zsl = {
        "template": projection["zsl_template"],
        "content": projection["zsl_content"],
        "output": projection["zsl_output"],
    }
    template_rank_to_ordinal = {
        int(rank): ordinal for ordinal, rank in enumerate(extraction.get("template_ranks") or [])
    }

    pos = [[] for _ in range(n_stages)]
    val: list[Any] = [None for _ in range(n_stages)]
    if not extraction["icl_active"]:
        return pos, val

    for k in range(n_stages):
        cur_pos, cur_val = [], []
        for slot_id, stages in active_stages_per_slot.items():
            if k not in stages:
                continue
            pt, kind = slot_id.split(":", 1)
            zsl_p = pt_zsl[pt]
            if not zsl_p:
                continue
            if kind == "block":
                row_start = pt_offset[pt]
                row_end = row_start + pt_count[pt]
                cur_pos.extend(zsl_p)
                cur_val.append(extraction["per_stage_values"][k][row_start:row_end, :])
            else:
                if pt == "template":
                    rank = int(kind)
                    if rank not in template_rank_to_ordinal:
                        continue
                    idx = template_rank_to_ordinal[rank]
                else:
                    idx = int(kind)
                    if idx >= pt_count[pt]:
                        continue
                row = pt_offset[pt] + idx
                cur_pos.append(zsl_p[idx])
                cur_val.append(extraction["per_stage_values"][k][row : row + 1, :])
        pos[k] = cur_pos
        val[k] = torch.cat(cur_val, dim=0) if cur_val else None
    return pos, val


# ---------------------------------------------------------- evaluate_config orchestrator
def _evaluate_config(
    model, tokenizer, samples, n_stages,
    T_active, c_active, o_active,
    *,
    active_stages_per_slot=None,
    extraction_cache=None,
    batch_size=8,
):
    n = len(samples)
    pp_active, pp_values, extractions = [], [], []
    for s_idx, s in enumerate(samples):
        if extraction_cache is not None and extraction_cache[s_idx] is not None:
            extr = extraction_cache[s_idx]
        else:
            extr = _extract_icl_for_sample(model, s, T_active, c_active, o_active, n_stages)
        extractions.append(extr)
        proj = _project_icl_to_zsl(s, extr)
        if active_stages_per_slot is None:
            pos, val = _build_per_stage_active_uniform(extr, proj, n_stages)
        else:
            pos, val = _build_per_stage_active_per_slot(extr, proj, active_stages_per_slot, n_stages)
        pp_active.append(pos)
        pp_values.append(val)

    if all(not e["icl_active"] for e in extractions):
        nlls = _batched_zsl_nll(
            model, tokenizer, list(range(n)), samples,
            per_sample_per_stage_active=None, per_sample_per_stage_values=None,
            batch_size=batch_size,
        )
    else:
        nlls = _batched_zsl_nll(
            model, tokenizer, list(range(n)), samples,
            per_sample_per_stage_active=pp_active, per_sample_per_stage_values=pp_values,
            batch_size=batch_size,
        )
    return nlls, extractions


# ---------------------------------------------------------------------- recovery formula
def _recovery_per_sample(per_sample_nll, floor_nll, ceil_nll) -> list[float]:
    out = []
    for nll, f, c in zip(per_sample_nll, floor_nll, ceil_nll):
        span = max(float(f) - float(c), 1e-9)
        out.append(float((float(f) - float(nll)) / span))
    return out


def _aggregate_min(rs: list[float]) -> float:
    return float(min(rs)) if rs else 0.0


def _recovery_from_mean_nll(nll_mean, floor_mean, ceil_mean) -> float:
    span = max(float(floor_mean) - float(ceil_mean), 1e-9)
    return float((float(floor_mean) - float(nll_mean)) / span)


# --------------------------------------------------------- trajectory persistence helpers
def _config_signature(phase, T_active, c, o, active_stages_per_slot=None):
    if T_active is None:
        t_str = "FULL"
    else:
        t_str = ",".join(str(int(i)) for i in T_active)
    sig = f"{phase}|t={t_str}|c={int(c)}|o={int(o)}"
    if active_stages_per_slot is not None:
        for slot_id in sorted(active_stages_per_slot):
            sig += f"|{slot_id}=" + ",".join(str(s) for s in sorted(active_stages_per_slot[slot_id]))
    return sig


def _find_in_trajectory(traj, sig):
    for entry in traj["trajectory"]:
        if entry.get("signature") == sig:
            return entry
    return None


def _build_slot_list(T_active, c_active, o_active):
    """Slot IDs:
        template:{actual_rank} — using the rank value from T_active (e.g., template:5)
        content:{i} | content:block
        output:{i} | output:block
    Template uses actual ranks (not ordinals) so the IDs align with what
    `task_operator._slot_id_for_record` produces, which is critical when phase 2
    keeps a non-contiguous subset.
    """

    slots = []
    if T_active is not None:
        for r in T_active:
            slots.append(f"template:{int(r)}")
    if c_active != 0:
        if int(c_active) > 0:
            for i in range(int(c_active)):
                slots.append(f"content:{i}")
        else:
            slots.append("content:block")
    if o_active != 0:
        if int(o_active) > 0:
            for i in range(int(o_active)):
                slots.append(f"output:{i}")
        else:
            slots.append("output:block")
    return slots


# --------------------------------------------------------------------- public entry point
def run_search(
    model,
    tokenizer,
    demos,
    validation_records,
    validation_icl_outputs,
    *,
    trajectory_path: Path,
    final_config_path: Path,
    settings: SearchSettings = SearchSettings(),
    demo_template: str,
    query_template: str,
    attention_sink: bool = True,
) -> dict:
    """Run the worstcase greedy search and persist trajectory + final_config.

    Skips probes already recorded in the trajectory (idempotent on rerun).
    """

    device = next(model.parameters()).device
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)

    samples: list[dict] = []
    sample_records: list[dict] = []     # parallel to `samples`; needed by recon path
    sample_icl_outputs: list[str] = []
    n_total = settings.n_validation if settings.n_validation is not None else len(validation_records)
    n_total = min(n_total, len(validation_records), len(validation_icl_outputs))
    for record, icl_output in zip(validation_records[:n_total], validation_icl_outputs[:n_total]):
        s = prepare_sample(
            tokenizer, demos, record, icl_output,
            demo_template=demo_template, query_template=query_template,
            attention_sink=attention_sink, device=device,
        )
        if s is not None:
            samples.append(s)
            sample_records.append(record)
            sample_icl_outputs.append(icl_output)
    if not samples:
        raise RuntimeError("no usable validation samples")

    if settings.metric not in ("substitution", "reconstruction"):
        raise ValueError(f"unknown SearchSettings.metric={settings.metric!r}")

    trajectory: dict
    if trajectory_path.exists():
        trajectory = json.loads(trajectory_path.read_text())
        # Cross-metric guard: a trajectory written under one metric isn't reusable
        # under another (probe signatures and ceiling differ). Default to
        # 'substitution' for legacy trajectories that pre-date this field.
        saved_metric = (trajectory.get("settings") or {}).get("metric", "substitution")
        if saved_metric != settings.metric:
            raise RuntimeError(
                f"trajectory at {trajectory_path} was made with metric='{saved_metric}', "
                f"but current settings.metric='{settings.metric}'. "
                f"Delete trajectory.json and final_config.json to re-search under the new metric, "
                f"or pass settings=SearchSettings(metric='{saved_metric}') to resume."
            )
    else:
        trajectory = {
            "settings": {
                "recovery_threshold": settings.recovery_threshold,
                "recovery_decimals": settings.recovery_decimals,
                "batch_size": settings.batch_size,
                "n_validation": len(samples),
                "patch_target": "attention_output_oproj",
                "layer_search_mode": "per_position",
                "recovery_aggregation": "min",
                "metric": settings.metric,
            },
            "reference": {},
            "trajectory": [],
            "final_config": None,
        }

    def _save():
        atomic_write_json(trajectory_path, trajectory)

    # ---------- metric-dispatched evaluator ----------
    def _eval(T_active, c, o, *, active_stages_per_slot=None, extraction_cache=None):
        if settings.metric == "reconstruction":
            return _evaluate_config_recon(
                model, tokenizer, samples, sample_records, sample_icl_outputs, n_layers,
                T_active, c, o,
                demos=demos, demo_template=demo_template, query_template=query_template,
                attention_sink=attention_sink,
                active_stages_per_slot=active_stages_per_slot,
                extraction_cache=extraction_cache, batch_size=settings.batch_size,
            )
        # substitution (legacy)
        return _evaluate_config(
            model, tokenizer, samples, n_layers, T_active, c, o,
            active_stages_per_slot=active_stages_per_slot,
            extraction_cache=extraction_cache, batch_size=settings.batch_size,
        )

    # ---- Phase 0 ---------------------------------------------------------------------
    # Phase-0 ceil is metric-specific: substitution uses the full-ICL no-hooks NLL;
    # reconstruction uses the all-on recon NLL (the best the recon method can deliver,
    # making the 0.95 threshold meaningful relative to that ceiling). The signature
    # encodes the metric so trajectories can't accidentally reuse a wrong-metric ceil.
    sig_ceil = _config_signature(f"phase0_ceil_{settings.metric}", None, -1, -1)
    existing = _find_in_trajectory(trajectory, sig_ceil)
    if existing is not None:
        per_sample_ceil_nll = list(existing["per_sample_nll"])
        nll_ceil_mean = float(existing["nll_mean"])
    else:
        if settings.metric == "substitution":
            per_sample_ceil_nll = _batched_icl_nll_no_hooks(
                model, list(range(len(samples))), samples, batch_size=settings.batch_size,
            )
        else:
            # All-on recon NLL — uses every (layer, position) at full strength.
            per_sample_ceil_nll, _ = _evaluate_config_recon(
                model, tokenizer, samples, sample_records, sample_icl_outputs, n_layers,
                T_active=None, c_active=-1, o_active=-1,
                demos=demos, demo_template=demo_template, query_template=query_template,
                attention_sink=attention_sink,
                active_stages_per_slot=None, extraction_cache=None,
                batch_size=settings.batch_size,
            )
        nll_ceil_mean = float(np.mean(per_sample_ceil_nll))
        trajectory["trajectory"].append({
            "signature": sig_ceil, "phase": "phase0_ceil",
            "config": {"template": "FULL", "n_content": -1, "n_output": -1, "active_stages": "all"},
            "metric": settings.metric,
            "nll_mean": nll_ceil_mean, "recovery": 1.0, "decision": "reference",
            "per_sample_nll": per_sample_ceil_nll,
            "per_sample_recovery": [1.0] * len(per_sample_ceil_nll),
            "min_recovery": 1.0,
        })
        _save()

    sig_floor = _config_signature("phase0_floor", [], 0, 0)
    existing = _find_in_trajectory(trajectory, sig_floor)
    if existing is not None:
        per_sample_floor_nll = list(existing["per_sample_nll"])
        nll_floor_mean = float(existing["nll_mean"])
    else:
        per_sample_floor_nll = _batched_zsl_nll(
            model, tokenizer, list(range(len(samples))), samples,
            per_sample_per_stage_active=None, per_sample_per_stage_values=None,
            batch_size=settings.batch_size,
        )
        nll_floor_mean = float(np.mean(per_sample_floor_nll))
        trajectory["trajectory"].append({
            "signature": sig_floor, "phase": "phase0_floor",
            "config": {"template": [], "n_content": 0, "n_output": 0, "active_stages": "all"},
            "nll_mean": nll_floor_mean, "recovery": 0.0, "decision": "reference",
            "per_sample_nll": per_sample_floor_nll,
            "per_sample_recovery": [0.0] * len(per_sample_floor_nll),
            "min_recovery": 0.0,
        })
        _save()

    trajectory["reference"] = {
        "nll_ceil_mean": nll_ceil_mean,
        "nll_floor_mean": nll_floor_mean,
        "per_sample_ceil_nll": per_sample_ceil_nll,
        "per_sample_floor_nll": per_sample_floor_nll,
    }
    _save()

    accept = lambda min_rec: round(min_rec, settings.recovery_decimals) >= settings.recovery_threshold

    def probe(phase, T_active, c, o):
        sig = _config_signature(phase, T_active, c, o)
        existing = _find_in_trajectory(trajectory, sig)
        if existing is not None:
            return existing
        nlls, _extr = _eval(T_active, c, o, active_stages_per_slot=None, extraction_cache=None)
        ps_recs = _recovery_per_sample(nlls, per_sample_floor_nll, per_sample_ceil_nll)
        min_rec = _aggregate_min(ps_recs)
        mean_rec = _recovery_from_mean_nll(float(np.mean(nlls)), nll_floor_mean, nll_ceil_mean)
        decision = "accept" if accept(min_rec) else "reject"
        entry = {
            "signature": sig, "phase": phase,
            "config": {
                "template": "FULL" if T_active is None else list(int(i) for i in T_active),
                "n_content": int(c), "n_output": int(o),
                "active_stages": "all",
            },
            "nll_mean": float(np.mean(nlls)),
            "recovery": min_rec,
            "mean_recovery": mean_rec,
            "per_sample_recovery": ps_recs,
            "min_recovery": min_rec,
            "decision": decision,
        }
        trajectory["trajectory"].append(entry)
        _save()
        return entry

    # ---- Phase 1 ---------------------------------------------------------------------
    T_active = None
    c_active = -1
    o_active = -1
    n_template_full = len(samples[0]["template_positions"])

    e = probe("phase1_drop_content", T_active, 0, o_active)
    if e["decision"] == "accept":
        c_active = 0

    e = probe("phase1_drop_template", [], c_active, o_active)
    if e["decision"] == "accept":
        T_active = []

    if o_active == -1:
        for k in settings.output_search_values:
            e = probe(f"phase1_output_descend_{k}", T_active, c_active, k)
            if e["decision"] == "accept":
                o_active = k
            else:
                break

    if c_active == -1:
        for k in settings.content_search_values:
            e = probe(f"phase1_content_descend_{k}", T_active, k, o_active)
            if e["decision"] == "accept":
                c_active = k
            else:
                break

    # ---- Phase 2 ---------------------------------------------------------------------
    if T_active is None:
        T_active = list(range(n_template_full))
    if T_active:
        kept = list(T_active)
        for pos in list(T_active):
            trial = [p for p in kept if p != pos]
            e = probe(f"phase2_drop_template_pos_{pos}", trial, c_active, o_active)
            if e["decision"] == "accept":
                kept = trial
        T_active = kept

    # ---- Phase 3 ---------------------------------------------------------------------
    extraction_cache = []
    if settings.metric == "reconstruction":
        cfg_for_cache = _build_to_config_for_recon(
            T_active, c_active, o_active,
            demo_template=demo_template, query_template=query_template, attention_sink=attention_sink,
        )
        for i, s in enumerate(samples):
            try:
                # keep_on_device=False — see _evaluate_config_recon for rationale
                # (caps GPU memory; required for reasoning tasks with long L).
                extr = _extract_one_sample(
                    model, tokenizer, demos, sample_records[i], sample_icl_outputs[i],
                    cfg_for_cache, keep_on_device=False,
                )
            except Exception:
                extr = None
            extraction_cache.append(extr)
    else:
        for s in samples:
            extraction_cache.append(_extract_icl_for_sample(model, s, T_active, c_active, o_active, n_layers))

    slots = _build_slot_list(T_active, c_active, o_active)
    active_stages_per_slot = {sid: set(range(n_layers)) for sid in slots}

    def probe_layer(phase, asps):
        sig = _config_signature(phase, T_active, c_active, o_active, asps)
        existing = _find_in_trajectory(trajectory, sig)
        if existing is not None:
            return existing
        nlls, _ = _eval(
            T_active, c_active, o_active,
            active_stages_per_slot=asps, extraction_cache=extraction_cache,
        )
        ps_recs = _recovery_per_sample(nlls, per_sample_floor_nll, per_sample_ceil_nll)
        min_rec = _aggregate_min(ps_recs)
        mean_rec = _recovery_from_mean_nll(float(np.mean(nlls)), nll_floor_mean, nll_ceil_mean)
        decision = "accept" if accept(min_rec) else "reject"
        entry = {
            "signature": sig, "phase": phase,
            "config": {
                "template": list(int(i) for i in T_active) if T_active is not None else "FULL",
                "n_content": int(c_active), "n_output": int(o_active),
                "active_stages_per_slot": {sid: sorted(stgs) for sid, stgs in asps.items()},
            },
            "nll_mean": float(np.mean(nlls)),
            "recovery": min_rec,
            "mean_recovery": mean_rec,
            "per_sample_recovery": ps_recs,
            "min_recovery": min_rec,
            "decision": decision,
        }
        trajectory["trajectory"].append(entry)
        _save()
        return entry

    slots_by_pt = {"template": [], "content": [], "output": []}
    for sid in slots:
        pt, _ = sid.split(":", 1)
        slots_by_pt[pt].append(sid)

    for pt in settings.layer_search_pt_order:
        if not slots_by_pt[pt]:
            continue
        for sid in slots_by_pt[pt]:
            for stage in sorted(active_stages_per_slot[sid]):
                trial = {sid2: set(stgs) for sid2, stgs in active_stages_per_slot.items()}
                trial[sid].discard(stage)
                e = probe_layer(f"phase3_drop_{sid}_stage_{stage}", trial)
                if e["decision"] == "accept":
                    active_stages_per_slot[sid].discard(stage)

    final_config = {
        "template_relative": list(int(i) for i in T_active) if T_active is not None else "FULL",
        "n_content": int(c_active),
        "n_output": int(o_active),
        "slots": slots,
        "active_stages_per_slot": {sid: sorted(stgs) for sid, stgs in active_stages_per_slot.items()},
        "n_stages_total": int(n_layers),
        "patch_target": "attention_output_oproj",
        "layer_search_mode": "per_position",
        "recovery_aggregation": "min",
        "metric": settings.metric,
    }
    trajectory["final_config"] = final_config
    _save()
    atomic_write_json(final_config_path, final_config)
    return {"trajectory": trajectory, "final_config": final_config}
