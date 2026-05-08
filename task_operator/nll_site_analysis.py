"""NLL-based per-site importance analysis for the standard task_operator.

For each validation sample `v` and each site `i = (layer, slot_id)` we
measure the apply-time-aligned reconstruction NLL on the teacher-forced
ICL target tokens under three conditions:

    nll_full[v]      — all sites active (full mean_circuit replay)
    nll_zsl[v]       — no sites active (zero-shot, no replay)
    nll_kill[v, i]   — every site active EXCEPT site i, which is patched
                       as identity (per-head w=1, ch=0) at every (token, head)
                       — equivalent to skipping the patch at that one site.

The three NLLs use the same reconstruction-hook arithmetic as
`search.py`'s `_batched_zsl_nll_recon` (matching `apply_task_operator`):

    y[token] = (x[token] * scale_w + ch.flat) @ W_o^T,  scale_w = w ⊗ 1_{head_dim}

so they are directly comparable.

Two ranking strategies are derived from `nll_kill`:

* **recovery score**:
      recovery[v, i] = (nll_zsl[v] - nll_kill[v, i]) / (nll_zsl[v] - nll_full[v])
  clamped to [0, 1] (semantically, killing a site cannot push nll_kill
  below nll_full or above nll_zsl; excursions are numerical noise),
  is the fraction of the ZSL→full-TO NLL drop that *remains* when site i is
  killed; large recovery ⇔ site i is unimportant. The complement
      mean_lost_complement[i] = 1 - mean_v(recovery[v, i])
  is per-site importance ('how much of the TO's gain is LOST when this
  site is killed'); large complement ⇔ kill-it-last.

* **RRF score**: per-sample, rank sites by `nll_kill[v, :]` descending
  (largest NLL = killing it hurt the most = most important), then fuse
  via reciprocal rank fusion `rrf[i] = sum_v 1 / (k + rank[v, i])`. RRF
  is sample-rank-based and is robust to per-sample NLL-scale variance,
  but its absolute scale is k-dependent and not interpretable as a
  recovered fraction.

Public API:

* `extract_nll_site_stats(model, tokenizer, demos, validation_records,
  validation_icl_outputs, mean_circuit, config: TaskOperatorConfig) -> dict`
* `compute_recovery_scores(nll_stats: dict) -> dict`
* `compute_rrf_scores(nll_stats: dict, *, k: int = 60) -> dict`
* `filter_mean_circuit_by_score_topp(mean_circuit, score, site_index, *,
  fraction_kept, selection='highest') -> mean_circuit`
* `flatten_score_sites(score: Tensor[n_sites], site_index: list) -> list[dict]`

Site granularity is **(layer, slot)**, not (layer, slot, head): the
expected number of slots is small (≤ ~12 per layer) so layer × slot
already produces O(10^3) sites and a per-head ablation would inflate
this by another factor of n_heads (32–40). The kill semantics still
fall back to the per-head identity `(w_h=1, ch_h=0)` applied to every
head of the slot. This module pairs naturally with the standard
mean_circuit produced by `core.extract_knowledge`.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .model import get_decoder_layers, get_text_config
from .prompting import build_full_icl_prompt, build_zsl_prompt
from .nll_utils import (
    per_sample_nll_from_logits,
    prepare_sample,
    zsl_positions_for_record,
)
from .spans import identify_context_query, resolve_query_positions
from .core import (
    TaskOperatorConfig,
    _slot_id_for_record,
)
from .tokenization import encode_prompt_ids


# ============================================================================
# Site enumeration
# ============================================================================
def _enumerate_sites(mean_circuit: dict) -> list[dict]:
    """Flatten the mean_circuit into a deterministic list of (layer, slot) sites.

    Order: layers ascending, within a layer the slot order is the order
    `_aggregate_circuits` emits records (template ranks first, then content,
    then output). Each entry carries enough metadata to reconstruct which
    record it points to and to display in figures.
    """

    site_index: list[dict] = []
    rb = mean_circuit["records_by_layer"]
    for layer_idx in sorted(rb):
        for slot_idx, rec in enumerate(rb[int(layer_idx)]):
            sid = _slot_id_for_record(rec)
            site_index.append({
                "flat_idx": len(site_index),
                "layer": int(layer_idx),
                "slot_idx_within_layer": int(slot_idx),
                "slot_id": sid,
                "position_type": str(rec["position_type"]),
                "rank": (int(rec["rank"]) if rec.get("rank") is not None else None),
                "broadcast": bool(rec.get("broadcast", False)),
            })
    return site_index


def _site_lookup_by_layer_slot(site_index: list[dict]) -> dict[tuple[int, str], int]:
    """Map (layer, slot_id) -> flat site index for fast hook-time lookup."""

    return {(int(s["layer"]), str(s["slot_id"])): int(s["flat_idx"]) for s in site_index}


# ============================================================================
# Per-sample preparation
# ============================================================================
def _prepare_sample_for_nll(
    tokenizer, demos, record, icl_output_text, config: TaskOperatorConfig, device,
) -> dict | None:
    """Wrap `search.prepare_sample` so we use the exact same per-sample structure
    that the search-time recon NLL uses (icl_ids, zsl_ids, target_ids, ICL/ZSL
    template/content/icl_target positions, context/query span, sink offset).
    """

    return prepare_sample(
        tokenizer, demos, record, icl_output_text,
        demo_template=config.demo_template,
        query_template=config.query_template,
        attention_sink=config.attention_sink,
        device=device,
    )


# ============================================================================
# Batched per-site forward
# ============================================================================
def _make_oproj_batched_recon_hook(
    batch_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    w_stack: torch.Tensor,
    ch_stack: torch.Tensor,
):
    """Forward hook on `decoder_layers[L].self_attn.o_proj` that overrides
    output[batch_idx[i], pos_idx[i], :] with the apply-time recon arithmetic
    `(x * scale_w + ch.flat) @ W_o^T`. Same arithmetic as `apply_task_operator`
    at test time; here we apply it on a teacher-forced ZSL+target forward to
    measure per-site NLL recovery."""

    def hook(module, inputs, output):
        if output.shape[1] == 1:
            return output
        x = inputs[0]
        out = output.clone()
        if batch_idx.numel() == 0:
            return out
        head_dim = ch_stack.shape[-1]
        b = batch_idx.to(x.device)
        p = pos_idx.to(x.device)
        x_at = x[b, p, :]
        scale = w_stack.repeat_interleave(head_dim, dim=-1).to(x.device, x.dtype)
        ch_flat = ch_stack.reshape(b.shape[0], -1).to(x.device, x.dtype)
        recon = (x_at * scale + ch_flat) @ module.weight.T.to(x.device, x.dtype)
        out[b, p, :] = recon
        return out
    return hook


def _build_per_layer_recon_tensors(
    sample: dict,
    mean_circuit: dict,
    site_lookup: dict[tuple[int, str], int],
    *,
    n_layers: int,
    site_active_per_row: torch.Tensor,
    device,
) -> list[dict]:
    """For one validation sample replicated across `B` batch rows, build per-layer
    (batch_idx, pos_idx, w_stack, ch_stack) tensors that drive the o_proj hook.

    `site_active_per_row[b, site_flat_idx]` = 1 if batch row b should APPLY the
    site's replay, 0 if it should KILL it (skip the patch — equivalent to
    per-head w=1, ch=0). Row 0 = full-TO (all 1), row 1 = ZSL (all 0), rows
    2..1+n_kill = kill-one configurations.

    Per layer, we produce one flat list of (b, p, w_vec, ch_mat) entries
    flattened across (slot, batch_row, position) for sites where the row is
    active.
    """

    rb = mean_circuit["records_by_layer"]
    B = int(site_active_per_row.shape[0])
    out: list[dict] = []
    for layer_idx in range(n_layers):
        bs_b: list[int] = []
        bs_p: list[int] = []
        bs_w: list[torch.Tensor] = []
        bs_ch: list[torch.Tensor] = []
        for rec in rb[int(layer_idx)]:
            slot_id = _slot_id_for_record(rec)
            site_flat = site_lookup.get((int(layer_idx), str(slot_id)))
            if site_flat is None:
                continue
            zsl_ps = zsl_positions_for_record(sample, rec)
            if not zsl_ps:
                continue
            for b in range(B):
                if int(site_active_per_row[b, site_flat].item()) == 0:
                    continue
                for p in zsl_ps:
                    bs_b.append(int(b))
                    bs_p.append(int(p))
                    bs_w.append(rec["w"])
                    bs_ch.append(rec["context_heads"])
        if not bs_b:
            out.append({
                "batch_idx": torch.empty(0, dtype=torch.long, device=device),
                "pos_idx": torch.empty(0, dtype=torch.long, device=device),
                "w_stack": None,
                "ch_stack": None,
            })
            continue
        out.append({
            "batch_idx": torch.tensor(bs_b, dtype=torch.long, device=device),
            "pos_idx": torch.tensor(bs_p, dtype=torch.long, device=device),
            "w_stack": torch.stack(bs_w, dim=0),
            "ch_stack": torch.stack(bs_ch, dim=0),
        })
    return out


def _register_layer_hooks(model, per_layer_tensors):
    decoder_layers = get_decoder_layers(model)
    handles = []
    for layer_idx, tens in enumerate(per_layer_tensors):
        if tens["w_stack"] is None:
            continue
        h = decoder_layers[layer_idx].self_attn.o_proj.register_forward_hook(
            _make_oproj_batched_recon_hook(
                tens["batch_idx"], tens["pos_idx"], tens["w_stack"], tens["ch_stack"],
            )
        )
        handles.append(h)
    return handles


def _clear_handles(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def _process_one_sample(
    model, tokenizer, sample, mean_circuit, site_index, site_lookup,
    *, n_sites, n_layers, sites_per_microbatch,
) -> tuple[float, float, torch.Tensor]:
    """Return (nll_full, nll_zsl, nll_kill_per_site[n_sites]) for one validation
    sample. Two phases:

    * Phase A (once, B=2): row 0 = full TO replay (all sites active), row 1 =
      ZSL (no sites active). Records `nll_full` and `nll_zsl`.
    * Phase B (B=sites_per_microbatch, repeated ceil(n_sites/B) times):
      row j kills site `kill_chunk[j]` (all other sites active). Records
      `nll_kill_per_site[kill_chunk[j]]`.

    The controls do not depend on which kill is being ablated, so they are
    evaluated exactly once per sample. Re-running them per chunk would only
    add bf16/cuBLAS-kernel-shape noise without information.

    Memory note: per-chunk hook intermediates scale with B * slots_per_layer
    * avg_n_zsl_pos_per_slot * hidden in bf16. For B=16, slots=10, avg=4,
    hidden=4096 that is ~5 MB per layer, so the full forward stays bounded.
    """

    device = next(model.parameters()).device
    extended = torch.cat([sample["zsl_ids"], sample["target_ids"]], dim=1)
    L = int(extended.shape[1])
    target_padded = sample["target_ids"][0]
    n_target = int(sample["n_target"])
    zsl_prompt_len = int(sample["zsl_prompt_len"])

    nll_kill_per_site = torch.full(
        (int(n_sites),), float("nan"), dtype=torch.float32,
    )

    # ---- Phase A: controls (full TO + ZSL), B=2 ------------------------
    mask_ctrl = torch.ones((2, int(n_sites)), dtype=torch.uint8, device=device)
    mask_ctrl[1, :] = 0  # ZSL row: no patches
    input_ids_ctrl = extended.expand(2, -1).contiguous()
    attn_mask_ctrl = torch.ones((2, L), dtype=torch.long, device=device)

    per_layer_ctrl = _build_per_layer_recon_tensors(
        sample, mean_circuit, site_lookup,
        n_layers=int(n_layers),
        site_active_per_row=mask_ctrl,
        device=device,
    )
    handles = _register_layer_hooks(model, per_layer_ctrl)
    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids_ctrl, attention_mask=attn_mask_ctrl,
                use_cache=False, output_hidden_states=False, return_dict=True,
            )
        ctrl_nlls = per_sample_nll_from_logits(
            out.logits, [zsl_prompt_len] * 2,
            [target_padded] * 2, [n_target] * 2,
        )
    finally:
        _clear_handles(handles)
        del out
    nll_full_val = float(ctrl_nlls[0])
    nll_zsl_val = float(ctrl_nlls[1])

    # ---- Phase B: kill rows only, B=chunk_size per chunk ---------------
    site_chunks: list[list[int]] = []
    for start in range(0, int(n_sites), int(sites_per_microbatch)):
        site_chunks.append(list(range(start, min(start + int(sites_per_microbatch), int(n_sites)))))

    for kill_chunk in site_chunks:
        chunk_size = len(kill_chunk)
        if chunk_size == 0:
            continue
        mask_kill = torch.ones((chunk_size, int(n_sites)), dtype=torch.uint8, device=device)
        for j, site_flat in enumerate(kill_chunk):
            mask_kill[j, int(site_flat)] = 0
        input_ids_kill = extended.expand(chunk_size, -1).contiguous()
        attn_mask_kill = torch.ones((chunk_size, L), dtype=torch.long, device=device)

        per_layer_kill = _build_per_layer_recon_tensors(
            sample, mean_circuit, site_lookup,
            n_layers=int(n_layers),
            site_active_per_row=mask_kill,
            device=device,
        )
        handles = _register_layer_hooks(model, per_layer_kill)
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids_kill, attention_mask=attn_mask_kill,
                    use_cache=False, output_hidden_states=False, return_dict=True,
                )
            kill_nlls = per_sample_nll_from_logits(
                out.logits, [zsl_prompt_len] * chunk_size,
                [target_padded] * chunk_size, [n_target] * chunk_size,
            )
        finally:
            _clear_handles(handles)
            del out
        for j, site_flat in enumerate(kill_chunk):
            nll_kill_per_site[int(site_flat)] = float(kill_nlls[j])
    return nll_full_val, nll_zsl_val, nll_kill_per_site


# ============================================================================
# Public extraction entry
# ============================================================================
def extract_nll_site_stats(
    model,
    tokenizer,
    demos,
    validation_records,
    validation_icl_outputs,
    mean_circuit: dict,
    config: TaskOperatorConfig,
    *,
    sites_per_microbatch: int = 16,
    return_per_sample: bool = True,
) -> dict:
    """Compute per-validation-sample NLL with each (layer, slot) site killed.

    Per kept validation sample, two teacher-forced forwards drive the
    measurement (see `_process_one_sample` for details):

      * **Phase A** (B=2): row 0 = full TO replay, row 1 = ZSL (no replay).
        Records `nll_full[v]` and `nll_zsl[v]` once per sample.
      * **Phase B** (B=`sites_per_microbatch` per chunk, repeated until all
        sites are covered): each row kills one site (all other sites active).
        Records `nll_kill[v, site]` for every site.

    The two controls are evaluated only once per sample because they don't
    depend on which kill is being ablated.

    Returns
    -------
    dict with keys:
        nll_full     : Tensor[n_val]
        nll_zsl      : Tensor[n_val]
        nll_kill     : Tensor[n_val, n_sites]   (NaN for failed extractions)
        site_index   : list[dict]               — flat (layer, slot) descriptions
        meta         : dict — model/task metadata, n_skipped, etc.
        skipped      : list[dict] — diagnostics for samples that couldn't be processed.

    Sites are flattened in (layer, slot) order matching `_aggregate_circuits`'s
    record ordering. The mean_circuit must be the one this analysis is being
    run against — the slot enumeration depends on it.

    Parameters
    ----------
    sites_per_microbatch : int, default 16
        Phase B chunks sites along the batch dim into sub-forward passes;
        peak batch size in Phase B equals `sites_per_microbatch` (Phase A
        always uses B=2 for the controls). Reduce on long prompts or
        limited GPU memory; increase to amortise the per-forward overhead.
        The hook-side intermediates scale with B × slots_per_layer
        × avg_n_zsl_positions_per_slot × hidden_size; for typical lex/alg
        prompts (~50 tokens) the default 16 produces O(10^7) bfloat16 floats
        per layer which fits comfortably.
    return_per_sample : bool, default True
        Reserved for API symmetry; the per-sample tensors are always returned
        (the kill ablation is inherently per-sample).
    """

    del return_per_sample  # the per-sample tensors are always emitted

    if not validation_records:
        raise ValueError("validation_records is empty")
    if len(validation_records) != len(validation_icl_outputs):
        raise ValueError(
            f"validation_records ({len(validation_records)}) and "
            f"validation_icl_outputs ({len(validation_icl_outputs)}) length mismatch"
        )

    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)

    site_index = _enumerate_sites(mean_circuit)
    site_lookup = _site_lookup_by_layer_slot(site_index)
    n_sites = len(site_index)
    if n_sites == 0:
        raise RuntimeError("mean_circuit has no slots; nothing to ablate")

    device = next(model.parameters()).device

    nll_full_list: list[float] = []
    nll_zsl_list: list[float] = []
    nll_kill_list: list[torch.Tensor] = []
    skipped: list[dict] = []
    kept_record_indices: list[int] = []

    for v, (record, icl_output) in enumerate(zip(validation_records, validation_icl_outputs)):
        try:
            sample = _prepare_sample_for_nll(
                tokenizer, demos, record, icl_output, config, device,
            )
            reason = None if sample is not None else "empty_target_or_span_mismatch"
        except Exception as exc:
            sample = None
            reason = f"{type(exc).__name__}: {exc}"
        if sample is None:
            skipped.append({
                "example_id": record.get("example_id"),
                "validation_index": int(v),
                "reason": reason,
            })
            continue
        try:
            nll_full, nll_zsl, nll_kill = _process_one_sample(
                model, tokenizer, sample, mean_circuit, site_index, site_lookup,
                n_sites=n_sites, n_layers=n_layers,
                sites_per_microbatch=int(sites_per_microbatch),
            )
        except Exception as exc:
            skipped.append({
                "example_id": record.get("example_id"),
                "validation_index": int(v),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        nll_full_list.append(float(nll_full))
        nll_zsl_list.append(float(nll_zsl))
        nll_kill_list.append(nll_kill)
        kept_record_indices.append(int(v))

    if not nll_full_list:
        raise RuntimeError(
            f"all {len(skipped)} validation samples failed; first 3: {skipped[:3]}"
        )

    nll_full_t = torch.tensor(nll_full_list, dtype=torch.float32)
    nll_zsl_t = torch.tensor(nll_zsl_list, dtype=torch.float32)
    nll_kill_t = torch.stack(nll_kill_list, dim=0).to(dtype=torch.float32)

    return {
        "nll_full": nll_full_t,
        "nll_zsl": nll_zsl_t,
        "nll_kill": nll_kill_t,
        "site_index": site_index,
        "meta": {
            "method": "task-operator-nll-site",
            "n_layers": int(n_layers),
            "n_sites": int(n_sites),
            "n_val_kept": int(nll_full_t.shape[0]),
            "n_val_input": int(len(validation_records)),
            "n_skipped": int(len(skipped)),
            "validation_indices_kept": kept_record_indices,
            "sites_per_microbatch": int(sites_per_microbatch),
            "config": {
                "n_template": int(config.n_template),
                "n_content": int(config.n_content),
                "n_output": int(config.n_output),
                "template_active_ranks": (
                    list(config.template_active_ranks)
                    if config.template_active_ranks is not None else None
                ),
            },
        },
        "skipped": skipped,
    }


# ============================================================================
# Score derivations
# ============================================================================
def compute_recovery_scores(nll_stats: dict, *, eps: float = 1e-6) -> dict:
    """Compute per-site recovery and lost-complement scores from `nll_kill`.

    For each validation sample v:
        recovery[v, i] = (nll_zsl[v] - nll_kill[v, i]) / (nll_zsl[v] - nll_full[v])

    A valid sample requires `nll_zsl - nll_full > eps` (TO actually helped).
    Samples with non-positive denominator are dropped from the per-site mean
    and flagged via `valid_mask`.

    Returns
    -------
    {
      'recovery': Tensor[n_val, n_sites]      — fraction of TO-vs-ZSL gap RETAINED
                                                 by killing site i (so high ⇒ kill it).
      'lost_complement': Tensor[n_val, n_sites] — 1 - recovery (per-site IMPORTANCE
                                                                under that sample).
      'mean_recovery': Tensor[n_sites]         — sample-mean of recovery.
      'mean_lost_complement': Tensor[n_sites]  — 1 - mean_recovery; use this as the
                                                 ranking score: large ⇔ site is
                                                 important ⇔ kill-it-last.
      'valid_mask': Tensor[n_val] bool         — which samples contributed.
      'eps'                                    — denominator guard threshold.
    }

    Sign convention: large `mean_lost_complement[i]` ⇔ site i is IMPORTANT
    (its removal loses a lot of the TO's improvement over ZSL). When passing
    this score into `filter_mean_circuit_by_score_topp(..., selection='highest')`
    the kept top-p sites are the most important ones.
    """

    nll_full = nll_stats["nll_full"].to(dtype=torch.float32)
    nll_zsl = nll_stats["nll_zsl"].to(dtype=torch.float32)
    nll_kill = nll_stats["nll_kill"].to(dtype=torch.float32)

    span = nll_zsl - nll_full                           # [n_val]
    valid_mask = span > float(eps)                      # [n_val] bool

    # Compute per-sample recovery for ALL samples (NaN-safe), then aggregate.
    # Clamp recovery to [0, 1]: semantically, killing a site cannot take
    # NLL_kill below NLL_full (recovery > 1 would mean the kill HELPS) or
    # above NLL_zsl (recovery < 0 would mean the kill is WORSE than no
    # replay). Excursions outside [0, 1] are bf16/numerical artifacts
    # (clamped away here to keep the [0, 1] "fraction" interpretation
    # honest) plus rare adversarial samples (also clamped because they
    # would otherwise distort the per-site mean).
    span_clamped = span.clamp(min=float(eps))
    recovery = (nll_zsl.unsqueeze(1) - nll_kill) / span_clamped.unsqueeze(1)
    finite = torch.isfinite(recovery)
    recovery = torch.where(finite, recovery.clamp(0.0, 1.0), recovery)
    lost_complement = 1.0 - recovery

    if valid_mask.any():
        recovery_valid = recovery[valid_mask]
        # Drop NaN rows defensively (extraction failures land as NaN in nll_kill).
        finite_mask = torch.isfinite(recovery_valid)
        recovery_valid = torch.where(finite_mask, recovery_valid, torch.zeros_like(recovery_valid))
        denom = finite_mask.sum(dim=0).clamp(min=1)
        mean_recovery = recovery_valid.sum(dim=0) / denom
    else:
        mean_recovery = torch.full((int(nll_kill.shape[1]),), float("nan"))
    mean_lost_complement = 1.0 - mean_recovery

    return {
        "recovery": recovery,
        "lost_complement": lost_complement,
        "mean_recovery": mean_recovery,
        "mean_lost_complement": mean_lost_complement,
        "valid_mask": valid_mask,
        "eps": float(eps),
        "n_valid": int(valid_mask.sum().item()),
    }


def compute_rrf_scores(nll_stats: dict, *, k: int = 60) -> dict:
    """Reciprocal-rank-fusion of per-sample site-importance ranklists.

    For each validation sample, sites are ranked by `nll_kill[v, :]` descending
    (larger NLL = killing the site hurt the model more = more important). Ranks
    are dense (1 for the most important site). Fused score:

        rrf_score[i] = sum_v 1 / (k + rank[v, i])

    Larger `rrf_score[i]` ⇔ site i is important across many samples ⇔ kill-it-last.

    Returns
    -------
    {
      'rrf_score': Tensor[n_sites]              — fused score, large = important.
      'per_sample_rank': Tensor[n_val, n_sites] — dense ranks, 1 = most important.
      'k'                                       — RRF parameter.
    }

    Default `k = 60` matches the canonical RRF parameter (Cormack et al. 2009).
    """

    nll_kill = nll_stats["nll_kill"].to(dtype=torch.float32)
    n_val, n_sites = int(nll_kill.shape[0]), int(nll_kill.shape[1])

    # Replace any NaN with -inf so failed sites don't poison sort order; their
    # rank ends up at the bottom (least important) which contributes ~0 to RRF.
    nll_kill_clean = torch.where(
        torch.isfinite(nll_kill), nll_kill, torch.full_like(nll_kill, float("-inf")),
    )

    # Rank descending: argsort(-x) gives the index of the k-th largest at position k.
    # Convert to dense ranks: rank[v, i] = position of site i in the desc-sorted list (1-indexed).
    order = nll_kill_clean.argsort(dim=-1, descending=True)        # [n_val, n_sites]
    ranks = torch.empty_like(order)
    arange = torch.arange(1, n_sites + 1, device=order.device).unsqueeze(0).expand(n_val, n_sites)
    ranks.scatter_(dim=-1, index=order, src=arange)
    rrf_score = (1.0 / (float(k) + ranks.to(dtype=torch.float32))).sum(dim=0)

    return {
        "rrf_score": rrf_score,
        "per_sample_rank": ranks,
        "k": int(k),
        "n_val": int(n_val),
        "n_sites": int(n_sites),
    }


# ============================================================================
# ICL upper-bound NLL extraction (for compute_recovery_scores_icl)
# ============================================================================
def extract_nll_icl_per_sample(
    model,
    tokenizer,
    demos,
    validation_records,
    validation_icl_outputs,
    config: TaskOperatorConfig,
) -> dict:
    """Per-validation-sample teacher-forced NLL on the same target tokens as
    `extract_nll_site_stats`, but evaluated under the **full ICL prompt**
    (with K demos prepended) and **no task_operator replay**. This is the
    true upper-bound that `nll_full` (full-TO replay) only approximates.

    The function reuses `_prepare_sample_for_nll` (= `prepare_sample`) so
    the kept-sample order matches `extract_nll_site_stats` exactly when
    called with the same arguments — `nll_icl[v]` then aligns row-by-row
    with `nll_kill[v, :]` from a paired `nll_site_stats.pt`.

    Forward shape: one B=1 forward per sample, on `cat([icl_ids,
    target_ids])`. No hooks. Cost is dominated by the long ICL prompt
    length (typically 200–500 tokens for K=8 lex/alg) but is independent
    of the number of sites.

    Returns
    -------
    dict with keys:
        nll_icl     : Tensor[n_val_kept]
        meta        : dict — n_val_kept, n_val_input, n_skipped,
                       validation_indices_kept, K_demos.
        skipped     : list[dict] — diagnostics for samples that couldn't be
                       prepared.
    """

    if not validation_records:
        raise ValueError("validation_records is empty")
    if len(validation_records) != len(validation_icl_outputs):
        raise ValueError(
            f"validation_records ({len(validation_records)}) and "
            f"validation_icl_outputs ({len(validation_icl_outputs)}) length mismatch"
        )

    device = next(model.parameters()).device
    nll_icl_list: list[float] = []
    skipped: list[dict] = []
    kept_record_indices: list[int] = []

    for v, (record, icl_output) in enumerate(zip(validation_records, validation_icl_outputs)):
        try:
            sample = _prepare_sample_for_nll(
                tokenizer, demos, record, icl_output, config, device,
            )
            reason = None if sample is not None else "empty_target_or_span_mismatch"
        except Exception as exc:
            sample = None
            reason = f"{type(exc).__name__}: {exc}"
        if sample is None:
            skipped.append({
                "example_id": record.get("example_id"),
                "validation_index": int(v),
                "reason": reason,
            })
            continue
        try:
            extended = torch.cat([sample["icl_ids"], sample["target_ids"]], dim=1)
            attn_mask = torch.ones_like(extended, dtype=torch.long)
            with torch.inference_mode():
                out = model(
                    input_ids=extended, attention_mask=attn_mask,
                    use_cache=False, output_hidden_states=False, return_dict=True,
                )
            nll_one = per_sample_nll_from_logits(
                out.logits,
                [int(sample["icl_prompt_len"])],
                [sample["target_ids"][0]],
                [int(sample["n_target"])],
            )[0]
            del out
        except Exception as exc:
            skipped.append({
                "example_id": record.get("example_id"),
                "validation_index": int(v),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        nll_icl_list.append(float(nll_one))
        kept_record_indices.append(int(v))

    if not nll_icl_list:
        raise RuntimeError(
            f"all {len(skipped)} validation samples failed; first 3: {skipped[:3]}"
        )

    return {
        "nll_icl": torch.tensor(nll_icl_list, dtype=torch.float32),
        "meta": {
            "method": "full-icl-nll",
            "n_val_kept": len(nll_icl_list),
            "n_val_input": len(validation_records),
            "n_skipped": len(skipped),
            "validation_indices_kept": kept_record_indices,
            "K_demos": int(len(demos)),
        },
        "skipped": skipped,
    }


def compute_recovery_scores_icl(
    nll_stats: dict,
    nll_icl: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> dict:
    """Variant of `compute_recovery_scores` that uses the **ICL** NLL as the
    upper-bound denominator instead of the full-TO-replay NLL.

    For each validation sample v:
        recovery_icl[v, i] = (nll_zsl[v] - nll_kill[v, i]) / (nll_zsl[v] - nll_icl[v])

    `nll_icl` is the per-sample teacher-forced NLL on the same target
    tokens but evaluated under the full ICL prompt (no task_operator
    replay). It is the true "ceiling" that the operator is trying to
    achieve; substituting it for `nll_full` normalizes per-site
    importance against the actual ZSL→ICL gap.

    The resulting `mean_lost_complement` score is rank-compatible with
    the non-icl variant (large ⇔ important ⇔ kill-it-last) and is
    consumed identically by `filter_mean_circuit_by_score_topp(...,
    selection='highest')`.

    Numerical handling matches `compute_recovery_scores`:
      * recovery is clamped to [0, 1] (excursions are bf16 / numerical
        noise plus rare adversarial samples; both kinds of excursion
        would distort the per-site mean if left unclamped).
      * samples with `nll_zsl - nll_icl <= eps` are dropped from the
        per-site mean (ICL no better than ZSL on that sample, so the
        recovery fraction has no signal — the mean is taken over the
        valid mask).
    """

    nll_zsl = nll_stats["nll_zsl"].to(dtype=torch.float32)
    nll_kill = nll_stats["nll_kill"].to(dtype=torch.float32)
    nll_icl = nll_icl.to(dtype=torch.float32)

    if nll_icl.shape != nll_zsl.shape:
        raise ValueError(
            f"nll_icl shape {tuple(nll_icl.shape)} != nll_zsl shape "
            f"{tuple(nll_zsl.shape)}; check kept-sample alignment "
            "(should match meta['validation_indices_kept'] from both extractions)"
        )

    span = nll_zsl - nll_icl                            # [n_val]
    valid_mask = span > float(eps)                      # [n_val] bool

    span_clamped = span.clamp(min=float(eps))
    recovery = (nll_zsl.unsqueeze(1) - nll_kill) / span_clamped.unsqueeze(1)
    finite = torch.isfinite(recovery)
    recovery = torch.where(finite, recovery.clamp(0.0, 1.0), recovery)
    lost_complement = 1.0 - recovery

    if valid_mask.any():
        recovery_valid = recovery[valid_mask]
        finite_mask = torch.isfinite(recovery_valid)
        recovery_valid = torch.where(
            finite_mask, recovery_valid, torch.zeros_like(recovery_valid),
        )
        denom = finite_mask.sum(dim=0).clamp(min=1)
        mean_recovery = recovery_valid.sum(dim=0) / denom
    else:
        mean_recovery = torch.full((int(nll_kill.shape[1]),), float("nan"))
    mean_lost_complement = 1.0 - mean_recovery

    return {
        "recovery": recovery,
        "lost_complement": lost_complement,
        "mean_recovery": mean_recovery,
        "mean_lost_complement": mean_lost_complement,
        "valid_mask": valid_mask,
        "eps": float(eps),
        "n_valid": int(valid_mask.sum().item()),
    }


# ============================================================================
# Score-driven filtering
# ============================================================================
def flatten_score_sites(
    score: torch.Tensor, site_index: Sequence[dict],
) -> list[dict]:
    """Return a flat list `[{layer, slot_id, position_type, rank, broadcast,
    score, rank_in_score}]`, sorted by descending `score`. Used by figures.
    """

    if int(score.numel()) != len(site_index):
        raise ValueError(
            f"score length {int(score.numel())} != site_index length {len(site_index)}"
        )
    rows: list[dict] = []
    s = score.to(dtype=torch.float32).cpu().numpy()
    for site in site_index:
        rows.append({
            "layer": int(site["layer"]),
            "slot_id": str(site["slot_id"]),
            "position_type": str(site["position_type"]),
            "rank": (None if site["rank"] is None else int(site["rank"])),
            "broadcast": bool(site["broadcast"]),
            "flat_idx": int(site["flat_idx"]),
            "score": float(s[int(site["flat_idx"])]),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    for k, r in enumerate(rows):
        r["rank_in_score"] = int(k)
    return rows


def filter_mean_circuit_by_score_topp(
    mean_circuit: dict,
    score: torch.Tensor,
    site_index: Sequence[dict],
    *,
    fraction_kept: float,
    selection: str = "highest",
) -> dict:
    """Return a copy of `mean_circuit` with all but the top-p fraction of
    (layer, slot) sites overridden so the standard task_operator replay hook
    is the identity at those slots (i.e., w[head]=1.0, ch[head]=0.0 at every
    head of the killed slot).

    Parameters
    ----------
    score : Tensor[n_sites]
        Per-(layer, slot) importance score; matched to `site_index` by flat
        index. Larger score ⇒ more important (under both the
        `mean_lost_complement` and `rrf_score` conventions, kill-it-last).
    site_index : list[dict]
        Output of `_enumerate_sites(mean_circuit)`, returned by
        `extract_nll_site_stats`. Determines (layer, slot_idx_within_layer)
        for each flat site index.
    fraction_kept : float in (0, 1]
        Top fraction of sites to retain (by `selection`).
    selection : 'highest' | 'lowest', default 'highest'
        'highest' keeps the LARGEST-score sites (most important under our
        sign convention). 'lowest' is the dual ablation.

    Killed slots are overridden per-head with `w_h = 1.0`, `ch_h = 0.0`.
    Under `apply_task_operator`'s recon arithmetic
    `y[token] = (x[token] * (w ⊗ 1_{head_dim})) @ W_o^T + ch.flat @ W_o^T`,
    `(w_h=1, ch_h=0)` produces the original head's o_proj contribution
    `x[head_h] @ W_o^h.T`, exactly skipping the patch at that head. Other
    slots keep their (w, ch) replay; refit fields (when present) are also
    zeroed at killed slots so `_select_factors` stays consistent.
    """

    if not (0.0 < float(fraction_kept) <= 1.0):
        raise ValueError(f"fraction_kept must be in (0, 1]; got {fraction_kept}")
    if selection not in ("highest", "lowest"):
        raise ValueError(f"selection must be 'highest' or 'lowest'; got {selection}")
    if int(score.numel()) != len(site_index):
        raise ValueError(
            f"score length {int(score.numel())} != site_index length {len(site_index)}"
        )

    n_total = int(score.numel())
    n_keep = int(round(float(fraction_kept) * n_total))
    n_keep = max(1, min(n_keep, n_total))
    reverse = selection == "highest"
    s = score.to(dtype=torch.float32).cpu().numpy()
    ordering = sorted(range(n_total), key=lambda i: float(s[i]), reverse=reverse)
    kept_flat_idx = set(int(i) for i in ordering[:n_keep])

    kept_layer_slot: set[tuple[int, int]] = set()
    for site in site_index:
        if int(site["flat_idx"]) in kept_flat_idx:
            kept_layer_slot.add((int(site["layer"]), int(site["slot_idx_within_layer"])))

    per_position_type: dict[str, dict[str, int]] = {}
    new_records: dict[int, list[dict]] = {}
    for layer_idx, recs in mean_circuit["records_by_layer"].items():
        new_recs = []
        for slot_idx, rec in enumerate(recs):
            new_rec = dict(rec)
            position_type = str(rec.get("position_type", "template"))
            stats = per_position_type.setdefault(
                position_type, {"total": 0, "active": 0, "masked": 0},
            )
            stats["total"] += 1
            if (int(layer_idx), int(slot_idx)) in kept_layer_slot:
                stats["active"] += 1
            else:
                stats["masked"] += 1
                new_w = torch.ones_like(rec["w"])
                new_ch = torch.zeros_like(rec["context_heads"])
                new_rec["w"] = new_w.to(dtype=rec["w"].dtype)
                new_rec["context_heads"] = new_ch.to(dtype=rec["context_heads"].dtype)
                if "w_refit" in rec and "ch_refit" in rec:
                    new_rec["w_refit"] = torch.ones_like(rec["w_refit"]).to(dtype=rec["w_refit"].dtype)
                    new_rec["ch_refit"] = torch.zeros_like(rec["ch_refit"]).to(dtype=rec["ch_refit"].dtype)
            new_recs.append(new_rec)
        new_records[int(layer_idx)] = new_recs

    new_meta = copy.deepcopy(mean_circuit.get("meta", {}))
    notes = dict(mean_circuit.get("notes", {}))
    notes["sparsity_stats"] = {
        "fraction_kept": float(fraction_kept),
        "selection": selection,
        "total_sites": int(n_total),
        "active_sites": int(n_keep),
        "masked_sites": int(n_total - n_keep),
        "sparsity_fraction": (n_total - n_keep) / max(n_total, 1),
        "per_position_type": per_position_type,
        "kill_w": 1.0,
        "kill_ch": 0.0,
        "site_granularity": "layer_slot",
    }
    return {
        "records_by_layer": new_records,
        "meta": new_meta,
        "notes": notes,
    }
