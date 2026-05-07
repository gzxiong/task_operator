"""L1 reconstruction-gap analysis for the standard task_operator method.

For each query-side row at attention layer `L`, the per-head L1
reconstruction gap is

    full_attn[h]    = sum_k    p_full[h,k] * v[k]                (per-head full output)
    context_heads[h] = sum_{k in C} p_full[h,k] * v[k]            (paper's b)
    log_full[h]     = logsumexp_k(s[h,k])
    log_ctx [h]     = logsumexp_{k in C}(s[h,k])
    w[h]            = 1 - exp(log_ctx - log_full)                 in [0, 1]
    gap[h]          = full_attn[h] * (w[h] - 1) + context_heads[h]
    gap_norm[h]     = || gap[h] ||_1                              (sum over head_dim)

Heads with small `gap_norm` are well approximated by
`w * full_attn + context_heads`, so the affine replay reproduces their
attention output. Heads with large `gap_norm` cannot be reconstructed
from the context-only summary at that position, so they carry signal
from outside the context (in particular, the query side).

The module mirrors `task_operator._extract_one_sample` structurally —
same prompt, same masking, same per-(layer, slot) tensor schema — but
piggy-backs on `_extract_replay_factors(..., compute_refit=True)` so we
get `x_full` for free and compute `gap = x_full * (w - 1) + ch` without
re-running the forward.

Public API:

* `extract_gap_stats(model, tokenizer, demos, validation_records,
  validation_icl_outputs, config: TaskOperatorConfig) -> gap_stats`
* `filter_mean_circuit_by_gap_topp(mean_circuit, gap_stats, *,
  fraction_kept, selection='highest') -> mean_circuit_filtered`
* `flatten_gap_sites(gap_stats) -> list[dict]`

Kill semantics: `apply_task_operator`'s replay arithmetic is

    y[token] = (x[token] * scale) @ W_o + ch_proj
    scale    = w.repeat_interleave(head_dim)
    ch_proj  = ch.reshape(1, -1) @ W_o^T

Setting per-head `w_h = 1.0` and `ch_h = 0.0` makes that head's
contribution `(x_h * 1.0) @ W_o^h^T + 0`, which is exactly the original
o_proj output for head h. The other heads still receive their (w, ch)
replay. This matches `src_v3.circuit.sparsify_circuit`'s convention
(`new_w[inactive] = 1.0; new_context[inactive] = 0.0`).
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch

from .task_operator import (
    TaskOperatorConfig,
    _extract_replay_factors,
    _per_layer_value_states,
    _select_first_k,
    _select_template,
    _slot_id_for_record,
)
from .attention import patched_attention
from .model import get_decoder_layers, get_text_config
from .prompting import build_full_icl_prompt, build_zsl_prompt
from .spans import identify_context_query, resolve_query_positions
from .tokenization import encode_prompt_ids


# -------------------------------------------------------- per-sample gap extraction
def _extract_one_sample_gap(
    model, tokenizer, demos, record, icl_output_text, config: TaskOperatorConfig,
) -> dict | None:
    """Run the same extended-ICL forward as `task_operator._extract_one_sample`,
    but record per-(layer, position-slot, head) `gap_norm` from
    `_extract_replay_factors(..., compute_refit=True)`.

    Returns None if the sample can't be processed (empty target, span mismatch).
    Returned shape:

        {
          "layers": [
            {"template": Tensor[n_template_pos, n_heads] | None,
             "content":  Tensor[n_content_pos,  n_heads] | None,
             "gen":      Tensor[n_gen_pos,      n_heads] | None}
            ... per layer
          ],
          "n_template", "n_content", "n_gen", "active_template_ranks",
          "n_layers", "n_heads", "head_dim",
        }
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
        scores = attn_trace[layer_idx]["scores_post"].to(device=device, dtype=model_dtype)
        probs = attn_trace[layer_idx]["attn_probs"].to(device=device, dtype=model_dtype)

        def _gap_at(positions: list[int]):
            if not positions:
                return None
            s = scores[:, positions, :].permute(1, 0, 2).contiguous()
            p = probs[:, positions, :].permute(1, 0, 2).contiguous()
            # compute_refit=True returns (w, ch, x_full, x_zsl)
            w, ch, x_full, _x_zsl = _extract_replay_factors(
                s, p, v_states, context_span, compute_refit=True,
            )
            # gap = x_full * (w - 1) + ch  →  L1 over head_dim → [n_pos, n_heads]
            gap = x_full * (w.unsqueeze(-1) - 1.0) + ch
            gap_norm = gap.abs().sum(dim=-1)
            return gap_norm.detach().cpu()

        layer_records.append({
            "template": _gap_at(active_template),
            "content": _gap_at(active_content),
            "gen": _gap_at(active_output),
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


# ------------------------------------------------------------- aggregation
def _stack_mean(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(tensors), dim=0).mean(dim=0)


def _aggregate_gap_stats(raw: list[dict], config: TaskOperatorConfig) -> dict:
    """Average per-(layer, slot, head) gap_norm across validation samples.

    Mirrors `task_operator._aggregate_circuits`'s slot-emission rules so the
    record list is order-aligned one-to-one with a `mean_circuit` extracted
    under the same `TaskOperatorConfig`.
    """

    if not raw:
        raise ValueError("no validation samples were processed")
    n_layers = raw[0]["n_layers"]
    n_heads = raw[0]["n_heads"]
    head_dim = raw[0]["head_dim"]

    common_template_ranks = sorted(set.intersection(*[set(c["active_template_ranks"]) for c in raw]))

    records_by_layer: dict[int, list[dict]] = {l: [] for l in range(n_layers)}

    def _make_record(position_type, rank, broadcast, gap_stack):
        return {
            "position_type": position_type,
            "rank": rank,
            "broadcast": broadcast,
            "gap_norm": _stack_mean(gap_stack),
        }

    # ---- template (per-rank) ----
    if common_template_ranks:
        for layer_idx in range(n_layers):
            for rank in common_template_ranks:
                stack = []
                for c in raw:
                    rank_idx = c["active_template_ranks"].index(rank)
                    t = c["layers"][layer_idx]["template"]
                    stack.append(t[rank_idx])
                records_by_layer[layer_idx].append(
                    _make_record("template", int(rank), False, stack)
                )

    # ---- content ----
    if config.n_content == 0:
        n_content_recs = 0
    elif config.n_content == -1:
        for layer_idx in range(n_layers):
            stack = []
            for c in raw:
                t = c["layers"][layer_idx]["content"]
                if t is None:
                    continue
                stack.append(t.mean(dim=0))
            if stack:
                records_by_layer[layer_idx].append(
                    _make_record("content", None, True, stack)
                )
        n_content_recs = 1
    else:
        k_max = int(config.n_content)
        produced_ranks = [
            r for r in range(k_max) if any(int(c["n_content"]) > r for c in raw)
        ]
        for layer_idx in range(n_layers):
            for rank in produced_ranks:
                stack = []
                for c in raw:
                    if rank >= int(c["n_content"]):
                        continue
                    t = c["layers"][layer_idx]["content"]
                    if t is None:
                        continue
                    stack.append(t[rank])
                if stack:
                    records_by_layer[layer_idx].append(
                        _make_record("content", int(rank), False, stack)
                    )
        n_content_recs = len(produced_ranks)

    # ---- output ----
    if config.n_output == 0:
        n_output_recs = 0
    elif config.n_output == -1:
        for layer_idx in range(n_layers):
            stack = []
            for c in raw:
                t = c["layers"][layer_idx]["gen"]
                if t is None:
                    continue
                stack.append(t.mean(dim=0))
            if stack:
                records_by_layer[layer_idx].append(
                    _make_record("output", None, True, stack)
                )
        n_output_recs = 1
    else:
        k_max = int(config.n_output)
        produced_ranks_out = [
            r for r in range(k_max) if any(int(c["n_gen"]) > r for c in raw)
        ]
        for layer_idx in range(n_layers):
            for rank in produced_ranks_out:
                stack = []
                for c in raw:
                    if rank >= int(c["n_gen"]):
                        continue
                    t = c["layers"][layer_idx]["gen"]
                    if t is None:
                        continue
                    stack.append(t[rank])
                if stack:
                    records_by_layer[layer_idx].append(
                        _make_record("output", int(rank), False, stack)
                    )
        n_output_recs = len(produced_ranks_out)

    return {
        "records_by_layer": records_by_layer,
        "meta": {
            "method": "task-operator-gap",
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


# ------------------------------------------------------------ public extraction entry
def extract_gap_stats(
    model,
    tokenizer,
    demos,
    validation_records,
    validation_icl_outputs,
    config: TaskOperatorConfig,
) -> dict:
    """Extract per-(layer, slot, head) `gap_norm` averaged across validation
    samples for the standard task_operator method.

    Structurally mirrors `task_operator.extract_knowledge` but produces a
    `gap_stats` dict whose `records_by_layer[L]` entries carry `gap_norm`
    instead of `(w, context_heads)`. Records align one-to-one with a
    `mean_circuit` extracted under the same `TaskOperatorConfig`.
    """

    raw, skipped = [], []
    for record, icl_output in zip(validation_records, validation_icl_outputs):
        try:
            sample = _extract_one_sample_gap(
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
    state = _aggregate_gap_stats(raw, config)
    state["meta"]["n_skipped"] = len(skipped)
    state["meta"]["skipped"] = skipped
    return state


# ------------------------------------------------ alignment + flattening + filtering
def _align_records(
    mean_circuit: dict, gap_stats: dict
) -> dict[int, list[tuple[dict, torch.Tensor]]]:
    """Pair each `gap_stats` record with its matching `mean_circuit` record
    by (position_type, rank, broadcast). Mismatches raise: callers must
    extract gap_stats under the same `TaskOperatorConfig` as the
    mean_circuit, otherwise the slot grammar can disagree.
    """

    out: dict[int, list[tuple[dict, torch.Tensor]]] = {}
    for layer_idx in mean_circuit["records_by_layer"]:
        mean_recs = mean_circuit["records_by_layer"][int(layer_idx)]
        gap_recs = gap_stats["records_by_layer"].get(int(layer_idx), [])
        if len(mean_recs) != len(gap_recs):
            raise ValueError(
                f"layer {layer_idx}: mean_circuit has {len(mean_recs)} records "
                f"but gap_stats has {len(gap_recs)}; extraction configs disagree."
            )
        pairs = []
        for mr, gr in zip(mean_recs, gap_recs):
            mr_key = (mr["position_type"], mr["rank"], bool(mr.get("broadcast", False)))
            gr_key = (gr["position_type"], gr["rank"], bool(gr.get("broadcast", False)))
            if mr_key != gr_key:
                raise ValueError(
                    f"layer {layer_idx}: slot mismatch mean={mr_key} gap={gr_key}"
                )
            pairs.append((mr, gr["gap_norm"]))
        out[int(layer_idx)] = pairs
    return out


def flatten_gap_sites(gap_stats: dict) -> list[dict]:
    """Return a flat list of `{layer, slot, head, gap}` rows, sorted by
    descending `gap`. Used for top-N inspection / bar-chart visualisations.
    """

    rows: list[dict] = []
    for layer_idx, recs in gap_stats["records_by_layer"].items():
        for r in recs:
            slot_id = (
                f"{r['position_type']}:block"
                if r.get("broadcast")
                else f"{r['position_type']}:{int(r['rank'])}"
            )
            g = r["gap_norm"].to(dtype=torch.float32).cpu().numpy()
            for h, val in enumerate(g):
                rows.append({
                    "layer": int(layer_idx),
                    "slot": slot_id,
                    "head": int(h),
                    "gap": float(val),
                })
    rows.sort(key=lambda r: r["gap"], reverse=True)
    return rows


def filter_mean_circuit_by_gap_topp(
    mean_circuit: dict,
    gap_stats: dict,
    *,
    fraction_kept: float,
    selection: str = "highest",
) -> dict:
    """Return a copy of `mean_circuit` with all but the top-p fraction of
    `(layer, slot, head)` sites overridden so the standard task_operator
    replay hook is the identity at those heads.

    Parameters
    ----------
    fraction_kept : float in (0, 1]
        Fraction of sites to retain.
    selection : 'highest' | 'lowest'
        'highest' keeps the heads with the LARGEST `gap_norm` (heads whose
        full-attention output cannot be reconstructed from the context-only
        summary at that position; arguably the most query-side or
        out-of-context content); 'lowest' keeps the smallest. The default
        'highest' matches `src_v3.circuit.sparsify_circuit`'s ranking
        order (max-by-score, descending).

    Killed sites are set to `(w_h = 1.0, ch_h = 0.0)`. Under
    `task_operator.apply_task_operator`'s replay arithmetic
    `y[token] = (x[token] * w_broadcast) @ W_o + ch.reshape(1, -1) @ W_o.T`
    this yields `y[head_h] = x[head_h] @ W_o^h.T + 0`, i.e. the original
    o_proj contribution from head h. Other heads keep their `(w, ch)`
    replay. Matches v3 `sparsify_circuit`'s `(new_w[inactive] = 1.0,
    new_context[inactive] = 0.0)` convention.
    """

    if not (0.0 < float(fraction_kept) <= 1.0):
        raise ValueError(f"fraction_kept must be in (0, 1]; got {fraction_kept}")
    if selection not in ("highest", "lowest"):
        raise ValueError(f"selection must be 'highest' or 'lowest'; got {selection}")

    aligned = _align_records(mean_circuit, gap_stats)

    # Collect every (layer, slot_idx, head, gap) site for global ranking.
    all_sites: list[tuple[int, int, int, float]] = []
    for layer_idx, pairs in aligned.items():
        for slot_idx, (_, gap_norm) in enumerate(pairs):
            n_h = int(gap_norm.shape[0])
            for h in range(n_h):
                all_sites.append((layer_idx, slot_idx, h, float(gap_norm[h].item())))

    n_total = len(all_sites)
    n_keep = int(round(float(fraction_kept) * n_total))
    n_keep = max(1, min(n_keep, n_total))
    reverse = selection == "highest"
    all_sites.sort(key=lambda s: s[3], reverse=reverse)
    kept_set = {(L, s, h) for (L, s, h, _) in all_sites[:n_keep]}

    # Per-position-type sparsity stats (mirrors v3's `sparsity_stats` shape).
    per_position_type: dict[str, dict[str, int]] = {}

    new_records: dict[int, list[dict]] = {}
    for layer_idx, recs in mean_circuit["records_by_layer"].items():
        new_recs = []
        for slot_idx, rec in enumerate(recs):
            new_rec = dict(rec)  # shallow copy: we'll override w/context_heads
            new_w = rec["w"].clone()
            new_ch = rec["context_heads"].clone()
            n_h = int(new_w.shape[0])
            position_type = str(rec.get("position_type", "template"))
            stats = per_position_type.setdefault(
                position_type, {"total": 0, "active": 0, "masked": 0}
            )
            for h in range(n_h):
                stats["total"] += 1
                if (int(layer_idx), slot_idx, h) in kept_set:
                    stats["active"] += 1
                else:
                    stats["masked"] += 1
                    new_w[h] = 1.0
                    new_ch[h, :] = 0.0
            new_rec["w"] = new_w.to(dtype=rec["w"].dtype)
            new_rec["context_heads"] = new_ch.to(dtype=rec["context_heads"].dtype)
            # Refit fields, when present (compute_refit=True at extraction):
            # zero them out at killed sites too, so any later ablation_mode
            # that consults w_refit / ch_refit also sees identity at killed
            # heads.
            if "w_refit" in rec and "ch_refit" in rec:
                new_w_refit = rec["w_refit"].clone()
                new_ch_refit = rec["ch_refit"].clone()
                for h in range(n_h):
                    if (int(layer_idx), slot_idx, h) not in kept_set:
                        new_w_refit[h] = 1.0
                        new_ch_refit[h, :] = 0.0
                new_rec["w_refit"] = new_w_refit.to(dtype=rec["w_refit"].dtype)
                new_rec["ch_refit"] = new_ch_refit.to(dtype=rec["ch_refit"].dtype)
            new_recs.append(new_rec)
        new_records[int(layer_idx)] = new_recs

    # Carry forward + extend mean_circuit['meta']. Track sparsity stats in a
    # dedicated 'notes' field so callers can inspect what was killed without
    # touching the existing 'meta' surface.
    new_meta = copy.deepcopy(mean_circuit.get("meta", {}))
    notes = dict(mean_circuit.get("notes", {}))
    notes["sparsity_stats"] = {
        "fraction_kept": float(fraction_kept),
        "selection": selection,
        "total_head_slots": int(n_total),
        "active_head_slots": int(n_keep),
        "masked_head_slots": int(n_total - n_keep),
        "sparsity_fraction": (n_total - n_keep) / max(n_total, 1),
        "per_position_type": per_position_type,
        "kill_w": 1.0,
        "kill_ch": 0.0,
    }
    return {
        "records_by_layer": new_records,
        "meta": new_meta,
        "notes": notes,
    }
