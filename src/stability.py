"""Stability analyses for extracted task-operator parameters.

Three claims underpin the broadcast-mode operator design and are flagged
in appendix §A.2:

  1. Cross-sample stability  — at each (layer, head, slot j), w_{j,i} and
     b_{j,i} vary little across the m=32 ICL prompts.
  2. Cross-token stability   — within a prompt, w_t varies little across
     token offsets t inside a category.
  3. Cross-demonstration-set stability — operators extracted from
     disjoint demo + validation pools agree per slot.

All three helpers operate on the per-sample `raw` list returned by
`extract_knowledge(..., return_per_sample=True)` (or, for §3, two
already-aggregated mean_circuits).

The per-sample dict layout (from `_extract_one_sample`):
    raw[i]["layers"][L]["template" | "content" | "gen"] == (w, ch[, ...])
        w  shape: [n_pos_in_category_i, n_heads]
        ch shape: [n_pos_in_category_i, n_heads, head_dim]
    raw[i]["active_template_ranks"]  — list of template ranks present
    raw[i]["n_content"], raw[i]["n_gen"]
"""

from __future__ import annotations

import torch

from .task_operator import TaskOperatorConfig


def _slot_id(position_type: str, rank: int | None, broadcast: bool) -> str:
    if broadcast:
        return f"{position_type}:block"
    return f"{position_type}:{int(rank)}"


def _site_stats(
    w_per_sample: list[torch.Tensor],
    b_per_sample: list[torch.Tensor],
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For one slot, return (cv_w[n_heads], mean_pairwise_cos_b[n_heads]).

    `w_per_sample[i]` is shape [n_heads]; `b_per_sample[i]` is shape
    [n_heads, head_dim]. `m = len(w_per_sample)`. CV is defined as
    std/|mean| with the unbiased=False std (population std) so it matches
    the appendix definition. Cosine sim is the mean of off-diagonal pairs
    in the m×m similarity matrix per head.
    """

    W = torch.stack([w.float() for w in w_per_sample], dim=0)  # [m, n_heads]
    B = torch.stack([b.float() for b in b_per_sample], dim=0)  # [m, n_heads, head_dim]
    m = W.shape[0]

    mean_w = W.mean(dim=0)
    std_w = W.std(dim=0, unbiased=False)
    cv_w = std_w / (mean_w.abs() + eps)

    if m < 2:
        cos_b = torch.full_like(mean_w, float("nan"))
        return cv_w, cos_b

    Bn = B / (B.norm(dim=-1, keepdim=True) + eps)
    sim = torch.einsum("mhd,nhd->mnh", Bn, Bn)              # [m, m, n_heads]
    eye = torch.eye(m, device=sim.device, dtype=sim.dtype).unsqueeze(-1)
    off_sum = (sim * (1.0 - eye)).sum(dim=(0, 1))            # [n_heads]
    cos_b = off_sum / (m * (m - 1))
    return cv_w, cos_b


def compute_cross_sample_stats(
    raw: list[dict],
    config: TaskOperatorConfig,
) -> dict:
    """Per-(layer, slot, head) CV(w) and mean pairwise cos-sim of b across
    the m samples in `raw`.

    The slot enumeration mirrors `_aggregate_circuits` exactly so the site
    list aligns 1:1 with what the aggregated mean_circuit would expose.
    """

    if not raw:
        raise ValueError("empty raw list")
    n_layers = int(raw[0]["n_layers"])

    common_template_ranks = sorted(
        set.intersection(*[set(c["active_template_ranks"]) for c in raw])
    )

    sites: list[dict] = []
    cv_rows: list[torch.Tensor] = []
    cos_rows: list[torch.Tensor] = []
    n_used: list[int] = []

    def _push(layer, slot_id, position_type, rank, broadcast, cv, cos, n):
        sites.append({
            "layer": int(layer),
            "slot_id": slot_id,
            "position_type": position_type,
            "rank": rank,
            "broadcast": bool(broadcast),
        })
        cv_rows.append(cv)
        cos_rows.append(cos)
        n_used.append(int(n))

    # ---- template (per-rank) ----
    for L in range(n_layers):
        for r in common_template_ranks:
            ws, bs = [], []
            for c in raw:
                ri = c["active_template_ranks"].index(r)
                t = c["layers"][L]["template"]
                if t is None:
                    continue
                ws.append(t[0][ri])
                bs.append(t[1][ri])
            if len(ws) < 2:
                continue
            cv, cos = _site_stats(ws, bs)
            _push(L, _slot_id("template", r, False), "template", int(r), False, cv, cos, len(ws))

    # ---- content / output ----
    for cat_key, cat_label, n_field, n_attr in (
        ("content", "content", "n_content", "n_content"),
        ("gen",     "output",  "n_output",  "n_gen"),
    ):
        n_setting = int(getattr(config, n_field))
        if n_setting == 0:
            continue
        if n_setting == -1:
            for L in range(n_layers):
                ws, bs = [], []
                for c in raw:
                    t = c["layers"][L][cat_key]
                    if t is None:
                        continue
                    ws.append(t[0].float().mean(dim=0))
                    bs.append(t[1].float().mean(dim=0))
                if len(ws) < 2:
                    continue
                cv, cos = _site_stats(ws, bs)
                _push(L, _slot_id(cat_label, None, True), cat_label, None, True, cv, cos, len(ws))
        else:
            k_max = n_setting
            for r in range(k_max):
                if not any(int(c[n_attr]) > r for c in raw):
                    continue
                for L in range(n_layers):
                    ws, bs = [], []
                    for c in raw:
                        if int(c[n_attr]) <= r:
                            continue
                        t = c["layers"][L][cat_key]
                        if t is None:
                            continue
                        ws.append(t[0][r])
                        bs.append(t[1][r])
                    if len(ws) < 2:
                        continue
                    cv, cos = _site_stats(ws, bs)
                    _push(L, _slot_id(cat_label, r, False), cat_label, int(r), False, cv, cos, len(ws))

    return {
        "site_index": sites,
        "cv_w":  torch.stack(cv_rows, dim=0)  if cv_rows  else torch.empty(0),
        "cos_b": torch.stack(cos_rows, dim=0) if cos_rows else torch.empty(0),
        "n_used": torch.tensor(n_used, dtype=torch.int32),
        "meta": {
            "n_layers": n_layers,
            "n_heads": int(raw[0]["n_heads"]),
            "head_dim": int(raw[0]["head_dim"]),
            "m_total": int(len(raw)),
            "n_template_ranks": list(common_template_ranks),
            "config": {
                "n_template": config.n_template,
                "n_content": config.n_content,
                "n_output": config.n_output,
            },
        },
    }


def compute_cross_token_within_prompt_cv(raw: list[dict]) -> dict:
    """For each prompt × (layer, head) × category in {template, content,
    generated}, compute the within-prompt CV of `w` across token offsets.

    Returns
    -------
    dict[category -> {
        cv_per_prompt: Tensor[m, n_layers, n_heads]   (NaN for prompts with <2 offsets)
        mean_cv:       Tensor[n_layers, n_heads]      (nan-mean over prompts)
        n_offsets_per_prompt: list[int]
    }]
    """

    if not raw:
        raise ValueError("empty raw list")
    n_layers = int(raw[0]["n_layers"])
    n_heads  = int(raw[0]["n_heads"])
    m        = len(raw)
    eps      = 1e-12

    cat_specs = [("template", "template"), ("content", "content"), ("gen", "generated")]
    out: dict = {}
    for cat_key, cat_label in cat_specs:
        cv = torch.full((m, n_layers, n_heads), float("nan"))
        n_t_per_prompt: list[int] = []
        for i, c in enumerate(raw):
            t_l0 = c["layers"][0][cat_key]
            n_t = int(t_l0[0].shape[0]) if t_l0 is not None else 0
            n_t_per_prompt.append(n_t)
            if n_t < 2:
                continue
            for L in range(n_layers):
                t = c["layers"][L][cat_key]
                if t is None:
                    continue
                W = t[0].float()                            # [n_t, n_heads]
                mean_w = W.mean(dim=0)
                std_w  = W.std(dim=0, unbiased=False)
                cv[i, L, :] = std_w / (mean_w.abs() + eps)
        # nan-aware mean across the m prompts
        mask = torch.isfinite(cv)
        denom = mask.sum(dim=0).clamp(min=1)
        mean_cv = torch.where(mask, cv, torch.zeros_like(cv)).sum(dim=0) / denom
        # mark layers/heads where no prompt contributed
        all_nan = ~mask.any(dim=0)
        mean_cv = torch.where(all_nan, torch.full_like(mean_cv, float("nan")), mean_cv)
        out[cat_label] = {
            "cv_per_prompt": cv,
            "mean_cv": mean_cv,
            "n_offsets_per_prompt": n_t_per_prompt,
        }
    return out


def compare_circuits(circuit_a: dict, circuit_b: dict) -> dict:
    """Compare two aggregated mean_circuits at the slot level.

    Returns
    -------
    {
      'pearson_w': float                      # one Pearson over all flat (layer, slot, head) w's
      'cos_b_per_site': Tensor[n_sites]       # one cos(b_a, b_b) per (layer, slot, head)
      'site_index':   list of {layer, slot_id, head, position_type, rank, broadcast}
      'summary': {pearson_w, cos_b_mean, cos_b_median, cos_b_p10}
    }

    Sites present in only one circuit are skipped silently. Position-rank
    indexing inside a slot's `w` / `context_heads` is collapsed when shapes
    differ — both circuits must have matching tensor shapes per matched
    slot (otherwise the slot is skipped).
    """

    a_recs = circuit_a["records_by_layer"]
    b_recs = circuit_b["records_by_layer"]

    flat_w_a, flat_w_b = [], []
    cos_per_site = []
    site_index: list[dict] = []
    skipped: list[str] = []

    for L in sorted(a_recs.keys()):
        a_by_slot = {_slot_id(r["position_type"], r.get("rank"), bool(r.get("broadcast"))): r for r in a_recs[L]}
        b_by_slot = {_slot_id(r["position_type"], r.get("rank"), bool(r.get("broadcast"))): r for r in b_recs.get(L, [])}
        for slot_id in sorted(set(a_by_slot) & set(b_by_slot)):
            ra, rb = a_by_slot[slot_id], b_by_slot[slot_id]
            wa = ra["w"].float().reshape(-1)
            wb = rb["w"].float().reshape(-1)
            if wa.shape != wb.shape:
                skipped.append(f"L{L}/{slot_id}: shape mismatch w {tuple(wa.shape)} vs {tuple(wb.shape)}")
                continue
            ba = ra["context_heads"].float()
            bb = rb["context_heads"].float()
            if ba.shape != bb.shape:
                skipped.append(f"L{L}/{slot_id}: shape mismatch ch {tuple(ba.shape)} vs {tuple(bb.shape)}")
                continue
            # Heads dimension is the last-but-one for ch; flatten the leading dims for w
            n_heads = ba.shape[-2]
            flat_w_a.append(wa)
            flat_w_b.append(wb)

            # Cosine over the head_dim axis
            cos_h = torch.nn.functional.cosine_similarity(
                ba.reshape(-1, ba.shape[-1]), bb.reshape(-1, bb.shape[-1]), dim=-1
            )                                                # [n_heads * leading]
            cos_per_site.append(cos_h)
            for h_idx in range(cos_h.shape[0]):
                site_index.append({
                    "layer": int(L),
                    "slot_id": slot_id,
                    "head": int(h_idx % n_heads),
                    "position_type": ra["position_type"],
                    "rank": ra.get("rank"),
                    "broadcast": bool(ra.get("broadcast")),
                })

    if not flat_w_a:
        raise RuntimeError("no overlapping slots between circuits")

    wa_flat = torch.cat(flat_w_a)
    wb_flat = torch.cat(flat_w_b)
    cos_flat = torch.cat(cos_per_site)

    if wa_flat.std(unbiased=False) < 1e-12 or wb_flat.std(unbiased=False) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(torch.corrcoef(torch.stack([wa_flat, wb_flat], dim=0))[0, 1])

    return {
        "pearson_w": pearson,
        "cos_b_per_site": cos_flat,
        "site_index": site_index,
        "w_a_flat": wa_flat,
        "w_b_flat": wb_flat,
        "summary": {
            "pearson_w": pearson,
            "cos_b_mean": float(cos_flat.mean()),
            "cos_b_median": float(cos_flat.median()),
            "cos_b_p10": float(torch.quantile(cos_flat, 0.10)),
        },
        "skipped": skipped,
    }
