"""NLL-recovery site importance maps (Section 3.3 / Figure 2).

For each (layer, slot) site, compute the next-token NLL on the teacher-forced
ICL output under three conditions:

    - full:    all task-operator factors active                 → NLL_full
    - zsl:     no factors (zero-shot)                           → NLL_zsl
    - kill_i:  every factor active EXCEPT site i (patched to 1) → NLL_kill[i]

Per-sample recovery is `(NLL_zsl - NLL_kill[i]) / (NLL_zsl - NLL_full)` and
per-sample importance is `1 - recovery`. Per-site importance is averaged via
reciprocal rank fusion (RRF) across samples to be scale-robust.

Requires `task_operator/<model>/<task>/allpos/knowledge.pt` from `run_to.py`
and `predictions/<model>/<task>/full_icl_K8_validation.json` from `run_zsl_icl.py`.

Outputs:
    task_operator/<model>/<task>/allpos_nll_site/nll_site_stats.pt
    figures/sparsity_<model>_<task>.png   (only with --plot)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from task_operator import (
    ALL_TASKS,
    N_DEMOS,
    N_VALIDATION,
    OUTPUTS_DIR,
    compute_recovery_scores,
    compute_rrf_scores,
    extract_nll_site_stats,
    load_hf_model,
    load_split,
    load_validation_icl_outputs,
    make_allpos_config,
    safe_model_name,
    task_operator_paths,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=ALL_TASKS)
    ap.add_argument("--m", type=int, default=N_VALIDATION)
    ap.add_argument("--sites-per-microbatch", type=int, default=16)
    ap.add_argument("--plot", action="store_true",
                    help="render a per-(layer, position) importance heatmap")
    args = ap.parse_args()

    knowledge_path, _ = task_operator_paths(args.model, args.task, "allpos")
    if not knowledge_path.exists():
        raise SystemExit(
            f"missing {knowledge_path} — run scripts/run_to.py for this (model, task) first"
        )
    blob = torch.load(knowledge_path, map_location="cpu", weights_only=False)
    mean_circuit = blob["mean_circuit"]

    val_outputs, kept_indices = load_validation_icl_outputs(args.model, args.task, args.m)
    if val_outputs is None:
        raise SystemExit("missing full_icl_K8_validation.json — run scripts/run_zsl_icl.py first")

    print(f"Loading {args.model} ...")
    model, tokenizer, info = load_hf_model(args.model)
    print(f"  family={info['family']} layers={info['num_hidden_layers']}")

    demos = load_split(args.task, "demos")[:N_DEMOS]
    validation_full = load_split(args.task, "validation")
    val_records = [validation_full[i] for i in kept_indices]
    config = make_allpos_config(args.task)

    print(f"\n[NLL site importance] {args.task} m_used={len(val_records)} "
          f"n_sites={sum(len(v) for v in mean_circuit['records_by_layer'].values())}")
    stats = extract_nll_site_stats(
        model, tokenizer, demos, val_records, val_outputs,
        mean_circuit, config,
        sites_per_microbatch=args.sites_per_microbatch,
    )
    rec = compute_recovery_scores(stats)
    rrf = compute_rrf_scores(stats)
    stats["recovery"] = rec
    stats["rrf"] = rrf

    out_dir = OUTPUTS_DIR / "task_operator" / safe_model_name(args.model) / args.task / "allpos_nll_site"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nll_site_stats.pt"
    torch.save(stats, out_path)
    print(f"  saved {out_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("  [warn] --plot requires matplotlib; skipping")
            return
        site_index = stats["site_index"]
        importance = 1.0 - rec["mean"]
        layers = sorted({s["layer"] for s in site_index})
        slot_ids = list(dict.fromkeys(s["slot_id"] for s in site_index))
        grid = np.full((len(layers), len(slot_ids)), np.nan, dtype=float)
        layer_to_row = {L: i for i, L in enumerate(layers)}
        slot_to_col = {sid: i for i, sid in enumerate(slot_ids)}
        for s, imp in zip(site_index, importance.tolist()):
            grid[layer_to_row[s["layer"]], slot_to_col[s["slot_id"]]] = float(imp)
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(slot_ids)), 0.18 * len(layers) + 1.5))
        im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(slot_ids)))
        ax.set_xticklabels(slot_ids, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_xlabel("slot")
        ax.set_ylabel("layer")
        ax.set_title(f"{args.model} × {args.task} — site importance (1 − recovery)")
        fig.colorbar(im, ax=ax, label="importance")
        fig.tight_layout()
        fig_dir = OUTPUTS_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"sparsity_{safe_model_name(args.model)}_{args.task}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"  figure: {fig_path}")


if __name__ == "__main__":
    main()
