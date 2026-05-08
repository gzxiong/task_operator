"""ZSL + 8-shot ICL on (model, task).

Writes three shards under `$TASK_OPERATOR_OUTPUTS/predictions/<model>/<task>/`:

    zero_shot.json                   — zero-shot test predictions
    full_icl_K8_test.json            — 8-shot ICL test predictions
    full_icl_K8_validation.json      — 8-shot ICL validation predictions
                                       (used as pseudo-labels for TO + baselines)

All three shards are resumable: rerunning skips records already present.
"""

from __future__ import annotations

import argparse

from task_operator import (
    ALL_TASKS,
    N_DEMOS,
    N_VALIDATION,
    apply_full_icl,
    apply_zero_shot,
    load_hf_model,
    load_split,
    make_default_config,
    run_predictions_shard,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-4B")
    ap.add_argument("--task", required=True, choices=ALL_TASKS)
    ap.add_argument("--n-validation", type=int, default=N_VALIDATION,
                    help=f"validation rows to teacher-force (default {N_VALIDATION})")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer, info = load_hf_model(args.model)
    print(f"  family={info['family']} layers={info['num_hidden_layers']}")

    demos = load_split(args.task, "demos")[:N_DEMOS]
    validation = load_split(args.task, "validation")[:args.n_validation]
    test = load_split(args.task, "test")
    config = make_default_config(args.task)

    # Zero-shot test
    print(f"\n[zero_shot] {args.task} n_test={len(test)}")
    res = run_predictions_shard(
        hf_id=args.model, task=args.task, condition="zero_shot",
        expected_n=len(test), record_iter=lambda: test,
        generate_one=lambda i, r: apply_zero_shot(model, tokenizer, r, config),
        config=config,
    )
    print(f"  acc={res['acc']:.4f} ({res['n_rows']} rows)")

    # 8-shot ICL test
    print(f"\n[full_icl_K8_test] {args.task} K={N_DEMOS} n_test={len(test)}")
    res = run_predictions_shard(
        hf_id=args.model, task=args.task, condition="full_icl_K8_test",
        expected_n=len(test), record_iter=lambda: test,
        generate_one=lambda i, r: apply_full_icl(model, tokenizer, demos, r, config),
        config=config,
        extra_settings={"n_demos": N_DEMOS},
    )
    print(f"  acc={res['acc']:.4f} ({res['n_rows']} rows)")

    # 8-shot ICL validation (pseudo-labels for TO + baselines)
    print(f"\n[full_icl_K8_validation] {args.task} K={N_DEMOS} n_val={len(validation)}")
    res = run_predictions_shard(
        hf_id=args.model, task=args.task, condition="full_icl_K8_validation",
        expected_n=len(validation), record_iter=lambda: validation,
        generate_one=lambda i, r: apply_full_icl(model, tokenizer, demos, r, config),
        config=config,
        extra_settings={"n_demos": N_DEMOS, "split": "validation"},
    )
    print(f"  done: {res['n_rows']} validation rows")


if __name__ == "__main__":
    main()
