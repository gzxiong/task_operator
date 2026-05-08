"""Task Operator (TO) on (model, task) with the all-positions / all-layers
configuration used for the paper's main table.

Reads validation pseudo-labels from
`predictions/<model>/<task>/full_icl_K8_validation.json` (produced by
`run_zsl_icl.py`), extracts a mean_circuit on the validation samples whose
ICL output didn't truncate, then applies it to the test set.

Outputs:
    task_operator/<model>/<task>/allpos/knowledge.pt      — extracted mean_circuit
    task_operator/<model>/<task>/allpos/predictions.json  — TO test predictions
"""

from __future__ import annotations

import argparse

from task_operator import (
    ALL_TASKS,
    N_DEMOS,
    N_VALIDATION,
    load_hf_model,
    load_split,
    load_validation_icl_outputs,
    make_allpos_config,
    run_task_operator_setting,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=ALL_TASKS)
    ap.add_argument("--m", type=int, default=N_VALIDATION,
                    help=f"validation samples for circuit extraction (default {N_VALIDATION})")
    args = ap.parse_args()

    val_outputs, kept_indices = load_validation_icl_outputs(args.model, args.task, args.m)
    if val_outputs is None:
        raise SystemExit(
            f"missing predictions/<{args.model}>/<{args.task}>/full_icl_K8_validation.json — "
            "run scripts/run_zsl_icl.py first"
        )

    print(f"Loading {args.model} ...")
    model, tokenizer, info = load_hf_model(args.model)
    print(f"  family={info['family']} layers={info['num_hidden_layers']}")

    demos = load_split(args.task, "demos")[:N_DEMOS]
    validation_full = load_split(args.task, "validation")
    val_records = [validation_full[i] for i in kept_indices]
    test = load_split(args.task, "test")

    config = make_allpos_config(args.task)
    print(f"\n[TO allpos] {args.task} m_used={len(val_records)} n_test={len(test)}")
    res = run_task_operator_setting(
        hf_id=args.model, task=args.task, setting="allpos",
        config=config, demos=demos,
        validation_records=val_records, validation_icl_outputs=val_outputs,
        test_records=test, model=model, tokenizer=tokenizer,
    )
    print(f"  acc={res['acc']:.4f} ({res['n_rows']} rows)")
    print(f"  knowledge: {res['knowledge_path']}")
    print(f"  predictions: {res['predictions_path']}")


if __name__ == "__main__":
    main()
