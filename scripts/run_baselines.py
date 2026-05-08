"""TV / FV / ICV / Conceptor on (model, task).

All four baselines extract a representation from K=8 ICL prompts paired with
the model's own K=8 ICL pseudo-labels (read from
`predictions/<model>/<task>/full_icl_K8_validation.json` — produced by
`run_zsl_icl.py`), then apply the representation at test time. For TV / FV /
Conceptor the apply layer is searched on the validation pseudo-labels using
NLL; ICV uses every layer by construction (only λ is searched).

Outputs:
    baselines/<model>/<task>/<NAME>/representation.pt
    baselines/<model>/<task>/<NAME>/predictions.json
"""

from __future__ import annotations

import argparse

from task_operator import (
    ALL_TASKS,
    ConceptorConfig,
    FVConfig,
    ICVConfig,
    N_DEMOS,
    N_VALIDATION,
    TVConfig,
    load_hf_model,
    load_split,
    load_validation_icl_outputs,
    make_default_config,
    run_baseline_setting,
)


def make_baseline_config(task: str, baseline: str):
    base = make_default_config(task)
    common = dict(
        attention_sink=base.attention_sink,
        demo_template=base.demo_template,
        query_template=base.query_template,
        repetition_penalty=base.repetition_penalty,
        max_new_tokens=base.max_new_tokens,
        stopping_strings=base.stopping_strings,
        answer_phrases=base.answer_phrases,
    )
    if baseline == "TV":
        return TVConfig(**common)
    if baseline == "FV":
        return FVConfig(**common)
    if baseline == "ICV":
        return ICVConfig(
            **common,
            lambda_grid=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        )
    if baseline == "Conceptor":
        return ConceptorConfig(**common)
    raise ValueError(f"unknown baseline {baseline!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=ALL_TASKS)
    ap.add_argument("--baseline", required=True, choices=["TV", "FV", "ICV", "Conceptor"])
    ap.add_argument("--m", type=int, default=N_VALIDATION)
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
    config = make_baseline_config(args.task, args.baseline)

    print(f"\n[{args.baseline}] {args.task} m_used={len(val_records)} n_test={len(test)}")
    res = run_baseline_setting(
        hf_id=args.model, task=args.task, baseline=args.baseline,
        config=config, demos=demos,
        validation_records=val_records, validation_icl_outputs=val_outputs,
        test_records=test, model=model, tokenizer=tokenizer,
    )
    print(f"  acc={res['acc']:.4f} ({res['n_rows']} rows)")
    print(f"  representation: {res['representation_path']}")
    print(f"  predictions: {res['predictions_path']}")


if __name__ == "__main__":
    main()
