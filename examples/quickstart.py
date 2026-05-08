"""End-to-end Task Operator demo on a single (model, task).

Loads Qwen3-4B + the `uppercase` task by default, runs:

    1. Zero-shot accuracy on the test set (small subset by default)
    2. 8-shot ICL accuracy on the same subset (and emits validation
       pseudo-labels needed for TO extraction)
    3. Extracts a Task Operator from the validation set
    4. Applies the TO at zero-shot inference and reports its accuracy

This is the smoke test reviewers run first. Override the model / task / sample
counts via CLI flags. Expect TO to recover most of the gap between ZSL and ICL.
"""

from __future__ import annotations

import argparse

import torch

from task_operator import (
    apply_full_icl,
    apply_task_operator,
    apply_zero_shot,
    evaluate_generation,
    extract_knowledge,
    load_hf_model,
    load_split,
    make_allpos_config,
    make_default_config,
)


def _accuracy(rows: list[dict]) -> float:
    return sum(float(r["score"]) for r in rows) / max(len(rows), 1)


def _generate_and_score(task, generate_fn, records):
    rows = []
    for rec in records:
        payload = generate_fn(rec)
        scored = evaluate_generation(task, payload["generated_text"], rec.get("target") or "")
        rows.append({"score": scored["score"], "is_correct": scored["is_correct"],
                     "generated_text": scored["generated_text"], "stop_reason": payload.get("stop_reason")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--task", default="uppercase")
    ap.add_argument("--n-test", type=int, default=8, help="test records to evaluate (default 8)")
    ap.add_argument("--m", type=int, default=32, help="validation samples for TO extraction")
    args = ap.parse_args()

    print(f"=== Quick start: {args.model} × {args.task} ===")
    print(f"Loading model ...")
    model, tokenizer, info = load_hf_model(args.model)
    print(f"  family={info['family']} layers={info['num_hidden_layers']}")

    demos = load_split(args.task, "demos")[:8]
    validation = load_split(args.task, "validation")[:args.m]
    test = load_split(args.task, "test")[:args.n_test]
    config = make_default_config(args.task)

    print("\n[1/4] Zero-shot ...")
    zsl_rows = _generate_and_score(
        args.task, lambda r: apply_zero_shot(model, tokenizer, r, config), test,
    )
    print(f"  ZSL accuracy: {_accuracy(zsl_rows):.4f}")

    print("\n[2/4] 8-shot ICL on test ...")
    icl_rows = _generate_and_score(
        args.task, lambda r: apply_full_icl(model, tokenizer, demos, r, config), test,
    )
    print(f"  ICL accuracy: {_accuracy(icl_rows):.4f}")

    print(f"\n[3/4] Generating ICL pseudo-labels on {len(validation)} validation samples ...")
    val_outputs = []
    for rec in validation:
        payload = apply_full_icl(model, tokenizer, demos, rec, config)
        val_outputs.append(payload["generated_text"])

    print("\n[4/4] Extracting Task Operator + applying at ZSL ...")
    allpos = make_allpos_config(args.task)
    mean_circuit = extract_knowledge(
        model, tokenizer, demos, validation, val_outputs, allpos,
    )
    n_sites = sum(len(v) for v in mean_circuit["records_by_layer"].values())
    print(f"  extracted: {n_sites} (layer, slot) sites across "
          f"{mean_circuit['meta']['n_layers']} layers, "
          f"{mean_circuit['meta']['n_val_samples']} samples")
    to_rows = _generate_and_score(
        args.task, lambda r: apply_task_operator(model, tokenizer, r, mean_circuit, allpos), test,
    )
    print(f"  TO accuracy: {_accuracy(to_rows):.4f}")

    print("\n=== Summary ===")
    print(f"  ZSL : {_accuracy(zsl_rows):.4f}")
    print(f"  ICL : {_accuracy(icl_rows):.4f}")
    print(f"  TO  : {_accuracy(to_rows):.4f}")


if __name__ == "__main__":
    main()
