"""Aggregate predictions into the paper's main table.

Walks `$TASK_OPERATOR_OUTPUTS` for every (model, task) pair you pass on the
command line and prints a Markdown-style table with:

    model | task | ZSL | ICL | TV | FV | ICV | Conceptor | TO

Cells with no shard are shown as `--`. Also writes the same table to
`outputs/main_table.csv`. Pass `--models` and `--tasks` (comma-separated) to
restrict the grid; defaults run the 4×8 grid from the paper.
"""

from __future__ import annotations

import argparse
import csv
import json

from task_operator import (
    ALL_TASKS,
    OUTPUTS_DIR,
    baselines_paths,
    predictions_path,
    safe_model_name,
    task_operator_paths,
)

DEFAULT_MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def _read_acc(path):
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    rows = blob.get("rows") or []
    if not rows:
        return None
    return sum(float(r.get("score", 0.0)) for r in rows) / len(rows)


def _row(model: str, task: str) -> dict:
    out = {"model": model, "task": task}
    out["ZSL"] = _read_acc(predictions_path(model, task, "zero_shot"))
    out["ICL"] = _read_acc(predictions_path(model, task, "full_icl_K8_test"))
    for b in ("TV", "FV", "ICV", "Conceptor"):
        _, p = baselines_paths(model, task, b)
        out[b] = _read_acc(p)
    _, p = task_operator_paths(model, task, "allpos")
    out["TO"] = _read_acc(p)
    return out


def _fmt(v):
    return "  --  " if v is None else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated HF ids (default: paper's 4 models)")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS),
                    help=f"comma-separated tasks (default: {','.join(ALL_TASKS)})")
    ap.add_argument("--csv", default=str(OUTPUTS_DIR / "main_table.csv"),
                    help="output CSV path")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    cols = ["model", "task", "ZSL", "ICL", "TV", "FV", "ICV", "Conceptor", "TO"]
    rows = [_row(m, t) for m in models for t in tasks]

    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("-" * (len(c) + 2) for c in cols) + "|")
    for r in rows:
        cells = [r["model"].split("/")[-1], r["task"]] + [_fmt(r[c]) for c in cols[2:]]
        print("| " + " | ".join(cells) + " |")

    out_path = args.csv
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r[c]) for c in cols})
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
