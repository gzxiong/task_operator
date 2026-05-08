# Task Operator

Code for **Capturing In-Context Learning Dynamics with Task Operators**.

A Task Operator (TO) extracts an analytic per-(layer, head, position) affine
operator `(w, b)` from one ICL forward pass, then replays it as a dynamic
update to the attention output projection at zero-shot inference. Across four
LLMs and eight tasks the method recovers most of the gap between zero-shot
and full ICL while paying only the zero-shot context cost.

The key identity, derived in the paper, is that for every attention head

```
    u_t = w_t · u'_t + b_t
```

where `u'_t` is the head output with attention to the demonstration context
masked out, `w_t = Z'_t / Z_t` is the fraction of softmax mass on non-context
positions, and `b_t = Σ_{i∈C} α_{t,i} v_i` is the context contribution. The
parameters are extracted from a few validation samples and applied as a
diagonal-scale-plus-bias update on `W_O` during ZSL inference.

## Install

Requires Python ≥ 3.10. From the repo root:

```bash
pip install -e .
```

The shipped data splits (`data/`, ~5 MB) are loaded automatically; no
external download is required for the lexical or algorithmic tasks. Reasoning
splits (`gsm8k`, `math500`, `gpqa`) are also shipped pre-built.

## Quick start

```bash
python examples/quickstart.py
```

Loads Qwen3-4B + the `uppercase` task, runs ZSL, 8-shot ICL, then extracts a
TO and applies it at ZSL. Prints the three accuracies side by side.

## Reproducing the main table

The paper reports a 4-model × 8-task grid of (ZSL, ICL, TV, FV, ICV,
Conceptors, TO). Every run is resumable. Rerunning a script skips records
already present in the shard. Outputs land under `outputs/` (override with
`TASK_OPERATOR_OUTPUTS=...`).

For one cell, run the four scripts in order:

```bash
MODEL=Qwen/Qwen3-4B
TASK=uppercase

# 1. Zero-shot, 8-shot ICL test, 8-shot ICL validation (used as pseudo-labels)
python scripts/run_zsl_icl.py --model "$MODEL" --task "$TASK"

# 2. Task Operator (all positions, all layers)
python scripts/run_to.py --model "$MODEL" --task "$TASK"

# 3. Baselines (one at a time; each is independent)
for B in TV FV ICV Conceptor; do
    python scripts/run_baselines.py --model "$MODEL" --task "$TASK" --baseline "$B"
done
```

Once all four (model, task) cells you care about are populated:

```bash
python scripts/summarize.py
```

prints the table to stdout and writes `outputs/main_table.csv`. Pass
`--models` and `--tasks` (comma-separated) to restrict the grid.

## Reproducing Figure 2 (sparsity maps)

Per-(layer, position) site importance via NLL recovery. Requires `run_to.py`
to have produced `knowledge.pt` for the cell first.

```bash
python scripts/run_sparsity.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --task uppercase \
    --plot
```

The `--plot` flag writes a heatmap to `outputs/figures/sparsity_<model>_<task>.png`.

## Layout

```
task_operator/         # the package
  core.py              # extract_knowledge / apply_task_operator (the TO method)
  attention.py         # patched eager attention (mask + capture)
  generation.py        # generate_with_hooks (replay at o_proj)
  baselines.py         # TV / FV / ICV / Conceptors
  nll_site_analysis.py # NLL-recovery site importance (Section 3.3)
  nll_utils.py         # shared NLL eval helpers
  data.py              # task registry, load_split
  prompting.py         # ICL / ZSL templates incl. CoT
  evaluation.py        # answer parsing + scoring
  experiments.py       # high-level runners (resumable)
  io.py                # path builders for outputs/
  model.py / spans.py / tokenization.py
scripts/               # CLI entrypoints
data/                  # 8-task splits (lexical, algorithmic, reasoning)
examples/quickstart.py # single-cell end-to-end demo
```

## API

```python
from task_operator import (
    load_hf_model, load_split,
    make_default_config, make_allpos_config,
    apply_zero_shot, apply_full_icl,
    extract_knowledge, apply_task_operator,
)

model, tokenizer, info = load_hf_model("Qwen/Qwen3-4B")
demos = load_split("uppercase", "demos")[:8]
validation = load_split("uppercase", "validation")[:32]
test = load_split("uppercase", "test")[:8]

config = make_allpos_config("uppercase")
val_outputs = [
    apply_full_icl(model, tokenizer, demos, r, config)["generated_text"]
    for r in validation
]
mean_circuit = extract_knowledge(
    model, tokenizer, demos, validation, val_outputs, config,
)
for rec in test:
    payload = apply_task_operator(model, tokenizer, rec, mean_circuit, config)
    print(payload["generated_text"])
```

## Tasks

| Category    | Tasks                                |
|-------------|--------------------------------------|
| Lexical     | translation, linguistic              |
| Algorithmic | uppercase, reverse, deduplicate      |
| Reasoning   | gsm8k, math500, gpqa                 |

Reasoning tasks use a Chain-of-Thought template ("Let's think step by step.")
in both demos and the zero-shot query; lexical and algorithmic tasks use the
bare `Input: ... \nOutput:` template.
