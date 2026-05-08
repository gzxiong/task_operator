"""Task registry, category mapping, and per-category generation defaults.

Three task categories:
    - lexical: translation, linguistic
    - algorithmic: uppercase, reverse, deduplicate
    - reasoning: gsm8k, math500, gpqa

Reasoning tasks generate up to 512 new tokens and stop on '\\nInput:' (substring
of the demo separator '\\n\\nInput:'); lexical and algorithmic tasks generate up
to 64 tokens and stop on '\\n'.

The data root is resolved from the `TASK_OPERATOR_DATA` env var, falling back
to the `data/` directory next to the package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
_DATA_DIR_ENV = "TASK_OPERATOR_DATA"
_DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _data_dir() -> Path:
    val = os.environ.get(_DATA_DIR_ENV)
    if not val:
        return _DEFAULT_DATA_DIR
    p = Path(val)
    return p if p.is_absolute() else (REPO_ROOT / p)


LEXICAL_TASKS = ["translation", "linguistic"]
ALGORITHMIC_TASKS = ["uppercase", "reverse", "deduplicate"]
REASONING_TASKS = ["gsm8k", "math500", "gpqa"]
ALL_TASKS = [*LEXICAL_TASKS, *ALGORITHMIC_TASKS, *REASONING_TASKS]

_CATEGORY_BY_TASK = {
    **{t: "lexical" for t in LEXICAL_TASKS},
    **{t: "algorithmic" for t in ALGORITHMIC_TASKS},
    **{t: "reasoning" for t in REASONING_TASKS},
}


def task_category(task: str) -> str:
    if task not in _CATEGORY_BY_TASK:
        raise ValueError(f"Unknown task: {task!r}")
    return _CATEGORY_BY_TASK[task]


def stop_strings_for_category(category: str) -> list[str]:
    """Per-category stop string set passed to generate_with_hooks.

    - reasoning: `"\\nInput:"` — substring of the actual demo separator
      `"\\n\\nInput:"`. Stops at the demo boundary the model has learned,
      letting reasoning traverse mid-output `\\n\\n` paragraph breaks.
    - others: `"\\n"` — covered by id-based stop (single-token in all
      our tokenizers); no per-step criterion needed.
    """

    return ["\nInput:"] if category == "reasoning" else ["\n"]


def max_new_tokens_for_category(category: str) -> int:
    return 512 if category == "reasoning" else 64


def split_path(task: str, split: str) -> Path:
    return _data_dir() / task_category(task) / task / f"{split}.json"


def load_split(task: str, split: str) -> list[dict]:
    """Load demos / validation / test records for a task. Returns the records list."""

    p = split_path(task, split)
    blob = json.loads(p.read_text())
    return blob["records"]


def assert_splits_exist(tasks: list[str] | None = None,
                         splits: tuple[str, ...] = ("demos", "validation", "test")) -> None:
    """Preflight: raise FileNotFoundError listing every missing `<task>/<split>.json`."""

    ts = list(tasks) if tasks else list(ALL_TASKS)
    missing = []
    for t in ts:
        for s in splits:
            p = split_path(t, s)
            if not p.exists():
                missing.append(f"{task_category(t)}/{t}/{s}.json")
    if missing:
        listing = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Missing data split files (set TASK_OPERATOR_DATA or run scripts/prepare_data.py):\n  {listing}"
        )
