"""Resumable shard I/O for predictions / task_operator / baselines outputs.

The outputs root is resolved on every call from the `TASK_OPERATOR_OUTPUTS`
env var, falling back to `./outputs/` next to the package. Set the env var
before running scripts to redirect outputs (e.g. `TASK_OPERATOR_OUTPUTS=/path/to/run`).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
_OUTPUTS_DIR_ENV = "TASK_OPERATOR_OUTPUTS"
_DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"


def _outputs_dir() -> Path:
    """Resolve the outputs root on every call. A relative path in the env var
    is resolved against the repo root."""

    val = os.environ.get(_OUTPUTS_DIR_ENV)
    if not val:
        return _DEFAULT_OUTPUTS_DIR
    p = Path(val)
    return p if p.is_absolute() else (REPO_ROOT / p)


# Snapshot at import time, for `from task_operator import OUTPUTS_DIR`.
OUTPUTS_DIR = _outputs_dir()


def safe_model_name(hf_id: str) -> str:
    """`Qwen/Qwen3-4B` → `Qwen__Qwen3-4B`. Used in shard paths."""

    return hf_id.replace("/", "__")


def predictions_path(hf_id: str, task: str, condition: str) -> Path:
    return _outputs_dir() / "predictions" / safe_model_name(hf_id) / task / f"{condition}.json"


def task_operator_paths(hf_id: str, task: str, setting: str) -> tuple[Path, Path]:
    base = _outputs_dir() / "task_operator" / safe_model_name(hf_id) / task / setting
    return base / "knowledge.pt", base / "predictions.json"


def baselines_paths(hf_id: str, task: str, baseline: str) -> tuple[Path, Path]:
    base = _outputs_dir() / "baselines" / safe_model_name(hf_id) / task / baseline
    return base / "representation.pt", base / "predictions.json"


def atomic_write_json(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load_or_init_shard(path: Path, init_fn: Callable[[], dict]) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return init_fn()


def append_row(shard: dict, row: dict, path: Path) -> None:
    shard.setdefault("rows", []).append(row)
    atomic_write_json(path, shard)


def is_shard_complete(shard: dict, expected_n: int) -> bool:
    """Complete iff every record_index in range(expected_n) is present at least once."""

    rows = shard.get("rows") or []
    seen: set[int] = set()
    for r in rows:
        idx = r.get("record_index")
        if idx is None:
            return False
        seen.add(int(idx))
    return seen.issuperset(range(expected_n))


def done_test_indices(shard: dict) -> set[int]:
    return {int(r["test_index"]) for r in shard.get("rows") or [] if "test_index" in r}


def _canonical_json(blob: Any) -> str:
    return json.dumps(blob, sort_keys=True, default=str, ensure_ascii=False)


def fingerprint(blob: Any, *, length: int = 16) -> str:
    """Stable short hash of any JSON-serializable blob; used to detect cached
    results produced under settings different from the current ones."""

    return hashlib.sha256(_canonical_json(blob).encode("utf-8")).hexdigest()[:length]


def diff_settings(saved: dict, current: dict) -> list[str]:
    """Return a list of keys whose values differ between two settings dicts."""

    keys = set(saved or {}) | set(current or {})
    out = []
    for k in sorted(keys):
        if (saved or {}).get(k) != (current or {}).get(k):
            out.append(k)
    return out
