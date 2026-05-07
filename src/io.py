"""Resumable shard I/O for predictions / search trajectories / task_operator outputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUTS_DIR_ENV = "CIRICL_OUTPUTS_DIR"
_DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs_final_v2"


def _outputs_dir() -> Path:
    """Resolve the outputs root on every call. If `CIRICL_OUTPUTS_DIR` is set in
    the environment, use that; otherwise fall back to `outputs_final_v2/` next to
    the project root. Notebooks that target a different tree (e.g.
    `outputs_final_v2_rep_1.0/`) should set the env var in their config cell
    *before* importing from `src_final_v2`. The path helpers below re-read the
    env var on every call so even late-set values take effect.

    A relative path in the env var is resolved against `PROJECT_ROOT` so
    notebooks can use `'outputs_final_v2_rep_1.0'` and stay portable across
    servers with the same project layout but different absolute roots.
    """

    val = os.environ.get(_OUTPUTS_DIR_ENV)
    if not val:
        return _DEFAULT_OUTPUTS_DIR
    p = Path(val)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# Snapshot at import time, for code that does `from src_final_v2 import OUTPUTS_DIR`
# (e.g., the summary cells at the bottom of every experiment notebook).
OUTPUTS_DIR = _outputs_dir()


def safe_model_name(hf_id: str) -> str:
    """`Qwen/Qwen3-4B` → `Qwen__Qwen3-4B`. Used in shard paths."""

    return hf_id.replace("/", "__")


def predictions_path(hf_id: str, task: str, condition: str) -> Path:
    return _outputs_dir() / "predictions" / safe_model_name(hf_id) / task / f"{condition}.json"


def search_paths(hf_id: str, task: str) -> tuple[Path, Path]:
    base = _outputs_dir() / "search" / safe_model_name(hf_id) / task
    return base / "trajectory.json", base / "final_config.json"


def task_operator_paths(hf_id: str, task: str, setting: str) -> tuple[Path, Path]:
    base = _outputs_dir() / "task_operator" / safe_model_name(hf_id) / task / setting
    return base / "knowledge.pt", base / "predictions.json"


def nto_paths(hf_id: str, task: str, setting: str, K_0: float) -> tuple[Path, Path]:
    """Path scheme for the Normalized Task Operator (canonical θ, μ_C).

    Knowledge is K_0-independent (one extraction reused across the K_0 sweep);
    predictions are K_0-specific.
    """

    base = _outputs_dir() / "nto" / safe_model_name(hf_id) / task / setting
    f = float(K_0)
    tag = str(int(f)) if f == int(f) else str(f).replace(".", "p")
    return base / "knowledge_nto.pt", base / f"predictions_K0_{tag}.json"


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
    """Complete iff every record_index in `range(expected_n)` is present at least once.

    Length alone is unsafe under parallel writers — a duplicated index plus a missing one
    would still pass `len(rows) >= expected_n`.
    """

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
    """Stable short hash of any JSON-serializable blob. Used to detect cached results that
    were produced under settings different from the current ones.
    """

    return hashlib.sha256(_canonical_json(blob).encode("utf-8")).hexdigest()[:length]


def diff_settings(saved: dict, current: dict) -> list[str]:
    """Return a list of keys whose values differ between two settings dicts."""

    keys = set(saved or {}) | set(current or {})
    out = []
    for k in sorted(keys):
        if (saved or {}).get(k) != (current or {}).get(k):
            out.append(k)
    return out
