"""Resumable per-(model, task) experiment loops shared by notebooks 02–06."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Callable

import torch

from .data import (
    ALL_TASKS,
    load_split,
    max_new_tokens_for_category,
    stop_strings_for_category,
    task_category,
)
from .evaluation import evaluate_generation
from .baselines import apply_baseline, extract_baseline
from .io import (
    append_row,
    atomic_write_json,
    baselines_paths,
    diff_settings,
    fingerprint as _fingerprint,
    is_shard_complete,
    load_or_init_shard,
    nto_paths,
    predictions_path,
    search_paths,
    task_operator_paths,
)
from .model import MODEL_REGISTRY, load_hf_model
from .prompting import (
    COT_DEMO_TEMPLATE,
    COT_QUERY_TEMPLATE,
    DEMO_TEMPLATE,
    QUERY_TEMPLATE,
)
from .task_operator import (
    TaskOperatorConfig,
    apply_task_operator,
    extract_knowledge,
)
from .nto import (
    NTOConfig,
    apply_canonical_task_operator,
    extract_canonical_knowledge,
)

DEFAULT_REP_PENALTY = 1.1
N_DEMOS = 8
N_VALIDATION = 32


def make_default_config(task: str, *, repetition_penalty: float = DEFAULT_REP_PENALTY) -> TaskOperatorConfig:
    """Default per-task config. Reasoning tasks use the CoT templates so that even
    zero-shot generation gets a "Let's think step by step." trigger; lexical and
    algorithmic stay on the bare IO templates (no CoT prefix needed for short
    string transformations)."""

    cat = task_category(task)
    if cat == "reasoning":
        demo_t, query_t = COT_DEMO_TEMPLATE, COT_QUERY_TEMPLATE
        # Single phrase "answer is" matches "The answer is", "the correct
        # answer is", "Therefore, the answer is", etc. via case-insensitive
        # regex (`re.IGNORECASE` in `_AnswerPhraseStop` and `_infer_stop_reason`).
        answer_phrases: tuple[str, ...] = ("answer is",)
    else:
        demo_t, query_t = DEMO_TEMPLATE, QUERY_TEMPLATE
        answer_phrases = ()
    return TaskOperatorConfig(
        attention_sink=True,
        demo_template=demo_t,
        query_template=query_t,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens_for_category(cat),
        stopping_strings=stop_strings_for_category(cat),
        answer_phrases=answer_phrases,
    )


def config_from_searched(
    task: str,
    fc: dict,
    *,
    all_layers: bool = False,
    all_positions: bool = False,
    repetition_penalty: float = DEFAULT_REP_PENALTY,
) -> TaskOperatorConfig:
    """Build a TaskOperatorConfig from a searched final_config blob.

    - all_layers=True   ⇒ active_layers_per_slot=None (notebook 05)
    - all_positions=True ⇒ template_active_ranks=None, n_template/n_content/n_output=-1,
                            active_layers_per_slot=None (notebook 06; ignores fc)
    - default            ⇒ use searched template_relative + n_content + n_output +
                            active_stages_per_slot (notebook 04)
    """

    base = make_default_config(task, repetition_penalty=repetition_penalty)
    if all_positions:
        base.template_active_ranks = None
        base.n_template = -1
        base.n_content = -1
        base.n_output = -1
        base.active_layers_per_slot = None
        return base
    t = fc.get("template_relative")
    base.template_active_ranks = list(t) if isinstance(t, list) else None
    base.n_content = int(fc.get("n_content", -1))
    base.n_output = int(fc.get("n_output", -1))
    if all_layers:
        base.active_layers_per_slot = None
    else:
        asps = fc.get("active_stages_per_slot") or {}
        base.active_layers_per_slot = {sid: list(stgs) for sid, stgs in asps.items()}
    return base


def load_validation_icl_outputs(
    hf_id: str,
    task: str,
    n: int,
    *,
    exclude_stops: tuple[str, ...] = ("max_new_tokens",),
) -> tuple[list[str] | None, list[int] | None]:
    """Read the K=8 full-ICL validation predictions saved by notebook 02, filtering
    rows whose `stop_reason` falls in `exclude_stops`.

    Default behavior drops rows that hit `max_new_tokens` — the model never reached
    its natural ending in those cases and the saved text is not a valid teacher-forced
    target for downstream search / task_operator extraction. The filter is a no-op
    for any (model, task) where every row stopped on `eos`, `answer_phrase`, or a
    stop-string match (e.g., almost all lex/alg pairs).

    Returns (outputs, kept_record_indices) — both lists are aligned and sorted by
    record_index. Callers should subset their `validation_records` by
    `kept_record_indices` before passing both into search / extract_knowledge so
    the per-sample zip is consistent. Returns (None, None) if the shard is missing.
    """

    p = predictions_path(hf_id, task, "full_icl_K8_validation")
    if not p.exists():
        return None, None
    blob = json.loads(p.read_text())
    rows = sorted(blob.get("rows", []), key=lambda r: int(r.get("record_index", 0)))[:n]
    outputs: list[str] = []
    kept: list[int] = []
    for r in rows:
        if r.get("stop_reason") in exclude_stops:
            continue
        outputs.append(r.get("generated_text", "") or "")
        kept.append(int(r.get("record_index", 0)))
    return outputs, kept


def load_searched_config(hf_id: str, task: str) -> dict | None:
    """Read the search trajectory's final_config.json (notebook 03 output) if present."""

    _, fp = search_paths(hf_id, task)
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def _verify_settings_match(saved_settings: dict, current_settings: dict, *, path: Path) -> None:
    saved_fp = (saved_settings or {}).get("fingerprint")
    current_fp = current_settings.get("fingerprint")
    if saved_fp is None:
        return  # legacy shard without fingerprint — accept and proceed
    if saved_fp == current_fp:
        return
    diff = diff_settings(saved_settings, current_settings)
    raise RuntimeError(
        f"settings fingerprint mismatch at {path}\n"
        f"  saved={saved_fp}  current={current_fp}\n"
        f"  differing keys: {diff}\n"
        f"  delete the file or change the output path to recompute."
    )


def _release_model(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _init_shard(hf_id: str, task: str, condition: str, settings: dict) -> dict:
    return {
        "hf_id": hf_id, "task": task, "condition": condition,
        "settings": settings, "rows": [],
    }


def run_predictions_shard(
    *,
    hf_id: str,
    task: str,
    condition: str,
    expected_n: int,
    record_iter: Callable[[], list[dict]],
    generate_one: Callable,  # (record) -> dict with generated_text + payload
    config: TaskOperatorConfig,
    extra_settings: dict | None = None,
):
    """Run a resumable per-record prediction shard.

    `generate_one(record_index, record) -> dict` should call apply_zero_shot /
    apply_full_icl / apply_task_operator and return the raw payload. This helper:
      1. loads or initializes the shard JSON,
      2. skips records whose `record_index` is already in the shard,
      3. for each remaining record runs `generate_one`, scores via evaluate_generation,
      4. appends the row + atomically rewrites the shard,
      5. returns the final shard accuracy summary.
    """

    path = predictions_path(hf_id, task, condition)
    settings = {
        "n_demos": N_DEMOS,
        "n_records": int(expected_n),
        "repetition_penalty": float(config.repetition_penalty),
        "attention_sink": bool(config.attention_sink),
        "max_new_tokens": int(config.max_new_tokens),
        "stopping_strings": list(config.stopping_strings or []),
        "demo_template": config.demo_template,
        "query_template": config.query_template,
        "do_sample": bool(config.do_sample),
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        **(extra_settings or {}),
    }
    # Add answer_phrases only when non-empty, so non-reasoning shards retain
    # their pre-answer_phrase fingerprint (no spurious cache invalidation).
    if config.answer_phrases:
        settings["answer_phrases"] = list(config.answer_phrases)
    settings["fingerprint"] = _fingerprint(settings)
    shard = load_or_init_shard(path, lambda: _init_shard(hf_id, task, condition, settings))
    _verify_settings_match(shard.get("settings") or {}, settings, path=path)
    if is_shard_complete(shard, expected_n):
        rows = shard["rows"]
        acc = sum(float(r.get("score", 0.0)) for r in rows) / max(len(rows), 1)
        return {"path": path, "rows": rows, "n_rows": len(rows), "acc": acc, "skipped": True}

    done_indices = {int(r["record_index"]) for r in shard.get("rows", [])}
    records = record_iter()
    for idx, record in enumerate(records):
        if idx >= expected_n or idx in done_indices:
            continue
        payload = generate_one(idx, record)
        scored = evaluate_generation(task, payload["generated_text"], record.get("target") or "")
        row = {
            "record_index": int(idx),
            "example_id": record.get("example_id"),
            "input": record.get("input"),
            "target": record.get("target"),
            "generated_text": scored["generated_text"],
            "parsed_answer": scored["parsed_answer"],
            "parsed_target": scored["parsed_target"],
            "is_correct": bool(scored["is_correct"]),
            "score": float(scored["score"]),
            "n_generated_tokens": int(payload.get("generated_tokens", 0)),
            "wall_time_sec": float(payload.get("wall_time_sec", 0.0)),
        }
        for k in ("n_prompt_records_total", "n_gen_records_total",
                  "n_template_positions", "n_content_positions"):
            if k in payload:
                row[k] = int(payload[k])
        if "stop_reason" in payload:
            row["stop_reason"] = str(payload["stop_reason"])
        append_row(shard, row, path)

    rows = shard["rows"]
    acc = sum(float(r.get("score", 0.0)) for r in rows) / max(len(rows), 1)
    return {"path": path, "rows": rows, "n_rows": len(rows), "acc": acc, "skipped": False}


def run_task_operator_setting(
    *,
    hf_id: str,
    task: str,
    setting: str,
    config: TaskOperatorConfig,
    demos,
    validation_records,
    validation_icl_outputs,
    test_records,
    model,
    tokenizer,
) -> dict:
    """Run one task_operator setting end-to-end with caching at both knowledge and row level.

    1. If knowledge.pt does not exist, run extract_knowledge and save.
    2. For each test record not yet in predictions.json, run apply_task_operator,
       evaluate, and append.

    Returns dict with paths and final accuracy.
    """

    knowledge_path, predictions_path_ = task_operator_paths(hf_id, task, setting)
    expected_n = len(test_records)

    demo_ids = [r.get("example_id") for r in (demos or [])][:N_DEMOS]
    val_ids = [r.get("example_id") for r in (validation_records or [])][:N_VALIDATION]
    val_outputs_digest = _fingerprint({
        "outputs": list(validation_icl_outputs or [])[:N_VALIDATION]
    })

    settings_blob = {
        "n_demos": N_DEMOS,
        "n_validation": len(validation_records),
        "n_records": int(expected_n),
        "repetition_penalty": float(config.repetition_penalty),
        "attention_sink": bool(config.attention_sink),
        "max_new_tokens": int(config.max_new_tokens),
        "stopping_strings": list(config.stopping_strings or []),
        "demo_template": config.demo_template,
        "query_template": config.query_template,
        "do_sample": bool(config.do_sample),
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "n_template": int(config.n_template),
        "n_content": int(config.n_content),
        "n_output": int(config.n_output),
        "template_active_ranks": list(config.template_active_ranks)
        if config.template_active_ranks is not None else None,
        "active_layers_per_slot": (
            {k: list(v) for k, v in config.active_layers_per_slot.items()}
            if config.active_layers_per_slot is not None else None
        ),
        "setting": setting,
        "demo_example_ids": demo_ids,
        "validation_example_ids": val_ids,
        "validation_icl_outputs_digest": val_outputs_digest,
    }
    # Add answer_phrases only when non-empty, so non-reasoning shards retain
    # their pre-answer_phrase fingerprint (no spurious cache invalidation).
    if config.answer_phrases:
        settings_blob["answer_phrases"] = list(config.answer_phrases)
    settings_blob["fingerprint"] = _fingerprint(settings_blob)

    shard = load_or_init_shard(
        predictions_path_,
        lambda: {
            "hf_id": hf_id, "task": task, "setting": setting,
            "settings": settings_blob, "rows": [],
        },
    )
    _verify_settings_match(shard.get("settings") or {}, settings_blob, path=predictions_path_)
    if is_shard_complete(shard, expected_n):
        rows = shard["rows"]
        acc = sum(float(r.get("score", 0.0)) for r in rows) / max(len(rows), 1)
        return {
            "knowledge_path": knowledge_path, "predictions_path": predictions_path_,
            "n_rows": len(rows), "acc": acc, "skipped": True,
        }

    # Load or extract knowledge, with fingerprint compatibility check on the saved blob.
    mean_circuit = None
    if knowledge_path.exists():
        loaded = torch.load(knowledge_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "mean_circuit" in loaded:
            saved_fp = loaded.get("fingerprint")
            if saved_fp == settings_blob["fingerprint"]:
                mean_circuit = loaded["mean_circuit"]
            else:
                diff = diff_settings(loaded.get("settings") or {}, settings_blob)
                raise RuntimeError(
                    f"knowledge fingerprint mismatch at {knowledge_path}\n"
                    f"  saved={saved_fp}  current={settings_blob['fingerprint']}\n"
                    f"  differing keys: {diff}\n"
                    f"  delete the file to recompute."
                )
        else:
            print(f"  [warn] legacy knowledge blob at {knowledge_path}; recomputing")

    if mean_circuit is None:
        mean_circuit = extract_knowledge(
            model, tokenizer, demos, validation_records, validation_icl_outputs, config,
        )
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mean_circuit": mean_circuit,
                "fingerprint": settings_blob["fingerprint"],
                "settings": settings_blob,
            },
            knowledge_path,
        )

    done = {int(r["record_index"]) for r in shard.get("rows", [])}
    for idx, record in enumerate(test_records):
        if idx in done:
            continue
        payload = apply_task_operator(model, tokenizer, record, mean_circuit, config)
        scored = evaluate_generation(task, payload["generated_text"], record.get("target") or "")
        row = {
            "record_index": int(idx),
            "example_id": record.get("example_id"),
            "input": record.get("input"),
            "target": record.get("target"),
            "generated_text": scored["generated_text"],
            "parsed_answer": scored["parsed_answer"],
            "parsed_target": scored["parsed_target"],
            "is_correct": bool(scored["is_correct"]),
            "score": float(scored["score"]),
            "n_generated_tokens": int(payload.get("generated_tokens", 0)),
            "n_prompt_records_total": int(payload.get("n_prompt_records_total", 0)),
            "n_gen_records_total": int(payload.get("n_gen_records_total", 0)),
            "wall_time_sec": float(payload.get("wall_time_sec", 0.0)),
        }
        if "stop_reason" in payload:
            row["stop_reason"] = str(payload["stop_reason"])
        append_row(shard, row, predictions_path_)

    rows = shard["rows"]
    acc = sum(float(r.get("score", 0.0)) for r in rows) / max(len(rows), 1)
    return {
        "knowledge_path": knowledge_path, "predictions_path": predictions_path_,
        "n_rows": len(rows), "acc": acc, "skipped": False,
    }


def make_nto_config(task: str, *, K_0: float = 8.0,
                    repetition_penalty: float = DEFAULT_REP_PENALTY,
                    all_positions: bool = True) -> NTOConfig:
    """NTO-equivalent of `config_from_searched(..., all_positions=True)`.

    Mirrors `make_default_config` but produces an `NTOConfig` with the K_0
    replay knob set; defaults to the all-positions/all-layers setting that
    notebook 06 uses for the standard task operator.
    """

    base = make_default_config(task, repetition_penalty=repetition_penalty)
    cfg = NTOConfig(
        attention_sink=base.attention_sink,
        demo_template=base.demo_template,
        query_template=base.query_template,
        n_template=-1 if all_positions else base.n_template,
        n_content=-1 if all_positions else base.n_content,
        n_output=-1 if all_positions else base.n_output,
        template_active_ranks=None if all_positions else base.template_active_ranks,
        active_layers_per_slot=None if all_positions else base.active_layers_per_slot,
        repetition_penalty=base.repetition_penalty,
        max_new_tokens=base.max_new_tokens,
        stopping_strings=base.stopping_strings,
        answer_phrases=base.answer_phrases,
        do_sample=base.do_sample,
        temperature=base.temperature,
        top_p=base.top_p,
        top_k=base.top_k,
        K_0=float(K_0),
    )
    return cfg


def run_nto_setting(
    *,
    hf_id: str,
    task: str,
    setting: str,
    K_0: float,
    config: NTOConfig,
    demos,
    validation_records,
    validation_icl_outputs,
    test_records,
    model,
    tokenizer,
) -> dict:
    """Run one NTO setting end-to-end with caching.

    Knowledge (canonical θ, μ_C) is K_0-independent and cached at
    `knowledge_nto.pt`; predictions are K_0-specific and cached at
    `predictions_K0_<K_0>.json`. Mirrors `run_task_operator_setting`.

    Returns dict with paths and final accuracy.
    """

    knowledge_path, predictions_path_ = nto_paths(hf_id, task, setting, K_0)
    expected_n = len(test_records)

    demo_ids = [r.get("example_id") for r in (demos or [])][:N_DEMOS]
    val_ids = [r.get("example_id") for r in (validation_records or [])][:N_VALIDATION]
    val_outputs_digest = _fingerprint({
        "outputs": list(validation_icl_outputs or [])[:N_VALIDATION]
    })

    knowledge_blob = {
        "method": "nto",
        "n_demos": N_DEMOS,
        "n_validation": len(validation_records),
        "repetition_penalty": float(config.repetition_penalty),
        "attention_sink": bool(config.attention_sink),
        "demo_template": config.demo_template,
        "query_template": config.query_template,
        "n_template": int(config.n_template),
        "n_content": int(config.n_content),
        "n_output": int(config.n_output),
        "template_active_ranks": list(config.template_active_ranks)
        if config.template_active_ranks is not None else None,
        "active_layers_per_slot": (
            {k: list(v) for k, v in config.active_layers_per_slot.items()}
            if config.active_layers_per_slot is not None else None
        ),
        "setting": setting,
        "demo_example_ids": demo_ids,
        "validation_example_ids": val_ids,
        "validation_icl_outputs_digest": val_outputs_digest,
    }
    knowledge_blob["fingerprint"] = _fingerprint(knowledge_blob)

    predictions_settings = {
        **{k: v for k, v in knowledge_blob.items() if k != "fingerprint"},
        "K_0": float(K_0),
        "n_records": int(expected_n),
        "max_new_tokens": int(config.max_new_tokens),
        "stopping_strings": list(config.stopping_strings or []),
        "do_sample": bool(config.do_sample),
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "knowledge_fingerprint": knowledge_blob["fingerprint"],
    }
    if config.answer_phrases:
        predictions_settings["answer_phrases"] = list(config.answer_phrases)
    predictions_settings["fingerprint"] = _fingerprint(predictions_settings)

    shard = load_or_init_shard(
        predictions_path_,
        lambda: {
            "method": "nto", "hf_id": hf_id, "task": task,
            "setting": setting, "K_0": float(K_0),
            "settings": predictions_settings, "rows": [],
        },
    )
    _verify_settings_match(shard.get("settings") or {}, predictions_settings, path=predictions_path_)
    if is_shard_complete(shard, expected_n):
        rows = shard["rows"]
        acc = sum(float(r.get("score", 0.0)) for r in rows) / max(len(rows), 1)
        return {
            "knowledge_path": knowledge_path, "predictions_path": predictions_path_,
            "n_rows": len(rows), "acc": acc, "skipped": True,
        }

    mean_circuit = None
    if knowledge_path.exists():
        loaded = torch.load(knowledge_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "mean_circuit" in loaded:
            saved_fp = loaded.get("fingerprint")
            if saved_fp is None or saved_fp == knowledge_blob["fingerprint"]:
                mean_circuit = loaded["mean_circuit"]
            else:
                diff = diff_settings(loaded.get("settings") or {}, knowledge_blob)
                raise RuntimeError(
                    f"NTO knowledge fingerprint mismatch at {knowledge_path}\n"
                    f"  saved={saved_fp}  current={knowledge_blob['fingerprint']}\n"
                    f"  differing keys: {diff}\n"
                    f"  delete the file to recompute."
                )

    if mean_circuit is None:
        mean_circuit = extract_canonical_knowledge(
            model, tokenizer, demos, validation_records, validation_icl_outputs, config,
        )
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mean_circuit": mean_circuit,
                "fingerprint": knowledge_blob["fingerprint"],
                "settings": knowledge_blob,
            },
            knowledge_path,
        )

    apply_config = NTOConfig(
        **{f.name: getattr(config, f.name) for f in __import__("dataclasses").fields(config)}
    )
    apply_config.K_0 = float(K_0)

    done = {int(r["record_index"]) for r in shard.get("rows", [])}
    for idx, record in enumerate(test_records):
        if idx in done:
            continue
        payload = apply_canonical_task_operator(model, tokenizer, record, mean_circuit, apply_config)
        scored = evaluate_generation(task, payload["generated_text"], record.get("target") or "")
        row = {
            "record_index": int(idx),
            "example_id": record.get("example_id"),
            "input": record.get("input"),
            "target": record.get("target"),
            "generated_text": scored["generated_text"],
            "parsed_answer": scored["parsed_answer"],
            "parsed_target": scored["parsed_target"],
            "is_correct": bool(scored["is_correct"]),
            "score": float(scored["score"]),
            "n_generated_tokens": int(payload.get("generated_tokens", 0)),
            "n_prompt_records_total": int(payload.get("n_prompt_records_total", 0)),
            "n_gen_records_total": int(payload.get("n_gen_records_total", 0)),
            "wall_time_sec": float(payload.get("wall_time_sec", 0.0)),
            "K_0": float(K_0),
        }
        if "stop_reason" in payload:
            row["stop_reason"] = str(payload["stop_reason"])
        append_row(shard, row, predictions_path_)

    rows = shard["rows"]
    acc = sum(float(r.get("score", 0.0)) for r in rows) / max(len(rows), 1)
    return {
        "knowledge_path": knowledge_path, "predictions_path": predictions_path_,
        "n_rows": len(rows), "acc": acc, "skipped": False,
    }


def iter_model_task_pairs(models: list[str] | None = None, tasks: list[str] | None = None):
    """Yield (hf_id, task) pairs filtered by user-provided subsets (None = use full registry)."""

    ms = list(models) if models else list(MODEL_REGISTRY)
    ts = list(tasks) if tasks else list(ALL_TASKS)
    for m in ms:
        for t in ts:
            yield m, t


def _baseline_settings_blob(
    *, hf_id, task, baseline, config, expected_n,
    demo_ids, val_ids, val_outputs_digest,
) -> dict:
    """Per-baseline settings blob → fingerprint. Includes the baseline name plus
    every field of the per-baseline dataclass, so each method's cache is keyed
    on its own knobs."""

    from dataclasses import asdict
    cfg_dict = asdict(config)
    # Drop empty `answer_phrases` so non-reasoning baselines keep their
    # pre-answer_phrases fingerprint (the field is new; empty tuple is the
    # backwards-compatible default and shouldn't show up in serialised settings).
    if not cfg_dict.get("answer_phrases"):
        cfg_dict.pop("answer_phrases", None)
    else:
        cfg_dict["answer_phrases"] = list(cfg_dict["answer_phrases"])
    blob = {
        "baseline": baseline,
        "hf_id": hf_id,
        "task": task,
        "n_demos": N_DEMOS,
        "n_validation": int(len(val_ids)),
        "n_records": int(expected_n),
        "demo_example_ids": demo_ids,
        "validation_example_ids": val_ids,
        "validation_icl_outputs_digest": val_outputs_digest,
        **{k: v for k, v in cfg_dict.items() if v is not None or k in {"layer", "lambda_grid"}},
    }
    blob["fingerprint"] = _fingerprint(blob)
    return blob


def run_baseline_setting(
    *,
    hf_id: str,
    task: str,
    baseline: str,
    config,
    demos,
    validation_records,
    validation_icl_outputs,
    test_records,
    model,
    tokenizer,
) -> dict:
    """Run one baseline (TV / FV / ICV / Conceptor) end-to-end with caching.

    1. If `representation.pt` exists with a matching fingerprint, load it.
       Otherwise call `extract_baseline` and save.
    2. For each test record not yet in `predictions.json`, call
       `apply_baseline`, score via `evaluate_generation`, and append the row.

    Mirrors `run_task_operator_setting` but with baselines/<m>/<task>/<baseline>/
    output paths and a baseline-specific settings blob.
    """

    rep_path, predictions_path_ = baselines_paths(hf_id, task, baseline)
    expected_n = len(test_records)

    demo_ids = [r.get("example_id") for r in (demos or [])][:N_DEMOS]
    val_ids = [r.get("example_id") for r in (validation_records or [])][:N_VALIDATION]
    val_outputs_digest = _fingerprint({
        "outputs": list(validation_icl_outputs or [])[:N_VALIDATION]
    })

    settings_blob = _baseline_settings_blob(
        hf_id=hf_id, task=task, baseline=baseline, config=config,
        expected_n=expected_n, demo_ids=demo_ids, val_ids=val_ids,
        val_outputs_digest=val_outputs_digest,
    )

    shard = load_or_init_shard(
        predictions_path_,
        lambda: {
            "hf_id": hf_id, "task": task, "baseline": baseline,
            "settings": settings_blob, "rows": [],
        },
    )
    _verify_settings_match(shard.get("settings") or {}, settings_blob, path=predictions_path_)
    if is_shard_complete(shard, expected_n):
        rows = shard["rows"]
        acc = sum(float(r.get("score", 0.0)) for r in rows) / max(len(rows), 1)
        return {
            "representation_path": rep_path, "predictions_path": predictions_path_,
            "n_rows": len(rows), "acc": acc, "skipped": True,
        }

    # Load or extract representation, fingerprint-checked.
    repr_dict = None
    if rep_path.exists():
        loaded = torch.load(rep_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "repr" in loaded:
            saved_fp = loaded.get("fingerprint")
            if saved_fp == settings_blob["fingerprint"]:
                repr_dict = loaded["repr"]
            else:
                diff = diff_settings(loaded.get("settings") or {}, settings_blob)
                raise RuntimeError(
                    f"baseline representation fingerprint mismatch at {rep_path}\n"
                    f"  saved={saved_fp}  current={settings_blob['fingerprint']}\n"
                    f"  differing keys: {diff}\n"
                    f"  delete the file to recompute."
                )
        else:
            print(f"  [warn] legacy baseline blob at {rep_path}; recomputing")

    if repr_dict is None:
        repr_dict = extract_baseline(
            baseline, model, tokenizer, demos, validation_records, validation_icl_outputs, config,
        )
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "repr": repr_dict,
                "fingerprint": settings_blob["fingerprint"],
                "settings": settings_blob,
            },
            rep_path,
        )

    done = {int(r["record_index"]) for r in shard.get("rows", [])}
    for idx, record in enumerate(test_records):
        if idx in done:
            continue
        payload = apply_baseline(baseline, model, tokenizer, record, repr_dict, config)
        scored = evaluate_generation(task, payload["generated_text"], record.get("target") or "")
        row = {
            "record_index": int(idx),
            "example_id": record.get("example_id"),
            "input": record.get("input"),
            "target": record.get("target"),
            "generated_text": scored["generated_text"],
            "parsed_answer": scored["parsed_answer"],
            "parsed_target": scored["parsed_target"],
            "is_correct": bool(scored["is_correct"]),
            "score": float(scored["score"]),
            "n_generated_tokens": int(payload.get("generated_tokens", 0)),
            "wall_time_sec": float(payload.get("wall_time_sec", 0.0)),
        }
        if "stop_reason" in payload:
            row["stop_reason"] = str(payload["stop_reason"])
        append_row(shard, row, predictions_path_)

    rows = shard["rows"]
    acc = sum(float(r.get("score", 0.0)) for r in rows) / max(len(rows), 1)
    return {
        "representation_path": rep_path, "predictions_path": predictions_path_,
        "n_rows": len(rows), "acc": acc, "skipped": False,
    }
