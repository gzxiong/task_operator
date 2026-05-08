"""HuggingFace model loading + family detection + decoder-layer accessors.

Models are referenced by their HF ids directly. `MODEL_REGISTRY` holds the 12 IDs
used by all v2 experiment notebooks. `load_hf_model` always uses
`attn_implementation='eager'` so attention captures and patches work.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REGISTRY = [
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-8B",
    "google/gemma-3-4b-pt",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def detect_model_family(model) -> str:
    """Returns one of 'qwen3' / 'gemma3' / 'llama' from config.model_type / architectures."""

    cfg = model.config
    text_cfg = get_text_config(cfg)
    type_str = (getattr(text_cfg, "model_type", "") or "").lower()
    arch_strs = " ".join(getattr(cfg, "architectures", None) or []).lower()
    blob = f"{type_str} {arch_strs}"
    for fam in ("qwen3", "gemma3", "llama"):
        if fam in blob:
            return fam
    raise ValueError(f"Could not detect model family from: {blob!r}")


def get_text_config(model_or_config):
    """Extract the text-side config from a model or config (handles multimodal wrappers)."""

    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "text_config", cfg)


def get_decoder_layers(model):
    """Find the list of decoder layers regardless of multimodal wrappers."""

    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None),
    ]
    for c in candidates:
        if c is not None:
            return c
    raise ValueError("Could not locate decoder layers on this model")


_MODULE_BY_FAMILY = {
    "qwen3": "transformers.models.qwen3.modeling_qwen3",
    "gemma3": "transformers.models.gemma3.modeling_gemma3",
    "llama": "transformers.models.llama.modeling_llama",
}


def get_modeling_backend(family: str) -> dict[str, Any]:
    """Resolve the HF modeling module + repeat_kv helper for one family. Used by attention.py."""

    module = import_module(_MODULE_BY_FAMILY[family])
    return {"module": module, "eager_name": "eager_attention_forward", "repeat_kv": module.repeat_kv}


def _resolve_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    raise TypeError(f"unrecognized dtype: {dtype!r}")


def load_hf_model(hf_id: str, *, device: str | None = None, dtype: Any = None):
    """Load a CausalLM with eager attention. Returns (model, tokenizer, info)."""

    resolved_dtype = _resolve_dtype(dtype)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=resolved_dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = model.to(resolved_device).eval()

    text_cfg = get_text_config(model)
    info = {
        "hf_id": hf_id,
        "family": detect_model_family(model),
        "device": resolved_device,
        "dtype": resolved_dtype,
        "num_hidden_layers": int(text_cfg.num_hidden_layers),
        "num_attention_heads": int(text_cfg.num_attention_heads),
        "num_key_value_heads": int(getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)),
        "hidden_size": int(text_cfg.hidden_size),
        "head_dim": int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)),
    }
    return model, tokenizer, info
