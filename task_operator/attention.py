"""Family-generalized eager-attention patching.

Provides a context manager that monkey-patches each model family's
`eager_attention_forward` symbol so we can:
    1. mask attention from specific query rows to a key span (`mask_spec` entries),
    2. capture per-layer post-softmax scores and probabilities (`capture_store`).

`mask_spec` entry shape (matching v1):
    {
      "layers": [int, ...],          # which decoder layer indices the entry applies to
      "query_rows": [int, ...],       # rows to zero/scale at attention computation
      "key_span": [int, int],         # [start, end) keys to mask
      "alpha": float = 0.0,           # 0 → -inf logit (full block); >0 → +log(alpha)
      "block_generated": bool = False, # if True, mask only when seq_len==1 (gen step)
      "skip_positions": tuple[int],   # gen-only override: skip masking when key length
                                      # equals one of these (used by replay paths)
    }

`capture_store` is populated as `{layer_idx: {"scores_post": Tensor, "attn_probs": Tensor}}`
(both detached, on CPU, with the batch dimension squeezed).
"""

from __future__ import annotations

import contextlib
import math


NEG_LARGE = -1e9


def _normalize_runtime_args(family: str, args: tuple, kwargs: dict) -> dict:
    """Normalize family-specific eager-attention arguments to a shared representation."""

    if family == "gemma3":
        dropout = kwargs.get("dropout", args[0] if len(args) > 0 else 0.0)
        scaling = kwargs.get("scaling", args[1] if len(args) > 1 else None)
        softcap = kwargs.get("softcap", args[2] if len(args) > 2 else None)
    else:
        scaling = kwargs.get("scaling", args[0] if len(args) > 0 else None)
        dropout = kwargs.get("dropout", args[1] if len(args) > 1 else 0.0)
        softcap = None
    return {
        "dropout": float(dropout or 0.0),
        "scaling": scaling,
        "softcap": softcap,
    }


@contextlib.contextmanager
def patched_attention(model, *, mask_spec=None, capture_store=None):
    """Patch eager attention for one supported model family.

    On exit, the original eager_attention_forward is restored even if the body raises.
    """

    import torch
    import torch.nn as nn

    from .model import detect_model_family, get_modeling_backend

    family = detect_model_family(model)
    backend = get_modeling_backend(family)
    module = backend["module"]
    original = getattr(module, backend["eager_name"])
    repeat_kv = backend["repeat_kv"]
    active_spec = list(mask_spec or [])

    def wrapped(attn_module, query, key, value, attention_mask, *args, **kwargs):
        runtime = _normalize_runtime_args(family, args, kwargs)
        scaling = runtime["scaling"]
        if scaling is None:
            scaling = float(attn_module.head_dim) ** -0.5
        dropout = runtime["dropout"]
        softcap = runtime["softcap"]

        key_states = repeat_kv(key, attn_module.num_key_value_groups)
        value_states = repeat_kv(value, attn_module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if softcap is not None:
            attn_weights = torch.tanh(attn_weights / softcap) * softcap
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        layer_idx = int(getattr(attn_module, "layer_idx", -1))
        for entry in active_spec:
            if layer_idx not in {int(layer) for layer in entry.get("layers", [])}:
                continue
            key_start, key_end = [int(v) for v in entry["key_span"]]
            alpha = float(entry.get("alpha", 0.0))
            if entry.get("block_generated", False) and attn_weights.shape[-2] == 1:
                if attn_weights.shape[-1] in set(entry.get("skip_positions", ())):
                    continue
                if alpha <= 0.0:
                    attn_weights[:, :, 0, key_start:key_end] = NEG_LARGE
                else:
                    attn_weights[:, :, 0, key_start:key_end] += math.log(alpha)
                continue
            for row in entry.get("query_rows", ()):
                row = int(row)
                if row < 0 or row >= attn_weights.shape[-2]:
                    continue
                if alpha <= 0.0:
                    attn_weights[:, :, row, key_start:key_end] = NEG_LARGE
                else:
                    attn_weights[:, :, row, key_start:key_end] += math.log(alpha)

        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if capture_store is not None:
            # Caller may set capture_store['_keep_on_device'] = True before invoking the
            # patched forward to avoid GPU→CPU sync per layer (a major bottleneck for
            # downstream code that immediately re-uploads scores/probs to GPU). The flag
            # is read here, layer-agnostic.
            if capture_store.get("_keep_on_device"):
                capture_store[layer_idx] = {
                    "scores_post": attn_weights.detach()[0],
                    "attn_probs": attn_probs.detach()[0],
                }
            else:
                capture_store[layer_idx] = {
                    "scores_post": attn_weights.detach()[0].cpu(),
                    "attn_probs": attn_probs.detach()[0].cpu(),
                }
        attn_probs = nn.functional.dropout(attn_probs, p=dropout, training=attn_module.training)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_probs

    setattr(module, backend["eager_name"], wrapped)
    try:
        yield
    finally:
        setattr(module, backend["eager_name"], original)
