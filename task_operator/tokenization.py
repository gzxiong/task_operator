"""Tokenization with `attention_sink` semantics.

The `attention_sink` flag (renamed from v1's `add_bos`) inserts a sink token at
position 0 *only if one is not already there*. Many tokenizers (Llama, Gemma)
auto-prepend BOS when calling them with default settings; tokenizing with
`add_special_tokens=False` skips that. This module always tokenizes with
`add_special_tokens=False` and then inserts a sink iff requested AND none is present.

Sink token preference: BOS → EOS (when BOS is unavailable).
"""

from __future__ import annotations

import torch


def resolve_sink_token_id(tokenizer) -> int | None:
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return None


def encode_prompt_ids(
    tokenizer,
    prompt_text: str,
    *,
    attention_sink: bool = True,
    device=None,
) -> torch.Tensor:
    """Tokenize prompt_text and (optionally) ensure a sink token at position 0.

    Returns a LongTensor of shape [1, L].
    """

    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    ids = encoded.input_ids
    if attention_sink:
        sink_id = resolve_sink_token_id(tokenizer)
        if sink_id is None:
            raise RuntimeError(
                "attention_sink=True but tokenizer has neither BOS nor EOS available"
            )
        first_id = int(ids[0, 0].item()) if ids.numel() > 0 else None
        if first_id != sink_id:
            sink = torch.tensor([[sink_id]], dtype=ids.dtype)
            ids = torch.cat([sink, ids], dim=1)
    if device is not None:
        ids = ids.to(device)
    return ids


def leading_sink_offset(tokenizer, input_ids: torch.Tensor) -> int:
    """Return 1 if the leading token is the resolved sink token, else 0.

    Accepts either a 1D tensor `[L]` (e.g., after squeezing the batch dim in
    `spans._ensure_1d_ids`) or a 2D tensor `[1, L]` from `encode_prompt_ids`.
    """

    sink_id = resolve_sink_token_id(tokenizer)
    if sink_id is None or input_ids.numel() == 0:
        return 0
    flat = input_ids if input_ids.ndim == 1 else input_ids[0]
    return 1 if int(flat[0].item()) == sink_id else 0
