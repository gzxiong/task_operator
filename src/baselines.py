"""Four ICL-steering baselines: TV, FV, ICV, Conceptor.

All four are training-free and forward-only with frozen weights. Each method
extracts a task representation from K=8 ICL prompts (paired with the model's
own K=8 ICL predictions as pseudo-labels — never validation gold), then injects
that representation into a zero-shot test query at inference time.

  TV (Hendel et al. 2023):
      replace the residual stream at layer L, last prompt position, with a
      mean of last-token residuals from K=8 ICL prompts.

  FV (Todd et al. 2024):
      capture per-attention-head clean means at the last token of K=8 ICL
      prompts; rank heads by Average Indirect Effect (AIE) using NLL on the
      validation pseudo-labels (zero-shot vs. zero-shot-with-clean-head-patched);
      sum the top-|A| projected through their o_proj input slices to form a
      function vector; add into the residual at layer L, last prompt position.

  ICV (Liu et al. 2024):
      forward `x_i` and `y_i` separately, capture last-token residuals at every
      layer, take Δ = h(y) − h(x); top-1 right-singular vector across pairs
      gives an (L, d) steering tensor. Apply via norm-preserving residual add
      at every layer / every position.

  Conceptor (Postmus & Abreu 2024):
      collect last-prompt-token residuals at layer L; build correlation
      R = X^T X / N; closed-form C = R(R + α^-2 I)^-1 via eigendecomposition;
      apply h' = β · C · h at layer L, last prompt position.

For TV, FV (apply layer), and Conceptor, the layer L is searched over
`range(0, n_layers, config.layer_step)` using the same NLL-on-pseudo-labels
metric as `task_operator`'s search. ICV uses every layer by construction; only
λ is tunable (optional grid).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .generation import generate_with_hooks
from .model import get_decoder_layers, get_text_config
from .prompting import DEMO_TEMPLATE, QUERY_TEMPLATE, build_full_icl_prompt, build_zsl_prompt
from .search import _pad_and_stack, _per_sample_nll_from_logits, prepare_sample
from .tokenization import encode_prompt_ids


# ============================================================================
# Configs
# ============================================================================

@dataclass
class _BaselineCommon:
    attention_sink: bool = True
    demo_template: str = DEMO_TEMPLATE
    query_template: str = QUERY_TEMPLATE
    repetition_penalty: float = 1.1
    max_new_tokens: int = 64
    stopping_strings: list[str] | None = None
    answer_phrases: tuple[str, ...] = ()
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    batch_size: int = 8


@dataclass
class TVConfig(_BaselineCommon):
    layer: int | None = None       # None ⇒ run layer search; int ⇒ pin
    layer_step: int = 1            # search grid stride


@dataclass
class FVConfig(_BaselineCommon):
    layer: int | None = None
    layer_step: int = 1
    n_top_heads: int = 20
    n_cie_prompts: int | None = None  # None ⇒ all filtered samples; int ⇒ cap


@dataclass
class ICVConfig(_BaselineCommon):
    scaling_lambda: float = 0.1
    lambda_grid: list[float] | None = None  # None ⇒ no λ search


@dataclass
class ConceptorConfig(_BaselineCommon):
    layer: int | None = None
    layer_step: int = 1
    aperture: float = 1.0
    scaling_beta: float = 1.0


@dataclass
class I2CLConfig(_BaselineCommon):
    """I2CL (Liu et al. NeurIPS 2024, arXiv:2405.14660v2). Defaults from the paper."""

    n_steps: int = 200
    lr: float = 1e-2            # paper §3.2: AdamW with cosine annealing 1e-2 -> lr_min
    lr_min: float = 1e-5        # paper §3.2: cosine annealing end LR
    init_lambda: float = 0.1
    init_beta: float = 1.0
    noise_gamma: float = 0.001  # paper §3.2: empirically τ=0.001
    calibration_seed: int = 0


# ============================================================================
# Hook helpers
# ============================================================================

def _split_layer_output(output):
    """HF decoder layers may return a Tensor or a tuple `(hidden, *rest)`. Return
    `(hidden, rest)` so the rewrap helper can preserve the original structure."""

    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rewrap_layer_output(hidden, rest):
    if rest is None:
        return hidden
    return (hidden, *rest)


def _clear_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def _make_per_sample_replace_hook(per_sample_pos: torch.Tensor, value: torch.Tensor):
    """NLL-eval hook on a decoder layer. On forward passes with seq_len > 1,
    overwrites output[i, per_sample_pos[i], :] with `value` (broadcast across
    samples in the batch)."""

    def hook(module, inputs, output):
        h, rest = _split_layer_output(output)
        if h.shape[1] == 1:
            return output
        h_new = h.clone()
        b = torch.arange(h.shape[0], device=h.device)
        pos = per_sample_pos.to(h.device)
        h_new[b, pos, :] = value.to(h.device, h.dtype)
        return _rewrap_layer_output(h_new, rest)
    return hook


def _make_per_sample_add_hook(per_sample_pos: torch.Tensor, value: torch.Tensor):
    def hook(module, inputs, output):
        h, rest = _split_layer_output(output)
        if h.shape[1] == 1:
            return output
        h_new = h.clone()
        b = torch.arange(h.shape[0], device=h.device)
        pos = per_sample_pos.to(h.device)
        h_new[b, pos, :] = h_new[b, pos, :] + value.to(h.device, h.dtype)
        return _rewrap_layer_output(h_new, rest)
    return hook


def _make_per_sample_matmul_hook(per_sample_pos: torch.Tensor, matrix: torch.Tensor, scale: float):
    """Replace output[i, per_sample_pos[i], :] with `scale * (matrix @ h_at_pos)`."""

    def hook(module, inputs, output):
        h, rest = _split_layer_output(output)
        if h.shape[1] == 1:
            return output
        h_new = h.clone()
        b = torch.arange(h.shape[0], device=h.device)
        pos = per_sample_pos.to(h.device)
        m = matrix.to(h.device, h.dtype)
        h_at = h_new[b, pos, :]
        h_new[b, pos, :] = float(scale) * (h_at @ m.T)
        return _rewrap_layer_output(h_new, rest)
    return hook


def _make_apply_replace_last_hook(value: torch.Tensor):
    """Generation-time hook: prefill-only (seq_len > 1), replaces output[:, -1, :]
    with `value`. On gen steps (seq_len == 1) no-op so the model continues
    normally after the task vector has been injected once."""

    def hook(module, inputs, output):
        h, rest = _split_layer_output(output)
        if h.shape[1] == 1:
            return output
        h_new = h.clone()
        h_new[:, -1, :] = value.to(h.device, h.dtype)
        return _rewrap_layer_output(h_new, rest)
    return hook


def _make_apply_add_last_hook(value: torch.Tensor):
    def hook(module, inputs, output):
        h, rest = _split_layer_output(output)
        if h.shape[1] == 1:
            return output
        h_new = h.clone()
        h_new[:, -1, :] = h_new[:, -1, :] + value.to(h.device, h.dtype)
        return _rewrap_layer_output(h_new, rest)
    return hook


def _make_apply_matmul_last_hook(matrix: torch.Tensor, scale: float):
    def hook(module, inputs, output):
        h, rest = _split_layer_output(output)
        if h.shape[1] == 1:
            return output
        h_new = h.clone()
        m = matrix.to(h.device, h.dtype)
        h_new[:, -1, :] = float(scale) * (h_new[:, -1, :] @ m.T)
        return _rewrap_layer_output(h_new, rest)
    return hook


def _make_icv_hook(value: torch.Tensor, scaling_lambda: float):
    """ICV's norm-preserving add at every layer and every position.

    h_new = h + λ·v;  h_final = h_new · ‖h‖ / ‖h_new‖

    Applied uniformly during both prefill and gen, since ICV is a continuous
    steering signal rather than a one-shot replacement.
    """

    def hook(module, inputs, output):
        h, rest = _split_layer_output(output)
        h_orig_norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = value.to(h.device, h.dtype)
        h_new = h + float(scaling_lambda) * v
        h_new_norm = h_new.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        h_final = h_new * (h_orig_norm / h_new_norm)
        return _rewrap_layer_output(h_final, rest)
    return hook


# ============================================================================
# Per-sample NLL machinery
# ============================================================================

def _eval_batched_nll(
    model,
    tokenizer,
    samples: list[dict],
    *,
    register_hooks_fn: Callable[[torch.Tensor], list],
    batch_size: int = 8,
) -> list[float]:
    """Compute teacher-forced NLL of `samples[i]["target_ids"]` given a forward
    pass on `cat([zsl_ids, target_ids])` with custom hooks registered.

    `register_hooks_fn(per_sample_pos)` is called once per batch with the tensor
    of positions where the intervention should apply (= zsl_prompt_len[i] - 1
    per sample). It returns the list of hook handles to remove after the batch.
    """

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    nlls: list[float] = []
    n = len(samples)
    for bs in range(0, n, batch_size):
        batch = samples[bs : bs + batch_size]
        extended = [torch.cat([s["zsl_ids"], s["target_ids"]], dim=1) for s in batch]
        input_ids, attn_mask = _pad_and_stack(extended, pad_id, device)
        per_sample_pos = torch.tensor(
            [int(s["zsl_prompt_len"]) - 1 for s in batch],
            dtype=torch.long, device=device,
        )
        handles = register_hooks_fn(per_sample_pos)
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids, attention_mask=attn_mask,
                    use_cache=False, output_hidden_states=False, return_dict=True,
                )
            prompt_lens = [s["zsl_prompt_len"] for s in batch]
            target_padded = [s["target_ids"][0] for s in batch]
            target_lens = [s["n_target"] for s in batch]
            batch_nlls = _per_sample_nll_from_logits(out.logits, prompt_lens, target_padded, target_lens)
        finally:
            _clear_hooks(handles)
        nlls.extend(batch_nlls)
        del out
    return nlls


def _eval_zsl_baseline_nll(model, tokenizer, samples, batch_size):
    """Plain zero-shot NLL with no hooks. Used as the FV CIE / AIE baseline and
    as a sanity reference."""

    return _eval_batched_nll(
        model, tokenizer, samples,
        register_hooks_fn=lambda _pos: [],
        batch_size=batch_size,
    )


# ============================================================================
# Sample preparation (shared)
# ============================================================================

def _prepare_samples(model, tokenizer, demos, validation_records, validation_icl_outputs, config):
    device = next(model.parameters()).device
    samples = []
    skipped = 0
    for record, icl_output in zip(validation_records, validation_icl_outputs):
        s = prepare_sample(
            tokenizer, demos, record, icl_output,
            demo_template=config.demo_template, query_template=config.query_template,
            attention_sink=config.attention_sink, device=device,
        )
        if s is None:
            skipped += 1
        else:
            samples.append(s)
    return samples, skipped


def _layer_grid(n_layers: int, layer_step: int) -> list[int]:
    return list(range(0, int(n_layers), max(1, int(layer_step))))


# ============================================================================
# TV
# ============================================================================

def _capture_residual_last_per_layer(model, tokenizer, samples, batch_size):
    """Run forwards on `samples[i]["icl_ids"]` with `output_hidden_states=True`,
    capture the last prompt-token residual at every decoder layer's output, and
    return the per-layer mean across samples as a tensor of shape (n_layers, d).
    """

    device = next(model.parameters()).device
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    hidden_size = int(text_cfg.hidden_size)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    accumulator = torch.zeros(n_layers, hidden_size, dtype=torch.float32)
    n_processed = 0
    for bs in range(0, len(samples), batch_size):
        batch = samples[bs : bs + batch_size]
        icl_ids_list = [s["icl_ids"] for s in batch]
        input_ids, attn_mask = _pad_and_stack(icl_ids_list, pad_id, device)
        per_sample_pos = [int(s["icl_ids"].shape[1]) - 1 for s in batch]
        with torch.inference_mode():
            out = model(
                input_ids=input_ids, attention_mask=attn_mask,
                use_cache=False, output_hidden_states=True, return_dict=True,
            )
        b = torch.arange(len(batch), device=device)
        pos_t = torch.tensor(per_sample_pos, dtype=torch.long, device=device)
        for layer_idx in range(n_layers):
            h = out.hidden_states[layer_idx + 1]   # output of layer `layer_idx`
            last = h[b, pos_t, :].detach().float().cpu()
            accumulator[layer_idx] += last.sum(dim=0)
        n_processed += len(batch)
        del out
    return accumulator / max(n_processed, 1)


def tv_extract(model, tokenizer, demos, validation_records, validation_icl_outputs, config: TVConfig) -> dict:
    """Extract TV vector at one layer L (searched if config.layer is None)."""

    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    samples, skipped = _prepare_samples(
        model, tokenizer, demos, validation_records, validation_icl_outputs, config,
    )
    if not samples:
        raise RuntimeError(f"TV extract: no usable validation samples (skipped={skipped})")

    tv_per_layer = _capture_residual_last_per_layer(model, tokenizer, samples, config.batch_size)
    decoder_layers = get_decoder_layers(model)
    model_dtype = next(model.parameters()).dtype

    per_layer_nll: list[tuple[int, float]] = []
    if config.layer is not None:
        chosen_layer = int(config.layer)
    else:
        for ell in _layer_grid(n_layers, config.layer_step):
            tv_l = tv_per_layer[ell].to(next(model.parameters()).device, dtype=model_dtype)
            def _register(per_sample_pos, _layer=ell, _tv=tv_l):
                hook = _make_per_sample_replace_hook(per_sample_pos, _tv)
                return [decoder_layers[_layer].register_forward_hook(hook)]
            nlls = _eval_batched_nll(
                model, tokenizer, samples,
                register_hooks_fn=_register, batch_size=config.batch_size,
            )
            per_layer_nll.append((int(ell), float(sum(nlls) / len(nlls))))
        chosen_layer = min(per_layer_nll, key=lambda kv: kv[1])[0]

    return {
        "tv_vector": tv_per_layer[chosen_layer].clone(),
        "tv_per_layer": tv_per_layer,
        "layer_used": int(chosen_layer),
        "per_layer_nll": per_layer_nll,
        "n_layers": int(n_layers),
        "n_val_samples": int(len(samples)),
        "n_val_skipped": int(skipped),
    }


def tv_apply(model, tokenizer, test_record, repr_dict: dict, config: TVConfig) -> dict:
    decoder_layers = get_decoder_layers(model)
    layer_idx = int(repr_dict["layer_used"])
    tv = repr_dict["tv_vector"]
    handle = decoder_layers[layer_idx].register_forward_hook(_make_apply_replace_last_hook(tv))
    try:
        zsl_prompt = build_zsl_prompt(test_record["input"], query_template=config.query_template)
        payload = generate_with_hooks(
            model, tokenizer, zsl_prompt,
            prompt_records=None, gen_records=None,
            attention_sink=config.attention_sink,
            max_new_tokens=config.max_new_tokens,
            stop_strings=config.stopping_strings,
            answer_phrases=config.answer_phrases,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
    finally:
        try:
            handle.remove()
        except Exception:
            pass
    return payload


# ============================================================================
# FV
# ============================================================================

def _capture_per_head_clean_means(model, tokenizer, samples, batch_size):
    """Capture per-head outputs at the last token of each ICL prompt (averaged
    across samples). Returns tensor of shape (n_layers, n_heads, head_dim).

    The per-head pre-projection signal lives in `o_proj.inputs[0]`, the
    concatenated heads of shape (B, T, n_heads * head_dim).
    """

    device = next(model.parameters()).device
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    n_heads = int(text_cfg.num_attention_heads)
    head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // n_heads))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    accumulator = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float32)
    n_processed = 0
    decoder_layers = get_decoder_layers(model)
    for bs in range(0, len(samples), batch_size):
        batch = samples[bs : bs + batch_size]
        icl_ids_list = [s["icl_ids"] for s in batch]
        input_ids, attn_mask = _pad_and_stack(icl_ids_list, pad_id, device)
        per_sample_pos = torch.tensor(
            [int(s["icl_ids"].shape[1]) - 1 for s in batch],
            dtype=torch.long, device=device,
        )
        captured: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                x = inputs[0]
                if x.shape[1] == 1:
                    return output
                b = torch.arange(x.shape[0], device=x.device)
                last = x[b, per_sample_pos.to(x.device), :].detach()
                captured[layer_idx] = last.view(x.shape[0], n_heads, head_dim)
                return output
            return hook

        handles = []
        for layer_idx, layer in enumerate(decoder_layers):
            handles.append(layer.self_attn.o_proj.register_forward_hook(make_hook(layer_idx)))
        try:
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, return_dict=True)
        finally:
            _clear_hooks(handles)
        for layer_idx in range(n_layers):
            accumulator[layer_idx] += captured[layer_idx].sum(dim=0).float().cpu()
        n_processed += len(batch)
    return accumulator / max(n_processed, 1)


def _fv_cie_one_head_nll(
    model, tokenizer, samples,
    layer_idx: int, head_idx: int, head_value: torch.Tensor,
    n_heads: int, head_dim: int, batch_size: int,
) -> list[float]:
    """Patched zero-shot NLL for one (layer, head) pair: pre-hook the layer's
    `o_proj` to overwrite inputs[0][i, last_pos[i], j*hd:(j+1)*hd] with the
    captured clean mean for head j, then teacher-force the validation pseudo
    target. Returns per-sample NLLs."""

    decoder_layers = get_decoder_layers(model)
    slot_start = head_idx * head_dim
    slot_end = (head_idx + 1) * head_dim

    def register(per_sample_pos):
        def pre_hook(module, inputs):
            x = inputs[0]
            if x.shape[1] == 1:
                return None
            x_new = x.clone()
            b = torch.arange(x.shape[0], device=x.device)
            pos = per_sample_pos.to(x.device)
            x_new[b, pos, slot_start:slot_end] = head_value.to(x.device, x.dtype)
            return (x_new, *inputs[1:])
        return [decoder_layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(pre_hook)]

    return _eval_batched_nll(
        model, tokenizer, samples,
        register_hooks_fn=register, batch_size=batch_size,
    )


def _fv_build_vector(model, head_clean_mean: torch.Tensor, top_heads: list[tuple[int, int]]) -> torch.Tensor:
    """FV = sum_{(ℓ,j) ∈ top_heads} head_clean_mean[ℓ,j,:] @ W_o[ℓ][:, j*hd:(j+1)*hd].T"""

    text_cfg = get_text_config(model)
    n_heads = int(text_cfg.num_attention_heads)
    head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // n_heads))
    hidden_size = int(text_cfg.hidden_size)
    decoder_layers = get_decoder_layers(model)
    fv = torch.zeros(hidden_size, dtype=torch.float32)
    for layer_idx, head_idx in top_heads:
        W = decoder_layers[layer_idx].self_attn.o_proj.weight  # (hidden_size, n_heads*head_dim)
        W_slice = W[:, head_idx * head_dim : (head_idx + 1) * head_dim].detach().float().cpu()
        v = head_clean_mean[layer_idx, head_idx, :].float().cpu()
        fv += v @ W_slice.T
    return fv


def fv_extract(model, tokenizer, demos, validation_records, validation_icl_outputs, config: FVConfig) -> dict:
    """Full paper-faithful FV with NLL-based AIE (no gold labels)."""

    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    n_heads = int(text_cfg.num_attention_heads)
    head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // n_heads))

    samples, skipped = _prepare_samples(
        model, tokenizer, demos, validation_records, validation_icl_outputs, config,
    )
    if not samples:
        raise RuntimeError(f"FV extract: no usable validation samples (skipped={skipped})")
    cie_samples = (
        samples if config.n_cie_prompts is None
        else samples[: int(config.n_cie_prompts)]
    )

    # 1) Per-head clean means at last position of each ICL prompt.
    head_clean_mean = _capture_per_head_clean_means(model, tokenizer, cie_samples, config.batch_size)

    # 2) Baseline (un-patched) zero-shot NLLs, one per validation sample.
    base_nlls = _eval_zsl_baseline_nll(model, tokenizer, cie_samples, config.batch_size)
    base_arr = torch.tensor(base_nlls, dtype=torch.float32)

    # 3) For each (ℓ, j): patched NLL → CIE_per_sample → AIE.
    aie_table = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    model_dtype = next(model.parameters()).dtype
    for ell in range(n_layers):
        for j in range(n_heads):
            head_val = head_clean_mean[ell, j, :].to(next(model.parameters()).device, dtype=model_dtype)
            patched = _fv_cie_one_head_nll(
                model, tokenizer, cie_samples,
                ell, j, head_val, n_heads, head_dim, config.batch_size,
            )
            patched_arr = torch.tensor(patched, dtype=torch.float32)
            cie = base_arr - patched_arr
            aie_table[ell, j] = cie.mean()

    # 4) Top-|A| heads by AIE.
    n_top = int(max(1, min(config.n_top_heads, n_layers * n_heads)))
    flat = aie_table.flatten()
    top_idx = torch.topk(flat, n_top).indices.tolist()
    top_heads = [(int(idx // n_heads), int(idx % n_heads)) for idx in top_idx]

    # 5) Build FV vector.
    fv_vector = _fv_build_vector(model, head_clean_mean, top_heads)

    # 6) Apply-layer search (or pin).
    decoder_layers = get_decoder_layers(model)
    per_layer_nll: list[tuple[int, float]] = []
    if config.layer is not None:
        chosen_layer = int(config.layer)
    else:
        fv_dev = fv_vector.to(next(model.parameters()).device, dtype=model_dtype)
        for ell in _layer_grid(n_layers, config.layer_step):
            def _register(per_sample_pos, _layer=ell, _v=fv_dev):
                hook = _make_per_sample_add_hook(per_sample_pos, _v)
                return [decoder_layers[_layer].register_forward_hook(hook)]
            nlls = _eval_batched_nll(
                model, tokenizer, samples,
                register_hooks_fn=_register, batch_size=config.batch_size,
            )
            per_layer_nll.append((int(ell), float(sum(nlls) / len(nlls))))
        chosen_layer = min(per_layer_nll, key=lambda kv: kv[1])[0]

    return {
        "fv_vector": fv_vector,
        "head_clean_mean": head_clean_mean,
        "aie_table": aie_table,
        "top_heads": top_heads,
        "n_top_heads": int(n_top),
        "layer_used": int(chosen_layer),
        "per_layer_nll": per_layer_nll,
        "base_nll_mean": float(sum(base_nlls) / max(len(base_nlls), 1)),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "head_dim": int(head_dim),
        "n_val_samples": int(len(samples)),
        "n_cie_samples": int(len(cie_samples)),
        "n_val_skipped": int(skipped),
    }


def fv_apply(model, tokenizer, test_record, repr_dict: dict, config: FVConfig) -> dict:
    decoder_layers = get_decoder_layers(model)
    layer_idx = int(repr_dict["layer_used"])
    fv = repr_dict["fv_vector"]
    handle = decoder_layers[layer_idx].register_forward_hook(_make_apply_add_last_hook(fv))
    try:
        zsl_prompt = build_zsl_prompt(test_record["input"], query_template=config.query_template)
        payload = generate_with_hooks(
            model, tokenizer, zsl_prompt,
            prompt_records=None, gen_records=None,
            attention_sink=config.attention_sink,
            max_new_tokens=config.max_new_tokens,
            stop_strings=config.stopping_strings,
            answer_phrases=config.answer_phrases,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
    finally:
        try:
            handle.remove()
        except Exception:
            pass
    return payload


# ============================================================================
# ICV
# ============================================================================

def _capture_last_token_residuals_all_layers(model, tokenizer, text: str, *, attention_sink: bool):
    """Tokenize `text`, run forward with `output_hidden_states=True`, and return
    the last-token residual at every layer as a tensor of shape (n_layers, d).
    """

    device = next(model.parameters()).device
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    ids = encode_prompt_ids(tokenizer, text or " ", attention_sink=attention_sink, device=device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=True)
    h = torch.stack(
        [out.hidden_states[layer_idx + 1][0, -1, :].detach().float().cpu() for layer_idx in range(n_layers)],
        dim=0,
    )
    del out
    return h


def icv_extract(model, tokenizer, demos, validation_records, validation_icl_outputs, config: ICVConfig) -> dict:
    """ICV: per-pair Δh = h(y) − h(x), top-1 right-singular vector across pairs.

    Per the paper, x and y are run as SEPARATE forwards on raw strings, NOT
    concatenated into an ICL prompt. We use:
      x_i = record["input"]
      y_i = the model's K=8 ICL prediction for that record (icl_output_text).
    """

    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    hidden_size = int(text_cfg.hidden_size)

    deltas: list[torch.Tensor] = []
    skipped = 0
    for record, icl_output in zip(validation_records, validation_icl_outputs):
        x_text = record.get("input") or ""
        y_text = icl_output or ""
        if not x_text.strip() or not y_text.strip():
            skipped += 1
            continue
        h_x = _capture_last_token_residuals_all_layers(
            model, tokenizer, x_text, attention_sink=config.attention_sink,
        )  # (n_layers, d)
        h_y = _capture_last_token_residuals_all_layers(
            model, tokenizer, y_text, attention_sink=config.attention_sink,
        )
        deltas.append((h_y - h_x).flatten())  # (n_layers * d,)
    if len(deltas) < 2:
        raise RuntimeError(f"ICV extract: need >=2 pairs but got {len(deltas)} (skipped={skipped})")

    M = torch.stack(deltas, dim=0)  # (k, L*d)
    # Top-1 right-singular vector of M: M = U S V^T → V[:, 0] is the principal direction.
    _, _, V = torch.svd_lowrank(M, q=1)
    icv_flat = V[:, 0]  # (L*d,)
    # Sign convention: align with mean Δh so the sign is task-meaningful.
    mean_delta = M.mean(dim=0)
    if torch.dot(icv_flat, mean_delta) < 0:
        icv_flat = -icv_flat
    icv = icv_flat.reshape(n_layers, hidden_size).contiguous()

    # Optional λ search via NLL on prepared samples.
    samples, samples_skipped = _prepare_samples(
        model, tokenizer, demos, validation_records, validation_icl_outputs, config,
    )
    decoder_layers = get_decoder_layers(model)
    model_dtype = next(model.parameters()).dtype
    per_lambda_nll: list[tuple[float, float]] = []
    if config.lambda_grid:
        for lam in list(config.lambda_grid):
            def _register(per_sample_pos, _lam=float(lam)):
                handles = []
                for layer_idx, layer in enumerate(decoder_layers):
                    v = icv[layer_idx].to(next(model.parameters()).device, dtype=model_dtype)
                    handles.append(layer.register_forward_hook(_make_icv_hook(v, _lam)))
                return handles
            nlls = _eval_batched_nll(
                model, tokenizer, samples,
                register_hooks_fn=_register, batch_size=config.batch_size,
            )
            per_lambda_nll.append((float(lam), float(sum(nlls) / max(len(nlls), 1))))
        chosen_lambda = min(per_lambda_nll, key=lambda kv: kv[1])[0]
    else:
        chosen_lambda = float(config.scaling_lambda)

    return {
        "icv": icv,
        "scaling_lambda": float(chosen_lambda),
        "per_lambda_nll": per_lambda_nll,
        "n_layers": int(n_layers),
        "hidden_size": int(hidden_size),
        "n_val_samples": int(len(deltas)),
        "n_val_skipped": int(skipped),
        "n_nll_samples": int(len(samples)),
        "n_nll_skipped": int(samples_skipped),
    }


def icv_apply(model, tokenizer, test_record, repr_dict: dict, config: ICVConfig) -> dict:
    decoder_layers = get_decoder_layers(model)
    icv = repr_dict["icv"]
    lam = float(repr_dict.get("scaling_lambda", config.scaling_lambda))
    handles = []
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    for layer_idx, layer in enumerate(decoder_layers):
        v = icv[layer_idx].to(device, dtype=model_dtype)
        handles.append(layer.register_forward_hook(_make_icv_hook(v, lam)))
    try:
        zsl_prompt = build_zsl_prompt(test_record["input"], query_template=config.query_template)
        payload = generate_with_hooks(
            model, tokenizer, zsl_prompt,
            prompt_records=None, gen_records=None,
            attention_sink=config.attention_sink,
            max_new_tokens=config.max_new_tokens,
            stop_strings=config.stopping_strings,
            answer_phrases=config.answer_phrases,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
    finally:
        _clear_hooks(handles)
    return payload


# ============================================================================
# Conceptor
# ============================================================================

def conceptor_extract(model, tokenizer, demos, validation_records, validation_icl_outputs, config: ConceptorConfig) -> dict:
    """Per-layer Conceptor matrices: build C[ℓ] = R[ℓ](R[ℓ] + α^-2 I)^-1 via
    eigendecomposition, layer-search by NLL, persist only C[L*]."""

    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    hidden_size = int(text_cfg.hidden_size)
    samples, skipped = _prepare_samples(
        model, tokenizer, demos, validation_records, validation_icl_outputs, config,
    )
    if not samples:
        raise RuntimeError(f"Conceptor extract: no usable validation samples (skipped={skipped})")

    # Capture last-token residuals at all layers via output_hidden_states=True.
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    layer_X: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    for bs in range(0, len(samples), config.batch_size):
        batch = samples[bs : bs + config.batch_size]
        icl_ids_list = [s["icl_ids"] for s in batch]
        input_ids, attn_mask = _pad_and_stack(icl_ids_list, pad_id, device)
        per_sample_pos = [int(s["icl_ids"].shape[1]) - 1 for s in batch]
        with torch.inference_mode():
            out = model(
                input_ids=input_ids, attention_mask=attn_mask,
                use_cache=False, output_hidden_states=True, return_dict=True,
            )
        b = torch.arange(len(batch), device=device)
        pos_t = torch.tensor(per_sample_pos, dtype=torch.long, device=device)
        for layer_idx in range(n_layers):
            h = out.hidden_states[layer_idx + 1]
            layer_X[layer_idx].append(h[b, pos_t, :].detach().float().cpu())
        del out

    candidates = _layer_grid(n_layers, config.layer_step) if config.layer is None else [int(config.layer)]
    alpha = float(config.aperture)
    inv_a2 = 1.0 / (alpha ** 2)
    C_per_layer: dict[int, torch.Tensor] = {}
    for layer_idx in candidates:
        X = torch.cat(layer_X[layer_idx], dim=0)  # (N, d)
        N = int(X.shape[0])
        R = X.T @ X / max(N, 1)
        # Symmetrize for numerical stability before eigh.
        R = 0.5 * (R + R.T)
        eigvals, U = torch.linalg.eigh(R)
        mu = eigvals / (eigvals + inv_a2)
        C = (U * mu.unsqueeze(0)) @ U.T  # = U @ diag(mu) @ U.T
        C_per_layer[int(layer_idx)] = C.contiguous()

    decoder_layers = get_decoder_layers(model)
    model_dtype = next(model.parameters()).dtype
    per_layer_nll: list[tuple[int, float]] = []
    if config.layer is not None:
        chosen_layer = int(config.layer)
    else:
        for ell in candidates:
            C_l = C_per_layer[ell].to(next(model.parameters()).device, dtype=model_dtype)
            scale = float(config.scaling_beta)
            def _register(per_sample_pos, _layer=ell, _C=C_l, _scale=scale):
                hook = _make_per_sample_matmul_hook(per_sample_pos, _C, _scale)
                return [decoder_layers[_layer].register_forward_hook(hook)]
            nlls = _eval_batched_nll(
                model, tokenizer, samples,
                register_hooks_fn=_register, batch_size=config.batch_size,
            )
            per_layer_nll.append((int(ell), float(sum(nlls) / len(nlls))))
        chosen_layer = min(per_layer_nll, key=lambda kv: kv[1])[0]

    chosen_C = C_per_layer[int(chosen_layer)]
    return {
        "conceptor": chosen_C,
        "layer_used": int(chosen_layer),
        "aperture": float(alpha),
        "scaling_beta": float(config.scaling_beta),
        "per_layer_nll": per_layer_nll,
        "n_layers": int(n_layers),
        "hidden_size": int(hidden_size),
        "n_val_samples": int(len(samples)),
        "n_val_skipped": int(skipped),
    }


def conceptor_apply(model, tokenizer, test_record, repr_dict: dict, config: ConceptorConfig) -> dict:
    decoder_layers = get_decoder_layers(model)
    layer_idx = int(repr_dict["layer_used"])
    C = repr_dict["conceptor"]
    scale = float(repr_dict.get("scaling_beta", config.scaling_beta))
    handle = decoder_layers[layer_idx].register_forward_hook(_make_apply_matmul_last_hook(C, scale))
    try:
        zsl_prompt = build_zsl_prompt(test_record["input"], query_template=config.query_template)
        payload = generate_with_hooks(
            model, tokenizer, zsl_prompt,
            prompt_records=None, gen_records=None,
            attention_sink=config.attention_sink,
            max_new_tokens=config.max_new_tokens,
            stop_strings=config.stopping_strings,
            answer_phrases=config.answer_phrases,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
    finally:
        try:
            handle.remove()
        except Exception:
            pass
    return payload


# ============================================================================
# I2CL (Liu et al. NeurIPS 2024)
# ============================================================================
# Vectorize each demo individually (per-layer attention output and MLP output at
# the last token), aggregate cross-demo means, then calibrate 4·L scalars
# (λ^a, β^a, λ^m, β^m) by Adam on teacher-forced demo-label NLL. At apply time
# the same scalars blend the captured means with the query's natural module
# outputs at every layer × every position.

def _make_i2cl_blend_capture_hook(captured: dict, key: str):
    """Capture last-position output of a module on a B=1 forward (used during
    extraction). Stored on GPU."""

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[key] = h[0, -1, :].detach()
        return output
    return hook


def _make_i2cl_blend_hook_param(v: torch.Tensor, lam: torch.Tensor, beta: torch.Tensor):
    """Forward post-hook: out_new = lam * v + beta * out, broadcast over (B, T, d).

    `lam` and `beta` are torch tensors so the hook participates in the autograd
    graph during calibration. At apply time these are passed as plain tensors
    (no grad) and the math is identical."""

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        v_dev = v.to(h.device, h.dtype)
        l_dev = lam.to(h.device, h.dtype)
        b_dev = beta.to(h.device, h.dtype)
        h_new = l_dev * v_dev + b_dev * h
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new
    return hook


def _make_i2cl_noise_hook(gamma: float):
    """Calibration-only hook on a decoder layer's residual-stream output. Adds
    Gaussian noise `gamma * ||o||_2 * eta` per token, η ~ N(0, I)."""

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        eta = torch.randn_like(h)
        h_new = h + float(gamma) * norm * eta
        if isinstance(output, tuple):
            return (h_new, *output[1:])
        return h_new
    return hook


def _i2cl_capture_demo_means(model, tokenizer, demos, config: I2CLConfig):
    """Run K single-demo forwards; return (a_mean, m_mean) of shape (L, hidden)
    on the model's device.

    Paper-faithful capture: the position is the last token of the FULL demo
    `Input: x \\nOutput: y` (after stripping trailing whitespace) — i.e., the
    position immediately *after* the model has produced the answer.
    """

    device = next(model.parameters()).device
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    hidden = int(text_cfg.hidden_size)
    decoder_layers = get_decoder_layers(model)

    demos_K = list(demos)
    if not demos_K:
        raise RuntimeError("I2CL extract: empty demo list")

    a_sum = torch.zeros(n_layers, hidden, dtype=torch.float32, device=device)
    m_sum = torch.zeros(n_layers, hidden, dtype=torch.float32, device=device)
    n_used = 0
    for d in demos_K:
        x_i = d.get("input")
        y_i = d.get("output") or d.get("target")
        if not x_i or not y_i:
            continue
        prompt = config.demo_template.format(input=x_i, output=y_i).strip()
        ids = encode_prompt_ids(tokenizer, prompt, attention_sink=config.attention_sink, device=device)

        captured: dict = {}
        handles = []
        for ell, layer in enumerate(decoder_layers):
            handles.append(layer.self_attn.o_proj.register_forward_hook(
                _make_i2cl_blend_capture_hook(captured, ("a", ell))
            ))
            handles.append(layer.mlp.register_forward_hook(
                _make_i2cl_blend_capture_hook(captured, ("m", ell))
            ))
        try:
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False, return_dict=True)
        finally:
            _clear_hooks(handles)

        for ell in range(n_layers):
            a_sum[ell] += captured[("a", ell)].float()
            m_sum[ell] += captured[("m", ell)].float()
        n_used += 1

    if n_used == 0:
        raise RuntimeError("I2CL extract: no usable demos")
    a_mean = a_sum / n_used
    m_mean = m_sum / n_used
    return a_mean, m_mean


def _i2cl_calibrate(
    model, tokenizer, demos, a_mean, m_mean, config: I2CLConfig,
):
    """Adam on the 4·L scalars by minimising teacher-forced NLL of demo labels."""

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    decoder_layers = get_decoder_layers(model)

    # Build per-demo (prompt_ids, target_ids) pairs. Target tokens are derived
    # from the natural tokenization of y *as it appears in the demo* (i.e., the
    # full demo's tokens minus the prefix's tokens) rather than from `tokenizer(y)`
    # standalone — this avoids the leading-space token mismatch that arises when y
    # is tokenized without surrounding context.
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for d in demos:
        x_i = d.get("input")
        y_i = d.get("output") or d.get("target")
        if not x_i or not y_i:
            continue
        prompt = build_zsl_prompt(x_i, query_template=config.query_template)
        prompt_ids = encode_prompt_ids(tokenizer, prompt, attention_sink=config.attention_sink, device=device)
        full_demo_text = config.demo_template.format(input=x_i, output=y_i).strip()
        full_ids = encode_prompt_ids(tokenizer, full_demo_text, attention_sink=config.attention_sink, device=device)
        prefix_len = int(prompt_ids.shape[1])
        if int(full_ids.shape[1]) <= prefix_len:
            continue
        target_ids = full_ids[:, prefix_len:]
        samples.append((prompt_ids, target_ids))
    if not samples:
        raise RuntimeError("I2CL calibration: no usable demos")

    # Freeze model weights but allow activations to carry grad to our scalars.
    model.eval()
    prev_requires_grad = {}
    for n, p in model.named_parameters():
        prev_requires_grad[n] = p.requires_grad
        p.requires_grad_(False)

    try:
        # 4 vectors of L scalars each, on device, in float32 for stable optim.
        la = torch.full((n_layers,), float(config.init_lambda), device=device, dtype=torch.float32, requires_grad=True)
        ba = torch.full((n_layers,), float(config.init_beta),   device=device, dtype=torch.float32, requires_grad=True)
        lm = torch.full((n_layers,), float(config.init_lambda), device=device, dtype=torch.float32, requires_grad=True)
        bm = torch.full((n_layers,), float(config.init_beta),   device=device, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.AdamW([la, ba, lm, bm], lr=float(config.lr))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(config.n_steps), eta_min=float(config.lr_min),
        )

        gen = torch.Generator(device="cpu").manual_seed(int(config.calibration_seed))
        loss_trace: list[float] = []

        for step in range(int(config.n_steps)):
            # Round-robin through demos with periodic shuffle.
            if step % len(samples) == 0:
                perm = torch.randperm(len(samples), generator=gen).tolist()
            i = perm[step % len(samples)]
            prompt_ids, target_ids = samples[i]
            extended = torch.cat([prompt_ids, target_ids], dim=1)

            # Register hooks: blend on each o_proj + each mlp; noise on each layer output.
            handles = []
            for ell, layer in enumerate(decoder_layers):
                handles.append(layer.self_attn.o_proj.register_forward_hook(
                    _make_i2cl_blend_hook_param(a_mean[ell], la[ell], ba[ell])
                ))
                handles.append(layer.mlp.register_forward_hook(
                    _make_i2cl_blend_hook_param(m_mean[ell], lm[ell], bm[ell])
                ))
                if float(config.noise_gamma) > 0:
                    handles.append(layer.register_forward_hook(_make_i2cl_noise_hook(config.noise_gamma)))

            try:
                opt.zero_grad(set_to_none=True)
                # Need grads — do NOT use inference_mode.
                out = model(input_ids=extended, use_cache=False, return_dict=True)
                # Teacher-forced NLL of target tokens.
                logits = out.logits[0]                 # (T_total, V)
                prompt_len = int(prompt_ids.shape[1])
                n_target = int(target_ids.shape[1])
                slice_logits = logits[prompt_len - 1 : prompt_len - 1 + n_target, :].float()
                log_probs = torch.nn.functional.log_softmax(slice_logits, dim=-1)
                tgt = target_ids[0]
                loss = -log_probs.gather(1, tgt.unsqueeze(-1)).squeeze(-1).mean()
                loss.backward()
                opt.step()
                sched.step()
                loss_trace.append(float(loss.item()))
            finally:
                _clear_hooks(handles)
                del out

        return la.detach(), ba.detach(), lm.detach(), bm.detach(), loss_trace
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(prev_requires_grad.get(n, False))


def i2cl_extract(model, tokenizer, demos, validation_records, validation_icl_outputs, config: I2CLConfig) -> dict:
    """Vectorize K demos into (a_mean, m_mean), then calibrate 4·L blend scalars
    on the same demos. `validation_records` / `validation_icl_outputs` are accepted
    for dispatcher signature compatibility but unused — paper-faithful I2CL trains
    on the demo set only.
    """

    text_cfg = get_text_config(model)
    n_layers = int(text_cfg.num_hidden_layers)
    hidden = int(text_cfg.hidden_size)

    a_mean, m_mean = _i2cl_capture_demo_means(model, tokenizer, demos, config)
    la, ba, lm, bm, loss_trace = _i2cl_calibrate(model, tokenizer, demos, a_mean, m_mean, config)

    return {
        "a_mean":     a_mean.detach().cpu(),
        "m_mean":     m_mean.detach().cpu(),
        "lambdas_a":  la.cpu(),
        "betas_a":    ba.cpu(),
        "lambdas_m":  lm.cpu(),
        "betas_m":    bm.cpu(),
        "loss_trace": loss_trace,
        "n_layers":   int(n_layers),
        "hidden_size": int(hidden),
        "n_demos":    int(len(demos)),
    }


def i2cl_apply(model, tokenizer, test_record, repr_dict: dict, config: I2CLConfig) -> dict:
    decoder_layers = get_decoder_layers(model)
    a_mean = repr_dict["a_mean"]
    m_mean = repr_dict["m_mean"]
    la = repr_dict["lambdas_a"]
    ba = repr_dict["betas_a"]
    lm = repr_dict["lambdas_m"]
    bm = repr_dict["betas_m"]

    handles = []
    for ell, layer in enumerate(decoder_layers):
        handles.append(layer.self_attn.o_proj.register_forward_hook(
            _make_i2cl_blend_hook_param(a_mean[ell], la[ell], ba[ell])
        ))
        handles.append(layer.mlp.register_forward_hook(
            _make_i2cl_blend_hook_param(m_mean[ell], lm[ell], bm[ell])
        ))
    try:
        zsl_prompt = build_zsl_prompt(test_record["input"], query_template=config.query_template)
        payload = generate_with_hooks(
            model, tokenizer, zsl_prompt,
            prompt_records=None, gen_records=None,
            attention_sink=config.attention_sink,
            max_new_tokens=config.max_new_tokens,
            stop_strings=config.stopping_strings,
            answer_phrases=config.answer_phrases,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
    finally:
        _clear_hooks(handles)
    return payload


# ============================================================================
# Dispatcher
# ============================================================================

_EXTRACT = {
    "TV": tv_extract,
    "FV": fv_extract,
    "ICV": icv_extract,
    "Conceptor": conceptor_extract,
    "I2CL": i2cl_extract,
}
_APPLY = {
    "TV": tv_apply,
    "FV": fv_apply,
    "ICV": icv_apply,
    "Conceptor": conceptor_apply,
    "I2CL": i2cl_apply,
}


def extract_baseline(name: str, model, tokenizer, demos, validation_records, validation_icl_outputs, config) -> dict:
    if name not in _EXTRACT:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(_EXTRACT)}")
    return _EXTRACT[name](model, tokenizer, demos, validation_records, validation_icl_outputs, config)


def apply_baseline(name: str, model, tokenizer, test_record, repr_dict: dict, config) -> dict:
    if name not in _APPLY:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(_APPLY)}")
    return _APPLY[name](model, tokenizer, test_record, repr_dict, config)
