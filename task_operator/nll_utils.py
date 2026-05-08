"""NLL evaluation utilities shared by baselines and site-importance analysis.

Three small primitives:

  prepare_sample              — tokenize one (record, icl_output_text) pair into
                                the structures needed for batched NLL eval (icl/zsl
                                ids, target ids, position spans).
  pad_and_stack               — right-pad a list of [1, L_i] token tensors into a
                                single [B, max(L_i)] batch + attention mask.
  per_sample_nll_from_logits  — given output logits and per-sample prompt/target
                                lengths, compute teacher-forced mean NLL on the
                                target tokens (one float per sample).
  make_oproj_recon_hook       — forward hook on decoder layer `self_attn.o_proj`
                                that overwrites `output[b, p, :]` with the affine
                                reconstruction `(x[b, p] * scale_w + ch_flat) @ W_o^T`.
                                Used by `nll_site_analysis` to replay the task
                                operator on selected sites for kill-one-out NLL.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .prompting import build_full_icl_prompt, build_zsl_prompt
from .spans import identify_context_query, resolve_query_positions
from .tokenization import encode_prompt_ids, leading_sink_offset


def prepare_sample(
    tokenizer,
    demos,
    record,
    icl_output_text,
    *,
    demo_template,
    query_template,
    attention_sink,
    device,
) -> dict | None:
    """Build per-sample objects used during NLL evaluation. Returns None on bad samples."""

    if not icl_output_text:
        return None
    target_token_ids = tokenizer(icl_output_text, add_special_tokens=False).input_ids
    if not target_token_ids:
        return None
    target_ids = torch.tensor([target_token_ids], device=device)
    n_target = int(target_ids.shape[1])

    icl_prompt = build_full_icl_prompt(
        demos, record["input"],
        demo_template=demo_template, query_template=query_template,
    )
    zsl_prompt = build_zsl_prompt(record["input"], query_template=query_template)
    icl_ids = encode_prompt_ids(tokenizer, icl_prompt, attention_sink=attention_sink, device=device)
    zsl_ids = encode_prompt_ids(tokenizer, zsl_prompt, attention_sink=attention_sink, device=device)

    context_span, query_span = identify_context_query(
        tokenizer, icl_ids, zsl_prompt, prompt_text=icl_prompt,
    )
    resolved = resolve_query_positions(
        tokenizer, icl_ids, query_span, record["input"], prompt_text=icl_prompt,
    )
    template_positions = list(resolved["template_positions"])
    content_positions = list(resolved["content_positions"])

    icl_prompt_len = int(icl_ids.shape[1])
    zsl_prompt_len = int(zsl_ids.shape[1])
    icl_target_positions = list(range(icl_prompt_len, icl_prompt_len + n_target))
    zsl_target_positions = list(range(zsl_prompt_len, zsl_prompt_len + n_target))

    n_q_icl = int(query_span[1]) - int(query_span[0])
    sink_offset = leading_sink_offset(tokenizer, zsl_ids)
    n_q_zsl = zsl_prompt_len - sink_offset
    if n_q_icl != n_q_zsl:
        return None

    return {
        "icl_ids": icl_ids,
        "zsl_ids": zsl_ids,
        "target_ids": target_ids,
        "icl_prompt_len": icl_prompt_len,
        "zsl_prompt_len": zsl_prompt_len,
        "n_target": n_target,
        "template_positions": template_positions,
        "content_positions": content_positions,
        "icl_target_positions": icl_target_positions,
        "zsl_target_positions": zsl_target_positions,
        "context_span": (int(context_span[0]), int(context_span[1])),
        "query_span": (int(query_span[0]), int(query_span[1])),
        "sink_offset": int(sink_offset),
    }


def pad_and_stack(tensor_list, pad_value, device):
    """Right-pad a list of [1, L_i] token tensors to [B, max(L_i)] + attention mask."""

    L_max = max(int(t.shape[1]) for t in tensor_list)
    out = torch.full(
        (len(tensor_list), L_max), pad_value,
        dtype=tensor_list[0].dtype, device=device,
    )
    attn_mask = torch.zeros((len(tensor_list), L_max), dtype=torch.long, device=device)
    for i, t in enumerate(tensor_list):
        L = int(t.shape[1])
        out[i, :L] = t[0, :L]
        attn_mask[i, :L] = 1
    return out, attn_mask


def per_sample_nll_from_logits(logits, prompt_lens, target_ids_padded, target_lens) -> list[float]:
    """Teacher-forced mean NLL over target tokens, one float per sample.

    `logits[i, prompt_lens[i] - 1 : prompt_lens[i] - 1 + target_lens[i], :]` are the
    logits that predict the target tokens of sample i. Mean log-prob across target
    tokens, negated.
    """

    out = []
    for i in range(logits.shape[0]):
        n = int(target_lens[i])
        start = int(prompt_lens[i]) - 1
        slice_logits = logits[i, start : start + n, :].float()
        log_probs = F.log_softmax(slice_logits, dim=-1)
        target = target_ids_padded[i].to(log_probs.device, dtype=torch.long)
        nll_per_token = -log_probs.gather(1, target.unsqueeze(-1)).squeeze(-1)
        out.append(float(nll_per_token.mean().item()))
    return out


def zsl_offset(sample, position_kind: str) -> int:
    """Position offset to translate ICL-prompt positions into the ZSL+target
    forward used for NLL eval. `position_kind` is one of {"query", "output"}."""

    if position_kind == "query":
        return sample["sink_offset"] - sample["query_span"][0]
    if position_kind == "output":
        return sample["zsl_prompt_len"] - sample["icl_prompt_len"]
    raise ValueError(position_kind)


def zsl_positions_for_record(sample: dict, rec: dict) -> list[int]:
    """ZSL positions where `rec` (a `mean_circuit` record) fires for `sample`.
    Same mapping `core._build_prompt_records` / `_build_gen_records` use, but
    for the combined `cat([zsl_ids, target_ids])` forward used in NLL eval."""

    pt = rec["position_type"]
    off_q = zsl_offset(sample, "query")
    off_o = zsl_offset(sample, "output")
    if pt == "template":
        rank = int(rec["rank"])
        if 0 <= rank < len(sample["template_positions"]):
            return [int(sample["template_positions"][rank] + off_q)]
        return []
    if pt == "content":
        if rec.get("broadcast"):
            return [int(p + off_q) for p in sample["content_positions"]]
        rank = int(rec["rank"])
        if 0 <= rank < len(sample["content_positions"]):
            return [int(sample["content_positions"][rank] + off_q)]
        return []
    if pt == "output":
        if rec.get("broadcast"):
            return [int(p + off_o) for p in sample["icl_target_positions"]]
        rank = int(rec["rank"])
        if 0 <= rank < len(sample["icl_target_positions"]):
            return [int(sample["icl_target_positions"][rank] + off_o)]
        return []
    raise ValueError(pt)


def make_oproj_recon_hook(
    batch_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    w_stack: torch.Tensor,
    ch_stack: torch.Tensor,
):
    """Forward hook on `decoder_layers[ℓ].self_attn.o_proj`. On forward passes
    with seq_len > 1, replaces `output[batch_idx[i], pos_idx[i], :]` with
    `(x[batch_idx[i], pos_idx[i], :] * scale_w + ch.flat) @ W_o^T`. Mirrors the
    attention-output reconstruction used by `apply_task_operator` at test time."""

    def hook(module, inputs, output):
        if output.shape[1] == 1:
            return output
        x = inputs[0]
        out = output.clone()
        N = batch_idx.shape[0]
        head_dim = ch_stack.shape[-1]
        b = batch_idx.to(x.device)
        p = pos_idx.to(x.device)
        x_at = x[b, p, :]
        scale = w_stack.repeat_interleave(head_dim, dim=-1).to(x.device, x.dtype)
        ch_flat = ch_stack.reshape(N, -1).to(x.device, x.dtype)
        recon = (x_at * scale + ch_flat) @ module.weight.T.to(x.device, x.dtype)
        out[b, p, :] = recon
        return out
    return hook
