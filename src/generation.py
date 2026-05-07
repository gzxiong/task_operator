"""Greedy / sampling generation under affine replay hooks, dispatched via `model.generate`.

Replay records arrive in two flavors:

    prompt_records: dict[layer_idx -> list of {token_idx, w, context_heads}]
        Applied during the prompt forward (seq_len > 1) at each token_idx.

    gen_records: dict[layer_idx -> list of {broadcast: bool, step: int|None,
                                              w, context_heads}]
        Applied at each generation step (seq_len == 1) — picks the broadcast record
        if any is present, else the per-step record matching step_counter[0].

A single forward post-hook on `model.lm_head` advances `step_counter[0]` once per
single-token forward, so o_proj hooks across all decoder layers see a consistent
step within a generation step. This decoupling lets us call `model.generate(...)`
directly with all of HF's native knobs (`do_sample`, `temperature`, `top_p`,
`top_k`, `stop_strings`, `repetition_penalty`).
"""

from __future__ import annotations

import re
import time
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .model import get_decoder_layers, get_text_config
from .tokenization import encode_prompt_ids


def _build_answer_phrase_pattern(phrases: tuple[str, ...]):
    """Compile `(?:p1|p2|...).+?\\n` (case-insensitive) for the given phrases,
    or return None if phrases is empty. `.+?` is non-greedy and (without
    `DOTALL`) cannot match `\\n`, so it forces ≥1 non-newline char between
    phrase and the stopping newline."""

    if not phrases:
        return None
    alt = "|".join(re.escape(p) for p in phrases)
    return re.compile(r"(?:" + alt + r").+?\n", flags=re.IGNORECASE)


class _AnswerPhraseStop(StoppingCriteria):
    """Stop at the first newline AFTER any phrase in `phrases` has been emitted,
    requiring at least one non-newline character between phrase and that newline.

    Single regex with `|` alternation across all phrases (case-insensitive),
    so `("answer is",)` matches "The answer is", "the correct answer is",
    "Therefore, the answer is", etc.

    Hot path is cheap: skip the regex unless the latest emitted token decodes
    to something containing '\\n'. Cold path (when '\\n' is in the new token)
    decodes the full gen suffix and runs `re.search` once.
    """

    def __init__(self, tokenizer, prompt_len: int, *, phrases: tuple[str, ...]):
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.phrases = tuple(phrases)
        self.pattern = _build_answer_phrase_pattern(self.phrases)

    def __call__(self, input_ids: torch.LongTensor, scores=None, **kwargs) -> bool:
        if self.pattern is None:
            return False
        n_gen = int(input_ids.shape[1]) - self.prompt_len
        if n_gen <= 0:
            return False
        last_decoded = self.tokenizer.decode(
            [int(input_ids[0, -1].item())], skip_special_tokens=False,
        )
        if "\n" not in last_decoded:
            return False
        gen_ids = input_ids[0, self.prompt_len:].tolist()
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return bool(self.pattern.search(decoded))


class _FastSubstringStop(StoppingCriteria):
    """Sliding-window substring stopping criterion.

    On each generation step decodes only the last `window` tokens and tests
    whether any `stop_strings` substring is present. ~0.1-0.5 ms / step,
    versus HF's `StopStringCriteria` which re-decodes the growing sequence
    every step (~190 ms / step).

    Designed for multi-token stop strings where id-based eos doesn't suffice
    (e.g., `"\\nInput:"` for reasoning). Single-token stops should be folded
    into `eos_token_id` instead — this criterion is then unnecessary.
    """

    def __init__(self, tokenizer, stop_strings: list[str], *, prompt_len: int, window: int = 8):
        self.tokenizer = tokenizer
        self.stop_strings = [s for s in stop_strings if s]
        self.prompt_len = int(prompt_len)
        self.window = int(window)

    def __call__(self, input_ids: torch.LongTensor, scores=None, **kwargs) -> bool:
        if not self.stop_strings:
            return False
        n_gen = int(input_ids.shape[1]) - self.prompt_len
        if n_gen <= 0:
            return False
        take = min(n_gen, self.window)
        tail_ids = input_ids[0, -take:].tolist()
        decoded = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
        return any(s in decoded for s in self.stop_strings)


def _make_prompt_replay_hook(records: list[dict], head_dim: int):
    def hook(module, inputs, output):
        x = inputs[0]
        if x.shape[1] == 1:
            return output
        y = output.clone()
        w_dtype = x.dtype
        weight = module.weight
        if weight.dtype != w_dtype:
            weight = weight.to(dtype=w_dtype)
        for r in records:
            token_idx = int(r["token_idx"])
            if token_idx >= y.shape[1]:
                continue
            w = r["w"].to(x.device, w_dtype)
            ch = r["context_heads"].to(x.device, w_dtype)
            scale = w.repeat_interleave(head_dim).unsqueeze(0)
            bias = ch.reshape(1, -1) @ weight.T
            y[:, token_idx, :] = (x[:, token_idx, :] * scale) @ weight.T + bias
        return y
    return hook


def _make_gen_replay_hook(records: list[dict], step_counter: list[int], head_dim: int):
    def hook(module, inputs, output):
        x = inputs[0]
        if x.shape[1] != 1:
            return output
        step = int(step_counter[0])
        chosen = None
        for r in records:
            if r.get("broadcast"):
                chosen = r
                break
            if r.get("step") == step:
                chosen = r
                break
        if chosen is None:
            return output
        y = output.clone()
        w_dtype = x.dtype
        weight = module.weight
        if weight.dtype != w_dtype:
            weight = weight.to(dtype=w_dtype)
        w = chosen["w"].to(x.device, w_dtype)
        ch = chosen["context_heads"].to(x.device, w_dtype)
        scale = w.repeat_interleave(head_dim).unsqueeze(0)
        bias = ch.reshape(1, -1) @ weight.T
        y[:, 0, :] = (x[:, 0, :] * scale) @ weight.T + bias
        return y
    return hook


def _make_step_advance_hook(step_counter: list[int]):
    def hook(module, inputs, output):
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        if hasattr(x, "shape") and len(x.shape) >= 2 and x.shape[1] == 1:
            step_counter[0] += 1
        return output
    return hook


def _truncate_on_stop_strings(text: str, stop_strings: list[str]) -> str:
    """Remove the earliest stop string match (HF's StopStringCriteria leaves it in)."""

    if not stop_strings:
        return text
    best_idx = None
    for s in stop_strings:
        if not s:
            continue
        i = text.find(s)
        if i == -1:
            continue
        if best_idx is None or i < best_idx:
            best_idx = i
    return text if best_idx is None else text[:best_idx]


def _resolve_pad_token_id(tokenizer) -> int | None:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        eos = tokenizer.eos_token_id
        if isinstance(eos, list):
            return int(eos[0]) if eos else None
        return int(eos)
    return None


def _resolve_eos_token_ids(tokenizer, model=None) -> list[int]:
    """Union of EOS token ids from the tokenizer and (when available) the
    model's `generation_config.eos_token_id`. The model side is needed for
    e.g. gemma-3, where `tokenizer.eos_token_id` is just `<eos>` (id 1) but
    the model's GenerationConfig also lists `<end_of_turn>` (id 106) as a
    natural turn boundary. Without the model side, gemma-3 keeps generating
    past `<end_of_turn>` until `max_new_tokens`."""

    out: set[int] = set()
    eos_t = tokenizer.eos_token_id
    if eos_t is not None:
        if isinstance(eos_t, list):
            out.update(int(x) for x in eos_t)
        else:
            out.add(int(eos_t))
    if model is not None:
        gc = getattr(model, "generation_config", None)
        eos_g = getattr(gc, "eos_token_id", None) if gc is not None else None
        if eos_g is not None:
            if isinstance(eos_g, list):
                out.update(int(x) for x in eos_g)
            else:
                out.add(int(eos_g))
    return sorted(out)


_STOP_ID_CACHE: dict[tuple, list[int]] = {}


def _resolve_stop_token_ids(tokenizer, stop_strings: list[str] | None) -> list[int]:
    """Single-token IDs whose decoded form contains any of `stop_strings`.

    Used to drive HF's `eos_token_id` (id-based stop) in place of `stop_strings`
    (HF's `StopStringCriteria`). The string criteria re-decodes the growing
    sequence on every gen step and is ~6× slower per call; an id list is just
    a `last_token in set` check.

    Cross-token stop matches (e.g., two consecutive `\n` tokens forming `\n\n`)
    are handled by the post-hoc `_truncate_on_stop_strings` on the decoded
    text — id-based stop fires only when a single token already contains the
    stop substring. The result is cached per (tokenizer identity, stop_strings).
    """

    if not stop_strings:
        return []
    targets = tuple(s for s in stop_strings if s)
    key = (id(tokenizer), targets)
    if key in _STOP_ID_CACHE:
        return _STOP_ID_CACHE[key]

    found: set[int] = set()
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    for tok_id in range(int(vocab_size)):
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            continue
        if any(s in decoded for s in targets):
            found.add(int(tok_id))

    _STOP_ID_CACHE[key] = sorted(found)
    return _STOP_ID_CACHE[key]


def _infer_stop_reason(generated_ids, decoded_text, stop_strings, eos_ids,
                        max_new_tokens, *, answer_phrases: tuple[str, ...] = ()):
    """Return one of: "answer_phrase:'...'", "stop_string:'...'", "eos",
    "max_new_tokens", "unknown".

    Priority: answer_phrase > stop_string > eos > max_new_tokens.

    For answer_phrase, two passes:
      1. Strict: `phrase.+?\\n` regex (case-insensitive). The standard signal.
      2. Permissive substring: case-insensitive `phrase in decoded_text`,
         used when no `\\n` is in `decoded_text` (e.g. gemma-3 where the
         post-answer turn-boundary token is special and stripped by
         `skip_special_tokens=True`). Without this fallback such rows would
         fall through to `max_new_tokens` even though the answer is plainly
         in the text.
    """

    if answer_phrases:
        pat = _build_answer_phrase_pattern(tuple(answer_phrases))
        # Strict pass: phrase + content + \n
        if pat is not None:
            m = pat.search(decoded_text)
            if m:
                # Identify which phrase actually matched (case-insensitive).
                ms = m.group(0).lower()
                for p in answer_phrases:
                    if p.lower() in ms:
                        return f"answer_phrase:{p!r}"
                return f"answer_phrase:{answer_phrases[0]!r}"
        # Permissive pass: case-insensitive substring search, no \n required.
        low = decoded_text.lower()
        for p in answer_phrases:
            if p.lower() in low:
                return f"answer_phrase:{p!r}"

    earliest = None
    matched = None
    for s in stop_strings or []:
        if not s:
            continue
        i = decoded_text.find(s)
        if i == -1:
            continue
        if earliest is None or i < earliest:
            earliest = i
            matched = s
    if matched is not None:
        return f"stop_string:{matched!r}"
    if eos_ids and generated_ids and int(generated_ids[-1]) in set(int(x) for x in eos_ids):
        return "eos"
    if len(generated_ids) >= int(max_new_tokens):
        return "max_new_tokens"
    return "unknown"


def generate_with_hooks(
    model,
    tokenizer,
    prompt_text: str,
    *,
    prompt_records: dict[int, list[dict]] | None = None,
    gen_records: dict[int, list[dict]] | None = None,
    attention_sink: bool = True,
    max_new_tokens: int = 64,
    stop_strings: list[str] | None = None,
    answer_phrases: tuple[str, ...] = (),
    repetition_penalty: float = 1.1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> dict:
    """Greedy / sampling generation with affine replay hooks installed on each layer's o_proj.

    The decoded text has any leading stop_string match truncated; the generated_ids are the
    raw HF outputs (i.e., they may include the tokens that decode into the stop string).
    """

    device = next(model.parameters()).device
    text_cfg = get_text_config(model)
    head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads))

    input_ids = encode_prompt_ids(tokenizer, prompt_text, attention_sink=attention_sink, device=device)

    step_counter = [0]
    handles = []
    decoder_layers = get_decoder_layers(model)
    for layer_idx, recs in (prompt_records or {}).items():
        if not recs:
            continue
        h = decoder_layers[int(layer_idx)].self_attn.o_proj.register_forward_hook(
            _make_prompt_replay_hook(recs, head_dim)
        )
        handles.append(h)
    for layer_idx, recs in (gen_records or {}).items():
        if not recs:
            continue
        h = decoder_layers[int(layer_idx)].self_attn.o_proj.register_forward_hook(
            _make_gen_replay_hook(recs, step_counter, head_dim)
        )
        handles.append(h)
    handles.append(model.lm_head.register_forward_hook(_make_step_advance_hook(step_counter)))

    pad_token_id = _resolve_pad_token_id(tokenizer)
    eos_ids = _resolve_eos_token_ids(tokenizer, model)
    # Stop strings are split into two groups:
    #   1. id-covered: at least one single-vocab-token contains the string
    #      (e.g., "\n" → 198/271/715/...). These ids are folded into
    #      eos_token_id; no per-step criterion needed.
    #   2. uncovered (multi-token, e.g., "\nInput:"): handled by the fast
    #      sliding-window _FastSubstringStop criterion.
    # Cross-token boundary cases are also caught by _truncate_on_stop_strings
    # post-hoc on the decoded text.
    id_covered_strings: list[str] = []
    uncovered_strings: list[str] = []
    for s in stop_strings or []:
        if not s:
            continue
        if _resolve_stop_token_ids(tokenizer, [s]):
            id_covered_strings.append(s)
        else:
            uncovered_strings.append(s)
    stop_token_ids = _resolve_stop_token_ids(tokenizer, id_covered_strings)
    all_eos_ids = sorted(set(eos_ids) | set(stop_token_ids))

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        use_cache=True,
        pad_token_id=pad_token_id,
        return_dict_in_generate=True,
    )
    if all_eos_ids:
        gen_kwargs["eos_token_id"] = all_eos_ids if len(all_eos_ids) > 1 else all_eos_ids[0]
    criteria_list: list[StoppingCriteria] = []
    prompt_len = int(input_ids.shape[1])
    if uncovered_strings:
        criteria_list.append(
            _FastSubstringStop(tokenizer, uncovered_strings, prompt_len=prompt_len, window=8)
        )
    if answer_phrases:
        criteria_list.append(
            _AnswerPhraseStop(tokenizer, prompt_len, phrases=tuple(answer_phrases))
        )
    if criteria_list:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria_list)

    start = time.time()
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, **gen_kwargs)
        seq = output.sequences[0]
        prompt_len = int(input_ids.shape[1])
        generated_ids = seq[prompt_len:].tolist()
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    truncated = _truncate_on_stop_strings(decoded, stop_strings or [])
    # If any answer_phrase fired, also trim at the end of "phrase + content + \\n"
    # match. The criterion only stops at the *next* token boundary, which can
    # be a multi-newline token (e.g., Qwen3 id 271 = "\\n\\n"). Without this
    # trim the saved generated_text would carry the extra trailing newline.
    if answer_phrases:
        ap_pat = _build_answer_phrase_pattern(tuple(answer_phrases))
        if ap_pat is not None:
            m = ap_pat.search(truncated)
            if m:
                truncated = truncated[: m.end()]
    stop_reason = _infer_stop_reason(
        generated_ids, decoded, stop_strings, all_eos_ids, max_new_tokens,
        answer_phrases=tuple(answer_phrases),
    )

    return {
        "generated_text": truncated,
        "raw_text": decoded,
        "generated_ids": generated_ids,
        "prompt_token_count": int(input_ids.shape[1]),
        "generated_tokens": len(generated_ids),
        "stop_reason": stop_reason,
        "wall_time_sec": max(0.0001, time.time() - start),
    }
