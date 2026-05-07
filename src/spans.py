"""Resolve context, query, template, and content positions on a tokenized prompt.

Algorithm parity with `src_final_v1/spans.py`, but uses the v2 `attention_sink` semantics
(via `leading_sink_offset`) instead of v1's `leading_bos_offset`.

The contracts:
    identify_context_query → returns (context_span, query_span) where the context is
    [sink, query_start) and the query is [query_start, query_end).

    resolve_query_positions → given a query_span and a content substring, returns
    {full_text, char_spans, query_char_span, content_positions, template_positions}.
    Template positions are all query positions not in content_positions.
"""

from __future__ import annotations

from .tokenization import leading_sink_offset


def _ensure_1d_ids(input_ids):
    if hasattr(input_ids, "ndim") and input_ids.ndim == 2:
        return input_ids[0]
    return input_ids


def token_char_spans(tokenizer, input_ids, *, prompt_text: str | None = None):
    """Return (decoded_text, char_spans). Each char_span is (start, end) in decoded text;
    sink tokens are encoded as (-1, -1)."""

    input_ids = _ensure_1d_ids(input_ids)
    sink_offset = leading_sink_offset(tokenizer, input_ids)
    core_ids = input_ids[sink_offset:].tolist()

    if prompt_text is not None:
        try:
            encoded = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded.get("offset_mapping")
            token_ids = encoded.get("input_ids")
            if offsets is not None and list(token_ids) == core_ids:
                spans = [(-1, -1)] * sink_offset
                spans.extend((int(s), int(e)) for s, e in offsets)
                return prompt_text, spans
        except Exception:
            pass

    full_text = tokenizer.decode(core_ids, skip_special_tokens=False)
    spans: list[tuple[int, int]] = [(-1, -1)] * sink_offset
    for idx in range(len(core_ids)):
        left = tokenizer.decode(core_ids[:idx], skip_special_tokens=False)
        right = tokenizer.decode(core_ids[: idx + 1], skip_special_tokens=False)
        spans.append((len(left), len(right)))
    return full_text, spans


def _find_char_span(full_text, needle, *, start_char=0, end_char=None, occurrence=0):
    if not needle:
        raise ValueError("needle must be non-empty")
    search_from = start_char
    match_start = -1
    limit = len(full_text) if end_char is None else int(end_char)
    for _ in range(occurrence + 1):
        match_start = full_text.find(needle, search_from, limit)
        if match_start < 0:
            raise ValueError(f"substring not found: {needle!r}")
        search_from = match_start + 1
    return match_start, match_start + len(needle)


def _token_span_from_char_span(char_spans, match_start, match_end):
    hits = [
        idx
        for idx, (cs, ce) in enumerate(char_spans)
        if not (ce <= match_start or cs >= match_end)
    ]
    if not hits:
        raise ValueError("substring matched chars but not tokens")
    return hits[0], hits[-1] + 1


def _query_char_bounds(char_spans, query_span):
    qs, qe = query_span
    if qe <= qs:
        return char_spans[qs][0], char_spans[qs][0]
    return char_spans[qs][0], char_spans[qe - 1][1]


def find_token_span_by_substring(
    tokenizer, input_ids, needle, *, start_char=0, occurrence=0, prompt_text=None
):
    full_text, char_spans = token_char_spans(tokenizer, input_ids, prompt_text=prompt_text)
    match_start, match_end = _find_char_span(
        full_text, needle, start_char=start_char, occurrence=occurrence
    )
    return _token_span_from_char_span(char_spans, match_start, match_end)


def identify_context_query(tokenizer, icl_ids, query_prompt, *, prompt_text=None):
    """Split an ICL prompt into (context_span, query_span).

    The context begins at the sink offset and ends where the (last occurrence of the)
    query_prompt substring begins; the query span covers the query_prompt itself.
    """

    start_char = max(0, len(prompt_text) - len(query_prompt) - 4) if prompt_text is not None else 0
    qs, qe = find_token_span_by_substring(
        tokenizer, icl_ids, query_prompt,
        start_char=start_char, occurrence=0, prompt_text=prompt_text,
    )
    return (leading_sink_offset(tokenizer, icl_ids), qs), (qs, qe)


def resolve_query_positions(tokenizer, input_ids, query_span, content_text, *, prompt_text=None):
    """Resolve template + content token positions inside a query span.

    Returns {full_text, char_spans, query_char_span, content_positions, template_positions}.
    """

    input_ids = _ensure_1d_ids(input_ids)
    full_text, char_spans = token_char_spans(tokenizer, input_ids, prompt_text=prompt_text)
    query_char_start, query_char_end = _query_char_bounds(char_spans, query_span)

    content_positions: list[int] = []
    if content_text:
        match_start, match_end = _find_char_span(
            full_text, content_text,
            start_char=query_char_start, end_char=query_char_end,
        )
        if match_start > query_char_start and full_text[match_start - 1] == " ":
            match_start -= 1
        cs, ce = _token_span_from_char_span(char_spans, match_start, match_end)
        qs, qe = query_span
        if cs < qs or ce > qe:
            raise ValueError("content span fell outside query span")
        content_positions = list(range(cs, ce))

    content_set = set(content_positions)
    template_positions = [
        position for position in range(query_span[0], query_span[1]) if position not in content_set
    ]
    return {
        "full_text": full_text,
        "char_spans": char_spans,
        "query_char_span": (query_char_start, query_char_end),
        "content_positions": content_positions,
        "template_positions": template_positions,
    }
