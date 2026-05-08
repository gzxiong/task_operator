"""Per-task answer extraction and scoring.

Reasoning evaluation is robust to two false-negative issues identified in v1:
    1. gsm8k: string-equality on the numeric answer mis-scores `7` vs `7.00`. Fix:
       compare as floats via math.isclose.
    2. math500: the v1 boxed extractor uses a non-greedy regex that stops at the first
       `}`, truncating nested LaTeX. Fix: balanced-brace walker (`last_boxed_only_string`)
       plus Hendrycks `is_equiv` (with `_strip_string` normalization for tfrac/dfrac,
       \\left/\\right, units, sqrt, slashes, leading `.` decimals).

The Hendrycks helpers (`last_boxed_only_string`, `remove_boxed`, `_fix_fracs`,
`_fix_a_slash_b`, `_remove_right_units`, `_fix_sqrt`, `_strip_string`, `is_equiv`) are
vendored verbatim from https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
so this module has no external dependency on that repo.
"""

from __future__ import annotations

import math
import re

# === Hendrycks MATH equivalence (vendored) ============================================


def last_boxed_only_string(string: str):
    """Return the last `\\boxed{...}` (or `\\fbox{...}`) substring, with balanced braces."""

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    if s.startswith("\\fbox{") and s.endswith("}"):
        return s[len("\\fbox{") : -1]
    return None


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_i = int(a)
        b_i = int(b)
        assert string == f"{a_i}/{b_i}"
        return "\\frac{" + str(a_i) + "}{" + str(b_i) + "}"
    except Exception:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def hendrycks_is_equiv(str1, str2) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


# === Per-task answer extractors =======================================================

_HASH_MARKER_RE = re.compile(r"####\s*([^\n]+?)\s*$", flags=re.MULTILINE)
# Phrase-locator only — does NOT capture the suffix (which used to truncate decimals).
_ANSWER_PHRASE_FIND_RE = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*[:\-]?|answer\s*[:\-])\s*",
    flags=re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_PAREN_LETTER_RE = re.compile(r"\(\s*([A-D])\s*\)", flags=re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?"

# LaTeX block patterns used by _extract_math_answer as a robust fallback when no
# `\boxed{...}` / `####` / "the answer is ..." marker is present. Inline `$...$`
# uses a non-`$` character class so it can't span across two separate inline blocks.
_INLINE_LATEX_RE = re.compile(r"\$([^$\n]+?)\$")
_DISPLAY_LATEX_RE = re.compile(r"\\\[(.+?)\\\]", flags=re.DOTALL)
_PAREN_LATEX_RE = re.compile(r"\\\((.+?)\\\)", flags=re.DOTALL)


def _last_latex_block(text: str) -> str | None:
    """Return the content of the last `$...$` / `\\(...\\)` / `\\[...\\]` block in
    `text`, choosing among the three patterns by which one ends *latest*. None if
    no LaTeX block is found."""

    candidates: list[tuple[int, str]] = []
    for pat in (_INLINE_LATEX_RE, _DISPLAY_LATEX_RE, _PAREN_LATEX_RE):
        for m in pat.finditer(text):
            candidates.append((m.end(), m.group(1)))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[0])
    return candidates[-1][1].strip().rstrip(_TRAILING_PUNCT).strip() or None


def _suffix_after_last_phrase(text: str) -> str | None:
    """Return the text after the final answer-phrase match, truncated at the next newline."""

    matches = list(_ANSWER_PHRASE_FIND_RE.finditer(text))
    if not matches:
        return None
    suffix = text[matches[-1].end():]
    return suffix.split("\n", 1)[0]


def _clean_numeric_token(tok: str) -> str:
    return tok.replace("$", "").replace(",", "")


def _extract_numeric_token(text: str) -> str | None:
    """gsm8k-style precedence: '#### N' → 'the answer is X' → last numeric. Returns the
    raw numeric token (string), with leading `$` and commas stripped. Decimals preserved.
    """

    if not text:
        return None
    matches = _HASH_MARKER_RE.findall(text)
    if matches:
        cand = matches[-1].strip().rstrip(_TRAILING_PUNCT).strip()
        m = _NUMERIC_TOKEN_RE.search(cand)
        if m:
            return _clean_numeric_token(m.group(0))
        cleaned = _clean_numeric_token(cand)
        if cleaned:
            return cleaned
    suffix = _suffix_after_last_phrase(text)
    if suffix is not None:
        m = _NUMERIC_TOKEN_RE.search(suffix)
        if m:
            return _clean_numeric_token(m.group(0))
    candidates = _NUMERIC_TOKEN_RE.findall(text)
    if candidates:
        return _clean_numeric_token(candidates[-1])
    return None


def _extract_math_answer(text: str) -> str | None:
    """math500 precedence:
        1. `\\boxed{...}` (balanced)
        2. `#### X`
        3. "the answer is X" — prefer LaTeX block in suffix, else numeric, else raw suffix
        4. last `$...$` / `\\(...\\)` / `\\[...\\]` block in the whole text
        5. last numeric token (final resort).
    """

    if not text:
        return None
    boxed = remove_boxed(last_boxed_only_string(text))
    if boxed is not None and boxed.strip():
        return boxed.strip()
    matches = _HASH_MARKER_RE.findall(text)
    if matches:
        cand = matches[-1].strip().rstrip(_TRAILING_PUNCT).strip()
        if cand:
            return cand
    suffix = _suffix_after_last_phrase(text)
    if suffix is not None:
        # Prefer LaTeX block over a bare numeric in the phrase suffix —
        # "The answer is $\\frac{14}{3}$" should yield "\\frac{14}{3}", not "14".
        latex_in_suffix = _last_latex_block(suffix)
        if latex_in_suffix:
            return latex_in_suffix
        m = _NUMERIC_TOKEN_RE.search(suffix)
        if m:
            return _clean_numeric_token(m.group(0))
        cand = suffix.strip().rstrip(_TRAILING_PUNCT).strip()
        if cand:
            return cand
    # Whole-text LaTeX-block fallback recovers cases like
    # "The point is $\\left(3\\sqrt{10},\\frac{\\pi}{2}\\right).$" where no
    # phrase or `\\boxed{}` was emitted.
    latex_anywhere = _last_latex_block(text)
    if latex_anywhere:
        return latex_anywhere
    candidates = _NUMERIC_TOKEN_RE.findall(text)
    if candidates:
        return _clean_numeric_token(candidates[-1])
    return None


def _extract_choice_letter(text: str) -> str | None:
    """gpqa precedence: 'the answer is (X)' → last '(LETTER)' → last bare A-D."""

    if not text:
        return None
    suffix = _suffix_after_last_phrase(text)
    if suffix is not None:
        m = _PAREN_LETTER_RE.search(suffix)
        if m:
            return m.group(1).upper()
        token = suffix.strip().rstrip(_TRAILING_PUNCT).strip()
        if len(token) == 1 and token.upper() in {"A", "B", "C", "D"}:
            return token.upper()
        if len(token) == 3 and token.startswith("(") and token.endswith(")"):
            inner = token[1].upper()
            if inner in {"A", "B", "C", "D"}:
                return inner
    paren_matches = _PAREN_LETTER_RE.findall(text)
    if paren_matches:
        return paren_matches[-1].upper()
    bare_matches = re.findall(r"\b([A-D])\b", text)
    if bare_matches:
        return bare_matches[-1].upper()
    return None


# === Equivalence ======================================================================


def _is_finite_float(token: str | None) -> bool:
    if token is None or token == "":
        return False
    try:
        return math.isfinite(float(token))
    except (TypeError, ValueError):
        return False


def is_equiv_gsm8k(target: str, prediction: str) -> bool:
    t = _extract_numeric_token(target or "")
    p = _extract_numeric_token(prediction or "")
    if not _is_finite_float(t) or not _is_finite_float(p):
        return False
    return math.isclose(float(t), float(p), rel_tol=1e-9, abs_tol=1e-6)


def is_equiv_math500(target: str, prediction: str) -> bool:
    t = _extract_math_answer(target or "")
    p = _extract_math_answer(prediction or "")
    if t is None or p is None:
        return False
    return hendrycks_is_equiv(t, p)


def is_equiv_gpqa(target: str, prediction: str) -> bool:
    t = _extract_choice_letter(target or "")
    p = _extract_choice_letter(prediction or "")
    if t is None or p is None:
        return False
    return t == p


def _is_equiv_lexical(target: str, prediction: str) -> bool:
    return (target or "").strip().lower() == (prediction or "").strip().lower()


def _is_equiv_algorithmic(target: str, prediction: str) -> bool:
    return (target or "").strip() == (prediction or "").strip()


def parse_answer(task: str, text: str) -> str | None:
    """Public extractor; dispatches by task."""

    if task == "gsm8k":
        return _extract_numeric_token(text)
    if task == "math500":
        return _extract_math_answer(text)
    if task == "gpqa":
        return _extract_choice_letter(text)
    if task in {"translation", "linguistic"}:
        return (text or "").strip().lower()
    if task in {"uppercase", "reverse", "deduplicate"}:
        return (text or "").strip()
    raise ValueError(f"Unknown task: {task!r}")


def evaluate_generation(task: str, generated_text: str, target: str) -> dict:
    """Score one prediction. Returns parsed answer / target tokens, score (float), is_correct."""

    if task == "gsm8k":
        is_correct = is_equiv_gsm8k(target, generated_text)
    elif task == "math500":
        is_correct = is_equiv_math500(target, generated_text)
    elif task == "gpqa":
        is_correct = is_equiv_gpqa(target, generated_text)
    elif task in {"translation", "linguistic"}:
        is_correct = _is_equiv_lexical(target, generated_text)
    elif task in {"uppercase", "reverse", "deduplicate"}:
        is_correct = _is_equiv_algorithmic(target, generated_text)
    else:
        raise ValueError(f"Unknown task: {task!r}")
    return {
        "generated_text": generated_text,
        "parsed_answer": parse_answer(task, generated_text or ""),
        "parsed_target": parse_answer(task, target or ""),
        "is_correct": bool(is_correct),
        "score": 1.0 if is_correct else 0.0,
    }
