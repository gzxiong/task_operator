"""Prompt templates for ICL and zero-shot."""

from __future__ import annotations

from typing import Sequence


DEMO_TEMPLATE = "Input: {input} \nOutput: {output}"
QUERY_TEMPLATE = "Input: {input} \nOutput:"

# Chain-of-thought variants — used by reasoning tasks. The CoT trigger phrase
# "Let's think step by step." appears immediately after "Output:" both in demo
# targets and in the query, biasing the model toward step-by-step reasoning
# even in zero-shot. Mirrors `src_final_v1/prompting.py`.
COT_DEMO_TEMPLATE = "Input: {input} \nOutput: Let's think step by step. {output}"
COT_QUERY_TEMPLATE = "Input: {input} \nOutput: Let's think step by step."


def build_full_icl_prompt(
    demos: Sequence[dict],
    question: str,
    *,
    demo_template: str = DEMO_TEMPLATE,
    query_template: str = QUERY_TEMPLATE,
) -> str:
    """Build a few-shot prompt by concatenating demo blocks with the query block."""

    def _output(demo: dict) -> str:
        if "output" in demo:
            return demo["output"]
        if "target" in demo:
            return demo["target"]
        raise KeyError("Demo record must contain 'output' or 'target'")

    demo_blocks = [
        demo_template.format(input=d["input"], output=_output(d)).strip() for d in demos
    ]
    query_block = query_template.format(input=question).strip()
    if not demo_blocks:
        return query_block
    return "\n\n".join([*demo_blocks, query_block])


def build_zsl_prompt(question: str, *, query_template: str = QUERY_TEMPLATE) -> str:
    return query_template.format(input=question).strip()
