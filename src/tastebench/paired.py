"""Paired-order evaluation questions built from the published trajectory prefixes."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tastebench.schema import Question

LETTERS = "ABCDEFGH"
OMISSION_MARKER = "...[middle transcript lines omitted to satisfy the 64K-token input cap]..."
TOKENIZER = "o200k_base"


@dataclass(frozen=True)
class PreparedQuestion:
    messages: list[dict[str, str]]
    correct: str
    input_tokens: int
    tokenizer_type: str
    full_prefix_chars: int
    visible_prefix_chars: int
    truncated: bool


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def option_blocks(question: Question, *, reverse: bool) -> tuple[str, str]:
    """Render the published option order, or its exact reverse, relabelling as we go.

    The release ships ``choices`` already in the evaluated order, so no shuffle seed
    is involved: the ``seeded`` order is the published one and the ``reversed`` order
    is its exact reverse. Option letters are recomputed after ordering, so the correct
    letter differs between the two presentations.
    """
    order = list(range(len(question.choices)))
    if reverse:
        order.reverse()
    blocks: list[str] = []
    correct = ""
    for position, choice_index in enumerate(order):
        letter = LETTERS[position]
        blocks.append(f"Option {letter}:\n{question.choices[choice_index].strip()}")
        if choice_index == question.answer_index:
            correct = letter
    if not correct:
        raise ValueError(f"{question.id}: answer_index is outside the option list")
    return "\n\n".join(blocks), correct


def render_messages(
    question: Question,
    prefix: str,
    *,
    reverse: bool,
    system_prompt: str,
    user_template: str,
) -> tuple[list[dict[str, str]], str]:
    options, correct = option_blocks(question, reverse=reverse)
    user = user_template.format(
        query=question.query.strip(),
        prefix=prefix.strip(),
        options=options,
    )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user},
    ], correct


def parse_final_answer(text: str, n_options: int) -> str | None:
    """Parse only an explicit final answer or a bare single-letter response."""
    valid = LETTERS[:n_options]
    stripped = text.strip()
    if stripped.upper() in valid:
        return stripped.upper()
    match = re.search(rf"(?:^|\n)\s*ANSWER:\s*([{valid}])\s*$", stripped, re.I)
    return match.group(1).upper() if match else None


TokenCounter = Callable[[list[dict[str, str]]], tuple[int, str]]


@functools.lru_cache(maxsize=1)
def _encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding(TOKENIZER)


def make_token_counter(multiplier: float = 1.0) -> TokenCounter:
    """Count the rendered messages locally with tiktoken, inflated by a safety factor.

    Tokenizers differ behind OpenAI-compatible APIs, so the preflight count is
    multiplied upward before it is compared against the protocol input cap.
    """
    if multiplier < 1.0:
        raise ValueError("token multiplier must be at least 1")

    def count(messages: list[dict[str, str]]) -> tuple[int, str]:
        encoding = _encoding()
        raw = sum(
            len(encoding.encode(message.get("content") or "", disallowed_special=()))
            for message in messages
        )
        return math.ceil(raw * multiplier), f"{TOKENIZER}*{multiplier:g}"

    return count


def prepare_question(
    question: Question,
    *,
    reverse: bool,
    system_prompt: str,
    user_template: str,
    max_input_tokens: int,
    count_tokens: TokenCounter,
) -> PreparedQuestion:
    """Use the published prefix unless the complete request exceeds the token cap."""
    full_prefix = question.prefix_text

    def build(prefix: str) -> tuple[list[dict[str, str]], str, int, str]:
        messages, correct = render_messages(
            question,
            prefix,
            reverse=reverse,
            system_prompt=system_prompt,
            user_template=user_template,
        )
        tokens, tokenizer_type = count_tokens(messages)
        return messages, correct, tokens, tokenizer_type

    messages, correct, tokens, tokenizer_type = build(full_prefix)
    if tokens <= max_input_tokens:
        return PreparedQuestion(
            messages=messages,
            correct=correct,
            input_tokens=tokens,
            tokenizer_type=tokenizer_type,
            full_prefix_chars=len(full_prefix),
            visible_prefix_chars=len(full_prefix),
            truncated=False,
        )

    lines = full_prefix.splitlines()
    head = lines[:25]
    remainder = lines[25:]
    low, high = 0, len(remainder)
    best: tuple[list[dict[str, str]], str, int, str, str] | None = None
    while low <= high:
        keep_tail = (low + high) // 2
        candidate_lines = head + [OMISSION_MARKER]
        if keep_tail:
            candidate_lines += remainder[-keep_tail:]
        candidate = "\n".join(candidate_lines)
        candidate_messages, candidate_correct, candidate_tokens, candidate_tokenizer = build(
            candidate
        )
        if candidate_tokens <= max_input_tokens:
            best = (
                candidate_messages,
                candidate_correct,
                candidate_tokens,
                candidate_tokenizer,
                candidate,
            )
            low = keep_tail + 1
        else:
            high = keep_tail - 1
    if best is None:
        raise ValueError(f"{question.id}: task and prefix head alone exceed the input-token cap")
    messages, correct, tokens, tokenizer_type, visible_prefix = best
    return PreparedQuestion(
        messages=messages,
        correct=correct,
        input_tokens=tokens,
        tokenizer_type=tokenizer_type,
        full_prefix_chars=len(full_prefix),
        visible_prefix_chars=len(visible_prefix),
        truncated=True,
    )
