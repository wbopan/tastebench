"""Paired-order evaluation questions built from full, redacted trajectory prefixes."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tastebench.redaction import redact_sensitive_text
from tastebench.render import render_steps
from tastebench.schema import Item
from tastebench.transcripts import TranscriptStore, checksum

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


def reconstruct_prefix(
    item: Item,
    store: TranscriptStore,
    transcript_manifest: Mapping[str, Mapping[str, Any]],
) -> str:
    """Rebuild the full pre-decision prefix from a manifest-verified transcript."""
    if not item.reference_traj or item.breakpoint_step is None:
        raise ValueError(f"{item.id}: missing reference trajectory or breakpoint")
    if not store.has(item.reference_traj):
        raise ValueError(f"{item.id}: transcript {item.reference_traj} is unavailable")
    manifest_entry = transcript_manifest.get(item.reference_traj)
    if not manifest_entry:
        raise ValueError(f"{item.id}: transcript is absent from the release manifest")
    steps = store.get(item.reference_traj)
    if checksum(steps) != manifest_entry.get("checksum"):
        raise ValueError(f"{item.id}: transcript checksum mismatch")
    if not 0 < item.breakpoint_step <= len(steps):
        raise ValueError(f"{item.id}: breakpoint is outside the transcript")
    # Source transcripts can contain credentials. Redaction is mandatory before
    # token counting, hashing, model calls, or artifact persistence.
    return redact_sensitive_text(render_steps(steps[: item.breakpoint_step]))


def option_blocks(item: Item, *, seed: int, reverse: bool) -> tuple[str, str]:
    order = list(range(item.arity))
    random.Random(f"{seed}-{item.id}").shuffle(order)
    if reverse:
        order.reverse()
    blocks: list[str] = []
    correct = ""
    for position, choice_index in enumerate(order):
        letter = LETTERS[position]
        blocks.append(f"Option {letter}:\n{item.choices[choice_index].text.strip()}")
        if item.choices[choice_index].is_correct:
            correct = letter
    if not correct:
        raise ValueError(f"{item.id}: no unique correct option")
    return "\n\n".join(blocks), correct


def render_messages(
    item: Item,
    prefix: str,
    *,
    seed: int,
    reverse: bool,
    system_prompt: str,
    user_template: str,
) -> tuple[list[dict[str, str]], str]:
    options, correct = option_blocks(item, seed=seed, reverse=reverse)
    user = user_template.format(
        query=item.query.strip(),
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
    item: Item,
    full_prefix: str,
    *,
    seed: int,
    reverse: bool,
    system_prompt: str,
    user_template: str,
    max_input_tokens: int,
    count_tokens: TokenCounter,
) -> PreparedQuestion:
    """Use the full prefix unless the complete request exceeds the token cap."""

    def build(prefix: str) -> tuple[list[dict[str, str]], str, int, str]:
        messages, correct = render_messages(
            item,
            prefix,
            seed=seed,
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
        raise ValueError(f"{item.id}: task and prefix head alone exceed the input-token cap")
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
