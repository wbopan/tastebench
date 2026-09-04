"""Render distilled steps into the plain-text transcript shown in a benchmark prompt.

Tolerant of both the canonical step schema (``kind``/``text``/``paths``) and the
legacy bundle schema (``k``/``t``/``changes``) so it can render older embedded
transcripts during migration.
"""

from __future__ import annotations

from typing import Any

MAX_PREFIX_CHARS = 16000


def _kind(step: dict[str, Any]) -> str | None:
    return step.get("kind") or step.get("k")


def render_steps(steps: list[dict[str, Any]]) -> str:
    """Render a list of distilled steps to a readable, line-indexed transcript."""
    lines: list[str] = []
    for idx, s in enumerate(steps):
        i = s.get("i", idx)
        kind = _kind(s)
        if kind == "agent":
            text = (s.get("text") or s.get("t") or "").strip()
            lines.append(f"[{i}] AGENT: {text}")
        elif kind == "cmd":
            lines.append(f"[{i}] $ {s.get('cmd', '')}   (exit={s.get('exit')})")
            out = (s.get("out") or "").strip()
            if out:
                lines.append("\n".join("    " + ln for ln in out.splitlines()))
        elif kind == "file":
            paths = s.get("paths") or s.get("changes") or []
            lines.append(f"[{i}] EDIT: {', '.join(paths)}")
    return "\n".join(lines)


def trim_prefix(text: str, max_chars: int = MAX_PREFIX_CHARS) -> str:
    """Keep the first ~25 lines (task setup) + a tail that fills the budget."""
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    head = "\n".join(lines[:25])
    remaining = max_chars - len(head) - 60
    tail: list[str] = []
    total = 0
    for ln in reversed(lines[25:]):
        total += len(ln) + 1
        if total > remaining:
            break
        tail.append(ln)
    tail.reverse()
    return head + "\n...[earlier steps omitted]...\n" + "\n".join(tail)


def render_prefix(steps: list[dict[str, Any]], breakpoint_step: int) -> str:
    """Render ``steps[:breakpoint_step]`` as a trimmed prefix transcript."""
    return trim_prefix(render_steps(steps[:breakpoint_step]))


# --- model-visibility whitelist ---------------------------------------------
# The ONLY item content a model under test may see: query, prefix, choice texts.
# Used both to build eval prompts and to assert no hidden field leaks (invariants).


def render_prompt_visible(item: Any) -> str:
    """Concatenate exactly the model-visible fields of an item (query + prefix + options).

    Order/shuffle of options is handled by the eval runner; this is the canonical
    visible content used for leak-checking.
    """
    parts = [item.query.strip(), item.prefix_text.strip()]
    parts += [c.text.strip() for c in item.choices]
    return "\n\n".join(p for p in parts if p)
