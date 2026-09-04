"""The published question schema, plus its correctness invariants.

A :class:`Question` is one row of the released parquet files, with the fourteen
columns the dataset ships and nothing else. Only ``query``, ``prefix_text`` and the
two entries of ``choices`` are ever shown to a model under test; the remaining
columns are metadata for grouping, provenance and contamination detection.

``choices`` arrives in the published order and ``answer`` is the letter of the
correct option *in that order*, so no shuffle seed is needed to reproduce the
presentation (see :mod:`tastebench.paired`).

``check_question`` returns a list of human-readable problems (empty == valid).
"""

from __future__ import annotations

from pydantic import BaseModel

SCHEMA_VERSION = 1
LETTERS = "AB"


class Question(BaseModel):
    """One released two-way taste question."""

    id: str
    domain: str  # "engineering" | "research"
    method: str  # "detour" | "parallel"
    cell: str  # f"{method}_{domain}"
    source: str  # upstream trajectory source, e.g. "swebench" / "malt"
    task_id: str  # upstream instance id, for reproduction
    query: str  # model-visible: the task given to the agent
    prefix_text: str  # model-visible: the rendered trajectory up to the fork
    prefix_text_short: str  # a shorter rendering, not used by the paired protocol
    prefix_steps: int  # number of trajectory steps behind ``prefix_text``
    choices: list[str]  # model-visible: the two candidate next steps, published order
    answer: str  # "A" or "B", the correct letter in the published order
    answer_index: int  # index into ``choices`` of the correct candidate
    canary: str  # contamination canary string, identical on every row

    @property
    def arity(self) -> int:
        return len(self.choices)


def check_question(q: Question) -> list[str]:
    """Return a list of invariant violations for ``q`` (empty == valid)."""
    problems: list[str] = []

    if not q.id:
        problems.append("empty id")
    if not q.query.strip():
        problems.append("empty query")
    if not q.prefix_text.strip():
        problems.append("empty prefix_text")

    if q.arity != 2:
        problems.append(f"expected 2 choices, found {q.arity}")
    for index, choice in enumerate(q.choices):
        if not choice.strip():
            problems.append(f"choice[{index}] empty text")

    if q.answer not in LETTERS:
        problems.append(f"answer {q.answer!r} is not one of {tuple(LETTERS)}")
    elif not 0 <= q.answer_index < q.arity:
        problems.append(f"answer_index {q.answer_index} is out of range")
    elif LETTERS[q.answer_index] != q.answer:
        problems.append(f"answer {q.answer} disagrees with answer_index {q.answer_index}")

    if q.cell != f"{q.method}_{q.domain}":
        problems.append(f"cell {q.cell!r} does not match method/domain {q.method}/{q.domain}")

    if q.prefix_steps < 1:
        problems.append(f"prefix_steps {q.prefix_steps} < 1")

    return problems
