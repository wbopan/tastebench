"""Unified, versioned schema for a benchmark item, plus its correctness invariants.

A single :class:`Item` model covers every extraction format. A binary ``detour``
item is just ``arity == 2``; an N-way ``parallel`` item has more choices. Exactly one
choice is correct. Only ``query``, ``prefix_text`` and each ``Choice.text`` are ever
shown to a model under test -- everything else is hidden metadata (see
:mod:`tastebench.render` for the visibility whitelist).

``check_item`` returns a list of human-readable problems (empty == valid). Structural
checks always run; transcript-grounded checks run only when a store is supplied.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from tastebench.render import render_prompt_visible

SCHEMA_VERSION = 1
Method = Literal["detour", "parallel"]


class Outcome(BaseModel):
    """Ground-truth result of the rollout a choice was taken from."""

    passed: bool | None = None
    score: float | None = None


class Choice(BaseModel):
    text: str  # model-visible: the candidate next step
    is_correct: bool
    quality: str | None = None  # "good" | "poor" (informational)
    traj: str | None = None  # ref into the transcript store
    choice_start_step: int | None = None  # step in `traj` where this choice begins
    outcome: Outcome | None = None  # ground-truth (parallel has it; detour usually None)
    rationale: str = ""  # hidden: why this is better/worse


class Provenance(BaseModel):
    source_run: str | None = None
    sse_sha: str | None = None
    extractor: Method | None = None
    prompt_version: str | None = None
    builder_model: str | None = None
    split: str | None = None  # e.g. "1P2F" for parallel
    extra: dict[str, Any] = Field(default_factory=dict)


class Item(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str
    method: Method
    dataset: str  # upstream source: swebench / malt
    task_id: str  # repo/data instance id, for reproduction
    query: str
    reference_traj: str | None = None  # transcript that defines the prefix
    breakpoint_step: int | None = None  # prefix = reference_traj.steps[:breakpoint_step]
    prefix_text: str = ""  # rendered prefix shown to the model (self-contained for eval)
    choices: list[Choice]
    rationale: str = ""  # overall: what the fork is, why the correct choice is better
    quality: dict[str, Any] | None = None  # quality metadata (too_easy, screen counts, ...)
    provenance: Provenance = Field(default_factory=Provenance)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def arity(self) -> int:
        return len(self.choices)

    def correct_index(self) -> int | None:
        idxs = [i for i, c in enumerate(self.choices) if c.is_correct]
        return idxs[0] if len(idxs) == 1 else None


def check_item(item: Item, store: object | None = None) -> list[str]:
    """Return a list of invariant violations for ``item`` (empty == valid).

    ``store`` is an optional object exposing ``has(traj)`` and ``get(traj)`` for
    transcript-grounded checks (breakpoint bounds, choice traj existence).
    """
    problems: list[str] = []

    if not item.id:
        problems.append("empty id")
    if not item.query.strip():
        problems.append("empty query")
    if not item.prefix_text.strip():
        problems.append("empty prefix_text")
    if item.arity < 2:
        problems.append(f"arity {item.arity} < 2")

    n_correct = sum(1 for c in item.choices if c.is_correct)
    if n_correct != 1:
        problems.append(f"expected exactly 1 correct choice, found {n_correct}")

    for j, c in enumerate(item.choices):
        if not c.text.strip():
            problems.append(f"choice[{j}] empty text")

    # the correct choice should come from a passing rollout when outcome is known
    ci = item.correct_index()
    if ci is not None:
        oc = item.choices[ci].outcome
        if oc is not None and oc.passed is False:
            problems.append("correct choice maps to a failing rollout (outcome.passed=False)")

    # leakage: hidden field *values* must not surface in the visible render
    visible = render_prompt_visible(item)
    for c in item.choices:
        if c.rationale and c.rationale.strip() and c.rationale.strip() in visible:
            problems.append("choice rationale text leaked into visible prompt")
    # transcript-grounded checks
    if store is not None:
        get = getattr(store, "get", None)
        has = getattr(store, "has", None)
        if item.reference_traj and item.breakpoint_step is not None and callable(get):
            if not has(item.reference_traj):  # type: ignore[misc]
                problems.append(f"reference_traj {item.reference_traj} not in store")
            else:
                n = len(get(item.reference_traj))  # type: ignore[misc]
                if not (0 < item.breakpoint_step <= n):
                    problems.append(
                        f"breakpoint_step {item.breakpoint_step} out of bounds (traj has {n} steps)"
                    )
        if callable(has):
            for j, c in enumerate(item.choices):
                if c.traj and not has(c.traj):  # type: ignore[misc]
                    problems.append(f"choice[{j}] traj {c.traj} not in store")

    return problems
