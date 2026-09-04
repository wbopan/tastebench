"""Scoring for the paired-order protocol.

Headline accuracy uses every release item as the denominator: request errors and
unparseable outputs count as incorrect. The headline number is ``both_correct_rate``
-- the share of items answered correctly under the seeded order and its exact reverse.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# The two release domains, each an item-weighted mean over its two cells. The
# headline Average weights the domains 1:1 regardless of their item counts.
DOMAIN_CELLS = {
    "research": ("detour_research", "parallel_research"),
    "engineering": ("detour_engineering", "parallel_engineering"),
}

# Display labels for the leaderboard; anything absent falls back to its raw name.
CELL_LABELS = {
    "detour_engineering": "D-Eng",
    "detour_research": "D-Res",
    "parallel_engineering": "P-Eng",
    "parallel_research": "P-Res",
}
MODEL_LABELS = {
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.4-nano": "GPT-5.4 Nano",
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "grok-4.5": "Grok 4.5",
    "grok-4.20-reasoning": "Grok 4.20 Reasoning",
    "glm-5.2": "GLM-5.2",
    "minimax-m3": "MiniMax M3",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "mistral-medium-3.5": "Mistral Medium 3.5",
}


def strict_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(record.get("ok") is True for record in records)
    parsed = sum(record.get("pred") is not None for record in records)
    errors = sum(bool(record.get("error")) for record in records)
    return {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else None,
        "parsed": parsed,
        "errors": errors,
        "unparsed": total - parsed - errors,
    }


def paired_summary(records: list[dict[str, Any]], cell_by_id: dict[str, str]) -> dict[str, Any]:
    by_key = {(record["item_id"], record["order"]): record for record in records}

    def summarize_ids(item_ids: list[str]) -> dict[str, Any]:
        seeded = [by_key[(item_id, "seeded")] for item_id in item_ids]
        reversed_rows = [by_key[(item_id, "reversed")] for item_id in item_ids]
        pairs = []
        choices: Counter[str] = Counter()
        for left, right in zip(seeded, reversed_rows, strict=True):
            left_ok, right_ok = left.get("ok") is True, right.get("ok") is True
            key = ("C" if left_ok else "W") + ("C" if right_ok else "W")
            pairs.append(key)
            if left.get("pred"):
                choices[f"seeded_{left['pred']}"] += 1
            if right.get("pred"):
                choices[f"reversed_{right['pred']}"] += 1
        pair_counts = Counter(pairs)
        n = len(item_ids)
        return {
            "seeded": strict_aggregate(seeded),
            "reversed": strict_aggregate(reversed_rows),
            "two_order_mean_accuracy": round(
                (sum(row.get("ok") is True for row in seeded + reversed_rows) / (2 * n)),
                6,
            )
            if n
            else None,
            "both_correct": pair_counts["CC"],
            "both_correct_rate": round(pair_counts["CC"] / n, 6) if n else None,
            "paired_outcomes": {key: pair_counts[key] for key in ("CC", "CW", "WC", "WW")},
            "position_choices": dict(sorted(choices.items())),
        }

    ids = sorted(cell_by_id)
    usage_totals: defaultdict[str, int] = defaultdict(int)
    for record in records:
        usage = record.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_totals[key] += int(usage.get(key) or 0)
        details = usage.get("completion_tokens_details") or {}
        usage_totals["reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)
    return {
        "n_items": len(ids),
        "overall": summarize_ids(ids),
        "per_cell": {
            cell: summarize_ids(sorted(item_id for item_id in ids if cell_by_id[item_id] == cell))
            for cell in sorted(set(cell_by_id.values()))
        },
        "usage": dict(usage_totals),
        "input": {
            "max_tokens": max(
                (int(record.get("input_tokens") or 0) for record in records), default=0
            ),
            "max_provider_reported_prompt_tokens": max(
                (int((record.get("usage") or {}).get("prompt_tokens") or 0) for record in records),
                default=0,
            ),
            "truncated_presentations": sum(
                bool(record.get("input_truncated")) for record in records
            ),
            "tokenizer_types": sorted(
                {
                    str(record["tokenizer_type"])
                    for record in records
                    if record.get("tokenizer_type")
                }
            ),
        },
    }


# --- headline average --------------------------------------------------------


def cell_counts(manifest: Any) -> dict[str, int]:
    """Normalise a release manifest (model, dict, or plain mapping) to cell -> n_items."""
    cells = getattr(manifest, "cells", None)
    if cells is None and isinstance(manifest, dict):
        cells = manifest.get("cells", manifest)
    if cells is None:
        raise ValueError("manifest carries no cell counts")
    counts: dict[str, int] = {}
    for cell, record in cells.items():
        if isinstance(record, int):
            counts[cell] = record
        elif isinstance(record, dict):
            counts[cell] = int(record["n_items"])
        else:
            counts[cell] = int(record.n_items)
    return counts


def counts_from_summary(summary: dict[str, Any]) -> dict[str, int]:
    """Read cell sizes back off a summary, for scoring a run without its release."""
    return {cell: int(block["seeded"]["n"]) for cell, block in summary["per_cell"].items()}


def domain_score(summary: dict[str, Any], counts: dict[str, int], domain: str) -> float:
    cells = DOMAIN_CELLS[domain]
    weighted = sum(summary["per_cell"][cell]["both_correct_rate"] * counts[cell] for cell in cells)
    return 100 * weighted / sum(counts[cell] for cell in cells)


def macro_average(summary: dict[str, Any], manifest: Any) -> float:
    """Headline score: the 1:1 mean of the research and engineering subset scores."""
    counts = cell_counts(manifest)
    return (
        domain_score(summary, counts, "research") + domain_score(summary, counts, "engineering")
    ) / 2


# --- leaderboard -------------------------------------------------------------


def leaderboard(results_dir: str | Path) -> list[dict[str, Any]]:
    """Rank every ``<results_dir>/<model>/summary.json`` by the headline Average."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        overall = summary["overall"]
        rows.append(
            {
                "model": path.parent.name,
                "average": macro_average(summary, counts_from_summary(summary)),
                "cells": {
                    cell: block["both_correct_rate"] for cell, block in summary["per_cell"].items()
                },
                "unparsed": overall["seeded"]["unparsed"] + overall["reversed"]["unparsed"],
                "errors": overall["seeded"]["errors"] + overall["reversed"]["errors"],
            }
        )
    rows.sort(key=lambda row: -row["average"])
    return rows


def leaderboard_markdown(rows: list[dict[str, Any]]) -> str:
    cells = sorted({cell for row in rows for cell in row["cells"]})
    header = ["Model", "Average", *(CELL_LABELS.get(cell, cell) for cell in cells), "Unparsed"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|" + "|".join(["---:"] * (len(header) - 1)) + "|",
    ]
    for row in rows:
        values = [
            MODEL_LABELS.get(row["model"], row["model"]),
            f"{row['average']:.1f}",
            *[f"{100 * row['cells'][cell]:.1f}" if cell in row["cells"] else "-" for cell in cells],
            str(row["unparsed"] + row["errors"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
