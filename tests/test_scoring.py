from tastebench.scoring import (
    cell_counts,
    counts_from_summary,
    leaderboard,
    macro_average,
    paired_summary,
    strict_aggregate,
)

CELLS = {
    "detour_research": 2,
    "parallel_research": 2,
    "detour_engineering": 2,
    "parallel_engineering": 2,
}


def _records():
    """Two items per cell; the first is right in both orders, the second only seeded."""
    records = []
    for cell in CELLS:
        for index in range(2):
            item_id = f"{cell}-{index}"
            for order in ("seeded", "reversed"):
                ok = index == 0 or order == "seeded"
                records.append(
                    {
                        "item_id": item_id,
                        "cell": cell,
                        "order": order,
                        "correct": "A",
                        "pred": "A" if ok else "B",
                        "ok": ok,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                        "input_tokens": 12,
                        "tokenizer_type": "o200k_base*1.02",
                    }
                )
    return records


def test_strict_aggregate_counts_errors_and_unparsed_as_wrong():
    rows = [
        {"ok": True, "pred": "A"},
        {"ok": None, "pred": None},
        {"ok": None, "pred": None, "error": "boom"},
    ]
    assert strict_aggregate(rows) == {
        "n": 3,
        "correct": 1,
        "accuracy": round(1 / 3, 6),
        "parsed": 1,
        "errors": 1,
        "unparsed": 1,
    }


def test_paired_summary_counts_both_orders_per_cell():
    records = _records()
    summary = paired_summary(records, {r["item_id"]: r["cell"] for r in records})
    assert summary["n_items"] == 8
    overall = summary["overall"]
    assert overall["seeded"]["correct"] == 8
    assert overall["reversed"]["correct"] == 4
    assert overall["both_correct"] == 4
    assert overall["paired_outcomes"] == {"CC": 4, "CW": 4, "WC": 0, "WW": 0}
    assert overall["two_order_mean_accuracy"] == 0.75
    assert summary["usage"]["total_tokens"] == 16 * 11
    for block in summary["per_cell"].values():
        assert block["both_correct"] == 1
        assert block["both_correct_rate"] == 0.5


def test_macro_average_weights_domains_one_to_one():
    records = _records()
    summary = paired_summary(records, {r["item_id"]: r["cell"] for r in records})
    manifest = {"cells": {cell: {"n_items": n} for cell, n in CELLS.items()}}
    assert cell_counts(manifest) == CELLS
    assert counts_from_summary(summary) == CELLS
    assert macro_average(summary, manifest) == 50.0

    # A domain's cells are item-weighted; the two domains are then averaged 1:1.
    skewed = {
        "cells": {
            "detour_research": {"n_items": 1},
            "parallel_research": {"n_items": 1},
            "detour_engineering": {"n_items": 100},
            "parallel_engineering": {"n_items": 100},
        }
    }
    summary["per_cell"]["detour_research"]["both_correct_rate"] = 1.0
    summary["per_cell"]["parallel_research"]["both_correct_rate"] = 1.0
    # research 100, engineering 50 -> 75
    assert macro_average(summary, skewed) == 75.0


def test_cell_counts_reads_the_export_manifest_shape():
    export_manifest = {
        "domains": {
            "engineering": {"cells": {"detour_engineering": 2, "parallel_engineering": 2}},
            "research": {"cells": {"detour_research": 2, "parallel_research": 2}},
        }
    }
    assert cell_counts(export_manifest) == CELLS
    assert cell_counts(CELLS) == CELLS


def test_leaderboard_ranks_published_summaries(tmp_path):
    records = _records()
    summary = paired_summary(records, {r["item_id"]: r["cell"] for r in records})
    import json

    for name, rate in (("weak", 0.25), ("strong", 0.75)):
        for block in summary["per_cell"].values():
            block["both_correct_rate"] = rate
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(summary))

    rows = leaderboard(tmp_path)
    assert [row["model"] for row in rows] == ["strong", "weak"]
    assert rows[0]["average"] == 75.0
    assert rows[0]["unparsed"] == 0
