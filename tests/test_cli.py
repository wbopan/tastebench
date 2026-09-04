import json

from typer.testing import CliRunner

from tastebench.cli import app
from tastebench.data import sha256

runner = CliRunner()

CELLS = ("detour_research", "parallel_research", "detour_engineering", "parallel_engineering")


def _item(item_id: str, traj: str) -> dict:
    return {
        "schema_version": 1,
        "id": item_id,
        "method": "detour",
        "dataset": "swebench",
        "task_id": "t",
        "query": "fix the bug",
        "reference_traj": traj,
        "breakpoint_step": 1,
        "prefix_text": "[0] AGENT: investigating",
        "choices": [
            {"text": "good next step", "is_correct": True},
            {"text": "bad next step", "is_correct": False},
        ],
    }


def _release(root):
    """Build a tiny release with the layout the dataset ships."""
    from tastebench.transcripts import TranscriptStore

    (root / "items").mkdir(parents=True)
    store = TranscriptStore(root / "transcripts")
    cells = {}
    for index, cell in enumerate(CELLS):
        traj = f"traj_{index}"
        store.put(traj, [{"i": 0, "kind": "agent", "text": "investigating"}], meta={"source": cell})
        path = root / "items" / f"{cell}.jsonl"
        path.write_text(json.dumps(_item(f"{cell}-0", traj)) + "\n", encoding="utf-8")
        cells[cell] = {"path": f"items/{cell}.jsonl", "n_items": 1, "sha256": sha256(path)}
    manifest_path = store.write_manifest()
    manifest_path.replace(root / "transcripts.manifest.json")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release": "test",
                "status": "canonical",
                "n_items": len(CELLS),
                "unique_ids": len(CELLS),
                "cells": cells,
                "transcript_manifest": {
                    "path": "transcripts.manifest.json",
                    "sha256": sha256(root / "transcripts.manifest.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_validate_accepts_a_well_formed_release(tmp_path):
    data = _release(tmp_path / "data")
    result = runner.invoke(app, ["validate", "--data", str(data)])
    assert result.exit_code == 0, result.output
    assert "4 items valid" in result.output


def test_validate_rejects_a_tampered_item(tmp_path):
    data = _release(tmp_path / "data")
    path = data / "items" / "detour_research.jsonl"
    item = json.loads(path.read_text())
    item["choices"][1]["is_correct"] = True
    path.write_text(json.dumps(item) + "\n")
    result = runner.invoke(app, ["validate", "--data", str(data)])
    assert result.exit_code != 0


def test_score_recomputes_summary_from_raw(tmp_path):
    run_dir = tmp_path / "run"
    for cell in CELLS:
        for order in ("seeded", "reversed"):
            directory = run_dir / "raw" / order
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{cell}-0.json").write_text(
                json.dumps(
                    {
                        "item_id": f"{cell}-0",
                        "cell": cell,
                        "order": order,
                        "correct": "A",
                        "pred": "A",
                        "ok": True,
                    }
                )
            )
    result = runner.invoke(app, ["score", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "both_correct 4" in result.output
    assert "Average 100.0" in result.output
    assert json.loads((run_dir / "summary.json").read_text())["n_items"] == 4


def test_leaderboard_renders_a_markdown_table(tmp_path):
    summary = {
        "n_items": 4,
        "overall": {
            "seeded": {"n": 4, "correct": 4, "errors": 0, "unparsed": 0},
            "reversed": {"n": 4, "correct": 4, "errors": 0, "unparsed": 0},
            "both_correct": 4,
            "both_correct_rate": 1.0,
        },
        "per_cell": {
            cell: {"seeded": {"n": 1}, "both_correct_rate": 1.0, "both_correct": 1}
            for cell in CELLS
        },
    }
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "summary.json").write_text(json.dumps(summary))
    result = runner.invoke(app, ["leaderboard", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "| m1 | 100.0 |" in result.output
