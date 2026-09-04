import json

from conftest import CELLS, row, write_release
from typer.testing import CliRunner

from tastebench.cli import app

runner = CliRunner()

CELL_NAMES = tuple(f"{method}_{domain}" for method, domain in CELLS)


def test_validate_accepts_a_well_formed_release(tmp_path):
    data = write_release(tmp_path / "data")
    result = runner.invoke(app, ["validate", "--data", str(data)])
    assert result.exit_code == 0, result.output
    assert "4 questions valid" in result.output
    assert "4 cells, checksums match" in result.output
    for cell in CELL_NAMES:
        assert f"{cell:<22} 1" in result.output


def test_validate_rejects_a_broken_question(tmp_path):
    rows = [row("detour", "engineering"), row("parallel", "research", answer="B")]
    data = write_release(tmp_path / "data", rows)
    result = runner.invoke(app, ["validate", "--data", str(data)])
    assert result.exit_code != 0
    assert "disagrees with answer_index" in result.output


def test_score_recomputes_summary_from_raw(tmp_path):
    run_dir = tmp_path / "run"
    for cell in CELL_NAMES:
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
            for cell in CELL_NAMES
        },
    }
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "summary.json").write_text(json.dumps(summary))
    result = runner.invoke(app, ["leaderboard", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "| m1 | 100.0 |" in result.output
