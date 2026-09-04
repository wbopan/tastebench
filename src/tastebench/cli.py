"""`tb` -- download, validate, run and score the Taste-Bench release."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from tastebench import runner
from tastebench.data import (
    DEFAULT_DATA_DIR,
    HF_DATASET,
    fetch,
    load_all_items,
    verify_release,
)
from tastebench.schema import check_item
from tastebench.scoring import (
    counts_from_summary,
    leaderboard_markdown,
    macro_average,
    paired_summary,
)
from tastebench.scoring import leaderboard as leaderboard_rows
from tastebench.transcripts import TranscriptStore

app = typer.Typer(add_completion=False, help="Taste-Bench evaluation CLI.")


@app.command()
def download(
    repo_id: str = typer.Option(HF_DATASET, help="Hugging Face dataset repository."),
    data: Path = typer.Option(DEFAULT_DATA_DIR, help="Local directory to download into."),
    revision: str = typer.Option("", help="Optional revision, branch, or tag."),
) -> None:
    """Download the released items and transcripts."""
    path = fetch(repo_id, data, revision or None)
    typer.echo(f"downloaded {repo_id} to {path}")


@app.command()
def validate(
    data: Path = typer.Option(DEFAULT_DATA_DIR, help="Release directory."),
) -> None:
    """Check manifest checksums and every item invariant against the transcripts."""
    report = verify_release(data)
    items = load_all_items(data / "items")
    store = TranscriptStore(data / "transcripts")
    grounded = store.is_populated()
    problems = []
    for item in items:
        for problem in check_item(item, store if grounded else None):
            problems.append(f"{item.id}: {problem}")
    for problem in problems[:20]:
        typer.echo(problem)
    if problems:
        typer.echo(f"{len(problems)} problems")
        raise typer.Exit(1)
    typer.echo(
        f"release {report['release']}: {len(items)} items valid, "
        f"{len(report['cells'])} cells, checksums match"
        + ("" if grounded else " (transcripts absent: structural checks only)")
    )


@app.command()
def run(
    model: str = typer.Option(..., help="Model id sent to the endpoint."),
    config: Path = typer.Option(Path("configs/eval.yaml"), help="Evaluation config."),
    models: Path = typer.Option(Path("configs/models.yaml"), help="Per-model request overrides."),
    api: str = typer.Option("", help="OpenAI-compatible chat/completions or responses URL."),
    data: Path = typer.Option(DEFAULT_DATA_DIR, help="Release directory."),
    out: Path = typer.Option(Path("runs"), help="Root directory for run outputs."),
    limit: int = typer.Option(0, help="Evaluate only the first N items (smoke runs)."),
) -> None:
    """Evaluate one model over both option orders for every release item."""
    out_dir = runner.run(
        model=model,
        config_path=config,
        models_path=models,
        api=api or None,
        data_dir=data,
        out_root=out,
        limit=limit,
    )
    typer.echo(f"wrote {out_dir}")
    _report(json.loads((out_dir / "summary.json").read_text(encoding="utf-8")))


@app.command()
def score(run_dir: Path = typer.Argument(..., help="A run directory containing raw/.")) -> None:
    """Recompute summary.json from the raw records and print the headline numbers."""
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "raw").glob("*/*.json"))
    ]
    if not records:
        raise typer.BadParameter(f"no raw records under {run_dir / 'raw'}")
    cell_by_id = {record["item_id"]: record["cell"] for record in records}
    summary = paired_summary(records, cell_by_id)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _report(summary)


@app.command()
def leaderboard(
    results_dir: Path = typer.Argument(Path("results"), help="Directory of <model>/summary.json."),
) -> None:
    """Print a markdown leaderboard over published summaries."""
    rows = leaderboard_rows(results_dir)
    if not rows:
        raise typer.BadParameter(f"no <model>/summary.json under {results_dir}")
    typer.echo(leaderboard_markdown(rows))


def _report(summary: dict) -> None:
    overall = summary["overall"]
    typer.echo(
        f"items {summary['n_items']}  "
        f"seeded {overall['seeded']['correct']}/{overall['seeded']['n']}  "
        f"reversed {overall['reversed']['correct']}/{overall['reversed']['n']}  "
        f"both_correct {overall['both_correct']} "
        f"({100 * (overall['both_correct_rate'] or 0):.1f}%)  "
        f"errors {overall['seeded']['errors'] + overall['reversed']['errors']}  "
        f"unparsed {overall['seeded']['unparsed'] + overall['reversed']['unparsed']}"
    )
    for cell, block in sorted(summary["per_cell"].items()):
        typer.echo(
            f"  {cell:<22} both_correct {block['both_correct']}/{block['seeded']['n']} "
            f"({100 * (block['both_correct_rate'] or 0):.1f}%)"
        )
    if set(summary["per_cell"]) == {
        "detour_research",
        "parallel_research",
        "detour_engineering",
        "parallel_engineering",
    }:
        typer.echo(f"Average {macro_average(summary, counts_from_summary(summary)):.1f}")


if __name__ == "__main__":
    app()
