"""Download, load and verify the published Taste-Bench dataset.

The expected local layout is exactly what the Hugging Face dataset repository
ships::

    <data_dir>/export_manifest.json
    <data_dir>/eval.yaml
    <data_dir>/data/engineering/test-00000-of-00001.parquet
    <data_dir>/data/research/test-00000-of-00001.parquet

Each parquet row is one :class:`~tastebench.schema.Question`. The release carries
no transcripts, rationales or outcomes: everything the protocol needs is already
rendered into ``prefix_text``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tastebench.schema import Question, check_question

HF_DATASET = "wenbopan/taste-bench"
HF_REVISION = "v1.0"
DEFAULT_DATA_DIR = Path("data")

MANIFEST_FILE = "export_manifest.json"

# The parquet column names, in the order the release writes them.
COLUMNS = tuple(Question.model_fields)


# --- export manifest ---------------------------------------------------------


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DomainFile(StrictModel):
    path: Path
    n_rows: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: dict[str, int]


class ExportManifest(StrictModel):
    schema_version: Literal[1]
    release: str
    source_release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Recorded for provenance: the option order was fixed by this seed at export
    # time and is now frozen into the published `choices` column.
    seed: int
    canary: str
    domains: dict[str, DomainFile]
    generated_at: str

    @property
    def n_rows(self) -> int:
        return sum(domain.n_rows for domain in self.domains.values())

    def cell_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for domain in self.domains.values():
            counts.update(domain.cells)
        return counts


def load_manifest(data_dir: str | Path) -> ExportManifest:
    path = Path(data_dir) / MANIFEST_FILE
    return ExportManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- questions ---------------------------------------------------------------


def read_parquet(path: str | Path) -> list[Question]:
    """Read one released parquet file into questions, preserving row order."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=list(COLUMNS))
    return [Question.model_validate(row) for row in table.to_pylist()]


def load_questions(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Question]:
    """Load every released question: engineering first, then research, in file order."""
    root = Path(data_dir)
    manifest = load_manifest(root)
    questions: list[Question] = []
    for domain in sorted(manifest.domains):  # "engineering" < "research"
        questions.extend(read_parquet(root / manifest.domains[domain].path))
    return questions


# --- verification ------------------------------------------------------------


def verify_release(data_dir: str | Path) -> dict[str, Any]:
    """Check the published parquet files against the export manifest.

    Verifies each file's sha256 and row count, the per-cell counts, that the
    contamination canary is identical on every row and equal to the manifest
    canary, and that every row satisfies :func:`check_question`.
    """
    root = Path(data_dir).resolve()
    manifest = load_manifest(root)

    seen: set[str] = set()
    problems: list[str] = []
    cells: dict[str, int] = {}
    domains: dict[str, Any] = {}
    canaries: set[str] = set()

    for domain in sorted(manifest.domains):
        record = manifest.domains[domain]
        path = root / record.path
        actual_hash = sha256(path)
        if actual_hash != record.sha256:
            raise ValueError(f"{domain}: checksum mismatch for {record.path}")
        questions = read_parquet(path)
        if len(questions) != record.n_rows:
            raise ValueError(f"{domain}: expected {record.n_rows} rows, found {len(questions)}")

        duplicates = seen.intersection(q.id for q in questions)
        if duplicates:
            raise ValueError(f"duplicate question ids: {sorted(duplicates)[:10]}")
        seen.update(q.id for q in questions)

        domain_cells: dict[str, int] = {}
        for q in questions:
            if q.domain != domain:
                problems.append(f"{q.id}: domain {q.domain!r} in the {domain!r} file")
            domain_cells[q.cell] = domain_cells.get(q.cell, 0) + 1
            canaries.add(q.canary)
            problems.extend(f"{q.id}: {problem}" for problem in check_question(q))
        if domain_cells != record.cells:
            raise ValueError(f"{domain}: cell counts {domain_cells} != manifest {record.cells}")
        cells.update(domain_cells)
        domains[domain] = {"n_rows": len(questions), "sha256": actual_hash}

    if canaries != {manifest.canary}:
        raise ValueError("canary is missing, not constant across rows, or not the manifest canary")
    if problems:
        raise ValueError(f"{len(problems)} invalid questions; first: " + "; ".join(problems[:5]))
    if len(seen) != manifest.n_rows:
        raise ValueError(f"release count mismatch: manifest={manifest.n_rows}, actual={len(seen)}")

    return {
        "passed": True,
        "release": manifest.release,
        "n_questions": len(seen),
        "cells": dict(sorted(cells.items())),
        "domains": domains,
        "canary": manifest.canary,
    }


# --- download ----------------------------------------------------------------

GATED_HELP = (
    "The Taste-Bench dataset is gated. Request access at "
    "https://huggingface.co/datasets/{repo_id}, then authenticate with `hf auth login` "
    "(or set HF_TOKEN) and retry."
)


def fetch(
    repo_id: str = HF_DATASET,
    local_dir: Path = DEFAULT_DATA_DIR,
    revision: str | None = HF_REVISION,
) -> Path:
    """Download the published dataset snapshot from the Hugging Face Hub."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
        )
    except GatedRepoError as exc:
        raise SystemExit(GATED_HELP.format(repo_id=repo_id)) from exc
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            raise SystemExit(GATED_HELP.format(repo_id=repo_id)) from exc
        raise
    return Path(path)
