"""Load and verify a released Taste-Bench snapshot.

The expected local layout is exactly what the dataset repository ships::

    <data_dir>/manifest.json
    <data_dir>/items/<cell>.jsonl
    <data_dir>/transcripts.manifest.json
    <data_dir>/transcripts/<traj_id>.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tastebench.schema import Item

HF_DATASET = "wbopan/tastebench"
DEFAULT_DATA_DIR = Path("data")


# --- item files --------------------------------------------------------------


def load_items(path: str | Path) -> list[Item]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Item.model_validate_json(line))
    return out


def load_all_items(root: str | Path) -> list[Item]:
    """Load and concatenate every ``*.jsonl`` under ``root``."""
    items: list[Item] = []
    for p in sorted(Path(root).glob("*.jsonl")):
        items.extend(load_items(p))
    return items


# --- release manifest --------------------------------------------------------


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReleaseFile(StrictModel):
    path: Path
    n_items: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChecksummedFile(StrictModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReleaseManifest(StrictModel):
    schema_version: Literal[1]
    release: str
    status: str
    n_items: int = Field(ge=0)
    unique_ids: int = Field(ge=0)
    cells: dict[str, ReleaseFile]
    transcript_manifest: ChecksummedFile
    # Recorded for provenance only; the released snapshot does not ship these files.
    source_pools: dict[str, ReleaseFile] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        item_id = value.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{path}:{line_number} has no string id")
        ids.append(item_id)
    return ids


def load_manifest(data_dir: str | Path) -> ReleaseManifest:
    path = Path(data_dir) / "manifest.json"
    return ReleaseManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def verify_release(data_dir: str | Path) -> dict[str, Any]:
    """Check every released item file against the manifest checksums and counts."""
    release_dir = Path(data_dir).resolve()
    manifest = load_manifest(release_dir)
    seen: set[str] = set()
    cells: dict[str, Any] = {}
    for cell, record in manifest.cells.items():
        path = release_dir / record.path
        ids = jsonl_ids(path)
        duplicates = seen.intersection(ids)
        if duplicates:
            raise ValueError(f"cross-cell duplicate ids: {sorted(duplicates)[:10]}")
        seen.update(ids)
        actual_hash = sha256(path)
        if actual_hash != record.sha256:
            raise ValueError(f"{cell}: checksum mismatch")
        if len(ids) != record.n_items:
            raise ValueError(f"{cell}: expected {record.n_items} items, found {len(ids)}")
        cells[cell] = {"n_items": len(ids), "sha256": actual_hash}

    transcript_hash = sha256(release_dir / manifest.transcript_manifest.path)
    if transcript_hash != manifest.transcript_manifest.sha256:
        raise ValueError("transcript manifest checksum mismatch")

    if len(seen) != manifest.n_items or len(seen) != manifest.unique_ids:
        raise ValueError(f"release count mismatch: manifest={manifest.n_items}, actual={len(seen)}")
    return {
        "passed": True,
        "release": manifest.release,
        "n_items": len(seen),
        "cells": cells,
        "transcript_manifest_sha256": transcript_hash,
    }


# --- download ----------------------------------------------------------------


def fetch(
    repo_id: str = HF_DATASET,
    local_dir: Path = DEFAULT_DATA_DIR,
    revision: str | None = None,
) -> Path:
    """Download the released snapshot from the Hugging Face Hub into ``local_dir``."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
    )
    return Path(path)
