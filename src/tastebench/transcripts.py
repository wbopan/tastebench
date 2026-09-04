"""Content-addressed store for distilled trajectory transcripts.

Items reference transcripts by ``traj`` id instead of embedding them, keeping item
files small and diffable. One JSON file per trajectory under ``root``; a released
``transcripts.manifest.json`` records a checksum + step count + source for each, so a
store can be verified against the release it belongs to.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def checksum(steps: list[dict[str, Any]]) -> str:
    blob = json.dumps(steps, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class TranscriptStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def _path(self, traj: str) -> Path:
        return self.root / f"{traj}.json"

    def has(self, traj: str) -> bool:
        return traj in self._cache or self._path(traj).exists()

    def is_populated(self) -> bool:
        """True if the store holds any transcripts (not just a manifest)."""
        return any(p.name != "manifest.json" for p in self.root.glob("*.json"))

    def put(self, traj: str, steps: list[dict[str, Any]], meta: dict | None = None) -> str:
        payload = {"traj": traj, "n_steps": len(steps), "meta": meta or {}, "steps": steps}
        self._path(traj).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self._cache[traj] = steps
        return checksum(steps)

    def get(self, traj: str) -> list[dict[str, Any]]:
        if traj not in self._cache:
            data = json.loads(self._path(traj).read_text(encoding="utf-8"))
            self._cache[traj] = data["steps"]
        return self._cache[traj]

    def write_manifest(self) -> Path:
        """(Re)write ``manifest.json`` from the files currently in the store."""
        entries = {}
        for p in sorted(self.root.glob("*.json")):
            if p.name == "manifest.json":
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            steps = data.get("steps", [])
            entries[data.get("traj", p.stem)] = {
                "n_steps": len(steps),
                "checksum": checksum(steps),
                "source": data.get("meta", {}).get("source"),
            }
        mpath = self.root / "manifest.json"
        mpath.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
        return mpath

    def verify(self, manifest: dict[str, Any] | None = None) -> list[str]:
        """Check store files against a manifest; return mismatches."""
        if manifest is None:
            mpath = self.root / "manifest.json"
            if not mpath.exists():
                return ["manifest.json missing"]
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        problems = []
        for traj, entry in (manifest or {}).items():
            if not self.has(traj):
                problems.append(f"{traj}: missing from store")
                continue
            if checksum(self.get(traj)) != entry["checksum"]:
                problems.append(f"{traj}: checksum mismatch")
        return problems
