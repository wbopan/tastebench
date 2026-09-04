import json

import huggingface_hub
import pytest
from conftest import CANARY, row, write_release
from huggingface_hub.errors import GatedRepoError

from tastebench.data import fetch, load_manifest, load_questions, verify_release


def test_verify_release_accepts_a_well_formed_release(tmp_path):
    data = write_release(tmp_path / "data")
    report = verify_release(data)
    assert report["passed"] is True
    assert report["release"] == "test"
    assert report["n_questions"] == 4
    assert report["cells"] == {
        "detour_engineering": 1,
        "detour_research": 1,
        "parallel_engineering": 1,
        "parallel_research": 1,
    }
    assert report["canary"] == CANARY


def test_load_questions_reads_engineering_before_research(tmp_path):
    data = write_release(tmp_path / "data")
    questions = load_questions(data)
    assert [q.domain for q in questions] == [
        "engineering",
        "engineering",
        "research",
        "research",
    ]
    assert questions[0].choices == ["good next step", "bad next step"]


def test_manifest_exposes_row_and_cell_counts(tmp_path):
    manifest = load_manifest(write_release(tmp_path / "data"))
    assert manifest.n_rows == 4
    assert manifest.cell_counts()["parallel_research"] == 1


def test_verify_release_rejects_a_tampered_parquet(tmp_path):
    data = write_release(tmp_path / "data")
    path = data / "data" / "research" / "test-00000-of-00001.parquet"
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_release(data)


def test_verify_release_rejects_a_wrong_row_count(tmp_path):
    data = write_release(tmp_path / "data")
    manifest_path = data / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["domains"]["research"]["n_rows"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="expected 99 rows"):
        verify_release(data)


def test_verify_release_rejects_a_broken_question(tmp_path):
    rows = [row(method, domain) for method, domain in (("detour", "engineering"),)]
    rows.append(row("parallel", "research", answer="B", answer_index=0))
    data = write_release(tmp_path / "data", rows)
    with pytest.raises(ValueError, match="disagrees with answer_index"):
        verify_release(data)


def test_verify_release_rejects_an_inconsistent_canary(tmp_path):
    rows = [
        row("detour", "engineering"),
        row("parallel", "research", canary="a different canary"),
    ]
    data = write_release(tmp_path / "data", rows)
    with pytest.raises(ValueError, match="canary"):
        verify_release(data)


def test_verify_release_rejects_wrong_cell_counts(tmp_path):
    data = write_release(tmp_path / "data")
    manifest_path = data / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["domains"]["engineering"]["cells"] = {"detour_engineering": 2}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cell counts"):
        verify_release(data)


def test_fetch_explains_how_to_get_past_the_gate(monkeypatch, tmp_path):
    class _Response:
        status_code = 403
        headers: dict[str, str] = {}
        request = None

    def _refuse(**kwargs):
        raise GatedRepoError("403 Client Error", response=_Response())

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _refuse)
    with pytest.raises(SystemExit) as excinfo:
        fetch("wenbopan/taste-bench", tmp_path, "v1.0")
    message = str(excinfo.value)
    assert "gated" in message
    assert "hf auth login" in message
