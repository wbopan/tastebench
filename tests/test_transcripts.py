import json

from tastebench.transcripts import TranscriptStore


def test_store_roundtrip_and_manifest(tmp_path):
    store = TranscriptStore(tmp_path / "store")
    steps = [{"i": 0, "kind": "agent", "text": "hi"}]
    store.put("traj_x", steps, meta={"source": "test"})
    assert store.has("traj_x")
    assert store.get("traj_x") == steps
    store.write_manifest()
    assert store.verify() == []
    # tamper -> verify catches it
    (tmp_path / "store" / "traj_x.json").write_text(
        json.dumps({"traj": "traj_x", "steps": [{"i": 0, "kind": "agent", "text": "changed"}]})
    )
    store2 = TranscriptStore(tmp_path / "store")
    assert store2.verify()  # nonempty -> mismatch detected
