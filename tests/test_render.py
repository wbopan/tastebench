from tastebench.render import render_prefix, render_steps, trim_prefix


def test_render_canonical_steps():
    steps = [
        {"i": 0, "kind": "agent", "text": "I'll start by reading the prompt."},
        {"i": 1, "kind": "cmd", "cmd": "cat prompt.md", "exit": 0, "out": "task text"},
        {"i": 2, "kind": "file", "paths": ["update:src/a.py"]},
    ]
    out = render_steps(steps)
    assert "[0] AGENT: I'll start by reading the prompt." in out
    assert "[1] $ cat prompt.md   (exit=0)" in out
    assert "    task text" in out
    assert "[2] EDIT: update:src/a.py" in out


def test_render_tolerates_legacy_keys():
    # legacy bundle schema: k / t / changes
    steps = [
        {"i": 0, "k": "agent", "t": "legacy message"},
        {"i": 1, "k": "file", "changes": ["add:x.py"]},
    ]
    out = render_steps(steps)
    assert "[0] AGENT: legacy message" in out
    assert "[1] EDIT: add:x.py" in out


def test_trim_prefix_short_passthrough():
    assert trim_prefix("a\nb\nc", 1000) == "a\nb\nc"


def test_trim_prefix_long_keeps_head_and_tail():
    lines = [f"line{i}" for i in range(200)]
    text = "\n".join(lines)
    out = trim_prefix(text, 400)
    assert out.startswith("line0")
    assert "earlier steps omitted" in out
    assert "line199" in out
    assert len(out) <= 400 + 100  # budget plus marker slack


def test_render_prefix_respects_breakpoint():
    steps = [{"i": i, "kind": "agent", "text": f"m{i}"} for i in range(5)]
    out = render_prefix(steps, 2)
    assert "m0" in out and "m1" in out
    assert "m2" not in out
