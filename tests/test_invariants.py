from tastebench.schema import Choice, Item, Outcome, check_item


def _valid():
    return Item(
        id="ok",
        method="parallel",
        dataset="swebench",
        task_id="t",
        query="fix the bug",
        prefix_text="[0] AGENT: investigating",
        reference_traj="traj_a",
        breakpoint_step=1,
        choices=[
            Choice(text="approach A", is_correct=True, traj="traj_a", outcome=Outcome(passed=True)),
            Choice(
                text="approach B", is_correct=False, traj="traj_b", outcome=Outcome(passed=False)
            ),
        ],
    )


class _Store:
    def __init__(self, trajs):
        self._t = trajs

    def has(self, t):
        return t in self._t

    def get(self, t):
        return self._t[t]


def test_valid_item_passes():
    assert check_item(_valid()) == []


def test_detects_two_correct():
    it = _valid()
    it.choices[1].is_correct = True
    assert any("exactly 1 correct" in p for p in check_item(it))


def test_detects_empty_query_and_prefix():
    it = _valid()
    it.query = "  "
    it.prefix_text = ""
    probs = check_item(it)
    assert any("empty query" in p for p in probs)
    assert any("empty prefix_text" in p for p in probs)


def test_detects_correct_from_failing_rollout():
    it = _valid()
    it.choices[0].outcome = Outcome(passed=False)
    assert any("failing rollout" in p for p in check_item(it))


def test_detects_rationale_leak():
    it = _valid()
    it.choices[0].rationale = "investigating"  # appears verbatim in prefix_text
    assert any("rationale text leaked" in p for p in check_item(it))


def test_store_breakpoint_bounds():
    it = _valid()
    it.breakpoint_step = 99
    store = _Store({"traj_a": [{}, {}], "traj_b": [{}]})
    assert any("out of bounds" in p for p in check_item(it, store))


def test_store_missing_choice_traj():
    it = _valid()
    store = _Store({"traj_a": [{}, {}]})  # traj_b missing
    assert any("traj_b not in store" in p for p in check_item(it, store))
