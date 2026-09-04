from tastebench.schema import SCHEMA_VERSION, Choice, Item, Outcome


def _item(**kw):
    base = dict(
        id="x1",
        method="parallel",
        dataset="swebench",
        task_id="t",
        query="do a thing",
        prefix_text="[0] AGENT: hi",
        choices=[
            Choice(text="good", is_correct=True, outcome=Outcome(passed=True)),
            Choice(text="bad", is_correct=False, outcome=Outcome(passed=False)),
        ],
    )
    base.update(kw)
    return Item(**base)


def test_arity_is_computed():
    it = _item()
    assert it.arity == 2
    it3 = _item(choices=[Choice(text=str(i), is_correct=(i == 0)) for i in range(3)])
    assert it3.arity == 3


def test_correct_index():
    assert _item().correct_index() == 0
    two = _item(choices=[Choice(text="a", is_correct=True), Choice(text="b", is_correct=True)])
    assert two.correct_index() is None  # ambiguous


def test_roundtrip_json_includes_arity():
    it = _item()
    dumped = it.model_dump_json()
    assert '"arity":2' in dumped
    back = Item.model_validate_json(dumped)
    assert back.id == it.id
    assert back.arity == 2
    assert back.schema_version == SCHEMA_VERSION
