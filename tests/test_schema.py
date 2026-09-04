import pytest
from pydantic import ValidationError

from tastebench.schema import Question


def _question(**kw) -> Question:
    base = dict(
        id="x1",
        domain="engineering",
        method="parallel",
        cell="parallel_engineering",
        source="swebench",
        task_id="t",
        query="do a thing",
        prefix_text="[0] AGENT: hi",
        prefix_text_short="[0] AGENT: hi",
        prefix_steps=1,
        choices=["good", "bad"],
        answer="A",
        answer_index=0,
        canary="canary",
    )
    base.update(kw)
    return Question(**base)


def test_model_carries_exactly_the_published_columns():
    assert tuple(Question.model_fields) == (
        "id",
        "domain",
        "method",
        "cell",
        "source",
        "task_id",
        "query",
        "prefix_text",
        "prefix_text_short",
        "prefix_steps",
        "choices",
        "answer",
        "answer_index",
        "canary",
    )


def test_arity_counts_the_choices():
    assert _question().arity == 2


def test_roundtrip_json():
    question = _question()
    back = Question.model_validate_json(question.model_dump_json())
    assert back == question


def test_missing_column_is_rejected():
    payload = _question().model_dump()
    del payload["canary"]
    with pytest.raises(ValidationError):
        Question.model_validate(payload)
