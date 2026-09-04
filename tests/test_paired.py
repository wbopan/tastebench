from tastebench.paired import (
    make_token_counter,
    parse_final_answer,
    prepare_question,
    render_messages,
)
from tastebench.schema import Choice, Item


def _item(item_id: str, n: int = 2) -> Item:
    return Item(
        id=item_id,
        method="parallel",
        dataset="swebench",
        task_id="task",
        query="fix the bug",
        choices=[Choice(text=f"option {index}", is_correct=index == 0) for index in range(n)],
    )


def test_prompt_only_requests_final_answer() -> None:
    messages, _ = render_messages(
        _item("lean", 2),
        "full prefix",
        seed=1234,
        reverse=False,
        system_prompt="Judge the decision.",
        user_template="{query}\n{prefix}\n{options}\nReturn exactly one line: ANSWER: X",
    )
    text = "\n".join(message["content"] for message in messages)
    assert "Think briefly" not in text
    assert "think carefully" not in text.lower()
    assert text.endswith("ANSWER: X")


def test_reversed_order_is_the_exact_reverse() -> None:
    kwargs = dict(
        system_prompt="Judge.",
        user_template="{query}\n{prefix}\n{options}",
    )
    seeded, seeded_correct = render_messages(
        _item("pair", 2), "prefix", seed=1234, reverse=False, **kwargs
    )
    flipped, flipped_correct = render_messages(
        _item("pair", 2), "prefix", seed=1234, reverse=True, **kwargs
    )
    assert {seeded_correct, flipped_correct} == {"A", "B"}
    assert seeded[1]["content"] != flipped[1]["content"]


def test_parse_final_answer_rejects_incidental_letters() -> None:
    assert parse_final_answer("ANSWER: B", 2) == "B"
    assert parse_final_answer("A", 2) == "A"
    assert parse_final_answer("Option A looks plausible, but B may work", 2) is None


def test_prepare_question_uses_full_prefix_below_token_cap() -> None:
    item = _item("full", 2)

    def count(messages):
        return len(messages[1]["content"]), "test"

    question = prepare_question(
        item,
        "line one\nline two",
        seed=1234,
        reverse=False,
        system_prompt="Judge.",
        user_template="{query}\n{prefix}\n{options}\nANSWER: X",
        max_input_tokens=10_000,
        count_tokens=count,
    )
    assert question.truncated is False
    assert question.full_prefix_chars == question.visible_prefix_chars


def test_prepare_question_truncates_middle_to_token_cap() -> None:
    item = _item("trim", 2)
    prefix = "\n".join(f"line {index} " + "x" * 20 for index in range(100))

    def count(messages):
        return len(messages[1]["content"]), "test"

    question = prepare_question(
        item,
        prefix,
        seed=1234,
        reverse=False,
        system_prompt="Judge.",
        user_template="{query}\n{prefix}\n{options}\nANSWER: X",
        max_input_tokens=1_300,
        count_tokens=count,
    )
    assert question.truncated is True
    assert question.input_tokens <= 1_300
    assert "middle transcript lines omitted" in question.messages[1]["content"]


def test_token_counter_applies_the_safety_multiplier(monkeypatch) -> None:
    class _Encoding:
        def encode(self, text, disallowed_special=()):
            return text.split()

    monkeypatch.setattr("tastebench.paired._encoding", lambda: _Encoding())
    count = make_token_counter(1.5)
    tokens, tokenizer = count([{"role": "user", "content": "a b c d"}])
    assert tokens == 6
    assert tokenizer == "o200k_base*1.5"
