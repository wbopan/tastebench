from tastebench.paired import (
    make_token_counter,
    option_blocks,
    parse_final_answer,
    prepare_question,
    render_messages,
)
from tastebench.schema import Question


def _question(question_id: str = "q", prefix_text: str = "prefix", **kw) -> Question:
    base = dict(
        id=question_id,
        domain="engineering",
        method="parallel",
        cell="parallel_engineering",
        source="swebench",
        task_id="task",
        query="fix the bug",
        prefix_text=prefix_text,
        prefix_text_short=prefix_text[:20],
        prefix_steps=3,
        choices=["option zero", "option one"],
        answer="A",
        answer_index=0,
        canary="canary",
    )
    base.update(kw)
    return Question(**base)


def _texts(blocks: str) -> list[str]:
    return [block.split("\n", 1)[1] for block in blocks.split("\n\n")]


def test_option_blocks_render_the_published_order() -> None:
    blocks, correct = option_blocks(_question(), reverse=False)
    assert blocks == "Option A:\noption zero\n\nOption B:\noption one"
    assert correct == "A"


def test_option_blocks_reverse_and_recompute_labels() -> None:
    question = _question()
    seeded, _ = option_blocks(question, reverse=False)
    blocks, correct = option_blocks(question, reverse=True)
    assert blocks == "Option A:\noption one\n\nOption B:\noption zero"
    # The candidate texts are reversed and the labels are recomputed, so the correct
    # letter moves while the correct candidate stays the same.
    assert correct == "B"
    assert _texts(blocks) == list(reversed(_texts(seeded)))


def test_option_blocks_follow_the_answer_index() -> None:
    _, correct = option_blocks(_question(answer="B", answer_index=1), reverse=False)
    assert correct == "B"
    _, flipped = option_blocks(_question(answer="B", answer_index=1), reverse=True)
    assert flipped == "A"


def test_prompt_only_requests_final_answer() -> None:
    messages, _ = render_messages(
        _question("lean"),
        "full prefix",
        reverse=False,
        system_prompt="Judge the decision.",
        user_template="{query}\n{prefix}\n{options}\nReturn exactly one line: ANSWER: X",
    )
    text = "\n".join(message["content"] for message in messages)
    assert "Think briefly" not in text
    assert "think carefully" not in text.lower()
    assert text.endswith("ANSWER: X")


def test_reversed_order_is_the_exact_reverse() -> None:
    kwargs = dict(system_prompt="Judge.", user_template="{query}\n{prefix}\n{options}")
    seeded, seeded_correct = render_messages(_question("pair"), "prefix", reverse=False, **kwargs)
    flipped, flipped_correct = render_messages(_question("pair"), "prefix", reverse=True, **kwargs)
    assert {seeded_correct, flipped_correct} == {"A", "B"}
    assert seeded[1]["content"] != flipped[1]["content"]


def test_parse_final_answer_rejects_incidental_letters() -> None:
    assert parse_final_answer("ANSWER: B", 2) == "B"
    assert parse_final_answer("A", 2) == "A"
    assert parse_final_answer("Option A looks plausible, but B may work", 2) is None


def test_prepare_question_uses_full_prefix_below_token_cap() -> None:
    question = _question("full", prefix_text="line one\nline two")

    def count(messages):
        return len(messages[1]["content"]), "test"

    prepared = prepare_question(
        question,
        reverse=False,
        system_prompt="Judge.",
        user_template="{query}\n{prefix}\n{options}\nANSWER: X",
        max_input_tokens=10_000,
        count_tokens=count,
    )
    assert prepared.truncated is False
    assert prepared.full_prefix_chars == prepared.visible_prefix_chars


def test_prepare_question_truncates_middle_to_token_cap() -> None:
    prefix = "\n".join(f"line {index} " + "x" * 20 for index in range(100))
    question = _question("trim", prefix_text=prefix)

    def count(messages):
        return len(messages[1]["content"]), "test"

    prepared = prepare_question(
        question,
        reverse=False,
        system_prompt="Judge.",
        user_template="{query}\n{prefix}\n{options}\nANSWER: X",
        max_input_tokens=1_300,
        count_tokens=count,
    )
    assert prepared.truncated is True
    assert prepared.input_tokens <= 1_300
    assert "middle transcript lines omitted" in prepared.messages[1]["content"]


def test_token_counter_applies_the_safety_multiplier(monkeypatch) -> None:
    class _Encoding:
        def encode(self, text, disallowed_special=()):
            return text.split()

    monkeypatch.setattr("tastebench.paired._encoding", lambda: _Encoding())
    count = make_token_counter(1.5)
    tokens, tokenizer = count([{"role": "user", "content": "a b c d"}])
    assert tokens == 6
    assert tokenizer == "o200k_base*1.5"
