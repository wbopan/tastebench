from tastebench.schema import Question, check_question


def _valid(**kw) -> Question:
    base = dict(
        id="ok",
        domain="research",
        method="parallel",
        cell="parallel_research",
        source="malt",
        task_id="t",
        query="fix the bug",
        prefix_text="[0] AGENT: investigating",
        prefix_text_short="[0] AGENT: investigating",
        prefix_steps=1,
        choices=["approach A", "approach B"],
        answer="A",
        answer_index=0,
        canary="canary",
    )
    base.update(kw)
    return Question(**base)


def test_valid_question_passes():
    assert check_question(_valid()) == []


def test_answer_b_is_valid_when_the_index_agrees():
    assert check_question(_valid(answer="B", answer_index=1)) == []


def test_detects_a_bad_answer_letter():
    problems = check_question(_valid(answer="C"))
    assert any("not one of" in problem for problem in problems)


def test_detects_answer_disagreeing_with_index():
    problems = check_question(_valid(answer="A", answer_index=1))
    assert any("disagrees with answer_index" in problem for problem in problems)


def test_detects_an_out_of_range_index():
    problems = check_question(_valid(answer="B", answer_index=7))
    assert any("out of range" in problem for problem in problems)


def test_detects_a_wrong_choice_count():
    problems = check_question(_valid(choices=["only one"]))
    assert any("expected 2 choices" in problem for problem in problems)


def test_detects_an_empty_choice():
    problems = check_question(_valid(choices=["approach A", "   "]))
    assert any("choice[1] empty text" in problem for problem in problems)


def test_detects_empty_id_query_and_prefix():
    problems = check_question(_valid(id="", query="  ", prefix_text=""))
    assert any("empty id" in problem for problem in problems)
    assert any("empty query" in problem for problem in problems)
    assert any("empty prefix_text" in problem for problem in problems)


def test_detects_a_cell_that_disagrees_with_method_and_domain():
    problems = check_question(_valid(cell="detour_research"))
    assert any("does not match method/domain" in problem for problem in problems)


def test_detects_a_prefix_with_no_steps():
    problems = check_question(_valid(prefix_steps=0))
    assert any("prefix_steps 0 < 1" in problem for problem in problems)
