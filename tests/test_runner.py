from tastebench.runner import model_request

CONFIG = {
    "models": {
        "reasoner": {"no_temperature": True, "request": {"max_completion_tokens": 4096}},
        "plain": {"temperature": 0},
        "capped": {"no_temperature": True, "max_tokens": 2048},
        "effortful": {"no_temperature": True, "reasoning_effort": "high"},
    }
}


def test_declared_completion_budget_suppresses_the_config_default():
    fields = model_request(CONFIG, "reasoner", 65536)
    assert fields == {"max_completion_tokens": 4096}


def test_config_default_applies_only_without_a_declared_budget():
    assert model_request(CONFIG, "plain", 65536) == {"temperature": 0, "max_tokens": 65536}
    assert model_request(CONFIG, "capped", 65536) == {"max_tokens": 2048}
    assert model_request(CONFIG, "effortful", 100) == {
        "reasoning_effort": "high",
        "max_tokens": 100,
    }


def test_unknown_model_gets_only_the_default_budget():
    assert model_request(CONFIG, "unlisted", 512) == {"max_tokens": 512}
