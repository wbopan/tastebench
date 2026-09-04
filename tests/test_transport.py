from io import BytesIO
from urllib.error import HTTPError

import pytest

from tastebench.transport import ChatTransport


def _transport(api: str, **kwargs) -> ChatTransport:
    return ChatTransport(api=api, api_key="unused", model="test", **kwargs)


def test_an_api_url_is_required() -> None:
    with pytest.raises(ValueError):
        ChatTransport(api="", api_key="k", model="m")


def test_responses_transport_keeps_only_final_text_and_normalizes_usage(monkeypatch) -> None:
    transport = _transport("https://example.test/v1/responses")
    monkeypatch.setattr(
        transport,
        "_post",
        lambda *_: {
            "model": "test",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"text": "private"}]},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ANSWER: B"}],
                },
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 125,
                "output_tokens_details": {"reasoning_tokens": 20},
            },
        },
    )
    result = transport.complete([{"role": "user", "content": "pick"}])
    assert result["text"] == "ANSWER: B"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 100
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 20
    assert "private" not in str(result)


def _reasoning_only_response(*_args, **_kwargs):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "", "reasoning_content": "private analysis\nANSWER: A"},
            }
        ]
    }


def test_reasoning_content_fallback_extracts_only_explicit_answer(monkeypatch) -> None:
    transport = _transport(
        "https://example.test/v1/chat/completions", reasoning_content_fallback=True
    )
    monkeypatch.setattr(transport, "_post", _reasoning_only_response)
    result = transport.complete([{"role": "user", "content": "pick"}])
    assert result["text"] == "ANSWER: A"
    assert "private analysis" not in str(result)


def test_reasoning_content_is_not_exposed_without_the_flag(monkeypatch) -> None:
    transport = _transport("https://example.test/v1/chat/completions")
    monkeypatch.setattr(transport, "_post", _reasoning_only_response)
    result = transport.complete([{"role": "user", "content": "pick"}])
    assert result["text"] == ""
    assert "private analysis" not in str(result)


def test_provider_safety_refusal_is_an_unparsed_model_outcome(monkeypatch) -> None:
    transport = _transport("https://example.test/v1/chat/completions")
    attempts = 0

    def refuse(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        body = b'{"error":{"message":"Failed check: SAFETY_CHECK_TYPE_CYBER"}}'
        raise HTTPError("https://example.test", 403, "Forbidden", {}, BytesIO(body))

    monkeypatch.setattr("tastebench.transport.urlrequest.urlopen", refuse)
    result = transport.complete([{"role": "user", "content": "pick"}])
    assert attempts == 1
    assert result["text"] == ""
    assert result["finish_reason"] == "content_filter"
    assert result["provider_metadata"] == {"refusal": "safety_policy"}
    assert "SAFETY_CHECK_TYPE_CYBER" not in str(result)


def test_http_failure_is_retried_then_raised(monkeypatch) -> None:
    transport = _transport("https://example.test/v1/chat/completions", retries=2)
    attempts = 0

    def fail(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise HTTPError(
            "https://example.test",
            429,
            "Too Many Requests",
            {},
            BytesIO(b'{"error":{"code":"RateLimitReached"}}'),
        )

    monkeypatch.setattr("tastebench.transport.urlrequest.urlopen", fail)
    monkeypatch.setattr("tastebench.transport.time.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError) as excinfo:
        transport.complete([{"role": "user", "content": "pick"}])
    assert attempts == 2
    assert "HTTP 429" in str(excinfo.value)
