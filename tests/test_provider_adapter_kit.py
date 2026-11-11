"""Tests for provider_adapter_kit."""

from __future__ import annotations

import json
from collections.abc import Sequence
from unittest import mock
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest

from provider_adapter_kit import (
    FakeLLM,
    Message,
    ProviderError,
    RecordedLLM,
    RecordingExhaustedError,
    Request,
    Response,
    UrllibTransport,
    make_adapter,
)


class FixedTransport:
    """Transport that returns a fixed response."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body
        self.calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def post(
        self, url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> tuple[int, bytes]:
        self.calls.append((url, headers, body, timeout_s))
        return self.status, self.body


def req(
    messages: list[Message] | None = None,
    model: str = "claude-x",
    max_tokens: int = 64,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> Request:
    if messages is None:
        messages = [{"role": "user", "content": "Hi"}]
    return Request(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or [],
    )


# ---------------------------------------------------------------------------
# Request body building tests (acceptance items 1-3)


def test_anthropic_build_acceptance() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(
        model="claude-x",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=64,
    )
    url, headers, body = a.build(r)
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert body == (
        b'{"model":"claude-x","max_tokens":64,'
        b'"system":"Be brief.",'
        b'"messages":[{"role":"user","content":"Hi"}]}'
    )


def test_anthropic_build_body_literal() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(
        model="claude-x",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=64,
    )
    _, _, body = a.build(r)
    assert body.decode("utf-8") == (
        '{"model":"claude-x","max_tokens":64,"system":"Be brief.","messages":[{"role":"user","content":"Hi"}]}'
    )


def test_gemini_build_acceptance() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("gemini", api_key="sk-test", transport=t)
    r = Request(
        model="gemini-x",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=64,
    )
    url, headers, body = a.build(r)
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"
    assert headers["x-goog-api-key"] == "sk-test"
    assert headers["content-type"] == "application/json"
    assert body == (
        b'{"contents":[{"role":"user","parts":[{"text":"Hi"}]}],'
        b'"systemInstruction":{"parts":[{"text":"Be brief."}]},'
        b'"generationConfig":{"maxOutputTokens":64}}'
    )


def test_gemini_assistant_role_becomes_model() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("gemini", api_key="sk-test", transport=t)
    r = Request(
        model="gemini-x",
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        max_tokens=64,
    )
    _, _, body = a.build(r)
    data = json.loads(body)
    assert data["contents"][0]["role"] == "user"
    assert data["contents"][1]["role"] == "model"


def test_openai_build_acceptance() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("openai_compat", api_key="sk-test", transport=t)
    r = Request(
        model="gpt-x",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=64,
    )
    url, headers, body = a.build(r)
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["content-type"] == "application/json"
    assert body == (
        b'{"model":"gpt-x",'
        b'"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],'
        b'"max_tokens":64}'
    )


def test_openai_build_body_literal() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("openai_compat", api_key="sk-test", transport=t)
    r = Request(
        model="gpt-x",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        max_tokens=64,
    )
    _, _, body = a.build(r)
    assert body.decode("utf-8") == (
        '{"model":"gpt-x","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}],"max_tokens":64}'
    )


_SYSTEM_MESSAGES: list[Message] = [
    {"role": "system", "content": "A"},
    {"role": "system", "content": "B"},
    {"role": "user", "content": "Hi"},
]


def test_anthropic_joins_system_messages_with_temperature_and_stop() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(
        model="claude-x",
        messages=_SYSTEM_MESSAGES,
        max_tokens=64,
        temperature=0.5,
        stop=["END", "\n\n"],
    )
    _, _, body = a.build(r)
    assert body == (
        b'{"model":"claude-x","max_tokens":64,"system":"A\\n\\nB",'
        b'"messages":[{"role":"user","content":"Hi"}],'
        b'"temperature":0.5,"stop_sequences":["END","\\n\\n"]}'
    )


def test_gemini_joins_system_messages_with_temperature_and_stop() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("gemini", api_key="sk-test", transport=t)
    r = Request(
        model="gemini-x",
        messages=_SYSTEM_MESSAGES,
        max_tokens=64,
        temperature=0.5,
        stop=["END", "\n\n"],
    )
    _, _, body = a.build(r)
    assert body == (
        b'{"contents":[{"role":"user","parts":[{"text":"Hi"}]}],'
        b'"systemInstruction":{"parts":[{"text":"A\\n\\nB"}]},'
        b'"generationConfig":{"maxOutputTokens":64,"temperature":0.5,'
        b'"stopSequences":["END","\\n\\n"]}}'
    )


def test_openai_joins_system_messages_with_temperature_and_stop() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("openai_compat", api_key="sk-test", transport=t)
    r = Request(
        model="gpt-x",
        messages=_SYSTEM_MESSAGES,
        max_tokens=64,
        temperature=0.5,
        stop=["END", "\n\n"],
    )
    _, _, body = a.build(r)
    assert body == (
        b'{"model":"gpt-x",'
        b'"messages":[{"role":"system","content":"A"},{"role":"system","content":"B"},'
        b'{"role":"user","content":"Hi"}],'
        b'"max_tokens":64,"temperature":0.5,"stop":["END","\\n\\n"]}'
    )


# ---------------------------------------------------------------------------
# Response parsing and error mapping


def test_anthropic_response_parsing() -> None:
    body = (
        b'{"content":[{"type":"text","text":"Hel"},{"type":"text","text":"lo"}],'
        b'"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2},"model":"claude-x"}'
    )
    t = FixedTransport(200, body)
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    rsp = a.complete(req())
    assert rsp == Response(
        text="Hello", stop_reason="end", input_tokens=5, output_tokens=2, model="claude-x"
    )


def test_anthropic_ignores_tool_use_block() -> None:
    body = (
        b'{"content":[{"type":"text","text":"Hello"},{"type":"tool_use","id":"x"}],'
        b'"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2},"model":"claude-x"}'
    )
    t = FixedTransport(200, body)
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    rsp = a.complete(req())
    assert rsp.text == "Hello"


def test_anthropic_stop_reason_mapping() -> None:
    for raw, expected in [
        ("max_tokens", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
        ("weird", "other"),
    ]:
        body = json.dumps(
            {
                "content": [{"type": "text", "text": "x"}],
                "stop_reason": raw,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "model": "claude-x",
            }
        ).encode()
        t = FixedTransport(200, body)
        a = make_adapter("anthropic", api_key="sk-test", transport=t)
        rsp = a.complete(req())
        assert rsp.stop_reason == expected


def test_gemini_response_parsing_and_mapping() -> None:
    body = (
        b'{"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":"MAX_TOKENS"}],'
        b'"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20}}'
    )
    t = FixedTransport(200, body)
    a = make_adapter("gemini", api_key="sk-test", transport=t)
    rsp = a.complete(
        Request(model="gemini-x", messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
    )
    assert rsp.text == "Hello"
    assert rsp.stop_reason == "max_tokens"
    assert rsp.input_tokens == 10
    assert rsp.output_tokens == 20
    assert rsp.model == "gemini-x"  # model taken from request


def test_gemini_missing_usage_metadata_defaults_to_zero() -> None:
    body = b'{"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"STOP"}]}'
    t = FixedTransport(200, body)
    a = make_adapter("gemini", api_key="sk-test", transport=t)
    rsp = a.complete(
        Request(model="gemini-x", messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
    )
    assert rsp.input_tokens == 0
    assert rsp.output_tokens == 0


def test_gemini_finish_reason_other() -> None:
    body = b'{"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"SAFETY"}]}'
    t = FixedTransport(200, body)
    a = make_adapter("gemini", api_key="sk-test", transport=t)
    rsp = a.complete(
        Request(model="gemini-x", messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
    )
    assert rsp.stop_reason == "other"


def test_openai_response_parsing_and_mapping() -> None:
    body = (
        b'{"choices":[{"message":{"content":"Hello"},"finish_reason":"length"}],'
        b'"usage":{"prompt_tokens":7,"completion_tokens":5},"model":"gpt-x"}'
    )
    t = FixedTransport(200, body)
    a = make_adapter("openai_compat", api_key="sk-test", transport=t)
    rsp = a.complete(
        Request(model="gpt-x", messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
    )
    assert rsp.text == "Hello"
    assert rsp.stop_reason == "max_tokens"
    assert rsp.input_tokens == 7
    assert rsp.output_tokens == 5
    assert rsp.model == "gpt-x"


def test_openai_stop_reason_stop_maps_to_end() -> None:
    body = (
        b'{"choices":[{"message":{"content":"Hello"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":7,"completion_tokens":5},"model":"gpt-x"}'
    )
    t = FixedTransport(200, body)
    a = make_adapter("openai_compat", api_key="sk-test", transport=t)
    rsp = a.complete(
        Request(model="gpt-x", messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
    )
    assert rsp.stop_reason == "end"


def test_openai_missing_usage_defaults_to_zero() -> None:
    body = b'{"choices":[{"message":{"content":"Hello"},"finish_reason":"stop"}],"model":"gpt-x"}'
    t = FixedTransport(200, body)
    a = make_adapter("openai_compat", api_key="sk-test", transport=t)
    rsp = a.complete(
        Request(model="gpt-x", messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
    )
    assert rsp.input_tokens == 0
    assert rsp.output_tokens == 0


def test_transport_status_mapping() -> None:
    mapping = {
        401: "auth",
        403: "auth",
        429: "rate_limited",
        500: "unavailable",
        503: "unavailable",
        529: "unavailable",
        400: "bad_request",
        404: "bad_request",
    }
    for status, code in mapping.items():
        t = FixedTransport(status, b'{"error":{"type":"foo"}}')
        a = make_adapter("anthropic", api_key="sk-test", transport=t)
        with pytest.raises(ProviderError) as exc_info:
            a.complete(req())
        assert exc_info.value.code == code
        assert exc_info.value.status == status
        assert "sk-test" not in str(exc_info.value)


def test_bad_response_non_json() -> None:
    t = FixedTransport(200, b"not json")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    with pytest.raises(ProviderError) as exc_info:
        a.complete(req())
    assert exc_info.value.code == "bad_response"
    assert exc_info.value.status == 200


def test_bad_response_missing_text_field() -> None:
    body = b'{"content":[{"type":"image"}],"stop_reason":"end_turn","usage":{}}'
    t = FixedTransport(200, body)
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    with pytest.raises(ProviderError) as exc_info:
        a.complete(req())
    assert exc_info.value.code == "bad_response"


def test_response_body_too_large() -> None:
    big_body = b"x" * (10_485_761)  # max_response_bytes + 1
    t = FixedTransport(200, big_body)
    a = make_adapter("anthropic", api_key="sk-test", transport=t, max_response_bytes=10_485_760)
    with pytest.raises(ProviderError) as exc_info:
        a.complete(req())
    assert exc_info.value.code == "bad_response"


# ---------------------------------------------------------------------------
# Validation tests


def test_validation_invalid_model() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(model="a/b", messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
    with pytest.raises(ProviderError) as exc_info:
        a.build(r)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.status is None


def test_validation_max_tokens_zero() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(model="claude-x", messages=[{"role": "user", "content": "Hi"}], max_tokens=0)
    with pytest.raises(ProviderError) as exc_info:
        a.build(r)
    assert exc_info.value.code == "invalid_request"


def test_validation_empty_messages() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(model="claude-x", messages=[], max_tokens=64)
    with pytest.raises(ProviderError) as exc_info:
        a.build(r)
    assert exc_info.value.code == "invalid_request"


def test_validation_no_user_message() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(
        model="claude-x",
        messages=[{"role": "system", "content": "Be brief."}],
        max_tokens=64,
    )
    with pytest.raises(ProviderError) as exc_info:
        a.build(r)
    assert exc_info.value.code == "invalid_request"


def test_validation_total_content_limit() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    content_ok = "a" * 1_000_000
    content_too_long = "a" * 1_000_001
    r_ok = Request(
        model="claude-x", messages=[{"role": "user", "content": content_ok}], max_tokens=64
    )
    a.build(r_ok)  # should not raise
    r_bad = Request(
        model="claude-x", messages=[{"role": "user", "content": content_too_long}], max_tokens=64
    )
    with pytest.raises(ProviderError) as exc_info:
        a.build(r_bad)
    assert exc_info.value.code == "invalid_request"


def test_validation_temperature_out_of_range() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(
        model="claude-x",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
        temperature=3.0,
    )
    with pytest.raises(ProviderError) as exc_info:
        a.build(r)
    assert exc_info.value.code == "invalid_request"


def test_validation_too_many_stop_sequences() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    r = Request(
        model="claude-x",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
        stop=["a", "b", "c", "d", "e"],
    )
    with pytest.raises(ProviderError) as exc_info:
        a.build(r)
    assert exc_info.value.code == "invalid_request"


# ---------------------------------------------------------------------------
# on_event tests


def test_on_event_request_and_response() -> None:
    events: list[dict[str, object]] = []
    t = FixedTransport(
        200,
        b'{"content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn",'
        b'"usage":{"input_tokens":5,"output_tokens":2},"model":"claude-x"}',
    )
    a = make_adapter("anthropic", api_key="sk-test", transport=t, on_event=events.append)
    a.complete(req())
    assert events[0] == {"event": "request", "provider": "anthropic", "model": "claude-x"}
    assert events[1] == {
        "event": "response",
        "provider": "anthropic",
        "status": 200,
        "input_tokens": 5,
        "output_tokens": 2,
    }
    # No key material
    for event in events:
        assert "sk-test" not in repr(event)


# ---------------------------------------------------------------------------
# as_port tests


def test_as_port_returns_completion() -> None:
    t = FixedTransport(
        200,
        b'{"choices":[{"message":{"content":"Hello"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":7,"completion_tokens":5},"model":"gpt-x"}',
    )
    a = make_adapter("openai_compat", api_key="sk-test", transport=t)
    port = a.as_port("gpt-x")
    result = port.complete([{"role": "user", "content": "Hi"}], max_tokens=8)
    assert result == "Hello"


def test_as_port_uses_adapter_complete() -> None:
    t = FixedTransport(
        200,
        b'{"content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn",'
        b'"usage":{"input_tokens":1,"output_tokens":1},"model":"claude-x"}',
    )
    a = make_adapter("anthropic", api_key="sk-test", transport=t)
    with mock.patch.object(a, "complete", wraps=a.complete) as mock_complete:
        port = a.as_port("claude-x")
        port.complete([{"role": "user", "content": "Hi"}], max_tokens=8)
        mock_complete.assert_called_once()
        args = mock_complete.call_args[0][0]
        assert isinstance(args, Request)
        assert args.model == "claude-x"
        assert args.max_tokens == 8
        assert args.messages == [{"role": "user", "content": "Hi"}]


# ---------------------------------------------------------------------------
# RecordedLLM and FakeLLM tests


def test_recorded_llm_returns_replies_in_order() -> None:
    llm = RecordedLLM(["first", "second"])
    assert llm.complete([], max_tokens=1) == "first"
    assert llm.complete([], max_tokens=1) == "second"


def test_recorded_llm_exhaustion_raises() -> None:
    llm = RecordedLLM(["only"])
    llm.complete([], max_tokens=1)
    with pytest.raises(RecordingExhaustedError):
        llm.complete([], max_tokens=1)


def test_fake_llm_uses_callable() -> None:
    def fn(messages: Sequence[Message]) -> str:
        return f"got {len(messages)} messages"

    llm = FakeLLM(fn)
    assert llm.complete([{"role": "user", "content": "Hi"}], max_tokens=1) == "got 1 messages"


# ---------------------------------------------------------------------------
# UrllibTransport tests (using mocks)


def test_urllib_transport_success() -> None:
    transport = UrllibTransport(max_response_bytes=10)
    mock_response = mock.MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b"hello"
    mock_response.__enter__.return_value = mock_response
    with mock.patch("provider_adapter_kit._OPENER") as mock_opener:
        mock_opener.open.return_value = mock_response
        status, body = transport.post("http://example.com", {"X-Test": "1"}, b"data", timeout_s=5.0)
    assert status == 200
    assert body == b"hello"
    mock_opener.open.assert_called_once()
    # Check that the Request object was created correctly
    req_arg = mock_opener.open.call_args[0][0]
    assert req_arg.full_url == "http://example.com"
    assert req_arg.data == b"data"
    assert req_arg.headers["X-test"] == "1"  # urllib normalizes case


def test_urllib_transport_http_error() -> None:
    transport = UrllibTransport(max_response_bytes=10)
    mock_error = urllib_error.HTTPError("http://example.com", 404, "not found", {}, None)
    with (
        mock.patch("provider_adapter_kit._OPENER") as mock_opener,
        mock.patch.object(mock_error, "read", return_value=b"not found"),
    ):
        mock_opener.open.side_effect = mock_error
        status, body = transport.post("http://example.com", {}, b"", timeout_s=5.0)
    assert status == 404
    assert body == b"not found"


def test_urllib_transport_read_limit() -> None:
    transport = UrllibTransport(max_response_bytes=5)
    mock_response = mock.MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b"123456"
    mock_response.__enter__.return_value = mock_response
    with mock.patch("provider_adapter_kit._OPENER") as mock_opener:
        mock_opener.open.return_value = mock_response
        _, body = transport.post("http://example.com", {}, b"", timeout_s=5.0)
    assert body == b"123456"  # read was called with limit+1=6, so we got 6 bytes
    mock_response.read.assert_called_once_with(6)


def test_urllib_transport_refuses_redirects() -> None:
    from provider_adapter_kit import _NoRedirectHandler  # type: ignore[attr-defined]

    handler = _NoRedirectHandler()
    result = handler.redirect_request(
        urllib_request.Request("http://example.com"),
        None,
        302,
        "Found",
        {},
        "http://evil.example",
    )
    assert result is None


# ---------------------------------------------------------------------------
# make_adapter edge cases


def test_make_adapter_invalid_provider() -> None:
    with pytest.raises(ValueError):
        make_adapter("invalid", api_key="sk-test")  # type: ignore[arg-type]


def test_make_adapter_custom_base_url_trailing_slash() -> None:
    t = FixedTransport(200, b"{}")
    a = make_adapter("anthropic", api_key="sk-test", transport=t, base_url="https://example.com/")
    url, _, _ = a.build(req())
    assert url == "https://example.com/v1/messages"


def test_make_adapter_default_base_urls() -> None:
    t = FixedTransport(200, b"{}")
    a_anthropic = make_adapter("anthropic", api_key="sk-test", transport=t)
    a_gemini = make_adapter("gemini", api_key="sk-test", transport=t)
    a_openai = make_adapter("openai_compat", api_key="sk-test", transport=t)
    url_anthropic, _, _ = a_anthropic.build(req(model="m"))
    assert url_anthropic.startswith("https://api.anthropic.com/")
    url_gemini, _, _ = a_gemini.build(req(model="m"))
    assert url_gemini.startswith("https://generativelanguage.googleapis.com/")
    url_openai, _, _ = a_openai.build(req(model="m"))
    assert url_openai.startswith("https://api.openai.com/v1/")


def test_adapter_repr_does_not_contain_api_key() -> None:
    a = make_adapter("anthropic", api_key="sk-secret", base_url="https://example.com")
    rep = repr(a)
    assert "sk-secret" not in rep


def test_provider_error_message_truncated() -> None:
    from provider_adapter_kit import _build_error_message  # type: ignore[attr-defined]

    msg = _build_error_message("anthropic", 400, "x" * 300)
    assert len(msg) <= 200
