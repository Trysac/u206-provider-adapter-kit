"""Uniform text-completion adapters for Anthropic, Gemini, and OpenAI-compatible APIs.

The module exposes a common :class:`Request` and :class:`Response` model, a
:class:`Transport` protocol for injecting HTTP calls, and a factory
:func:`make_adapter` that returns an :class:`Adapter`. The adapter builds
byte-exact request bodies for each provider and maps transport responses to a
shared :class:`ProviderError` model.

All public helpers (:class:`RecordedLLM`, :class:`FakeLLM`) implement the
:class:`LLMPort` protocol for testing and composition.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict
from urllib import error as urllib_error
from urllib import request as urllib_request

__all__ = [
    "Adapter",
    "FakeLLM",
    "LLMPort",
    "Message",
    "ProviderError",
    "RecordedLLM",
    "RecordingExhaustedError",
    "Request",
    "Response",
    "Transport",
    "UrllibTransport",
    "make_adapter",
]

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

_StopReason = Literal["end", "max_tokens", "stop_sequence", "other"]
_ErrorCode = Literal[
    "invalid_request",
    "auth",
    "rate_limited",
    "bad_request",
    "unavailable",
    "bad_response",
]
_Provider = Literal["anthropic", "gemini", "openai_compat"]

_MODEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMPort(Protocol):
    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Return a completion for the given messages."""
        ...


class RecordingExhaustedError(Exception):
    """Raised when a :class:`RecordedLLM` runs out of scripted replies."""


class RecordedLLM:
    """An :class:`LLMPort` that replays a fixed sequence of replies."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = iter(replies)

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        del messages, max_tokens
        try:
            return next(self._replies)
        except StopIteration:
            raise RecordingExhaustedError("no more recorded replies") from None


class FakeLLM:
    """An :class:`LLMPort` whose completions come from a callable."""

    def __init__(self, fn: Callable[[Sequence[Message]], str]) -> None:
        self._fn = fn

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        del max_tokens
        return self._fn(messages)


@dataclass(frozen=True)
class Request:
    model: str
    messages: Sequence[Message]
    max_tokens: int
    temperature: float | None = None
    stop: Sequence[str] = ()


@dataclass(frozen=True)
class Response:
    text: str
    stop_reason: _StopReason
    input_tokens: int
    output_tokens: int
    model: str


class Transport(Protocol):
    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout_s: float
    ) -> tuple[int, bytes]:
        """Send an HTTP POST and return the status code and raw body bytes."""
        ...


class _NoRedirectHandler(urllib_request.HTTPRedirectHandler):
    """Refuse redirects so credentials never reach a different origin.

    :mod:`urllib` forwards the original request headers (including API keys) to
    a redirect target. Returning ``None`` here makes the opener raise the 3xx as
    an :class:`urllib.error.HTTPError` instead of following it.
    """

    def redirect_request(
        self,
        req: urllib_request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_OPENER = urllib_request.build_opener(_NoRedirectHandler())


class UrllibTransport:
    """Default :class:`Transport` using :mod:`urllib.request`.

    Reads at most ``max_response_bytes + 1`` bytes from a response. If the
    response is larger, only the first ``max_response_bytes + 1`` bytes are
    returned; the adapter will then raise :class:`ProviderError` with code
    ``bad_response``. Redirects are never followed.
    """

    def __init__(self, max_response_bytes: int = 10_485_760) -> None:
        self.max_response_bytes = max_response_bytes

    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout_s: float
    ) -> tuple[int, bytes]:
        # The base URL is provider-supplied and the path is built by the adapter, so
        # only http(s) endpoints are reachable here; S310's scheme warning does not apply.
        req = urllib_request.Request(  # noqa: S310
            url, data=body, headers=dict(headers), method="POST"
        )
        try:
            with _OPENER.open(req, timeout=timeout_s) as resp:
                status = resp.getcode()
                raw = resp.read(self.max_response_bytes + 1)
        except urllib_error.HTTPError as exc:
            status = exc.code
            raw = exc.read(self.max_response_bytes + 1)
        return status, raw


class Adapter(Protocol):
    def build(self, req: Request) -> tuple[str, dict[str, str], bytes]:
        """Build URL, headers, and request body for the given request."""
        ...

    def complete(self, req: Request) -> Response:
        """Send the request through the transport and parse the response."""
        ...

    def as_port(self, model: str) -> LLMPort:
        """Return an :class:`LLMPort` bound to this adapter and model."""
        ...


class _Port:
    def __init__(self, adapter: _Adapter, model: str) -> None:
        self._adapter = adapter
        self._model = model

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        req = Request(model=self._model, messages=messages, max_tokens=max_tokens)
        resp = self._adapter.complete(req)
        return resp.text


class _Adapter:
    def __init__(
        self,
        provider: _Provider,
        *,
        api_key: str,
        transport: Transport,
        base_url: str,
        timeout_s: float,
        max_response_bytes: int,
        on_event: Callable[[dict[str, object]], None] | None,
    ) -> None:
        self._provider = provider
        self._api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_response_bytes = max_response_bytes
        self._on_event = on_event

    def __repr__(self) -> str:
        return (
            f"_Adapter(provider={self._provider!r}, base_url={self._base_url!r}, "
            f"timeout_s={self._timeout_s!r}, max_response_bytes={self._max_response_bytes!r})"
        )

    def build(self, req: Request) -> tuple[str, dict[str, str], bytes]:
        _validate_request(self._provider, req)
        if self._provider == "anthropic":
            url = f"{self._base_url}/v1/messages"
            headers = {
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body_obj = _build_anthropic_body(req)
        elif self._provider == "gemini":
            url = f"{self._base_url}/v1beta/models/{req.model}:generateContent"
            headers = {
                "x-goog-api-key": self._api_key,
                "content-type": "application/json",
            }
            body_obj = _build_gemini_body(req)
        else:  # openai_compat
            url = f"{self._base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
            }
            body_obj = _build_openai_body(req)
        body_bytes = json.dumps(body_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return url, headers, body_bytes

    def complete(self, req: Request) -> Response:
        url, headers, body = self.build(req)
        event: dict[str, object] = {
            "event": "request",
            "provider": self._provider,
            "model": req.model,
        }
        if self._on_event is not None:
            self._on_event(event)
        status, raw = self._transport.post(url, headers, body, self._timeout_s)
        if status != 200:
            code = _map_http_status(status)
            err_type = _extract_provider_error_type(self._provider, raw)
            msg = _build_error_message(self._provider, status, err_type)
            raise ProviderError(code=code, status=status, message=msg)
        if len(raw) > self._max_response_bytes:
            raise ProviderError(
                code="bad_response",
                status=status,
                message=_build_error_message(self._provider, status, "response too large"),
            )
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderError(
                code="bad_response",
                status=status,
                message=_build_error_message(self._provider, status, "invalid JSON"),
            ) from None

        if self._provider == "anthropic":
            resp = _parse_anthropic_response(data, req.model)
        elif self._provider == "gemini":
            resp = _parse_gemini_response(data, req.model)
        else:
            resp = _parse_openai_response(data, req.model)
        if self._on_event is not None:
            response_event: dict[str, object] = {
                "event": "response",
                "provider": self._provider,
                "status": status,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            }
            self._on_event(response_event)
        return resp

    def as_port(self, model: str) -> LLMPort:
        return _Port(self, model)


def make_adapter(
    provider: _Provider,
    *,
    api_key: str,
    transport: Transport | None = None,
    base_url: str | None = None,
    timeout_s: float = 60.0,
    max_response_bytes: int = 10_485_760,
    on_event: Callable[[dict[str, object]], None] | None = None,
) -> Adapter:
    """Create an adapter for the given provider.

    Args:
        provider: One of ``'anthropic'``, ``'gemini'``, or ``'openai_compat'``.
        api_key: Provider API key. Never included in logs, events, or reprs.
        transport: Optional custom :class:`Transport`. Defaults to
            :class:`UrllibTransport` with ``max_response_bytes``.
        base_url: Optional base URL. Provider-specific defaults are used when
            omitted.
        timeout_s: Request timeout in seconds.
        max_response_bytes: Maximum accepted response size in bytes.
        on_event: Optional callback receiving ``{'event': 'request', ...}`` and
            ``{'event': 'response', ...}`` dictionaries.

    Returns:
        An :class:`Adapter` instance.
    """
    if provider not in ("anthropic", "gemini", "openai_compat"):
        raise ValueError(f"unsupported provider: {provider!r}")
    if transport is None:
        transport = UrllibTransport(max_response_bytes=max_response_bytes)
    if base_url is None:
        if provider == "anthropic":
            base_url = "https://api.anthropic.com"
        elif provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com"
        else:
            base_url = "https://api.openai.com/v1"
    return _Adapter(
        provider=provider,
        api_key=api_key,
        transport=transport,
        base_url=base_url,
        timeout_s=timeout_s,
        max_response_bytes=max_response_bytes,
        on_event=on_event,
    )


class ProviderError(Exception):
    """Error raised for invalid requests, provider errors, and bad responses."""

    def __init__(self, *, code: _ErrorCode, status: int | None, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# Validation


def _validate_request(provider: _Provider, req: Request) -> None:
    if not _MODEL_RE.match(req.model):
        raise ProviderError(
            code="invalid_request",
            status=None,
            message=_build_error_message(provider, None, "invalid model name"),
        )
    if not (1 <= req.max_tokens <= 1_000_000):
        raise ProviderError(
            code="invalid_request",
            status=None,
            message=_build_error_message(provider, None, "max_tokens out of range"),
        )
    if not (1 <= len(req.messages) <= 1000):
        raise ProviderError(
            code="invalid_request",
            status=None,
            message=_build_error_message(provider, None, "invalid message count"),
        )
    has_user = False
    total_chars = 0
    for msg in req.messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            has_user = True
        total_chars += len(content)
        if total_chars > 1_000_000:
            raise ProviderError(
                code="invalid_request",
                status=None,
                message=_build_error_message(provider, None, "content too large"),
            )
    if not has_user:
        raise ProviderError(
            code="invalid_request",
            status=None,
            message=_build_error_message(provider, None, "no user message"),
        )
    if req.temperature is not None and (
        not math.isfinite(req.temperature) or not (0.0 <= req.temperature <= 2.0)
    ):
        raise ProviderError(
            code="invalid_request",
            status=None,
            message=_build_error_message(provider, None, "temperature out of range"),
        )
    if len(req.stop) > 4:
        raise ProviderError(
            code="invalid_request",
            status=None,
            message=_build_error_message(provider, None, "too many stop sequences"),
        )


def _build_anthropic_body(req: Request) -> dict[str, Any]:
    system_parts = [m["content"] for m in req.messages if m["role"] == "system"]
    non_system = [m for m in req.messages if m["role"] != "system"]
    body: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    body["messages"] = [
        {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
        for m in non_system
    ]
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.stop:
        body["stop_sequences"] = list(req.stop)
    return body


def _build_gemini_body(req: Request) -> dict[str, Any]:
    system_parts = [m["content"] for m in req.messages if m["role"] == "system"]
    non_system = [m for m in req.messages if m["role"] != "system"]
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in non_system
    ]
    body: dict[str, Any] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    generation_config: dict[str, Any] = {"maxOutputTokens": req.max_tokens}
    if req.temperature is not None:
        generation_config["temperature"] = req.temperature
    if req.stop:
        generation_config["stopSequences"] = list(req.stop)
    body["generationConfig"] = generation_config
    return body


def _build_openai_body(req: Request) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": req.model,
        "messages": [{"role": m["role"], "content": m["content"]} for m in req.messages],
        "max_tokens": req.max_tokens,
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.stop:
        body["stop"] = list(req.stop)
    return body


# ---------------------------------------------------------------------------
# Response parsing


def _parse_anthropic_response(data: Any, model: str) -> Response:
    try:
        content_blocks = data["content"]
        text_blocks = [b for b in content_blocks if b.get("type") == "text"]
        if not text_blocks:
            raise KeyError("text")
        text = "".join(b["text"] for b in text_blocks)
        stop_reason_raw = data.get("stop_reason", "other")
        stop_reason: _StopReason
        if stop_reason_raw == "end_turn":
            stop_reason = "end"
        elif stop_reason_raw == "max_tokens":
            stop_reason = "max_tokens"
        elif stop_reason_raw == "stop_sequence":
            stop_reason = "stop_sequence"
        else:
            stop_reason = "other"
        usage = data.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        resp_model = data.get("model", model)
        return Response(
            text=text,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=resp_model,
        )
    except (KeyError, TypeError, AttributeError):
        raise ProviderError(
            code="bad_response",
            status=200,
            message=_build_error_message("anthropic", 200, "malformed response"),
        ) from None


def _parse_gemini_response(data: Any, model: str) -> Response:
    try:
        candidates = data["candidates"]
        first = candidates[0]
        parts = first["content"]["parts"]
        text_parts = [p for p in parts if "text" in p]
        if not text_parts:
            raise KeyError("text")
        text = "".join(p["text"] for p in text_parts)
        finish_reason_raw = first.get("finishReason", "OTHER")
        stop_reason: _StopReason
        if finish_reason_raw == "STOP":
            stop_reason = "end"
        elif finish_reason_raw == "MAX_TOKENS":
            stop_reason = "max_tokens"
        else:
            stop_reason = "other"
        usage_meta = data.get("usageMetadata", {})
        input_tokens = int(usage_meta.get("promptTokenCount", 0))
        output_tokens = int(usage_meta.get("candidatesTokenCount", 0))
        return Response(
            text=text,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
    except (KeyError, TypeError, IndexError, AttributeError):
        raise ProviderError(
            code="bad_response",
            status=200,
            message=_build_error_message("gemini", 200, "malformed response"),
        ) from None


def _parse_openai_response(data: Any, model: str) -> Response:
    try:
        choices = data["choices"]
        first = choices[0]
        text = first["message"]["content"] or ""
        finish_reason_raw = first.get("finish_reason", "other")
        stop_reason: _StopReason
        if finish_reason_raw == "stop":
            stop_reason = "end"
        elif finish_reason_raw == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "other"
        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        resp_model = data.get("model", model)
        return Response(
            text=text,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=resp_model,
        )
    except (KeyError, TypeError, IndexError, AttributeError):
        raise ProviderError(
            code="bad_response",
            status=200,
            message=_build_error_message("openai_compat", 200, "malformed response"),
        ) from None


# ---------------------------------------------------------------------------
# Error handling helpers


def _map_http_status(status: int) -> _ErrorCode:
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limited"
    if status in (500, 503, 529):
        return "unavailable"
    if status in (400, 404):
        return "bad_request"
    return "unavailable"


def _extract_provider_error_type(provider: _Provider, raw: bytes) -> str | None:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        return None
    if provider == "gemini":
        status = error.get("status")
        return str(status) if status is not None else None
    # anthropic and openai_compat both use error.type
    error_type = error.get("type")
    return error_type if isinstance(error_type, str) else None


def _build_error_message(provider: _Provider, status: int | None, error_type: str | None) -> str:
    provider_name = {
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "openai_compat": "OpenAI-compatible",
    }[provider]
    parts = [provider_name]
    if status is not None:
        parts.append(f"status {status}")
    if error_type:
        parts.append(error_type)
    else:
        parts.append("unknown error")
    msg = ", ".join(parts)
    return msg[:200]
