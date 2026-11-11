# provider-adapter-kit

[![PyPI version](https://img.shields.io/pypi/v/trysac-provider-adapter-kit)](https://pypi.org/project/trysac-provider-adapter-kit/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-provider-adapter-kit)](LICENSE)
[![CI](https://github.com/Trysac/u206-provider-adapter-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u206-provider-adapter-kit/actions/workflows/ci.yml)

Uniform text-completion adapters for the Anthropic Messages, Gemini generateContent and OpenAI-compatible chat APIs over an injected HTTP transport, with one error model

## Problem

Code that must switch between Anthropic, Gemini and OpenAI-compatible servers usually ends up with
three request builders, three response parsers and three ways of failing, typically through three
heavy SDKs. The differences that matter for plain text completion are small but easy to get wrong:
where the system prompt goes, the role names, the stop reason names and the token usage fields.

This library implements one contract over the three wire formats with byte-exact request bodies, a
shared error model and a transport that tests can fake.

## Install

```sh
pip install trysac-provider-adapter-kit
# or
uv add trysac-provider-adapter-kit
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
from provider_adapter_kit import Request, make_adapter


class T:
    def post(self, url, headers, body, timeout_s):
        return (
            200,
            b'{"content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2},"model":"claude-x"}',
        )


resp = make_adapter("anthropic", api_key="sk-test", transport=T()).complete(
    Request(model="claude-x", messages=[{"role": "user", "content": "Hi"}], max_tokens=64)
)
print(resp.text)
```

## Public API

`make_adapter(provider, *, api_key, transport=None, base_url=None, timeout_s=60.0, max_response_bytes=10_485_760, on_event=None) -> Adapter` supports `anthropic`, `gemini`, and `openai_compat`. Default base URLs are `https://api.anthropic.com`, `https://generativelanguage.googleapis.com`, and `https://api.openai.com/v1`; trailing slashes on a supplied `base_url` are removed. `transport` defaults to `UrllibTransport(max_response_bytes=max_response_bytes)`.

`Request(model, messages, max_tokens, temperature=None, stop=())` and `Response(text, stop_reason, input_tokens, output_tokens, model)` are frozen dataclasses. `Message` is a `TypedDict` with `role` in `system`, `user`, `assistant` and string `content`. Stop reasons are `end`, `max_tokens`, `stop_sequence`, and `other`.

`Adapter` exposes `build(req) -> (url, headers, body_bytes)`, `complete(req) -> Response`, and `as_port(model) -> LLMPort`. `LLMPort.complete(messages, *, max_tokens) -> str` is implemented by `Adapter.as_port`, `RecordedLLM(replies)`, and `FakeLLM(fn)`; `RecordedLLM` raises `RecordingExhaustedError` when its replies are exhausted.

`ProviderError` carries `code`, `status`, and `message`. Codes are `invalid_request`, `auth`, `rate_limited`, `bad_request`, `unavailable`, and `bad_response`; HTTP 401/403 map to `auth`, 429 to `rate_limited`, 400/404 to `bad_request`, 500/503/529 to `unavailable`, and other non-200 statuses to `unavailable`. Transport exceptions from an injected `Transport` propagate unchanged. Error messages name the provider, status, and provider error type only, are truncated to 200 characters, and never include request bodies, headers, or the API key.

Validation raises `invalid_request` unless the model matches `^[A-Za-z0-9._-]{1,128}$`, `max_tokens` is 1 through 1,000,000, there are 1 through 1000 messages with at least one user message, total content is at most 1,000,000 characters, `temperature` is finite and between 0 and 2, and there are at most 4 stop strings. Responses larger than `max_response_bytes` (default 10,485,760) and malformed JSON or malformed provider payloads raise `bad_response`; `UrllibTransport` reads at most `max_response_bytes + 1` bytes and never follows redirects.

Request and response bodies are UTF-8 JSON produced with `separators=(',', ':')` and `ensure_ascii=False`. `on_event`, when supplied, receives `{'event': 'request', 'provider': ..., 'model': ...}` and `{'event': 'response', 'provider': ..., 'status': ..., 'input_tokens': ..., 'output_tokens': ...}` dictionaries without keys, headers, or message contents. The API key is never included in events, errors, or the adapter repr.

Streaming, tools, images, retries, Vertex AI, and Bedrock auth are not implemented.

## Why not the official SDKs

The official SDKs add runtime dependencies and their own provider-specific error and response types. This library uses only the standard library, produces byte-exact request bodies that tests can assert, and maps all three providers onto one `ProviderError` model and one `Response` shape. It does not provide streaming, tools, images, retries, Vertex AI, or Bedrock auth.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
