# Architecture

## Purpose

`trysac-provider-adapter-kit` is a small, dependency-free Python library: uniform text-completion adapters for the Anthropic Messages, Gemini generateContent and OpenAI-compatible chat APIs over an injected HTTP transport, with one error model.

## Shape

```text
caller code --> provider_adapter_kit (public API in src/provider_adapter_kit/__init__.py)
                    |
                    +-- _Adapter.build / complete / as_port
                    +-- pure builders and parsers per provider
                    +-- Transport protocol --> UrllibTransport (urllib.request, only I/O)
```

- **Public API**: the names in `__all__` of `src/provider_adapter_kit/__init__.py`: `Adapter`, `FakeLLM`, `LLMPort`, `Message`, `ProviderError`, `RecordedLLM`, `RecordingExhaustedError`, `Request`, `Response`, `Transport`, `UrllibTransport`, and `make_adapter`.
- **Providers**: `_Adapter.build` selects `/v1/messages` for Anthropic, `/v1beta/models/<model>:generateContent` for Gemini, and `/chat/completions` for `openai_compat`, using provider-specific headers and pure body builders.
- **Errors**: invalid requests, HTTP failures, oversized responses, invalid JSON, and malformed provider payloads raise `ProviderError(code, status, message)` with codes `invalid_request`, `auth`, `rate_limited`, `bad_request`, `unavailable`, or `bad_response`. Transport exceptions from an injected transport propagate unchanged.
- **I/O**: request builders and response parsers are pure. The only I/O is `Transport.post`; `UrllibTransport` performs the default HTTP POST, reads at most `max_response_bytes + 1` bytes, and refuses redirects.
- **Port shape**: `Adapter.as_port(model)` returns an `LLMPort` whose `complete(messages, *, max_tokens)` returns only `Response.text`; `RecordedLLM` and `FakeLLM` provide test and composition implementations.

## Quality contract

The reviewed files contain `tests/test_provider_adapter_kit.py`, which exercises the public API with a fake `Transport`. No separate gate script or coverage threshold is present in the reviewed files.

## Decisions

No architecture decision record directory is present in the reviewed files. Provider-specific body order, headers, response parsing, validation limits, and error mapping are implemented directly in `src/provider_adapter_kit/__init__.py`.
