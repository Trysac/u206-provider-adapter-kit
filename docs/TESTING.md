# Testing

## Running

```sh
pytest tests/test_provider_adapter_kit.py
```

The repository contains `tests/test_provider_adapter_kit.py` and a compiled pytest cache named `tests/__pycache__/test_provider_adapter_kit.cpython-312-pytest-9.2.0.pyc`. No other test files or fixtures are present in the reviewed files.

## Policy

- Tests use the public import path `provider_adapter_kit`.
- HTTP is faked through a `Transport` implementation; the tests do not use the network.
- The visible helper `FixedTransport` stores each `(url, headers, body, timeout_s)` call and returns a configured `(status, body)` pair.
- No coverage threshold is encoded in the reviewed files.

## Tiers

Only the single unit-test module is visible. No separate local tier, CI tier, conformance marker, or optional conformance setup is present in the reviewed files.

## Layout

`tests/test_provider_adapter_kit.py` covers request construction and parsing. The visible tests include:

- `test_anthropic_build_acceptance`: Anthropic URL, `x-api-key`, `anthropic-version`, content type, and request body for system plus user messages.
- `test_anthropic_build_body_literal`: exact Anthropic body bytes for `model`, `max_tokens`, joined `system`, and `messages`.
- `test_gemini_build_acceptance`: Gemini URL with `:generateContent`, `x-goog-api-key`, content type, `contents`, `systemInstruction`, and `generationConfig`.
- `test_gemini_assistant_role_becomes_model`: an assistant message is serialized as Gemini role `model`.

No `tests/conftest.py` is present in the reviewed files.
