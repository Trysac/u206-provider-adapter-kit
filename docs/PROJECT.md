# Project record

## Objective

Uniform text-completion adapters for the Anthropic Messages, Gemini generateContent and OpenAI-compatible chat APIs over an injected HTTP transport, with one error model.

## Scope

- In scope: `provider_adapter_kit.make_adapter` and the public names exported by `src/provider_adapter_kit/__init__.py`.
- In scope: byte-exact request construction for Anthropic (`/v1/messages`), Gemini (`/v1beta/models/<model>:generateContent`), and OpenAI-compatible (`/chat/completions`) text completion.
- In scope: response parsing into `Response(text, stop_reason, input_tokens, output_tokens, model)` and `ProviderError(code, status, message)`.
- In scope: the `Transport` protocol, `UrllibTransport`, `LLMPort`, `RecordedLLM`, `FakeLLM`, and `RecordingExhaustedError`.
- In scope: request validation, response-size limits, event callbacks, and a standard-library-only implementation.

## Out of scope

- Streaming, tools, and images.
- Retries; callers may compose a retry library around `Transport` or `Adapter`.
- Vertex AI and Bedrock authentication.
- Runtime dependencies and I/O beyond the HTTP POST performed by `UrllibTransport` or a caller-supplied `Transport`.
- Provider features not represented by `Request`, `Response`, and `ProviderError`, including provider-specific content blocks other than text.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 2144523 / 81667 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.1179 |

## Factory run u206 (measured from API usage)

| Date (UTC) | Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | generation | deepseek-v4-pro | yes | 2340 | 18472 | 0 | 3328 | 0.0382 |
| 2026-11-11 | fix-1 | deepseek-flash | no | 28367 | 19931 | 0 | 628864 | 0.0181 |
| 2026-11-11 | fix-2 | deepseek-flash | no | 11783 | 1048 | 0 | 48256 | 0.0025 |
| 2026-11-11 | audit | deepseek-flash | no | 22999 | 16680 | 0 | 272384 | 0.0143 |
| 2026-11-11 | fix-3 | deepseek-flash | no | 15946 | 4222 | 0 | 179840 | 0.0055 |
| 2026-11-11 | threat-model | deepseek-flash | no | 58859 | 6564 | 0 | 665088 | 0.0148 |
| 2026-11-11 | fix-4 | deepseek-flash | no | 13894 | 5646 | 0 | 103808 | 0.0058 |
| 2026-11-11 | docs | deepseek-flash | no | 88511 | 9104 | 0 | 256 | 0.0187 |

Total: $0.1179. Prices: deepseek line, as measured by its adapter. Providers serve the current version of each model id, so the date tells which version answered; versions, launch stages and price sources per line are in the factory's model register (docs/MODELS.md).
