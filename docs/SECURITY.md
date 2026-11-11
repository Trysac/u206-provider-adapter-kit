# Security notes

## Threat model

`trysac-provider-adapter-kit` is a standard-library-only library. It performs I/O only through `Transport.post`; the default `UrllibTransport` uses `urllib.request` and never follows redirects.

Defended:

- **Request validation**: `_validate_request` rejects model names outside `^[A-Za-z0-9._-]{1,128}$`, `max_tokens` outside 1 through 1,000,000, message counts outside 1 through 1000, requests without a user message, total content over 1,000,000 characters, non-finite or out-of-range temperature, and more than 4 stop strings with `ProviderError(code="invalid_request")`.
- **Response size**: `UrllibTransport` reads at most `max_response_bytes + 1` bytes, and `Adapter.complete` raises `bad_response` when the returned body is longer than `max_response_bytes` (default 10,485,760 bytes). A custom transport does not bypass the adapter check once it returns bytes.
- **Redirects**: `_NoRedirectHandler.redirect_request` returns `None`, so `urllib` raises the 3xx response as `urllib.error.HTTPError` instead of forwarding credentials to a redirect target.
- **Malformed responses**: invalid UTF-8, invalid JSON, missing text fields, and provider payloads with unexpected shapes raise `bad_response` instead of being partially interpreted.
- **Error text**: `_build_error_message` includes only the provider name, HTTP status, and provider error type (`error.type` for Anthropic and OpenAI-compatible responses, `error.status` for Gemini), and truncates to 200 characters. It does not include the response body, request headers, or API key.
- **API key exposure in local objects**: `_Adapter.__repr__` includes only provider, base URL, timeout, and maximum response size. `on_event` payloads contain event name, provider, model, status, and token counts only. The key is sent only in provider authentication headers.
- **Logging**: the module creates `logging.getLogger(__name__)` with a `NullHandler` and does not log request bodies, headers, or keys.

Not defended:

- **Caller-supplied `base_url`**: a hostile or mistaken base URL can point requests at an arbitrary host and receive the API key in the provider header. Callers must pass trusted base URLs.
- **Custom transports**: an injected `Transport` receives the headers, including the API key, and can log, store, or forward them. Transport exceptions propagate unchanged.
- **TLS policy**: the default transport relies on `urllib` and the platform certificate store; there is no certificate pinning or custom trust policy.
- **Availability and abuse controls**: there are no retries, timeouts beyond the `timeout_s` passed to the transport, rate limiting, circuit breaking, or request-size limits beyond validation of model, message count, content characters, `max_tokens`, temperature, and stop strings.
- **Secret storage**: the library does not read, store, rotate, or redact secrets outside its own events, errors, and repr; callers are responsible for API-key handling.
- **Non-text content**: only text completion is handled. Tool-use blocks are ignored when parsing Anthropic responses; tools, images, streaming, Vertex AI, and Bedrock auth are not implemented.

## Publishing

No publishing configuration is present in the reviewed code. Release credentials are not part of the package, and package publication must not place long-lived tokens in the repository.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).
