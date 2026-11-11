# Provider-agnostic LLM adapter layer with injected transport

Status: accepted (generated, pending human review)

## Context

Code that switches between Anthropic, Gemini, and OpenAI-compatible servers ends up with three request builders, three response parsers, and three error models, usually via heavy SDKs. The differences for plain text completion are small but critical: system prompt placement, role names, stop reason names, and token usage fields.

## Decision

Implement a single library that provides uniform Request/Response models, a Transport protocol for HTTP injection, and a factory make_adapter that builds byte-exact provider-specific requests and normalizes responses into a shared ProviderError error model. No runtime dependencies; stdlib urllib provides the default transport.

## Consequences

Users get one contract for three providers, easy testing via fake transports, and a unified error model. The library remains lightweight and auditable. Any provider-specific extension beyond the three supported formats would require adding a new builder/parser pair. The transport abstraction allows composition with retry libraries and custom HTTP clients.
