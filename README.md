# provider-adapter-kit

[![PyPI version](https://img.shields.io/pypi/v/trysac-provider-adapter-kit)](https://pypi.org/project/trysac-provider-adapter-kit/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-provider-adapter-kit)](LICENSE)
[![CI](https://github.com/Trysac/u206-provider-adapter-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u206-provider-adapter-kit/actions/workflows/ci.yml)

Uniform text-completion adapters for the Anthropic Messages, Gemini generateContent and OpenAI-compatible chat APIs over an injected HTTP transport, with one error model

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-provider-adapter-kit
# or
uv add trysac-provider-adapter-kit
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import provider_adapter_kit

# normalize() trims surrounding whitespace and lower-cases the text.
print(provider_adapter_kit.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    provider_adapter_kit.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
