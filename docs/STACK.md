# Stack

## Used

| Tool | Why |
|---|---|
| Python >= 3.10 | Required for `str | None` syntax used by the public API. |
| Standard library | No runtime dependencies; the code uses `json`, `logging`, `math`, `re`, `dataclasses`, `typing`, `collections.abc`, and `urllib`. |
| `py.typed` | The package ships a `py.typed` marker for type checkers. |
| pytest | The compiled test cache is named `test_provider_adapter_kit.cpython-312-pytest-9.2.0.pyc`; tests use `pytest` and `unittest.mock` imports. |

## Discarded

| Tool | Why not |
|---|---|
| Official provider SDKs | They add runtime dependencies and provider-specific response and error types; this library needs byte-exact request bodies and one error model. |
| `requests` | The default transport uses the standard library's `urllib.request`, keeping the runtime dependency set empty. |
| Streaming and tool abstractions | The library handles plain text completion only; streaming, tools, images, retries, Vertex AI, and Bedrock auth are out of scope. |
