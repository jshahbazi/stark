# Worker protocols

Stark ships two native workers. Each is a standalone executable that loads one converted model and serves requests over its own standard input and output.

| Worker | Purpose | Contract |
| --- | --- | --- |
| `stark-generate` | Tokenize, decode and generate text; prepare and generate one image request for its admitted Qwen3.5 family | [Generation](GENERATION.md) |
| `stark-score` | Next-token logits and probabilities for candidate tokens | [Scoring](SCORING.md) |

This page describes what the two protocols share. The per-worker pages define the requests, replies and error codes.

## Starting a worker

An installed bundle exposes both workers through the `stark` launcher:

```sh
stark generate --model-path /absolute/path/to/converted-model
stark score --score-model-path /absolute/path/to/converted-model
```

The launcher execs the binary under `libexec/` with the arguments unchanged. From a source checkout, run `target/release/stark-generate` or `target/release/stark-score` directly. Each worker accepts exactly the documented arguments and nothing else; any other argument shape produces a startup error.

The caller owns the process: start it, hold its pipes, watch for exit, and terminate and reap it when done. Closing the worker's standard input ends the session; the worker exits with status 0 after finishing the current request.

## Framing

Requests and replies are UTF-8 JSON objects, one per line, terminated by `\n`. A request line, including its newline, is limited to 1 MiB (1,048,576 bytes). Both workers reply with exactly one terminal line per request, in request order. The generation worker can additionally emit zero or more `{"event":"delta",...}` lines before the terminal reply of a streaming request.

Requests are parsed strictly: unknown fields, wrong types, and missing required fields are rejected. A client cannot discover optional fields by sending them; use the readiness line instead.

Image-capable generation workers keep the same 1 MiB framing. They pass image data through a private file reference rather than embedding bytes in JSON. The optional `image_input` readiness object describes the `mediaref-v1` contract and the `owner-private-file-v1` transport. See [Generation](GENERATION.md#prepared-image-operations) for the request sequence, limits and qualification boundary.

## Readiness

The first line a worker writes is its readiness reply. On success it starts with `"ready":true` and carries the capability fields documented on the per-worker page. On failure it is `{"ready":false,"error":CODE}` and the process exits with status 1. Read the readiness line before sending any request.

`version` in the readiness line is the worker crate version (`0.1.0`), which does not change with the release version of the installed bundle. Use `stark version` for the bundle version. Compatibility is signalled by the tag fields instead:

| Worker | Tag field | Current value |
| --- | --- | --- |
| `stark-generate` | `mode` | `text-generation-v1` |
| `stark-generate` | `streaming` | `text-delta-v1` |
| `stark-generate` | `cancellation` | `pipe-v1` |
| `stark-generate` | `image_input.version` | `mediaref-v1` (when the loaded model supports the implemented vision layout) |
| `stark-generate` | `image_input.transport` | `owner-private-file-v1` |
| `stark-generate` | `timing_method` (in generate replies) | `first-token-wall-clock-v1` |
| `stark-score` | `mode` | `exact-token-scoring-cache-v1` or `exact-token-digest-cache-v1` |
| `stark-score` | `cache_policy` | `consecutive-full-vocabulary-logits-v1` or `fresh-state` |

Each tag names the exact request and reply shape documented on the per-worker page at the time of writing. A tag changes when the shape it covers changes incompatibly; new readiness keys can appear without a tag change. Treat an unrecognised tag as incompatible rather than guessing.

Image readiness describes implemented functionality and loaded identity. Its `qualified: false` field leaves the evidence decision to the host. A matching [execution catalog](../tools/converter/stark_convert/vision_runtime_qualifications.json) record enables the exact schema-2 tuple as `reference_verified`. The [family path](VISION_FAMILY_CONTRACT.md) enables an admitted compatible model as `locally_tested` after bounded automatic execution checks. Both bind all five artifact-file hashes, the worker hash observed before launch and checked again at readiness, the Metal device and macOS version; local receipts additionally bind the actual family/version, admitted vocabulary, checker and Python dependency identities. Current readers accept fixed version-1 and config-derived version-2 families; older readers reject the unknown version-2 profile. [Generation](GENERATION.md#image-qualification) defines the evidence levels and scope.

## Error classes

Every error reply carries a fixed, content-free `error` code. The codes fall into three classes:

| Class | Behaviour | How to handle |
| --- | --- | --- |
| Recoverable | The worker writes `{"error":CODE,...}` and keeps serving. The scorer also clears its cache. | Fix the request and continue with the same process. |
| Fencing | The worker writes `{"error":CODE,...}`, then `{"ready":false,"error":"native_operation_failed"}`, and exits with status 1. | Reap the process and start a fresh worker. Native failures never leave a reusable process. |
| Startup | The worker writes `{"ready":false,"error":CODE}` and exits with status 1 before serving anything. | Fix the arguments or model directory and start again. |

Each per-worker page lists which codes belong to which class. Error codes never contain request content, file paths, or native diagnostic text.

## Privacy and diagnostics

Before loading a model, both workers redirect their own standard output and standard error to `/dev/null` and reserve a private descriptor for protocol replies, so library diagnostics cannot enter the protocol stream. They install a silent panic hook and remove any `STARK_CAPTURE_*` or `UZU_CAPTURE_*` environment variables before engine threads exist, so tensor capture cannot be enabled by the environment.

The protocol stream itself carries prompts, generated text and token IDs. Worker image frames carry opaque references to files in the owning host's private staging directory. The host creates an empty mode-0700 directory before launch and writes mode-0600 image files. After validating and reading a file, the worker immediately unlinks its name and retains the open-file lease through completion. The host cleans up remaining staged names when the transaction exits; worker teardown removes only valid opaque, owner-private regular files in its pinned directory. Keep worker pipes private to the owning process and do not log request or reply lines.

## Model-free checks

These commands exercise the protocol parsers, bounds and failure fencing without a model:

```sh
cargo test --locked -p stark-generate --no-default-features
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
uv run --no-project --python 3.12.13 python scripts/test_generation_process.py --binary target/release/stark-generate
```

For a smoke test that loads a real model, see [Contributing](../CONTRIBUTING.md#run-a-model-bound-smoke-test).
