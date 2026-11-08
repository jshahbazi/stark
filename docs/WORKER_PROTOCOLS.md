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

Requests and replies are UTF-8 JSON objects, one per line, terminated by `\n`. Request lines, including the newline, follow the configurable resource budget below. Both workers reply with exactly one terminal line per request, in request order. The generation worker can additionally emit zero or more `{"event":"delta",...}` lines before the terminal reply of a streaming request.

Requests are parsed strictly: unknown fields, wrong types, and missing required fields are rejected. A client cannot discover optional fields by sending them; use the readiness line instead.

Image-capable generation workers keep the same framing budget. They pass image data through a private file reference rather than embedding bytes in JSON. The optional `image_input` readiness object describes the `mediaref-v1` contract and the `owner-private-file-v1` transport. See [Generation](GENERATION.md#prepared-image-operations) for the request sequence, limits and qualification boundary.

## Resource budget

Normal app use requires no budget configuration. At process startup, the host
reads available RAM and selects `floor(available_bytes / (128 × 1 MiB))` MiB,
clamped to 16 through 64 MiB. If memory telemetry is unavailable, it uses the
16 MiB floor. The host pins that snapshot for its child workers and template
helpers so a request uses one consistent policy. Native standalone workers
apply the same automatic policy at startup.

For development or controlled runtime testing, `STARK_REQUEST_BUDGET_MIB`
overrides the budget with an integer from 1 through 64. Invalid values fail
startup. The override is read at process startup; restart the owning app and
workers to apply a change, and give standalone host tools and native helpers
the same override. There is no budget setting in the app interface.

For a selected logical budget `B` bytes, the policy is:

| Resource | Budget | Automatic range |
| --- | --- | --- |
| Logical input and output content, including rendered templates | `B` | 16–64 MiB |
| Encoded worker and template request, including framing | `6B + 2 MiB` | 98–386 MiB |
| Text HTTP body | `6B + 2 MiB` | 98–386 MiB |
| Image HTTP body | Text body budget plus 6 MiB | 104–392 MiB |
| Aggregate pending HTTP bodies | Two image body budgets | 208–784 MiB |
| Template live Rust allocation requests and sampled RSS | `128 MiB + 8B` | 256–640 MiB |
| Template cumulative Rust allocation requests | `32 MiB + 32B` | 544–2,080 MiB |

Transport headroom accommodates JSON escaping and protocol fields. Token vectors
are also resource bounded using their four-byte-per-ID storage. Native readiness
reports `request_budget_bytes` and `max_request_bytes`; hosts must respect the
effective native request limit. Byte admission is independent of model context.
The worker still applies its model's actual token limit after tokenization.
Byte admission does not guarantee that every possible encoding of a full token
window fits the resource budget. Automatic admission does not qualify
full-window accuracy, execution time or memory use on a particular machine.

The HTTP parser bounds JSON structure before allocating the parsed tree: at
most 65,536 nodes and 64 nested containers, counting object keys as nodes.
Tool schemas and argument objects retain their stricter structural limits.
Direct native worker frames also pass a structural scan before Serde allocates
the request: at most 65,536 string/container nodes and 64 nested containers.
The frame's scalar count has a separate allowance of `B / 4 + 65,536` so
long flat token-ID vectors can reach actual model context; the typed token vector
must still fit its four-byte-per-ID content budget. The HTTP node ceiling does
not impose a 65,536-token ceiling on native worker requests.
Template source size, sandbox restrictions, execution fuel and deadlines remain
separate controls; see [chat template evaluation](GENERATION.md#chat-template-evaluation).

The HTTP API and private app command pipe also check current parsing headroom
for frames above 1 MiB, using twelve times the raw size as a conservative
Unicode/JSON allowance. HTTP reserves that allowance across concurrent readers.
The serialized app pipe checks while receiving a frame and immediately before
decoding; a rejected frame drains through its newline so smaller commands and
Stop remain available. This estimate does not reserve RAM or impose a hard RSS
ceiling.

## Automatic execution for undeclared contexts

The memory planner also admits Auto output for finite-context text models. Their
declared model capacity is the upper bound; it remains unchanged in discovery.

Workers advertising `generation_budget: "memory-context-v1"` accept
`plan_generation` after tokenization:

```json
{"operation":"plan_generation","input_tokens":120,"context_tokens":4294967295}
```

`input_tokens` is the actual rendered input count. `context_tokens` is an upper
budget; the protocol ceiling permits automatic selection. The reply contains
a finite `context_tokens`, `max_new_tokens` equal to that allowance minus the
input, `context_capacity_known` and `budget_kind: "memory"`. Generation then uses
those explicit finite values. A model without a declared capacity continues to
report an undeclared HTTP model capacity.

The native planner performs checked, allocation-free estimates using the loaded
attention cache dimensions and precision, ring bounds, recurrent state and
conservative scratch-pool requirements. It admits a context against one quarter
of currently available RAM after a 256 MiB reserve, also bounded by token-storage
and caller limits. Planning leaves another quarter of that allowance unused for
serialization and ordinary memory-pressure fluctuations before allocation is
rechecked. This is conservative admission evidence; it is not a memory
reservation or a hard process RSS ceiling. Cancellation, output byte limits and
deadlines remain active. Insufficient memory rejects the request cleanly.

Native text generation also checks allocation admission for explicit output
requests. A finite context using sparse Metal buffers is charged for the actual
input plus requested output span; unused reserved address capacity is not mapped
cache memory. Dense buffers and undeclared contexts are charged for their full
selected context. Finite retained states are checked before additional generation
and may be rejected conservatively under memory pressure. Exact reuse of an
undeclared-context state keeps its previously admitted whole-context allocation.

The separate GLM, Qwen and Flint execution paths inventory their loaded cache
geometry, growth rounding, transient replacement buffers and host workspaces.
Retained Lily uses its selected K/V format and loaded draft head when counting
state, snapshots and decode/prefill scratch. The EXL3 bridge already bills device
phases at its installed full context before loading; its host planner additionally
admits token/readout workspaces. These checks do not change inference precision
or replay tokens. Image processing keeps its separately qualified admission path.

Planning never replays a prompt or generated token. A successful query preserves
the retained prefix; the worker keeps an existing planned capacity while it still
fits current admission so context-keyed prefix reuse remains possible. If capacity
changes, the existing context key prevents reuse across the change. Failures clear
retained state through the ordinary worker error path.

## Readiness

The first line a worker writes is its readiness reply. On success it starts with `"ready":true` and carries the capability fields documented on the per-worker page. On failure it is `{"ready":false,"error":CODE}` and the process exits with status 1. Read the readiness line before sending any request.

`version` in the readiness line is the worker crate version (`0.1.0`), which does not change with the release version of the installed bundle. Use `stark version` for the bundle version. Compatibility is signalled by the tag fields instead:

| Worker | Tag field | Current value |
| --- | --- | --- |
| `stark-generate` | `mode` | `text-generation-v1` |
| `stark-generate` | `streaming` | `text-delta-v1` |
| `stark-generate` | `cancellation` | `pipe-v1` |
| `stark-generate` | `generation_budget` | `memory-context-v1` when the loaded text backend exposes an allocation profile |
| `stark-generate` | `image_input.version` | `mediaref-v1` (when the loaded model supports the implemented vision layout) |
| `stark-generate` | `image_input.transport` | `owner-private-file-v1` |
| `stark-generate` | `timing_method` (in generate replies) | `first-token-wall-clock-v1` |
| `stark-generate` | `state_policy` | `fresh-per-request`, or `consecutive-prefix-v1` with `--prefix-cache` on an eligible model ([prefix reuse](GENERATION.md#prefix-reuse)) |
| `stark-score` | `mode` | `exact-token-scoring-cache-v1` or `exact-token-digest-cache-v1` |
| `stark-score` | `cache_policy` | `consecutive-full-vocabulary-logits-v1` or `fresh-state` |

Each tag names the exact request and reply shape documented on the per-worker page at the time of writing. A tag changes when the shape it covers changes incompatibly; new readiness keys can appear without a tag change. Treat an unrecognised tag as incompatible rather than guessing.

Image readiness describes implemented functionality and loaded identity. Its `qualified: false` field leaves the evidence decision to the host. A matching [execution catalog](../tools/converter/stark_convert/vision_runtime_qualifications.json) record enables the exact schema-2 tuple as `reference_verified`. The [family path](VISION_FAMILY_CONTRACT.md) enables an admitted compatible model as `locally_tested` after bounded automatic execution checks. Both bind all five artifact-file hashes, the worker hash observed before launch and checked again at readiness, the Metal device and macOS version; local receipts additionally bind the actual family/version, admitted vocabulary, checker and Python dependency identities. Current readers accept fixed version-1 and config-derived version-2 families; older readers reject the unknown version-2 profile. [Generation](GENERATION.md#image-qualification) defines the evidence levels and scope.

## GPU memory telemetry

An owning host can opt into a separate, private telemetry pipe by setting
`STARK_GPU_MEMORY_FD` to an inherited writable, nonblocking FIFO descriptor of 3
or higher. Both workers remove the variable before loading a model and sample
Metal's `MTLDevice.currentAllocatedSize` about once a second, beginning before
model loading. Each telemetry line contains only `{"gpu_memory_bytes":N}`,
where `N` is the current allocation in bytes. It includes Metal resources in the
worker process, excludes CPU-only allocations and other applications, and can
decrease when resources are released. It is independent of the scoring
readiness field `peak_memory_bytes`.

Telemetry never enters the request/reply stream or changes its framing. The
worker closes the telemetry descriptor on exit; an unavailable Metal device,
invalid pipe or failed write disables reporting without failing inference.
Hosts must tolerate unavailable measurements and keep the pipe private. Without
the variable, workers create no telemetry thread.

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
