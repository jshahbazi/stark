# Text generation

`stark-generate` is the native generation worker. It loads one converted model and serves tokenize, decode and generate requests over a private JSON-lines protocol. Its Qwen3.5 vision path accepts one private image-file reference through a prepared-image transaction. Image availability depends on the host evidence described under [image qualification](#image-qualification), including automatic local checks for the [supported model family](VISION_FAMILY_CONTRACT.md). The calling application renders the chat template, owns tools and the agent loop, and sends either the rendered prompt or token IDs. Conventions shared with the scoring worker (framing, readiness, error classes, privacy) are in [worker protocols](WORKER_PROTOCOLS.md).

## Interactive CLI chat

GLM-5.3-Flash UD-Q2_K_XL and Qwen3.8-Flash-Next mixed 4/8-bit use the
[native hybrid loader](NATIVE_HYBRID_MODELS.md). Their prepared original-format
folders use this same text protocol, sampling, cancellation and fresh-request
state policy. Their initial native implementation is text-only.

The installed bundle includes a terminal chat client built on the worker:

```sh
stark chat --model-path /absolute/path/to/converted-model
```

From a source checkout, build the worker first and use the launcher, which runs the client through the locked converter environment:

```sh
cargo build --release --locked -p stark-generate
./scripts/stark chat --model-path /absolute/path/to/converted-model
./scripts/stark chat --model-path /absolute/path/to/converted-model --image /absolute/path/to/image.png
```

The client renders the converted model's chat template in a Jinja sandbox, keeps the conversation in memory, and sends the complete conversation to one worker for each turn. Replies appear when generation completes; the client does not stream. Type one message per line. `/help` lists commands, `/reset` clears the history, and `/exit`, `/quit`, Ctrl-D or Ctrl-C quit. Conversation history stays in memory.

| Option | Default | Meaning |
| --- | --- | --- |
| `--system TEXT` | model metadata | System message; the template decides how it is rendered |
| `--image PATH` | none | Attach one local PNG or JPEG image; requires passing host image evidence |
| `--context-tokens N` | smaller of 8192 and the worker limit | Context budget for prompt plus reply |
| `--max-new-tokens N` | 512 | Reply budget |
| `--thinking` | off | Enable reasoning for templates that support `enable_thinking` |
| `--timeout SECONDS` | 300 | Startup or text-reply wait; one total budget for each image transaction |

Generation is greedy. A turn whose rendered prompt plus reply budget exceeds the context budget is refused without changing the history; use `/reset` or a larger `--context-tokens`. The menu bar app exposes the same settings through **Advanced Options** ([menu bar guide](../apps/menu-bar/README.md#models-and-requests)).

With `--image`, the CLI reads one regular local file with a 4 MiB limit and accepts PNG or JPEG signatures. The first user turn contains an image placeholder followed by its text. Later turns keep that placeholder in the conversation and resubmit the same owned image bytes for each fresh request; `/reset` clears the conversation and attaches the image to the next first turn. The image stays in process memory between turns. Each request uses the [private staging and transaction lifecycle](#prepared-image-operations). A worker without passing image evidence and an invalid image produce fixed, path-free errors. See the [CLI implementation](../tools/converter/stark_convert/chat.py) and [private image staging](../tools/converter/stark_convert/media.py).

## Image qualification

The [execution catalog](../tools/converter/stark_convert/vision_runtime_qualifications.json) contains two passed schema-2 records for BF16 `Qwen/Qwen3.5-0.8B`: the developer worker `5fc92516…` and Developer ID signed worker `741a3ea7…`. Both use Metal on an Apple M5 Max with 40 GPU cores, Apple GPU family 1010 and MXU enabled, running macOS 26.7.1. Each record binds all five artifact-file hashes, the exact full executable hash, and its source-bound evidence: [developer execution](archive/runs/2026-10-26-qwen35-image-execution.md) and [signed-worker execution](archive/runs/2026-10-27-qwen35-signed-worker.md).

An exact catalog match enables images with `image_verification: "reference_verified"`. The host checks the supported schema-2 transport, all five loaded file hashes, the worker executable hash, the reported Metal device properties and current macOS version. It snapshots the executable hash before launch and hashes the path again at readiness; both must match. Conversion metadata can identify a named qualified build, while serving verifies the executable being launched.

The [supported family profiles](VISION_FAMILY_CONTRACT.md) also admit complete merged BF16 dense Qwen3.5 models and fine-tunes. New version-2 artifacts derive supported decoder/tower dimensions from bounded configuration; version 1 retains the fixed 0.8B graph. After native admission, the host runs bounded text, image, recovery, vocabulary accounting and cleanup checks. Passing enables images with `image_verification: "locally_tested"`. The desktop can reuse a private receipt for an identical artifact, worker, actual family/version, vocabulary, checker, dependencies, device and OS. Terminal chat runs the checks at startup without a persistent receipt. This path also serves a supported schema-2 model when its worker has no exact catalog match. Local checks do not establish reference parity or visual accuracy. Native readiness always reports `image_input.qualified: false`; the host supplies both evidence decisions.

The [Swift and local signed-package replay](archive/runs/2026-10-27-qwen35-swift-packaged-app.md) covers the earlier exact-tuple attachment, GUI, recovery and cleanup checks. Public notarized distribution remains pending. Rebuilding or signing the worker changes its identity, requiring matching reference evidence or fresh local checks. See the [host gate](../tools/converter/stark_convert/vision_runtime.py) and [local checker](../tools/converter/stark_convert/image_checks.py).

## Private JSON-lines protocol

Start the worker with `--model-path PATH`, where `PATH` is a converted model directory. The owning host may also pass `--vision-staging-root PATH` to enable the private media-reference transport. Models that contain a `speculator` directory are refused.

### Readiness

On success the first line is:

```json
{"ready":true,"engine":"stark","version":"0.1.0","mode":"text-generation-v1","operations":["tokenize","decode","generate"],"state_policy":"fresh-per-request","vocab_size":151936,"max_context_tokens":16384,"max_new_tokens":4096,"eos_token_ids":[151645,151643],"sampling_modes":["greedy","stochastic"],"streaming":"text-delta-v1","stop_sequences":true,"cancellation":"pipe-v1"}
```

| Field | Meaning |
| --- | --- |
| `mode` | Protocol tag for this contract |
| `operations` | Accepted `operation` values |
| `state_policy` | `fresh-per-request`: no state survives between requests |
| `vocab_size` | Upper bound for every token ID |
| `max_context_tokens` | Smaller of 16,384 and the model's maximum context |
| `max_new_tokens` | Fixed cap of 4,096 output tokens per request |
| `eos_token_ids` | Stop IDs from the model's generation metadata, always honoured |
| `streaming`, `stop_sequences`, `cancellation` | Capability tags for the features below |

An image-capable model adds an `image_input` object without changing `mode: "text-generation-v1"` or the 1 MiB line framing. Its `version` is `mediaref-v1`, its `transport` is `owner-private-file-v1`, and it lists the `prepare_image`, `generate_image` and `discard_image` operations. The object identifies artifact schema 2 or 3, family profile/version, source/revision, SHA-256 hashes for the weights, config, tokenizer, processor and template, and the actual Metal runtime identity. The host applies its [evidence checks](#image-qualification) before exposing image input. See the [readiness implementation](../crates/stark-generate/src/generation.rs).

### Prepared image operations

The owning host creates an empty mode-0700 staging directory and launches the worker with `--vision-staging-root PATH`. At startup the worker requires an absolute, owner-private directory with no entries. The host then writes mode-0600 regular files named with opaque IDs. The JSON line contains only an ID, byte length and media type; image bytes never enter worker JSON. The engine accepts PNG and JPEG up to 20 MiB compressed, while the terminal and HTTP services apply their narrower 4 MiB limit. Image processing is bounded by 32 million source pixels, a 16,384-pixel maximum source dimension, 1,048,576 processed pixels, 4,096 patches and 1,024 image tokens.

The owning host sends `prepare_image` with the rendered prompt, file reference, context limit and output limit. The worker validates the opened file's owner, regular-file type, private permissions, single link, declared length, stable metadata and image signature. It immediately unlinks the name before processing and retains the opened-file lease through terminal completion. A successful prepare reply provides a single-use `prepared_id` and token counts. The host sends `generate_image` with that ID and the usual generation options, then consumes a `discard_image` reply to clear any remaining prepared state. Prepared state belongs to one loaded model and is consumed once. A different operation, request error, cancellation, process failure or explicit discard clears it. Each image request repeats this transaction with fresh model state.

The host serializes the whole transaction and uses one deadline across frame writes, preparation, generation, streaming reads and discard; the budget does not restart for each operation. A protocol failure or expired deadline closes and reaps the worker. The host's staging cleanup removes an unclaimed file on every exit path. Worker teardown also removes remaining opaque, owner-private, single-link regular files in its pinned directory. The exact request structs and cleanup fence are in [`generation.rs`](../crates/stark-generate/src/generation.rs); the file checks and teardown are in [`media.rs`](../crates/stark-generate/src/media.rs).

### Tokenize

```json
{"operation":"tokenize","text":"rendered text","add_special_tokens":false}
```

Reply: `{"token_ids":[...],"token_count":N}`. `add_special_tokens` defaults to `false`; most chat templates already include the special tokens.

### Decode

```json
{"operation":"decode","token_ids":[3,4],"skip_special_tokens":false}
```

Reply: `{"text":"...","token_count":N}`. An empty vector is valid. The whole vector is decoded at once so multi-token characters are reproduced correctly.

### Generate

```json
{"operation":"generate","rendered_prompt":"complete rendered chat template","context_tokens":8192,"max_new_tokens":512,"seed":0,"sampling":{"mode":"greedy"},"stop_token_ids":[],"stop":[],"stream":false,"request_id":1}
```

| Field | Rule |
| --- | --- |
| `rendered_prompt` | String, tokenized with `add_special_tokens=false`. Supply exactly one of this field or `input_token_ids`. |
| `input_token_ids` | Non-empty array of token IDs below `vocab_size`. |
| `context_tokens` | Required, 1 through `max_context_tokens`. The input must be strictly shorter than this value so at least one output slot remains. |
| `max_new_tokens` | Required, 1 through 4,096. |
| `seed` | Optional unsigned 64-bit integer, default 0. Only stochastic sampling uses it. |
| `sampling` | Optional, default `{"mode":"greedy"}`. See [Sampling](#sampling). |
| `mtp_drafts` | Optional integer, default 0 (disabled). Values 1–3 enable exact MTP on a supported native Lily or official mixed-affine Qwen3.8 checkpoint containing its trained head; other models return `mtp_not_supported`. Applies to text `generate` only. |
| `stop_token_ids` | Optional, at most 256 distinct token IDs below `vocab_size`. Added to the model's EOS IDs; a model EOS cannot be removed. |
| `stop` | Optional, at most four non-empty strings of at most 1,024 characters each. See [Stop sequences](#stop-sequences). |
| `stream` | Optional boolean, default `false`. See [Streaming](#streaming). |
| `request_id` | Optional unsigned 64-bit integer, default 0. See [Cancellation](#cancellation). |

The worker honors the union of `generation_config.stop_token_ids` and the tokenizer ID for a nonempty `token_codec_config.eos_token`. The chat EOS must resolve to a single vocabulary token; an unresolved declaration fails model loading with `invalid_model_config`. This covers checkpoints whose end-of-text and end-of-turn declarations differ, including existing converted models. The ColGrep checkpoint declares both EOS IDs, `1` and `130073`; both are honored automatically. Caller stops supplement model stops. A terminal stop is excluded from `generated_token_ids` and `text`.

The terminal reply:

| Field | Meaning |
| --- | --- |
| `generated_token_ids` | Generated IDs, excluding a terminal EOS or stop token |
| `text` | Decoding of those IDs with special tokens preserved, minus any matched stop sequence |
| `input_tokens` | Number of prompt token IDs used |
| `output_tokens` | Length of `generated_token_ids` |
| `sampled_tokens` | `output_tokens` plus one when a terminal stop token was consumed |
| `finish_reason` | `eos`, `stop_token`, `stop_sequence`, `length` (output budget reached) or `context` (fewer output slots were available than `max_new_tokens`) |
| `stop_token_id` | The terminal stop token, otherwise `null` |
| `generation_seconds` | Wall-clock time for state setup, generation, cleanup and decoding |
| `prefill_seconds` | Time from state setup until the first sampled token |
| `decode_seconds` | Time between the first and last sampled tokens; `null` when fewer than two tokens were sampled |
| `input_tokens_per_second` | `input_tokens / prefill_seconds`, or `null` when unmeasurable |
| `output_tokens_per_second` | `(sampled_tokens - 1) / decode_seconds`, or `null` when unmeasurable |
| `timing_method` | `first-token-wall-clock-v1` |
| `mtp` | Present for an MTP-capable model: selected `enabled`/`drafts`, `sampling: "exact-rejection-v1"`, the admitted `verification` route, and aggregate round, proposal, acceptance, verified-row and phase-time counters. |

The prefill figure is a wall-clock proxy that includes native state allocation, command encoding, GPU waiting and any lookahead decode the engine schedules before returning its first token. The output rate counts sampling steps after the first token, including a terminal stop token. Prompt tokenization is excluded from all timings. The prompt plus sampled tokens never exceed `context_tokens`; there is no silent truncation.

The desktop bridge forwards these counts and timings in a content-free `metrics` event after successful built-in chat and local API requests, including streaming API requests. The menu-bar Input/Output display consumes this event independently of chat history. Failed, cancelled, or stale requests do not publish completion metrics; a missing or unmeasurable rate remains unavailable.

### Sampling

Greedy decoding is the default. Stochastic sampling supplies a temperature and optional filters:

```json
{"mode":"stochastic","temperature":0.6,"top_k":20,"top_p":0.95,"min_p":0.0}
```

`temperature` must be finite, positive and at most 100. `top_k` is 1 through `vocab_size`. `top_p` is greater than 0 and at most 1. `min_p` is 0 through 1. Omitted filters are disabled. Repetition, frequency and presence penalties are not part of this interface.

The converted-model sampler applies temperature before filtering and evaluates
top-k and top-p against the same full-vocabulary distribution. The native hybrid
sampler retains top-k candidates first, then computes the mass used by top-p
over those candidates; omitting top-k uses the full vocabulary. Seeded requests
are reproducible on the same build and hardware; they are not expected to match
another engine token for token.

### Qwen3.8 multi-token prediction

An admitted native Qwen3.8 checkpoint containing its trained MTP head adds
`mtp: {"supported": true, "max_drafts": 3, "default_drafts": 0,
"sampling": "exact-rejection-v1", "verification": "batched-decode-exact-v1"}` to readiness for Lily-format weights.
Official mixed-affine weights report `verification: "serial-decode-exact-v1"`.
Responses retain the same route label. The generation worker loads
the head with the checkpoint's existing quantization, including its memory
cost when drafting is disabled. Set `mtp_drafts` to 1,
2 or 3 on a text request; omit it or use 0 for ordinary autoregressive decode.
Requests on one worker remain serialized and start with fresh state.

Both target and draft use complete F32 readouts and the existing temperature,
top-k, top-p and min-p filtering policy. The host accepts a proposed token with
probability `min(1, p/q)` and samples a rejection from normalized `max(p-q, 0)`.
The bonus token uses the target distribution. This is exact rejection sampling;
there is no candidate cap or approximate acceptance threshold. Target
verification retains ordinary decode arithmetic and its precision boundaries. Exact sampling preserves the target
distribution, rather than requiring identical token sequences at the same seed
with MTP on and off. Each route is reproducible on matching builds and hardware.

Draft randomness is separate from target randomness. Draft distributions remain
in request memory only. Rejected suffixes rewind GDN recurrent state,
convolution windows, n-gram history and the draft head's committed history.
Early termination drains work and rewinds a partly delivered group without
calling the sampler during cleanup. MTP adds memory and verification work, so
its benefit depends on acceptance and the workload. The Lily verifier evaluates
up to four positions in one graph, reusing weights while retaining each
position's decode arithmetic and recurrent-state update order.
The official mixed-affine path executes ordinary target decode separately
for each row. Its ValueKey GDN state, stock HC arithmetic, affine-64 cache and
BF16-valued vocabulary head require that serial fallback; the Lily batch
kernels do not preserve those boundaries. Uzu text conversions omit the head.

Unbounded nucleus requests can reuse a proved prefix for both draft and
target distributions. Retained IDs, probability terms, cumulative totals
and RNG consumption match the full-order sampler; uncertain boundaries
fall back to that reference. Top-k requests retain their existing route.
It is disabled by default.

### Streaming

With `stream: true`, the worker writes zero or more delta lines while generating, then the normal terminal reply:

```json
{"event":"delta","text":"Hel"}
{"event":"delta","text":"lo"}
{"generated_token_ids":[...],"text":"Hello", ...}
```

Consume every delta and the terminal reply before sending another request. The concatenated deltas equal the terminal `text`. Incremental decoding buffers incomplete multi-byte characters, so a delta never splits a character; a final delta may be emitted after native cleanup to flush such a buffer. Streaming timings include decoding and output back-pressure.

### Stop sequences

`stop` strings are matched against the decoded text as it is produced, including across token boundaries. Text from the first matching sequence onward is withheld from both deltas and the terminal `text`, and `finish_reason` is `stop_sequence`. `generated_token_ids` and the token counts still include the token that completed the match, so decoding that vector yourself can reveal the withheld text. No extra token is generated after a match.

### Cancellation

If the environment variable `STARK_CANCEL_FD` names an inherited pipe descriptor (3 or higher, a FIFO), the worker reads it for cancellation frames: eight little-endian bytes holding the `request_id` to cancel. The worker checks the pipe before allocating state and around every token. Image requests also check before and after CPU preparation and after vision work has drained, before decoder work starts. A frame matching the running request stops generation after native cleanup and returns the recoverable error `generation_cancelled`; the model stays loaded. Frames for other IDs are discarded. End of file on the pipe cancels the running request and every later one. The worker removes `STARK_CANCEL_FD` from its environment before loading the model. Without the variable, the only way to interrupt a request is to terminate the process.

### Error codes

Errors are replies of the form `{"error":CODE}`. Codes are fixed strings; see [error classes](WORKER_PROTOCOLS.md#error-classes) for the behaviour of each class.

| Class | Codes |
| --- | --- |
| Recoverable (worker continues) | `invalid_request` (unparseable JSON, unknown operation or field, wrong type), `invalid_generation_request`, `invalid_sampling_request`, `invalid_decoding_request`, `tokenization_failed`, `context_overflow` (context above the model maximum, or input not shorter than `context_tokens`), `request_too_large` (line over 1 MiB; the rest of the line is discarded), `generation_cancelled`, `mtp_not_supported`, `images_not_supported`, `invalid_image_reference`, `image_read_failed`, `invalid_image_request`, `invalid_prepared_image`, `image_request_setup_failed` |
| Fencing (error reply, then `{"ready":false,"error":"native_operation_failed"}` and exit 1) | `native_generation_failed`, `invalid_generation_output`, `invalid_tokenization_output`, `invalid_image_preparation_output`, `decoding_failed`, `cancel_channel_failed` |
| Startup (`{"ready":false,"error":CODE}` and exit 1) | `invalid_arguments`, `output_setup_failed`, `invalid_cancel_channel`, `invalid_vision_staging_root`, `model_config_unavailable`, `invalid_model_config`, `speculator_not_supported`, `metal_initialization_failed`, `model_load_failed`, `native_backend_unavailable` (a build without the `metal` feature), `native_operation_failed` (panic during startup) |

A write failure on the reply pipe also ends the process with exit status 1.

## State and privacy

Every generate request creates a fresh native state sized to `context_tokens`, runs the prompt through prefill, generates, and drops the state before the terminal reply is written. No KV cache or prompt prefix survives between requests, so each turn of a conversation re-processes the whole conversation. This is the `fresh-per-request` policy advertised in readiness.

The worker suppresses native diagnostics and capture variables as described in [worker protocols](WORKER_PROTOCOLS.md#privacy-and-diagnostics). It opens no network listener and executes no tools.

## Validation

Model-free tests cover request parsing and bounds, stop handling, UTF-8 decoding, stream cleanup, fresh state per request, seeded option dispatch, sanitized startup errors and failure fencing:

```sh
cargo test --locked -p stark-generate --no-default-features
cargo test --release --locked -p stark-generate -- --test-threads=1
uv run --no-project --python 3.12.13 python scripts/test_generation_process.py --binary target/release/stark-generate
```

The model-bound acceptance record for the `lightonai/colgrep-agent-2B` checkpoint, including how to reproduce it, is in [ColGrep generation validation](validation/GENERATION_COLGREP.md).
