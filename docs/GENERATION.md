# Text generation

`stark-generate` is the native generation worker. It loads one converted model and serves tokenize, decode and generate requests over a private JSON-lines protocol. Its Qwen3.5 and Gemma 4 E2B vision paths accept one private image-file reference through a prepared-image transaction. Image availability depends on the host evidence described under [image qualification](#image-qualification), including automatic model-specific checks. The Gemma 4 path uses a separate schema-4 BF16 artifact; full worker end-to-end validation remains pending. The calling application renders the chat template, owns tools and the agent loop, and sends either the rendered prompt or token IDs. Conventions shared with the scoring worker (framing, readiness, error classes, privacy) are in [worker protocols](WORKER_PROTOCOLS.md).

## Interactive CLI chat

GLM-5.3-Flash UD-Q2_K_XL and Qwen3.8-Flash-Next mixed 4/8-bit use the
[native hybrid loader](NATIVE_HYBRID_MODELS.md). Their prepared original-format
folders use this same text protocol, sampling, cancellation and scoped
[prefix-reuse policy](#prefix-reuse). Their native implementation is text-only.

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

The client renders the converted model's chat template in a Jinja sandbox, keeps the conversation in memory, and sends the complete conversation to one worker for each turn. For eligible text models the worker reuses the prompt prefix shared with the previous turn ([prefix reuse](#prefix-reuse)); `/reset` starts a new reuse scope. Replies appear when generation completes; the client does not stream. Type one message per line. `/help` lists commands, `/reset` clears the history, and `/exit`, `/quit`, Ctrl-D or Ctrl-C quit. Conversation history stays in memory.

When a backend has no declared finite context maximum, the CLI tokenizes the conversation and asks the loaded worker to plan a finite execution allowance from its actual state geometry and available memory. No context or output option is required. The unsigned 32-bit ceiling remains a protocol range. A custom context is an upper bound; an explicit output limit is honored subject to memory admission. Image chat still requires a declared finite model context.

| Option | Default | Meaning |
| --- | --- | --- |
| `--system TEXT` | model metadata | System message; the template decides how it is rendered |
| `--image PATH` | none | Attach one local PNG or JPEG image; requires passing host image evidence |
| `--context-tokens N` | model maximum, or automatic memory planning when undeclared | Upper context budget for prompt plus reply |
| `--max-new-tokens N` | remaining context or memory-planned execution allowance | Explicit reply budget when supplied |
| `--thinking` | off | Enable reasoning where supported; GLM uses Max effort when on and Low when off |
| `--mtp`, `--no-mtp` | off | Enable or disable exact MTP for this terminal session; enabling uses the loaded model's recommended draft count |
| `--timeout SECONDS` | 300 | Startup or text-reply wait; one total budget for each image transaction |

Generation is greedy. By default, the client tokenizes each rendered prompt and uses the remaining finite context for output, stopping on EOS or another termination condition. An explicit reply budget that does not fit beside the rendered prompt is refused without changing the history; use `/reset` or a larger `--context-tokens`. The menu bar app exposes generation limits, thinking and a supported-model MTP toggle through **Advanced Options** ([menu bar guide](../apps/menu-bar/README.md#models-and-requests)). The app uses model-specific MTP defaults and remembers an explicit choice per model; the terminal session defaults off.

Terminal chat consumes worker streaming frames internally, including prefill
progress, while displaying the completed reply. `--timeout` limits inactivity
between frames. GLM and Lily report content-free prefill progress at existing
execution control boundaries, so an active long prefill renews the timeout.
Lily checks cancellation between its existing prefill passes. A submitted pass
finishes before cancellation resets the request and clears its prefix state.

With `--image`, the CLI reads one regular local file with a 4 MiB limit and accepts PNG or JPEG signatures. The first user turn contains an image placeholder followed by its text. Later turns keep that placeholder in the conversation and resubmit the same owned image bytes for each fresh request; `/reset` clears the conversation and attaches the image to the next first turn. The image stays in process memory between turns. Each request uses the [private staging and transaction lifecycle](#prepared-image-operations). A worker without passing image evidence and an invalid image produce fixed, path-free errors. See the [CLI implementation](../tools/converter/stark_convert/chat.py) and [private image staging](../tools/converter/stark_convert/media.py).

## Image qualification

The [execution catalog](../tools/converter/stark_convert/vision_runtime_qualifications.json) contains two passed schema-2 records for BF16 `Qwen/Qwen3.5-0.8B`: the developer worker `5fc92516…` and Developer ID signed worker `741a3ea7…`. Both use Metal on an Apple M5 Max with 40 GPU cores, Apple GPU family 1010 and MXU enabled, running macOS 26.7.1. Each record binds all five artifact-file hashes, the exact full executable hash, and its source-bound evidence: [developer execution](archive/runs/2026-10-26-qwen35-image-execution.md) and [signed-worker execution](archive/runs/2026-10-27-qwen35-signed-worker.md).

An exact catalog match enables images with `image_verification: "reference_verified"`. The host checks the supported schema-2 transport, all five loaded file hashes, the worker executable hash, the reported Metal device properties and current macOS version. It snapshots the executable hash before launch and hashes the path again at readiness; both must match. Conversion metadata can identify a named qualified build, while serving verifies the executable being launched.

The [supported family profiles](VISION_FAMILY_CONTRACT.md) also admit complete merged BF16 dense Qwen3.5 models and fine-tunes. New version-2 artifacts derive supported decoder/tower dimensions from bounded configuration; version 1 retains the fixed 0.8B graph. After native admission, the host runs bounded text, image, recovery, vocabulary accounting and cleanup checks. Passing enables images with `image_verification: "locally_tested"`. The desktop can reuse a private receipt for an identical artifact, worker, actual family/version, vocabulary, checker, dependencies, device and OS. Terminal chat runs the checks at startup without a persistent receipt. This path also serves a supported schema-2 model when its worker has no exact catalog match. Local checks do not establish reference parity or visual accuracy. Native readiness always reports `image_input.qualified: false`; the host supplies both evidence decisions.

Gemma 4 E2B uses its own schema-4 `gemma4-e2b-bf16-v1` artifact and fixed vision geometry. Its source-pinned converter and worker path, terminal chat, desktop handling and family gates are implemented. The canonical Hugging Face CPU image-feature comparison passes a 2% error gate on three fixtures. Full worker end-to-end validation and performance measurement remain pending; this feature result is not a complete chat qualification. The artifact contains five admitted components: config, model weights, tokenizer, processor and template. Gemma image preprocessing is bounded to 2,520 patches and 280 pooled soft tokens; see [OLMo 3 and Gemma 4 support](OLMO3_GEMMA4.md) for conversion and reference-profile details.

The [Swift and local signed-package replay](archive/runs/2026-10-27-qwen35-swift-packaged-app.md) covers the earlier exact-tuple attachment, GUI, recovery and cleanup checks. Public notarized distribution remains pending. Rebuilding or signing the worker changes its identity, requiring matching reference evidence or fresh local checks. See the [host gate](../tools/converter/stark_convert/vision_runtime.py) and [local checker](../tools/converter/stark_convert/image_checks.py).

## Chat template evaluation

Model-provided Jinja chat templates are compiled and rendered by the Rust
`stark-template` helper using MiniJinja 2.24.0. Each compilation check and render
starts a fresh native process. No Python interpreter is started for evaluation,
and there is no Python rendering fallback. The caller transfers only bounded
JSON over private pipes and keeps the existing model-worker transaction
serialized.

The evaluator limits template source to 256 KiB, source tokens to 16,384,
VM fuel to 1,000,000 and runtime recursion to 128. Input collections, value
depth and node counts are also bounded. Request and output sizes use the
[automatic shared resource budget](WORKER_PROTOCOLS.md#resource-budget), selected
from available RAM at startup with a 16 MiB floor and no fixed content ceiling. With
logical budget `B`, the encoded request is limited to `6B + 2 MiB`, and rendered
output to `B`. Hosts pin
the selected budget for child helpers; `STARK_REQUEST_BUDGET_MIB` is a
developer/operator override, with no app setting required.

A process-wide allocator bounds live requests at `128 MiB + 8B` and cumulative
requests at `32 MiB + 32B`.
Accounting includes
JSON parsing, compilation, intermediate values, captures, slices and output.
Reallocation charges the full replacement size. This counts actual Rust
allocation requests; it does not measure resident memory, native stacks or
allocator overhead.

Each child has a hard 2-second CPU limit and disables core dumps and regular-file
output. The parent enforces a 5-second wall deadline and samples RSS every 10 ms,
terminating the child above the selected live-allocation budget. This RSS
watchdog is not a kernel-enforced memory ceiling. Other POSIX
systems additionally apply that budget as an address-space limit. Cancellation
terminates and reaps the disposable child. Template source,
inputs, rendered output and compiler diagnostics are never logged or persisted
by the evaluator.

The helper exposes a restricted set of globals, methods and filters, with no
filesystem loader or network access API. Unsupported operations, reflective
attributes, includes/imports and over-limit programs fail with fixed errors.
Reviewed model templates are tested against Jinja2 for exact output, including
tool history and JSON formatting. MiniJinja is not a complete Python runtime;
unsupported method variants and out-of-range integer inputs are rejected.
Plain `set` and `for` bindings may declare single-underscore temporary names.
Dunder names, undeclared underscore names and underscore-prefixed member access
remain rejected. Underscore-prefixed root argument bindings are omitted before
rendering, so an unbound local cannot fall back to caller-provided private data.
Nested JSON keys are preserved. Loop metadata includes `previtem` and `nextitem` so templates
can group tool results across multiple completed tool cycles.

`tojson` accepts an optional `indent` from 0 through 16 or `none`, and an
explicit boolean `ensure_ascii` (default `true`). Keys are sorted and HTML
characters remain escaped under either ASCII policy. Other keywords and
non-boolean ASCII options are rejected; the same allocation and output limits
apply.

Tuple-valued expressions are unsupported because this MiniJinja version
represents them as lists. Constant-string tuple literals are admitted only as
direct `in` / `not in` membership operands, where those collection semantics
agree. Other tuple literals are rejected; list literals and unpacking remain
supported.
See the [native implementation](../crates/stark-template/src/engine.rs) and
[host lifecycle wrapper](../tools/converter/stark_convert/template_runtime.py).

Fresh process startup and template compilation remain intentional security
costs. Normal chat renders once per turn before tokenization and inference;
decision scoring renders twice to verify the assistant answer boundary.
The evaluator adds no work to the per-token decode loop. The
[native latency study](benchmarks/NATIVE_CHAT_TEMPLATE_2026_11_03.md) compares
this path with the preserved [Python isolation baseline](benchmarks/CHAT_TEMPLATE_ISOLATION_2026_11_03.md)
and records the scope and limits of the measurements.

Installed distributions bundle `libexec/stark-template`, checked by `stark doctor`.
Source checkouts must build it before running chat or converter tests:

```sh
cargo +1.98.1 build --release --locked -p stark-template
```

## Private JSON-lines protocol

The readiness example below uses the automatic 16 MiB floor; the effective
budget may be larger on a machine with more available memory.

Start the worker with `--model-path PATH`, where `PATH` is a converted model directory. The owning host may also pass `--vision-staging-root PATH` to enable the private media-reference transport and `--prefix-cache` to request [prefix reuse](#prefix-reuse). Models that contain a `speculator` directory are refused.

### Readiness

On success the first line is a readiness object. This example's vocabulary,
context limit, EOS IDs and MTP support vary by loaded model:

```json
{"ready":true,"engine":"stark","version":"0.1.0","mode":"text-generation-v1","operations":["tokenize","decode","generate"],"state_policy":"fresh-per-request","vocab_size":151936,"max_context_tokens":262144,"max_new_tokens":262143,"request_budget_bytes":16777216,"max_request_bytes":102760448,"eos_token_ids":[151645,151643],"sampling_modes":["greedy","stochastic"],"streaming":"text-delta-v1","stop_sequences":true,"cancellation":"pipe-v1","mtp":{"supported":false,"max_drafts":0,"recommended_drafts":0,"default_drafts":0,"operations":[]}}
```

| Field | Meaning |
| --- | --- |
| `mode` | Protocol tag for this contract |
| `operations` | Accepted `operation` values |
| `generation_budget` | `memory-context-v1` when the loaded text backend supports native memory planning |
| `state_policy` | `fresh-per-request`: no state survives between requests. `consecutive-prefix-v1` when [prefix reuse](#prefix-reuse) is enabled |
| `prefix_cache` | Present only when the worker was started with `--prefix-cache`: `supported`, `enabled`, `chunk_tokens` (the reuse grid, a multiple of 1,024: 1,024 for converted models and native GLM-5.3, 4,096 for native mixed-precision Qwen3.8), `idle_seconds` (600) and the reuse `key`. When enabled, `operations` also lists `reset_cache` |
| `vocab_size` | Upper bound for every token ID |
| `max_context_tokens` | Loaded model's configured maximum context; no global 16,384-token cap. A backend with no finite maximum reports the protocol's unsigned 32-bit maximum. |
| `max_new_tokens` | Model context capacity minus one, or the unsigned 32-bit protocol ceiling minus one when no finite context is declared; actual input still consumes context |
| `request_budget_bytes`, `max_request_bytes` | Configured logical content budget and encoded request-line budget, including the newline; see [resource budget](WORKER_PROTOCOLS.md#resource-budget) |
| `eos_token_ids` | Stop IDs from the model's generation metadata, always honoured |
| `streaming`, `stop_sequences`, `cancellation` | Capability tags for the features below |
| `mtp` | Loaded-model MTP support, recommended/maximum draft counts, supported operations and exact-sampling policy; see [MTP](#multi-token-prediction) |

The configured context is a token admission boundary. Available memory and the configurable content/frame budgets still apply; accepting a model's published maximum does not qualify full-length numerical accuracy or performance.

An image-capable model adds an `image_input` object without changing `mode: "text-generation-v1"` or the shared line-framing budget. Its `version` is `mediaref-v1`, its `transport` is `owner-private-file-v1`, and it lists the `prepare_image`, `generate_image` and `discard_image` operations. The object identifies artifact schema 2, 3 or 4, family profile/version, source/revision, SHA-256 hashes for the weights, config, tokenizer, processor and template, and the actual Metal runtime identity. The host applies its [evidence checks](#image-qualification) before exposing image input. See the [readiness implementation](../crates/stark-generate/src/generation.rs).

### Prepared image operations

The owning host creates an empty mode-0700 staging directory and launches the worker with `--vision-staging-root PATH`. At startup the worker requires an absolute, owner-private directory with no entries. The host then writes mode-0600 regular files named with opaque IDs. The JSON line contains only an ID, byte length and media type; image bytes never enter worker JSON. The engine accepts PNG and JPEG up to 20 MiB compressed, while the terminal and HTTP services apply their narrower 4 MiB limit. Image processing is bounded by 32 million source pixels, a 16,384-pixel maximum source dimension, 1,048,576 processed pixels, 4,096 patches and 1,024 image tokens.

The owning host sends `prepare_image` with the rendered prompt, file reference, context limit and output limit. The worker validates the opened file's owner, regular-file type, private permissions, single link, declared length, stable metadata and image signature. It immediately unlinks the name before processing and retains the opened-file lease through terminal completion. A successful prepare reply provides a single-use `prepared_id` and token counts. The host sends `generate_image` with that ID and the usual generation options, then consumes a `discard_image` reply to clear any remaining prepared state. Prepared state belongs to one loaded model and is consumed once. A different operation, request error, cancellation, process failure or explicit discard clears it. Each image request repeats this transaction with fresh model state. Gemma E2B additionally limits processed image input to at most 2,520 16-pixel patches and 280 pooled soft tokens; this model-specific bound is lower than the Qwen3.5 image limits above.

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
| `max_new_tokens` | Required, positive and at most `context_tokens - 1`; there is no separate fixed output-token ceiling. The actual rendered input and requested output must fit the context. |
| `seed` | Optional unsigned 64-bit integer, default 0. Only stochastic sampling uses it. |
| `sampling` | Optional, default `{"mode":"greedy"}`. See [Sampling](#sampling). |
| `mtp` | Optional boolean, default `false`. `true` uses this model's `recommended_drafts`; `false` disables MTP. Unsupported models reject `true` with `mtp_not_supported`. Applies to text `generate` only. |
| `mtp_drafts` | Optional explicit draft count, default 0. Nonzero values must fit the loaded model's `max_drafts`. Retained for advanced callers and benchmarks. Supply either `mtp` or `mtp_drafts`; supplying both, including `false` and `0`, is invalid. |
| `stop_token_ids` | Optional, at most 256 distinct token IDs below `vocab_size`. Added to the model's EOS IDs; a model EOS cannot be removed. |
| `stop` | Optional, at most four non-empty strings of at most 1,024 characters each. See [Stop sequences](#stop-sequences). |
| `stream` | Optional boolean, default `false`. See [Streaming](#streaming). |
| `request_id` | Optional unsigned 64-bit integer, default 0. See [Cancellation](#cancellation). |
| `cache_scope` | Optional, only when prefix reuse is enabled: 32 lowercase hexadecimal characters chosen by the host. See [Prefix reuse](#prefix-reuse). Otherwise the field is rejected with `invalid_generation_request`. |

The worker honors the union of `generation_config.stop_token_ids` and the tokenizer ID for a nonempty `token_codec_config.eos_token`. The chat EOS must resolve to a single vocabulary token; an unresolved declaration fails model loading with `invalid_model_config`. This covers checkpoints whose end-of-text and end-of-turn declarations differ, including existing converted models. The ColGrep checkpoint declares both EOS IDs, `1` and `130073`; both are honored automatically. Caller stops supplement model stops. A terminal stop is excluded from `generated_token_ids` and `text`.

The terminal reply:

| Field | Meaning |
| --- | --- |
| `generated_token_ids` | Generated IDs, excluding a terminal EOS or stop token |
| `text` | Decoding of those IDs with special tokens preserved, minus any matched stop sequence |
| `input_tokens` | Number of prompt token IDs used |
| `output_tokens` | Length of `generated_token_ids` |
| `sampled_tokens` | `output_tokens` plus one when a terminal stop token was consumed |
| `finish_reason` | `eos`, `stop_token`, `stop_sequence`, `length` (output budget reached) or `context` (context boundary reached) |
| `stop_token_id` | The terminal stop token, otherwise `null` |
| `generation_seconds` | Wall-clock time for state setup, generation, cleanup and decoding |
| `prefill_seconds` | Time from state setup until the first sampled token |
| `decode_seconds` | Time between the first and last sampled tokens; `null` when fewer than two tokens were sampled |
| `input_tokens_per_second` | `(input_tokens - reused_input_tokens) / prefill_seconds`, the prompt tokens actually processed, or `null` when unmeasurable |
| `reused_input_tokens` | Present only when prefix reuse is enabled: prompt tokens taken from the retained prefix instead of prefill, a multiple of `prefix_cache.chunk_tokens` below `input_tokens` |
| `output_tokens_per_second` | `(sampled_tokens - 1) / decode_seconds`, or `null` when unmeasurable |
| `timing_method` | `first-token-wall-clock-v1` |
| `mtp` | Present for an MTP-capable model: selected `enabled`/`drafts`, `sampling: "exact-rejection-v1"`, the admitted `verification` route, and aggregate round, proposal, accepted-prefix-count, verified-row and phase-time counters when available. |

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

<a id="qwen38-multi-token-prediction"></a>

### Multi-token prediction

Every worker reports `mtp` capabilities from its loaded backend. Unsupported
models report `supported: false`, zero draft counts and an empty `operations`
list. Supported models report their own recommendation and maximum:

```json
{"supported":true,"max_drafts":3,"recommended_drafts":2,"default_drafts":0,"sampling":"exact-rejection-v1","verification":"batched-affine-decode-exact-v1","operations":["generate"]}
```

The contract supports additional model backends without a global draft limit
or model-name check in callers. A backend advertises support only after it
has loaded and admitted its MTP implementation. A checkpoint's declared head
alone does not establish runtime support. Capabilities stay fixed for the
worker lifetime; a contradictory backend capability or completion
acknowledgement fences the worker.

Use `"mtp": true` for the recommendation, `"mtp": false` to disable it, or
`"mtp_drafts": N` for an explicit supported depth. If neither selector is
present, the worker uses the loaded backend's `default_drafts`, reported in
readiness; this is normally zero. The explicitly staged GLM EXL3
`bf16-native-experimental` profile sets that default to three. Send
`"mtp": false` or `"mtp_drafts": 0` to request serial generation for that
profile. The default `headless-reference` stage and other model families retain
their existing defaults. `mtp` requires a boolean and `mtp_drafts` an unsigned
integer; `null` and conflicting selectors are rejected. Completion metadata
reports the resolved `enabled` state and `drafts` limit,
`sampling` and `verification`, plus backend counters when available.
Selection is request-local, so changing it requires no model reload and does
not carry a mode into later native requests. Current operations are text-only.

Native Lily and official mixed-affine Qwen3.8 checkpoints with their trained
head currently implement this capability. They recommend two drafts with a
maximum of three. Lily-format weights report
`verification: "batched-decode-exact-v1"`; official mixed-affine weights
report `verification: "batched-affine-decode-exact-v1"`.
Responses retain the same route label. The generation worker loads
the head with the checkpoint's existing quantization, including its memory
cost when drafting is disabled. Set `mtp_drafts` to 1,
2 or 3 on a text request; omit it or use 0 for ordinary autoregressive decode.
Requests on one worker remain serialized. With `--prefix-cache`, a request resumes a [prefix snapshot](#prefix-reuse) only when it enables or disables MTP the same way as the request that captured it.

Converted dense `qwen3_5` models with an admitted quantized one-layer MTP head
report `verification: "dense-qwen35-target-verify-v1"`, recommend three drafts
and allow up to three. Drafting defaults to zero. This route includes
`Jundot/Qwen3.8-27B-oQ4e-mtp`. The target and predictor share the model's
embedding and vocabulary head; their BF16 logits are converted to F32 for the
rejection sampler. The draft policy applies the target's final normalization
to its captured residual, then supplies that hidden state and the next token
embedding to the predictor. Chained drafts use the predictor's normalized
output, matching the pinned oMLX post-normalization proposal variant. Accepted rows commit target
recurrent and attention state; the predictor rewinds and rebuilds from the
committed target residuals after every round. A zero-draft request uses the
ordinary target stream with the head resident, allocating no predictor session.
It retains an exact target-only chunk snapshot when prefix caching is enabled.
The round repair preserves the predictor's already-valid root KV and replays
only accepted proposal rows using the preceding target hidden rows; root-only
acceptance needs no replay. An internal zero-draft verification round primes
the root through the same cache-only route. That route keeps embedding, hidden normalization, fusion,
pre-attention normalization, Q/K/V projection, key normalization, RoPE and KV
writes, while skipping unused attention output, FFN and final normalization.
Target commit, repair and last-hidden ownership share one command completion:
the three-draft path uses five normal command completions, down from eight.
Completion metadata records aggregate round counts for zero through three
accepted drafts; it contains no token or probability data.
An optional `draft_readout_bits` model setting of three or four builds a
coarse proposal head from the original affine vocabulary matrix. Each draft
step proposes 64 shortlist IDs and rescores them with the original head. The
full target vocabulary readout and exact rejection sampler are unchanged;
omission keeps the full draft readout. Coarse-head performance and prefix
behavior remain subject to artifact-bound qualification.
Flat verification batches of two to four rows use serial per-row
TreeVerify/StateAdvance arithmetic with batched projections; zero-draft
generation uses the same verification arithmetic. This arithmetic differs from
the ordinary stream's fused full-accept DeltaNet update and needs its own
checkpoint-specific qualification.
For the qualified BF16 128-dimensional, 16-key-head/48-value-head MXU geometry,
a Metal kernel fuses those per-row operations without changing their arithmetic.
Other supported layouts retain the serial verifier.

The desktop backend defaults MTP off for Gemma and full-BF16 Qwen3.5-0.8B,
following the measured speed comparison. Other supported models retain their
existing default on. It remembers an explicit on/off preference per model;
its API can inherit or override that preference. See
[the backend configuration contract](LOCAL_API.md#mtp-capabilities-and-per-model-preferences).
Terminal `--mtp` applies only to that terminal session and does not modify the
model library. Older exact-MTP workers without a recommendation remain
compatible: the Python host uses at most two drafts, bounded by their
advertised maximum. Missing or malformed capabilities keep MTP unavailable.

The native hybrid routes use complete F32 readouts and the existing temperature,
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
The official mixed-affine path batches up to four positions with separate
decode-order HC, MoE and vocabulary-head kernels, affine-64 cache writes and
ValueKey GDN recurrence/captures. Each row retains ordinary decode's
precision boundaries and reduction order. The serial verifier remains a
reference control. Uzu text conversions omit the head.

Unbounded nucleus requests can reuse a proved prefix for both draft and
target distributions. Retained IDs, probability terms, cumulative totals
and RNG consumption match the full-order sampler; uncertain boundaries
fall back to that reference. Top-k requests retain their existing route.
It is disabled by default.

### Streaming

With `stream: true`, the worker writes progress and delta lines while generating, then the normal terminal reply:

```json
{"event":"progress","input_tokens":12,"sampled_tokens":0,"output_token_budget":64,"elapsed_seconds":0.0,"prefill_seconds":null}
{"event":"progress","input_tokens":12,"sampled_tokens":1,"output_token_budget":64,"elapsed_seconds":0.2,"prefill_seconds":0.2}
{"event":"delta","text":"Hel"}
{"event":"delta","text":"lo"}
{"generated_token_ids":[...],"text":"Hello", ...}
```

Consume every progress event, delta and the terminal reply before sending another request. Progress events contain counts and timings only. With prefix reuse enabled, every progress event also carries the request's `reused_input_tokens`. `output_token_budget` is the smaller of the requested output limit and the remaining context capacity. `sampled_tokens` includes sampled EOS and stop tokens, even when they produce no text delta. `prefill_seconds` becomes available at the first sampled token; `elapsed_seconds` starts before native generation setup, after prompt preparation. Progress is provisional until the terminal reply succeeds. Nonstreaming replies are unchanged.

Progress frames are written as they arrive through the same bounded pipe. There
is no cumulative progress-byte cutoff that can stop later count/timing updates.

The concatenated deltas equal the terminal `text`. Incremental decoding buffers incomplete multi-byte characters, so a delta never splits a character; a final delta may be emitted after native cleanup to flush such a buffer. Streaming timings include decoding and output back-pressure.

The desktop app uses progress events, for Playground chats and [local API](LOCAL_API.md#progress-in-stark) requests, to show live rates, the first-token time and elapsed time against an estimated total. The backend estimates the time to the first token and a remaining-completion range; once calibrated, the app shows the middle of that range as the estimated total. Calibration uses bounded, in-memory timing and count history from successful comparable requests; it stores no conversation content. With no matching history, the first-token estimate is unavailable and live decode measurements can establish a completion range after generation starts. The range estimates an unknown response length within the output budget and can change as generation continues. It is not a deadline or a statistical coverage guarantee. Loading, queueing, prompt tokenization and CPU image decoding/preprocessing are outside the generation timing; vision-tower and decoder inference are included. Failed, cancelled and stale requests do not train the estimator.

Calibration profiles distinguish the effective MTP draft count. Turning MTP on
or off, or overriding it for an API request, uses the matching profile without
mixing measurements from the other mode.

### Stop sequences

`stop` strings are matched against the decoded text as it is produced, including across token boundaries. Text from the first matching sequence onward is withheld from both deltas and the terminal `text`, and `finish_reason` is `stop_sequence`. `generated_token_ids` and the token counts still include the token that completed the match, so decoding that vector yourself can reveal the withheld text. No extra token is generated after a match.

### Cancellation

If the environment variable `STARK_CANCEL_FD` names an inherited pipe descriptor (3 or higher, a FIFO), the worker reads it for cancellation frames: eight little-endian bytes holding the `request_id` to cancel. The worker checks the pipe before allocating state and around every token. Image requests also check before and after CPU preparation and after vision work has drained, before decoder work starts. A frame matching the running request stops generation after native cleanup and returns the recoverable error `generation_cancelled`; the model stays loaded. Frames for other IDs are discarded. End of file on the pipe cancels the running request and every later one. The worker removes `STARK_CANCEL_FD` from its environment before loading the model. Without the variable, the only way to interrupt a request is to terminate the process.

### Prefix reuse

Agent clients resend the same instructions, tool definitions and history on every step. With `--prefix-cache`, the worker can keep the state of the previous text generation and reuse the part that covers the prompt prefix it shares with the next request, so only the new part of the prompt is prefilled. Without the flag, behaviour is unchanged.

Eligibility and key:

- Converted text models whose owned layers are all full, non-ring attention caches (for example Qwen3 dense models) rewind their attention cache.
- Converted dense `qwen3_5` models with an admitted MTP head restore a snapshot on a 1,024-token grid. With drafts enabled it holds each DeltaNet layer's F32 recurrence and convolution history, the BF16 normalized target hidden state, F32 next-token logits and the predictor's committed position. Full-attention prefix rows in the target and predictor remain in the retained session. MTP0 instead retains the ordinary stream's target-only recurrence/convolution snapshot and attention prefix rows. Switching between ordinary and predictor sessions misses; the exact selected draft count and text graph are part of the reuse key. Draft-readout width is immutable loaded-model configuration. The packed 27B artifact and its separately configured three/four-bit draft-head variants passed artifact-bound prefix checks on the older build named in the [follow-up report](benchmarks/2026-11-06-qwen38-27b-packed-followup.md); that evidence does not qualify a changed driver.
- Native GGUF GLM-5.3 resumes from a snapshot on a 1,024-token grid. The snapshot includes KDA recurrence and convolution history, plus the MLA/indexer position and incomplete pooling tail. Completed attention and indexer prefix rows remain in the loaded model's own state.
- The ordinary [EXL3/FP8 GLM backend](validation/GLM_EXL3_BACKEND.md) resumes on a 16,384-token grid with MTP off and its head unloaded. A private native MTP candidate, enabled through a separately prepared capability stage and explicit `STARK_GLM_EXL3_VALIDATE_NATIVE=1`, snapshots the draft-head state with the native prefix state. Its native greedy, sampled and prefix mechanism/invalidation checks pass; full Rust-worker and quality validation remain pending, so production MTP is unqualified. The state-hash diagnostic build setting is separate. See the [native mechanism record](archive/runs/2026-11-04-glm-exl3-native-mtp-prefix.md) and [comparison record](benchmarks/2026-11-04-glm-mlx-stream-comparison.md).
- The native mixed-precision Qwen3.8 backend (Lily-format, external mixed-affine and Uzu-converted Qwen3.8 artifacts) resumes from a recurrent-state snapshot on a 4,096-token grid. This includes the admitted `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` layout. Its attention, indexer and draft-head caches below the snapshot stay in the loaded model's own state.
- Ordinary text requests on native Qwen vision artifacts can resume from an exact decoder snapshot on the 1,024-token grid when every owned mixer supports boundary restoration. The snapshot includes DeltaNet recurrence and convolution history; completed full-attention rows stay in the retained state. Image operations clear this entry and never populate it. The host requests caching for vision workers and uses loaded readiness to determine support.
- Ordinary text on Gemma 4 image artifacts can restore the last complete non-final 1,024-token boundary. Each owning sliding-attention layer retains independent physical K/V rows and its ring offset, length and source traversal origin; shared-KV consumers reuse that owner's restored state. Images clear the entry and never populate it. Exact checkpoint qualification is recorded in the [Gemma 4 attempt](model-experiments/2026-11-09-gemma4-image-text-parity.md); larger component evidence is not full-model cache qualification.
- Gemma 3 image models implement the same owned ring snapshot capture/restore and suffix-only text prefill. Actual-artifact qualification is pending in the [continuation attempt](model-experiments/2026-11-09-gemma-next-image-parity.md); no other checkpoint inherits the Gemma 4 evidence.
- Readiness reports `prefix_cache.supported: false` and keeps `fresh-per-request` for other recurrent, sliding-window text exports and speculator models, and the native Flint and F32 Qwen graphs.
- A request reuses state only when it carries a `cache_scope`, the scope and `context_tokens` equal those of the retained entry, and the loaded model instance is the same. On Qwen3.8 and the private native EXL3 candidate, whether MTP is enabled must also match (`key` reports this). Dense MTP uses the exact selected draft count; model-level coarse-readout configuration is bound by the loaded model instance.

The F32 `qwen4_exp` route used by the PipeNetwork `qwen38-mixed` catalog
artifact has no boundary snapshot implementation. Exact restoration needs to
save the Delta recurrence and convolution history, QSA attention/indexer state,
and PLE/ngram history while retaining completed attention rows. Its current
`Forward::Qwen` dispatch has no capture/restore hooks for these owners, so prefix
reuse stays disabled. The separate mixed-affine Lily artifact's qualification
does not cover this route.

Exactness, converted models:

- Prompts are prefilled in 1,024-token chunks from position zero. The worker retains only rows produced by full, non-final prefill chunks of the previous prompt. It reuses the longest common token prefix rounded down to a chunk boundary and kept below the new prompt's own final chunk.
- The rest of the prompt is then prefilled with the same chunk boundaries, kernels and arithmetic as on a fresh state. Rows from the previous final chunk and from decoding are never reused, and the sampler is reseeded from the request.

Exactness, native hybrid snapshots:

- GGUF GLM captures at the start of the final 1,024-token reuse chunk. That boundary is divisible by its admitted 4/8-token prefill group widths, so splitting for capture preserves their grouping and arithmetic. After all GPU writers finish, capture copies the exact stored KDA state and convolution histories, and the MLA/indexer cursor and incomplete pooling tail. Restore resets the model, restores those values and exposes only the completed prefix rows retained below the boundary.
- Qwen3.8 prefill runs in 4,096-token passes counted from the position it starts at. A scoped request splits its prefill at the start of its final pass and copies the recurrent state there: GDN states and convolution windows, the n-gram history and window, and, when present, the draft head's last hidden. Both parts run the same passes as one prefill.
- Dense Qwen3.5-family MTP prefill captures after the last complete non-final 1,024-token chunk. Restore copies the recurrence, convolution history and normalized target hidden state, rewinds both attention cursors, then prefills the remaining suffix with the fresh route's chunk boundaries. The snapshot belongs to one loaded model instance and one retained session.
- Qwen vision text requests capture after the last complete non-final 1,024-token prefill chunk, in the same ordered command after decoder writes. Restoration checks loaded identity, execution graph, capacity, text-only state and saved position, restores all owned mixer snapshots and reseeds the sampler. Repeated prompts keep an unchanged restored snapshot; longer prompts replace it at their new eligible boundary. An image-prefilled or failed state is rejected. BF16 dense, affine dense and affine MoE ordinary text select the shared-weight Legacy graph; image requests select the reference graph and clear retained prefixes. Admitted complete dense heads use their separate MTP-aware snapshot path and key the selected draft count. BF16-head and other artifact qualification remains separate; see the [remaining Qwen attempt](archive/runs/2026-11-09-qwen-remaining-image-text-parity.md).
- A snapshot cannot be rewound. The next request reuses it only when its prompt begins with the whole snapshotted prefix and leaves at least two tokens to prefill; otherwise it reuses nothing. The remaining prompt uses the backend's original prefill grouping, final-token route and draw.

Restored execution must produce greedy and seeded outputs identical to a fresh-state run. The [validation record](archive/runs/2026-11-01-generation-prefix-reuse.md) checks this on the pinned Qwen3-4B artifact, and the [Qwen3.8 record](archive/runs/2026-11-02-hybrid-prefix-reuse.md) on the native Lily-format Qwen3.8 checkpoint. The [GLM and external mixed-affine record](archive/runs/2026-11-03-glm-qwen-prefix-reuse.md) records the artifact-specific scope for those paths.

The [vision text-prefix record](archive/runs/2026-11-08-vision-text-prefix-reuse.md)
qualifies the official Qwen3.5-0.8B BF16 schema-2 vision artifact at context 8,192,
including seeded text outputs and completed-image invalidation. Other vision
artifacts do not inherit its numerical qualification. The
[paired timing](benchmarks/2026-11-08-qwen35-vision-text-prefix.md) reports the
cache-cold cost and bounded follow-up latency observations separately.

Lifetime:

- There is one entry. It is published only after the terminal reply has been written and flushed.
- A request that cannot reuse it drops it before allocating new state, so two request states never coexist.
- The entry is dropped on:
  - any error reply, including invalid requests, `context_overflow`, cancellation and oversized lines
  - a different scope or context size, or a `generate` without `cache_scope`
  - any image operation or `reset_cache`
  - 600 seconds without a request
  - process exit
- `tokenize` and `decode` keep it.

Converted text models using Metal4 and native mixed-precision Qwen models
maintain their existing command queue every 500 ms between complete requests.
This owner-thread maintenance preserves retained state and the original
600-second deadline; ticks cannot renew it. Queue/input failures clear the
entry and exit for reload. Vision wrappers and other native hybrid backends
retain their existing idle scheduling. The [cross-backend wake study](benchmarks/2026-11-08-cross-backend-idle-wake.md)
records artifact-specific latency and restoration checks.

Memory: on converted models the entry holds the KV cache of the previous request. On Qwen3-4B that is 144 KiB per token, about 2.7 GiB after an 18,500-token step, bounded by the state's capacity of `context_tokens` plus 1,024 rows. A request of that size already maps this memory while it runs; reuse keeps it resident until the next request or idle expiry. Native GLM and Qwen3.8 keep their completed attention caches in the loaded model; retention adds the snapshot of mutable recurrence and history. GLM also preserves the incomplete indexer pool. The Lily-format Qwen3.8 snapshot is about 110 MiB (36 GDN layers of 48 × 128 × 128 F32 state plus convolution windows); other graph geometries have their own memory cost.

Privacy: the retained state and its prefix token IDs exist only in worker memory. They are never written to disk, logged, or included in replies; replies carry counts only. A faster first token reveals that the previous request shared a prefix, so hosts give each principal its own random scope. The desktop backend uses one scope per Playground conversation, per API session and master key, and per managed client grant. It rotates scopes on reset, revocation and key change, and sends `reset_cache` on revocation and API restart.

### Reset cache

```json
{"operation":"reset_cache"}
```

Drops the retained prefix. The reply is `{"cache_reset":true}`. Available only when prefix reuse is enabled; otherwise the reply is `{"error":"invalid_request"}`.

### Error codes

Errors are replies of the form `{"error":CODE}`. Codes are fixed strings; see [error classes](WORKER_PROTOCOLS.md#error-classes) for the behaviour of each class.

| Class | Codes |
| --- | --- |
| Recoverable (worker continues) | `invalid_request` (unparseable JSON, unknown operation or field, wrong type), `invalid_generation_request`, `invalid_sampling_request`, `invalid_decoding_request`, `tokenization_failed`, `context_overflow` (context above the model maximum, or input and requested output exceed `context_tokens`), `resource_budget_exceeded` (content, token storage or native allocation admission exceeds the resource allowance), `request_too_large` (line exceeds the configured frame budget; the rest of the line is discarded), `generation_cancelled`, `mtp_not_supported`, `images_not_supported`, `invalid_image_reference`, `image_read_failed`, `invalid_image_request`, `invalid_prepared_image`, `image_request_setup_failed` |
| Fencing (error reply, then `{"ready":false,"error":"native_operation_failed"}` and exit 1) | `native_generation_failed`, `invalid_generation_output`, `invalid_tokenization_output`, `invalid_image_preparation_output`, `decoding_failed`, `cancel_channel_failed` |
| Startup (`{"ready":false,"error":CODE}` and exit 1) | `invalid_arguments`, `invalid_resource_budget`, `output_setup_failed`, `invalid_cancel_channel`, `invalid_vision_staging_root`, `model_config_unavailable`, `invalid_model_config`, `speculator_not_supported`, `metal_initialization_failed`, `model_load_failed`, `native_backend_unavailable` (a build without the `metal` feature), `native_operation_failed` (panic during startup) |

A write failure on the reply pipe also ends the process with exit status 1.

## State and privacy

By default every generate request creates a fresh native state sized to `context_tokens`, runs the prompt through prefill, generates, and drops the state before the terminal reply is written. No KV cache or prompt prefix survives between requests, so each turn of a conversation re-processes the whole conversation. This is the `fresh-per-request` policy advertised in readiness.

With `--prefix-cache` on an eligible model, the policy is `consecutive-prefix-v1`: the state of the previous successful text generation (on GLM-5.3 and Qwen3.8, its recurrent-state snapshot) is kept in memory for the next request under the key, lifetime and privacy rules in [Prefix reuse](#prefix-reuse). Nothing is ever persisted.

The worker suppresses native diagnostics and capture variables as described in [worker protocols](WORKER_PROTOCOLS.md#privacy-and-diagnostics). It opens no network listener and executes no tools.

## Validation

Model-free tests cover request parsing and bounds, stop handling, UTF-8 decoding, stream cleanup, fresh state per request, seeded option dispatch, sanitized startup errors and failure fencing:

```sh
cargo test --locked -p stark-generate --no-default-features
cargo test --release --locked -p stark-generate -- --test-threads=1
uv run --no-project --python 3.12.13 python scripts/test_generation_process.py --binary target/release/stark-generate
```

The model-free tests also cover prefix-reuse alignment, backend reuse grids, whole-prefix snapshot matching, key matching, publication after the reply is flushed and every invalidation path. A model-bound script compares a `--prefix-cache` worker with fresh-state workers on synthetic agent sessions rendered by the model's chat template. It compares generated token IDs, text and parsed tool calls (when the template has a qualified tool parser), checks every reuse count against the documented rule, checks each invalidation path and measures first-token time. Workers run one at a time, so models that fill most of unified memory can be checked. It writes aggregates and hashes only. Add `--reuse boundary` for the native GLM-5.3 and Qwen3.8 backends. For text on a Qwen vision artifact, add both `--vision --reuse boundary`; the validator creates private staging and checks invalidation after a completed image request. Follow the [model onboarding workflow](MODEL_ONBOARDING.md) for every new model or state-affecting path:

```sh
uv run --project tools/converter --locked --python 3.13.13 python scripts/validate_generation_prefix_cache.py --model MODEL_DIR --baseline BASELINE_WORKER --candidate CANDIDATE_WORKER --output runtime/prefix-cache-validation.json
```

An ignored engine test checks at the model level that capturing a snapshot leaves the prefill unchanged and that a resumed prefix matches a fresh prefill bit for bit:

```sh
STARK_LILY_TEST_MODEL_PATH=/absolute/path/to/qwen38 cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal -- --ignored private_checkpoint_prefix_capture_and_restore_match_fresh_prefill
```

The model-bound acceptance record for the `lightonai/colgrep-agent-2B` checkpoint, including how to reproduce it, is in [ColGrep generation validation](validation/GENERATION_COLGREP.md).
