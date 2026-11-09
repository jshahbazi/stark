# Probability scoring

`stark-score` is the native scoring worker. It loads one converted language model and, for a given token sequence, returns the next-token logits and probabilities for a small set of candidate tokens. It never samples. Conventions shared with the generation worker (framing, readiness, error classes, privacy) are in [worker protocols](WORKER_PROTOCOLS.md).

Build the worker with the `optimized` feature, which enables the scoring-only engine optimizations ([feature flags](TECHNICAL_OVERVIEW.md#engine-feature-flags)):

```sh
cargo build --release --locked -p stark-score --features optimized
./target/release/stark-score --score-model-path /absolute/path/to/converted-model
```

Installed bundles ship this build as `stark score`.

## Arguments

```text
stark-score --score-model-path PATH [--consecutive-cache]
```

`PATH` is a converted model directory. Models that contain a `speculator` directory are refused. `--consecutive-cache` enables the [consecutive cache](#consecutive-cache); it is off by default.

## Readiness

```json
{"ready":true,"engine":"stark","version":"0.1.0","mode":"exact-token-scoring-cache-v1","upstream_version":"0.6.1","upstream_revision":"8d01581474760bd1fc7f4a547ff1d02f7dada437","scoring_patch_sha256":"...","cache_policy":"fresh-state","cache_enabled":false,"cache_key":"exact-token-ids+context-tokens+model-instance","cache":{"enabled":false,"hits":0,"native_score_calls":0},"load_seconds":1.9,"vocab_size":151936,"max_context_tokens":262144,"peak_memory_bytes":1234567890}
```

| Field | Meaning |
| --- | --- |
| `mode` | `exact-token-scoring-cache-v1` for `stark-score`, `exact-token-digest-cache-v1` for the digest probe |
| `upstream_version`, `upstream_revision` | The Uzu release and commit the engine derives from |
| `scoring_patch_sha256` | Identity of the original scoring patch, recorded at build time |
| `cache_policy` | `consecutive-full-vocabulary-logits-v1` when the cache is enabled, otherwise `fresh-state` |
| `cache_enabled`, `cache_key`, `cache` | Cache state and counters; `cache` is repeated on every reply |
| `load_seconds` | Model load time |
| `vocab_size` | Upper bound for every token ID |
| `max_context_tokens` | Loaded model's context maximum, or `null` when it has no finite maximum |
| `request_budget_bytes`, `max_request_bytes` | Configured logical content budget and encoded request-line budget; see [resource budget](WORKER_PROTOCOLS.md#resource-budget) |
| `peak_memory_bytes` | Engine peak memory, or `null` when unavailable |

## Requests

### Tokenize

```json
{"operation":"tokenize","text":"rendered text","add_special_tokens":false}
```

`text` must be non-empty. The reply contains `token_ids`, `token_count`, `token_ids_sha256` (SHA-256 over the IDs as little-endian 32-bit integers), `vocab_size`, `tokenization_seconds` and `cache`.

### Score

```json
{"operation":"score","input_token_ids":[1,2,3],"candidate_token_ids":[10,11],"context_tokens":8192,"temperature":1.0}
```

| Field | Rule |
| --- | --- |
| `input_token_ids` | Non-empty array of token IDs below `vocab_size`; the sequence to score the next token after |
| `candidate_token_ids` | 2 through 16 distinct token IDs below `vocab_size`; order is preserved in the reply |
| `context_tokens` | Positive 32-bit integer at most the model maximum reported in readiness; `input_token_ids` may be at most this long |
| `temperature` | Finite and positive; the calibration temperature applied to the candidate distribution |

The worker applies the loaded model's configured context capacity. It has no separate fixed token ceiling. The [automatic shared resource budget](WORKER_PROTOCOLS.md#resource-budget) applies to request content, four-byte-per-ID token storage and encoded lines. Available RAM at startup selects a logical budget with a 16 MiB floor and no fixed content ceiling; encoded request lines allow six times that content budget plus 2 MiB for JSON escaping and framing. `STARK_REQUEST_BUDGET_MIB` is a developer/operator override accepting positive integers whose derived byte allowances fit the platform's allocation address space. Resource admission does not guarantee every encoding of a full token window or establish full-context accuracy or performance.

Before a fresh native score, the worker also admits the loaded model's state and
workspace estimate against one quarter of available RAM after a 256 MiB reserve.
Generic scoring retains independent scratch pools for every 1,024-token prompt
chunk until completion, so its peak estimate counts all those pools. Sparse
finite state counts the actual input span; dense and undeclared state count the
selected allocation. Hybrid families use their loaded cache/workspace profiles.
The prompt may exactly fill the context because scoring reserves no output token.
An insufficient-memory rejection clears cached logits and leaves the worker
available for a smaller retry. Admission is an estimate, not a memory reservation
or hard RSS ceiling.

The reply:

| Field | Meaning |
| --- | --- |
| `logits` | Raw next-token logits for each candidate, in request order |
| `log_probabilities` | `logit - logsumexp(all vocabulary logits)` for each candidate |
| `candidate_probabilities` | Softmax over the candidate logits divided by `temperature`; sums to 1 over the candidates |
| `candidate_token_ids`, `temperature` | Echoed from the request |
| `input_tokens`, `prompt_tokens_sha256` | Length and hash of the input sequence |
| `cache_hit` | Whether the logits came from the consecutive cache |
| `truncated` | Always `false`; inputs are never truncated |
| `metrics` | `state_creation_seconds`, `prefill_seconds`, `readout_seconds`, `score_seconds` (native call plus readout) and `total_seconds` (whole request); the native stages are `0` on a cache hit |
| `peak_memory_bytes`, `cache` | As in readiness |

Logits are read from the engine as BF16 and widened, then all arithmetic is done in f64. `log_probabilities` normalise over the complete vocabulary; `candidate_probabilities` normalise over the candidates only. The decision rule that consumes these values belongs to the caller.

### Reset cache

```json
{"operation":"reset_cache"}
```

Clears the consecutive cache. The reply is `{"cache_reset":true,"cache":{...}}`. Hit and call counters are cumulative for the process and are not reset.

## Consecutive cache

With `--consecutive-cache`, the worker keeps the full-vocabulary logits of the most recent successful score request. The next score request reuses them, skipping native execution, only when its `input_token_ids` and `context_tokens` are identical. Candidates and temperature may differ because normalisation and calibration are recomputed from the stored logits. Only the immediately previous request can hit: sequences A, B, A score A twice. Any error, an explicit reset, or process exit discards the entry; a successful tokenize request preserves it. An entry is published only after its reply has been written and flushed, so a reply the caller never received cannot produce a later hit. Nothing is written to disk.

## Error codes

Errors are replies of the form `{"error":CODE,"cache":{...}}`. Every error discards the cached entry. See [error classes](WORKER_PROTOCOLS.md#error-classes).

| Class | Codes |
| --- | --- |
| Recoverable (worker continues) | `invalid_request` (unparseable JSON, unknown operation or field, wrong type), `invalid_scoring_request` (bounds or duplicate candidates), `context_overflow` (context above the model maximum, or input longer than `context_tokens`), `resource_budget_exceeded` (content, token storage or native memory admission exceeds the resource allowance), `invalid_tokenization_request` (empty text), `tokenization_failed` |
| Fencing (error reply, then `{"ready":false,"error":"native_operation_failed"}` and exit 1) | `native_scoring_failed`, `invalid_score_values` (non-finite or wrong-sized logits) |
| Terminating (error reply, then exit 0) | `request_too_large` (line exceeds the configured encoded-frame budget) |
| Startup (`{"ready":false,"error":CODE}` and exit 1) | `invalid_arguments`, `invalid_resource_budget`, `output_setup_failed`, `model_config_unavailable`, `invalid_model_config`, `speculator_not_supported`, `metal_initialization_failed`, `model_load_failed`, `native_operation_failed` (panic during startup) |

## Digest probe

`stark-score-digest` is built alongside the worker from the same crate. It serves the same protocol, including the cache, but each score reply carries `vocab_logits_sha256`, a SHA-256 over the complete raw f32 vocabulary logits, instead of candidate values. It exists for parity testing between builds; see [scoring parity validation](validation/SCORING_PARITY.md).

## State and privacy

Every score request creates a fresh native state, runs the prompt in batches of up to 1,024 tokens, reads the logits at the final position, and drops the state. The worker suppresses native diagnostics and capture variables as described in [worker protocols](WORKER_PROTOCOLS.md#privacy-and-diagnostics) and opens no network listener.

For native mixed-precision Qwen models, both workers maintain the Metal queue
every 500 ms while waiting for a new complete request. Maintenance runs on the
request owner thread, changes no model state or cached logits, and leaves hit
and native-call counters unchanged. Buffered input and EOF bypass the wait.
Input or maintenance failure discards the cache and exits with a static fatal
error so the host can reload the worker. Dense scoring retains its existing
blocking input wait. The [cross-backend wake study](benchmarks/2026-11-08-cross-backend-idle-wake.md)
records the qualified artifact and cache/parity controls.

## Validation

```sh
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
```

The model-bound parity record against frozen Uzu references is in [scoring parity validation](validation/SCORING_PARITY.md).
