# Text generation

`stark-generate` is a standalone Rust/Metal inference worker for converted language models. It serializes complete requests against one loaded model. The app owns chat-template rendering, repository indexing, search, tool execution and the agent loop.

## Build and prepare ColGrep-Agent

Use an Apple silicon Mac, Rust 1.98.1 and Xcode's Metal tools. Build the default Metal package:

```sh
cargo build --release --locked -p stark-generate
```

This package uses its default generation build; the scoring worker's `optimized` feature bundle is outside this build.

Convert the pinned checkpoint into a new output folder:

```sh
uv run --project tools/converter --locked stark-convert lightonai/colgrep-agent-2B \
  --revision f798a61129c6decb5166864e0a3c98373e62c764 \
  --metadata-profile hf-generic \
  --write-source-manifest runtime/converter-sources/colgrep-agent-2B.json \
  --output models/colgrep-agent-2B-stark
./target/release/stark-generate --model-path /absolute/path/to/models/colgrep-agent-2B-stark
```

Model files, source downloads and private IPC artifacts stay outside Git. Preserve the conversion receipt and immutable source revision. Generic conversion readiness and generation acceptance have separate evidence.

For this checkpoint, the app must render the exact Hugging Face/upstream MiniCPM5 XML chat template with the upstream tools and `enable_thinking=False`. The validation harness pins upstream agent assets at revision `c6c091d7ee14c81206bb28af90d8ff506fd0235e` and checks their hashes. Pass the complete rendered string to the worker, including the generation prefix. The worker tokenizes that string with `add_special_tokens=false` and performs inference.

## Private JSON-lines protocol

Start the worker with exactly `--model-path PATH`. Read its readiness reply before sending requests. Readiness includes `ready`, `engine`, `version`, `mode` (`text-generation-v1`), supported `operations`, `state_policy` (`fresh-per-request`), `vocab_size`, context/output bounds, `eos_token_ids` and `sampling_modes`.

Send one JSON object per line and read one reply per request. Each request line is limited to 1 MiB. Unknown fields, wrong types and invalid IDs are rejected with fixed error codes. Malformed and oversized lines are consumed independently so a subsequent valid request can proceed. A native failure fences the process; the caller must observe exit and restart.

### Tokenize and decode

```json
{"operation":"tokenize","text":"rendered text","add_special_tokens":false}
```

The reply contains `token_ids` and `token_count`. `add_special_tokens` defaults to `false`.

```json
{"operation":"decode","token_ids":[3,4],"skip_special_tokens":false}
```

The reply contains `text` and `token_count`. `skip_special_tokens` defaults to `false`. Empty vectors are valid. Decoding uses the complete vector to preserve UTF-8 boundaries.

### Generate

```json
{"operation":"generate","rendered_prompt":"complete rendered chat template","context_tokens":16384,"max_new_tokens":2048,"seed":0,"sampling":{"mode":"greedy"},"stop_token_ids":[]}
```

| Field | Contract |
| --- | --- |
| `rendered_prompt` | String; supply exactly one of this field or `input_token_ids` |
| `input_token_ids` | Nonempty array of unsigned 32-bit vocabulary IDs; mutually exclusive with `rendered_prompt` |
| `context_tokens` | Required integer from 1 through 16,384, also bounded by the model maximum; the input must leave at least one output slot |
| `max_new_tokens` | Required integer from 1 through 4,096 |
| `seed` | Optional unsigned 64-bit integer, defaults to 0 |
| `sampling` | Optional sampling object, defaults to `{"mode":"greedy"}` |
| `stop_token_ids` | Optional array of at most 256 distinct valid vocabulary IDs, defaults to empty; adds stops to model EOS metadata |

The ColGrep checkpoint declares both EOS IDs, `1` and `130073`; both are honored automatically. Caller stops supplement model stops. A terminal stop is excluded from `generated_token_ids` and `text`.

The reply contains:

| Field | Meaning |
| --- | --- |
| `generated_token_ids` | Generated IDs excluding a terminal EOS or caller stop |
| `text` | Full-vector decoding of those IDs with special tokens preserved |
| `input_tokens` | Number of input IDs actually used |
| `output_tokens` | Length of `generated_token_ids` |
| `sampled_tokens` | Returned output IDs plus a consumed terminal stop, when present |
| `finish_reason` | `eos`, `stop_token`, `length` or `context` |
| `stop_token_id` | Terminal stop ID, otherwise `null` |
| `generation_seconds` | Generation, cleanup and decoding duration in seconds |

`length` means the requested output limit was reached; `context` means fewer output slots were available than requested. Returned prompt-plus-sampled counts stay within `context_tokens`. There is no silent input truncation.

## Sampling semantics

Greedy decoding is the default and the intended reference-qualification path. Stochastic requests explicitly provide temperature and optional filters:

```json
{"mode":"stochastic","temperature":0.6,"top_k":20,"top_p":0.95,"min_p":0.0}
```

Temperature must be finite, positive and at most 100. `top_k`, when present, is from 1 through the vocabulary size. `top_p` is finite and greater than 0 through 1; `min_p` is finite from 0 through 1. Filters omitted from the request are disabled. Repetition, frequency and presence penalties are outside this interface.

The inherited native sampler applies temperature before filtering and evaluates top-k and top-p against the same full-vocabulary distribution. Its combined filters differ from sequential top-k then top-p sampling with renormalization, and its temperature ordering can differ from a llama.cpp sampler chain. Seeded repeat checks establish reproducibility on the named native build; they do not establish stochastic distribution or token parity with another backend.

## State, cleanup and cancellation

Every generation request creates an empty state with the supplied context and seed. The worker retains no KV cache between requests. The native stream can schedule lookahead; dropping it waits for pending work. The stream and state are dropped before decoded text is returned, including on normal stops and native errors.

Native stdout/stderr diagnostics, panic details and both `STARK_CAPTURE_*` and historical `UZU_CAPTURE_*` environment settings are suppressed before model threads start. Protocol requests and replies contain private content; keep them in private pipes or protected files. The worker exposes no network listener and executes no shell or repository tools.

For cancellation, terminate and reap the worker process, then start a fresh worker. The interface does not expose an in-process cancellation operation. The app must own the process and handle EOF, timeouts, fixed errors and unexpected exit.

## Validation status and replay

Native ColGrep reference and controlled tool-protocol acceptance are pending. Model-free tests and process smoke checks cover protocol parsing, bounds, stop handling, UTF-8 decoding, stream cleanup, serial fresh state, seeded option dispatch, sanitized startup errors and failure fencing. These checks do not qualify the checkpoint's generated content.

```sh
cargo test --offline --locked -p stark-generate --lib --no-default-features
cargo test --release --offline --locked -p stark-generate
uv run --project tools --locked python scripts/test_generation_process.py \
  --binary target/release/stark-generate
```

The native replay requires the complete pinned Hugging Face source snapshot, the converted model and an upstream agent checkout at the revision above:

```sh
uv run --project tools --locked python scripts/validate_colgrep_generation.py \
  --source runtime/colgrep-agent-2B-source \
  --model models/colgrep-agent-2B-stark \
  --worker target/release/stark-generate \
  --upstream runtime/colgrep-agent-upstream \
  --device mps \
  --report provenance/COLGREP_GENERATION_VALIDATION.json
```

The harness compares BF16 greedy generation with a named PyTorch/Transformers reference and checks exact tokenization, decoded text, stops, XML calls and file/line-range output. Its tool observations are controlled fixtures, and its parser covers a controlled XML subset. It does not run a live ColGrep index or qualify the app's search/read/finish loop.

The report records aggregate results and hashes for the checkpoint, converted model, worker, dependency locks, validator and upstream assets. Keep prompts, generated text, token vectors and model weights private. Scoring acceptance remains governed by [its separate validation record](VALIDATION.md).
