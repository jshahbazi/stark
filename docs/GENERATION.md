# Text generation

`stark-generate` is the native text generation worker. It loads one converted language model and serves tokenize, decode and generate requests over a private JSON-lines protocol. The calling application renders the chat template, owns tools and the agent loop, and sends either the rendered prompt or token IDs. Conventions shared with the scoring worker (framing, readiness, error classes, privacy) are in [worker protocols](WORKER_PROTOCOLS.md).

## Interactive CLI chat

The installed bundle includes a terminal chat client built on the worker:

```sh
stark chat --model-path /absolute/path/to/converted-model
```

From a source checkout, build the worker first and use the launcher, which runs the client through the locked converter environment:

```sh
cargo build --release --locked -p stark-generate
./scripts/stark chat --model-path /absolute/path/to/converted-model
```

The client renders the converted model's chat template in a Jinja sandbox, keeps the conversation in memory, and sends the complete conversation to one worker for each turn. Replies appear when generation completes; the client does not stream. Type one message per line. `/help` lists commands, `/reset` clears the history, and `/exit`, `/quit`, Ctrl-D or Ctrl-C quit. Nothing is written to disk.

| Option | Default | Meaning |
| --- | --- | --- |
| `--system TEXT` | model metadata | System message; the template decides how it is rendered |
| `--context-tokens N` | smaller of 8192 and the worker limit | Context budget for prompt plus reply |
| `--max-new-tokens N` | 512 | Reply budget |
| `--thinking` | off | Enable reasoning for templates that support `enable_thinking` |
| `--timeout SECONDS` | 300 | Maximum wait for each worker reply, including startup |

Generation is greedy. A turn whose rendered prompt plus reply budget exceeds the context budget is refused without changing the history; use `/reset` or a larger `--context-tokens`. The menu bar app exposes the same settings through **Advanced Options** ([menu bar guide](../apps/menu-bar/README.md#models-and-requests)).

## Private JSON-lines protocol

Start the worker with exactly `--model-path PATH`, where `PATH` is a converted model directory. Models that contain a `speculator` directory are refused.

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
| `stop_token_ids` | Optional, at most 256 distinct token IDs below `vocab_size`. Added to the model's EOS IDs; a model EOS cannot be removed. |
| `stop` | Optional, at most four non-empty strings of at most 1,024 characters each. See [Stop sequences](#stop-sequences). |
| `stream` | Optional boolean, default `false`. See [Streaming](#streaming). |
| `request_id` | Optional unsigned 64-bit integer, default 0. See [Cancellation](#cancellation). |

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

The prefill figure is a wall-clock proxy that includes native state allocation, command encoding, GPU waiting and any lookahead decode the engine schedules before returning its first token. The output rate counts sampling steps after the first token, including a terminal stop token. Prompt tokenization is excluded from all timings. The prompt plus sampled tokens never exceed `context_tokens`; there is no silent truncation.

### Sampling

Greedy decoding is the default. Stochastic sampling supplies a temperature and optional filters:

```json
{"mode":"stochastic","temperature":0.6,"top_k":20,"top_p":0.95,"min_p":0.0}
```

`temperature` must be finite, positive and at most 100. `top_k` is 1 through `vocab_size`. `top_p` is greater than 0 and at most 1. `min_p` is 0 through 1. Omitted filters are disabled. Repetition, frequency and presence penalties are not part of this interface.

The sampler applies temperature before filtering and evaluates top-k and top-p against the same full-vocabulary distribution, so its output differs from samplers that apply top-k and then renormalise before top-p. Seeded requests are reproducible on the same build and hardware; they are not expected to match another engine token for token.

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

If the environment variable `STARK_CANCEL_FD` names an inherited pipe descriptor (3 or higher, a FIFO), the worker reads it for cancellation frames: eight little-endian bytes holding the `request_id` to cancel. The worker checks the pipe before allocating state and around every token. A frame matching the running request stops generation after native cleanup and returns the recoverable error `generation_cancelled`; the model stays loaded. Frames for other IDs are discarded. End of file on the pipe cancels the running request and every later one. The worker removes `STARK_CANCEL_FD` from its environment before loading the model. Without the variable, the only way to interrupt a request is to terminate the process.

### Error codes

Errors are replies of the form `{"error":CODE}`. Codes are fixed strings; see [error classes](WORKER_PROTOCOLS.md#error-classes) for the behaviour of each class.

| Class | Codes |
| --- | --- |
| Recoverable (worker continues) | `invalid_request` (unparseable JSON, unknown operation or field, wrong type), `invalid_generation_request`, `invalid_sampling_request`, `invalid_decoding_request`, `tokenization_failed`, `context_overflow` (context above the model maximum, or input not shorter than `context_tokens`), `request_too_large` (line over 1 MiB; the rest of the line is discarded), `generation_cancelled` |
| Fencing (error reply, then `{"ready":false,"error":"native_operation_failed"}` and exit 1) | `native_generation_failed`, `invalid_generation_output`, `invalid_tokenization_output`, `decoding_failed`, `cancel_channel_failed` |
| Startup (`{"ready":false,"error":CODE}` and exit 1) | `invalid_arguments`, `output_setup_failed`, `invalid_cancel_channel`, `model_config_unavailable`, `invalid_model_config`, `speculator_not_supported`, `metal_initialization_failed`, `model_load_failed`, `native_backend_unavailable` (a build without the `metal` feature), `native_operation_failed` (panic during startup) |

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
