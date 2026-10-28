# Technical guide

How Stark is put together: the crates, the Python package, the app, how they talk to each other, and the engine's feature flags. For protocol contracts see [worker protocols](WORKER_PROTOCOLS.md); for measured results see [benchmarks](benchmarks/README.md).

## Components

```text
apps/menu-bar/            SwiftUI menu bar app (swiftc build, no SwiftPM)
  |  JSON lines over the app's private pipe
tools/converter/stark_convert/
  desktop.py              app backend: model library, worker lifecycle, loopback API
  local_api.py            HTTP server for /v1/chat/completions
  api_request.py          request validation and normalisation
  chat.py                 terminal chat client and the Worker pipe wrapper
  decision.py             Decision-mode scoring client
  cli.py, adapters/, ...  the converter
  |  JSON lines over stdin/stdout
crates/stark-generate/    text generation worker
crates/stark-score/       next-token scoring worker (+ stark-score-digest probe)
crates/stark-engine/      Rust/Metal inference engine
crates/stark-engine-macros/
```

Everything a user runs is one of: `Stark.app`, the `stark` shell launcher (`scripts/stark`), or a worker binary. The app and the launcher both start `desktop.py` or `chat.py` from a private Python 3.13 runtime bundled with the installation, and those start the worker binaries from `libexec/`.

## Engine

`crates/stark-engine` is the Uzu 0.6.1 engine, renamed, with Stark's scoring API and optimizations added. Its module layout:

| Path | Contents |
| --- | --- |
| `src/engine/` | `Engine` (Metal context, capture manager), `LanguageModel` loading, `language_model/stream/` (generation: prefill, decode loop, sampling, lookahead), `language_model/scoring.rs` (next-token readout), `language_model/state.rs` (per-request state), `classifier_model.rs` (loads classifiers produced by the converter's ModernBERT path; no worker runs one) |
| `src/encodable_block/` | The model graph as encodable blocks: `decoder.rs`, `transformer.rs`, `transformer_layer.rs`, `mixer/attention/`, `mixer/delta_net.rs`, `mixer/mamba2.rs`, `mlp/`, `linear/`, `sampling/`, embeddings and normalisation |
| `src/backends/metal/` | Metal context, command buffers, buffer allocation and retirement, kernel wrappers under `kernel/` (matmul GEMM and GEMV, attention, GDN, sampling, normalization); `.metal` sources live next to their Rust wrappers |
| `src/backends/cpu/` | CPU reference kernels used by the unit tests (`cpu` feature); not used by the workers |
| `src/backends/common/` | Backend traits shared by both |
| `src/config/` | Model configuration types read from `config.json`, with validation |
| `src/parameters/` | Safetensors header parsing and bounded tensor loading |
| `build/` | The build script: scans GPU type definitions, generates Rust bindings for each kernel, and compiles `.metal` sources with Xcode's Metal toolchain into `.metallib` files embedded in the binary (`build/metal/`), or compiles CPU kernels (`build/cpu/`) |
| `unit/` | The engine unit and benchmark suite; see [Contributing](../CONTRIBUTING.md#engine-tests) |

A generation request creates a `LanguageModelState` sized to the requested context, runs prefill in batches of up to 1,024 tokens, then decodes one token per command buffer with the sampler on the GPU. A scoring request runs the same prefill batching with a flat topology, reads the logits at the final prompt position as BF16 and drops the state. No state outlives a request in either worker.

### Engine feature flags

| Feature | Default | Used by | Effect |
| --- | --- | --- | --- |
| `metal` | on | both workers | Metal backend and kernel compilation |
| `cpu` | on | unit tests only | CPU reference kernels; required by the unit harness |
| `grammar` | on | nothing shipped | Grammar-constrained sampling via `xgrammar-rs` (needs `vendor/cxx`); the workers do not enable it and the API offers no structured output |
| `deferred-state-init` | off | `stark-score` (`optimized`) | Encodes the zeroing of DeltaNet state into the first command buffer instead of a synchronous fill per layer at state creation. The flag that records "fills encoded" is cleared at encode time, so a state whose first encoding fails before submission must be discarded; the scorer does this by dropping its state on every error. Not enabled for generation |
| `scoring-last-layer-pruning` | off | `stark-score` (`optimized`) | Skips the query path of the final attention layer for prompt chunks whose logits are not read. Requires the per-state opt-in that only `score_next_token` sets; ordinary generation is unaffected even when the feature is compiled in |
| `attention-kv-only` | off | `stark-score` (`optimized`) | With the pruning above, uses a KV-only preparation kernel for the pruned final layer. Only fires for non-final chunks of prompts longer than 1,024 tokens whose final layer is owned, non-ring full attention |
| `direct-delta-conv` | off | `stark-score` (`optimized`) | Removes padded DeltaNet convolution tokens with exact arithmetic |
| `compact-delta-conv` | off | `stark-score` (`optimized`) | Compact convolved QKV on top of `direct-delta-conv`; the original gate and control projection is still computed for beta, alpha and gate reads |
| `rope-cache` | off | `stark-score` (`optimized`) | Caches CPU-side RoPE tables for up to 8,192 positions (8 MiB); GPU arithmetic is unchanged |

`stark-score --features optimized` enables the six scoring features. `stark-generate` enables only `metal`. The distribution builder compiles the two workers in separate Cargo invocations so feature unification cannot enable the scoring features in the generator. The six features were validated for bit-exact scoring parity ([scoring parity](validation/SCORING_PARITY.md)); their effect on ordinary generation has not been validated, which is why the generator is built without them.

## Workers

`crates/stark-generate` and `crates/stark-score` are small: argument parsing, protocol parsing with strict field validation, a backend trait with one native implementation, and the request loop. Each redirects its own stdout and stderr to `/dev/null` before loading the model and writes replies through a duplicated descriptor, so no library output can enter the protocol stream. Error codes are fixed strings and never carry request content. The generator reads a cancellation pipe when `STARK_CANCEL_FD` is set. Contracts: [generation](GENERATION.md), [scoring](SCORING.md).

## Python package

`tools/converter/stark_convert` is one package with three roles:

| Role | Modules |
| --- | --- |
| Converter | `cli.py`, `sources.py` (source resolution, hashing, manifests), `model_support.py` (shared family/mode registry and bounded metadata preflight), `model_catalog.py` (representative metadata-only family discovery), `streaming.py` (safetensors planning and bounded tensor streaming), `compressed.py` (MLX repacking), `metadata.py`, `profiles.py`, `registry.py`, `qualifications.py`, `provenance.py`, `classifier_metadata.py`, `adapters/` (one planner per architecture), `gguf_import.py`, `benchmark.py`, `progress.py` |
| Chat and workers | `chat.py` (terminal chat and the `Worker` class that owns a worker process), `decision.py` (`ScoreWorker` for Decision mode) |
| App backend | `desktop.py` (`Desktop`: model library, catalogue requests, install and conversion jobs, worker lifecycle, request queue), `model_inventory.py` (bound saved-model display metadata), `model_removal.py` (owned-folder Trash and library-only removal), `local_api.py` (`LocalAPI`: the HTTP server), `api_request.py` (chat request validation) |

Two Python projects exist: `tools/converter/pyproject.toml` is the product package (Python 3.13.13, locked in `tools/converter/uv.lock`). `tools/pyproject.toml` is an inherited developer environment for the unused `data_exporter`, `gpu_trace` and `model_compare` tools and is not part of the product. Repository policy scripts under `scripts/ci/` run on Python 3.12.13 with their own script lockfile.

## App control protocol

`Stark.app` launches `desktop.py` with the paths of the two worker binaries and its data directory, and exchanges JSON lines over the process's stdin and stdout. Each line is capped at 1 MiB. Commands from the app:

| Command | Fields | Effect |
| --- | --- | --- |
| `list` | | Re-emit the model library |
| `support` | | Re-emit the shared family/mode/format matrix without loading a model |
| `catalog` | `family_id`, `request_id` | Start bounded metadata-only discovery for one application model family |
| `catalog_cancel` | `request_id` | Cancel only the matching catalogue lookup and clear its cached choices |
| `install_catalog` | `catalog_request_id`, `entry_id`, `support_request_id`, optional `artifact_mode` | Resolve a server-cached choice and repeat install checks at its immutable source revision |
| `install` | `source`, `kind`, optional `artifact_mode`, `support_mode`, `support_request_id` | Register a converted folder or inspect raw-checkpoint metadata before downloading/converting; opt into `vision-language` output explicitly. Intended support mode is `text`, `decision` or `vision-language`; decision use still produces a text artifact. |
| `remove_model` | `model_id`, `action`, `request_id` | Recheck `trash` or `unregister`, stop that model's worker if loaded, then remove its saved entry |
| `run` | `model_id`, optional `mode` | Load the model into the `chat` generation worker (default) or `decision` scoring worker |
| `stop`, `quit` | | Unload the worker; `quit` also stops the API and exits |
| `generate` | `text`, optional `image` | One chat turn; the image is an opaque private-file reference, resupplied on every image-conversation turn |
| `decide` | context, question, choices | Score the choices with the scoring worker |
| `reset` | | Clear the conversation while keeping the model loaded |
| `configure` | `context_tokens`, `max_new_tokens`, `system_prompt`, `thinking` | Apply chat settings and start a new conversation |

Events to the app: `models` (the library), `model_families` (shared structural registry in `report`), `model_support` (metadata verdict in `report` with the matching `request_id`), `status` (`state` and `message`), `progress` (download and conversion bytes), `installed`, `ready` (worker limits after `run`), `configured`, `reply`, `decision`, `connection` (the API base URL and key), and `error` with a fixed message. The backend fences every command and request with an epoch that changes when a model is stopped or replaced, so a late reply from a previous worker is discarded. Chat and API requests share one FIFO queue in front of the single worker.

Catalogue replies use `model_catalog` with `request_id`, `family_id` and a
schema-1 `report`. Installs echo `support_request_id` in `installed`; errors echo
the matching `request_id`. Removal completes with `model_removed` or
`model_remove_error`, carrying the removal `request_id`, `model_id` and optional
`stopped_model_id`. Clients accept these results only for the active request.

Catalogue items may include `model_group: {id, name, publisher}` as optional
presentation metadata. It groups visible model-size and quantization rows but
does not replace the item's immutable source ID, repository or revision. Missing
or malformed grouping metadata falls back to grouping by exact repository.
Search queries are local and transient; sort preferences are persisted separately
for the saved-model list, family overview and family detail. Typing in a search
field does not issue another catalogue request.

The app delegate owns the independent Manage Models window. Explicit management
buttons open it; Install Model has no management-sheet binding. The menu and
Playground show the running name and open management instead of model pickers.
Saved-row Run uses a live private-ID lookup before selecting the model and
sending the existing `run` command. It inherits the current app mode, rejects
busy/stale rows and keeps management open. Stop follows the worker or pending
load's private ID and rejects unrelated installation/removal operations.
The filtered active version is extracted into a separate first section. Remaining
versions retain their source groups and are sorted again without the active
version's size. A search can hide the active section.
Loading, stopping and runtime errors remain visible in management. Closing management
cancels only catalogue browsing, including when using the native close button.

Library entries can include an optional `display` object with schema version 1:
`group_id`, `group_name`, nullable `family_id`, `quantization`, `weight_bytes` and
`source_revision` and `artifact_mode`. It describes saved variants for the model browser. Grouping
uses unambiguous source identity from a receipt with a consistent source-manifest
schema/digest, current config hash and declared weight-file size; otherwise each
installed ID stays separate.
Header inspection is bounded at 1 MiB and config/receipt reads at 2 MiB each,
with no tensor payload reads. Missing or invalid metadata yields partial display
information and never blocks library startup or changes admission. The Swift
client ignores malformed/unknown descriptors and retains the original name.
This presentation object is not persisted into model registry or artifact files.

Each library row also reports its `artifact_mode` and a schema-1 `removal`
descriptor (`action`, `label`, `message`). The app confirms the offered action.
Exclusively owned managed directories are moved through macOS Foundation's
Trash API; imported, shared or unowned paths are unregistered without moving
files. The backend rechecks action and directory identity, serializes the
registry update with library projection, and restores the registry and attempts
file rollback if persistence fails. Only the removed model's active worker is
closed. Receipts remain independent of library registration.

A family click starts a separate isolated `--family-models` child. The backend
caps its report at 256 KiB and its lifetime at 120 seconds, validates the
schema, and fences replies with a catalogue generation and request ID. Cancelling
the lookup terminates only that child. Discovery considers at most 45 candidates
and returns at most 30 representative compatible choices. It inspects bounded
configuration, processing metadata and repository file information without
reading weight payloads. Size and byte estimates remain nullable.

Accepted choices are cached server-side by opaque entry ID with their repository,
immutable primary revision and allowed artifact modes. `install_catalog` resolves
that cache, validates the selected mode, and passes the cached revision to
preflight. A mismatched revision stops installation before weights; the actual
conversion receives the checked revision. Installed markers intersect bound
repository/revision matches with the saved artifact mode. The Swift browser
retains download/preparation progress and does not change model selection after
a catalogue install completes.

The backend emits `model_families` at startup and for `support`. Its schema-1
matrix records implemented paths and constraints. A raw install runs a bounded,
cancellable `--preflight` child and emits `model_support` before any weight
download. The app gives each install a fresh `support_request_id` and accepts a
report only for the active installation and matching identifier. The backend
also checks the selected mode and logical primary source/revision before
continuing. Unsupported, inconclusive, malformed or missing reports cannot start
the weight downloader. A rejected check preserves the previously loaded model.

A conditional report pins the primary Hugging Face revision for conversion.
External catalog metadata is independently resolved and rechecked by the actual
source resolver; the preflight does not lock all dependency revisions or bytes.
The final conversion manifest records the sources actually consumed. A family
row, correlation identifier or conditional report cannot enable runtime images,
prove native execution, or establish decision calibration. Converted imports
retain artifact/native admission; vision artifacts remain Chat-only. The full
report fields and state meanings are in [model-family support](MODEL_SUPPORT.md).

Operation-completion events are published after their job releases the admission
slot. A client receiving Ready can immediately submit its next command. Buffered
completion events retain the original epoch and are discarded after Stop.

The Swift process also supplies an initially empty private `--attachment-root`.
The optional image object contains exactly `id` (32 lowercase hex characters),
`media_type` (`image/png` or `image/jpeg`), and `byte_length` (1 to 4 MiB). Python
claims a descriptor-relative private regular file and unlinks it before decoding;
paths and encoded bytes never enter this control protocol. `ready`/`configured`
negotiate `images`, `image_attachment_transport: private-file-v1`, and
`max_image_bytes: 4194304`. `image_progress` reports `preparing` or `generating`;
these are phases, not measured percentages. Native validation remains
authoritative. See [native image support](MULTIMODAL_IMPLEMENTATION.md).

## Where validation lives

- Scoring parity and generation acceptance records: [docs/validation](validation/).
- Benchmark studies and the current comparison: [docs/benchmarks](benchmarks/README.md).
- Aggregate receipts with hashes: [provenance](../provenance/).
- Historical milestone and run records: [docs/archive](archive/).
