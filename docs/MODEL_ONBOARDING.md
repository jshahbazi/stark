# Adding model support

Record each compatibility, memory, precision or performance attempt in the
[model experiment ledger](model-experiments/README.md), including failed,
inconclusive and abandoned trials. Use its [template](model-experiments/TEMPLATE.md)
to bind the exact artifact, source, precision and workload. Keep implemented
support, numerical qualification, measured speed and deployment as separate
outcomes; preserve older source-bound records when an attempt changes.

Every new model family, checkpoint template and import route must assess function
tools as well as native execution. Complete the [tool-calling acceptance
checks](#assess-and-qualify-function-tools) below. A missing parser must have an
explicit disposition; ordinary chat can remain available.

Every new model family, native backend, conversion route or state-affecting
inference path needs an explicit generation prefix-reuse assessment. Implement
reuse when the backend can restore an exact prefill boundary within the model's
memory constraints. A recurrent, sliding-window or unfamiliar architecture needs
a state inventory and a restoration design before it is classified as blocked.
Record a concrete correctness, implementation or memory blocker when reuse
cannot be provided. Model support work is incomplete until that disposition and
its evidence are recorded.

This is a development acceptance requirement. Installation metadata and catalogue
entries do not enable reuse; the loaded worker's readiness contract remains the
runtime authority. Follow the [model-family support contract](MODEL_SUPPORT.md),
[generation contract](GENERATION.md#prefix-reuse) and [repository rules](../AGENTS.md).

## Bind the implementation and artifact

Record the source repository and immutable revision, weight/configuration/
tokenizer/template hashes, conversion or native-preparation revision, worker
source and executable hashes, Rust toolchain, device and OS. Preserve upstream
licenses and patch provenance.

Describe the precision policy for weights, activations, accumulation, KV caches
and recurrent state. Identify the actual graph and artifact layout selected by
the loader. Quantizations and formats can share implementation code while still
requiring their own model-bound qualification. A result for one checkpoint does
not qualify a different artifact, precision policy or kernel path.

## Inventory state before implementing reuse

Trace every value read by prefill, decoding and any draft/verification route.
Classify its ownership, precision, position semantics, lifetime and restoration
method at the chosen boundary:

| State | Questions to resolve |
| --- | --- |
| Attention KV and indexer caches | Which rows remain valid? Are caches full, ring, sliding or shared between layers? Which logical lengths, positions and storage aliases must be restored? |
| Recurrent state | Which recurrence accumulators, convolution windows and rolling histories need a snapshot? What copy/synchronization and memory costs does it add? |
| Draft heads and auxiliary graphs | Which hidden states, caches and committed positions are read when MTP or another option changes? What sampler, verification and token-commit policy does a native MTP round use? Which options must enter the reuse key? |
| Position and token history | Which absolute positions, n-gram windows and history lengths affect the next prefill operation? |
| Sampler and temporary buffers | Which values must be reseeded or overwritten for each request? Which scratch buffers require a GPU completion fence before reuse or destruction? |

Rewindable full-attention caches can retain proven prefix rows. Non-rewindable
state needs a snapshot at a prefill boundary. Document why the restored suffix
uses the same chunk boundaries, kernels, precision and reduction order as the
fresh execution. Never reuse rows produced by a final prefill chunk or decoding
unless that route has its own exactness proof and qualification.

## Implement the loaded-backend contract

Keep the existing worker and host lifecycle unless a scoped protocol change is
required:

- For a backend without a declared context maximum, implement and test the
  loaded-state memory estimate used by automatic execution planning. Include
  owned/shared and ring/full KV state, recurrent buffers, activation precision,
  scratch allocation classes and checked overflow. Report the resulting
  allowance separately from trained model capacity.
- Derive support from the admitted backend and actual state layout. Keep
  `prefix_cache.supported` false for a path whose restoration is incomplete.
- Implement the backend's support, chunk size, key, planned reuse, retention,
  publication and clearing hooks. The worker's `GenerationBackend` trait and
  native adapters define those hooks; the [technical guide](TECHNICAL_OVERVIEW.md)
  locates the engine and hybrid implementations.
- Match the immediately preceding request's model instance, host `cache_scope`,
  context size and every state-affecting option, including MTP selection. Reuse
  only its shared token prefix at a supported chunk boundary. A scope identifies
  an existing host principal; model names and client-supplied metadata cannot
  grant another principal's state.
- Publish an entry only after its successful terminal reply has been delivered.
  On any error, cancellation, image operation, reset, scope/key mismatch,
  disconnect, deadline, revocation, idle expiry or worker restart, apply the
  [documented invalidation rules](GENERATION.md#prefix-reuse).
- Retain at most one preceding request's state in memory. Do not persist or log
  it. Keep `reused_input_tokens`, progress events and processed-token rates
  consistent with the actual native work.

## Qualify correctness and lifecycle

Add model-free tests for the backend-specific state and capability decisions.
Run the shared worker and host tests for changes to their contract. The existing
worker tests cover key mismatches, failed publication, invalidations and idle
expiry; host tests cover scope ownership, reset and revocation.

Run [the model-bound validator](../scripts/validate_generation_prefix_cache.py)
with the exact admitted artifact. It loads workers sequentially and keeps
prompts, generated text and token vectors in memory. Output records contain
aggregate results and hashes only. Use a unique ignored output path:

```sh
uv run --project tools/converter --locked --python 3.13.13 python \
  scripts/validate_generation_prefix_cache.py \
  --model /absolute/path/to/model \
  --baseline /absolute/path/to/reference/stark-generate \
  --candidate /absolute/path/to/candidate/stark-generate \
  --reuse boundary \
  --output runtime/model-prefix-reuse-validation.json
```

Use `--reuse rewind` for a backend that can rewind to any eligible full chunk.
Use `--reuse boundary` for a backend that can restore only its retained snapshot.
Choose prompts long enough to cross the advertised chunk boundary, and a context
large enough for the synthetic follow-ups and output budget. `--first-tokens`
sets the first-turn target; `--tools`, `--instruction-paragraphs` and
`--follow-up-sentences N N N` control the synthetic session size. Keep their
values in the validation record. Use `--mode exactness` and `--mode timing` for
separate runs when the model's resource cost requires it. The exactness run
includes a new worker using the same host scope and requires zero reused tokens
after that restart.

Build each source checkout with a distinct `CARGO_TARGET_DIR` as described in
[Contributing](../CONTRIBUTING.md#local-checks).

The acceptance evidence must cover:

- Successful greedy and seeded stochastic outputs identical between cached
  execution, the same candidate with reuse disabled, and the matching fresh
  reference. Equal error replies do not establish output equivalence.
- Cold first turn, growing follow-ups, repeated prompts, shorter prompts,
  divergence before and after the snapshot, and final-chunk edge cases. Assert
  the expected reused token count for every request.
- Scope and context changes; every state-affecting option, including MTP on/off
  where supported; an unscoped request; tokenize/decode preservation.
- Error, cancellation, image operation, explicit reset and process restart.
  Use worker/host lifecycle tests for idle expiry, failed output delivery,
  disconnect, deadline and principal revocation. A backend-specific retained
  state path must participate in those same cleanup hooks.

Run the applicable native state tests on Metal and the repository checks listed
in [Contributing](../CONTRIBUTING.md). Model-free tests establish lifecycle
behavior; the artifact run establishes numerical behavior for the tested path.

## Measure and record the disposition

Measure both cold first-token cost and warm follow-up benefit with matching
prompts, output budgets, sampling, context and model artifact. Alternate fresh
and cached runs, serialize GPU work, record repeated timings and retained-state
memory accounting, and report the actual reused token counts. Derive snapshot
payload bytes from admitted allocations or measure them directly. Report retained
cache capacity separately, and distinguish payload accounting from process RSS,
allocator overhead and transient copies. Record observed peak memory when
available. Keep timing observations separate
from correctness results. A successful restore does not establish a speedup.

Update the current support documentation with one of these dispositions:

| Disposition | Required evidence |
| --- | --- |
| Implemented and qualified | Loaded capability, restoration design, passing model-free and artifact-bound exactness/lifecycle results, timing measurements and retained-state memory accounting with the tested scope. |
| Implemented, artifact qualification pending | Supported code path and state proof, plus the exact formats/options/artifacts that still lack a model-bound run. Do not inherit another artifact's qualification. |
| Blocked | State inventory, the concrete reason exact restoration or practical retention is unavailable, and the work needed to remove the blocker. Keep the runtime capability disabled for that path. |

Put a dated, source-bound record under `docs/archive/runs/` when the evidence
must be preserved, and link it from the current reference page. Keep existing
records unchanged. Publish aggregate results and hashes only; keep native
validation artifacts private and ignored.

## Assess and qualify function tools

Inspect the exact artifact's `token_codec_config.prompt_template`, not its
family name, model card, catalogue flags or a claimed `supports_tools` field.
Import completion and Installed rows report `available`, `unavailable` or
`unchecked` from that actual codec. This reports protocol compatibility;
`model_behavior` remains `unverified`. The loaded template and
`/v1/capabilities` remain the authority for serving requests.

For each new template:

1. Pin its upstream source and revision, template hash, license and program
   fingerprint. Preserve the original template. Review definitions, assistant
   calls, typed arguments, tool results, result ordering and the next assistant
   prefix. Reuse a parser only when that complete protocol conforms.
   Review the template's expected tool-definition shape and result roles, ID
   constraints, supported schema subset and required stop tokens. Keep render
   adapters separate from public API IDs and preserve the original codec. Any
   reviewed runtime template correction needs a pinned packaged source, license,
   patch provenance and a separately recorded effective-template hash.
2. Add source-bound native-rendering and parsing tests. Cover null assistant
   content, multiple calls, results arriving out of order, all supported scalar
   and structured argument types, multiline/Unicode strings, streaming,
   malformed/truncated output, duplicate/unknown names, limits and tool choices.
   Keep unknown or changed programs disabled. Marker presence is insufficient.
3. If the model uses a different trained format, implement and test a separate
   bounded parser. A model without tool training needs an appropriate trained
   checkpoint or training/evaluation work; adding instructions or a template
   does not establish that capability.
4. Run a model-bound workflow on the actual checkpoint/quantization and worker.
   At minimum check correct call selection/arguments, returned-result use and
   ordinary chat. Check buffered/streamed API behavior and cancellation with a
   real client before claiming an agent integration. Record failures honestly.

The bounded native smoke command below runs synthetic requests serially, returns
a synthetic tool result without executing external tools, and writes only
aggregate checks and hashes. It uses greedy sampling with MTP disabled and hashes
the actual config, tokenizer, stopping metadata, shard index and Safetensors/GGUF
payloads, along with the generation worker and native template helper. It checks
bytes and the optional file set again after execution. Hashing a large model
can take time. It rejects external expert-streaming configurations and EXL3 banks.
Choose a new ignored output filename and run without another native generation
workload:

```sh
uv run --project tools/converter --locked --python 3.13.13 python \
  scripts/validate_tool_calling.py \
  --model /absolute/path/to/model \
  --worker /absolute/path/to/stark-generate \
  --output runtime/model-tools-validation.json
```

A passing run qualifies only those three bounded checks for those sealed bytes,
budgets and sampling settings. It does not establish broad task competence.
Store a separate dated record with the source/build identity, device/OS, parser,
template and artifact hashes, aggregate outcomes and remaining limitations.
Record protocol support and native model-loading results separately. A parser
can pass source-bound synthetic tests while its checkpoint fails loading or
ordinary chat; that artifact has not passed the workflow. Do not weaken a failed
check or infer a loading failure's inner cause from a generic worker error.
Never persist prompts, generated calls, arguments, responses or token vectors.
Do not reuse another artifact's behavioral qualification or silently promote
import metadata into one. Consult the [current coverage matrix](TOOL_CALLING.md)
for implemented formats and outstanding work.
