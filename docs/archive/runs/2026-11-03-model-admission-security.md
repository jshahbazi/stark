# Model admission security validation

## Source and scope

The fixes start from public `main` at
`2ee0b56f948198f7f559d97214348fedb42c43c1`, fetched on 2026-11-03.
The implementation is on `codex/security-model-admission`. The source hashes
below bind this record to the tested implementation without depending on the
record's eventual commit identifier.

Before pushing, the branch was rebased onto updated `main` at
`4545a409d32703bbf69ccc9be85d2d2f43aa5c4d`. The intervening changes affect only
menu-bar status colors and their README description. The tested security
implementation and test inputs are unchanged; all implementation hashes below
were checked again after the rebase.

This run covers three controls:

- Native Lily tensor admission checks shape/dtype byte agreement, checked
  offsets and file bounds before exposing metadata. Fused loading separately
  checks destination capacity before constructing native mutable slices.
- Native shard headers must fit both the fixed 100,000,000-byte maximum and
  the opened regular file before allocation. Native, external MLX and canonical
  Uzu storage retain their separate supported representations.
- Model-provided templates compile and render in disposable processes with
  restricted capabilities, bounded input/output, CPU/wall deadlines and a
  shared conservative expansion budget. Guards cover arithmetic, captured
  output, concatenation, slices, literal containers, methods and filters.
  Size failures retain the existing oversized-request error contract;
  cancellation terminates the evaluator and preserves the loaded worker.

No kernel, checkpoint precision policy or numerical operation was changed.
Upstream source attribution is retained in `vendor/lily-core/UPSTREAM.md`.
Dependency manifests and lockfiles are unchanged.

## Environment

- Apple silicon arm64, macOS 26.7.1.
- Rust 1.98.1 for native builds/checks; `nightly-2026-10-01` for workspace
  formatting.
- uv 0.12.24, managed Python 3.13.13 for converter checks and Python 3.12.13
  for repository policy checks.
- Isolated worktree with its own build outputs. Standalone Lily checks use
  `CARGO_TARGET_DIR=$PWD/target/lily-core-tests`.

## Verification

All commands run from the repository root, with the rustup binaries on `PATH`.

| Check | Result |
| --- | --- |
| `uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py` | Passed |
| Same script with `--test` | 41 passed |
| `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q --tb=short -rs` | 5,902 passed, 89 skipped |
| `uv run --project tools/converter --locked --python 3.13.13 ruff check tools/converter` | Passed |
| Same environment, `ruff format --check tools/converter` | All 136 files formatted |
| `uv lock --check --directory tools/converter` | Passed |
| `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 python tools/converter/scripts/check_installed_package.py` | Editable, sdist, wheel, license, provenance, conversion and benchmark controls passed; includes an actual isolated render from the installed wheel outside the checkout |
| `cargo +1.98.1 test --locked --manifest-path vendor/lily-core/Cargo.toml safetensors -- --test-threads=1` | 20 synthetic storage/admission tests passed; local checkpoint check is opt-in |
| Same standalone crate, `weights::destination_tests -- --test-threads=1` | 1 passed |
| `cargo +1.98.1 test --locked -p stark-generate --no-default-features` | 72 passed |
| `cargo +1.98.1 check --locked -p stark-generate` | Passed |
| `cargo +1.98.1 check --locked -p stark-score --features optimized` | Passed, separately from generation |
| `cargo +1.98.1 clippy --locked -p stark-generate --all-targets -- -D warnings` | Passed |
| `cargo +nightly-2026-10-01 fmt -p stark-engine -p stark-engine-macros -p stark-score -p stark-generate --check` | Passed |
| Rust 1.98.1 `rustfmt --edition 2024 --config skip_children=true --check` on the three changed vendor Rust files | Passed |
| `cargo +1.98.1 build --release --locked -p stark-generate` | Passed |
| `git diff --check` | Passed |

The converter skips are explicit: 88 independent external-reference metadata
parity cases require `STARK_CONVERT_REFERENCE_ARTIFACTS`, which was not supplied;
one Hermes parser assertion does not apply because its arguments preserve JSON
types directly. The new security regressions executed.

Regression controls reject the reported shape-small/payload-large tensor
before a native read, enormous or truncated header declarations before large
allocation, and malformed or overflowing intervals. Valid BF16/F32/U32/I64/U64,
scalar/empty tensors, cached-file symlinks, external MLX aliases and canonical
Uzu physical/logical restoration remain covered. Template controls exercise
compile-time expansion, computation deadlines, child cleanup, eager aggregate
allocation, slice copies, retained containers, sandbox errors, normal rendering,
tool history, decision/image callers and API size/cancellation recovery. The CPU
test verifies child CPU consumption and termination by the resource-limit signal.

An independent read-only patch review identified eager aggregate allocation in
the first candidate. The final shared-budget and incremental-evaluation controls
and regressions close that path, including equivalent slice and literal forms.

## Model-bound controls

### Native Lily header admission

The opt-in test was run against the existing local native Lily checkpoint:

```sh
STARK_TEST_LILY_CHECKPOINT=/absolute/path/to/native-checkpoint \
  CARGO_TARGET_DIR="$PWD/target/lily-core-tests" \
  cargo +1.98.1 test --locked --manifest-path vendor/lily-core/Cargo.toml \
  local_native_checkpoint_headers_remain_admissible -- --ignored --test-threads=1
```

It passed and admitted all 55 native shards without reading tensor payloads or
loading the full model. The header inventory SHA-256 is
`a9d4305760ffa0ed1b1dcc96a74f242d96e6cd04383e7e1be74948e55a1d99fe`.
The digest covers the index bytes, then each sorted unique shard's relative
name, file length as little-endian u64, eight-byte header-length field and header
bytes. It identifies this metadata-admission control, not full weight integrity
or generation quality.

### Conversion, isolated rendering and generation

The complete cached `Qwen/Qwen3-0.6B` checkpoint at revision
`c1899de289a04d12100db370d81485cdf75e47ca` was converted offline into the
ignored `runtime/security-smoke-qwen3` directory using the catalog profile.
The newly built release worker reached readiness, accepted a prompt from the
new isolated renderer and completed greedy generation with a 512-token context,
32-token output maximum and seed 0. The result was nonempty, 34 UTF-8 bytes,
with finish reason `eos`. Only aggregate status was emitted; prompt, response
and token vectors were not saved or printed.

Release worker SHA-256:
`078b654bd3691b55012db695ef52b0cd4a7723140bc0c25a7af0dcb317cb2847`.

## Limits and pre-existing checks

The evaluator's 16 MiB cumulative expansion budget is a conservative work and
allocation budget, not a peak-RSS claim. macOS uses a sampled 256 MiB child-RSS
watchdog in addition to the primitive guards and hard CPU/wall/output limits;
it has no kernel-enforced address-space ceiling in this implementation.
Unsupported generic template constructs can fail closed.

A stricter standalone Lily `clippy --all-targets -- -D warnings` check reports
the same pre-existing failures on clean base `main` and the candidate:
63 diagnostics at 48 unique locations, with no diagnostics in the changed
Rust files. General vendor lint cleanup is outside this patch. The required
generation workspace Clippy check passed.

This run does not establish full Lily generation, whole-model numerical parity,
throughput, all external reference compatibility, signed distribution upgrades,
or GitHub-side repository/release protection settings.

## Tested implementation hashes

| File | SHA-256 |
| --- | --- |
| `vendor/lily-core/src/safetensors.rs` | `aea4ce615094acc7e9cc537983dbaa7d10facecf7514c7781d721ecfac1676b0` |
| `vendor/lily-core/src/weights.rs` | `54a347a45b5617ad054a46173e43c7e0848a44b962b5f162e84c14cac0c77797` |
| `tools/converter/stark_convert/template_runtime.py` | `1ca2a313bfd7b48059aeab52d5f3077053cb7ea663b15db91b1e48813f6d9353` |
| `tools/converter/stark_convert/chat.py` | `4821a29054b6581fc05464c86b4787bf82d089ba685ac3538cf136b9f71fe950` |
| `tools/converter/stark_convert/desktop.py` | `1661336ccdf9e6f38558bb8e742aa46d7434fb01a9165cba0948aae189770054` |
