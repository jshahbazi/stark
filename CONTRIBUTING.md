# Contributing to Stark

Read [AGENTS.md](AGENTS.md) for the repository rules (provenance, logging, caching, commit style). This page covers the practical setup. The documentation index is [docs/README.md](docs/README.md); the architecture is in the [technical guide](docs/TECHNICAL_OVERVIEW.md).

## Toolchains

| Tool | Version | Used for |
| --- | --- | --- |
| Rust | 1.98.1 via rustup (`rust-toolchain.toml`) | Building and testing the engine and workers |
| Rust nightly | `nightly-2026-10-01` | `cargo fmt`, because the inherited rustfmt configuration needs nightly options, and the engine unit suite (see below) |
| Xcode with the Metal Toolchain | current | Compiling `.metal` kernels; the Command Line Tools package alone is not enough |
| uv | 0.12.24 | Every Python environment |
| Python | 3.13.13 for the converter and installer tests, 3.12.13 for the repository policy scripts | Managed by uv; nothing uses the system Python |

Run commands from the repository root. Keep dependencies locked: `Cargo.lock`, `tools/converter/uv.lock` and `scripts/ci/check_repository.py.lock` are checked in CI.

## Layout

| Area | Code | Tests |
| --- | --- | --- |
| Engine | `crates/stark-engine/src`, kernels under `src/backends/metal/kernel`, build script under `build/` | `crates/stark-engine/unit` (nightly, see below), `tests/cxx_string.rs` |
| Workers | `crates/stark-generate/src`, `crates/stark-score/src` | Unit tests inside each module; `scripts/test_generation_process.py`, `scripts/test_scoring_process.py` |
| Converter, chat, API, app backend | `tools/converter/stark_convert` | `tools/converter/tests` (pytest, model-free, fixtures with recorded provenance) |
| Menu bar app | `apps/menu-bar/Sources` | `scripts/test_build_menubar.py`, `scripts/test_app_update.py`, `scripts/test_menubar_controller.py` (needs Xcode, not run in CI) |
| Installer and release | `install.sh`, `scripts/stark`, `scripts/*.py`, `.github/workflows` | `scripts/test_*.py`, `scripts/ci/test_*.py` |

`tools/data_exporter`, `tools/gpu_trace` and `tools/model_compare` are inherited from Uzu and are not part of the product.

## Local checks

The repository policy check is the first thing CI runs and the first thing to run locally:

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
```

Then run the checks for the area you changed; the complete list, matching CI, is in [CI local checks](docs/CI.md#local-checks). The two commands most changes need:

```sh
cargo +nightly-2026-10-01 fmt -p stark-engine -p stark-engine-macros -p stark-score -p stark-generate --check
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q
```

Build the workers separately so the scorer's `optimized` features never unify into the generator:

```sh
cargo +1.98.1 build --release --locked -p stark-generate
cargo +1.98.1 build --release --locked -p stark-score --features optimized
```

## Engine tests

The engine unit suite under `crates/stark-engine/unit` is compiled into the library's test target. It needs a nightly toolchain because `src/lib.rs` enables `custom_test_frameworks` under `cfg(test)`, and the `cpu` feature for its reference kernels. CI runs it in the `engine-unit` job on the Apple silicon runner. To run it locally with the same pinned nightly:

```sh
cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal -- --test-threads=1
```

Nothing in this run needs a model or the network. The ignored `test_metadata_loading` test and the model-loading benchmark read a converted model from `STARK_TEST_MODEL_PATH`; point it at a directory such as the one the smoke test below produces and add `--include-ignored` (or `--bench`, see `crates/stark-engine/unit/BENCHMARKS.md`). Kernel changes should be accompanied by a run of this suite and, where a kernel feeds the scorer, by the [scoring parity](docs/validation/SCORING_PARITY.md) replay.

## Run a model-bound smoke test

CI never loads a model. After changing the engine, a worker, or the converter, check the real path once:

```sh
cargo +1.98.1 build --release --locked -p stark-generate
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B --output runtime/smoke-qwen3
printf '%s\n' '{"operation":"generate","rendered_prompt":"<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n","context_tokens":512,"max_new_tokens":32}' \
  | ./target/release/stark-generate --model-path "$PWD/runtime/smoke-qwen3"
```

The first line of output is the readiness reply and the second is the generated completion; `finish_reason` should be `eos` or `length`. For the scorer:

```sh
cargo +1.98.1 build --release --locked -p stark-score --features optimized
printf '%s\n' '{"operation":"score","input_token_ids":[151644,872,198],"candidate_token_ids":[9707,14990],"context_tokens":512,"temperature":1.0}' \
  | ./target/release/stark-score --score-model-path "$PWD/runtime/smoke-qwen3"
```

`runtime/` is ignored by Git. Delete the converted model when you are done. For an end-to-end check through the app, build it with `scripts/build_menubar.py --install --open` as described in the [menu bar guide](apps/menu-bar/README.md#build-and-install).

## Documentation and validation records

- Reference pages under `docs/` describe current behaviour. When a change alters a protocol, a limit, an option or a default, update the page in the same pull request.
- Write Markdown without emojis and without shell heredocs in examples.
- Put the evidence for a change (test counts, manual checks, measurements) in the pull request description. A record that must outlive the pull request goes under `docs/archive/runs/` with its date and the commit it describes; a benchmark goes under `docs/benchmarks/` and is added to the [benchmarks index](docs/benchmarks/README.md) with a note on what it supersedes.
- Keep models, private fixtures, converted outputs and runtime artifacts out of Git; publish aggregate results and hashes only.

## Commits and pull requests

Use Conventional Commits. Push authorized work to its branch; open a pull request only when asked. CI must pass `CI completed` and `Security completed` before merge, with the branch up to date with `main`.
