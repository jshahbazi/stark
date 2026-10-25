# Continuous integration and security checks

CI, Security and CodeQL run for pull requests to `main`, pushes to `main` and manual dispatches; Security and CodeQL also run weekly. To check a branch before opening a pull request, choose the branch under **Actions > CI > Run workflow**. All CI inputs are synthetic: no workflow downloads a model, and private validation artifacts never enter hosted runners.

| Workflow | Coverage |
| --- | --- |
| CI | Repository policy, Rust formatting, locked dependency resolution, workflow syntax, converter tests and lint, installer and app script tests, process lifecycle tests on Linux and macOS, Apple silicon builds, Clippy and tests of both workers in the default and `optimized` configurations, the engine unit and kernel suite on the pinned nightly compiler, and the vendored cxx panic-safety test with the engine's `grammar` feature. |
| Build | Builds an unsigned development bundle on pushes to `main` or `codex/**` that touch source or build inputs, verifies it, and uploads it as a seven-day artifact. |
| Security | cargo-deny advisories, soundness, licenses and sources across all Cargo features; OSV scans of the seven tracked lockfiles; dependency review on pull requests. |
| CodeQL | Rust, C/C++, Python, JavaScript/TypeScript and GitHub Actions in build mode `none`. Metal kernels and Swift are not analyzed. |
| Prebuilt release | Manual; builds an unsigned archive from an existing version tag and uploads a draft. See [releasing](RELEASING.md). |
| Signed release | Manual from protected `main`; builds a combined CLI and app distribution from the dispatched `main` commit, verifies it before credentials are imported, signs and notarizes, verifies the installation and DMG, and in a separate publishing job creates the version tag and publishes the latest stable release. |

The obsolete inherited `Tests` workflow was removed. The remaining inherited `Release`, `Run Benchmark` and `Run Stark Cargo Command` workflows are present but every job is `if: false`; the repository policy check requires them to stay disabled.

## Toolchains and runners

- Rust code builds with Rust 1.98.1 on the `macos-26` Apple silicon runner, whose Xcode must support the engine's Metal 4 / macOS 26.4 target. Formatting uses `nightly-2026-10-01` rustfmt because the inherited rustfmt configuration needs nightly options, and the engine unit suite builds with the same nightly compiler because it uses `custom_test_frameworks`.
- Python checks use uv 0.12.24. Repository policy scripts run on Python 3.12.13 with an adjacent script lockfile; the converter and the installer tests run on Python 3.13.13.
- actionlint, cargo-deny and OSV Scanner downloads are verified against pinned hashes. External Actions are pinned to full commit SHAs, checkout does not persist credentials, build jobs hold read-only tokens, and no job uses `pull_request_target` or a self-hosted runner.

## What CI does and does not cover

**Product code.** Both workers are built, linted with Clippy at `-D warnings`, and tested with synthetic inputs; the generation worker's process isolation is checked by a Python harness. The converter runs its full model-free pytest suite with Hugging Face offline mode. Installer, app builder, signer, updater and release-version scripts run their unit tests. The C process reaper is compiled with warnings as errors and exercised on Linux and macOS.

**Engine unit tests.** The `engine-unit` job runs the engine's own unit suite under `crates/stark-engine/unit` on the Apple silicon runner with `nightly-2026-10-01` (the suite needs `custom_test_frameworks`) and the `cpu` and `metal` features. This includes the kernel tests that compare each Metal kernel with the CPU reference backend in `src/backends/cpu`. The ignored tests that read a converted model from `STARK_TEST_MODEL_PATH` do not run in CI; run them locally as described in [Contributing](../CONTRIBUTING.md#engine-tests).

**Grammar support.** The `cxx` job runs the vendored cxx panic-safety test with the `grammar` feature, which keeps the XGrammar bridge building although neither worker enables structured output.

**Not covered.** Model loading, numerical parity, GPU performance and converted-model compatibility need private, model-bound runs; see [validation](validation/SCORING_PARITY.md) and the [benchmarks index](benchmarks/README.md).

## Dependency gates

OSV Scanner downloads the public advisory databases and matches the seven tracked lockfiles locally: the root `Cargo.lock`, `tools/uv.lock`, `tools/converter/uv.lock`, three `tools/stock_benchmark` uv lockfiles, and the policy script lockfile. A strict JSON checker requires every declared lockfile to appear in the report; adding or removing a lockfile means editing `security.yml` and `SCANNED_LOCKFILES` in `scripts/ci/check_repository.py` in the same change. Every known vulnerability fails both cargo-deny and the OSV gate; no advisory is ignored. Unmaintained-package notices are warnings.

## Rust build caches

Rust jobs use `.github/actions/rust-cache`, a thin wrapper over the pinned `actions/cache`. Each job and configuration has its own cache of registry sources, Git dependencies and `target` output, keyed by runner image, compiler and Xcode identity, Cargo manifests and lockfiles, build scripts and native sources (`scripts/ci/rust-cache-key.sh`). Incremental compilation is off so no per-checkout incremental state is stored. Every build, Clippy and test step still runs after a restore; caches are saved only when the job succeeds. Pull requests can restore `main` caches, but caches saved on a merge ref do not populate `main`. Increment the `stark-rust-v1` namespace in the key script to force a fresh cache. The Build workflow does not use the cache and compiles from scratch each run.

## Download a development build

Open **Actions > Build**, select a successful run and download its artifact. Each build has a unique `0.0.0-dev.COMMIT.RUN.ATTEMPT` version with the source commit and file hashes recorded in the bundle manifest. The artifact holds the archive, its `.sha256`, `install.sh` and `verification.json`; install it with:

```sh
sh install.sh --archive "$PWD/stark-VERSION-aarch64-apple-darwin.tar.gz"
```

Development builds are unsigned and expire after seven days. Use the [published release](INSTALLATION.md) for normal installation.

## Release workflows

The prebuilt release job has `contents: write` only for its final upload step; build and verification receive no publishing token. It requires an existing version tag on a commit merged into `main`.

The signed release workflow runs in the protected `macos-release` environment. Only its credential-import step receives the six Apple secrets; the signing step receives the public identity; a separate job with `contents: write` creates the immutable version tag, uploads the verified assets and publishes the latest stable release after installation, DMG and checksum verification. The helper creates a temporary Keychain, removes credential files and restores the previous Keychain search list unconditionally. Repository policy permits `secrets.*` references only in this workflow and the disabled inherited ones, and rejects other events, branches, missing cleanup and alias reuse.

## Local checks

Run the same checks CI runs before opening a pull request. Repository policy and converter:

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py --test
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q
uv run --project tools/converter --locked --python 3.13.13 ruff check tools/converter
uv run --project tools/converter --locked --python 3.13.13 ruff format --check tools/converter
uv lock --check --directory tools/converter
```

Installer, app and release scripts (shell syntax plus every unit test CI runs):

```sh
sh -n install.sh && sh -n scripts/install-from-source.sh && sh -n scripts/stark
uv run --no-project --python 3.13.13 python -m unittest \
  scripts/test_install.py scripts/test_install_app.py scripts/test_stark_cli.py \
  scripts/test_build_distribution.py scripts/test_build_menubar.py scripts/test_sign_distribution.py \
  scripts/test_apple_credentials.py scripts/test_build_dmg.py scripts/test_release_version.py scripts/test_app_update.py
```

Rust formatting, dependencies and the workers (Apple silicon with Xcode's Metal tools):

```sh
cargo +nightly-2026-10-01 fmt -p stark-engine -p stark-engine-macros -p stark-score -p stark-generate --check
cargo metadata --locked --format-version=1 > /dev/null
cargo deny --locked --all-features check --warn unmaintained advisories licenses bans sources
cargo test --locked -p stark-generate --no-default-features
cargo build --release --locked -p stark-score --features optimized
cargo clippy --release --locked -p stark-score --all-targets --features optimized -- -D warnings
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
cargo build --release --locked -p stark-generate
cargo clippy --release --locked -p stark-generate --all-targets -- -D warnings
cargo test --release --locked -p stark-generate -- --test-threads=1
cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal -- --test-threads=1
mkdir -p runtime
cc -std=c11 -O2 -Wall -Wextra -Werror scripts/stark-validation-reap.c -o runtime/ci-stark-validation-reap
uv run --python 3.12.13 python scripts/test_scoring_process.py --reaper runtime/ci-stark-validation-reap
uv run --no-project --python 3.12.13 python scripts/test_generation_process.py --binary target/release/stark-generate
```

Run the scorer commands a second time without `--features optimized` to cover the default configuration. None of these commands loads a model.

## Repository controls

Keep secret scanning, push protection, Dependabot alerts, private vulnerability reporting and read-only default Actions permissions enabled. `main` requires a pull request, conversation resolution, the `CI completed` and `Security completed` checks with the branch up to date, and blocks force pushes and deletion, including for administrators. The Actions allowlist is limited to the pinned checkout, dependency review, CodeQL, setup-uv, cache and artifact actions used by the workflows; the `macos-release` environment is restricted to protected branches.

To update a tool or action, resolve its release to an immutable commit or verified archive hash, update the pin and any lockfile together, and record the validation run in the pull request. Never ignore an advisory to make CI pass. An exception the maintainer accepts needs its scope, rationale and expiry recorded and enforced in both cargo-deny and the OSV gate; the retired [rustls exception](archive/runs/RUSTLS_TEST_EXCEPTION.md) shows the previous arrangement.
